"""Blender (Cycles) re-render of the t6_06 hero execution clip: MuJoCo is the physics ground truth, Blender the look.

blender -b -P scripts/blender06/build_scene.py -- [--save out.blend] [--pilot FRAME] [--samples N] [--out PNG]
                                                  [--res 1920x1080] [--orbit]

Reads footage/tables_sim/hero/exec_blender/_export/ (written by export_traj.py + make_textures.py):
  06_traj.npz        per frame (747 = the clip's frames @30 fps): geom world poses, body poses, flex (cloth)
                     vertices, camera (position, forward, up, fy px, principal-point offset), arm on/off flags
  06_traj_model.npz  meshes (Franka, SAM 3D, table, knife ...), geoms, materials; textures in 06_traj_tex/
Builds: every visible MuJoCo geom as a Blender object (parented to an animated empty per moving body), the towel as
a mesh driven per frame by the flex vertices (Mesh Cache .pc2) + subdivision + solidify, PBR materials, a dark
"simulation studio" (near-black void, glowing grid floor that fades out, key/rim/fill area lights), the MuJoCo
camera keyed per frame (same pose, fov and principal-point shift), depth of field and motion blur.
"""
import json
import math
import struct
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(n, d=None):
    return argv[argv.index(n) + 1] if n in argv else d


ROOT = Path(__file__).resolve().parents[3]                      # .../meta
EXP = ROOT / "footage/tables_sim/hero/exec_blender/_export"
TEX = EXP / "06_traj_tex"
D = np.load(EXP / "06_traj.npz")
M = np.load(EXP / "06_traj_model.npz")
NF = int(D["xpos"].shape[0])
FPS, W, H = int(D["fps"]), int(D["W"]), int(D["H"])
TABLE_C = Vector((0.4817, 0.0, 0.0))
FLOOR_Z = -0.75
ORBIT = "--orbit" in argv

GT = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh", "sdf"]
DROP_MATS = {"hero_wall_mat", "hero_floor_mat"}                 # kitchen room: replaced by the dark studio
ARM_BODIES = {"link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger",
              "right_finger"}


def s(x):
    return str(x)


bname = [s(x) for x in M["body_name"]]
gname = [s(x) for x in M["geom_name"]]
mname = [s(x) for x in M["mat_name"]]
texfile = [s(x) for x in M["tex_file"]]

# ------------------------------------------------------------------------------------------------ reset scene
bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene
scn.render.fps = FPS
scn.frame_start, scn.frame_end = 0, NF - 1
scn.unit_settings.system = "METRIC"


def coll(name):
    c = bpy.data.collections.get(name)
    if c is None:
        c = bpy.data.collections.new(name)
        scn.collection.children.link(c)
    return c


# ------------------------------------------------------------------------------------------------ materials
def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    return m, nt, out


def P(nt, **kw):
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    names = {"color": "Base Color", "rough": "Roughness", "metal": "Metallic", "ior": "IOR",
             "aniso": "Anisotropic", "aniso_rot": "Anisotropic Rotation", "coat": "Coat Weight",
             "coat_rough": "Coat Roughness", "sheen": "Sheen Weight", "sheen_rough": "Sheen Roughness",
             "sss": "Subsurface Weight", "sss_scale": "Subsurface Scale", "spec": "Specular IOR Level",
             "trans": "Transmission Weight"}
    for k, v in kw.items():
        sock = b.inputs[names[k]]
        if k == "color":
            v = (*v[:3], 1.0)
        sock.default_value = v
    return b


def img(nt, path, cs="sRGB", interp="Cubic", ext="REPEAT"):
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(str(path), check_existing=True)
    t.image.colorspace_settings.name = cs
    t.interpolation = interp
    t.extension = ext
    return t


def fade_wrap(m, nt, out, shader_out):
    """Arm/pedestal fade-in (the MuJoCo clip fades the arm in during the camera swing): object property 'fade'."""
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_type = "OBJECT"
    at.attribute_name = "fade"
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mx = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(at.outputs["Fac"], mx.inputs[0])
    nt.links.new(tr.outputs[0], mx.inputs[1])
    nt.links.new(shader_out, mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs[0])


def noise(nt, scale, detail=4.0, rough=0.5, coord=None, dims="3D"):
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.noise_dimensions = dims
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = rough
    if coord is not None:
        nt.links.new(coord, n.inputs["Vector"])
    return n


def maprange(nt, src, a, b, c, d):
    r = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(src, r.inputs[0])
    r.inputs[1].default_value, r.inputs[2].default_value = a, b
    r.inputs[3].default_value, r.inputs[4].default_value = c, d
    return r.outputs[0]


MATS = {}


def franka_mat(kind, fade=True):
    key = ("franka", kind, fade)
    if key in MATS:
        return MATS[key]
    m, nt, out = new_mat(f"franka_{kind}")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    if kind in ("white", "off_white"):
        col = (0.60, 0.605, 0.61) if kind == "white" else (0.54, 0.54, 0.525)
        b = P(nt, color=col, rough=0.32, coat=0.35, coat_rough=0.12, spec=0.5)
        # painted cast aluminium: very fine orange-peel micro bump
        n = noise(nt, 900.0, 2.0, 0.5, tc.outputs["Object"])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.04
        bump.inputs["Distance"].default_value = 0.0002
        nt.links.new(n.outputs[0], bump.inputs["Height"])
        nt.links.new(bump.outputs[0], b.inputs["Normal"])
    elif kind == "black":
        b = P(nt, color=(0.018, 0.018, 0.02), rough=0.42, spec=0.45, coat=0.15, coat_rough=0.3)
    elif kind == "light_blue":
        b = P(nt, color=(0.25, 0.55, 0.95), rough=0.3)
        b.inputs["Emission Color"].default_value = (0.3, 0.6, 1.0, 1)
        b.inputs["Emission Strength"].default_value = 2.0
    elif kind == "green":
        b = P(nt, color=(0.2, 0.9, 0.3), rough=0.3)
        b.inputs["Emission Color"].default_value = (0.2, 1.0, 0.35, 1)
        b.inputs["Emission Strength"].default_value = 2.0
    else:
        b = P(nt, color=(0.5, 0.5, 0.5), rough=0.4)
    if fade:
        fade_wrap(m, nt, out, b.outputs[0])
    else:
        nt.links.new(b.outputs[0], out.inputs[0])
    MATS[key] = m
    return m


def pedestal_mat():
    if "pedestal" in MATS:
        return MATS["pedestal"]
    m, nt, out = new_mat("pedestal_graphite")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    b = P(nt, color=(0.035, 0.037, 0.04), metal=0.85, rough=0.38, aniso=0.5)
    n = noise(nt, 4.0, 6.0, 0.6, tc.outputs["Object"])
    nt.links.new(maprange(nt, n.outputs[0], 0.3, 0.7, 0.32, 0.46), b.inputs["Roughness"])
    fade_wrap(m, nt, out, b.outputs[0])
    MATS["pedestal"] = m
    return m


def brushed_steel(name, base=(0.62, 0.63, 0.64), rough=(0.16, 0.30), aniso=0.65, tex=None, uv_brush=True,
                  radial=False, scratch=True):
    """Stainless steel: anisotropic (brushed) specular, roughness broken up by noise and fine scratches."""
    m, nt, out = new_mat(name)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    b = P(nt, color=base, metal=1.0, rough=rough[0], aniso=aniso)
    if radial:
        tg = nt.nodes.new("ShaderNodeTangent")
        tg.direction_type = "RADIAL"
        tg.axis = "Z"
        nt.links.new(tg.outputs[0], b.inputs["Tangent"])
    # roughness: broad smudges + streaks along the brush direction
    n1 = noise(nt, 3.0, 8.0, 0.65, tc.outputs["Object"])
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 120.0, 1.0)
    nt.links.new(tc.outputs["Object"], mp.inputs[0])
    n2 = noise(nt, 6.0, 4.0, 0.6, mp.outputs[0])
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "FLOAT"
    mx.inputs["Factor"].default_value = 0.45
    nt.links.new(n1.outputs[0], mx.inputs["A"])
    nt.links.new(n2.outputs[0], mx.inputs["B"])
    nt.links.new(maprange(nt, mx.outputs["Result"], 0.3, 0.7, rough[0], rough[1]), b.inputs["Roughness"])
    col_src = None
    if tex is not None:
        t = img(nt, tex)
        nt.links.new(tc.outputs["UV"], t.inputs[0])
        # the texture holds the photo's steel (scuffs, light falloff): use its detail, recentred on `base`
        hsv = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(t.outputs[0], hsv.inputs[0])
        lum = nt.nodes.new("ShaderNodeRGBToBW")
        nt.links.new(t.outputs[0], lum.inputs[0])
        cr = nt.nodes.new("ShaderNodeValToRGB")
        cr.color_ramp.elements[0].position = 0.35
        cr.color_ramp.elements[0].color = (base[0] * 0.78, base[1] * 0.78, base[2] * 0.8, 1)
        cr.color_ramp.elements[1].position = 0.85
        cr.color_ramp.elements[1].color = (base[0] * 1.08, base[1] * 1.08, base[2] * 1.08, 1)
        nt.links.new(lum.outputs[0], cr.inputs[0])
        nt.links.new(cr.outputs[0], b.inputs["Base Color"])
        col_src = lum.outputs[0]
        # darker texels (scuffs) are also rougher
        r2 = maprange(nt, lum.outputs[0], 0.4, 0.8, rough[1] + 0.06, rough[0])
        mx2 = nt.nodes.new("ShaderNodeMix")
        mx2.data_type = "FLOAT"
        mx2.inputs["Factor"].default_value = 0.5
        nt.links.new(maprange(nt, mx.outputs["Result"], 0.3, 0.7, rough[0], rough[1]), mx2.inputs["A"])
        nt.links.new(r2, mx2.inputs["B"])
        nt.links.new(mx2.outputs["Result"], b.inputs["Roughness"])
    if scratch:
        # micro scratches: stretched noise as a faint bump
        mp3 = nt.nodes.new("ShaderNodeMapping")
        mp3.inputs["Scale"].default_value = (2.0, 400.0, 2.0) if not radial else (300.0, 300.0, 3.0)
        nt.links.new(tc.outputs["Object"], mp3.inputs[0])
        n3 = noise(nt, 8.0, 2.0, 0.5, mp3.outputs[0])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.035
        bump.inputs["Distance"].default_value = 0.0001
        nt.links.new(n3.outputs[0], bump.inputs["Height"])
        nt.links.new(bump.outputs[0], b.inputs["Normal"])
    del col_src
    nt.links.new(b.outputs[0], out.inputs[0])
    return m


def onion_mat(name, tex):
    """Red onion: the SAM 3D colour, papery glossy outer skin, subsurface flesh, meridian veins root->tip."""
    m, nt, out = new_mat(name)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    t = img(nt, tex, interp="Linear")
    nt.links.new(tc.outputs["UV"], t.inputs[0])
    # angle around the onion's vertical axis -> meridian lines
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    at = nt.nodes.new("ShaderNodeMath")
    at.operation = "ARCTAN2"
    nt.links.new(sep.outputs[1], at.inputs[0])
    nt.links.new(sep.outputs[0], at.inputs[1])
    # wobble the veins with noise
    nz = noise(nt, 18.0, 3.0, 0.6, tc.outputs["Object"])
    wob = nt.nodes.new("ShaderNodeMath")
    wob.operation = "MULTIPLY_ADD"
    nt.links.new(nz.outputs[0], wob.inputs[0])
    wob.inputs[1].default_value = 0.35
    nt.links.new(at.outputs[0], wob.inputs[2])
    sn = nt.nodes.new("ShaderNodeMath")
    sn.operation = "SINE"
    k = nt.nodes.new("ShaderNodeMath")
    k.operation = "MULTIPLY"
    k.inputs[1].default_value = 34.0
    nt.links.new(wob.outputs[0], k.inputs[0])
    nt.links.new(k.outputs[0], sn.inputs[0])
    vein = maprange(nt, sn.outputs[0], 0.80, 1.0, 0.0, 1.0)          # thin lines
    # second, finer vein set (irregular spacing)
    k2 = nt.nodes.new("ShaderNodeMath")
    k2.operation = "MULTIPLY"
    k2.inputs[1].default_value = 77.0
    nt.links.new(wob.outputs[0], k2.inputs[0])
    sn2 = nt.nodes.new("ShaderNodeMath")
    sn2.operation = "SINE"
    nt.links.new(k2.outputs[0], sn2.inputs[0])
    vein2 = maprange(nt, sn2.outputs[0], 0.88, 1.0, 0.0, 0.6)
    vsum = nt.nodes.new("ShaderNodeMath")
    vsum.operation = "MAXIMUM"
    nt.links.new(vein, vsum.inputs[0])
    nt.links.new(vein2, vsum.inputs[1])
    vein = vsum.outputs[0]
    # colour: SAM 3D texel colour pulled toward deep burgundy-violet
    hs = nt.nodes.new("ShaderNodeHueSaturation")
    hs.inputs["Saturation"].default_value = 0.9
    hs.inputs["Value"].default_value = 0.45
    nt.links.new(t.outputs[0], hs.inputs["Color"])
    base = nt.nodes.new("ShaderNodeMix")
    base.data_type = "RGBA"
    base.blend_type = "MIX"
    base.inputs["Factor"].default_value = 0.85
    nt.links.new(hs.outputs[0], base.inputs[6])
    base.inputs[7].default_value = (0.10, 0.007, 0.032, 1)
    # veins: darker violet lines
    vm = nt.nodes.new("ShaderNodeMix")
    vm.data_type = "RGBA"
    vm.blend_type = "MULTIPLY"
    vm.inputs[7].default_value = (0.55, 0.42, 0.55, 1)
    nt.links.new(vein, vm.inputs["Factor"])
    nt.links.new(base.outputs[2], vm.inputs[6])
    # dry papery patches of outer skin (copper-tan, matte)
    pn = noise(nt, 9.0, 6.0, 0.62, tc.outputs["Object"])
    patch = maprange(nt, pn.outputs[0], 0.58, 0.68, 0.0, 1.0)
    pm = nt.nodes.new("ShaderNodeMix")
    pm.data_type = "RGBA"
    pm.blend_type = "MIX"
    pf = nt.nodes.new("ShaderNodeMath")
    pf.operation = "MULTIPLY"
    pf.inputs[1].default_value = 0.75
    nt.links.new(patch, pf.inputs[0])
    nt.links.new(pf.outputs[0], pm.inputs["Factor"])
    nt.links.new(vm.outputs[2], pm.inputs[6])
    pm.inputs[7].default_value = (0.30, 0.13, 0.10, 1)
    b = P(nt, rough=0.3, sss=0.1, sss_scale=0.005, spec=0.4, coat=0.2, coat_rough=0.2)
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.2, 0.3)
    nt.links.new(pm.outputs[2], b.inputs["Base Color"])
    nt.links.new(maprange(nt, patch, 0.0, 1.0, 0.3, 0.7), b.inputs["Roughness"])
    nt.links.new(maprange(nt, patch, 0.0, 1.0, 0.3, 0.0), b.inputs["Coat Weight"])
    hmix = nt.nodes.new("ShaderNodeMath")
    hmix.operation = "MULTIPLY_ADD"
    nt.links.new(patch, hmix.inputs[0])
    hmix.inputs[1].default_value = 0.6
    nt.links.new(vein, hmix.inputs[2])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.03
    bump.inputs["Distance"].default_value = 0.0002
    nt.links.new(hmix.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    nt.links.new(b.outputs[0], out.inputs[0])
    return m


def towel_mat(L, Wd):
    """Cotton kitchen towel: the photo decal colour, plain-weave height map (normal), fibre fuzz, sheen."""
    m, nt, out = new_mat("towel_cotton")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    dec = img(nt, EXP / "towel_decal_4x.png", ext="EXTEND")
    nt.links.new(tc.outputs["UV"], dec.inputs[0])
    pitch = 0.0011                                                # ~1.1 mm per thread
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (L / (16 * pitch), Wd / (16 * pitch), 1.0)
    nt.links.new(tc.outputs["UV"], mp.inputs[0])
    wv = img(nt, EXP / "weave_height.png", cs="Non-Color", interp="Linear")
    nt.links.new(mp.outputs[0], wv.inputs[0])
    # fuzz: fine noise
    fz = noise(nt, 2500.0, 4.0, 0.7, tc.outputs["Object"])
    hgt = nt.nodes.new("ShaderNodeMix")
    hgt.data_type = "FLOAT"
    hgt.inputs["Factor"].default_value = 0.25
    nt.links.new(wv.outputs[0], hgt.inputs["A"])
    nt.links.new(fz.outputs[0], hgt.inputs["B"])
    # colour: decal, slightly desaturated + darkened in the weave's valleys (thread shadowing)
    hs = nt.nodes.new("ShaderNodeHueSaturation")
    hs.inputs["Saturation"].default_value = 1.1
    hs.inputs["Value"].default_value = 1.0
    nt.links.new(dec.outputs[0], hs.inputs["Color"])
    occ = nt.nodes.new("ShaderNodeMix")
    occ.data_type = "RGBA"
    occ.blend_type = "MULTIPLY"
    nt.links.new(maprange(nt, wv.outputs[0], 0.0, 1.0, 0.0, 0.0), occ.inputs["Factor"])
    occ.inputs["Factor"].default_value = 0.55
    nt.links.new(hs.outputs[0], occ.inputs[6])
    shade = maprange(nt, wv.outputs[0], 0.0, 0.8, 0.62, 1.0)
    cmb = nt.nodes.new("ShaderNodeCombineColor")
    for i in range(3):
        nt.links.new(shade, cmb.inputs[i])
    nt.links.new(cmb.outputs[0], occ.inputs[7])
    b = P(nt, rough=0.92, sheen=0.85, sheen_rough=0.35, spec=0.3, sss=0.06, sss_scale=0.002)
    b.inputs["Sheen Tint"].default_value = (1.0, 0.98, 0.95, 1)
    nt.links.new(occ.outputs[2], b.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    bump.inputs["Distance"].default_value = 0.0006
    nt.links.new(hgt.outputs["Result"], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    nt.links.new(b.outputs[0], out.inputs[0])
    return m


def tex_principled(name, tex, rough=0.4, metal=0.0, coat=0.0, gain=1.0, interp="Linear"):
    m, nt, out = new_mat(name)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    t = img(nt, tex, interp=interp, ext="EXTEND")
    nt.links.new(tc.outputs["UV"], t.inputs[0])
    b = P(nt, rough=rough, metal=metal, coat=coat, coat_rough=0.1)
    if gain != 1.0:
        g = nt.nodes.new("ShaderNodeMix")
        g.data_type = "RGBA"
        g.blend_type = "MULTIPLY"
        g.inputs["Factor"].default_value = 1.0
        nt.links.new(t.outputs[0], g.inputs[6])
        g.inputs[7].default_value = (gain, gain, gain, 1)
        nt.links.new(g.outputs[2], b.inputs["Base Color"])
    else:
        nt.links.new(t.outputs[0], b.inputs["Base Color"])
    nt.links.new(b.outputs[0], out.inputs[0])
    return m


def simple(name, color, rough=0.5, metal=0.0, coat=0.0):
    m, nt, out = new_mat(name)
    b = P(nt, color=color, rough=rough, metal=metal, coat=coat, coat_rough=0.1)
    nt.links.new(b.outputs[0], out.inputs[0])
    return m


def tex_of_mat(k):
    t = int(M["mat_texid"][k]) if k >= 0 else -1
    return TEX / texfile[t] if t >= 0 and texfile[t] else None


# ------------------------------------------------------------------------------------------------ geometry
def mesh_from_mj(did, R=None, p=None, name="m"):
    """MuJoCo mesh -> Blender mesh, vertices in the geom frame (or R @ v + p), UVs and custom normals."""
    va, vn = int(M["mesh_vertadr"][did]), int(M["mesh_vertnum"][did])
    fa, fn = int(M["mesh_faceadr"][did]), int(M["mesh_facenum"][did])
    V = M["mesh_vert"][va:va + vn].astype(np.float64)
    F = M["mesh_face"][fa:fa + fn].astype(np.int64)
    if R is not None:
        V = V @ np.asarray(R).T + np.asarray(p)
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], F.tolist())
    me.update()
    ta = int(M["mesh_texcoordadr"][did])
    if ta >= 0:
        FT = M["mesh_facetexcoord"][fa:fa + fn].astype(np.int64) + ta
        T = M["mesh_texcoord"][FT.reshape(-1)]
        uv = me.uv_layers.new(name="UVMap")
        T2 = np.c_[T[:, 0], 1.0 - T[:, 1]]                     # MuJoCo v runs down the image, Blender's up
        uv.data.foreach_set("uv", T2.astype(np.float32).ravel())
    try:
        na = int(M["mesh_normaladr"][did])
        FN = M["mesh_facenormal"][fa:fa + fn].astype(np.int64) + na
        N = M["mesh_normal"][FN.reshape(-1)].astype(np.float64)
        if R is not None:
            N = N @ np.asarray(R).T
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
        me.normals_split_custom_set(N.tolist())
    except Exception as e:                                         # noqa: BLE001
        print("normals:", e)
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    return me


def prim_obj(t, size, name):
    """MuJoCo primitive (half-sizes) -> Blender mesh object (geom frame)."""
    if t == "box":
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        o = bpy.context.active_object
        o.scale = tuple(size[:3])
        bpy.ops.object.transform_apply(scale=True)
    elif t == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=size[0], depth=2 * size[1])
        o = bpy.context.active_object
    elif t == "capsule":
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=size[0], depth=2 * size[1])
        o = bpy.context.active_object
        for zc in (-size[1], size[1]):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=size[0], location=(0, 0, zc))
            sp = bpy.context.active_object
            sp.select_set(True)
            o.select_set(True)
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.join()
    elif t == "ellipsoid":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0)
        o = bpy.context.active_object
        o.scale = tuple(size[:3])
        bpy.ops.object.transform_apply(scale=True)
    elif t == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=size[0])
        o = bpy.context.active_object
    else:
        raise ValueError(t)
    o.name = name
    for c in o.users_collection:
        c.objects.unlink(o)
    if t in ("cylinder", "capsule", "ellipsoid", "sphere"):
        bpy.ops.object.select_all(action="DESELECT")
        o.data.polygons.foreach_set("use_smooth", [True] * len(o.data.polygons))
    return o


def bevel(o, w, seg=3, angle=50):
    b = o.modifiers.new("bevel", "BEVEL")
    b.width, b.segments = w, seg
    b.limit_method = "ANGLE"
    b.angle_limit = math.radians(angle)
    b.harden_normals = False
    o.data.polygons.foreach_set("use_smooth", [True] * len(o.data.polygons))
    try:
        o.modifiers.new("smooth", "NODES")
        o.modifiers.remove(o.modifiers["smooth"])
    except Exception:
        pass


def lathe(profile, n=96, name="lathe"):
    """profile: [(r, z)] -> closed revolve mesh."""
    V, F = [], []
    k = len(profile)
    for i in range(n):
        a = 2 * math.pi * i / n
        for r, z in profile:
            V.append((r * math.cos(a), r * math.sin(a), z))
    for i in range(n):
        j = (i + 1) % n
        for q in range(k - 1):
            F.append((i * k + q, j * k + q, j * k + q + 1, i * k + q + 1))
    me = bpy.data.meshes.new(name)
    me.from_pydata(V, [], F)
    me.update()
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    return me


def quat_mat(q):
    return np.array(Quaternion(q).to_matrix())


# ------------------------------------------------------------------------------------------------ build
C_SET, C_ARM, C_OBJ = coll("set"), coll("robot"), coll("objects")
geom_type = [GT[int(t)] for t in M["geom_type"]]
body_of = M["geom_bodyid"].astype(int)
moving = sorted({int(b) for b in body_of if b != 0})
EMPTY = {}
xpos, xquat = D["xpos"].astype(np.float64), D["xquat"].astype(np.float64)


def fcurve_set(idblock, path, index, values, interp="LINEAR"):
    ad = idblock.animation_data or idblock.animation_data_create()
    if ad.action is None:
        ad.action = bpy.data.actions.new(idblock.name + "_act")
    try:
        fc = ad.action.fcurves.new(path, index=index)
    except Exception:                                                     # Blender 5 layered actions
        fc = ad.action.fcurve_ensure_for_datablock(idblock, path, index=index)
    kp = fc.keyframe_points
    kp.add(len(values))
    co = np.empty(2 * len(values), np.float32)
    co[0::2] = np.arange(len(values))
    co[1::2] = values
    kp.foreach_set("co", co)
    kp.foreach_set("interpolation", [{"LINEAR": 1, "CONSTANT": 0, "BEZIER": 2}[interp]] * len(values))
    fc.update()
    return fc


def anim_obj(o, P_, Q_):
    o.rotation_mode = "QUATERNION"
    Q_ = Q_.copy()
    for i in range(1, len(Q_)):                                          # hemisphere continuity
        if np.dot(Q_[i], Q_[i - 1]) < 0:
            Q_[i] = -Q_[i]
    for k in range(3):
        fcurve_set(o, "location", k, P_[:, k])
    for k in range(4):
        fcurve_set(o, "rotation_quaternion", k, Q_[:, k])


for b in moving:
    e = bpy.data.objects.new(f"B_{bname[b]}", None)
    e.empty_display_size = 0.03
    (C_ARM if bname[b] in ARM_BODIES else C_OBJ).objects.link(e)
    anim_obj(e, xpos[:, b], xquat[:, b])
    EMPTY[b] = e

# ---- fade (arm + pedestal): 0 before the camera swing, smoothstep while the swing passes its middle
t = np.arange(NF) / FPS
T_SW0, T_SWING = 1.3 + 1.2 + 0.8, 4.0
x = (t - T_SW0) / T_SWING
fade = np.clip((x - 0.5) / 0.45, 0, 1)
fade = fade * fade * (3 - 2 * fade)
fade[t < T_SW0] = 0.0
fade[t >= T_SW0 + T_SWING] = 1.0
arm_on, arm_off = D["arm_on"].astype(bool), D["arm_off"].astype(bool)
assert ((fade > 0) == arm_on).all() and ((fade < 1) == arm_off).all(), "fade timeline mismatch"


def add_fade(o):
    o["fade"] = 1.0
    fcurve_set(o, '["fade"]', 0, fade)


bowl_body = bname.index("mixing_bowl_1")
spice_body = bname.index("spice_box_1")
SPICES = [(0.62, 0.36, 0.015), (0.42, 0.035, 0.015), (0.30, 0.17, 0.07), (0.075, 0.045, 0.03),
          (0.36, 0.30, 0.13), (0.55, 0.20, 0.03), (0.72, 0.70, 0.64)]


def spice_mat(i, col):
    m, nt, out = new_mat(f"spice_{i}")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = noise(nt, 1800.0, 2.0, 0.6, tc.outputs["Object"])
    n2 = noise(nt, 140.0, 4.0, 0.6, tc.outputs["Object"])
    b_ = P(nt, color=col, rough=0.85, spec=0.3)
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].color = (col[0] * 0.55, col[1] * 0.55, col[2] * 0.55, 1)
    cr.color_ramp.elements[1].color = (min(col[0] * 1.3, 1), min(col[1] * 1.3, 1), min(col[2] * 1.3, 1), 1)
    nt.links.new(n1.outputs[0], cr.inputs[0])
    nt.links.new(cr.outputs[0], b_.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.5
    bump.inputs["Distance"].default_value = 0.0006
    hm = nt.nodes.new("ShaderNodeMath")
    hm.operation = "MULTIPLY_ADD"
    nt.links.new(n1.outputs[0], hm.inputs[0])
    hm.inputs[1].default_value = 0.4
    nt.links.new(n2.outputs[0], hm.inputs[2])
    nt.links.new(hm.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b_.inputs["Normal"])
    nt.links.new(b_.outputs[0], out.inputs[0])
    return m


def bowl_profile(R_, z0, h, th=0.0008, rim=0.0025, flat=0.42, curve=1.6):
    prof = []
    for i in range(25):
        u = i / 24
        r = flat * R_ + (R_ - flat * R_) * math.sin(u * math.pi / 2) ** 0.9
        prof.append((r, z0 + h * (u ** curve) * 0.985))
    rr = prof[-1][0]
    z1 = z0 + h
    prof += [(rr + rim * math.cos(a), z1 - rim + rim * math.sin(a)) for a in np.linspace(-math.pi / 2, math.pi, 10)]
    for (r, z) in reversed(prof[:25]):
        prof.append((max(r - th, 0.0), z + th))
    return [(0.0, z0)] + prof + [(0.0, z0 + th)]


def spice_box(did, Rg, pg, parent):
    """Masala dabba (the SAM 3D spice box, re-modelled): steel tin, 7 steel cups of ground spices."""
    va, vn = int(M["mesh_vertadr"][did]), int(M["mesh_vertnum"][did])
    Vb = M["mesh_vert"][va:va + vn] @ Rg.T + pg
    z0, z1 = float(Vb[:, 2].min()), float(Vb[:, 2].max())
    cxy = (Vb[:, :2].max(0) + Vb[:, :2].min(0)) / 2
    R_ = float(np.percentile(np.linalg.norm(Vb[:, :2] - cxy, axis=1), 99.0))
    h = z1 - z0
    steel = brushed_steel("dabba_steel", base=(0.70, 0.70, 0.71), rough=(0.12, 0.26), aniso=0.5, radial=True)
    # tin: straight wall with a rolled rim
    prof = [(0.0, z0), (R_ - 0.004, z0), (R_, z0 + 0.004), (R_, z1 - 0.003)]
    prof += [(R_ - 0.0015 + 0.0015 * math.cos(a), z1 - 0.0015 + 0.0015 * math.sin(a))
             for a in np.linspace(0, math.pi, 8)]
    prof += [(R_ - 0.003, z1 - 0.004), (R_ - 0.003, z0 + 0.002), (0.0, z0 + 0.002)]
    tin = bpy.data.objects.new("spice_box_tin", lathe(prof, 128, "dabba"))
    tin.location = (float(cxy[0]), float(cxy[1]), 0.0)
    tin.data.materials.append(steel)
    rc = 0.30 * R_
    centres = [(0.0, 0.0)] + [(0.64 * R_ * math.cos(a), 0.64 * R_ * math.sin(a))
                              for a in np.linspace(0, 2 * math.pi, 7)[:-1] + 0.3]
    for i, (dx, dy) in enumerate(centres):
        ch = 0.72 * h
        cup = bpy.data.objects.new(f"spice_cup_{i}", lathe(bowl_profile(rc, z0 + 0.003, ch, rim=0.0012,
                                                                        flat=0.55, curve=1.3), 64, "cup"))
        cup.location = (float(cxy[0] + dx), float(cxy[1] + dy), 0.0)
        cup.data.materials.append(steel)
        # spice: a slightly domed disc inside the cup
        zf = z0 + 0.003 + ch * 0.86
        rf = rc * 0.93
        prof = [(0.0, zf + 0.004)] + [(rf * u, zf + 0.004 * (1 - u * u) - 0.001 * u ** 4) for u in np.linspace(0.1, 1, 8)]
        prof += [(rf, zf - 0.01), (0.0, zf - 0.01)]
        sp = bpy.data.objects.new(f"spice_{i}", lathe(prof, 64, "spice"))
        sp.location = cup.location
        sp.data.materials.append(spice_mat(i, SPICES[i % len(SPICES)]))
        for ob in (cup, sp):
            C_OBJ.objects.link(ob)
            ob.parent = parent
    return tin
pedestal_geoms = []
created = []
for g in range(len(geom_type)):
    rgba = M["geom_rgba"][g]
    grp = int(M["geom_group"][g])
    if rgba[3] == 0 or grp > 2:
        continue
    k = int(M["geom_matid"][g])
    mat_nm = mname[k] if k >= 0 else ""
    if mat_nm in DROP_MATS or geom_type[g] == "plane":
        continue
    if mat_nm.startswith("dmat"):          # photo decals of the phone: replaced by a modelled phone back
        continue
    if body_of[g] == 0 and mat_nm == "" and abs(rgba[0] - 0.62) < 0.01:      # skirting board
        continue
    b = int(body_of[g])
    gt = geom_type[g]
    nm = gname[g] or f"g{g}_{bname[b]}"
    # geom frame relative to its body (static geoms: world)
    Rg = quat_mat(M["geom_quat"][g])
    pg = M["geom_pos"][g].astype(np.float64)
    if gt == "mesh":
        did = int(M["geom_dataid"][g])
        if b == bowl_body:
            # SAM 3D bowl -> clean lathed steel bowl fitted to the SAM 3D mesh's extent (in the body frame)
            va, vn = int(M["mesh_vertadr"][did]), int(M["mesh_vertnum"][did])
            Vb = M["mesh_vert"][va:va + vn] @ Rg.T + pg
            z0, z1 = float(Vb[:, 2].min()), float(Vb[:, 2].max())
            cxy = (Vb[:, :2].max(0) + Vb[:, :2].min(0)) / 2
            R_ = float(np.percentile(np.linalg.norm(Vb[:, :2] - cxy, axis=1), 99.5))
            h = z1 - z0
            th = 0.0012
            prof = []
            for i in range(25):                                         # outer wall: bottom -> rim
                u = i / 24
                r = 0.42 * R_ + (R_ - 0.42 * R_) * math.sin(u * math.pi / 2) ** 0.9
                z = z0 + h * (u ** 1.6) * 0.985
                prof.append((r, z))
            rr = prof[-1][0]
            prof += [(rr + 0.0035 * math.cos(a), z1 - 0.0035 + 0.0035 * math.sin(a))
                     for a in np.linspace(-math.pi / 2, math.pi, 10)]                         # rolled rim
            for (r, z) in reversed(prof[:25]):                          # inner wall
                prof.append((max(r - th, 0.0), z + th))
            prof = [(0.0, z0)] + prof + [(0.0, z0 + th)]
            me = lathe(prof, 128, "bowl")
            o = bpy.data.objects.new("mixing_bowl_steel", me)
            o.location = (float(cxy[0]), float(cxy[1]), 0.0)
            mat = brushed_steel("bowl_steel", base=(0.78, 0.78, 0.79), rough=(0.07, 0.16), aniso=0.55,
                                radial=True)
            o.data.materials.append(mat)
            sub = o.modifiers.new("sub", "SUBSURF")
            sub.levels = sub.render_levels = 1
        elif b == spice_body:
            o = spice_box(did, Rg, pg, EMPTY[b])
        else:
            me = mesh_from_mj(did, Rg, pg, name=nm)
            o = bpy.data.objects.new(nm, me)
    else:
        o = prim_obj(gt, M["geom_size"][g], nm)
        o.data.transform(Matrix.Translation(Vector(pg)) @ Matrix(Rg.tolist()).to_4x4())
    # ---- material
    bn = bname[b]
    if bn in ARM_BODIES:
        mat = franka_mat(mat_nm if mat_nm in ("white", "off_white", "black", "light_blue", "green") else "white")
    elif b in (bowl_body, spice_body):
        pass
    elif b == 0 and gt == "box" and mat_nm == "" and rgba[3] > 0 and abs(pg[0] - 0.0) < 0.3 \
            and pg[0] + M["geom_size"][g][0] < TABLE_C.x - 0.29 + 0.05:
        mat = pedestal_mat()
        pedestal_geoms.append(o)
        bevel(o, 0.006, 3)
    elif mat_nm == "tablemat":
        mat = brushed_steel("table_top_steel", base=(0.66, 0.665, 0.67), rough=(0.18, 0.34), aniso=0.6,
                            tex=tex_of_mat(k))
    elif mat_nm == "hero_steel":
        mat = MATS.get("hero_steel") or brushed_steel("frame_steel", base=(0.62, 0.62, 0.63), rough=(0.2, 0.32))
        MATS["hero_steel"] = mat
        if gt == "box":
            bevel(o, 0.004, 2)
    elif mat_nm == "hero_pot":
        mat = MATS.get("hero_pot") or brushed_steel("pot_steel", base=(0.74, 0.74, 0.75), rough=(0.1, 0.2),
                                                    radial=True)
        MATS["hero_pot"] = mat
        bevel(o, 0.004, 3)
    elif mat_nm == "hero_blade":
        mat = MATS.get("hero_blade") or brushed_steel("blade_steel", base=(0.80, 0.80, 0.81), rough=(0.08, 0.18),
                                                      aniso=0.8)
        MATS["hero_blade"] = mat
        if gt == "mesh":
            bevel(o, 0.0004, 2, 30)
    elif mat_nm == "hero_black":
        mat = simple("knife_handle", (0.008, 0.008, 0.009), rough=0.3, coat=0.0)
        mat.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.3
    elif mat_nm.startswith("sam3d_onion"):
        mat = onion_mat(mat_nm, tex_of_mat(k))
        sub = o.modifiers.new("sub", "SUBSURF")
        sub.levels = sub.render_levels = 1
    elif mat_nm.startswith("sam3d_spice"):
        mat = tex_principled("spice_box", tex_of_mat(k), rough=0.3, metal=0.35, coat=0.2, gain=1.1)
    elif mat_nm.startswith("dmat"):            # phone decals (top = the photo's phone back)
        mat = tex_principled(mat_nm, tex_of_mat(k), rough=0.22, coat=0.6, interp="Cubic")
    elif bn == "smartphone_1" and gt == "box":
        # phone lying face down (as in the photo): dark frosted-glass back, satin frame, camera module
        mat = simple("phone_back", (0.015, 0.016, 0.019), rough=0.3, metal=0.0, coat=0.0)
        mat.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.25
        bevel(o, 0.0035, 4, 60)
        hx, hy, hz = (float(v) for v in M["geom_size"][g][:3])
        long_x = hx >= hy
        mod_c = (pg[0] + (hx - 0.022) * (1 if long_x else 1), pg[1] + (hy - 0.022), pg[2] + hz)
        if not long_x:
            mod_c = (pg[0] + (hx - 0.022), pg[1] + (hy - 0.022), pg[2] + hz)
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        cm = bpy.context.active_object
        for c_ in cm.users_collection:
            c_.objects.unlink(cm)
        cm.name = "phone_cam_module"
        cm.data.transform(Matrix.Diagonal((0.034, 0.034, 0.0024, 1.0)))
        cm.data.transform(Matrix.Translation(Vector(mod_c) + Vector((0, 0, 0.0008))))
        bevel(cm, 0.005, 5, 60)
        cm.data.materials.append(simple("phone_cam_glass", (0.05, 0.052, 0.058), rough=0.12, coat=1.0))
        C_OBJ.objects.link(cm)
        cm.parent = EMPTY[b]
        for (dx, dy) in ((-0.0075, 0.0075), (0.0075, -0.0075), (-0.0075, -0.0075)):
            lens = prim_obj("cylinder", (0.0052, 0.0016, 0), "phone_lens")
            lens.data.transform(Matrix.Translation(Vector(mod_c) + Vector((dx, dy, 0.0028))))
            lens.data.materials.append(simple("phone_lens_mat", (0.004, 0.004, 0.006), rough=0.05, metal=0.0,
                                              coat=1.0) if "phone_lens_mat" not in bpy.data.materials
                                       else bpy.data.materials["phone_lens_mat"])
            bevel(lens, 0.0008, 2, 40)
            C_OBJ.objects.link(lens)
            lens.parent = EMPTY[b]
    else:
        print("default material for", nm, mat_nm, gt, bn)
        mat = simple(f"m_{nm}", tuple(M["geom_rgba"][g][:3]), rough=0.4)
    if b not in (bowl_body, spice_body):
        o.data.materials.append(mat)
    if b == 0:
        C_SET.objects.link(o)
    else:
        (C_ARM if bn in ARM_BODIES else C_OBJ).objects.link(o)
        o.parent = EMPTY[b]
    if bn in ARM_BODIES or o in pedestal_geoms:
        add_fade(o)
    created.append(o)

# ---- towel: flex vertices per frame -> .pc2 point cache -> Mesh Cache modifier, subdivision, solidify
FL = D["flex"].astype(np.float32)                                         # (NF, nv, 3) world coords
nv = int(M["flex_vertnum"][0])
E = M["flex_elem"].reshape(-1, 3)[:int(M["flex_elemnum"][0])].astype(int)
ta = int(M["flex_texcoordadr"][0]) if "flex_texcoordadr" in M else 0
TC = M["flex_texcoord"][ta:ta + nv]
pc2 = EXP / "towel_flex.pc2"
with open(pc2, "wb") as f:
    f.write(struct.pack("<12siiffi", b"POINTCACHE2\0", 1, nv, 0.0, 1.0, NF))
    f.write(FL.astype("<f4").tobytes())
me = bpy.data.meshes.new("towel")
me.from_pydata(FL[0].tolist(), [], E.tolist())
me.update()
# consistent upward normals
for p in me.polygons:
    pass
uvl = me.uv_layers.new(name="UVMap")
loops_v = np.empty(len(me.loops), np.int64)
me.loops.foreach_get("vertex_index", loops_v)
T2 = np.c_[TC[loops_v, 0], 1.0 - TC[loops_v, 1]]
uvl.data.foreach_set("uv", T2.astype(np.float32).ravel())
me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
towel = bpy.data.objects.new("towel_flex", me)
C_OBJ.objects.link(towel)
mc = towel.modifiers.new("flex_cache", "MESH_CACHE")
mc.cache_format = "PC2"
mc.filepath = str(pc2)
mc.frame_start = 0.0
mc.play_mode = "SCENE"
mc.forward_axis, mc.up_axis = "POS_Y", "POS_Z"
sub = towel.modifiers.new("subdiv", "SUBSURF")
sub.levels, sub.render_levels = 2, 3
sub.uv_smooth = "PRESERVE_BOUNDARIES"
sol = towel.modifiers.new("thickness", "SOLIDIFY")
sol.thickness = 0.0042
sol.offset = -1.0
sol.use_even_offset = True
sol.use_rim = True
sol.use_quality_normals = True
P0 = FL[0][:, :2] - FL[0][:, :2].mean(0)
_, _, Vt = np.linalg.svd(P0, full_matrices=False)
ext = (P0 @ Vt.T).max(0) - (P0 @ Vt.T).min(0)                          # towel extent along its own axes
L_t, W_t = float(max(ext)), float(min(ext))
# which UV axis runs along the long side
du = np.ptp(TC[:, 0]) / max(np.ptp(P0 @ Vt[0]), 1e-6)
if np.corrcoef(TC[:, 0], P0 @ Vt[0])[0, 1] ** 2 < 0.5:
    L_t, W_t = W_t, L_t
towel.data.materials.append(towel_mat(L_t, W_t))
# keep the sheet's normals pointing up
up = sum(p.normal.z for p in me.polygons)
if up < 0:
    me.flip_normals()

# ------------------------------------------------------------------------------------------------ studio
# floor: near-black glossy with a glowing grid that fades out from the table
bpy.ops.mesh.primitive_plane_add(size=30.0, location=(TABLE_C.x - 0.3, 0.0, FLOOR_Z - 0.0015))  # not coplanar with the pedestal plate (fade-out)
floor = bpy.context.active_object
floor.name = "studio_floor"
for c in floor.users_collection:
    c.objects.unlink(floor)
C_SET.objects.link(floor)
m, nt, out = new_mat("studio_grid_floor")
tc = nt.nodes.new("ShaderNodeTexCoord")
sep = nt.nodes.new("ShaderNodeSeparateXYZ")
nt.links.new(tc.outputs["Object"], sep.inputs[0])


def grid_lines(spacing, width):
    """-> socket: 1 on grid lines (anti-aliased by width), from object x/y (object at the floor centre)."""
    acc = None
    for ax in (0, 1):
        d = nt.nodes.new("ShaderNodeMath")
        d.operation = "PINGPONG"                                    # triangle wave: distance to nearest line
        nt.links.new(sep.outputs[ax], d.inputs[0])
        d.inputs[1].default_value = spacing / 2
        ln = maprange(nt, d.outputs[0], 0.0, width, 1.0, 0.0)
        if acc is None:
            acc = ln
        else:
            mx = nt.nodes.new("ShaderNodeMath")
            mx.operation = "MAXIMUM"
            nt.links.new(acc, mx.inputs[0])
            nt.links.new(ln, mx.inputs[1])
            acc = mx.outputs[0]
    return acc


minor = grid_lines(0.25, 0.0022)
major = grid_lines(1.0, 0.0035)
lines = nt.nodes.new("ShaderNodeMath")
lines.operation = "MULTIPLY_ADD"
nt.links.new(minor, lines.inputs[0])
lines.inputs[1].default_value = 0.45
nt.links.new(major, lines.inputs[2])
# radial fade (object origin is 0.3 m behind the table centre, roughly under the robot + table)
ln_ = nt.nodes.new("ShaderNodeVectorMath")
ln_.operation = "LENGTH"
cmb = nt.nodes.new("ShaderNodeCombineXYZ")
nt.links.new(sep.outputs[0], cmb.inputs[0])
nt.links.new(sep.outputs[1], cmb.inputs[1])
nt.links.new(cmb.outputs[0], ln_.inputs[0])
fd = maprange(nt, ln_.outputs["Value"], 0.4, 2.8, 1.0, 0.0)
fd2 = nt.nodes.new("ShaderNodeMath")
fd2.operation = "POWER"
nt.links.new(fd, fd2.inputs[0])
fd2.inputs[1].default_value = 1.6
glow = nt.nodes.new("ShaderNodeMath")
glow.operation = "MULTIPLY"
nt.links.new(lines.outputs[0], glow.inputs[0])
nt.links.new(fd2.outputs[0], glow.inputs[1])
gs = nt.nodes.new("ShaderNodeMath")
gs.operation = "MULTIPLY"
nt.links.new(glow.outputs[0], gs.inputs[0])
gs.inputs[1].default_value = 0.35
fb = P(nt, color=(0.006, 0.0065, 0.008), rough=0.4, spec=0.22, coat=0.0)
n_fl = noise(nt, 2.5, 6.0, 0.6, tc.outputs["Object"])
nt.links.new(maprange(nt, n_fl.outputs[0], 0.3, 0.7, 0.32, 0.55), fb.inputs["Roughness"])
fb.inputs["Emission Color"].default_value = (0.35, 0.62, 1.0, 1)
nt.links.new(gs.outputs[0], fb.inputs["Emission Strength"])
nt.links.new(fb.outputs[0], out.inputs[0])
floor.data.materials.append(m)

# world: near-black void with the faintest cool gradient
wd = bpy.data.worlds.new("void")
scn.world = wd
wd.use_nodes = True
wn = wd.node_tree
bg = next(n for n in wn.nodes if n.type == "BACKGROUND")
bg.inputs[0].default_value = (0.0025, 0.0028, 0.0035, 1)
bg.inputs[1].default_value = 1.0


def area(name, loc, target, size, power, color=(1, 1, 1), shape="RECTANGLE", size_y=None, spread=180):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = shape
    ld.size = size
    if size_y is not None:
        ld.size_y = size_y
    ld.energy = power
    ld.color = color
    try:
        ld.spread = math.radians(spread)
    except Exception:
        pass
    o = bpy.data.objects.new(name, ld)
    C_SET.objects.link(o)
    o.location = loc
    d = (Vector(target) - Vector(loc)).normalized()
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = d.to_track_quat("-Z", "Y")
    o.visible_camera = False
    return o


def sph(c, dist, az, el):
    a, e = math.radians(az), math.radians(el)
    return Vector(c) + dist * Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))


TGT = TABLE_C + Vector((-0.15, 0.0, 0.15))
LIGHTS = {
    # key: big soft box high camera-left, slightly warm
    "key": area("key_softbox", sph(TGT, 2.3, -170, 58), TGT, 1.4, 110, (1.0, 0.94, 0.86), size_y=0.9),
    # rim: two cool strips behind the set (opposite the camera), carving the robot and steel edges
    "rim_a": area("rim_strip_a", sph(TGT, 2.2, 75, 14), TGT, 0.25, 140, (0.72, 0.84, 1.0), size_y=2.0, spread=60),
    "rim_b": area("rim_strip_b", sph(TGT, 2.4, 150, 22), TGT, 0.25, 90, (0.78, 0.88, 1.0), size_y=1.8, spread=60),
    # fill: wide and dim from camera right
    "fill": area("fill_wide", sph(TGT, 2.6, -95, 22), TGT, 2.5, 14, (0.85, 0.9, 1.0), size_y=1.5),
    # overhead strip for long reflections in the steel table top
    "top": area("top_strip", TGT + Vector((0.1, 0.0, 1.6)), TGT, 0.3, 35, (1.0, 1.0, 1.0), size_y=1.6),
}

# ------------------------------------------------------------------------------------------------ camera
CAM = D["cam"].astype(np.float64)                                         # C(3) f(3) up(3) fy off(2)
cd = bpy.data.cameras.new("hero_cam")
cd.sensor_fit = "VERTICAL"
cd.sensor_height = 24.0
cd.sensor_width = 24.0 * W / H
cd.clip_start, cd.clip_end = 0.02, 60.0
cam = bpy.data.objects.new("hero_cam", cd)
scn.collection.objects.link(cam)
scn.camera = cam
locs, quats = CAM[:, 0:3], []
for i in range(NF):
    f, upv = CAM[i, 3:6], CAM[i, 6:9]
    r = np.cross(f, upv)
    Rm = Matrix(np.c_[r, upv, -f].tolist())
    quats.append(list(Rm.to_quaternion()))
quats = np.array(quats)
if ORBIT:
    # alternative camera: a slow 70 deg orbit around the table at the hero distance/elevation (task frames)
    tt = np.clip((t - (T_SW0 + T_SWING)) / (t[-1] - (T_SW0 + T_SWING)), 0, 1)
    look = Vector((0.36, 0.02, 0.05))
    for i in range(NF):
        az = -150 + 75 * (tt[i] * tt[i] * (3 - 2 * tt[i]))
        Cp = sph(look, 1.75, az, 34)
        q = (look - Cp).normalized().to_track_quat("-Z", "Y")
        locs[i] = np.array(Cp)
        quats[i] = list(q)
anim_obj(cam, locs, quats)
fy = CAM[:, 9]
offx, offy = CAM[:, 10], CAM[:, 11]
if ORBIT:
    fy[:] = 1500.0
    offx[:] = 0
    offy[:] = 0
fcurve_set(cd, "lens", 0, fy * 24.0 / H)
fcurve_set(cd, "shift_x", 0, -offx / W)
fcurve_set(cd, "shift_y", 0, offy / W)

# depth of field: focus between the table centre and the hand, smoothed
hand = bname.index("hand")
foc = 0.45 * xpos[:, hand] + 0.55 * np.array([0.52, 0.05, 0.06])
foc[fade < 1] = (foc[fade < 1] * fade[fade < 1, None] + np.array([0.48, 0.0, 0.02]) * (1 - fade[fade < 1, None]))
ker = np.exp(-0.5 * (np.arange(-24, 25) / 9.0) ** 2)
ker /= ker.sum()
foc = np.stack([np.convolve(np.pad(foc[:, k], 24, mode="edge"), ker, "valid") for k in range(3)], 1)
fwd = np.array([(Quaternion(q).to_matrix() @ Vector((0, 0, -1)))[:] for q in quats])
fdist = np.einsum("ij,ij->i", foc - locs, fwd)
cd.dof.use_dof = True
cd.dof.aperture_fstop = float(arg("--fstop", 2.0))
cd.dof.aperture_blades = 7
cd.dof.aperture_ratio = 1.0
fcurve_set(cd, "dof.focus_distance", 0, fdist)

# ------------------------------------------------------------------------------------------------ render
scn.render.engine = "CYCLES"
cy = scn.cycles
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for dv in prefs.devices:
        dv.use = True
    cy.device = "GPU"
except Exception as e:                                                       # noqa: BLE001
    print("GPU setup failed:", e)
cy.samples = int(arg("--samples", 96))
cy.use_adaptive_sampling = True
cy.adaptive_threshold = float(arg("--thresh", 0.025))
cy.adaptive_min_samples = 16
cy.use_denoising = True
try:
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.denoising_use_gpu = True
except Exception:
    pass
cy.max_bounces, cy.diffuse_bounces, cy.glossy_bounces = 8, 3, 5
cy.transmission_bounces, cy.transparent_max_bounces = 6, 64
cy.sample_clamp_indirect = 4.0
cy.caustics_reflective = cy.caustics_refractive = False
cy.blur_glossy = 1.0
scn.render.use_motion_blur = True
scn.render.motion_blur_shutter = 0.5
scn.render.use_persistent_data = True
scn.render.film_transparent = False
res = arg("--res", f"{W}x{H}").split("x")
scn.render.resolution_x, scn.render.resolution_y = int(res[0]), int(res[1])
scn.render.resolution_percentage = 100
scn.render.image_settings.file_format = "PNG"
scn.render.image_settings.color_depth = "8"
vs = scn.view_settings
try:
    vs.view_transform = "AgX"
    vs.look = arg("--look", "AgX - Medium High Contrast")
except Exception as e:                                                       # noqa: BLE001
    print("look:", e)
vs.exposure = float(arg("--exposure", -0.6))
scn.render.use_overwrite = False
scn.render.use_placeholder = True

# ------------------------------------------------------------------------------------------------ outputs
info = {"frames": NF, "geoms": len(created), "towel_size": [L_t, W_t], "bowl": "lathe",
        "fade_frames": int(((fade > 0) & (fade < 1)).sum())}
print("BUILD", json.dumps(info))
if arg("--save"):
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(arg("--save")).resolve()), compress=True)
if arg("--pilot") is not None:
    scn.frame_set(int(arg("--pilot")))
    scn.render.filepath = str(arg("--out", str(EXP / "pilot.png")))
    scn.render.use_overwrite = True
    bpy.ops.render.render(write_still=True)
