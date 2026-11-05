"""Re-modelled objects and materials for the refined hero renders (imported by build_refined.py inside Blender).

Visual only. Every model is built in its body's frame and sized to the twin's physics primitive (scene json
`shape`), so it moves exactly as the MuJoCo body does.  Objects without a builder keep their SAM 3D mesh (or the
primitive) with a material from `sam_mat`.
"""
import math
import re

import bpy
import numpy as np
from mathutils import Matrix, Vector

G = {}


def setup(g):
    G.update(g)


def _g(n):
    return G[n]


def link(o, b, name=None):
    if name:
        o.name = name
    _g("C_OBJ").objects.link(o)
    o.parent = _g("EMPTY")[b]
    return o


def z_bottom(b):
    """Bottom of the object in its body frame (objects start upright, resting on the table at z=0)."""
    return -float(_g("xpos")[0, b, 2])


def lin(c):
    """sRGB 0..1 (the scene's mask colours) -> linear."""
    return tuple((v / 12.92) if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c[:3])


def mesh_obj(name, V, F, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in V], [], [tuple(f) for f in F])
    me.update()
    if smooth:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    return bpy.data.objects.new(name, me)


def lathe_obj(name, prof, n=96):
    return bpy.data.objects.new(name, _g("lathe")(prof, n, name))


# ------------------------------------------------------------------------------------------------ materials
def principled(name, color, rough=0.5, **kw):
    m, nt, out = _g("new_mat")(name)
    b = _g("P")(nt, color=color, rough=rough, **kw)
    nt.links.new(b.outputs[0], out.inputs[0])
    return m, nt, b


def ceramic(name, color, rough=0.12, speck=0.0):
    m, nt, b = principled(name, color, rough, coat=0.6, coat_rough=0.05, spec=0.5, sss=0.05, sss_scale=0.003)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = _g("noise")(nt, 6.0, 4.0, 0.55, tc.outputs["Object"])
    nt.links.new(_g("maprange")(nt, n.outputs[0], 0.3, 0.7, rough * 0.7, rough * 1.6), b.inputs["Roughness"])
    if speck:
        n2 = _g("noise")(nt, 700.0, 2.0, 0.5, tc.outputs["Object"])
        mx = nt.nodes.new("ShaderNodeMix")
        mx.data_type = "RGBA"
        nt.links.new(_g("maprange")(nt, n2.outputs[0], 0.62, 0.68, 0.0, speck), mx.inputs["Factor"])
        mx.inputs[6].default_value = (*color, 1)
        mx.inputs[7].default_value = (color[0] * 0.35, color[1] * 0.33, color[2] * 0.3, 1)
        nt.links.new(mx.outputs[2], b.inputs["Base Color"])
    return m


def wood(name, base, dark, scale=(1.0, 1.0, 60.0), rough=0.45, coat=0.15, axis_scale=None):
    """Long-grain wood: stretched noise along the object's x axis (rings) + fine pores."""
    m, nt, b = principled(name, base, rough, coat=coat, coat_rough=0.3, spec=0.4)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = axis_scale or (2.0, 40.0, 40.0)
    nt.links.new(tc.outputs["Object"], mp.inputs[0])
    w = nt.nodes.new("ShaderNodeTexWave")
    w.wave_type = "RINGS"
    w.rings_direction = "X"
    w.inputs["Scale"].default_value = 1.2
    w.inputs["Distortion"].default_value = 6.0
    w.inputs["Detail"].default_value = 3.0
    nt.links.new(mp.outputs[0], w.inputs["Vector"])
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].color = (*dark, 1)
    cr.color_ramp.elements[1].color = (*base, 1)
    cr.color_ramp.elements[0].position = 0.25
    cr.color_ramp.elements[1].position = 0.75
    nt.links.new(w.outputs["Fac"], cr.inputs[0])
    nt.links.new(cr.outputs[0], b.inputs["Base Color"])
    n = _g("noise")(nt, 900.0, 2.0, 0.5, tc.outputs["Object"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.0003
    nt.links.new(n.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    return m


def rustic_top_mat():
    """Reclaimed-plank farmhouse top (t6_31 photo): planks along world y with dark seams, long grain, dark knots,
    worn lighter patches, matte oiled finish.  Colours sampled from the photo by eye."""
    lin8_ = lambda c: lin([v / 255 for v in c])
    m, nt, out = _g("new_mat")("table_top")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    co = tc.outputs["Object"]
    b = _g("P")(nt, rough=0.62, coat=0.05, coat_rough=0.4, spec=0.35)
    # long grain: noise stretched along y
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (140.0, 1.2, 140.0)
    nt.links.new(co, mp.inputs[0])
    grain0 = _g("noise")(nt, 1.5, 14.0, 0.75, mp.outputs[0])
    wv = nt.nodes.new("ShaderNodeTexWave")            # long grain lines along the planks (world y)
    wv.wave_type = "BANDS"
    wv.bands_direction = "X"
    wv.inputs["Scale"].default_value = 45.0
    wv.inputs["Distortion"].default_value = 9.0
    wv.inputs["Detail"].default_value = 6.0
    wv.inputs["Detail Scale"].default_value = 1.5
    nt.links.new(co, wv.inputs["Vector"])
    grain = nt.nodes.new("ShaderNodeMix")
    grain.data_type = "FLOAT"
    grain.inputs["Factor"].default_value = 0.6
    nt.links.new(grain0.outputs[0], grain.inputs["A"])
    nt.links.new(wv.outputs["Fac"], grain.inputs["B"])
    grain.outputs[0].name
    # planks: seams every 0.19 m in x, each plank a slightly different tone
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(co, sep.inputs[0])
    dv = nt.nodes.new("ShaderNodeMath"); dv.operation = "DIVIDE"; dv.inputs[1].default_value = 0.19
    nt.links.new(sep.outputs[0], dv.inputs[0])
    fr = nt.nodes.new("ShaderNodeMath"); fr.operation = "FRACT"
    nt.links.new(dv.outputs[0], fr.inputs[0])
    fl = nt.nodes.new("ShaderNodeMath"); fl.operation = "FLOOR"
    nt.links.new(dv.outputs[0], fl.inputs[0])
    wn = nt.nodes.new("ShaderNodeTexWhiteNoise"); wn.noise_dimensions = "1D"
    nt.links.new(fl.outputs[0], wn.inputs["W"])
    seam = _g("maprange")(nt, fr.outputs[0], 0.0, 0.012, 0.0, 1.0)            # 0 in the seam
    # knots / worn patches
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 9.0
    nt.links.new(mp.outputs[0], vor.inputs["Vector"])
    knot = _g("maprange")(nt, vor.outputs["Distance"], 0.0, 0.06, 1.0, 0.0)
    wear = _g("noise")(nt, 3.0, 4.0, 0.6, co)
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].color = (*lin8_((30, 18, 10)), 1)
    cr.color_ramp.elements[1].color = (*lin8_((118, 78, 44)), 1)
    cr.color_ramp.elements.new(0.5).color = (*lin8_((72, 46, 26)), 1)
    mix_g = nt.nodes.new("ShaderNodeMath"); mix_g.operation = "MULTIPLY_ADD"
    nt.links.new(grain.outputs[0], mix_g.inputs[0]); mix_g.inputs[1].default_value = 0.7
    nt.links.new(wn.outputs["Value"], mix_g.inputs[2])
    sc = nt.nodes.new("ShaderNodeMath"); sc.operation = "MULTIPLY"; sc.inputs[1].default_value = 0.62
    nt.links.new(mix_g.outputs[0], sc.inputs[0])
    nt.links.new(sc.outputs[0], cr.inputs[0])
    c1 = nt.nodes.new("ShaderNodeMix"); c1.data_type = "RGBA"; c1.blend_type = "MULTIPLY"
    nt.links.new(knot, c1.inputs["Factor"]); nt.links.new(cr.outputs[0], c1.inputs[6])
    c1.inputs[7].default_value = (0.25, 0.2, 0.17, 1)
    c2 = nt.nodes.new("ShaderNodeMix"); c2.data_type = "RGBA"; c2.blend_type = "MULTIPLY"
    inv = nt.nodes.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0
    nt.links.new(seam, inv.inputs[1])
    nt.links.new(inv.outputs[0], c2.inputs["Factor"]); nt.links.new(c1.outputs[2], c2.inputs[6])
    c2.inputs[7].default_value = (0.08, 0.06, 0.05, 1)
    c3 = nt.nodes.new("ShaderNodeMix"); c3.data_type = "RGBA"; c3.blend_type = "SCREEN"
    nt.links.new(_g("maprange")(nt, wear.outputs[0], 0.6, 0.8, 0.0, 0.12), c3.inputs["Factor"])
    nt.links.new(c2.outputs[2], c3.inputs[6]); c3.inputs[7].default_value = (*lin8_((200, 160, 112)), 1)
    nt.links.new(c3.outputs[2], b.inputs["Base Color"])
    bh = nt.nodes.new("ShaderNodeMath"); bh.operation = "MULTIPLY"
    nt.links.new(grain.outputs[0], bh.inputs[0]); nt.links.new(seam, bh.inputs[1])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.45
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(bh.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    nt.links.new(b.outputs[0], out.inputs[0])
    return m


def table_top_mat(tex):
    """The table top: the hero texture (marble / laminate / wood) as albedo, a polished clear coat."""
    if G.get("TOP_KIND") == "rustic":         # photo wood planks (plank_tex.py): matte oiled, grain bump
        m, nt, out = _g("new_mat")("table_top")
        tc = nt.nodes.new("ShaderNodeTexCoord")
        t = _g("img")(nt, tex, interp="Cubic", ext="REPEAT")
        nt.links.new(tc.outputs["UV"], t.inputs[0])
        b = _g("P")(nt, rough=0.6, coat=0.06, coat_rough=0.35, spec=0.35)
        gr = nt.nodes.new("ShaderNodeMix")          # grade to the photo's darker, warmer tone
        gr.data_type = "RGBA"
        gr.blend_type = "MULTIPLY"
        gr.inputs["Factor"].default_value = 1.0
        nt.links.new(t.outputs[0], gr.inputs[6])
        gr.inputs[7].default_value = (*(G.get("TOP_TINT") or (0.6, 0.52, 0.45)), 1.0)
        nt.links.new(gr.outputs[2], b.inputs["Base Color"])
        bw = nt.nodes.new("ShaderNodeRGBToBW")
        nt.links.new(t.outputs[0], bw.inputs[0])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.35
        bump.inputs["Distance"].default_value = 0.0015
        nt.links.new(bw.outputs[0], bump.inputs["Height"])
        nt.links.new(bump.outputs[0], b.inputs["Normal"])
        nt.links.new(b.outputs[0], out.inputs[0])
        return m
    m, nt, out = _g("new_mat")("table_top")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    t = _g("img")(nt, tex, interp="Cubic", ext="REPEAT")
    nt.links.new(tc.outputs["UV"], t.inputs[0])
    kind = G.get("TOP_KIND", "marble")
    b = _g("P")(nt, rough=0.18 if kind == "marble" else 0.35, coat=0.5 if kind == "marble" else 0.2,
                coat_rough=0.06, spec=0.5)
    tint = G.get("TOP_TINT")
    if tint:                                   # per-scene grade towards the photo's table (visual only)
        mx = nt.nodes.new("ShaderNodeMix")
        mx.data_type = "RGBA"
        mx.blend_type = "MULTIPLY"
        mx.inputs["Factor"].default_value = 1.0
        nt.links.new(t.outputs[0], mx.inputs[6])
        mx.inputs[7].default_value = (*tint, 1.0)
        nt.links.new(mx.outputs[2], b.inputs["Base Color"])
    else:
        nt.links.new(t.outputs[0], b.inputs["Base Color"])
    n = _g("noise")(nt, 4.0, 6.0, 0.6, tc.outputs["Object"])
    nt.links.new(_g("maprange")(nt, n.outputs[0], 0.3, 0.7, 0.12, 0.28), b.inputs["Roughness"])
    nt.links.new(b.outputs[0], out.inputs[0])
    return m


def frame_mat():
    if "frame_mat" in G:
        return G["frame_mat"]
    rgb = G.get("FRAME_RGB", (0.45, 0.25, 0.10))
    G["frame_mat"] = wood("table_frame", rgb, tuple(c * 0.65 for c in rgb), axis_scale=(1.0, 30.0, 4.0), rough=0.5)
    return G["frame_mat"]


SAM_LOOK = {}      # body -> (rough, coat) override per scene


def sam_mat(bn, mat_nm, tex):
    rough, coat = SAM_LOOK.get(bn, (0.45, 0.1))
    return _g("tex_principled")(f"sam_{bn}", tex, rough=rough, coat=coat)


# ------------------------------------------------------------------------------------------------ builders
def flour_bag(oid, o, b):
    """Paper flour sack standing upright, open top rolled outward, flour inside (t6_09 photo)."""
    sx, sy, sz = o["shape"]["size"]
    z0 = z_bottom(b)
    hx, hy = sx / 2, sy / 2
    body_h = sz * 0.80                     # straight part; the rest is the open, flared paper collar
    # body: a box with a soft bevel, slight bulge (filled with flour) via displacement noise
    nu, nv = 40, 40
    V, F = [], []
    ring = []
    for i in range(nu):                    # rectangle perimeter, rounded corners
        a = 2 * math.pi * i / nu
        c, s = math.cos(a), math.sin(a)
        p = 6.0
        r = 1.0 / ((abs(c) / hx) ** p + (abs(s) / hy) ** p) ** (1 / p)
        ring.append((r * c, r * s))
    rows = []
    for j in range(nv + 1):
        t = j / nv
        z = z0 + t * sz
        if t * sz <= body_h:
            bul = 1.0 + 0.045 * math.sin(math.pi * min(1.0, t * sz / body_h)) ** 0.6
            k = bul
        else:                              # collar: flares out a little, then folds
            u = (t * sz - body_h) / (sz - body_h)
            k = 1.0 + 0.10 * u
        rows.append(len(V))
        for (x, y) in ring:
            V.append((x * k, y * k, z))
    for j in range(nv):
        for i in range(nu):
            a, bb = rows[j] + i, rows[j] + (i + 1) % nu
            F.append((a, bb, bb + nu, a + nu))
    V.append((0, 0, z0))
    ci = len(V) - 1
    for i in range(nu):
        F.append((rows[0] + (i + 1) % nu, rows[0] + i, ci))
    sack = mesh_obj("flour_sack", V, F)
    sol = sack.modifiers.new("paper", "SOLIDIFY")
    sol.thickness = 0.0012
    tx = bpy.data.textures.new("crinkle", "VORONOI")
    tx.noise_scale = 0.018
    dm = sack.modifiers.new("crinkle", "DISPLACE")
    dm.texture = tx
    dm.strength = 0.004
    dm.mid_level = 0.5
    sub = sack.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 1
    m, nt, bsdf = principled("kraft_white_paper", (0.74, 0.66, 0.55), 0.75, sheen=0.3, sss=0.15, sss_scale=0.004)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = _g("noise")(nt, 260.0, 6.0, 0.7, tc.outputs["Object"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.0005
    nt.links.new(n.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], bsdf.inputs["Normal"])
    sack.data.materials.append(m)
    link(sack, b)
    # flour surface inside, just below the body top, mounded
    zf = z0 + body_h - 0.01
    prof = [(0.0, zf + 0.018)] + [(u, zf + 0.018 * (1 - (u / 0.07) ** 2)) for u in np.linspace(0.01, 0.07, 6)]
    fl = lathe_obj("flour", prof + [(0.07, zf - 0.02), (0.0, zf - 0.02)], 64)
    fl.scale = (hx * 0.98 / 0.07, hy * 0.98 / 0.07, 1.0)
    m2, nt2, b2 = principled("flour", (0.93, 0.90, 0.84), 0.95, sss=0.2, sss_scale=0.003)
    tc2 = nt2.nodes.new("ShaderNodeTexCoord")
    n2 = _g("noise")(nt2, 400.0, 6.0, 0.7, tc2.outputs["Object"])
    bump2 = nt2.nodes.new("ShaderNodeBump")
    bump2.inputs["Strength"].default_value = 0.6
    bump2.inputs["Distance"].default_value = 0.0012
    nt2.links.new(n2.outputs[0], bump2.inputs["Height"])
    nt2.links.new(bump2.outputs[0], b2.inputs["Normal"])
    fl.data.materials.append(m2)
    link(fl, b)
    return sack


def mug(oid, o, b, color=None, handle_world_deg=None):
    sh = o["shape"]
    body = next((p for p in sh.get("parts", []) if p.get("name") == "body"), None) or sh
    R, h = body["radius"], body["height"]
    z0 = z_bottom(b)
    th = 0.004
    prof = [(0.0, z0), (R - 0.006, z0), (R - 0.001, z0 + 0.003), (R, z0 + 0.012), (R, z0 + h - 0.002),
            (R - th / 2, z0 + h), (R - th, z0 + h - 0.002), (R - th, z0 + 0.012), (R - th - 0.004, z0 + 0.008),
            (0.0, z0 + 0.008)]
    m = lathe_obj("mug_body", prof, 96)
    col = color or lin(o.get("rgba", (0.6, 0.65, 0.6)))
    mat = ceramic(f"glaze_{oid}", col, 0.14)
    m.data.materials.append(mat)
    # handle: a torus segment on +x (template: handle box at +x)
    hr, hz = 0.026, z0 + h * 0.52
    bpy.ops.mesh.primitive_torus_add(major_radius=hr, minor_radius=0.0055, major_segments=48, minor_segments=16,
                                     location=(R + 0.004, 0, hz), rotation=(math.pi / 2, 0, 0))
    hd = bpy.context.active_object
    for c in hd.users_collection:
        c.objects.unlink(hd)
    hd.scale = (0.85, 1.0, 1.25)
    bm_ = hd.modifiers.new("cut", "BOOLEAN")
    hd.data.materials.append(mat)
    hd.modifiers.remove(bm_)
    hd.data.polygons.foreach_set("use_smooth", [True] * len(hd.data.polygons))
    # keep only the outer half of the torus (x > R - 0.002)
    import bmesh
    bmx = bmesh.new()
    bmx.from_mesh(hd.data)
    sx, sz_ = hd.scale[0], hd.scale[2]
    kill = [v for v in bmx.verts if (v.co.x * 1.0) < -0.004]
    bmesh.ops.delete(bmx, geom=kill, context="VERTS")
    bmx.to_mesh(hd.data)
    bmx.free()
    if handle_world_deg is not None:   # visual only: the photo's handle direction (the template put it at +x)
        hd.data.transform(Matrix.Rotation(math.radians(handle_world_deg) - o.get("yaw", 0.0), 4, "Z") @ Matrix.Translation(hd.location))
        hd.location = (0, 0, 0)
    sub = m.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 1
    link(m, b, f"mug_{oid}")
    link(hd, b, f"mug_handle_{oid}")
    return m


def bowl_ceramic(oid, o, b, color=(0.86, 0.84, 0.79)):
    R, h = o["shape"]["radius"], o["shape"]["height"]
    z0 = z_bottom(b)
    th = 0.005
    prof = [(0.0, z0 + 0.004), (R * 0.40, z0 + 0.004), (R * 0.42, z0), (R * 0.46, z0), (R * 0.47, z0 + 0.008)]
    for i in range(1, 25):
        u = i / 24
        r = R * 0.47 + (R - R * 0.47) * math.sin(u * math.pi / 2) ** 0.75
        prof.append((r, z0 + 0.008 + (h - 0.008) * u ** 1.25))
    prof += [(R - th / 2, z0 + h + 0.0015), (R - th, z0 + h)]
    for (r, z) in reversed(prof[5:-2]):
        prof.append((max(r - th, 0.0), z + th * 0.7))
    prof.append((0.0, z0 + 0.008 + th))
    o_ = lathe_obj("bowl_ceramic", prof, 128)
    sub = o_.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 1
    o_.data.materials.append(ceramic(f"glaze_{oid}", color, 0.08))
    return link(o_, b, f"bowl_{oid}")


def egg(oid, o, b):
    r, h = o["shape"]["radius"], o["shape"]["height"]
    z0 = z_bottom(b)
    # an egg lying on its side as in the photo: long axis horizontal (x), radius r, length ~ 1.3 * 2r
    L = min(h, 2.75 * r)
    prof = []
    for i in range(33):
        t = i / 32
        x = -L / 2 + L * t
        u = 2 * t - 1
        rr = r * math.sqrt(max(0.0, 1 - u * u)) * (1 + 0.12 * u)
        prof.append((rr, x))
    e = lathe_obj("egg", prof, 64)
    e.data.transform(Matrix.Rotation(math.pi / 2, 4, "Y"))
    e.data.transform(Matrix.Translation((0, 0, z0 + r * 1.0)))
    col = lin(o.get("rgba", (0.65, 0.42, 0.24)))
    col = (min(1, col[0] * 1.1), min(1, col[1] * 1.0), col[2] * 0.95)
    m, nt, bsdf = principled(f"eggshell_{oid}", col, 0.55, spec=0.35, sss=0.05, sss_scale=0.002)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = _g("noise")(nt, 900.0, 2.0, 0.5, tc.outputs["Object"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.0002
    nt.links.new(n.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], bsdf.inputs["Normal"])
    n2 = _g("noise")(nt, 180.0, 3.0, 0.6, tc.outputs["Object"])
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    nt.links.new(_g("maprange")(nt, n2.outputs[0], 0.35, 0.65, 0.0, 0.25), mx.inputs["Factor"])
    mx.inputs[6].default_value = (*col, 1)
    mx.inputs[7].default_value = (col[0] * 0.8, col[1] * 0.75, col[2] * 0.7, 1)
    nt.links.new(mx.outputs[2], bsdf.inputs["Base Color"])
    e.data.materials.append(m)
    return link(e, b, f"egg_{oid}")


def rolling_pin(oid, o, b):
    r, L = o["shape"]["radius"], o["shape"]["height"]
    z0 = z_bottom(b)
    hl = 0.085                              # handle length each side
    rb = r
    prof = [(0.0, -L / 2), (0.011, -L / 2), (0.0135, -L / 2 + 0.006), (0.0125, -L / 2 + hl - 0.012),
            (0.009, -L / 2 + hl - 0.004), (rb - 0.004, -L / 2 + hl), (rb, -L / 2 + hl + 0.004),
            (rb, L / 2 - hl - 0.004), (rb - 0.004, L / 2 - hl), (0.009, L / 2 - hl + 0.004),
            (0.0125, L / 2 - hl + 0.012), (0.0135, L / 2 - 0.006), (0.011, L / 2), (0.0, L / 2)]
    p = lathe_obj("rolling_pin", prof, 64)
    p.data.transform(Matrix.Rotation(math.pi / 2, 4, "Y"))
    p.data.transform(Matrix.Translation((0, 0, z0 + rb)))
    sub = p.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 1
    p.data.materials.append(wood("beech", (0.52, 0.28, 0.10), (0.40, 0.20, 0.07), axis_scale=(1.5, 60.0, 60.0),
                                 rough=0.5, coat=0.05))
    return link(p, b, f"pin_{oid}")


def phone(oid, o, b):
    sx, sy, sz = o["shape"]["size"]
    z0 = z_bottom(b)
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    ph = bpy.context.active_object
    for c in ph.users_collection:
        c.objects.unlink(ph)
    ph.data.transform(Matrix.Diagonal((sx, sy, sz, 1.0)))
    ph.data.transform(Matrix.Translation((0, 0, z0 + sz / 2)))
    _g("bevel")(ph, min(0.006, sz * 0.45), 4, 60)
    m, nt, bsdf = principled("phone_back", (0.018, 0.019, 0.022), 0.32, spec=0.3)
    ph.data.materials.append(m)
    link(ph, b, f"phone_{oid}")
    long_x = sx >= sy
    cx = sx / 2 - 0.024 if long_x else -sx / 2 + 0.024
    cy = sy / 2 - 0.024
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    cm = bpy.context.active_object
    for c in cm.users_collection:
        c.objects.unlink(cm)
    cm.data.transform(Matrix.Diagonal((0.032, 0.032, 0.0022, 1.0)))
    cm.data.transform(Matrix.Translation((cx, cy, z0 + sz + 0.0008)))
    _g("bevel")(cm, 0.005, 5, 60)
    cm.data.materials.append(principled("phone_cam_glass", (0.05, 0.052, 0.058), 0.12, coat=1.0)[0])
    link(cm, b, "phone_cam")
    lm = principled("phone_lens", (0.004, 0.004, 0.006), 0.05, coat=1.0)[0]
    for (dx, dy) in ((-0.0075, 0.0075), (0.0075, -0.0075), (-0.0075, -0.0075)):
        ln = _g("prim_obj")("cylinder", (0.0052, 0.0016, 0), "phone_lens")
        ln.data.transform(Matrix.Translation((cx + dx, cy + dy, z0 + sz + 0.0028)))
        ln.data.materials.append(lm)
        link(ln, b)
    return ph


# category regex -> builder (first match).  Per-scene extra builders: EXTRA (set by the scene's style block).
BUILDERS = [
    (r"flour", flour_bag),
    (r"\bmug\b|\bcup\b", mug),
    (r"rolling pin", rolling_pin),
    (r"\begg\b", egg),
    (r"phone", phone),
]
EXTRA = {}           # body id -> builder (scene-specific), takes precedence
SKIP = set()         # body ids kept as SAM 3D / primitive


def build(oid, o, b):
    if oid in SKIP:
        return None
    if oid in EXTRA:
        return EXTRA[oid](oid, o, b)
    txt = (o.get("category", "") + " " + o.get("label", "")).lower()
    for pat, fn in BUILDERS:
        if re.search(pat, txt):
            return fn(oid, o, b)
    return None


# ================================================================================ more builders (02, 31, 46, 43)
def lin8(c):
    """sRGB 0..255 -> linear."""
    return lin([v / 255 for v in c])


def _cube(name, sx, sy, sz, loc=(0, 0, 0), bev=0.0, seg=3):
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    o = bpy.context.active_object
    for c in o.users_collection:
        c.objects.unlink(o)
    o.name = name
    o.data.transform(Matrix.Translation(loc) @ Matrix.Diagonal((sx, sy, sz, 1.0)))
    if bev:
        _g("bevel")(o, bev, seg, 60)
    return o


def _sphere(name, r, loc=(0, 0, 0), scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=r)
    o = bpy.context.active_object
    for c in o.users_collection:
        c.objects.unlink(o)
    o.name = name
    o.data.transform(Matrix.Translation(loc) @ Matrix.Diagonal((*scale, 1.0)))
    o.data.polygons.foreach_set("use_smooth", [True] * len(o.data.polygons))
    return o


def plate_profile(R, z0, h):
    well = R * 0.62
    prof = [(0.0, z0 + 0.003), (well * 0.9, z0 + 0.003), (well * 0.92, z0), (well * 0.97, z0), (well, z0 + 0.004)]
    prof += [(well + (R - well) * u, z0 + 0.004 + (h - 0.006) * u ** 1.6) for u in np.linspace(0.1, 1.0, 10)]
    prof += [(R, z0 + h - 0.001), (R - 0.003, z0 + h), (R - 0.004, z0 + h - 0.002)]
    prof += [(well + (R - well) * u - 0.003, z0 + 0.004 + (h - 0.006) * u ** 1.6 + 0.003) for u in np.linspace(1.0, 0.0, 10)]
    prof += [(0.0, z0 + 0.007)]
    return prof


CERAMIC = {}


def _ceramic_cached(key, col, rough=0.1):
    if key not in CERAMIC:
        CERAMIC[key] = ceramic(key, col, rough)
    return CERAMIC[key]


def plate(oid, o, b, color=None, n=1):
    R, h = o["shape"]["radius"], o["shape"]["height"]
    z0 = z_bottom(b)
    hp = h / n if n > 1 else h
    mat = _ceramic_cached(f"plate_glaze_{color}", color or lin8((236, 230, 216)))
    first = None
    for k in range(n):
        p = lathe_obj(f"plate_{oid}_{k}", plate_profile(R - 0.002 * (n - 1 - k), z0 + k * hp, min(hp, 0.024)), 128)
        sub = p.modifiers.new("sub", "SUBSURF")
        sub.levels = sub.render_levels = 1
        p.data.materials.append(mat)
        link(p, b)
        first = first or p
    return first


def bread_mat(name, crust=None, crumb=None):
    """Sourdough crust: golden-brown with darker blistered ridges and flour dust (t6_31 photo)."""
    m, nt, bsdf = principled(name, lin8((160, 98, 40)), 0.6, sss=0.05, sss_scale=0.003)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = _g("noise")(nt, 35.0, 10.0, 0.7, tc.outputs["Object"])
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].color = (*lin8((70, 34, 10)), 1)
    cr.color_ramp.elements[1].color = (*lin8((150, 92, 34)), 1)
    cr.color_ramp.elements.new(0.5).color = (*lin8((118, 66, 20)), 1)
    nt.links.new(n.outputs[0], cr.inputs[0])
    # flour dusting on the top
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Normal"], sep.inputs[0])
    dust = nt.nodes.new("ShaderNodeMix"); dust.data_type = "RGBA"
    n2 = _g("noise")(nt, 80.0, 3.0, 0.5, tc.outputs["Object"])
    f = nt.nodes.new("ShaderNodeMath"); f.operation = "MULTIPLY"
    nt.links.new(_g("maprange")(nt, sep.outputs[2], 0.8, 1.0, 0.0, 0.15), f.inputs[0])
    nt.links.new(_g("maprange")(nt, n2.outputs[0], 0.5, 0.7, 0.0, 1.0), f.inputs[1])
    nt.links.new(f.outputs[0], dust.inputs["Factor"])
    nt.links.new(cr.outputs[0], dust.inputs[6]); dust.inputs[7].default_value = (*lin8((236, 226, 206)), 1)
    nt.links.new(dust.outputs[2], bsdf.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.8
    bump.inputs["Distance"].default_value = 0.003
    nt.links.new(n.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], bsdf.inputs["Normal"])
    return m


def crumb_mat():
    if "crumb" in CERAMIC:
        return CERAMIC["crumb"]
    m, nt, bsdf = principled("crumb", lin8((236, 220, 186)), 0.9, sss=0.25, sss_scale=0.004)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    v = nt.nodes.new("ShaderNodeTexVoronoi")
    v.inputs["Scale"].default_value = 300.0
    nt.links.new(tc.outputs["Object"], v.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = 0.0015
    nt.links.new(v.outputs["Distance"], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], bsdf.inputs["Normal"])
    CERAMIC["crumb"] = m
    return m


def _bread_slice(name, rx, ry, th):
    """An oval slice: crumb faces, crust rim (per-face materials)."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=1.0, depth=1.0)
    o = bpy.context.active_object
    for c in o.users_collection:
        c.objects.unlink(o)
    o.name = name
    V = o.data.vertices
    for v in V:                                   # flatten the bottom of the oval (sliced loaf base)
        x, y, z = v.co
        v.co = (x * rx, max(y, -0.75) * ry, z * th)
    o.data.materials.append(crumb_mat())
    o.data.materials.append(bread_mat("crust_slice"))
    for p in o.data.polygons:
        p.material_index = 0 if abs(p.normal.z) > 0.5 else 1
    _g("bevel")(o, 0.002, 2, 60)
    return o


def board_with_bread(oid, o, b):
    parts = {p.get("name"): p for p in o["shape"]["parts"]}
    bd, lf, sl = parts["board"], parts["loaf"], parts["slices"]
    sx, sy, sz = bd["size"]
    board = _cube("board", sx, sy, sz, tuple(bd["pos"]), bev=0.004)
    board.data.materials.append(wood("walnut_board", lin8((104, 66, 40)), lin8((92, 58, 34)),
                                     axis_scale=(1.0, 6.0, 6.0), rough=0.55, coat=0.05))
    link(board, b)
    r, h = lf["radius"], lf["height"]
    cx, cy, cz = lf["pos"]
    zb = cz - h / 2
    # boule, cut face towards the slices (half loaf in the photo): a sphere squashed, clipped flat on one side
    loaf = _sphere("loaf", 1.0, (0, 0, 0), (r * 1.2, r * 1.1, h * 0.8))
    dx, dy = sl["pos"][0] - cx, sl["pos"][1] - cy
    ang = math.atan2(dy, dx)
    for v in loaf.data.vertices:                  # cut face
        x, y, z = v.co
        u = x * math.cos(ang) + y * math.sin(ang)
        if u > r * 0.7:
            v.co = (x - (u - r * 0.7) * math.cos(ang), y - (u - r * 0.7) * math.sin(ang), z)
        if z < 0:
            v.co.z = z * 0.25
    loaf.data.transform(Matrix.Translation((cx, cy, zb + h * 0.17)))
    tx = bpy.data.textures.new("loaf_score", "CLOUDS")
    tx.noise_scale = 0.025
    dm = loaf.modifiers.new("lumps", "DISPLACE")
    dm.texture = tx
    dm.strength = 0.005
    loaf.data.materials.append(bread_mat("crust_loaf"))
    link(loaf, b)
    # two slices leaning against each other in front of the cut face
    for k in range(2):
        s_ = _bread_slice(f"slice_{k}", 0.05, 0.045, 0.016)
        s_.rotation_euler = (0.0, 1.15 - 0.12 * k, ang)
        s_.location = (sl["pos"][0] + 0.02 * k * math.cos(ang), sl["pos"][1] + 0.02 * k * math.sin(ang),
                       sl["pos"][2] - sl["size"][2] / 2 + 0.035)
        link(s_, b)
    return board


def glass_mat(name="glass_clear"):
    if name in CERAMIC:
        return CERAMIC[name]
    m, nt, bsdf = principled(name, (0.95, 0.97, 0.97), 0.02, trans=1.0, ior=1.5)
    bsdf.inputs["Alpha"].default_value = 0.3      # thin glass in a black studio: let the contents read
    CERAMIC[name] = m
    return m


def jar_with_jam(oid, o, b):
    R, h = o["shape"]["radius"], o["shape"]["height"]
    z0 = z_bottom(b)
    hj = h * 0.68
    th = 0.003
    prof = [(0.0, z0), (R - 0.004, z0), (R, z0 + 0.006), (R, z0 + hj - 0.012), (R - 0.006, z0 + hj - 0.004),
            (R - 0.006, z0 + hj), (R - 0.006 - th, z0 + hj), (R - 0.006 - th, z0 + hj - 0.006), (R - th, z0 + hj - 0.014),
            (R - th, z0 + 0.008), (0.0, z0 + 0.006)]
    j = lathe_obj("jar", prof, 96)
    j.data.materials.append(glass_mat())
    link(j, b, f"jar_{oid}")
    jam = lathe_obj("jam", [(0.0, z0 + 0.007), (R - th - 0.0005, z0 + 0.007), (R - th - 0.0005, z0 + hj * 0.85),
                            (0.0, z0 + hj * 0.87)], 64)
    m, nt, bsdf = principled("jam", lin8((120, 18, 22)), 0.15, sss=0.6, sss_scale=0.01)
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.15, 0.1)
    jam.data.materials.append(m)
    link(jam, b)
    # the spoon standing in the jar (visual: the photo's spoon in the jam)
    sp = _cube("jar_spoon", 0.012, 0.004, 0.17, (0.012, 0.0, z0 + hj * 0.55 + 0.06), bev=0.0015)
    sp.rotation_euler = (0.0, 0.42, 0.6)
    sp.data.materials.append(steel_simple())
    link(sp, b)
    return j


def steel_simple():
    if "steel_simple" not in CERAMIC:
        CERAMIC["steel_simple"] = principled("steel_simple", (0.8, 0.8, 0.82), 0.18, metal=1.0)[0]
    return CERAMIC["steel_simple"]


def butter_dish(oid, o, b):
    parts = {p.get("name"): p for p in o["shape"]["parts"]}
    d, bu = parts["dish"], parts["butter"]
    sx, sy, sz = d["size"]
    dish = _cube("dish", sx, sy, sz, tuple(d["pos"]), bev=0.008, seg=4)
    dish.data.materials.append(_ceramic_cached("cream_glaze", lin8((200, 182, 146))))
    link(dish, b)
    x, y, z = bu["pos"]
    bt = _cube("butter", *bu["size"], (x, y, z), bev=0.003)
    bt.data.materials.append(principled("butter", lin8((246, 222, 120)), 0.35, sss=0.4, sss_scale=0.01)[0])
    link(bt, b)
    return dish


def butter_lid(oid, o, b):
    sx, sy, sz = o["shape"]["size"]
    z0 = z_bottom(b)
    lid = _cube("lid", sx, sy, sz * 0.85, (0, 0, z0 + sz * 0.85 / 2), bev=min(0.02, sz * 0.4), seg=6)
    knob = _sphere("knob", 0.012, (0, 0, z0 + sz * 0.88), (1, 1, 0.8))
    for x_ in (lid, knob):
        x_.data.materials.append(_ceramic_cached("cream_glaze", lin8((200, 182, 146))))
        link(x_, b)
    return lid


def napkin(oid, o, b, stripes=True):
    sx, sy, sz = o["shape"]["size"]
    z0 = z_bottom(b)
    n_ = _cube("napkin", sx, sy, max(sz, 0.008), (0, 0, z0 + max(sz, 0.008) / 2), bev=0.004)
    m, nt, bsdf = principled("linen", lin8((224, 216, 198)), 0.9, sheen=0.6)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n2 = _g("noise")(nt, 1500.0, 2.0, 0.5, tc.outputs["Object"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.3
    nt.links.new(n2.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], bsdf.inputs["Normal"])
    n_.data.materials.append(m)
    return link(n_, b)


def keys(oid, o, b):
    sx, sy, sz = o["shape"]["size"]
    z0 = z_bottom(b)
    brass = principled("key_metal", (0.78, 0.74, 0.66), 0.25, metal=1.0)[0]
    first = None
    for k, (dx, dy, a) in enumerate(((-sx * 0.15, 0.004, 0.25), (-sx * 0.1, -0.006, -0.2))):
        blade = _cube(f"key_blade_{k}", sx * 0.55, 0.008, 0.0022, (dx - sx * 0.05, dy, z0 + 0.0012 + k * 0.0022), bev=0.0006)
        head = _cube(f"key_head_{k}", 0.022, 0.02, 0.0024, (dx + sx * 0.28, dy, z0 + 0.0012 + k * 0.0022), bev=0.006)
        for p in (blade, head):
            p.data.transform(Matrix.Rotation(a, 4, "Z"))
            p.data.materials.append(brass)
            link(p, b)
        first = first or blade
    bpy.ops.mesh.primitive_torus_add(major_radius=0.011, minor_radius=0.0011, location=(sx * 0.38, 0, z0 + 0.0012))
    ring = bpy.context.active_object
    for c in ring.users_collection:
        c.objects.unlink(ring)
    ring.data.materials.append(steel_simple())
    link(ring, b)
    return first


def fruit(oid, o, b, color, kind="orange"):
    r = o["shape"].get("radius", 0.035)
    z0 = z_bottom(b)
    f = _sphere(f"{kind}_{oid}", r, (0, 0, z0 + r * 0.95), (1, 1, 0.95))
    m, nt, bsdf = principled(f"{kind}_skin_{oid}", color, 0.35 if kind == "orange" else 0.25, coat=0.3, coat_rough=0.2,
                             sss=0.1, sss_scale=0.004)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    if kind == "orange":
        v = nt.nodes.new("ShaderNodeTexVoronoi")
        v.inputs["Scale"].default_value = 900.0
        nt.links.new(tc.outputs["Object"], v.inputs["Vector"])
        h = v.outputs["Distance"]
    else:
        h = _g("noise")(nt, 300.0, 2.0, 0.5, tc.outputs["Object"]).outputs[0]
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.0006
    nt.links.new(h, bump.inputs["Height"])
    nt.links.new(bump.outputs[0], bsdf.inputs["Normal"])
    f.data.materials.append(m)
    link(f, b)
    stem = _cube("stem", 0.004, 0.004, 0.006, (0, 0, z0 + r * 1.9), bev=0.0015)
    stem.data.materials.append(principled("stem", lin8((70, 60, 30)), 0.8)[0])
    link(stem, b)
    return f


def carton(oid, o, b):
    sx, sy, sz = o["shape"]["size"]
    z0 = z_bottom(b)
    body_h = sz * 0.80
    hx, hy = sx / 2, sy / 2
    V = [(-hx, -hy, z0), (hx, -hy, z0), (hx, hy, z0), (-hx, hy, z0),
         (-hx, -hy, z0 + body_h), (hx, -hy, z0 + body_h), (hx, hy, z0 + body_h), (-hx, hy, z0 + body_h),
         (-hx, 0, z0 + sz * 0.97), (hx, 0, z0 + sz * 0.97)]
    F = [(0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 9, 8), (6, 7, 8, 9),
         (5, 6, 9), (7, 4, 8)]
    c = mesh_obj("carton", V, F, smooth=False)
    _g("bevel")(c, 0.002, 2, 30)
    fin = _cube("carton_fin", sx, 0.004, sz * 0.05, (0, 0, z0 + sz * 0.975))
    m = principled("carton_board", lin8((240, 238, 232)), 0.45, coat=0.3, coat_rough=0.2)[0]
    for x_ in (c, fin):
        x_.data.materials.append(m)
        link(x_, b)
    cap = lathe_obj("cap", [(0.0, z0 + body_h + 0.02), (0.012, z0 + body_h + 0.02), (0.012, z0 + body_h + 0.034),
                            (0.0, z0 + body_h + 0.034)], 32)
    cap.data.transform(Matrix.Translation((hx * 0.45, -hy * 0.45, 0)) @ Matrix.Rotation(-0.6, 4, "X"))
    cap.data.materials.append(principled("cap", lin8((236, 236, 236)), 0.4)[0])
    link(cap, b)
    return c


def juice_glass(oid, o, b, juice=0.54):
    R, h = o["shape"]["radius"], o["shape"]["height"]
    z0 = z_bottom(b)
    th = 0.0025
    prof = [(0.0, z0), (R * 0.88, z0), (R * 0.9, z0 + 0.004), (R, z0 + h), (R - th, z0 + h),
            (R * 0.9 - th, z0 + 0.008), (0.0, z0 + 0.008)]
    g = lathe_obj("glass", prof, 96)
    g.data.materials.append(glass_mat())
    link(g, b, f"glass_{oid}")
    zj = z0 + 0.008 + (h - 0.008) * juice
    rj = R * 0.9 - th + (R - R * 0.9) * juice - 0.0005
    li = lathe_obj("juice", [(0.0, z0 + 0.0085), (R * 0.9 - th - 0.0005, z0 + 0.0085), (rj, zj), (0.0, zj)], 64)
    m, nt, bsdf = principled("orange_juice", lin8((246, 150, 10)), 0.08, sss=0.6, sss_scale=0.01, coat=0.5, coat_rough=0.02)
    li.data.materials.append(m)
    link(li, b)
    return g


def cereal_bowl(oid, o, b):
    bw = bowl_ceramic(oid, o, b, lin8((236, 214, 170)))
    R, h = o["shape"]["radius"], o["shape"]["height"]
    z0 = z_bottom(b)
    zm = z0 + h * 0.62
    milk = lathe_obj("milk", [(0.0, zm), (R * 0.86, zm), (R * 0.8, zm - 0.02), (0.0, zm - 0.02)], 64)
    milk.data.materials.append(principled("milk", lin8((246, 244, 236)), 0.08, sss=0.8, sss_scale=0.01)[0])
    link(milk, b)
    rng = np.random.default_rng(7)
    flake = principled("cornflake", lin8((222, 160, 70)), 0.5, sss=0.1, sss_scale=0.003)[0]
    for k in range(70):
        a, rr = rng.uniform(0, 2 * np.pi), R * 0.78 * np.sqrt(rng.uniform())
        f = _cube("flake", 0.016, 0.013, 0.0015, (rr * np.cos(a), rr * np.sin(a), zm + rng.uniform(0.0, 0.01)), bev=0.005)
        f.rotation_euler = tuple(rng.uniform(-0.6, 0.6, 3))
        f.data.materials.append(flake)
        link(f, b)
    return bw


def spoon(oid, o, b):
    sx, sy, sz = o["shape"]["size"]
    z0 = z_bottom(b)
    h_ = _cube("spoon_handle", sx * 0.68, 0.009, 0.0025, (-sx * 0.16, 0, z0 + 0.004), bev=0.003)
    bowl_ = _sphere("spoon_bowl", 1.0, (sx * 0.32, 0, z0 + 0.006), (sx * 0.17, 0.018, 0.005))
    for x_ in (h_, bowl_):
        x_.data.materials.append(steel_simple())
        link(x_, b)
    return h_


def hair_tie(oid, o, b):
    sx, sy, sz = o["shape"]["size"]
    z0 = z_bottom(b)
    r = max(sx, sy) / 2 * 0.85
    bpy.ops.mesh.primitive_torus_add(major_radius=r, minor_radius=0.0028, major_segments=64, minor_segments=12,
                                     location=(0, 0, z0 + 0.0028))
    t = bpy.context.active_object
    for c in t.users_collection:
        c.objects.unlink(t)
    t.data.polygons.foreach_set("use_smooth", [True] * len(t.data.polygons))
    t.data.materials.append(principled("elastic", lin8((14, 14, 15)), 0.55, sheen=0.4)[0])
    return link(t, b)


def rice_bowl(oid, o, b, color):
    bw = bowl_ceramic(oid, o, b, color)
    R, h = o["shape"]["radius"], o["shape"]["height"]
    z0 = z_bottom(b)
    mound = _sphere("rice", 1.0, (0, 0, z0 + h * 0.62), (R * 0.82, R * 0.82, h * 0.42))
    tx = bpy.data.textures.new("rice_grain", "VORONOI")
    tx.noise_scale = 0.004
    dm = mound.modifiers.new("grains", "DISPLACE")
    dm.texture = tx
    dm.strength = 0.003
    mound.data.materials.append(principled("rice", lin8((246, 244, 236)), 0.45, sss=0.5, sss_scale=0.004)[0])
    link(mound, b)
    return bw


def _ring(name, rx, ry, minor, z, flat=0.35, rot=0.0, off=(0, 0)):
    bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=minor, major_segments=96, minor_segments=12)
    t = bpy.context.active_object
    for c in t.users_collection:
        c.objects.unlink(t)
    t.name = name
    for v in t.data.vertices:                     # elliptical ring of a flat strap (section scaled, not the ring)
        x, y, zz = v.co
        r = math.hypot(x, y)
        ux, uy = x / r, y / r
        d = r - 1.0
        v.co = (ux * (rx + d), uy * (ry + d), zz * flat)
    t.data.transform(Matrix.Translation((off[0], off[1], z)) @ Matrix.Rotation(rot, 4, "Z"))
    t.data.polygons.foreach_set("use_smooth", [True] * len(t.data.polygons))
    return t


def strap_mat(name, rgb, stitch=None):
    m, nt, bsdf = principled(name, lin8(rgb), 0.8, spec=0.2)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = _g("noise")(nt, 2500.0, 2.0, 0.5, tc.outputs["Object"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.05
    nt.links.new(n.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], bsdf.inputs["Normal"])
    return m


def leash(oid, o, b):
    """Coiled flat red nylon leash with a steel snap hook (t6_43 photo)."""
    sx = o["shape"]["size"][0]
    z0 = z_bottom(b)
    m = strap_mat("leash_red", (120, 6, 10))
    first = None
    for k, (rx, ry, dx, dy, a) in enumerate(((0.15, 0.045, 0.0, 0.0, 0.05), (0.14, 0.05, 0.01, 0.01, -0.08),
                                             (0.13, 0.04, -0.01, -0.01, 0.15))):
        r = _ring(f"leash_loop_{k}", min(rx, sx / 2), ry, 0.009, z0 + 0.003 + k * 0.003, 0.22, a, (dx, dy))
        r.data.materials.append(m)
        link(r, b)
        first = first or r
    hook = _cube("leash_hook", 0.05, 0.012, 0.006, (sx / 2 + 0.01, -0.03, z0 + 0.004), bev=0.004)
    hook.data.materials.append(steel_simple())
    link(hook, b)
    return first


def collar(oid, o, b):
    """Blue nylon dog collar lying curled, black plastic buckle, steel D-ring."""
    sx, sy, sz = o["shape"]["size"]
    z0 = z_bottom(b)
    r = _ring("collar_band", sx * 0.42, sy * 0.42, 0.011, z0 + 0.004, 0.32)
    r.data.materials.append(strap_mat("collar_blue", (18, 40, 112)))
    link(r, b)
    bk = _cube("collar_buckle", 0.035, 0.028, 0.012, (0.0, -sy * 0.42, z0 + 0.006), bev=0.003)
    bk.data.materials.append(principled("buckle_black", lin8((20, 20, 22)), 0.4)[0])
    link(bk, b)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.011, minor_radius=0.0015, location=(sx * 0.3, -sy * 0.32, z0 + 0.003))
    dr = bpy.context.active_object
    for c in dr.users_collection:
        c.objects.unlink(dr)
    dr.data.materials.append(steel_simple())
    link(dr, b)
    return r


def bamboo_mat(name="bamboo", base=(196, 142, 78), dark=(160, 104, 52)):
    if name in CERAMIC:
        return CERAMIC[name]
    m = wood(name, lin8(base), lin8(dark), axis_scale=(1.5, 30.0, 30.0), rough=0.45, coat=0.25)
    CERAMIC[name] = m
    return m


def tray(oid, o, b):  # noqa: D401
    """Open bamboo serving tray: base + four rounded walls with hand-hole slots (t6_46 photo)."""
    sx, sy, sz = o["shape"]["bbox"]
    z0 = z_bottom(b)
    m = bamboo_mat("bamboo_tray", (176, 112, 44), (138, 82, 30))
    first = _cube("tray_base", sx, sy, 0.008, (0, 0, z0 + 0.004), bev=0.002)
    first.data.materials.append(m)
    link(first, b)
    t = 0.012
    for k, (lx, ly, px, py) in enumerate(((t, sy, sx / 2 - t / 2, 0), (t, sy, -sx / 2 + t / 2, 0),
                                         (sx - 2 * t, t, 0, sy / 2 - t / 2), (sx - 2 * t, t, 0, -sy / 2 + t / 2))):
        w = _cube(f"tray_wall_{k}", lx, ly, sz, (px, py, z0 + sz / 2), bev=0.005, seg=4)
        w.data.materials.append(m)
        link(w, b)
    return first


def sushi_mat(oid, o, b):
    """Bamboo rolling mat: ~30 round slats tied with cotton string."""
    sx, sy, sz = o["shape"]["size"]
    z0 = z_bottom(b)
    n = 30
    r = sy / n / 2 * 0.95
    m = bamboo_mat("bamboo_mat", (214, 190, 140), (186, 156, 104))
    first = None
    for k in range(n):
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=r, depth=sx, location=(0, -sy / 2 + r + k * 2 * r / 0.95, z0 + r),
                                            rotation=(0, math.pi / 2, 0))
        c_ = bpy.context.active_object
        for c in c_.users_collection:
            c.objects.unlink(c_)
        c_.data.transform(c_.matrix_basis)
        c_.matrix_basis.identity()
        c_.data.materials.append(m)
        link(c_, b)
        first = first or c_
    for k in range(4):
        st = _cube(f"mat_string_{k}", 0.0025, sy, 0.0025, (-sx / 2 + sx * (k + 0.5) / 4, 0, z0 + 2 * r + 0.0005))
        st.data.materials.append(principled("string", lin8((236, 234, 226)), 0.9)[0])
        link(st, b)
    return first


def nori(oid, o, b):
    sx, sy, sz = o["shape"]["size"]
    z0 = z_bottom(b)
    lift = G.get("NORI_LIFT", 0.0)            # visual: resting on the rolling mat as in the photo
    sh = _cube("nori", sx, sy, 0.0012, (0, 0, z0 + 0.0006 + lift))
    m, nt, bsdf = principled("nori", lin8((28, 38, 26)), 0.45, coat=0.3)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = _g("noise")(nt, 300.0, 6.0, 0.7, tc.outputs["Object"])
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].color = (*lin8((14, 20, 14)), 1)
    cr.color_ramp.elements[1].color = (*lin8((60, 78, 46)), 1)
    nt.links.new(n.outputs[0], cr.inputs[0])
    nt.links.new(cr.outputs[0], bsdf.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    nt.links.new(n.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], bsdf.inputs["Normal"])
    sh.data.materials.append(m)
    return link(sh, b)


def chopsticks(oid, o, b):
    sx, sy, sz = o["shape"]["size"]
    z0 = z_bottom(b)
    m = principled("chopstick_wood", lin8((72, 40, 28)), 0.4, coat=0.4)[0]
    first = None
    for k, dy in enumerate((-0.006, 0.006)):
        bpy.ops.mesh.primitive_cone_add(vertices=8, radius1=0.0045, radius2=0.002, depth=sx * 0.95,
                                        location=(0, dy, z0 + 0.0045), rotation=(0, math.pi / 2, 0))
        c_ = bpy.context.active_object
        for c in c_.users_collection:
            c.objects.unlink(c_)
        c_.data.transform(c_.matrix_basis)
        c_.matrix_basis.identity()
        c_.data.materials.append(m)
        link(c_, b)
        first = first or c_
    return first


def pouch(oid, o, b):
    sx, sy, sz = o["shape"]["bbox"]
    z0 = z_bottom(b)
    p_ = _cube("pouch", sx, sy, sz * 0.9, (0, 0, z0 + sz * 0.45), bev=min(sy, sz) * 0.45, seg=6)
    m, nt, bsdf = principled("pouch_canvas", lin8((112, 96, 76)), 0.85, sheen=0.5)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = _g("noise")(nt, 1800.0, 2.0, 0.5, tc.outputs["Object"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.4
    nt.links.new(n.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], bsdf.inputs["Normal"])
    p_.data.materials.append(m)
    link(p_, b)
    z = _cube("pouch_zip", sx * 0.92, 0.006, 0.004, (0, 0, z0 + sz * 0.9), bev=0.0015)
    z.data.materials.append(principled("zip", lin8((40, 38, 36)), 0.5)[0])
    link(z, b)
    return p_


def cucumber_plate(oid, o, b):
    R, h = o["shape"]["radius"], o["shape"]["height"]
    pl = plate(oid, {"shape": {"radius": R, "height": h}}, b, lin8((214, 210, 196)))
    z0 = z_bottom(b)
    rng = np.random.default_rng(46)
    skin = principled("cucumber_skin", lin8((40, 78, 34)), 0.4)[0]
    flesh = principled("cucumber_flesh", lin8((196, 216, 150)), 0.35, sss=0.5, sss_scale=0.006)[0]
    for k in range(12):
        a, rr = rng.uniform(0, 2 * np.pi), R * 0.55 * np.sqrt(rng.uniform())
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.017, depth=0.005,
                                            location=(rr * math.cos(a), rr * math.sin(a), z0 + 0.008 + 0.004 * (k % 3)),
                                            rotation=(rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), 0))
        c_ = bpy.context.active_object
        for c in c_.users_collection:
            c.objects.unlink(c_)
        c_.data.materials.append(flesh)
        c_.data.materials.append(skin)
        for p in c_.data.polygons:
            p.material_index = 0 if abs(p.normal.z) > 0.5 else 1
        link(c_, b)
    return pl


def coaster(oid, o, b):
    sx, sy, sz = o["shape"]["size"]
    z0 = z_bottom(b)
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=min(sx, sy) / 2, depth=sz * 0.6, location=(0, 0, z0 + sz * 0.3))
    c_ = bpy.context.active_object
    for c in c_.users_collection:
        c.objects.unlink(c_)
    m, nt, bsdf = principled("cork", lin8((168, 120, 74)), 0.9)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    v = nt.nodes.new("ShaderNodeTexVoronoi")
    v.inputs["Scale"].default_value = 500.0
    nt.links.new(tc.outputs["Object"], v.inputs["Vector"])
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].color = (*lin8((110, 72, 40)), 1)
    cr.color_ramp.elements[1].color = (*lin8((190, 146, 98)), 1)
    nt.links.new(v.outputs["Distance"], cr.inputs[0])
    nt.links.new(cr.outputs[0], bsdf.inputs["Base Color"])
    c_.data.materials.append(m)
    return link(c_, b)


def gloves(oid, o, b):
    """A pair of yellow rubber gloves lying flat (t6_27 photo)."""
    sx, sy, sz = o["shape"]["size"]
    z0 = z_bottom(b)
    m = principled("rubber_yellow", lin8((236, 196, 20)), 0.3, coat=0.4, coat_rough=0.15)[0]
    first = None
    for k, (dy, a) in enumerate(((-sy * 0.22, 0.12), (sy * 0.22, -0.18))):
        th = 0.012
        parts = [_cube(f"glove_palm_{k}", 0.09, 0.07, th, (0.0, 0, z0 + th / 2), bev=0.008, seg=4),
                 _cube(f"glove_cuff_{k}", 0.1, 0.075, th * 0.8, (-0.09, 0, z0 + th * 0.4), bev=0.006, seg=4)]
        for f in range(4):
            parts.append(_cube(f"glove_f{f}_{k}", 0.075 - 0.01 * abs(f - 1.5), 0.016, th * 0.8,
                               (0.08, -0.026 + f * 0.0175, z0 + th * 0.4), bev=0.0075, seg=4))
        parts.append(_cube(f"glove_th_{k}", 0.05, 0.018, th * 0.8, (0.02, 0.05 * (1 if k else -1), z0 + th * 0.4),
                           bev=0.008, seg=4))
        parts[-1].data.transform(Matrix.Rotation(0.6 * (1 if k else -1), 4, "Z"))
        for p_ in parts:
            p_.data.transform(Matrix.Translation((0.03, dy, 0)) @ Matrix.Rotation(a, 4, "Z"))
            p_.data.materials.append(m)
            link(p_, b)
        first = first or parts[0]
    return first


BUILDERS[:0] = [
    (r"cork coaster", coaster),
    (r"rubber glove", gloves),
    (r"serving tray", tray),
    (r"sushi mat", sushi_mat),
    (r"seaweed|nori", nori),
    (r"chopsticks", chopsticks),
    (r"zipper pouch", pouch),
    (r"sliced cucumber", cucumber_plate),
    (r"\bleash\b", leash),
    (r"dog collar", collar),
    (r"hair tie", hair_tie),
    (r"plate stack", lambda i, o, b: plate(i, o, b, n=3)),
    (r"\bplate\b", plate),
    (r"cutting board with bread", board_with_bread),
    (r"jam jar", jar_with_jam),
    (r"butter dish with butter", butter_dish),
    (r"butter dish lid", butter_lid),
    (r"napkin", napkin),
    (r"\bkeys?\b", keys),
    (r"milk carton", carton),
    (r"glass of orange juice", juice_glass),
    (r"cereal bowl", cereal_bowl),
    (r"\bspoon\b", spoon),
]
