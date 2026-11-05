"""Blender HQ wall: physics collision shapes coloured from the photo (muted), table top/frame from the photo,
onion-scene studio (dark void, glowing grid floor, warm key + cool rims), Cycles + OIDN.
  blender -b --factory-startup -P render_hq.py -- --json T.json --colors C.json --still OUT.png [--orbit DIR --frames 75 --deg 20]
Frame is 1280x720 with the photo's vertical field of view (the 4:3 photo scaled to 720 px high, centred, overlays it).
"""
import colorsys, json, math, sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
arg = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d
J = json.loads(Path(arg("--json")).read_text())
COL = json.loads(Path(arg("--colors")).read_text())
W, H = 1280, 720

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


def lin(c):
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


MATS = {}


def matte(rgb, rough=0.6, key=None, coat=0.0, spec=0.4):
    key = key or tuple(round(x, 3) for x in rgb)
    if key in MATS:
        return MATS[key]
    m = bpy.data.materials.new(f"m{len(MATS)}")
    m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    for k_, v_ in (("Specular IOR Level", spec), ("Coat Weight", coat), ("Coat Roughness", 0.25)):
        try:
            b.inputs[k_].default_value = v_
        except Exception:
            pass
    MATS[key] = m
    return m


def add(me, name, mat, smooth=False):
    o = bpy.data.objects.new(name, me)
    sc.collection.objects.link(o)
    me.materials.append(mat)
    if smooth:
        me.shade_smooth()
    return o


def prim(t, s):
    import bmesh
    bm = bmesh.new()
    if t == 6:      # box
        bmesh.ops.create_cube(bm, size=2.0)
        bmesh.ops.scale(bm, vec=Vector(s[:3]), verts=bm.verts)
    elif t == 5:    # cylinder
        bmesh.ops.create_cone(bm, cap_ends=True, segments=64, radius1=s[0], radius2=s[0], depth=2 * s[1])
    elif t in (2, 4):   # sphere / ellipsoid
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0)
        sz = (s[0], s[0], s[0]) if t == 2 else s[:3]
        bmesh.ops.scale(bm, vec=Vector(sz), verts=bm.verts)
    elif t == 3:    # capsule
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=s[0])
        for v in bm.verts:
            v.co.z += s[1] if v.co.z > 0 else -s[1]
    else:
        bm.free()
        return None
    me = bpy.data.meshes.new("p")
    bm.to_mesh(me)
    bm.free()
    return me


# ---------------------------------------------------------------- geometry: collision shapes, photo colours
th, ts, tv = colorsys.rgb_to_hsv(*COL["table"])
TABLE = lin(colorsys.hsv_to_rgb(th, min(1, ts * 1.25), tv * 0.8))
FRAME = lin(colorsys.hsv_to_rgb(th, min(ts * 1.1, 1), tv * 0.42))
grain = COL.get("table_spread", 0.05)
OBJC = COL["objects"]
for i, g in enumerate(J["geoms"]):
    if g["kind"] == "arm" or (g["kind"] == "table" and g["body"] == "world" and g["pos"][0] < 0.2):
        continue
    R = np.array(g["mat"]).reshape(3, 3)
    M4 = Matrix(((*R[0], g["pos"][0]), (*R[1], g["pos"][1]), (*R[2], g["pos"][2]), (0, 0, 0, 1)))
    if g["type"] == 7:
        me = bpy.data.meshes.new("m")
        me.from_pydata(g["verts"], [], g["faces"])
        me.update()
    else:
        me = prim(g["type"], g["size"])
        if me is None:
            continue
    if g["kind"] == "table":
        top = g["pos"][2] > -0.05
        mat = (matte(TABLE, 0.38 if grain < 0.06 else 0.5, key="top", coat=0.3 if grain < 0.06 else 0.12) if top
               else matte(FRAME, 0.5, key="frame"))
    else:
        c = OBJC.get(g["body"], list(g["rgba"][:3]))
        mat = matte(lin(c), 0.55, key=("o", g["body"]), spec=0.45)
    me.transform(M4)
    o = add(me, f"g{i}", mat, smooth=g["type"] in (2, 3, 4))
    if g["type"] == 5:
        for p in me.polygons:
            p.use_smooth = abs(p.normal.z) < 0.5
    if g["type"] in (5, 6):
        bv = o.modifiers.new("bv", "BEVEL")
        bv.width = 0.002 if g["kind"] == "obj" else 0.004
        bv.segments = 2
        bv.limit_method = "ANGLE"

cx, cy = J["table"]["center"]

# ---------------------------------------------------------------- photo camera (OpenCV K, R, t) -> Blender
K, Rc, t = (np.array(J["camera"][k], float) for k in ("K", "R", "t"))
C = -Rc.T @ t
Rbw = Rc.T @ np.diag([1, -1, -1])
cd = bpy.data.cameras.new("c")
cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
cd.sensor_fit = "VERTICAL"; cd.sensor_height = 24.0
cd.lens = K[1, 1] * 24.0 / J["H"]
cd.shift_y = -(K[1, 2] - J["H"] / 2) / J["H"]
cd.shift_x = 0.0
cd.clip_start, cd.clip_end = 0.02, 60
cam.matrix_world = Matrix(((*Rbw[0], C[0]), (*Rbw[1], C[1]), (*Rbw[2], C[2]), (0, 0, 0, 1)))
cam_m0 = cam.matrix_world.copy()
fwd = Rc.T @ np.array([0, 0, 1.0])
s_ = -C[2] / fwd[2] if fwd[2] < -1e-3 else 1.5
P = C + s_ * fwd

# ---------------------------------------------------------------- studio (onion look), tinted by the photo's mood
mh, ms, mv = colorsys.rgb_to_hsv(*COL["mood"])
tint = colorsys.hsv_to_rgb(mh, min(0.4, ms * 1.6 + 0.08), 1.0)
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
bgn = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
bgn.inputs[0].default_value = (*[0.004 * c for c in tint], 1)
bgn.inputs[1].default_value = 1.0
FLOOR_Z = min([g["pos"][2] - g["size"][2] for g in J["geoms"] if g["kind"] == "table" and g["type"] == 6] + [-0.7])
bpy.ops.mesh.primitive_plane_add(size=30, location=(cx, cy, FLOOR_Z - 0.0015))
floor = bpy.context.active_object
fm = bpy.data.materials.new("floor"); fm.use_nodes = True; nt = fm.node_tree
fb = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
fb.inputs["Base Color"].default_value = (*[0.010 * c for c in tint], 1)
fb.inputs["Roughness"].default_value = 0.8
fb.inputs["Specular IOR Level"].default_value = 0.05
tc = nt.nodes.new("ShaderNodeTexCoord"); sep = nt.nodes.new("ShaderNodeSeparateXYZ")
nt.links.new(tc.outputs["Object"], sep.inputs[0])


def mr(src_, a, b, c, d):
    m_ = nt.nodes.new("ShaderNodeMapRange"); m_.clamp = True
    nt.links.new(src_, m_.inputs[0])
    m_.inputs[1].default_value = a; m_.inputs[2].default_value = b
    m_.inputs[3].default_value = c; m_.inputs[4].default_value = d
    return m_.outputs[0]


def math_(op, a_, b_):
    m_ = nt.nodes.new("ShaderNodeMath"); m_.operation = op
    for k_, v_ in enumerate((a_, b_)):
        if isinstance(v_, float):
            m_.inputs[k_].default_value = v_
        else:
            nt.links.new(v_, m_.inputs[k_])
    return m_.outputs[0]


def grid(sp, wd):
    acc = None
    for ax in (0, 1):
        d = math_("PINGPONG", sep.outputs[ax], sp / 2)
        ln = mr(d, 0.0, wd, 1.0, 0.0)
        acc = ln if acc is None else math_("MAXIMUM", acc, ln)
    return acc


lines = math_("ADD", math_("MULTIPLY", grid(0.25, 0.0022), 0.45), grid(1.0, 0.0035))
cmb = nt.nodes.new("ShaderNodeCombineXYZ")
nt.links.new(sep.outputs[0], cmb.inputs[0]); nt.links.new(sep.outputs[1], cmb.inputs[1])
vl = nt.nodes.new("ShaderNodeVectorMath"); vl.operation = "LENGTH"; nt.links.new(cmb.outputs[0], vl.inputs[0])
fade_ = math_("POWER", mr(vl.outputs["Value"], 0.4, 2.8, 1.0, 0.0), 1.6)
fb.inputs["Emission Color"].default_value = (0.35, 0.62, 1.0, 1)
nt.links.new(math_("MULTIPLY", math_("MULTIPLY", lines, fade_), 0.35), fb.inputs["Emission Strength"])
floor.data.materials.append(fm)

TGT = Vector((cx, cy, 0.08))
caz = math.atan2(C[1] - cy, C[0] - cx)


def sph(dist, az, el):
    a, e = caz + math.radians(az), math.radians(el)
    return TGT + dist * Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))


def area(name, loc, size, power, color, size_y=None, spread=180):
    ld = bpy.data.lights.new(name, "AREA"); ld.shape = "RECTANGLE"; ld.size = size; ld.size_y = size_y or size
    ld.energy = power; ld.color = color
    try:
        ld.spread = math.radians(spread)
    except Exception:
        pass
    o = bpy.data.objects.new(name, ld); sc.collection.objects.link(o); o.location = loc
    o.rotation_mode = "QUATERNION"; o.rotation_quaternion = (TGT - loc).normalized().to_track_quat("-Z", "Y")
    o.visible_camera = False


L_ = float(arg("--light", 1.0))
area("key", sph(2.3, 40, 58), 1.4, 110 * L_, (1.0, 0.94, 0.86), 0.9)          # big soft box, camera-left, warm
area("rim_a", sph(2.2, 160, 16), 0.25, 90 * L_, (0.72, 0.84, 1.0), 2.0, 40)   # cool strips behind the set
area("rim_b", sph(2.4, -150, 22), 0.25, 60 * L_, (0.78, 0.88, 1.0), 1.8, 40)
area("fill", sph(2.6, -60, 22), 2.5, 16 * L_, (0.85, 0.9, 1.0), 1.5)           # wide, dim, camera-right
area("top", TGT + Vector((0, 0, 1.6)), 0.3, 35 * L_, (1, 1, 1), 1.6)           # overhead strip

# ---------------------------------------------------------------- render
r = sc.render
r.resolution_x, r.resolution_y, r.resolution_percentage = W, H, 100
r.engine = "CYCLES"
cy_ = sc.cycles
try:
    pr = bpy.context.preferences.addons["cycles"].preferences
    pr.compute_device_type = "METAL"; pr.get_devices()
    for dv in pr.devices:
        dv.use = True
    cy_.device = "GPU"
except Exception as e:
    print("GPU setup failed:", e)
cy_.samples = int(arg("--spp", 24)); cy_.use_adaptive_sampling = True; cy_.adaptive_threshold = 0.04
cy_.use_denoising = True
try:
    cy_.denoiser = "OPENIMAGEDENOISE"
except Exception:
    pass
cy_.max_bounces, cy_.diffuse_bounces, cy_.glossy_bounces = 6, 3, 3
cy_.sample_clamp_indirect = 4.0
r.use_persistent_data = True
sc.view_settings.view_transform = "AgX"
try:
    sc.view_settings.look = arg("--look", "AgX - Medium High Contrast")
except Exception:
    pass
sc.view_settings.exposure = float(arg("--exposure", -1.0))
r.image_settings.file_format = "PNG"

if arg("--still"):
    r.filepath = str(Path(arg("--still")).resolve())
    bpy.ops.render.render(write_still=True)

if arg("--orbit"):
    od = Path(arg("--orbit")); od.mkdir(parents=True, exist_ok=True)
    n = int(arg("--frames", 75))
    amp = math.radians(float(arg("--deg", 20)))
    piv = Vector(P.tolist())
    for i in range(n):
        u = i / (n - 1)
        e = u * u * (3 - 2 * u)
        Rz = Matrix.Rotation(amp * e, 4, "Z")
        cam.matrix_world = Matrix.Translation(piv) @ Rz @ Matrix.Translation(-piv) @ cam_m0
        r.filepath = str(od / f"f_{i:03d}.png")
        bpy.ops.render.render(write_still=True)
print("WALL_DONE", J["name"])
