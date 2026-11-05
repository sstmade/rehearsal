"""Blender: primitive (physics collision shapes, matte) twin of one table, from its photo camera.
  blender -b --factory-startup -P render_wall.py -- --json T.json --still OUT.png [--orbit FRAMEDIR --frames 75]
Frame is 1280x720 with the photo's vertical field of view (the 4:3 photo scaled to 720 px high, centred, overlays it).
"""
import json, math, sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
arg = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d
J = json.loads(Path(arg("--json")).read_text())
W, H = 1280, 720

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene


def lin(c):
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


MATS = {}


def matte(rgb, rough=0.9, key=None):
    key = key or tuple(round(x, 3) for x in rgb)
    if key in MATS:
        return MATS[key]
    m = bpy.data.materials.new(f"m{len(MATS)}")
    m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
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
        bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=s[0], radius2=s[0], depth=2 * s[1])
    elif t in (2, 4):   # sphere / ellipsoid
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
        sz = (s[0], s[0], s[0]) if t == 2 else s[:3]
        bmesh.ops.scale(bm, vec=Vector(sz), verts=bm.verts)
    elif t == 3:    # capsule
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=s[0])
        for v in bm.verts:
            v.co.z += s[1] if v.co.z > 0 else -s[1]
    else:
        bm.free()
        return None
    me = bpy.data.meshes.new("p")
    bm.to_mesh(me)
    bm.free()
    return me


TABLE = lin((92 / 255, 94 / 255, 98 / 255))
for i, g in enumerate(J["geoms"]):
    if g["kind"] == "arm" or (g["kind"] == "table" and g["body"] == "world" and g["pos"][0] < 0.2):
        continue                                    # no arm / pedestal: it stands between the photo camera and the table
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
    rgb = lin(g["rgba"][:3])
    if g["kind"] == "table":
        top = g["pos"][2] > -0.05
        mat = matte(TABLE if top else lin((0.2, 0.2, 0.22)), 0.85, key=("table", top))
    elif g["kind"] == "arm":
        mat = matte(rgb, 0.45)
    else:
        mat = matte(rgb, 0.9)
    me.transform(M4)
    o = add(me, f"g{i}", mat, smooth=g["type"] in (2, 3, 4) or (g["type"] == 7 and g["kind"] == "arm"))
    if g["type"] == 5:
        for p in me.polygons:
            p.use_smooth = abs(p.normal.z) < 0.5

# studio: near-black floor, dark world
bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, -0.75))
bpy.context.active_object.data.materials.append(matte(lin((0.035, 0.037, 0.042)), 0.6, key="floor"))
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.006, 0.007, 0.009, 1)
cx, cy = J["table"]["center"]


def area(loc, power, size):
    ld = bpy.data.lights.new("a", "AREA"); ld.energy = power; ld.size = size
    o = bpy.data.objects.new("a", ld); sc.collection.objects.link(o); o.location = loc
    d = Vector((cx, cy, 0)) - Vector(loc)
    o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


area((cx - 0.4, cy + 0.9, 2.2), 420, 2.0)
area((cx + 1.6, cy - 1.4, 1.4), 120, 2.5)
area((cx - 1.8, cy - 0.8, 1.0), 60, 2.0)

# photo camera (OpenCV K, R, t) -> Blender
K, Rc, t = (np.array(J["camera"][k], float) for k in ("K", "R", "t"))
C = -Rc.T @ t
Rbw = Rc.T @ np.diag([1, -1, -1])
cd = bpy.data.cameras.new("c")
cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
cd.sensor_fit = "VERTICAL"; cd.sensor_height = 24.0
cd.lens = K[1, 1] * 24.0 / J["H"]
cd.shift_y = -(K[1, 2] - J["H"] / 2) / J["H"] * 1.0 * (J["H"] / J["H"])
cd.shift_x = 0.0
cd.clip_start, cd.clip_end = 0.02, 60
cam.matrix_world = Matrix(((*Rbw[0], C[0]), (*Rbw[1], C[1]), (*Rbw[2], C[2]), (0, 0, 0, 1)))
cam_m0 = cam.matrix_world.copy()
# pivot: the optical axis hitting the table plane
fwd = Rc.T @ np.array([0, 0, 1.0])
s_ = -C[2] / fwd[2] if fwd[2] < -1e-3 else 1.5
P = C + s_ * fwd

r = sc.render
r.resolution_x, r.resolution_y, r.resolution_percentage = W, H, 100
try:
    r.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    try:
        r.engine = "BLENDER_EEVEE"
    except TypeError as e:
        print("engines:", e); raise
ee = sc.eevee
for k, v in (("taa_render_samples", int(arg("--taa", 24))), ("use_shadows", True), ("use_gtao", True), ("use_raytracing", False)):
    try:
        setattr(ee, k, v)
    except Exception:
        pass
sc.view_settings.view_transform = "AgX"
try:
    sc.view_settings.look = "AgX - Medium High Contrast"
except Exception:
    pass
sc.view_settings.exposure = 0.0
r.image_settings.file_format = "PNG"

if arg("--still"):
    r.filepath = str(Path(arg("--still")).resolve())
    bpy.ops.render.render(write_still=True)

if arg("--orbit"):
    od = Path(arg("--orbit")); od.mkdir(parents=True, exist_ok=True)
    n = int(arg("--frames", 75))
    amp = math.radians(float(arg("--deg", 22)))
    piv = Vector(P.tolist())
    for i in range(n):
        u = i / (n - 1)
        e = u * u * (3 - 2 * u)                       # smoothstep: slow in, slow out
        a = amp * e
        Rz = Matrix.Rotation(a, 4, "Z")
        T = Matrix.Translation(piv) @ Rz @ Matrix.Translation(-piv)
        cam.matrix_world = T @ cam_m0
        r.filepath = str(od / f"f_{i:03d}.png")
        bpy.ops.render.render(write_still=True)
print("WALL_DONE", J["name"])
