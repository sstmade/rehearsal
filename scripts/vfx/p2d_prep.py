"""Blender (headless): back-project the 06 photo through the photo camera onto the twin geometry.

blender -b footage/tables_sim/hero/exec_blender/06_cycles.blend --python p2d_prep.py
Writes footage/vfx/_data/p2d_points.npz: per sampled frame pixel (1920x1080 grid, step S):
  xy (frame px), dir (world ray dir), origin, depth (ray hit distance, nan = no twin geometry),
  hit (index into names), normal; plus cam matrix / lens / shift of the photo camera (frame 98).
The twin (frame 98 of the Cycles scene = still the photo camera, nothing moved yet) is real data:
SAM 3D meshes + fitted primitives placed by the shipped perception pipeline.
"""
import bpy, numpy as np, json, os, mathutils
ROOT = os.path.abspath(os.path.join(os.path.dirname(bpy.data.filepath), "../../../.."))
OUT = os.path.join(ROOT, "footage/vfx/_data"); os.makedirs(OUT, exist_ok=True)
S = 3
sc = bpy.context.scene
sc.frame_set(98)
for o in bpy.data.objects:   # hide the robot, its pedestal and the studio floor for the ray cast
    if (o.parent and o.parent.name.startswith("B_link")) or o.parent and o.parent.name in ("B_hand", "B_left_finger", "B_right_finger") \
            or o.name in ("g1_world", "g2_world", "studio_floor"):
        o.hide_set(True); o.hide_viewport = True
dg = bpy.context.evaluated_depsgraph_get()
cam = sc.camera
M = cam.matrix_world.copy()
fr = [M @ v for v in cam.data.view_frame(scene=sc)]   # tr, br, bl, tl (world, at unit-ish distance)
o = M.translation
W, H = sc.render.resolution_x, sc.render.resolution_y
tr, br, bl, tl = [np.array(v) for v in fr]
xs = np.arange(S / 2, W, S); ys = np.arange(S / 2, H, S)
X, Y = np.meshgrid(xs, ys)
u = (X / W)[..., None]; v = (Y / H)[..., None]
top = tl + (tr - tl) * u; bot = bl + (br - bl) * u
P = top + (bot - top) * v
D = P - np.array(o); D /= np.linalg.norm(D, axis=-1, keepdims=True)
names = []
depth = np.full(X.shape, np.nan, np.float32); hit = np.full(X.shape, -1, np.int16); nrm = np.zeros(X.shape + (3,), np.float32)
ov = mathutils.Vector(o)
for j in range(X.shape[0]):
    for i in range(X.shape[1]):
        ok, loc, n, idx, ob, _ = sc.ray_cast(dg, ov, mathutils.Vector(D[j, i]))
        if ok:
            depth[j, i] = (loc - ov).length
            nm = ob.parent.name[2:] if ob.parent and ob.parent.name.startswith("B_") else ("towel_1" if ob.name == "towel_flex" else "table")
            if nm not in names: names.append(nm)
            hit[j, i] = names.index(nm); nrm[j, i] = n
np.savez_compressed(os.path.join(OUT, "p2d_points.npz"), x=X.astype(np.float32), y=Y.astype(np.float32), dir=D.astype(np.float32),
                    origin=np.array(o, np.float32), depth=depth, hit=hit, normal=nrm, names=np.array(names),
                    cam=np.array(M, np.float32), lens=cam.data.lens, sensor=cam.data.sensor_width,
                    shift=np.array([cam.data.shift_x, cam.data.shift_y], np.float32), frame=np.array([tr, br, bl, tl], np.float32))
print("PREP", names, np.isfinite(depth).mean(), np.nanmin(depth), np.nanmax(depth))
