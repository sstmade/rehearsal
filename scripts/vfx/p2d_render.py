"""Sequence 1 "Photo -> Digital": the Blender (Eevee) part.

blender -b footage/tables_sim/hero/exec_blender/06_cycles.blend --python p2d_render.py -- --frames 0:70 [--step 1] [--out DIR]
       ... -- --wire-still   (wireframe twin alone through the photo camera, for the 2D transitions)

Opens the shipped Cycles twin read-only (never saved), freezes it at frame 98 (photo camera, nothing has moved),
hides the robot, and renders:
  * the photo as a floating card (phase A, f0-69): the card sits on the photo camera's frustum, so when the
    camera arrives at the photo camera (f70) the card fills the frame exactly = the photo;
  * the photo as a point cloud (phase D/E, f240-444): one point per 3x3 frame pixels (p2d_prep.py), coloured by
    the photo pixel, positioned along its camera ray at the depth where that ray meets the twin geometry;
    rays that miss the twin (wall, floor) have no depth and dissolve into the void;
  * the twin's own meshes as glowing wireframes, built per object with a Build modifier (phase E).
Timeline numbers are global frames of the 30 fps sequence (see p2d_compose.py).
"""
import bpy, bmesh, sys, os, math, argparse
import numpy as np
from mathutils import Matrix, Vector, Quaternion

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="0:70")
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--list", default=None, help="comma list of frames")
ap.add_argument("--out", default=None)
ap.add_argument("--wire-still", action="store_true")
ap.add_argument("--samples", type=int, default=24)
ap.add_argument("--res", type=float, default=1.0)
A = ap.parse_args(argv)

ROOT = os.path.abspath(os.path.join(os.path.dirname(bpy.data.filepath), "../../../.."))
DATA = os.path.join(ROOT, "footage/vfx/_data")
OUT = A.out or os.path.join(ROOT, "footage/vfx/_work/p2d_blender")
os.makedirs(OUT, exist_ok=True)
PHOTO = os.path.join(ROOT, "rehearsal-mvp/photos/ai_gen/tables_v6/table_06_indian-kitchen.png")

sc = bpy.context.scene
sc.frame_set(98)
# ---------------------------------------------------------------- freeze the twin at frame 98
mats = {o.name: o.matrix_world.copy() for o in bpy.data.objects}
for o in bpy.data.objects:
    o.animation_data_clear()
for o in bpy.data.objects:
    if o.parent is None:
        o.matrix_world = mats[o.name]
for o in bpy.data.objects:
    if o.parent is not None:
        o.matrix_world = mats[o.name]
PHOTO_CAM = sc.camera
M0 = mats[PHOTO_CAM.name]
LENS, SENSOR = PHOTO_CAM.data.lens, PHOTO_CAM.data.sensor_width
SHIFT0 = Vector((PHOTO_CAM.data.shift_x, PHOTO_CAM.data.shift_y))
for m in bpy.data.objects["towel_flex"].modifiers:
    if m.type == "MESH_CACHE":
        m.frame_start = -1000.0       # hold the settled first cache frame (the flex moves < 5 mm in this run)

ROBOT = {"B_link0", "B_link1", "B_link2", "B_link3", "B_link4", "B_link5", "B_link6", "B_link7", "B_hand",
         "B_left_finger", "B_right_finger"}
TWIN = []
for o in bpy.data.objects:
    if o.type == "LIGHT":
        o.hide_render = True
        continue
    if o.type != "MESH":
        continue
    if (o.parent and o.parent.name in ROBOT) or o.name in ("g1_world", "g2_world", "studio_floor"):
        o.hide_render = True
        continue
    o.hide_render = True            # solid twin is never rendered here (the 2D comp crossfades to the real renders)
    TWIN.append(o)

# ---------------------------------------------------------------- render settings
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
sc.render.resolution_percentage = int(100 * A.res)
sc.render.film_transparent = False
sc.eevee.taa_render_samples = A.samples
sc.view_settings.view_transform = "Standard"
sc.view_settings.look = "None"
sc.view_settings.exposure = 0.0
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_depth = "8"
try:
    sc.eevee.use_shadows = False
except Exception:
    pass
wn = sc.world.node_tree.nodes
bg = next(n for n in wn if n.type == "BACKGROUND")
bg.inputs[0].default_value = (0.0035, 0.0045, 0.006, 1)
bg.inputs[1].default_value = 1.0


def emit_mat(name, color=(1, 1, 1), strength=1.0, attr=None, attr_strength=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1)
    em.inputs["Strength"].default_value = strength
    if attr:
        at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = attr; at.attribute_type = "GEOMETRY"
        nt.links.new(at.outputs["Color"], em.inputs["Color"])
    if attr_strength:
        at2 = nt.nodes.new("ShaderNodeAttribute"); at2.attribute_name = attr_strength; at2.attribute_type = "GEOMETRY"
        nt.links.new(at2.outputs["Fac"], em.inputs["Strength"])
    nt.links.new(em.outputs[0], out.inputs[0])
    return m


# ---------------------------------------------------------------- data from p2d_prep
P = np.load(os.path.join(DATA, "p2d_points.npz"))
Dirs = P["dir"].reshape(-1, 3).astype(np.float64)
O = np.array(M0.translation)
FWD = -np.array(M0.to_3x3().col[2])
RIGHT = np.array(M0.to_3x3().col[0])
depth = P["depth"].reshape(-1).astype(np.float64)
hit = P["hit"].reshape(-1)
names = list(P["names"])
gx, gy = P["x"].reshape(-1), P["y"].reshape(-1)
N = len(depth)
valid = np.isfinite(depth)
D0 = float(np.nanmedian(depth)) * 0.93          # the photo card's distance from the camera
cosf = Dirs @ FWD
d_flat = D0 / cosf
pivot = O + FWD * float(np.nanmedian(depth[np.abs(gx - 960) + np.abs(gy - 540) < 120]))

# photo colours (frame px -> photo px: scale 1920/1448, centre crop 1080 of 1440)
img = bpy.data.images.load(PHOTO)
pw, ph = img.size
px = np.array(img.pixels[:], np.float32).reshape(ph, pw, 4)[::-1]     # top row first, linear
k = 1920.0 / pw
u = np.clip((gx / k).astype(int), 0, pw - 1)
v = np.clip(((gy + (ph * k - 1080) / 2) / k).astype(int), 0, ph - 1)
col = px[v, u, :3]
rng = np.random.default_rng(6)
rnd = rng.random(N)

# per-point depth for the extrusion: twin hit, or (no twin geometry) pushed out and dissolved
d_true = np.where(valid, depth, d_flat * 1.0)
pix_world = SENSOR / LENS / 1920.0                 # world size of one frame pixel at unit distance
S = float(P["x"][0, 1] - P["x"][0, 0])
rad0 = 0.62 * S * pix_world * d_flat * 1.06
rad_pt = 0.30 * S * pix_world * d_flat

# depth-discontinuity bridges: glowing points strung along the camera ray between neighbours whose depths jump,
# so disoccluded gaps (behind the spice box, bowl, onions) glow instead of showing black
GH, GW = P["depth"].shape
dg_ = P["depth"].astype(np.float64)
BA, BB, BS = [], [], []
for dy, dx in ((0, 1), (1, 0)):
    a = dg_[:GH - dy, :GW - dx]; b = dg_[dy:, dx:]
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) > 0.012)
    jj, ii = np.nonzero(ok)
    ia = jj * GW + ii; ib = (jj + dy) * GW + (ii + dx)
    nk = np.clip(np.ceil(np.abs(a - b)[ok] / 0.005), 2, 10).astype(int)
    for kk in range(1, 11):
        sel = nk > kk
        BA.append(ia[sel]); BB.append(ib[sel]); BS.append(np.full(sel.sum(), kk) / nk[sel])
BA, BB, BS = np.concatenate(BA), np.concatenate(BB), np.concatenate(BS)
rngb = np.random.default_rng(1)
if len(BA) > 70000:
    keep = rngb.choice(len(BA), 70000, replace=False); BA, BB, BS = BA[keep], BB[keep], BS[keep]
NB = len(BA)
print("BRIDGES", NB)
NT = N + NB
# ---------------------------------------------------------------- point cloud object
me = bpy.data.meshes.new("pc")
me.vertices.add(NT)
me.vertices.foreach_set("co", np.r_[O + Dirs * d_flat[:, None], np.repeat(O[None], NB, 0)].astype(np.float32).ravel())
for nm, typ in (("pcol", "FLOAT_COLOR"), ("prad", "FLOAT"), ("pem", "FLOAT")):
    me.attributes.new(nm, typ, "POINT")
me.attributes["pcol"].data.foreach_set("color", np.c_[np.r_[col, np.ones((NB, 3))], np.ones(NT)].astype(np.float32).ravel())
pc = bpy.data.objects.new("pc", me)
sc.collection.objects.link(pc)
ng = bpy.data.node_groups.new("pc_gn", "GeometryNodeTree")
ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
gi = ng.nodes.new("NodeGroupInput"); go = ng.nodes.new("NodeGroupOutput")
m2p = ng.nodes.new("GeometryNodeMeshToPoints")
na = ng.nodes.new("GeometryNodeInputNamedAttribute"); na.data_type = "FLOAT"; na.inputs["Name"].default_value = "prad"
sm = ng.nodes.new("GeometryNodeSetMaterial")
pmat = bpy.data.materials.new("pc_mat"); pmat.use_nodes = True
nt = pmat.node_tree; nt.nodes.clear()
o_ = nt.nodes.new("ShaderNodeOutputMaterial"); em = nt.nodes.new("ShaderNodeEmission")
ac = nt.nodes.new("ShaderNodeAttribute"); ac.attribute_name = "pcol"
ae = nt.nodes.new("ShaderNodeAttribute"); ae.attribute_name = "pem"
nt.links.new(ac.outputs["Color"], em.inputs["Color"]); nt.links.new(ae.outputs["Fac"], em.inputs["Strength"])
nt.links.new(em.outputs[0], o_.inputs[0])
sm.inputs["Material"].default_value = pmat
ng.links.new(gi.outputs[0], m2p.inputs["Mesh"])
ng.links.new(na.outputs[0] if "Attribute" not in na.outputs else na.outputs["Attribute"], m2p.inputs["Radius"])
ng.links.new(m2p.outputs[0], sm.inputs["Geometry"])
ng.links.new(sm.outputs[0], go.inputs[0])
mod = pc.modifiers.new("gn", "NODES"); mod.node_group = ng

# ---------------------------------------------------------------- photo card (+ 3D corner brackets)
fr = P["frame"].astype(np.float64)     # tr, br, bl, tl world points on the photo camera's view frame
def on_card(pw_):
    d = pw_ - O; d /= np.linalg.norm(d); return O + d * (D0 / (d @ FWD))
corners = [on_card(c) for c in fr]
cm = bpy.data.meshes.new("card")
cm.from_pydata([tuple(c) for c in corners], [], [(0, 1, 2, 3)])
uvl = cm.uv_layers.new()
vtop = 1 - ((ph * k - 1080) / 2) / (ph * k); vbot = 1 - ((ph * k + 1080) / 2) / (ph * k)
for li, uvv in zip(cm.polygons[0].loop_indices, [(1, vtop), (1, vbot), (0, vbot), (0, vtop)]):
    uvl.data[li].uv = uvv
card = bpy.data.objects.new("card", cm); sc.collection.objects.link(card)
cmat = bpy.data.materials.new("card_mat"); cmat.use_nodes = True
nt = cmat.node_tree; nt.nodes.clear()
o_ = nt.nodes.new("ShaderNodeOutputMaterial"); em = nt.nodes.new("ShaderNodeEmission")
tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = img; tx.interpolation = "Cubic"
nt.links.new(tx.outputs["Color"], em.inputs["Color"]); nt.links.new(em.outputs[0], o_.inputs[0])
card.data.materials.append(cmat)
CARD_EM = em

# thin glowing border + corner brackets around the card (slightly behind it)
bmat = emit_mat("brk", (0.75, 0.86, 1.0), 2.5)
def bar(a, b, w):
    a, b = np.asarray(a), np.asarray(b)
    mid = (a + b) / 2; L = np.linalg.norm(b - a)
    bpy.ops.mesh.primitive_cube_add(size=1)
    ob = bpy.context.active_object
    ob.scale = (L / 2 * 2, w, w)
    xa = (b - a) / L
    za = -FWD
    ya = np.cross(za, xa)
    R = Matrix(((xa[0], ya[0], za[0]), (xa[1], ya[1], za[1]), (xa[2], ya[2], za[2])))
    ob.matrix_world = Matrix.Translation(Vector(mid)) @ R.to_4x4() @ Matrix.Diagonal((L, w, w, 1))
    ob.data.materials.append(bmat)
    return ob
BRK = []
cc = np.array(corners)
ctr = cc.mean(0)
cexp = ctr + (cc - ctr) * 1.025
for i in range(4):
    a, b = cexp[i], cexp[(i + 1) % 4]
    L = 0.07 * np.linalg.norm(cexp[0] - cexp[1])
    da = (b - a) / np.linalg.norm(b - a)
    BRK.append(bar(a, a + da * L, 0.0016))
    BRK.append(bar(b, b - da * L, 0.0016))

# ---------------------------------------------------------------- floor grid (emissive, faint, radial falloff)
floor_z = min((o.matrix_world @ Vector(c)).z for o in TWIN for c in o.bound_box) - 0.001
gm = bpy.data.meshes.new("grid"); gm.from_pydata([(-6, -6, 0), (6, -6, 0), (6, 6, 0), (-6, 6, 0)], [], [(0, 1, 2, 3)])
grid = bpy.data.objects.new("grid", gm); sc.collection.objects.link(grid)
grid.location = (pivot[0], pivot[1], floor_z)
gmat = bpy.data.materials.new("grid_mat"); gmat.use_nodes = True
nt = gmat.node_tree; nt.nodes.clear()
o_ = nt.nodes.new("ShaderNodeOutputMaterial"); em = nt.nodes.new("ShaderNodeEmission")
tc = nt.nodes.new("ShaderNodeTexCoord")
# lines: 1 - smoothstep(frac distance)
sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(tc.outputs["Object"], sep.inputs[0])
def line(c):
    m1 = nt.nodes.new("ShaderNodeMath"); m1.operation = "MULTIPLY"; m1.inputs[1].default_value = 4.0
    nt.links.new(c, m1.inputs[0])
    m2 = nt.nodes.new("ShaderNodeMath"); m2.operation = "PINGPONG"; m2.inputs[1].default_value = 0.5
    nt.links.new(m1.outputs[0], m2.inputs[0])
    m3 = nt.nodes.new("ShaderNodeMapRange"); m3.inputs[1].default_value = 0.0; m3.inputs[2].default_value = 0.006
    m3.inputs[3].default_value = 1.0; m3.inputs[4].default_value = 0.0
    nt.links.new(m2.outputs[0], m3.inputs[0]); return m3.outputs[0]
lx, ly = line(sep.outputs[0]), line(sep.outputs[1])
mx = nt.nodes.new("ShaderNodeMath"); mx.operation = "MAXIMUM"; nt.links.new(lx, mx.inputs[0]); nt.links.new(ly, mx.inputs[1])
vl = nt.nodes.new("ShaderNodeVectorMath"); vl.operation = "LENGTH"; nt.links.new(tc.outputs["Object"], vl.inputs[0])
fall = nt.nodes.new("ShaderNodeMapRange"); fall.inputs[1].default_value = 0.4; fall.inputs[2].default_value = 3.2
fall.inputs[3].default_value = 1.0; fall.inputs[4].default_value = 0.0
nt.links.new(vl.outputs["Value"], fall.inputs[0])
mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; nt.links.new(mx.outputs[0], mul.inputs[0]); nt.links.new(fall.outputs[0], mul.inputs[1])
GRID_K = nt.nodes.new("ShaderNodeMath"); GRID_K.operation = "MULTIPLY"; GRID_K.inputs[1].default_value = 0.10
nt.links.new(mul.outputs[0], GRID_K.inputs[0])
em.inputs["Color"].default_value = (0.55, 0.68, 0.9, 1)
nt.links.new(GRID_K.outputs[0], em.inputs["Strength"]); nt.links.new(em.outputs[0], o_.inputs[0])
grid.data.materials.append(gmat)

# ---------------------------------------------------------------- wireframes of the twin meshes
WIRE_ORDER = ["onion_1", "onion_2", "onion_3", "mixing_bowl_1", "knife_1", "smartphone_1", "spice_box_1", "spoon_1",
              "towel_1", "table"]
def key_of(o):
    if o.name == "towel_flex": return "towel_1"
    if o.parent and o.parent.name.startswith("B_"): return o.parent.name[2:]
    return "table"
wmat = emit_mat("wire_mat", (0.78, 0.88, 1.0), 1.0)
wem = next(n for n in wmat.node_tree.nodes if n.type == "EMISSION")
WIRES = []
for o in TWIN:
    if o.name.startswith(("g3_", "g4_", "g5_", "g6_", "g10_", "g11_", "g12_", "g13_")):   # legs and under-shelf pots: keep it clean
        continue
    w = o.copy(); w.data = o.data.copy()
    for m in list(w.modifiers):
        if m.type in ("SUBSURF", "SOLIDIFY"):
            w.modifiers.remove(m)
    sc.collection.objects.link(w)
    w.parent = None; w.matrix_world = o.matrix_world.copy()
    w.data.materials.clear(); w.data.materials.append(wmat)
    for poly in w.data.polygons: poly.material_index = 0
    nv = len(w.data.vertices)
    nf = len(w.data.polygons)
    if key_of(o) == "table" or nf < 200:
        dm = w.modifiers.new("dis", "DECIMATE"); dm.decimate_type = "DISSOLVE"; dm.angle_limit = math.radians(4)
    else:
        dm = w.modifiers.new("dec", "DECIMATE"); dm.decimate_type = "COLLAPSE"; dm.ratio = min(1.0, 700.0 / nf)
    b = w.modifiers.new("build", "BUILD"); b.use_random_order = True; b.seed = len(WIRES) + 3
    wf = w.modifiers.new("wire", "WIREFRAME")
    wf.thickness = 0.0006 if key_of(o) != "table" else 0.0011
    wf.use_replace = True; wf.use_even_offset = False
    w.hide_render = True
    WIRES.append((key_of(o), w, b))
WIRE_START = {k: 378 + 5 * i for i, k in enumerate(WIRE_ORDER)}
for kk, w, b in WIRES:
    b.frame_start = WIRE_START.get(kk, 420); b.frame_duration = 34 if kk != "table" else 28

# ---------------------------------------------------------------- camera
cam = PHOTO_CAM
cam.animation_data_clear()
cam.data.dof.use_dof = False
cam.data.clip_start = 0.02


def orbit(az, el, dist):
    """camera pose: photo camera rotated about the pivot by az (world Z) and el (camera right), distance scale."""
    rel = Vector(O - pivot) * dist
    Rz = Matrix.Rotation(math.radians(az), 4, "Z")
    Rx = Matrix.Rotation(math.radians(el), 4, Vector(RIGHT))
    R = Rz @ Rx
    pos = Vector(pivot) + (R @ rel.to_4d()).to_3d()
    rot = (R.to_3x3() @ M0.to_3x3())
    return Matrix.Translation(pos) @ rot.to_4x4()


def smoother(t):
    t = min(max(t, 0.0), 1.0); return t * t * t * (t * (t * 6 - 15) + 10)


def catmull(keys, f):
    """keys: [(frame, (az, el, dist))]; C1 path through keys with zero velocity at the ends."""
    fs = [k[0] for k in keys]; vs = [np.array(k[1], float) for k in keys]
    if f <= fs[0]: return vs[0]
    if f >= fs[-1]: return vs[-1]
    i = max(j for j in range(len(fs) - 1) if fs[j] <= f)
    t = (f - fs[i]) / (fs[i + 1] - fs[i])
    def tang(j):
        if j == 0 or j == len(fs) - 1: return np.zeros(3)
        return (vs[j + 1] - vs[j - 1]) / (fs[j + 1] - fs[j - 1]) * (fs[i + 1] - fs[i])
    m0, m1 = tang(i), tang(i + 1)
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * vs[i] + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * vs[i + 1] + (t3 - t2) * m1 * 1.0


CAM_A = [(0, (-24.0, -5.0, 1.55)), (70, (0.0, 0.0, 1.0))]
CAM_D = [(238, (0.0, 0.0, 1.0)), (330, (30.0, 9.0, 0.97)), (382, (40.0, 13.0, 0.9)), (446, (0.0, 0.0, 1.0))]


def cam_params(f):
    if f <= 70:
        t = smoother(f / 70.0)
        v = np.array(CAM_A[0][1]) * (1 - t) + np.array(CAM_A[1][1]) * t
        sh = t ** 3
    elif f < 238:
        v = np.array((0.0, 0.0, 1.0)); sh = 1.0
    else:
        v = catmull(CAM_D, f)
        sh = 1.0 - min(1.0, abs(v[0]) / 8.0)
        sh = smoother(sh)
    return v, sh


def set_cam(f):
    (az, el, dist), sh = cam_params(f)
    cam.matrix_world = orbit(az, el, dist)
    cam.data.shift_x, cam.data.shift_y = SHIFT0.x * sh, SHIFT0.y * sh
    cam.data.lens = LENS
    still = (abs(az) < 1e-3 and abs(el) < 1e-3)
    cam.data.dof.use_dof = bool(not still)
    cam.data.dof.focus_distance = float(np.linalg.norm(np.array(cam.matrix_world.translation) - pivot))
    cam.data.dof.aperture_fstop = 11.0


# ---------------------------------------------------------------- per-frame state
def ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)

# extrusion wave: starts at the image centre-bottom (nearest table edge) and travels outward
wave = gx / 1920.0 * 0.9 + 0.10 * rnd                  # left -> right, like the scan line
# density falloff: sparser towards the frame edges; background (no twin geometry): sparse dust only
rr = np.hypot((gx - 960) / 960, (gy - 540) / 540)
keep_p = np.where(valid, 1 - 0.85 * np.clip((rr - 0.75) / 0.55, 0, 1) ** 1.5, 0.05)
alive = rnd < keep_p
alive_dust = (~valid) & (rng.random(N) < 0.6) & alive
# depth tint: near = cool white, far = blue
dn = np.clip((np.nan_to_num(depth, nan=np.nanmax(depth)) - np.nanmin(depth)) / (np.nanmax(depth) - np.nanmin(depth) + 1e-9), 0, 1)
tint = np.c_[0.75 - 0.45 * dn, 0.86 - 0.25 * dn, np.ones(N)] * 1.0
UP = np.array([0.0, 0.0, 1.0])
layer = (np.round(gy / S).astype(int) % 4).astype(float)        # spatially coherent strata: every 4th scan row shares a layer
lift_amp = 0.008 * (1 + layer) + 0.003 * rng.random(N)   # 4 discrete Z layers -> visible depth strata
BCOL = np.tile(np.array([[0.35, 0.65, 1.0]]), (NB, 1))
order_i = {nm: i for i, nm in enumerate(WIRE_ORDER)}
obj_rank = np.array([order_i.get(names[h], 9) if h >= 0 else 10 for h in hit])
bg_drift = rng.normal(0, 1, (N, 3)) * np.array([0.05, 0.05, 0.02])


def set_points(f):
    show = 238 <= f <= 446
    pc.hide_render = not show
    if not show:
        return
    # staggered extrusion, left -> right (f246..~330): each point LIFTS off the photo plane, separates in Z, settles on the twin
    t = (f - 246) / 58.0 - wave * 0.75
    e = np.clip(t, 0, 1)
    e = 1 - (1 - e) ** 3
    d = d_flat + (d_true - d_flat) * e
    pos = O + Dirs * d[:, None]
    lift = np.sin(np.pi * np.clip(t, 0, 1)) * lift_amp * (t < 1.0)
    breathe = 0.003 * lift_amp / 0.035 * np.sin(f * 0.08 + rnd * 6.28) * ss(300, 330, f) * (1 - ss(360, 380, f))
    pos += UP * (lift + breathe)[:, None]
    # dust (no twin geometry: wall / floor in the photo): slow drift away, depth-tinted, sparse
    fade_bg = ss(244, 300, f + 30 * (1 - rnd))
    pos[~valid] += (bg_drift[~valid] * 0.6 + Dirs[~valid] * 0.12) * fade_bg[~valid, None]
    crest = np.exp(-((t - 0.35) / 0.18) ** 2) * (t > 0)
    em = 1.1 + 0.6 * crest
    st = 370 + obj_rank * 4 + rnd * 8
    c = ss(0, 22, f - st)
    ramp = ss(242, 262, f)
    rad = (rad0 + (rad_pt - rad0) * ramp) * (1 - c)
    rad = np.where(alive | (f < 242), rad, rad * (1 - ramp))
    rad[~valid] *= np.where(alive_dust[~valid], 0.7, 1 - ramp)
    em = em * (1 + 2.0 * c * (1 - c) * 4)
    em[~valid] *= 0.55
    em *= ss(237, 246, f)
    # colour: photo mixed with the depth tint as the point lifts (crest), keeps 35% tint afterwards
    mixk = np.clip(0.15 * ramp + 0.2 * crest, 0, 0.4)[:, None]
    cc = col * (1 - mixk) + tint * mixk
    cc[~valid] = tint[~valid] * 0.8
    # bridges between depth layers: glow on the seams, appear once both ends have moved apart
    pb = pos[BA] + (pos[BB] - pos[BA]) * BS[:, None]
    gap = np.abs(d[BA] - d[BB])
    eb = np.clip(gap / 0.02, 0, 1) * (1 - np.maximum(c[BA], c[BB]))
    rb = rad_pt[BA] * 0.8 * eb
    emb = (1.6 + 1.5 * np.maximum(crest[BA], crest[BB])) * eb
    pos = np.r_[pos, pb]
    rad = np.r_[rad, rb]
    em = np.r_[em, emb]
    cc = np.r_[cc, BCOL]
    pos = np.where(np.isfinite(pos), pos, O)
    me.vertices.foreach_set("co", pos.astype(np.float32).ravel())
    me.attributes["prad"].data.foreach_set("value", np.maximum(rad, 0).astype(np.float32))
    me.attributes["pcol"].data.foreach_set("color", np.c_[cc, np.ones(len(cc))].astype(np.float32).ravel())
    me.attributes["pem"].data.foreach_set("value", em.astype(np.float32))
    me.update()


def set_card(f):
    a = 1.0 if f < 238 else 1.0 - ss(238, 250, f)
    card.hide_render = bool(a <= 0.001)
    CARD_EM.inputs["Strength"].default_value = float(a)
    vb = 1.0 if f <= 70 else max(0.0, 1.0 - (f - 70) / 12.0)
    for b in BRK:
        b.hide_render = bool(vb <= 0.01)
    next(n for n in bmat.node_tree.nodes if n.type == "EMISSION").inputs["Strength"].default_value = float(2.5 * vb)


def set_wires(f, force=False):
    for kk, w, b in WIRES:
        w.hide_render = bool(not (force or f >= b.frame_start))
    wem.inputs["Strength"].default_value = 1.2


def set_grid(f):
    GRID_K.inputs[1].default_value = 0.10 * (0.6 + 0.4 * ss(240, 320, f))


def render(f, path):
    sc.frame_set(int(f))
    set_cam(f); set_card(f); set_points(f); set_wires(f); set_grid(f)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


if A.wire_still:
    sc.frame_set(500)
    set_cam(500); set_card(500); pc.hide_render = True; set_wires(500, force=True); grid.hide_render = True
    for b in BRK: b.hide_render = True
    sc.render.filepath = os.path.join(OUT, "wire_still.png")
    bpy.ops.render.render(write_still=True)
    # per-object id still (flat colours) is not needed: the comp uses SAM masks
    sys.exit(0)

frames = [int(x) for x in A.list.split(",")] if A.list else list(range(*[int(x) for x in A.frames.split(":")], A.step))
for f in frames:
    p = os.path.join(OUT, f"b_{f:04d}.png")
    if os.path.exists(p) and not A.list:
        continue
    render(f, p)
    print("FRAME", f, flush=True)
