"""Sequence 2 "Many futures": Blender (Eevee) render of the REAL rehearsed candidate plans of one table.

blender -b --factory-startup --python fut_render.py -- --nn 50 --frames 0:340 [--res 0.5] [--out DIR] [--list a,b]

Scene: the MuJoCo model's own visual geometry (Franka Panda menagerie meshes, object primitives, table), exported by
futures_export.py, in a dark simulation studio. Every ghost is one recorded simulation from the planner's search
(log_futures.py): body poses at 50 Hz exactly as simulated. Playback is time-scaled (SPEED x real sim time).
Colour logic: white = a rehearsal in progress; red = that rehearsal failed (flashes at the failure time recorded in
the simulation, then dissolves); green = the one plan that passed every goal (the plan that was sent).
Writes frames b_XXXX.png and chips.json (screen anchors of failure chips) for fut_compose.py.
"""
import bpy, sys, os, json, math, argparse
import numpy as np
from mathutils import Matrix, Vector, Quaternion
from bpy_extras.object_utils import world_to_camera_view

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--nn", default="50")
ap.add_argument("--frames", default="0:340")
ap.add_argument("--list", default=None)
ap.add_argument("--res", type=float, default=1.0)
ap.add_argument("--samples", type=int, default=24)
ap.add_argument("--out", default=None)
ap.add_argument("--chips-only", action="store_true")
A = ap.parse_args(argv)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA = os.path.join(ROOT, "footage/vfx/_data")
OUT = A.out or os.path.join(ROOT, f"footage/vfx/_work/fut{A.nn}_blender")
os.makedirs(OUT, exist_ok=True)
Z = np.load(os.path.join(DATA, f"{A.nn}_blend.npz"))
SUM = json.load(open(os.path.join(DATA, f"{A.nn}_futures_summary.json")))
GEOMS = json.loads(str(Z["geoms"]))
BN = list(Z["bnames"])
ARM = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger"]
OBJ_IDS = [o["id"] for o in SUM["objects"]]

# ------------------------------------------------------------------ choose rehearsals and the playback clock
sims = [s for s in SUM["sims"] if s["n_act"] > 0]
plan = SUM["plan"]["actions"]
sol = next(s for s in sims if s["success"] and s["actions"] == plan)
SOL = sol["i"]
FPS = 30
F_START, F_END_SOL = 84, 300            # rehearsal playback window (the solution finishes at F_END_SOL)
SPEED = sol["sim_time"] / ((F_END_SOL - F_START) / FPS)
MOVERS = sorted({a["obj"] for s in sims for a in s["actions"]} | {a["dest"] for s in sims for a in s["actions"]})


def evid_time(s):
    import re
    for e in s.get("evidence") or []:
        mm = re.match(r"t=([\d.]+)s", e)
        if mm:
            return float(mm.group(1)), e
    return None, None


for s in sims:
    te, ev = evid_time(s)
    s["t_show_fail"] = None
    if not s["success"]:
        s["t_show_fail"] = te if te is not None else (s["t_fail"] if s["t_fail"] is not None else s["sim_time"])
        s["ev"] = ev


def f_of(t):
    return F_START + t / SPEED * FPS


def t_of(f):
    return max(0.0, (f - F_START) / FPS * SPEED)


# ------------------------------------------------------------------ scene
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
sc.render.resolution_percentage = int(100 * A.res)
sc.eevee.taa_render_samples = A.samples
sc.view_settings.view_transform = "AgX"
try:
    sc.view_settings.look = "AgX - Medium High Contrast"
except Exception:
    pass
sc.render.image_settings.file_format = "PNG"
world = bpy.data.worlds.new("void"); sc.world = world; world.use_nodes = True
bgn = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
bgn.inputs[0].default_value = (0.004, 0.005, 0.007, 1); bgn.inputs[1].default_value = 1.0


def principled(name, rgba, rough=0.45, metal=0.0, coat=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = (*rgba[:3], 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    try:
        b.inputs["Coat Weight"].default_value = coat
    except Exception:
        pass
    return m


def ghost_mat(name):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mx = nt.nodes.new("ShaderNodeMixShader")
    lw = nt.nodes.new("ShaderNodeLayerWeight"); lw.inputs["Blend"].default_value = 0.35
    # fresnel-ish rim: edges brighter
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
    add = nt.nodes.new("ShaderNodeMath"); add.operation = "ADD"; add.inputs[1].default_value = 0.35
    nt.links.new(lw.outputs["Facing"], add.inputs[0])
    nt.links.new(add.outputs[0], mul.inputs[0])
    mul.inputs[1].default_value = 0.2            # <- alpha, driven per frame
    nt.links.new(mul.outputs[0], mx.inputs[0])
    nt.links.new(tr.outputs[0], mx.inputs[1]); nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], o.inputs[0])
    em.inputs["Strength"].default_value = 2.0
    try:
        m.surface_render_method = "BLENDED"
    except Exception:
        m.blend_method = "BLEND"
    m.use_backface_culling = False
    return m, em, mul


def quat(q):
    return Quaternion((q[0], q[1], q[2], q[3]))


def geom_mesh(g):
    t, s = g["type"], g["size"]
    me = bpy.data.meshes.new(f"g{g['g']}")
    import bmesh
    bm = bmesh.new()
    if t == 6:
        bmesh.ops.create_cube(bm, size=2.0)
        for v in bm.verts:
            v.co = Vector((v.co.x * s[0], v.co.y * s[1], v.co.z * s[2]))
    elif t == 5:
        bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=s[0], radius2=s[0], depth=2 * s[1])
    elif t in (2, 4):
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
        r = (s[0], s[0], s[0]) if t == 2 else (s[0], s[1], s[2])
        for v in bm.verts:
            v.co = Vector((v.co.x * r[0], v.co.y * r[1], v.co.z * r[2]))
    elif t == 3:
        bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=s[0], radius2=s[0], depth=2 * s[1])
    elif t == 7:
        V = Z[f"mesh{g['mesh']}_v"]; Fc = Z[f"mesh{g['mesh']}_f"]
        me.from_pydata(V.tolist(), [], Fc.tolist())
        me.update()
        for p in me.polygons:
            p.use_smooth = True
        bm.free()
        return me
    else:
        bm.free(); return None
    bm.to_mesh(me); bm.free()
    if t in (5, 3, 2, 4):
        for p in me.polygons:
            p.use_smooth = True
    return me


def local_mat(g):
    return Matrix.Translation(Vector(g["pos"])) @ quat(g["quat"]).to_matrix().to_4x4()


# body template meshes: one joined mesh per body (geoms in the body frame)
by_body = {}
for g in GEOMS:
    if g["body"] == 0:
        continue
    by_body.setdefault(g["bname"], []).append(g)

MAT_CACHE = {}


def mat_for(bname, rgba):
    if bname in ARM:
        lum = sum(rgba[:3]) / 3
        key = ("arm", lum > 0.5)
        if key not in MAT_CACHE:
            MAT_CACHE[key] = principled("arm_w" if lum > 0.5 else "arm_k", (0.86, 0.87, 0.88) if lum > 0.5 else (0.05, 0.05, 0.055),
                                        rough=0.35, coat=0.3)
        return MAT_CACHE[key]
    if bname == "table":
        if "tbl" not in MAT_CACHE:
            MAT_CACHE["tbl"] = principled("tbl", (0.10, 0.11, 0.125), rough=0.3, coat=0.5)
        return MAT_CACHE["tbl"]
    key = ("o", bname, tuple(round(c, 2) for c in rgba[:3]))
    if key not in MAT_CACHE:
        c = np.array(rgba[:3]); c = c * 0.85 + 0.05
        MAT_CACHE[key] = principled("m_" + bname, c, rough=0.55)
    return MAT_CACHE[key]


def body_template(bname, decimate=None):
    import bmesh
    bm = bmesh.new()
    mats = []
    for g in by_body[bname]:
        me = geom_mesh(g)
        if me is None:
            continue
        me.transform(local_mat(g))
        mat = mat_for(bname, g["rgba"])
        if mat not in mats:
            mats.append(mat)
        mi = mats.index(mat)
        for p in me.polygons:
            p.material_index = mi
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    me = bpy.data.meshes.new("T_" + bname)
    bm.to_mesh(me); bm.free()
    for mt in mats:
        me.materials.append(mt)
    if decimate:
        ob = bpy.data.objects.new("tmp", me); sc.collection.objects.link(ob)
        dm = ob.modifiers.new("d", "DECIMATE"); dm.ratio = decimate
        dg = bpy.context.evaluated_depsgraph_get()
        me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
        bpy.data.objects.remove(ob); me = me2
    return me


SOLID = {}
for bname in by_body:
    me = body_template(bname)
    ob = bpy.data.objects.new("S_" + bname, me); sc.collection.objects.link(ob)
    ob.rotation_mode = "QUATERNION"
    SOLID[bname] = ob
# world geoms (table top, legs, pedestal)
tabmat = principled("table", (0.045, 0.05, 0.058), rough=0.3, coat=0.5)
for g in GEOMS:
    if g["body"] != 0 or g["type"] == 0:
        continue
    me = geom_mesh(g)
    if me is None:
        continue
    ob = bpy.data.objects.new(f"W{g['g']}", me); sc.collection.objects.link(ob)
    R = Matrix(np.array(g["xmat"]).reshape(3, 3).tolist())
    ob.matrix_world = Matrix.Translation(Vector(g["pos"])) @ R.to_4x4()
    c = g["rgba"]
    ob.data.materials.append(tabmat if max(g["size"]) > 0.25 else principled("w", (0.22, 0.23, 0.25), 0.5))
# edge light strip around the table top (cool white) - the studio look
tb = SUM["table"]

# ghost templates (decimated) per arm body + movers
GT = {b: body_template(b, decimate=0.18 if b in ARM else None) for b in by_body if b in ARM or b in MOVERS}
PH = bpy.data.materials.new("placeholder")
for me in GT.values():
    me.materials.clear(); me.materials.append(PH)
    for p in me.polygons:
        p.material_index = 0

# ghost ARMS for a diverse subset (distinct action sequences, capped); every rehearsal still gets its trail
sig_seen, ARMSET = set(), set()
for s in sorted(sims, key=lambda s: (s["i"] != SOL, -s["n_act"], s["i"])):
    sig = json.dumps(s["actions"][-1], sort_keys=True) + str(s["n_act"])
    if sig in sig_seen:
        continue
    sig_seen.add(sig); ARMSET.add(s["i"])
    if len(ARMSET) >= 38:
        break
ARMSET.add(SOL)
print("ARM GHOSTS", len(ARMSET))
GHOSTS = []
for s in sims:
    gm, em, alpha = ghost_mat(f"gm{s['i']}")
    obs = {}
    for b, me in (GT.items() if s["i"] in ARMSET else []):
        ob = bpy.data.objects.new(f"G{s['i']}_{b}", me); sc.collection.objects.link(ob)
        ob.rotation_mode = "QUATERNION"
        ob.material_slots[0].link = "OBJECT"; ob.material_slots[0].material = gm
        obs[b] = ob
    # TCP trail (hand body path) as a curve
    T = Z[f"sim{s['i']}_t"]; P = Z[f"sim{s['i']}_p"]
    hi = BN.index("hand") - 1
    pts = P[:, hi, :3]
    cu = bpy.data.curves.new(f"trail{s['i']}", "CURVE"); cu.dimensions = "3D"
    sp = cu.splines.new("POLY"); sp.points.add(len(pts) - 1)
    for j, p in enumerate(pts):
        sp.points[j].co = (float(p[0]), float(p[1]), float(p[2]) - 0.0, 1.0)
    cu.bevel_depth = 0.0009; cu.bevel_resolution = 2
    cu.bevel_factor_mapping_end = "SPLINE"
    tro = bpy.data.objects.new(f"trail{s['i']}", cu); sc.collection.objects.link(tro)
    tm, tem, talpha = ghost_mat(f"tm{s['i']}")
    tro.data.materials.append(tm)
    GHOSTS.append({"s": s, "obs": obs, "mat": (gm, em, alpha), "trail": tro, "tmat": (tm, tem, talpha), "T": T, "P": P})

# ------------------------------------------------------------------ studio: grid floor + lights
FLOOR_Z = min(g["pos"][2] - (g["size"][2] if g["type"] == 6 else 0) for g in GEOMS if g["body"] == 0 and g["type"] == 6)
gmesh = bpy.data.meshes.new("grid"); gmesh.from_pydata([(-6, -6, 0), (6, -6, 0), (6, 6, 0), (-6, 6, 0)], [], [(0, 1, 2, 3)])
grid = bpy.data.objects.new("grid", gmesh); sc.collection.objects.link(grid)
grid.location = (0.3, 0, min(FLOOR_Z, 0.0) - 0.002)
gmat = bpy.data.materials.new("grid_mat"); gmat.use_nodes = True
nt = gmat.node_tree; nt.nodes.clear()
o_ = nt.nodes.new("ShaderNodeOutputMaterial"); bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
bs.inputs["Base Color"].default_value = (0.012, 0.013, 0.016, 1); bs.inputs["Roughness"].default_value = 0.32
em = nt.nodes.new("ShaderNodeEmission"); add = nt.nodes.new("ShaderNodeAddShader")
tc = nt.nodes.new("ShaderNodeTexCoord"); sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(tc.outputs["Object"], sep.inputs[0])


def line(c):
    m1 = nt.nodes.new("ShaderNodeMath"); m1.operation = "MULTIPLY"; m1.inputs[1].default_value = 2.0
    nt.links.new(c, m1.inputs[0])
    m2 = nt.nodes.new("ShaderNodeMath"); m2.operation = "PINGPONG"; m2.inputs[1].default_value = 0.5
    nt.links.new(m1.outputs[0], m2.inputs[0])
    m3 = nt.nodes.new("ShaderNodeMapRange"); m3.inputs[1].default_value = 0.0; m3.inputs[2].default_value = 0.004
    m3.inputs[3].default_value = 1.0; m3.inputs[4].default_value = 0.0
    nt.links.new(m2.outputs[0], m3.inputs[0]); return m3.outputs[0]


lx, ly = line(sep.outputs[0]), line(sep.outputs[1])
mx = nt.nodes.new("ShaderNodeMath"); mx.operation = "MAXIMUM"; nt.links.new(lx, mx.inputs[0]); nt.links.new(ly, mx.inputs[1])
vl = nt.nodes.new("ShaderNodeVectorMath"); vl.operation = "LENGTH"; nt.links.new(tc.outputs["Object"], vl.inputs[0])
fall = nt.nodes.new("ShaderNodeMapRange"); fall.inputs[1].default_value = 0.3; fall.inputs[2].default_value = 3.5
fall.inputs[3].default_value = 1.0; fall.inputs[4].default_value = 0.0
nt.links.new(vl.outputs["Value"], fall.inputs[0])
mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; nt.links.new(mx.outputs[0], mul.inputs[0]); nt.links.new(fall.outputs[0], mul.inputs[1])
k_ = nt.nodes.new("ShaderNodeMath"); k_.operation = "MULTIPLY"; k_.inputs[1].default_value = 0.35; nt.links.new(mul.outputs[0], k_.inputs[0])
em.inputs["Color"].default_value = (0.5, 0.62, 0.85, 1)
nt.links.new(k_.outputs[0], em.inputs["Strength"])
nt.links.new(bs.outputs[0], add.inputs[0]); nt.links.new(em.outputs[0], add.inputs[1]); nt.links.new(add.outputs[0], o_.inputs[0])
grid.data.materials.append(gmat)


def light(name, typ, loc, rot, energy, size, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, typ); ld.energy = energy; ld.color = color
    if typ == "AREA":
        ld.size = size
    ob = bpy.data.objects.new(name, ld); sc.collection.objects.link(ob)
    ob.location = loc; ob.rotation_euler = [math.radians(a) for a in rot]
    return ob


light("key", "AREA", (0.6, -0.9, 2.2), (25, 0, 20), 260, 1.6, (1.0, 0.98, 0.95))
light("rim", "AREA", (-0.6, 1.2, 1.4), (-60, 0, 200), 160, 1.2, (0.7, 0.8, 1.0))
light("fill", "AREA", (1.8, 0.6, 0.8), (0, 70, 20), 50, 2.0, (0.8, 0.85, 1.0))

# ------------------------------------------------------------------ camera
cam_d = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_d); sc.collection.objects.link(cam); sc.camera = cam
cam_d.lens = 35; cam_d.sensor_width = 36
cam_d.dof.use_dof = True; cam_d.dof.aperture_fstop = 2.8
TGT = Vector((tb["center"][0] - 0.05, tb["center"][1] * 0.5, 0.86 if FLOOR_Z < 0.2 else 0.3))


def tbl_top_z():
    zs = [g["pos"][2] + g["size"][2] for g in GEOMS if g["body"] == 0 and g["type"] == 6 and max(g["size"]) > 0.25]
    return max(zs) if zs else 0.0


TOP = tbl_top_z()
TGT = Vector((tb["center"][0] * 0.62, 0.0, TOP + 0.24))


def set_cam(f):
    u = f / 340.0
    az = math.radians(-38 + 14 * (u - 0.5) * 2 * 0.5)      # slow drift, -45 .. -31 deg
    el = math.radians(31 - 3 * u)
    dist = 2.45 - 0.22 * (1 - (1 - u) ** 2)
    pos = TGT + Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el))) * dist
    cam.location = pos
    cam.rotation_euler = (TGT - pos).to_track_quat("-Z", "Y").to_euler()
    cam_d.dof.focus_distance = (TGT - pos).length


# ------------------------------------------------------------------ per frame
def pose_at(G, t):
    T, P = G["T"], G["P"]
    k = int(np.clip(np.searchsorted(T, t), 0, len(T) - 1))
    return P[k]


def apply_pose(obs, row):
    for b, ob in obs.items():
        p = row[BN.index(b) - 1]
        ob.location = Vector(p[:3].tolist()); ob.rotation_quaternion = Quaternion(p[3:7].tolist())


SOL_G = next(G for G in GHOSTS if G["s"]["i"] == SOL)
WHITE, RED, GREEN = (0.75, 0.86, 1.0), (1.0, 0.16, 0.10), (0.35, 1.0, 0.55)


def ss(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0); return t * t * (3 - 2 * t)


chip_log = {}


def frame_state(f):
    t = t_of(f)
    start = ss(F_START - 14, F_START + 10, f)                 # ghosts born from the solid arm
    for G in GHOSTS:
        s = G["s"]
        tf = s["t_show_fail"]
        tt = min(t, tf if tf is not None else s["sim_time"])
        apply_pose(G["obs"], pose_at(G, tt))
        gm, em, alpha = G["mat"]; tm, tem, talpha = G["tmat"]
        is_sol = s["i"] == SOL
        if is_sol:
            col = tuple(np.array(WHITE) * (1 - ss(F_END_SOL - 60, F_END_SOL - 20, f)) + np.array(GREEN) * ss(F_END_SOL - 60, F_END_SOL - 20, f))
            a = 0.16 * start + 0.5 * ss(F_END_SOL - 40, F_END_SOL, f)
            a *= 1 - ss(F_END_SOL + 4, F_END_SOL + 24, f)       # hands over to the solid arm
            strength = 2.5
        elif tf is not None:
            ff = f_of(tf)
            red = ss(ff - 2, ff + 1, f)
            col = tuple(np.array(WHITE) * (1 - red) + np.array(RED) * red)
            flash = math.exp(-max(0.0, f - ff) / 6.0) * red
            a = (0.035 * start + 0.13 * flash + 0.03 * red) * (1 - ss(ff + 10, ff + 32, f))
            strength = 2.0 + 6.0 * flash
            if ff - 1 <= f:
                chip_log.setdefault(s["i"], {"f_fail": ff})
        else:                                               # a partial plan that passed: the search builds on it
            fe = f_of(s["sim_time"])
            col = WHITE
            a = 0.035 * start * (1 - ss(fe - 4, fe + 14, f))
            strength = 2.0
        em.inputs["Color"].default_value = (*col, 1); em.inputs["Strength"].default_value = strength
        alpha.inputs[1].default_value = max(0.0, a)
        tem.inputs["Color"].default_value = (*col, 1); tem.inputs["Strength"].default_value = strength * 1.4
        talpha.inputs[1].default_value = max(0.0, min(1.0, a * (2.0 if is_sol else 1.6)))
        frac = 0.0 if t <= 0 else min(1.0, np.searchsorted(G["T"], tt) / max(1, len(G["T"]) - 1))
        G["trail"].data.bevel_factor_end = float(frac)
        hide = bool(a <= 0.002)
        for ob in G["obs"].values():
            ob.hide_render = hide
        G["trail"].hide_render = bool(a <= 0.002 or frac <= 0.001)
    # solid arm: at home before the rehearsal, fades out while the ghosts run, then the sent plan plays solid
    if f < F_END_SOL:
        row = pose_at(SOL_G, 0.0 if f < F_START else 0.0)
    else:
        row = pose_at(SOL_G, sol["sim_time"])
    for b, ob in SOLID.items():
        if b in ARM or b in MOVERS:
            p = row[BN.index(b) - 1]
            ob.location = Vector(p[:3].tolist()); ob.rotation_quaternion = Quaternion(p[3:7].tolist())
    vis_solid = f < F_START - 2 or f >= F_END_SOL + 10
    for b, ob in SOLID.items():
        if b in ARM or b in MOVERS:
            ob.hide_render = bool(not vis_solid)
        else:
            p = pose_at(SOL_G, 0.0)[BN.index(b) - 1]
            ob.location = Vector(p[:3].tolist()); ob.rotation_quaternion = Quaternion(p[3:7].tolist())
    # movers: during the rehearsal they exist only as ghosts; keep the solid ones at their start pose until then
    set_cam(f)


def project(p):
    co = world_to_camera_view(sc, cam, Vector(p))
    return [co.x * 1920, (1 - co.y) * 1080, co.z]


frames = [int(x) for x in A.list.split(",")] if A.list else list(range(*[int(x) for x in A.frames.split(":")]))
# chip anchors for every failing rehearsal (computed for all frames cheaply)
chips = {}
for G in GHOSTS:
    s = G["s"]
    if s["t_show_fail"] is None:
        continue
    ff = f_of(s["t_show_fail"])
    set_cam(int(round(ff)))
    row = pose_at(G, s["t_show_fail"])
    obj = None
    if s.get("ev"):
        import re
        mm = re.search(r"held (\S+) touched (\S+)", s["ev"])
        obj = mm.group(2) if mm else None
    if not obj and s["disturbed"]:
        obj = s["disturbed"][0]["obj"]
    anchor = row[BN.index(obj) - 1][:3] if obj in BN else row[BN.index("hand") - 1][:3]
    chips[s["i"]] = {"f_fail": ff, "xy": project(anchor.tolist()), "obj": obj, "ev": s.get("ev"),
                     "disturbed": s["disturbed"], "failed": s["failed"], "error": s["execution_error"]}
meta = {"nn": A.nn, "speed": SPEED, "F_START": F_START, "F_END_SOL": F_END_SOL, "sol": SOL, "n_sims": len(SUM["sims"]),
        "n_shown": len(sims), "chips": chips,
        "sol_end_xy": project(pose_at(SOL_G, sol["sim_time"])[BN.index("hand") - 1][:3].tolist())}
json.dump(meta, open(os.path.join(OUT, "chips.json"), "w"), indent=1, default=float)
if A.chips_only:
    sys.exit(0)
for f in frames:
    p = os.path.join(OUT, f"b_{f:04d}.png")
    if os.path.exists(p) and not A.list:
        continue
    frame_state(f)
    sc.render.filepath = p
    bpy.ops.render.render(write_still=True)
    print("FRAME", f, flush=True)
