"""Export the logged futures of one table (log_futures.py output) + the MuJoCo model's visual geometry to an npz
that Blender can read (Blender has numpy but not mujoco).

.venv/bin/python scripts/vfx/futures_export.py 50
Writes footage/vfx/_data/NN_blend.npz and NN_futures_summary.json (per-simulation outcome, failure time and reason).
Everything here is copied from the MuJoCo model / the recorded simulations; nothing is invented.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mujoco
from rehearsal.scene import build_model

NN = sys.argv[1] if len(sys.argv) > 1 else "50"
DATA = ROOT.parent / "footage/vfx/_data"
sc = json.loads((ROOT.parent / f"footage/tables_sim/hero/exec/_cache/{NN}_scene.json").read_text())
FU = json.loads((DATA / f"{NN}_futures.json").read_text())
m = build_model(sc)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
bnames = [m.body(b).name for b in range(m.nbody)]
assert FU["body_names"] == bnames[1:], "trajectory body order mismatch"
ARM = {"link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger"}

geoms, meshes = [], {}
for g in range(m.ngeom):
    b = int(m.geom_bodyid[g]); bn = bnames[b]
    grp = int(m.geom_group[g])
    if bn in ARM and grp != 2:          # arm: visual meshes only
        continue
    if bn not in ARM and grp not in (0,):
        continue
    if int(m.geom_rgba[g][3]) == 0 and m.geom_rgba[g][3] < 0.01:
        continue
    t = int(m.geom_type[g])
    rec = {"g": g, "body": b, "bname": bn, "type": t, "size": m.geom_size[g].tolist(), "pos": m.geom_pos[g].tolist(),
           "quat": m.geom_quat[g].tolist(), "rgba": m.geom_rgba[g].tolist(), "mesh": int(m.geom_dataid[g]) if t == 7 else -1}
    if b == 0:                            # world geoms: store world pose directly
        rec["pos"], rec["quat"] = d.geom_xpos[g].tolist(), None
        rec["xmat"] = d.geom_xmat[g].tolist()
    if t == 7:
        mid = rec["mesh"]
        if mid not in meshes:
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            meshes[mid] = (m.mesh_vert[va:va + vn].copy(), m.mesh_face[fa:fa + fn].copy())
    if t == 7 and bn in ARM:
        # material colour of the menagerie mesh
        mat = int(m.geom_matid[g])
        if mat >= 0:
            rec["rgba"] = m.mat_rgba[mat].tolist()
    geoms.append(rec)

objs = [o["id"] for o in sc["objects"]]
out = {}
summ = []
for s in FU["sims"]:
    tr = np.asarray(s["traj"], np.float32)              # [T, 1 + 7 * (nbody-1)]
    T = tr[:, 0]
    P = tr[:, 1:].reshape(len(tr), m.nbody - 1, 7)
    out[f"sim{s['i']}_t"] = T
    out[f"sim{s['i']}_p"] = P
    # failure moment: first time a disturbed object has moved > 1 cm (xy) or tilted, from its own trajectory
    t_fail, why = None, None
    if not s["success"]:
        cands = []
        for dd in s["disturbed"]:
            ob = dd.get("obj")
            if ob not in bnames:
                continue
            bi = bnames.index(ob) - 1
            p = P[:, bi, :3]
            mv = np.linalg.norm(p[:, :2] - p[0, :2], axis=1)
            k = np.argmax(mv > 0.01) if (mv > 0.01).any() else None
            if k is not None:
                cands.append((float(T[k]), ob, dd))
        if cands:
            t_fail, ob, dd = min(cands, key=lambda c: c[0])
            why = dd
        elif s["execution_error"]:
            why = s["execution_error"]
        else:
            why = {"failed": s["failed"]}
    summ.append({"i": s["i"], "n_act": len(s["actions"]), "actions": s["actions"], "success": s["success"],
                 "sim_time": s["sim_time"], "t_fail": t_fail, "why": why, "failed": s["failed"],
                 "disturbed": s["disturbed"], "evidence": s["evidence"], "execution_error": s["execution_error"],
                 "events": s["events"]})

np.savez_compressed(DATA / f"{NN}_blend.npz", geoms=np.array(json.dumps(geoms)), bnames=np.array(bnames),
                    **{f"mesh{k}_v": v[0] for k, v in meshes.items()}, **{f"mesh{k}_f": v[1] for k, v in meshes.items()},
                    **out)
(DATA / f"{NN}_futures_summary.json").write_text(json.dumps({"scene": NN, "stats": FU["stats"], "plan": FU["plan"],
                                                              "shipped_stats": FU["shipped_stats"], "shipped_plan": FU["shipped_plan"],
                                                              "table": sc["table"], "objects": [{"id": o["id"], "label": o["label"]} for o in sc["objects"]],
                                                              "sims": summ}, indent=1, default=float))
print("geoms", len(geoms), "meshes", len(meshes), "sims", len(summ))
