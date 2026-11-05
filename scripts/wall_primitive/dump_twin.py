"""Physics twin of one AI eval table -> npz of drawable geoms (world frame) + the photo camera, for the primitive wall.
  .venv/bin/python scripts/wall_primitive/dump_twin.py OUT_DIR t6_NN [t6_NN ...]
Draws what the physics model collides with (objects: group 0 geoms), the table and the arm at its home pose (settled).
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import mujoco  # noqa: E402
import render_tables as RT  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for name in sys.argv[2:]:
    sc = json.loads((ROOT / "scenes" / f"{name}.json").read_text())
    assert sc.get("ai_generated") and sc.get("privacy") == "public", name
    sc2 = dict(sc); sc2["visual"] = {k: v for k, v in (sc.get("visual") or {}).items() if k != "table_texture"}
    m = RT.finish(RT.base_spec(sc2))
    d = RT.settle(m, sc)
    objs = {o["id"] for o in sc["objects"]}
    G = []
    for g in range(m.ngeom):
        t, grp, rgba = int(m.geom_type[g]), int(m.geom_group[g]), m.geom_rgba[g].copy()
        if t == mujoco.mjtGeom.mjGEOM_PLANE or grp > 2:
            continue
        bn = m.body(m.geom_bodyid[g]).name
        kind = "obj" if bn in objs else ("table" if bn in ("table", "world") else "arm")
        if kind == "arm" or (bn == "world" and d.geom_xpos[g][0] < 0.2):
            continue                      # the arm / pedestal stand between the photo camera and the table
        if kind == "obj" and grp != 0:
            continue
        if rgba[3] == 0:
            if kind != "obj":
                continue
            rgba[3] = 1.0
        if kind == "world" and m.geom_xpos[g][2] < -0.6:
            continue
        rec = dict(type=t, size=m.geom_size[g].tolist(), pos=d.geom_xpos[g].tolist(), mat=d.geom_xmat[g].tolist(),
                   rgba=rgba.tolist(), kind=kind, body=bn)
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            rec["verts"] = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].tolist()
            rec["faces"] = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]].tolist()
        G.append(rec)
    ph = sc["photo"]
    J = dict(name=name, geoms=G, camera=ph["camera"], photo=ph["file"], W=ph["width"], H=ph["height"],
             table=sc["table"], title=sc.get("title"))
    (out / f"{name}.json").write_text(json.dumps(J))
    print(name, len(G), "geoms", flush=True)
