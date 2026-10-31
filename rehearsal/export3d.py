"""Export a compiled MuJoCo model's visual geometry for the Three.js replay viewer.

scene3d.json lists bodies and visual geoms (shape, local pose, color); meshes.bin holds each mesh
as non-indexed float32 triangle positions followed by per-vertex normals. Trajectories recorded with Sim.start_recording("full") then
give per-frame world poses for bodies 1..nbody-1, in the same order.

Optional look-alike meshes (visual only, physics unchanged): when assets/sam3d/<scene>/<obj>.glb exists (made by
scripts/sam3d_assets.py with Meta SAM 3D Objects), the object's body gets
    "visual_mesh": {"url": "/assets/sam3d/<scene>/<obj>.glb", "scale": s, "pos": [x, y, z], "quat": [w, x, y, z]}
a transform from the GLB frame (already the twin object frame, bbox centred, longest side 1) to the body frame
(origin = centre of the primitive's bounding box). Scale is uniform (keeps the mesh's own proportions): the geometric
mean of the size ratios on the primitive's two largest axes; the mesh then sits on the primitive's bottom face.

Standalone re-export (same model, so recorded trajectories stay valid):
    python -m rehearsal.export3d scenes/<scene>.json out/<scene>
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets" / "sam3d"

G = mujoco.mjtGeom
TYPES = {int(G.mjGEOM_PLANE): "plane", int(G.mjGEOM_SPHERE): "sphere",
         int(G.mjGEOM_CAPSULE): "capsule", int(G.mjGEOM_ELLIPSOID): "ellipsoid",
         int(G.mjGEOM_CYLINDER): "cylinder", int(G.mjGEOM_BOX): "box", int(G.mjGEOM_MESH): "mesh"}
VISUAL_GROUPS = (0, 1, 2)  # menagerie puts collision geoms in group 3


def uniform_scale(bbox, ext) -> float:
    """Geometric mean of bbox/ext on the bbox's two largest axes (same formula as scripts/sam3d_assets.py)."""
    k = np.argsort(bbox)[-2:]
    return float(np.exp(np.mean(np.log(np.asarray(bbox, float)[k] / np.maximum(np.asarray(ext, float)[k], 1e-6)))))


def visual_meshes(scene: dict | None, assets: Path = ASSETS) -> dict:
    """{object id: visual_mesh transform} for the objects of `scene` that have a SAM 3D GLB + sidecar JSON."""
    if not scene or not scene.get("name"):
        return {}
    from .scene import shape_bbox
    out, d = {}, assets / scene["name"]
    for o in scene.get("objects", []):
        glb, side = d / f"{o['id']}.glb", d / f"{o['id']}.json"
        if not (glb.exists() and side.exists()):
            continue
        meta = json.loads(side.read_text())
        if meta.get("status") != "ok" or meta.get("use") is False:
            continue
        bb, ext = shape_bbox(o["shape"]), meta["extent"]
        s = uniform_scale(bb, ext)
        out[o["id"]] = {"url": f"/assets/sam3d/{scene['name']}/{o['id']}.glb?v={int(glb.stat().st_mtime)}",
                        "scale": round(s, 5), "pos": [0.0, 0.0, round(-bb[2] / 2 + s * ext[2] / 2, 5)],
                        "quat": [1.0, 0.0, 0.0, 0.0], "source": "SAM 3D Objects"}
    return out


def export_scene3d(m: mujoco.MjModel, out: Path, scene: dict | None = None):
    chunks, mesh_ref, offset = [], {}, 0
    geoms = []
    for g in range(m.ngeom):
        if m.geom_group[g] not in VISUAL_GROUPS:
            continue
        t = TYPES.get(int(m.geom_type[g]))
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        if t is None or rgba[3] == 0:
            continue
        entry = {"body": int(m.geom_bodyid[g]), "type": t,
                 "size": [round(float(v), 5) for v in m.geom_size[g]],
                 "pos": [round(float(v), 5) for v in m.geom_pos[g]],
                 "quat": [round(float(v), 6) for v in m.geom_quat[g]],
                 "rgba": [round(float(v), 3) for v in rgba]}
        if t == "mesh":
            mid = int(m.geom_dataid[g])
            if mid not in mesh_ref:
                va, fa = m.mesh_vertadr[mid], m.mesh_faceadr[mid]
                verts = m.mesh_vert[va:va + m.mesh_vertnum[mid]]
                faces = m.mesh_face[fa:fa + m.mesh_facenum[mid]]
                tri = verts[faces].astype(np.float32).ravel()
                na = m.mesh_normaladr[mid]
                normals = m.mesh_normal[na:na + m.mesh_normalnum[mid]]
                nrm = normals[m.mesh_facenormal[fa:fa + m.mesh_facenum[mid]]].astype(np.float32).ravel()
                mesh_ref[mid] = [offset, int(tri.size)]  # positions, then the same count of normals
                chunks += [tri, nrm]
                offset += 2 * tri.size
            entry["mesh"] = mesh_ref[mid]
        geoms.append(entry)

    bodies = [{"id": b, "name": m.body(b).name, "root": int(m.body_rootid[b])} for b in range(m.nbody)]
    vis = visual_meshes(scene)
    for b in bodies:
        if b["name"] in vis:
            b["visual_mesh"] = vis[b["name"]]
    out.mkdir(parents=True, exist_ok=True)
    (out / "meshes.bin").write_bytes(np.concatenate(chunks).tobytes() if chunks else b"")
    (out / "scene3d.json").write_text(json.dumps({"bodies": bodies, "geoms": geoms}))


if __name__ == "__main__":
    import sys
    from .scene import build_model, load_scene
    sc = load_scene(Path(sys.argv[1]))
    export_scene3d(build_model(sc), Path(sys.argv[2]), sc)
    print(f"wrote {sys.argv[2]}/scene3d.json ({len(visual_meshes(sc))} visual meshes)")
