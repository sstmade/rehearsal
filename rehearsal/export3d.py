"""Export a compiled MuJoCo model's visual geometry for the Three.js replay viewer.

scene3d.json lists bodies and visual geoms (shape, local pose, color); meshes.bin holds each mesh
as non-indexed float32 triangle positions followed by per-vertex normals. Trajectories recorded with Sim.start_recording("full") then
give per-frame world poses for bodies 1..nbody-1, in the same order.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

G = mujoco.mjtGeom
TYPES = {int(G.mjGEOM_PLANE): "plane", int(G.mjGEOM_SPHERE): "sphere",
         int(G.mjGEOM_CAPSULE): "capsule", int(G.mjGEOM_ELLIPSOID): "ellipsoid",
         int(G.mjGEOM_CYLINDER): "cylinder", int(G.mjGEOM_BOX): "box", int(G.mjGEOM_MESH): "mesh"}
VISUAL_GROUPS = (0, 1, 2)  # menagerie puts collision geoms in group 3


def export_scene3d(m: mujoco.MjModel, out: Path):
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
    out.mkdir(parents=True, exist_ok=True)
    (out / "meshes.bin").write_bytes(np.concatenate(chunks).tobytes() if chunks else b"")
    (out / "scene3d.json").write_text(json.dumps({"bodies": bodies, "geoms": geoms}))
