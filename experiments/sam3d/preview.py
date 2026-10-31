"""Render a GLB to a PNG preview (3 views) with matplotlib; print bbox and convex hull stats."""
import sys, json
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

glb, out = sys.argv[1], sys.argv[2]
scene = trimesh.load(glb)
mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
try:
    cv = mesh.visual if mesh.visual.kind == "vertex" else mesh.visual.to_color()
    vc = cv.vertex_colors[:, :3] / 255.0
    fc = vc[mesh.faces].mean(1)
except Exception as e:
    print("no colors:", e); fc = np.full((len(mesh.faces), 3), 0.7)
v = mesh.vertices
ext = v.max(0) - v.min(0)
hull = mesh.convex_hull
print(json.dumps({"verts": len(v), "faces": len(mesh.faces), "bbox_extent": ext.round(4).tolist(),
                  "watertight": bool(mesh.is_watertight), "hull_faces": len(hull.faces),
                  "texture": type(mesh.visual).__name__}))
# light shading
n = mesh.face_normals; shade = 0.55 + 0.45 * np.clip(n @ np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8]), 0, 1)
cols = np.clip(fc * shade[:, None], 0, 1)
fig = plt.figure(figsize=(12, 4))
c = (v.max(0) + v.min(0)) / 2; r = ext.max() / 2
for i, (el, az) in enumerate([(20, -60), (20, 30), (80, -90)]):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.add_collection3d(Poly3DCollection(v[mesh.faces], facecolors=cols, edgecolors="none"))
    for f, cc in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), c): f(cc - r, cc + r)
    ax.view_init(el, az); ax.set_axis_off()
plt.tight_layout(); plt.savefig(out, dpi=90)
