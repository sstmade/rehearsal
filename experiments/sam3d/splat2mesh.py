"""SAM 3D Gaussian PLY -> vertex-coloured mesh GLB (marching cubes on a splatted occupancy grid).

Workaround for mlx-spatial's FlexiCubes mesh export needing ~44 GB on a 36 GB Mac.
usage: splat2mesh.py gaussians.ply out.glb [grid=128] [min_opacity_logit=0]
"""
import sys, time, json
import numpy as np, trimesh
from plyfile import PlyData
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

t0 = time.time()
src, dst = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 128
thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
e = PlyData.read(src)["vertex"]
xyz = np.stack([e["x"], e["y"], e["z"]], 1).astype(np.float32)
keep = e["opacity"] > thr                       # opacity is a logit; >0 means alpha > 0.5
xyz = xyz[keep]
C0 = 0.28209479177387814                        # SH degree-0 constant
rgb = np.clip(0.5 + C0 * np.stack([e["f_dc_0"], e["f_dc_1"], e["f_dc_2"]], 1)[keep], 0, 1)
lo, hi = xyz.min(0), xyz.max(0)
pad = (hi - lo).max() * 0.03
lo, hi = lo - pad, hi + pad
step = (hi - lo).max() / (N - 1)
idx = np.clip(((xyz - lo) / step).round().astype(int), 0, N - 1)
grid = np.zeros((N, N, N), np.float32)
np.add.at(grid, tuple(idx.T), 1.0)
grid = gaussian_filter(grid, 1.0)
level = 0.5 * np.percentile(grid[grid > 0], 50)
v, f, n, _ = marching_cubes(grid, level=level)
v = v * step + lo
m = trimesh.Trimesh(v, f[:, ::-1], process=True)
m = max(m.split(only_watertight=False), key=lambda c: len(c.faces))   # largest component
_, nn = cKDTree(xyz).query(m.vertices, k=8)
col = rgb[nn].mean(1)
m.visual.vertex_colors = np.c_[(col * 255).astype(np.uint8), np.full(len(col), 255, np.uint8)]
m.export(dst)
print(json.dumps({"gaussians_kept": int(keep.sum()), "grid": N, "verts": len(m.vertices), "faces": len(m.faces),
                  "watertight": bool(m.is_watertight), "extent": (m.bounds[1] - m.bounds[0]).round(3).tolist(),
                  "seconds": round(time.time() - t0, 1)}))
