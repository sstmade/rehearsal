"""t6_09 twin fix: the flour bag stands upright in the photo, but the twin has it lying on its side.

Root cause (rehearsal/perception.py refine_orientation): Muse Spark listed the bag as 0.16 x 0.12 x 0.28 m
(standing, 28 cm tall).  The silhouette refinement tried the box variants and switched to the variant labelled
"on_end" = size [H, W, L] = [0.28, 0.12, 0.16] -- a 16 cm high box lying on its side -- because its IoU with the
SAM mask was higher (0.869 vs 0.729).  That comparison is unfair: a pose change re-fits xy freely (it moved 7.2 cm),
while the standing pose keeps its xy.  The settle is not the problem (the lying box is simply what the scene says).

Fix (this scene only; scenes/t6_09.json is NOT modified): flour_bag_1 back to the standing box [0.1614, 0.1211, 0.2825]
(same dims, Muse order), with xy + yaw re-fitted to the SAM mask under the same free-xy rule the lying pose got.
Writes footage/tables_sim/hero/refined/09_eggs/twin/t6_09_fixed.json and a fit report.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rehearsal.calib import Camera  # noqa: E402
from rehearsal.perception import _nelder_mead, shape_point_sets  # noqa: E402

OUT = ROOT.parent / "footage/tables_sim/hero/refined/09_eggs/twin"
sc = json.loads((ROOT / "scenes/t6_09.json").read_text())
cam = Camera(**{k: sc["photo"]["camera"][k] for k in ("K", "R", "t")})
w, h = sc["photo"]["width"], sc["photo"]["height"]
bag = next(o for o in sc["objects"] if o["id"] == "flour_bag_1")
poly = [tuple(p) for p in bag["perception"]["mask_px"]]
im = Image.new("L", (w, h), 0)
ImageDraw.Draw(im).polygon(poly, fill=255)
M = np.asarray(im) > 127


def hull(uv):
    from scipy.spatial import ConvexHull
    return uv[ConvexHull(uv).vertices]


def sil(shape, x, y, yaw):
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    for pts in shape_point_sets(shape, x, y, yaw):
        d.polygon([tuple(p) for p in hull(cam.project(pts))], fill=255)
    return np.asarray(im) > 127


def iou(shape, x, y, yaw):
    P = sil(shape, x, y, yaw)
    return (P & M).sum() / max((P | M).sum(), 1)


lying = bag["shape"]
L, W_, H_ = lying["size"]                       # [0.2825, 0.1211, 0.1614]: the "on_end" variant [H, W, L]
x0, y0 = bag["pos"]
iou_lying = iou(lying, x0, y0, bag["yaw"])
# 1) Muse's own standing dims (0.1614 x 0.1211 x 0.2825), xy + yaw free
muse_standing = {"type": "box", "size": [H_, W_, L]}
best = None
for yw in np.linspace(0, np.pi, 12, endpoint=False):
    p, v = _nelder_mead(lambda p: 1 - iou(muse_standing, p[1], p[2], p[0]), np.array([yw, x0, y0]),
                        [0.15, 0.015, 0.015], iters=120)
    if best is None or v < best[1]:
        best = (p, v)
iou_muse_standing = 1 - best[1]
# 2) standing box, dims free within a paper flour sack's range (footprint 8-18 x 9-15 cm, 20-30 cm tall, height
#    >= 1.5x the footprint: it must stand, as the photo shows) -- unconstrained, the silhouette alone prefers lying


def pen(p):
    sx, sy, sz = abs(p[3]), abs(p[4]), abs(p[5])
    v = sum(max(0, lo - a) + max(0, a - hi) for a, lo, hi in ((sx, 0.08, 0.18), (sy, 0.09, 0.15), (sz, 0.20, 0.30)))
    return 20 * (v + max(0, 1.5 * max(sx, sy) - sz))


best = None
for yw in np.linspace(0, np.pi, 8, endpoint=False):
    loss = lambda p: 1 - iou({"type": "box", "size": [abs(p[3]), abs(p[4]), abs(p[5])]}, p[1], p[2], p[0]) + pen(p)
    p, v = _nelder_mead(loss, np.array([yw, x0 - 0.05, y0 - 0.03, 0.13, 0.10, 0.24]),
                        [0.15, 0.015, 0.015, 0.02, 0.02, 0.03], iters=300)
    if best is None or v < best[1]:
        best = (p, v)
p, v = best
sx, sy, sz = (round(abs(float(a)), 4) for a in p[3:6])
if sy > sx:
    sx, sy, p[0] = sy, sx, p[0] + np.pi / 2
standing = {"type": "box", "size": [sx, sy, sz]}
yaw = float(((p[0] + np.pi / 2) % np.pi) - np.pi / 2)
xs, ys = float(p[1]), float(p[2])
v = 1 - iou(standing, xs, ys, yaw)
rep = {"lying_shape": lying, "lying_pos": [x0, y0], "lying_yaw": bag["yaw"], "iou_lying": round(float(iou_lying), 3),
       "standing_shape": standing, "standing_pos": [round(xs, 4), round(ys, 4)], "standing_yaw": round(yaw, 4),
       "iou_standing_refit": round(1 - v, 3),
       "iou_muse_dims_standing_refit": round(float(iou_muse_standing), 3)}
print(json.dumps(rep, indent=1))
fixed = json.loads(json.dumps(sc))
fb = next(o for o in fixed["objects"] if o["id"] == "flour_bag_1")
fb["shape"] = standing
fb["pos"] = [round(xs, 4), round(ys, 4)]
fb["yaw"] = round(yaw, 4)
fb["perception"]["fidelity"]["pose"] = "standing"
fb["perception"]["fidelity"]["refined_fix"] = ("standing restored (refined/09 fix): the 'on_end' switch laid the "
                                               "28 cm bag on its side; xy+yaw re-fitted to the SAM mask")
fb["perception"].setdefault("adjusted", []).append("upright_fix_refined09")
fixed["source"] += " | refined/09: flour_bag_1 restored to standing (photo shows it upright), xy+yaw re-fitted."
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "t6_09_fixed.json").write_text(json.dumps(fixed, indent=1))
(OUT / "flour_fit.json").write_text(json.dumps(rep, indent=1))
# overlay: mask (green), lying silhouette (red), standing (blue)
ph = Image.open(ROOT / sc["photo"]["file"]).convert("RGB")
ov = np.asarray(ph).astype(float)
for m_, c in ((M, (0, 255, 0)), (sil(lying, x0, y0, bag["yaw"]), (255, 0, 0)), (sil(standing, xs, ys, yaw), (0, 120, 255))):
    from scipy.ndimage import binary_erosion
    e = m_ & ~binary_erosion(m_, iterations=3)
    ov[e] = c
Image.fromarray(ov.astype(np.uint8)).crop((100, 200, 700, 850)).save(OUT / "flour_fit_overlay.png")
