"""Real photo -> scene, end to end, for a YCB-Video frame (MIT) with a known camera and 6-DoF ground truth.

    .venv/bin/python scripts/real_photo_scene.py datasets/tierA_ycbv/ycbv_test_000059_001152.json --name real_ycbv59

1. copies the RGB frame to photos/real/<name>.png
2. runs Muse Spark (object listing) + SAM 3.1 (masks) on it via perception.perceive_raw (data="public")
3. places every object with perception.estimate_scene(camera=<dataset K + table pose>), method "guarded"
4. writes scenes/<name>.json, photos/real/<name>.overlay.png and a per-object error table vs ground truth.

Robot frame: the robot stands on the camera's side of the table, facing away from the camera
(robot +x = dataset table +y, robot +y = dataset table -x), shifted by --offset-x so the objects are in reach.
No position is hand-edited; the only post-processing is perception.tidy (documented: clamp to table, de-overlap).
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rehearsal.calib import Camera  # noqa: E402
from rehearsal.perception import draw_overlay, estimate_scene, perceive_raw  # noqa: E402
from rehearsal.scene import footprint_polygon  # noqa: E402

def _poly(o):
    return footprint_polygon(o, n=16)


def _mtv(P, Q):
    """Minimum translation (2D SAT) that moves Q off P, or None if the convex polygons do not overlap."""
    best = None
    for poly in (P, Q):
        e = np.roll(poly, -1, 0) - poly
        for n in np.c_[-e[:, 1], e[:, 0]]:
            n = n / (np.linalg.norm(n) + 1e-12)
            a, b = P @ n, Q @ n
            ov = min(a.max(), b.max()) - max(a.min(), b.min())
            if ov <= 0:
                return None
            if best is None or ov < best[0]:
                best = (ov, n if (Q.mean(0) - P.mean(0)) @ n >= 0 else -n)
    return best


def separate(objects, gap=0.003, iters=100):
    """Resolve interpenetrating footprints with the exact rotated-rectangle / circle shapes (SAT), splitting each
    push equally.  perception.tidy uses circumscribed circles, which over-pushes elongated boxes standing
    side by side (8-10 cm on this photo); this keeps the same idea with exact footprints."""
    for _ in range(iters):
        moved = False
        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                m = _mtv(_poly(a), _poly(b))
                if m is None:
                    continue
                d = (m[0] + gap) / 2 * m[1]
                a["pos"] = [float(a["pos"][0] - d[0]), float(a["pos"][1] - d[1])]
                b["pos"] = [float(b["pos"][0] + d[0]), float(b["pos"][1] + d[1])]
                for o in (a, b):
                    adj = o["perception"].setdefault("adjusted", [])
                    if "overlap_sat" not in adj:
                        adj.append("overlap_sat")
                moved = True
        if not moved:
            break
    for o in objects:
        o["pos"] = [round(o["pos"][0], 4), round(o["pos"][1], 4)]


A = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # dataset table frame -> robot frame rotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--offset-x", type=float, default=0.62)
    ap.add_argument("--table", default="0.55,0.0,0.90,0.80", help="cx,cy,sx,sy in the robot frame")
    ap.add_argument("--title", default="")
    ap.add_argument("--instruction", default="")
    a = ap.parse_args()
    t0 = time.time()
    gt = json.loads(a.gt.read_text())
    b = np.array([a.offset_x, 0.0, 0.0])
    T = np.asarray(gt["table_plane"]["T_table_from_cam"], float)
    R_t = T[:3, :3].T                       # table -> cam
    t_t = -R_t @ T[:3, 3]
    R_r = R_t @ A.T                         # robot -> cam
    t_r = t_t - R_r @ b
    cam = Camera(gt["camera"]["K"], R_r, t_r)

    dst_dir = ROOT / "photos" / "real"
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_png = a.gt.with_suffix(".png")
    photo = dst_dir / f"{a.name}.png"
    shutil.copyfile(src_png, photo)
    rel = str(photo.relative_to(ROOT))

    raw = perceive_raw(photo, data="public")
    cx, cy, sx, sy = (float(v) for v in a.table.split(","))
    table = {"center": [cx, cy], "size": [sx, sy]}
    scene = estimate_scene(raw, None, method="guarded", camera=cam, table=table, privacy="public",
                           name=a.name, photo_file=rel, clamp=False)
    raw_pos = {o["id"]: list(o["pos"]) for o in scene["objects"]}
    separate(scene["objects"])
    scene["source"] = (f"real photo {src_png.name} (YCB-Video BOP'19 test, MIT) -> Muse Spark listing + SAM 3.1 masks "
                       f"-> guarded fit with the dataset's known camera K and table pose; robot frame = dataset table "
                       f"frame rotated so the robot faces away from the camera, shifted {a.offset_x} m. "
                       f"De-overlap: exact-footprint SAT (scripts/real_photo_scene.py), not perception.tidy. No hand-edited positions.")
    if a.title:
        scene["title"] = a.title
    if a.instruction:
        scene["demo_instruction"] = a.instruction
    scene["photo"]["camera_info"]["dataset"] = "YCB-Video (BOP'19 test), MIT"
    scene["photo"]["camera_info"]["robot_from_dataset_table"] = {"R": A.tolist(), "t": b.tolist()}

    # ground truth in the robot frame (bbox-centre footprint), greedy nearest association
    gts = []
    for o in gt["objects"]:
        p = A @ np.asarray(o["center_table_m"]) + b
        gts.append({"id": o["name"], "xy": p[:2], "on_table": o.get("on_table")})
    rows, used = [], set()
    pairs = sorted(((float(np.linalg.norm(np.subtract(e["pos"], g["xy"]))), i, j)
                    for i, e in enumerate(scene["objects"]) for j, g in enumerate(gts)))
    em, gm = {}, {}
    for d, i, j in pairs:
        if i in em or j in gm or d > 0.15:
            continue
        em[i], gm[j] = j, d
    for j, g in enumerate(gts):
        i = next((i for i, jj in em.items() if jj == j), None)
        e = scene["objects"][i] if i is not None else None
        rows.append({"gt": g["id"], "gt_xy": np.round(g["xy"], 4).tolist(), "est": e and e["id"],
                     "est_xy": e and e["pos"], "err_cm": None if e is None else round(gm[j] * 100, 2),
                     "err_before_deoverlap_cm": None if e is None else
                     round(float(np.linalg.norm(np.subtract(raw_pos[e["id"]], g["xy"]))) * 100, 2),
                     "source": e and e["perception"]["source"], "sigma_m": e and e["perception"]["sigma_m"],
                     "adjusted": e and e["perception"].get("adjusted", [])})
    extras = [o["id"] for i, o in enumerate(scene["objects"]) if i not in em]
    scene["perception_notes"]["ground_truth_eval"] = {"rows": rows, "extras_not_in_gt": extras,
                                                      "note": "bbox-centre footprint vs estimated pos, robot frame"}

    (ROOT / "scenes" / f"{a.name}.json").write_text(json.dumps(scene, ensure_ascii=False, indent=1))
    draw_overlay(photo, scene).save(dst_dir / f"{a.name}.overlay.png")
    for r in rows:
        print(r)
    print("extras:", extras)
    print("not segmented:", scene["perception_notes"]["not_segmented"])
    print(f"wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
