"""Batch: AI-generated table photos (photos/ai_gen/tables_v6) -> textured MuJoCo scenes + intent translation.

No known-size reference object is needed (aigen_photo_scene.py needs one):
  1. Muse Spark lists the objects, SAM 3.1 masks them (perceive_raw, Contributor mode: synthetic images).
  2. Tabletop outline read by eye (TOPS: corners, or the extreme points of a round top).  An automatic SAM
     "table" mask was tried first: it covers the legs and apron too, and the near edge could not be found reliably.
  3. Camera: focal length assumed (iPhone main camera, 26 mm equivalent); pitch/roll from the tabletop outline
     (rectangular top: corners must be right angles on the plane; round top: ellipse minor/major axis; oval top:
     pitch assumed).
  4. Scale: no reference object.  The camera height is set so that the objects' fitted sizes match Muse Spark's
     typical sizes for them (median of the mask-fit size factors -> 1).  Approximate by construction.
  5. Table texture: the photo is rectified onto the table plane, object pixels are filled from the surrounding
     surface, and the result is drawn on the sim tabletop (visual only, no physics change).
  6. The task sentence is compiled to goals by Muse Spark (the shipped compile step) - the "translation".

Run:  set -a; source .env; set +a; .venv/bin/python scripts/batch_tables.py [01 02 ...]
Writes scenes/t6_<NN>.json, photos/ai_gen/tables_v6/t6_<NN>.{table_tex.png,overlay.png} and
out/tables_v6/<NN>.json (intent).  All Meta calls are cached; reruns are free.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aigen_photo_scene import cam_tilt, plane_h, to_robot, _angles_cos  # noqa: E402
from rehearsal.perception import _nelder_mead, draw_overlay, estimate_scene, perceive_raw  # noqa: E402
from real_photo_scene import separate  # noqa: E402
from rehearsal.calib import apply_h  # noqa: E402

PH = ROOT / "photos" / "ai_gen" / "tables_v6"
OUT = ROOT / "out" / "tables_v6"
NEAR_X = 0.15

TASKS = {
    "01": "Put the loose pens into the pen cup.",
    "02": "Move the cereal bowl to the side of the table without knocking over the juice glass.",
    "03": "Put the socks into the laundry basket.",
    "04": "Stack the books neatly.",
    "05": "Move the phone next to the lamp without knocking over the glass of water.",
    "06": "Put the onions into the steel bowl.",
    "07": "Put both tea cups onto the tray.",
    "08": "Put the screwdrivers into the toolbox.",
    "09": "Put the eggs into the mixing bowl.",
    "10": "Put the plush toys into the basket.",
    "11": "Move the brush jar away from the sketchbook.",
    "12": "Clear the empty cans off the desk.",
    "13": "Put the thread spools into the tin box.",
    "14": "Line up the beakers next to the test-tube rack.",
    "15": "Put the tea glasses on the tray around the teapot.",
    "16": "Put the wrapped plates into the box.",
    "17": "Stack the empty rice bowls.",
    "18": "Put the small pots into the tray.",
    "19": "Bring the cup with the straw closer without spilling it.",
    "20": "Put the loose bike parts into the tray.",
    "21": "Take the apples out of the bag and put them in the bowl.",
    "22": "Put the food tins next to the cat bowl.",
    "23": "Move the mug away from the cables.",
    "24": "Put the crayons back in the box.",
    "25": "Put the cups next to the moka pot.",
    "26": "Move the mug off the puzzle.",
    "27": "Put the sponge and the cloth into the bucket.",
    "28": "Put the small bowls inside the big bowl.",
    "29": "Stack the small boxes.",
    "30": "Stand the fallen water bottle upright.",
    "31": "Stack the plates.",
    "32": "Gather the mugs together.",
    "33": "Put the glasses in a row.",
    "34": "Put nothing on the table: it should stay empty.",
    "35": "Put the tiny screws into the magnetic tray.",
    "36": "Put the paper filters into the filter holder.",
    "37": "Put the clay tools into the tool cup.",
    "38": "Put the wooden tokens into the box.",
    "39": "Put the brushes and the cloth into the caddy.",
    "40": "Put the test tubes into the tray.",
    "41": "Put the stones into the bowl.",
    "42": "Put the envelopes into the archive box.",
    "43": "Put the grooming brushes into the basket.",
    "44": "Put the yarn balls into the project bag.",
    "45": "Put the loose connectors into the compartment box.",
    "46": "Put the small dishes onto the tray.",
    "47": "Put the cubes into the case.",
    "48": "Put the envelopes into the letter tray.",
    "49": "Put the cups into the picnic basket.",
    "50": "Put the small bottles into the toiletry pouch.",
}

# Tabletop outline, read by eye off a 5% grid (percent of image width / height), like aigen_photo_scene.py's
# --table-px.  Rectangular tops: near-left, near-right, far-right, far-left corners of the top surface.
# ("R", left, right, far, near): round top (a circle: its outline sets the camera pitch).
# ("O", left, right, far, near): oval / kidney / triangular top (aspect unknown: pitch assumed 55 deg).
TOPS = {
    "01": [(-1, 84), (100, 84), (88, 29), (9, 30)], "02": [(6, 89), (94, 81), (92, 12), (15, 13)],
    "03": [(-2, 90), (100, 88), (94, 31), (12, 28)], "04": [(0, 92), (102, 91), (98, 37), (10, 38)],
    "05": ("R", 2, 94, 5, 91), "06": [(0, 80), (100, 80), (90, 22), (11, 22)],
    "07": [(5, 81), (94, 81), (70, 14), (17, 15)], "08": [(4, 73), (97, 73), (91, 26), (8, 26)],
    "09": ("O", 2, 92, 22, 82), "10": [(0, 77), (100, 76), (92, 31), (2, 32)],
    "11": [(4, 73), (95, 72), (88, 21), (8, 22)], "12": [(2, 58), (70, 51), (73, 22), (3, 24)],
    "13": [(-3, 76), (102, 72), (84, 18), (14, 22)], "14": [(1, 75), (99, 73), (94, 23), (12, 24)],
    "15": ("R", 9, 92, 9, 82), "16": [(1, 85), (100, 81), (96, 30), (5, 30)],
    "17": ("R", -2, 99, 2, 95), "18": [(-2, 84), (99, 84), (93, 22), (6, 22)],
    "19": [(1, 72), (96, 70), (92, 28), (7, 29)], "20": [(-1, 80), (100, 79), (91, 18), (8, 18)],
    "21": [(3, 84), (98, 74), (98, 27), (9, 26)], "22": ("O", 2, 98, 21, 84),
    "23": [(-1, 83), (101, 82), (88, 25), (9, 25)], "24": [(12, 73), (89, 72), (78, 17), (20, 17)],
    "25": [(12, 64), (88, 64), (82, 25), (18, 25)], "26": [(3, 78), (98, 84), (94, 8), (24, 9)],
    "27": [(2, 80), (95, 79), (85, 22), (16, 22)], "28": [(2, 77), (99, 74), (88, 14), (12, 15)],
    "29": [(0, 83), (99, 80), (87, 14), (8, 15)], "30": [(6, 73), (98, 73), (92, 16), (14, 16)],
    "31": [(-2, 92), (101, 90), (86, 27), (16, 26)], "32": [(0, 90), (100, 88), (93, 37), (11, 38)],
    "33": [(3, 63), (100, 60), (95, 34), (11, 36)], "34": [(2, 69), (95, 67), (84, 19), (11, 20)],
    "35": [(2, 91), (98, 90), (90, 14), (5, 14)], "36": [(3, 86), (96, 85), (88, 28), (10, 28)],
    "37": ("R", 5, 93, 2, 94), "38": [(6, 84), (95, 85), (87, 7), (13, 8)],
    "39": [(1, 67), (99, 67), (93, 30), (4, 30)], "40": [(2, 80), (97, 78), (82, 23), (16, 23)],
    "41": ("R", 11, 86, 20, 92), "42": ("O", 0, 99, 17, 91),
    "43": [(4, 81), (97, 78), (80, 19), (13, 20)], "44": ("O", 3, 96, 18, 84),
    "45": [(1, 85), (99, 84), (87, 26), (13, 26)], "46": [(0, 82), (99, 82), (95, 17), (12, 17)],
    "47": ("O", 11, 82, 13, 97), "48": [(2, 61), (99, 59), (94, 32), (7, 33)],
    "49": ("R", 4, 94, 14, 92), "50": [(0, 79), (100, 78), (96, 21), (5, 23)],
}


# ------------------------------------------------------------------------------------------------ tabletop

DIMS_SCHEMA = {"type": "object", "additionalProperties": False,
               "required": ["tabletop_width_cm", "tabletop_depth_cm", "camera_pitch_deg"],
               "properties": {"tabletop_width_cm": {"type": "number"}, "tabletop_depth_cm": {"type": "number"},
                              "camera_pitch_deg": {"type": "number"}}}
DIMS_PROMPT = ("Look at the table in this photo. Estimate the real size of its tabletop: width (left-right as seen "
               "in the photo) and depth (front-back), in cm, using the everyday objects on it for scale. Also estimate "
               "how steeply the camera looks down at the tabletop, in degrees below horizontal (0 = level, 90 = "
               "straight down). Answer with numbers only.")


def table_dims(nn):
    """Muse Spark's estimate of the tabletop's width/depth (cm) and camera pitch: AI-generated photos do not obey
    perspective well enough for right angles alone (near and far edges often converge too little), so the
    table's aspect ratio is a second cue for the calibration.  Cached per photo (synthetic data: Contributor)."""
    import json as _j
    from rehearsal.meta_client import chat_json
    from rehearsal.perception import data_url
    f = OUT / "table_dims.json"
    allv = _j.loads(f.read_text()) if f.exists() else {}
    if nn not in allv:
        photo = next(PH.glob(f"table_{nn}_*.png"))
        allv[nn] = chat_json([{"role": "user", "content": [{"type": "text", "text": DIMS_PROMPT},
                                                           {"type": "image_url", "image_url": {"url": data_url(photo)}}]}],
                             DIMS_SCHEMA, "TableDims", caller="batch_tables.table_dims", data="public", effort="low")
        OUT.mkdir(parents=True, exist_ok=True)
        f.write_text(_j.dumps(allv, indent=1))
    return allv[nn]


def _rect_aspect(Hp, q):
    p = apply_h(Hp, q)
    wid = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2
    dep = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2
    return wid / max(dep, 1e-9)

def calibrate_top(nn, w, h):
    """-> (f, pitch, roll, kind, outline, mask of the top in the image)."""
    from PIL import ImageDraw
    f = 26 / 43.27 * np.hypot(w, h)          # phone main camera, 26 mm full-frame equivalent
    t = TOPS[nn]
    M = Image.new("L", (w, h), 0)
    if isinstance(t, list):
        q = np.array([(x / 100 * w, y / 100 * h) for x, y in t])
        ImageDraw.Draw(M).polygon([tuple(p) for p in q], fill=255)

        # Roll is fixed at 0 (hand-held phone shots, level horizon).  Pitch and focal length are fitted from two
        # cues: right angles at the corners that are really in view, and the tabletop's width/depth ratio as
        # Muse Spark estimates it from the objects on it (AI-generated photos often have too little perspective
        # convergence for right angles alone, which drove the pitch to its bound on 18 of 50 photos).  Weak priors
        # keep the focal length near a phone's 26 mm equivalent and the pitch near Muse's estimate.
        lo, hi = np.radians(25), np.radians(80)
        vis = [i for i, (x, y) in enumerate(t) if 0.5 < x < 99.5 and 0.5 < y < 99.5]
        dims = table_dims(nn)
        asp = float(np.clip(dims["tabletop_width_cm"] / max(dims["tabletop_depth_cm"], 1), 0.3, 4.0))
        pit = np.radians(float(np.clip(dims["camera_pitch_deg"], 25, 80)))
        w_asp = 4.0 if len(vis) == 4 else 1.0          # a cut-off corner shortens one side

        def cost(x):
            ff = f * np.exp(x[1])
            if not (lo < x[0] < hi) or not (18 / 26 < ff / f < 40 / 26):
                return 1e3
            Hp = plane_h(cam_tilt(ff, x[0], 0.0, w, h))
            c = _angles_cos(Hp, q)
            c = [c[i] for i in vis] if len(vis) >= 2 else c
            return (sum(v * v for v in c) + w_asp * np.log(_rect_aspect(Hp, q) / asp) ** 2
                    + 0.5 * x[1] ** 2 + 0.3 * (x[0] - pit) ** 2)
        best = min((_nelder_mead(cost, np.array([p0, 0.0]), np.array([0.1, 0.1]), iters=400)
                    for p0 in (0.6, 0.9, 1.2, pit)), key=lambda r: r[1])
        return f * float(np.exp(best[0][1])), float(best[0][0]), 0.0, "rect", q, np.asarray(M) > 0
    kind, l, r, fa, ne = t
    ctr = np.array([(l + r) / 200 * w, (fa + ne) / 200 * h])
    major, minor = (r - l) / 100 * w, (ne - fa) / 100 * h
    ImageDraw.Draw(M).ellipse([l / 100 * w, fa / 100 * h, r / 100 * w, ne / 100 * h], fill=255)
    if kind == "R":
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        ray = np.linalg.solve(K, [ctr[0], ctr[1], 1.0])
        ray_el = np.arctan2(ray[1], np.hypot(ray[0], ray[2]))
        pitch = float(np.arcsin(np.clip(minor / major, 0, 1)) - ray_el)
    else:
        pitch = float(np.radians(55))
    return f, pitch, 0.0, "round" if kind == "R" else "oval", (ctr, major, minor, 0.0), np.asarray(M) > 0


def table_corners_px(cam, M, kind, q):
    """NL, NR, FR, FL pixel corners of the table rectangle used by the sim (round tops: bounding rectangle)."""
    H = plane_h(cam)
    if kind == "rect":
        P = q
    else:  # round / oval
        ctr, major, minor, ang = q
        t = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        P = ctr + np.c_[major / 2 * np.cos(t), minor / 2 * np.sin(t)] @ np.array([[np.cos(ang), np.sin(ang)],
                                                                               [-np.sin(ang), np.cos(ang)]])
    X = np.c_[P, np.ones(len(P))] @ H.T
    X = X[:, :2] / X[:, 2:]
    if kind == "rect":
        idx = np.argsort(X[:, 0])
        near, far = idx[:2], idx[2:]
        nl, nr = sorted(near, key=lambda i: -X[i, 1])
        fl, fr = sorted(far, key=lambda i: -X[i, 1])
        return q[[nl, nr, fr, fl]]
    x0, x1 = X[:, 0].min(), X[:, 0].max()
    y0, y1 = X[:, 1].min(), X[:, 1].max()
    C = np.array([[x0, y1, 0], [x0, y0, 0], [x1, y0, 0], [x1, y1, 0]])
    return cam.project(C)


CORD = ("cable", "cord", "leash", "rope", "string", "strap", "chain", "lanyard", "wire", "charger", "yarn strand",
        "hair tie", "rubber band", "necklace", "bracelet")
FLAT = ("towel", "cloth", "napkin", "rag", "leash", "paper", "envelope", "mat", "notebook", "sketchbook", "book",
        "map", "sheet", "glove", "fabric", "leaf", "nori", "tissue paper", "puzzle", "coaster", "sticky", "card",
        "photo", "print", "filter", "dish towel", "collar", "pad", "board", "palette", "folder", "magazine")


def flatten_from_masks(sc, raw, cam, w, h):
    """Flat / soft things (cloth, paper, books lying down, leashes...): their outline in the photo IS their footprint
    on the table, so the footprint comes from the SAM mask projected onto the table plane (minimum-area rectangle)
    instead of a typical size; thickness 0.5-3 cm.  Objects whose fitted height is already under 3.5 cm count too."""
    from rehearsal.calib import _hull
    from rehearsal.perception import _mask_full
    from rehearsal.scene import shape_bbox
    by_box = {tuple(int(v) for v in o["box"]): o for o in raw["objects"] if o.get("box")}
    for ob in sc["objects"]:
        name = (ob["category"] + " " + ob["label"]).lower()
        hgt = shape_bbox(ob["shape"])[2]
        soft = any(k in name for k in FLAT) and not any(k in name for k in ("box", "stack", "case", "holder", "bag",
                                                                             "cube", "roll", "cup", "jar", "bottle"))
        flatbox = ob["shape"]["type"] == "box" and hgt < 0.035      # phones, laptops, pens, keys
        cord = any(k in name for k in CORD) and "ball" not in name
        soft = soft or cord
        if not (soft or flatbox) or ob["shape"]["type"] in ("bowl",) or \
                ob["shape"].get("template") in ("open_box", "mug", "bottle"):
            continue
        r = by_box.get(tuple(int(v) for v in ob["perception"]["box_px"]))
        M = None if r is None else _mask_full(r, w, h, erode=1)
        if M is None:
            continue
        ys, xs = np.nonzero(M)
        if len(xs) > 4000:
            k = np.random.default_rng(0).choice(len(xs), 4000, replace=False)
            xs, ys = xs[k], ys[k]
        t = float(np.clip(min(hgt, 0.03) if soft else hgt, 0.005, 0.06))
        P = np.array([cam.intersect_z(u, v, t) for u, v in zip(xs, ys)])[:, :2]
        H = _hull(P)
        best = None
        for i in range(len(H)):                      # rotating edges: minimum-area enclosing rectangle
            e = H[(i + 1) % len(H)] - H[i]
            a = np.arctan2(e[1], e[0])
            R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
            Q = H @ R.T
            lo, hi = Q.min(0), Q.max(0)
            area = np.prod(hi - lo)
            if best is None or area < best[0]:
                best = (area, a, (lo + hi) / 2 @ R, hi - lo)
        _, a, c, (L, W) = best
        # a loop (cable, leash) covers a small part of its enclosing rectangle: keep the rectangle's direction but
        # shrink it to the area the object really covers (pixels -> table area via the plane homography)
        u0, v0 = float(np.mean(xs)), float(np.mean(ys))
        Q = np.array([cam.intersect_z(u, v, t)[:2] for u, v in ((u0, v0), (u0 + 1, v0), (u0, v0 + 1))])
        a1, a2 = Q[1] - Q[0], Q[2] - Q[0]
        cover = len(np.nonzero(M)[0]) * abs(a1[0] * a2[1] - a1[1] * a2[0])
        fill = cover / max(L * W, 1e-6)
        if cord and max(L, W) > 0.06:
            # a cable or leash lies as a loop: show it as a thin strip along the region's long side
            if W > L:
                a += np.pi / 2
            L, W, t = max(L, W), 0.012, 0.006
        elif fill < 0.35:
            k = np.sqrt(max(fill, 0.05))
            L, W = max(L * k, 0.02), max(W * k, 0.01)
        if L < 0.01 or W < 0.01 or max(L, W) > 1.2:
            continue
        ob["shape"] = {"type": "box", "size": [round(float(L), 4), round(float(W), 4), round(t, 4)]}
        ob["pos"] = [round(float(c[0]), 4), round(float(c[1]), 4)]
        ob["yaw"] = round(float(((a + np.pi / 2) % np.pi) - np.pi / 2), 4)
        ob["perception"].setdefault("adjusted", []).append("flat_footprint_from_mask")


# ------------------------------------------------------------------------------------------------ scene
def _floor_shape(sh, m=0.006):
    if "size" in sh:
        sh["size"] = [max(v, m) for v in sh["size"]]
    for k in ("radius", "height"):
        if k in sh:
            sh[k] = max(sh[k], m / 2 if k == "radius" else m)
    for p in sh.get("parts", []):
        _floor_shape(p, m)


def build(nn, raw, M, f, pitch, roll, kind, q, w, h, rel):
    height, info = 1.0, []
    for _ in range(3):
        cam0 = cam_tilt(f, pitch, roll, w, h, height)
        tpx = table_corners_px(cam0, M, kind, q)
        cam, table, tloc = to_robot(cam0, tpx, NEAR_X)
        sc = estimate_scene(raw, None, camera=cam, table=table, privacy="public", name=f"t6_{nn}",
                            photo_file=rel, clamp=False, fidelity=False)
        s = [o["perception"]["size_scale"] for o in sc["objects"]
             if o["perception"].get("placement") == "mask_fit" and 0.45 < o["perception"]["size_scale"] < 2.4]
        if len(s) < 2:
            s = [o["perception"]["size_scale"] for o in sc["objects"]]
        k = float(np.median(s)) if s else 1.0
        info.append({"cam_height_m": round(height, 3), "median_size_factor": round(k, 3), "n": len(s)})
        height = float(np.clip(height / k, 0.3, 2.0))
        if abs(k - 1) < 0.03:
            break
    cam0 = cam_tilt(f, pitch, roll, w, h, height)
    tpx = table_corners_px(cam0, M, kind, q)
    cam, table, tloc = to_robot(cam0, tpx, NEAR_X)
    sc = estimate_scene(raw, None, camera=cam, table=table, privacy="public", name=f"t6_{nn}",
                        photo_file=rel, clamp=False, fidelity=True)
    # Robot stand-off: a top-down suction tool cannot reach objects closer than ~0.28 m to its base.  Move the base
    # back (table edge further away) until the nearest object is >= 0.30 m away, but not so far that most objects
    # (80th percentile) leave the ~0.80 m working radius.
    xs = np.array([o["pos"][0] for o in sc["objects"]
                   if abs(o["pos"][1] - table["center"][1]) < table["size"][1] / 2 + 0.05])
    near_x = NEAR_X
    if len(xs):
        shift = max(0.0, 0.30 - float(xs.min()))
        shift = min(shift, max(0.0, 0.80 - float(np.percentile(xs, 80))))
        if shift > 0.005:
            near_x = NEAR_X + shift
            cam, table, tloc = to_robot(cam0, tpx, near_x)
            sc = estimate_scene(raw, None, camera=cam, table=table, privacy="public", name=f"t6_{nn}",
                                photo_file=rel, clamp=False, fidelity=True)
    info.append({"robot_standoff_m": round(near_x, 3)})
    flatten_from_masks(sc, raw, cam, w, h)
    separate(sc["objects"])
    (cx, cy), (sx, sy) = table["center"], table["size"]
    kept, dropped = [], []
    for o in sc["objects"]:
        x, y = o["pos"]
        bx = o["perception"]["box_px"]
        foot = (int(np.clip((bx[0] + bx[2]) / 2, 0, w - 1)), int(np.clip(bx[3] - 0.04 * (bx[3] - bx[1]), 0, h - 1)))
        on_top_in_photo = bool(M[foot[1], foot[0]])   # the object's base sits inside the tabletop outline
        off = abs(x - cx) > sx / 2 + 0.03 or abs(y - cy) > sy / 2 + 0.03
        if o["category"] in ("table", "chair") or (off and not on_top_in_photo):
            dropped.append(o["id"])
        else:
            # keep the whole footprint on the top (a photo object at the edge must not tip off when the sim settles)
            from rehearsal.scene import shape_bbox
            L, W = shape_bbox(o["shape"])[:2]
            c, si = abs(np.cos(o.get("yaw", 0.0))), abs(np.sin(o.get("yaw", 0.0)))
            hx, hy = min((L * c + W * si) / 2, sx / 2), min((L * si + W * c) / 2, sy / 2)
            nx_ = float(np.clip(x, cx - sx / 2 + hx, cx + sx / 2 - hx))
            ny_ = float(np.clip(y, cy - sy / 2 + hy, cy + sy / 2 - hy))
            if abs(nx_ - x) + abs(ny_ - y) > 1e-4:
                o["perception"].setdefault("adjusted", []).append("kept_on_table")
            o["pos"] = [round(nx_, 4), round(ny_, 4)]
            kept.append(o)
    for o in kept:                    # tiny parts (screws, beads): keep MuJoCo's minimum mass / inertia
        _floor_shape(o["shape"])
        o["mass_kg"]["nominal"] = max(o["mass_kg"]["nominal"], 0.005)
        o["mass_kg"]["range"] = [max(v, 0.003) for v in o["mass_kg"]["range"]]
    if len(kept) > 40:                 # the scene format allows 40 objects: keep the 40 largest footprints
        from rehearsal.scene import shape_bbox
        kept.sort(key=lambda o: -np.prod(shape_bbox(o["shape"])[:2]))
        dropped += [o["id"] for o in kept[40:]]
        kept = kept[:40]
    sc["objects"] = kept
    sc["perception_notes"]["dropped_off_table"] = dropped
    sc["perception_notes"]["dropped_note"] = "objects whose fitted centre is more than 3 cm off the table top"
    sc["perception_notes"]["auto_calibration"] = {
        "f_px": round(f, 1), "f_source": "assumed: phone main camera, 26 mm equivalent",
        "pitch_deg": round(np.degrees(pitch), 2), "roll_deg": round(np.degrees(roll), 2), "table_top": kind,
        "scale": "camera height set so fitted object sizes match Muse Spark's typical sizes (median)",
        "iterations": info, "cam_height_m": round(height, 3), "table_px": np.round(tpx, 1).tolist()}
    return sc, cam


# ------------------------------------------------------------------------------------------------ texture
def _blur(a, r):
    return np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)),
                      np.float32)


def _norm_fill(T, V, radii=(6, 16, 40, 120)):
    """Low-frequency fill of invalid texels by normalized convolution."""
    out, todo = T.copy(), ~V
    vf = V.astype(np.float32)
    for r in radii:
        wgt = _blur(vf * 255, r) / 255
        acc = np.stack([_blur(T[..., c] * vf, r) for c in range(3)], -1)
        ok = todo & (wgt > 0.1)
        out[ok] = acc[ok] / wgt[ok, None]
        todo &= ~ok
    if todo.any():
        out[todo] = T[V].mean(0) if V.any() else 128
    return out


def table_texture(photo, cam, table, M, raw, res=768):
    """Top view of the tabletop (row 0 = far edge, column 0 = left edge), objects removed.
    The photo is rectified onto the table plane; texels under objects (SAM masks, dilated for shadows) or outside
    the photo are filled with a clean patch of the same surface, mirror-tiled, carrying the local colour/light of
    the photo (low frequencies by normalized convolution)."""
    img = np.asarray(Image.open(photo).convert("RGB")).astype(np.float32)
    h, w = M.shape
    obj = np.zeros((h, w), bool)
    for o in raw["objects"]:
        if o.get("mask") is None:
            continue
        x1, y1 = o["box"][:2]
        m = o["mask"].astype(bool)
        y2, x2 = min(h, y1 + m.shape[0]), min(w, x1 + m.shape[1])
        obj[y1:y2, x1:x2] |= m[:y2 - y1, :x2 - x1]
    obj = np.asarray(Image.fromarray(obj.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(41))) > 0   # wide: shadows and mask edges too
    valid_px = M & ~obj
    (cx, cy), (sx, sy) = table["center"], table["size"]
    nx = res
    ny = max(8, int(round(res * sy / sx)))
    xs = cx + sx / 2 - (np.arange(nx) + 0.5) / nx * sx       # far -> near
    ys = cy + sy / 2 - (np.arange(ny) + 0.5) / ny * sy       # left -> right
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    uv = cam.project(np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)])
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    T = img[v, u].reshape(nx, ny, 3)
    V = (valid_px[v, u] & inside).reshape(nx, ny)
    # 1) smooth colour/light field: robust quadratic fit + large-radius residual (no sharp shadow edges carried)
    gi, gj = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), indexing="ij")
    B = np.stack([np.ones_like(gi), gi, gj, gi * gi, gj * gj, gi * gj], -1).reshape(-1, 6)

    def poly(Vm):
        keep = Vm.ravel().copy()
        out = np.zeros((nx * ny, 3))
        for _ in range(3):
            if keep.sum() < 50:
                break
            co, *_ = np.linalg.lstsq(B[keep], T.reshape(-1, 3)[keep], rcond=None)
            out = B @ co
            res = np.abs(T.reshape(-1, 3) - out).sum(1)
            keep = keep & (res < np.percentile(res[keep], 85) * 1.5)
        return out.reshape(nx, ny, 3)
    P = poly(V)
    # 2) leftover shadows / object fringes in the "clean" area: clearly darker (or brighter) than the field
    lum = lambda A: A @ np.array([0.3, 0.59, 0.11])
    ratio = _blur(lum(T), 9) / np.maximum(lum(P), 1)      # broad dark areas only; thin veins/grain stay
    bad = V & ((ratio < 0.74) | (ratio > 1.35))
    bad = np.asarray(Image.fromarray(bad.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(7))) > 0
    V = V & ~bad
    resid = _norm_fill(T - P, V, radii=(40, 120))
    low = P + resid
    # 3) fine surface detail from the cleanest window (lowest outlier energy), mirror-tiled
    D = T - np.stack([_blur(T[..., c], 12) for c in range(3)], -1)
    ii = np.pad(np.cumsum(np.cumsum(V.astype(np.int32), 0), 1), ((1, 0), (1, 0)))
    patch = None
    for k in range(min(nx, ny) // 3, 23, -8):
        full = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
        idx = np.argwhere(full == k * k)
        if len(idx):
            rng = np.random.default_rng(0)
            cand = idx[rng.choice(len(idx), min(200, len(idx)), replace=False)]
            score = [np.percentile(np.abs(D[i:i + k, j:j + k]).sum(-1), 99) for i, j in cand]
            i, j = cand[int(np.argmin(score))]
            patch = D[i:i + k, j:j + k]
            break
    if patch is None:
        fill = low
    else:
        tile = np.concatenate([patch, patch[::-1]], 0)
        tile = np.concatenate([tile, tile[:, ::-1]], 1)
        reps = (nx // tile.shape[0] + 1, ny // tile.shape[1] + 1, 1)
        Dt = np.tile(tile, reps)[:nx, :ny]
        lim = np.percentile(np.abs(patch), 80)          # keep grain, drop distinctive marks (they repeat visibly)
        fill = low + 0.7 * np.clip(Dt, -lim, lim)
    a = _blur(V.astype(np.float32) * 255, 2)[..., None] / 255   # feather the seam
    a = np.where(V[..., None], a, 0)
    tex = a * T + (1 - a) * fill
    mean = T[V].mean(0) / 255 if V.any() else np.array([150.0, 120, 95]) / 255
    return Image.fromarray(np.clip(tex, 0, 255).astype(np.uint8)), [round(float(c), 3) for c in mean] + [1.0]


# ------------------------------------------------------------------------------------------------ intent
def compile_task(scene, instruction):
    from rehearsal.agents import MuseAgent, scene_summary
    from rehearsal.executor import reachability_report
    from rehearsal import intent_gate
    summary = scene_summary(scene, reachability_report(scene))
    intent = MuseAgent(data="public").compile_intent(instruction, summary)
    try:
        gate = intent_gate.check(intent.get("goals") or [], scene, physics=False)
    except Exception as e:                      # the gate is a check on top; never block the batch on it
        gate = {"error": str(e)}
    return {"instruction": instruction, "intent": intent, "gate": gate}


def run(nn):
    photo = next(PH.glob(f"table_{nn}_*.png"))
    rel = str(photo.relative_to(ROOT))
    w, h = Image.open(photo).size
    raw = perceive_raw(photo, data="public")
    f, pitch, roll, kind, q, M = calibrate_top(nn, w, h)
    sc, cam = build(nn, raw, M, f, pitch, roll, kind, q, w, h, rel)
    tex, mean = table_texture(photo, cam, sc["table"], M, raw)
    tex_path = PH / f"t6_{nn}.table_tex.png"
    tex.save(tex_path)
    if os.environ.get("TEX_ONLY"):          # re-make the tabletop texture only; scene + intent files untouched
        print(nn, "texture", flush=True)
        return
    sc["title"] = photo.stem[9:].replace("-", " ").capitalize() + " (AI-generated test photo)"
    sc["provenance"] = "AI-generated test photo · not a real capture"
    sc["ai_generated"] = True
    sc["demo_instruction"] = TASKS[nn]
    sc["visual"] = {"table_texture": str(tex_path.relative_to(ROOT)), "table_rgba": mean,
                    "table_top": kind}
    sc["source"] = (f"AI-generated test photo {photo.name} (photos/ai_gen/tables_v6). Muse Spark listing + SAM 3.1 "
                    "masks (Contributor mode, synthetic image); tabletop outline read by eye; camera pitch/roll from the "
                    "tabletop outline with an assumed focal length; scale from Muse Spark's typical object sizes; "
                    "scripts/batch_tables.py. No hand-edited object positions.")
    # an empty table is not a valid Rehearsal scene (nothing to do): it is kept beside the renders only
    dest = ROOT / "scenes" if sc["objects"] else OUT
    OUT.mkdir(parents=True, exist_ok=True)
    (dest / f"t6_{nn}.json").write_text(json.dumps(sc, ensure_ascii=False, indent=1))
    draw_overlay(photo, sc).save(PH / f"t6_{nn}.overlay.png")
    res = compile_task(sc, TASKS[nn]) if sc["objects"] or nn == "34" else {"instruction": TASKS[nn], "intent": None}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{nn}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    a = sc["perception_notes"]["auto_calibration"]
    print(f"{nn} {kind:5s} pitch {a['pitch_deg']:5.1f} h {a['cam_height_m']:.2f} "
          f"table {sc['table']['size']} objs {len(sc['objects'])} dropped {sc['perception_notes']['dropped_off_table']} "
          f"status {(res.get('intent') or {}).get('status')}", flush=True)


if __name__ == "__main__":
    ids = sys.argv[1:] or sorted(TASKS)
    for nn in ids:
        try:
            run(nn)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{nn} FAILED: {e}", flush=True)
