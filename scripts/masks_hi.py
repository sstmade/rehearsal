"""High-res per-object SAM 3.1 masks for the 50 AI-generated tables (data only; no scene or render code is touched).

For every object in scenes/t6_NN.json (out/tables_v6/t6_NN.json for 34) this re-segments the full-resolution photo
with SAM 3.1 box prompts seeded from the object's own perception.box_px, so object ids stay the same:

  pass A  full photo, all boxes of the table in one request (<=16 per request)              -> native-res mask
  pass B  per object: crop around the box (+pad), upscaled so SAM sees the object at <=4x   -> sub-pixel mask

The outline is traced at sub-pixel precision (contourpy on the lightly smoothed mask) and simplified with
Douglas-Peucker at 0.5 px (full-res units), so thin handles, cloth hems and folds keep their shape.
Holes (mug handle loops, scissor rings) are written separately as holes_px.

Choice per object: B if it agrees with A and with the old outline, else A, else the old mask_px is kept
(status says which).  Writes out/tables_v6/masks_hi/t6_NN.json, t6_NN_overlay.jpg (old red, new green) and
review_contact.jpg.  SAM calls go through rehearsal.meta_client (cached, ledgered); synthetic photos only.

    .venv/bin/python scripts/masks_hi.py 17 13 03        # pilot
    .venv/bin/python scripts/masks_hi.py                 # all
    .venv/bin/python scripts/masks_hi.py --contact       # contact sheet only
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import contourpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rehearsal.meta_client import sam  # noqa: E402

OUT = ROOT / "out" / "tables_v6" / "masks_hi"
PHOTO_DIR = ROOT / "photos" / "ai_gen" / "tables_v6"
SAM_META = {"include_confidence": "true"}
CROP_TARGET = 1024       # upscaled crop long side (px)
MAX_UP = 4.0
EPS_PX = 0.5             # Douglas-Peucker tolerance in full-res pixels
MAX_PTS = 800
# open containers seen from above: SAM often leaves the contents (cereal, soup, food) out of the vessel, and a spoon
# across the rim opens that gap to the outside; for these the outline is the convex hull of the SAM mask
CONTAINER = re.compile(r"^(?!moka|wine)(?:.*_)?(?:bowl|plate|dish|platter|saucer|tray|glass|tumbler|jar|pot|pan|wok|ramekin)_\d+$")
# manual review (overlays read by eye): objects whose new outline is worse than the old one -> keep the old mask_px
FORCE_OLD: dict[str, set[str]] = {"06": {"spice_box_1"}, "17": {"rice_bowl_4"}, "49": {"banana_bunch_1"}}
# ... and the reverse: old outline is wrong/inflated (chords across a cable, zigzag), the SAM outline is right
FORCE_NEW: dict[str, set[str]] = {"01": {"charging_cable_1"}, "40": {"fish_net_1"}, "46": {"key_ring_1"}}


REC_RE = re.compile(r"(\d+)<\|box;([^|]*)\|>(?:<\|mask;([^|]*?)data=(\d+),(\d+),(.)([^|]*)\|>)?")


# ----------------------------------------------------------------------------------------------- SAM
def _decode(H, W, enc, payload):
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    m = SegmentationMask(encoding="lossless" if enc == "~" else "one_bit", payload=enc + payload, width=W, height=H)
    return np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(H, W).copy()


def sam_call(img_url: str, prompt: str) -> dict[int, dict]:
    text = sam(prompt, img_url, caller="masks_hi.sam", metadata=SAM_META)
    out = {}
    for m in REC_RE.finditer(text):
        oid, box, _, H, W, enc, payload = m.groups()
        kv = dict(t.split("=", 1) for t in box.split(";") if "=" in t)
        b = tuple(int(float(kv[k])) for k in ("x1", "y1", "x2", "y2"))
        try:
            mask = _decode(int(H), int(W), enc, payload) if H else None
        except Exception:
            mask = None
        out[int(oid)] = {"box": b, "conf": float(kv["c"]) if "c" in kv else None, "mask": mask}
    return out


def png_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def box_prompt(items, w, h):
    return "<0f>" + ",".join(f"{i}<|box;x1={b[0]};y1={b[1]};x2={b[2]};y2={b[3]};w={w};h={h}|>" for i, b in items)


def paste_full(rec, w, h) -> np.ndarray | None:
    mk = rec.get("mask") if rec else None
    if mk is None or not mk.any():
        return None
    full = np.zeros((h, w), bool)
    x1, y1 = rec["box"][0], rec["box"][1]
    y2, x2 = min(h, y1 + mk.shape[0]), min(w, x1 + mk.shape[1])
    full[y1:y2, x1:x2] = mk[:y2 - y1, :x2 - x1] > 0
    return full


# ------------------------------------------------------------------------------------------ geometry
def _blur(a: np.ndarray, n: int = 1) -> np.ndarray:
    a = a.astype(np.float32)
    for _ in range(n):
        p = np.pad(a, 1, mode="edge")
        a = sum(p[1 + dy:1 + dy + a.shape[0], 1 + dx:1 + dx + a.shape[1]] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9
    return a


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _dp(pts, eps):
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    L = np.linalg.norm(ab)
    if L < 1e-9:
        dist = np.linalg.norm(pts - a, axis=1)
    else:
        dist = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / L
    k = int(np.argmax(dist))
    if dist[k] <= eps:
        return np.array([a, b])
    return np.r_[_dp(pts[:k + 1], eps)[:-1], _dp(pts[k:], eps)]


def simplify(poly: np.ndarray, eps: float) -> np.ndarray:
    if len(poly) < 8:
        return poly
    i = 0
    j = int(np.argmax(np.linalg.norm(poly - poly[0], axis=1)))
    out = poly
    while True:
        out = np.r_[_dp(poly[i:j + 1], eps)[:-1], _dp(np.r_[poly[j:], poly[:i + 1]], eps)[:-1]]
        if len(out) <= MAX_PTS:
            return out
        eps *= 1.4


def trace(mask: np.ndarray, to_full, blur: int, eps: float):
    """mask (bool, local raster) -> (outer polygon, [holes]) in full-res pixel-index coords via to_full(xy)."""
    z = np.pad(_blur(mask, blur) if blur else mask.astype(np.float32), 1)
    lines = contourpy.contour_generator(z=z, line_type="Separate").lines(0.5)
    loops = [np.asarray(l, float) - 1.0 for l in lines if len(l) >= 4]
    if not loops:
        return None, []
    areas = [_area(l) for l in loops]
    k = int(np.argmax(np.abs(areas)))
    outer = loops[k]
    sgn = np.sign(areas[k])
    from matplotlib.path import Path as MPath
    op = MPath(outer)
    holes = [l for l, a in zip(loops, areas) if np.sign(a) != sgn and abs(a) > max(30.0, 0.004 * abs(areas[k]))
             and op.contains_point(l[0])]
    f = lambda l: simplify(to_full(l), eps)
    return f(outer), [f(hh) for hh in holes]


def convex_hull(poly):
    P = sorted(map(tuple, np.asarray(poly, float)))
    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h
    lo, up = half(P), half(P[::-1])
    return np.asarray(lo[:-1] + up[:-1], float)


def raster(poly, w, h, holes=()) -> np.ndarray:
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    if poly is not None and len(poly) >= 3:
        d.polygon([tuple(map(float, q)) for q in poly], fill=255)
    for hh in holes:
        if len(hh) >= 3:
            d.polygon([tuple(map(float, q)) for q in hh], fill=0)
    return np.asarray(im) > 0


def see_through(arr, poly, holes, w, h):
    """Keep only holes that show the background (mug-handle loop, gap between chopsticks): the hole's median colour
    must match (chromaticity, brightness 0.4-1.3x for shadow) the median of a 6 px ring just outside the outer outline.  Holes where SAM left out an object's
    contents (food in a bowl, a print on a shirt) are dropped, so the decal keeps those pixels."""
    from PIL import ImageFilter
    O = Image.fromarray((raster(poly, w, h) * 255).astype(np.uint8))
    ring = (np.asarray(O.filter(ImageFilter.MaxFilter(13))) > 0) & ~(np.asarray(O) > 0)
    if ring.sum() < 20:
        return []
    bg = np.median(arr[ring], 0)
    keep = []
    for hh in holes:
        H = raster(hh, w, h)
        if H.sum() < 6:
            continue
        hc = np.median(arr[H], 0)
        chroma = np.abs(hc / max(hc.sum(), 1) - bg / max(bg.sum(), 1)).sum()
        ratio = hc.sum() / max(bg.sum(), 1)
        small = H.sum() < 0.12 * max(int((np.asarray(O) > 0).sum()), 1)   # handle loop / finger ring
        if chroma < (0.1 if small else 0.05) and 0.4 < ratio < 1.3:      # same surface, maybe in the object's shadow
            keep.append(hh)
    return keep


def bbox_iou(poly, box):
    if poly is None or box is None or len(poly) < 3:
        return 0.0
    p = np.asarray(poly, float)
    a = (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
    b = (box[0] - 0.5, box[1] - 0.5, box[2] + 0.5, box[3] + 0.5)
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def iou(a, b):
    if a is None or b is None:
        return 0.0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


# ---------------------------------------------------------------------------------------------- table
def load_scene(nn: str) -> dict:
    for p in (ROOT / "scenes" / f"t6_{nn}.json", ROOT / "out" / "tables_v6" / f"t6_{nn}.json"):
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(nn)


def run_table(nn: str, pool) -> dict:
    sc = load_scene(nn)
    photo = ROOT / sc["photo"]["file"]
    assert "ai_gen/tables_v6" in str(photo), photo        # synthetic photos only
    im = Image.open(photo).convert("RGB")
    w, h = im.size
    assert (w, h) == (sc["photo"]["width"], sc["photo"]["height"]), "frame mismatch"
    objs = []
    for o in sc.get("objects", []):
        pr = o.get("perception") or {}
        box = pr.get("box_px")
        old = pr.get("mask_px")
        if box is None and old:
            a = np.asarray(old)
            box = [int(a[:, 0].min()), int(a[:, 1].min()), int(np.ceil(a[:, 0].max())), int(np.ceil(a[:, 1].max()))]
        objs.append({"id": o["id"], "label": o.get("label", ""), "box": box, "old": old})
    full_url = png_url(im)
    arr = np.asarray(im).astype(np.float32)

    # pass A: full frame, all boxes
    items = [(k, o["box"]) for k, o in enumerate(objs, 1) if o["box"]]
    chunks = [items[i:i + 16] for i in range(0, len(items), 16)]
    resA = {}
    for r in pool.map(lambda ch: sam_call(full_url, box_prompt(ch, w, h)), chunks):
        resA.update(r)

    # pass B: per-object upscaled crop
    def crop_job(o):
        x1, y1, x2, y2 = o["box"]
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        pad = int(round(0.18 * max(bw, bh) + 16))
        X0, Y0 = max(0, x1 - pad), max(0, y1 - pad)
        X1, Y1 = min(w, x2 + pad + 1), min(h, y2 + pad + 1)
        cw, ch = X1 - X0, Y1 - Y0
        s = min(MAX_UP, CROP_TARGET / max(cw, ch))
        if s < 1.25:
            return None
        Cw, Chh = int(round(cw * s)), int(round(ch * s))
        sx, sy = Cw / cw, Chh / ch
        crop = im.crop((X0, Y0, X1, Y1)).resize((Cw, Chh), Image.LANCZOS)
        cb = [int(np.floor((x1 - X0) * sx)), int(np.floor((y1 - Y0) * sy)),
              min(Cw - 1, int(np.ceil((x2 - X0 + 1) * sx)) - 1), min(Chh - 1, int(np.ceil((y2 - Y0 + 1) * sy)) - 1)]
        r = sam_call(png_url(crop), box_prompt([(1, cb)], Cw, Chh)).get(1)
        if not r or r["mask"] is None or not r["mask"].any():
            return None
        return {"rec": r, "X0": X0, "Y0": Y0, "sx": sx, "sy": sy, "Cw": Cw, "Ch": Chh}

    jobs = list(pool.map(lambda o: crop_job(o) if o["box"] else None, objs))

    out = {"masks": {}, "holes": {}, "status": {}, "metrics": {}}
    rejected, raw_own = {}, {}
    for k, (o, jb) in enumerate(zip(objs, jobs), 1):
        oldM = raster(o["old"], w, h) if o["old"] else None
        recA = resA.get(k)
        polyA, holesA = None, []
        if recA and recA["mask"] is not None and recA["mask"].any():
            bx, by = recA["box"][0], recA["box"][1]
            polyA, holesA = trace(recA["mask"] > 0, lambda l, bx=bx, by=by: l + [bx, by], 1, EPS_PX)
        polyB, holesB = None, []
        if jb:
            r = jb["rec"]
            cx, cy = r["box"][0], r["box"][1]

            def to_full(l, jb=jb, cx=cx, cy=cy):
                q = l + [cx, cy]                     # upscaled-crop pixel index
                return np.c_[jb["X0"] + (q[:, 0] + 0.5) / jb["sx"] - 0.5, jb["Y0"] + (q[:, 1] + 0.5) / jb["sy"] - 0.5]
            blur = 1 if max(jb["sx"], jb["sy"]) < 2 else 2
            polyB, holesB = trace(r["mask"] > 0, to_full, blur, EPS_PX)
        rawA, rawB = polyA, polyB
        hull = False
        if CONTAINER.search(o["id"].lower()):
            for nm in ("A", "B"):
                pp = polyA if nm == "A" else polyB
                if pp is not None:
                    hp = convex_hull(pp)
                    if abs(_area(hp)) <= 3.5 * abs(_area(pp)):
                        if nm == "A":
                            polyA, holesA = hp, []
                        else:
                            polyB, holesB = hp, []
                        hull = True
        MA = raster(polyA, w, h) if polyA is not None else None
        MB = raster(polyB, w, h) if polyB is not None else None
        iAo, iBo, iAB = iou(MA, oldM), iou(MB, oldM), iou(MA, MB)
        # choice: the new outline must hug the object's own box (box_px); where the old outline is sane it must
        # also broadly agree with it (guards against SAM grabbing a neighbour or the table)
        bA, bB = bbox_iou(polyA, o["box"]), bbox_iou(polyB, o["box"])
        bO = bbox_iou(np.asarray(o["old"], float) if o["old"] else None, o["box"])
        box_area = (o["box"][2] - o["box"][0] + 1) * (o["box"][3] - o["box"][1] + 1) if o["box"] else 1
        old_ok = oldM is not None and bO >= 0.6 and oldM.sum() >= 0.15 * box_area
        okB = MB is not None and bB >= 0.6 and (MA is None or iAB >= 0.7)
        okA = MA is not None and bA >= 0.6
        if not okB and MB is not None and bB >= 0.6 and old_ok and iBo >= 0.7:
            okB = True        # full-frame prompt grabbed a neighbour (clutter), the crop agrees with the old outline
        reject = None
        if o["id"] in FORCE_OLD.get(nn, ()) and o["old"]:
            okA = okB = False
        if o["id"] in FORCE_NEW.get(nn, ()):
            old_ok = False
        if okB and (not old_ok or iBo >= 0.45):
            st, poly, holes = "new_crop", polyB, holesB
        elif okA and (not old_ok or iAo >= 0.45):
            st, poly, holes = "new_full", polyA, holesA
        elif o["old"]:
            st, poly, holes = "kept_old", np.asarray(o["old"], float), []
            reject = polyB if polyB is not None else polyA
        else:
            st, poly, holes = "missing", None, []
            reject = polyB if polyB is not None else polyA
        rejected[o["id"]] = reject
        hull = hull and st.startswith("new")
        if hull:
            raw_own[o["id"]] = rawB if st == "new_crop" else rawA
        if holes:
            holes = see_through(arr, poly, holes, w, h)
        out["status"][o["id"]] = st
        out["metrics"][o["id"]] = {"label": o["label"], "iou_full_vs_old": round(iAo, 3),
                                   "iou_crop_vs_old": round(iBo, 3), "iou_crop_vs_full": round(iAB, 3),
                                   "n_old": len(o["old"] or []), "n_new": 0 if poly is None else len(poly),
                                   "sam_conf_full": recA and recA["conf"], "sam_conf_crop": jb and jb["rec"]["conf"],
                                   "upscale": jb and round(jb["sx"], 2), "bbox_iou_new": round(max(bA, bB), 3),
                                   "bbox_iou_old": round(bO, 3), "old_sane": bool(old_ok), "hull_filled": hull,
                                   "area_new_over_old": round(float(abs(_area(np.asarray(poly, float))) /
                                                                    max(abs(_area(np.asarray(o["old"], float))), 1)), 3)
                                   if (poly is not None and o["old"]) else None}
        if poly is not None:
            out["masks"][o["id"]] = [[round(float(u), 1), round(float(v), 1)] for u, v in poly]
            if holes:
                out["holes"][o["id"]] = [[[round(float(u), 1), round(float(v), 1)] for u, v in hh] for hh in holes]
    # a hull must not swallow a neighbouring object that lies on top (chopsticks across a bowl, an overlapping plate):
    # remove pixels that belong to another object's outline unless SAM gave them to this object itself
    for oid, raw in raw_own.items():
        hp = np.asarray(out["masks"][oid], float)
        x0, y0 = max(0, int(hp[:, 0].min()) - 2), max(0, int(hp[:, 1].min()) - 2)
        x1, y1 = min(w, int(np.ceil(hp[:, 0].max())) + 3), min(h, int(np.ceil(hp[:, 1].max())) + 3)
        sub = lambda P: raster(np.asarray(P, float) - [x0, y0], x1 - x0, y1 - y0)
        H, own = sub(hp), sub(raw)
        others = np.zeros_like(H)
        for oid2, p2 in out["masks"].items():
            if oid2 != oid:
                others |= sub(p2)
        cut = H & others & ~own
        if cut.sum() > 20:
            poly2, _ = trace(H & ~cut, lambda l: l + [x0, y0], 0, EPS_PX)
            # an occluder lying right across (chopsticks over a bowl) would split the vessel: keep the full hull then
            if poly2 is not None and sub(poly2).sum() >= 0.9 * (H & ~cut).sum():
                out["masks"][oid] = [[round(float(u), 1), round(float(v), 1)] for u, v in poly2]
                out["metrics"][oid]["hull_cut_px"] = int(cut.sum())
    rec = {"table": f"t6_{nn}", "photo": sc["photo"]["file"], "width": w, "height": h,
           "frame": "full-resolution photo pixels, (u, v) = (column, row), pixel centres at integers; "
                    "same frame as scene objects[*].perception.mask_px (no rescale)",
           "model": "sam-3.1 box prompts (perception.box_px), pass A full frame + pass B upscaled crop", **out}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"t6_{nn}.json").write_text(json.dumps(rec, indent=0))
    overlay(nn, im, objs, rec, rejected).save(OUT / f"t6_{nn}_overlay.jpg", quality=88)
    return rec


def _font(sz):
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(f, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def overlay(nn, im, objs, rec, rejected=None) -> Image.Image:
    S = 2   # supersample for crisp thin lines
    w, h = im.size
    base = im.resize((w * S, h * S), Image.LANCZOS).point(lambda v: int(v * 0.8))
    d = ImageDraw.Draw(base)
    f = _font(13 * S)
    for o in objs:
        if o["old"]:
            d.line([(u * S, v * S) for u, v in o["old"]] + [(o["old"][0][0] * S, o["old"][0][1] * S)],
                   fill=(255, 50, 50), width=2 * S // 2 + 1)
    for o in objs:
        st = rec["status"].get(o["id"])
        p = rec["masks"].get(o["id"])
        if p and st != "kept_old":
            d.line([(u * S, v * S) for u, v in p] + [(p[0][0] * S, p[0][1] * S)], fill=(40, 255, 90), width=2)
            for hh in rec["holes"].get(o["id"], []):
                d.line([(u * S, v * S) for u, v in hh] + [(hh[0][0] * S, hh[0][1] * S)], fill=(40, 200, 255), width=2)
        rj = (rejected or {}).get(o["id"])
        if rj is not None and len(rj) >= 3:
            d.line([(u * S, v * S) for u, v in rj] + [(rj[0][0] * S, rj[0][1] * S)], fill=(255, 150, 0), width=2)
        if o["box"]:
            col = {"new_crop": (40, 255, 90), "new_full": (160, 255, 160), "kept_old": (255, 210, 0),
                   "missing": (255, 0, 255)}.get(st, (255, 255, 255))
            d.text((o["box"][0] * S + 2, o["box"][1] * S - 15 * S), f"{o['id']} [{st}]", fill=col, font=f)
    d.text((10, 10), f"t6_{nn}  red = old mask_px   green = masks_hi   cyan = holes   yellow tag = kept old (orange = rejected SAM)",
           fill=(255, 255, 255), font=_font(18 * S))
    return base.resize((w, h), Image.LANCZOS)


def contact():
    tiles = []
    for nn in [f"{i:02d}" for i in range(1, 51)]:
        p = OUT / f"t6_{nn}_overlay.jpg"
        if p.exists():
            tiles.append(Image.open(p).resize((482, 362), Image.LANCZOS))
    cols = 7
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 486 + 4, rows * 366 + 4), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, (4 + (i % cols) * 486, 4 + (i // cols) * 366))
    sheet.save(OUT / "review_contact.jpg", quality=85)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--contact" in sys.argv:
        contact()
        return
    nns = args or [f"{i:02d}" for i in range(1, 51)]
    summ = {}
    with ThreadPoolExecutor(8) as pool:
        for nn in nns:
            rec = run_table(nn, pool)
            c = {}
            for s in rec["status"].values():
                c[s] = c.get(s, 0) + 1
            summ[nn] = c
            print(nn, c, flush=True)
    contact()


if __name__ == "__main__":
    main()
