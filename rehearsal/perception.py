"""Photo -> scene JSON (schema in CONTRACTS.md §2).

1. Muse Spark lists the movable objects (id, label, SAM category, rough center, primitive shape, dims, mass,
   friction).                                                                   -> list_objects
2. SAM 3.1 boxes every instance of each category (concept prompt); each listed object takes the nearest
   unclaimed box; leftovers fall back to a point prompt at Muse Spark's center ("low" confidence).
                                                                                -> segment_objects
3. A sheet of A4 paper lying flat on the table gives the pixel->table homography (rehearsal.calib).  From it we
   recover a full pinhole camera (focal length from the two rotation constraints), then place each object by
   fitting its 3D primitive so that the projected 2D bounding box matches SAM's box ("box_fit"; the object's
   overall scale is fitted too, its proportions come from Muse Spark).          -> estimate_scene
4. Objects are clamped onto the table and pushed apart if they overlap (flagged in `perception.adjusted`).

The measured accuracy of every placement method on synthetic renders with exact ground truth is in
docs/perception.md (python -m rehearsal.synth eval).

    python -m rehearsal.perception photo.jpg -o scenes/my_desk.json            # auto-detect the A4 sheet
    python -m rehearsal.perception photo.jpg --paper-corners "u,v u,v u,v u,v" -o scenes/my_desk.json

Backend entry point: perceive_photo(path, privacy, corners=None, ...) -> (scene, overlay_path)
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .calib import A4, apply_h, camera_from_homography, detect_paper, fit_homography, order_corners, paper_world
from .scene import footprint_polygon, shape_bbox

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAPER_XY = (0.30, 0.10)     # robot-frame xy of the paper's near-left corner (see calib.py)
DEFAULT_TABLE = {"center": [0.45, 0.0], "size": [1.3, 1.2]}
# Per-axis position sigma (m) for the stress test (box_fit / no mask), from the synthetic evaluation.
SIGMA_M = {"high": 0.010, "low": 0.020}


def sigma_for(source: str | None, sam_conf: float | None, mask_iou: float | None) -> tuple[float, str]:
    """Data-driven per-axis position sigma (m) for the stress test, and the confidence label.

    Calibrated on 83 real objects with 6-DoF ground truth (scout's YCB-Video + HOPE frames, known camera, default
    "guarded" placement; docs/perception.md): sigma = radial P90 error / 2.146 (Rayleigh P90), binned by mask_iou
    (how well the fitted 3D primitive explains SAM's mask), the best predictor we found (r = -0.37 real,
    -0.44 synthetic).  SAM's own confidence did NOT predict placement error (r = +0.04 real, -0.12 synthetic);
    it only sets the "low" label below 0.7 (median error 2.6 vs 1.1-1.7 cm above).
      mask_iou >= 0.95: P90 2.5 cm -> 0.012 m | 0.85-0.95: P90 6.1 cm -> 0.028 m | < 0.85 or no mask: P90 8.2 cm -> 0.038 m
      point-prompt instances: P90 10.7 cm -> floor 0.050 m;  SAM confidence < 0.7: floor 0.028 m.
    """
    if mask_iou is None:
        sig = 0.038
    else:
        sig = 0.012 if mask_iou >= 0.95 else 0.028 if mask_iou >= 0.85 else 0.038
    low_conf = sam_conf is not None and sam_conf < 0.7
    if source != "concept":
        sig = max(sig, 0.050)
    if low_conf:
        sig = max(sig, 0.028)
    low = source != "concept" or low_conf or (mask_iou is not None and mask_iou < 0.85)
    return sig, "low" if low else "high"

OBJECTS_SCHEMA = {
    "type": "object",
    "properties": {"objects": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "label": {"type": "string"},
            "category": {"type": "string"},
            "center_norm": {"type": "array", "items": {"type": "number"}},
            "shape": {"type": "string", "enum": ["box", "cylinder", "bowl"]},
            "dims_m": {"type": "array", "items": {"type": "number"}},
            "mass_kg_range": {"type": "array", "items": {"type": "number"}},
            "friction_range": {"type": "array", "items": {"type": "number"}},
            "fragile": {"type": "boolean"},
        },
        "required": ["id", "label", "category", "center_norm", "shape", "dims_m", "mass_kg_range",
                     "friction_range", "fragile"],
        "additionalProperties": False}}},
    "required": ["objects"], "additionalProperties": False,
}

PROMPT = """List every movable object on the table that a robot arm could manipulate.
For each: a snake_case id (e.g. mug_1); a short label; a category that is a plain 1-3 word noun
phrase shared by all objects of that kind (e.g. "water bottle", "bowl", "stapler" - no colors,
positions or other qualifiers); center_norm, the [x, y] center of the object in the image as
fractions of width and height (0-1, origin top-left); a primitive shape; and real-world dimensions in meters
(box: [length, width, height]; cylinder/bowl: [radius, height]). Give plausible ranges for mass
and friction coefficient. Ignore the reference sheet of paper."""

import re  # noqa: E402

BOX_RE = re.compile(r"(?:^|<\d+f>|,)(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)")


# ------------------------------------------------------------------------------------------- Meta API
def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


def list_objects(client, img: str, effort: str = "low", data: str = "private") -> list[dict]:
    # "low" measured ~2x faster than the default on desk photos with the same object coverage.
    # `client` is unused (kept for call-site compatibility); all calls go through meta_client.
    from .meta_client import chat_json
    return chat_json([{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": img}}]}],
        OBJECTS_SCHEMA, "Objects", caller="perception.list_objects", data=data, effort=effort)["objects"]


SAM_META = {"include_confidence": "true"}
REC_RE = re.compile(r"(\d+)<\|box;([^|]*)\|>(?:<\|mask;([^|]*?)data=(\d+),(\d+),(.)([^|]*)\|>)?")


def _decode_mask(H: int, W: int, enc: str, payload: str):
    """SAM mask token -> uint8 raster (H, W) covering the box crop, inclusive corners (raster[0,0] = (x1, y1)).
    The encoding marker ('~' lossless, otherwise one-bit) must stay the first payload character."""
    try:
        from meta_sam_parser import SegmentationMask, decode_mask_to_raster
        m = SegmentationMask(encoding="lossless" if enc == "~" else "one_bit", payload=enc + payload,
                             width=W, height=H)
        return np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(H, W).copy()
    except Exception:
        return None


def _sam(client, img: str, prompt: str) -> dict[int, dict]:
    """One SAM 3.1 call; returns {object id: {"box": (x1, y1, x2, y2), "conf": c|None, "mask": raster|None}}."""
    from .meta_client import sam
    text = sam(prompt, img, caller="perception.sam", metadata=SAM_META)
    out = {}
    for m in REC_RE.finditer(text):
        oid, box, _, H, W, enc, payload = m.groups()
        kv = dict(t.split("=", 1) for t in box.split(";") if "=" in t)
        b = tuple(int(float(kv[k])) for k in ("x1", "y1", "x2", "y2"))
        mask = _decode_mask(int(H), int(W), enc, payload) if H else None
        out[int(oid)] = {"box": b, "conf": float(kv["c"]) if "c" in kv else None, "mask": mask}
    return out


def sam_boxes(client, img: str, phrase: str) -> list[dict]:
    """Concept prompt: every instance of a short noun phrase."""
    return list(_sam(client, img, phrase).values())


def sam_points(client, img: str, points: dict[int, tuple[int, int]], w: int, h: int):
    """Point prompts: one positive point per object id, all objects in one request."""
    prompt = "<0f>" + ",".join(f"{i}<|point;x={x};y={y};w={w};h={h}|>" for i, (x, y) in points.items())
    return _sam(client, img, prompt)


def _attach(o: dict, rec: dict, source: str):
    o["box"], o["box_source"] = rec["box"], source
    o["sam_conf"], o["mask"] = rec["conf"], rec["mask"]


def segment_objects(client, img: str, objs: list[dict], w: int, h: int, pool) -> list[dict]:
    """Assign a SAM instance to each object Muse Spark listed.

    1. One concept request per category returns every instance of that kind.
    2. Each object takes the unclaimed instance box nearest to its Muse Spark center
       (box containing the point preferred).
    3. Objects still without a box are segmented by a point prompt at their center.
    Adds "box" (pixels), "box_source" ("concept" | "point" | None), "sam_conf" and "mask" (raster over the box).
    """
    cats = sorted({o["category"] for o in objs})
    found = dict(zip(cats, pool.map(lambda c: sam_boxes(client, img, c), cats)))
    centers = {o["id"]: (o["center_norm"][0] * w, o["center_norm"][1] * h) for o in objs}

    def cost(o, b):
        cx, cy = centers[o["id"]]
        inside = b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
        d = np.hypot(cx - (b[0] + b[2]) / 2, cy - (b[1] + b[3]) / 2) / max(w, h)
        return d - (1.0 if inside else 0.0)

    for cat in cats:
        members, recs = [o for o in objs if o["category"] == cat], list(found[cat])
        pairs = sorted((cost(o, r["box"]), i, j) for i, o in enumerate(members) for j, r in enumerate(recs))
        used_o, used_b = set(), set()
        for c, i, j in pairs:
            if i in used_o or j in used_b or c > 0.15:
                continue
            _attach(members[i], recs[j], "concept")
            used_o.add(i); used_b.add(j)

    missing = [o for o in objs if "box" not in o]
    if missing:
        pts = {k: tuple(int(v) for v in centers[o["id"]]) for k, o in enumerate(missing, 1)}
        got = sam_points(client, img, pts, w, h)
        for k, o in enumerate(missing, 1):
            if k in got:
                _attach(o, got[k], "point")
            else:
                o["box"], o["box_source"], o["sam_conf"], o["mask"] = None, None, None, None
    return objs


def perceive_raw(photo: Path, data: str = "private") -> dict:
    """All Meta API work for one photo (cached on disk by request hash, so reruns are free)."""
    photo = Path(photo)
    img = data_url(photo)
    w, h = Image.open(photo).size
    with ThreadPoolExecutor(8) as pool:
        objs = segment_objects(None, img, list_objects(None, img, data=data), w, h, pool)
    return {"file": str(photo), "width": w, "height": h, "objects": objs}


# ------------------------------------------------------------------------------------ geometry / fit
def _dims(o: dict) -> dict:
    """Sanitised primitive from Muse Spark's dims (meters)."""
    d = [float(abs(v)) for v in o["dims_m"]] + [0.05, 0.05, 0.05]
    if o["shape"] == "box":
        out = {"type": "box", "size": [float(np.clip(d[0], 0.02, 0.5)), float(np.clip(d[1], 0.02, 0.5)),
                                       float(np.clip(d[2], 0.005, 0.4))]}
    else:
        out = {"type": o["shape"], "radius": float(np.clip(d[0], 0.01, 0.2)), "height": float(np.clip(d[1], 0.01, 0.4))}
    if o.get("z0"):  # object resting on something: height of its bottom above the table (reality check)
        out["z0"] = float(o["z0"])
    return out


def _scaled(shape: dict, s: float) -> dict:
    if shape["type"] == "box":
        return {"type": "box", "size": [round(v * s, 4) for v in shape["size"]]}
    return {"type": shape["type"], "radius": round(shape["radius"] * s, 4), "height": round(shape["height"] * s, 4)}


def _height(shape):
    return shape["size"][2] if shape["type"] == "box" else shape["height"]


def _silhouette(shape: dict, x: float, y: float, s: float = 1.0, n: int = 32) -> np.ndarray:
    """3D points whose projection bounds the object's image (yaw unknown -> boxes as a vertical cylinder
    of radius sqrt(l*w)/2... measured to be a decent stand-in averaged over yaw)."""
    if shape["type"] == "box":
        r, hgt = 0.5 * np.sqrt(shape["size"][0] * shape["size"][1]) * s, shape["size"][2] * s
    else:
        r, hgt = shape["radius"] * s, shape["height"] * s
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[x + r * np.cos(a), y + r * np.sin(a)]
    z0 = shape.get("z0", 0.0)
    return np.r_[np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z0 + hgt)]]


def _pred_box(cam, shape, x, y, s=1.0):
    uv = cam.project(_silhouette(shape, x, y, s))
    return np.r_[uv.min(0), uv.max(0)]


def fit_position(cam, shape: dict, box, w: int, h: int, fit_scale: bool = True, init=None):
    """Least squares: object (x, y[, log scale]) such that the projected primitive's 2D bbox matches `box`.
    Box edges that touch the image border are ignored (truncated)."""
    box = np.asarray(box, float)
    wts = np.array([0.0 if box[0] <= 1 else 1.0, 0.0 if box[1] <= 1 else 1.0,
                    0.0 if box[2] >= w - 2 else 1.0, 0.0 if box[3] >= h - 2 else 1.0])
    scale = float(np.hypot(box[2] - box[0], box[3] - box[1])) + 1.0
    p = np.array([*(init if init is not None else cam.intersect_z((box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                                                                     shape.get("z0", 0.0) + _height(shape) / 2)), 0.0])
    free = 3 if fit_scale else 2

    def res(q):
        r = (_pred_box(cam, shape, q[0], q[1], np.exp(q[2])) - box) * wts / scale
        prior = [0.5 * q[2]] if fit_scale else []  # weak prior: keep Muse Spark's size unless evidence
        return np.r_[r, prior]

    lam = 1e-3
    for _ in range(40):
        r0 = res(p)
        J = np.zeros((len(r0), free))
        for k in range(free):
            dp = np.zeros(3); dp[k] = 1e-4
            J[:, k] = (res(p + dp) - r0) / 1e-4
        A, g = J.T @ J, J.T @ r0
        step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
        q = p.copy(); q[:free] += step
        q[2] = np.clip(q[2], np.log(0.4), np.log(2.5))
        if np.sum(res(q) ** 2) < np.sum(r0 ** 2):
            p, lam = q, lam * 0.3
            if np.linalg.norm(step[:2]) < 1e-5:
                break
        else:
            lam *= 10
    return p[:2], float(np.exp(p[2]))


def _nelder_mead(f, x0, step, iters=160, tol=1e-6):
    n = len(x0)
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def _solid_points(shape: dict, x, y, s, yaw, n: int = 24) -> np.ndarray:
    """3D points whose convex hull's projection is the primitive's silhouette (boxes with yaw)."""
    if shape["type"] == "box":
        L, Wd, Hh = (v * s for v in shape["size"])
        c, si = np.cos(yaw), np.sin(yaw)
        xy = np.array([[sx * L / 2, sy * Wd / 2] for sx in (-1, 1) for sy in (-1, 1)])
        xy = xy @ np.array([[c, si], [-si, c]]) + [x, y]
        z0 = shape.get("z0", 0.0)
        return np.r_[np.c_[xy, np.full(4, z0)], np.c_[xy, np.full(4, z0 + Hh)]]
    r, Hh = shape["radius"] * s, shape["height"] * s
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[x + r * np.cos(a), y + r * np.sin(a)]
    z0 = shape.get("z0", 0.0)
    return np.r_[np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z0 + Hh)]]


def _roi_mask(o: dict, w: int, h: int, others=(), pad: float = 0.35, max_side: int = 120):
    """Object mask and don't-care map (other objects' masks) on a padded, downsampled ROI around the box."""
    x1, y1, x2, y2 = o["box"]
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    X1, Y1 = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
    X2, Y2 = min(w, int(x2 + pad * bw) + 1), min(h, int(y2 + pad * bh) + 1)
    k = max_side / max(X2 - X1, Y2 - Y1)
    def paste(ob):
        full = np.zeros((Y2 - Y1, X2 - X1), bool)
        bx1, by1 = ob["box"][0], ob["box"][1]
        mk = ob["mask"].astype(bool)
        ys, xs = slice(max(by1, Y1), min(by1 + mk.shape[0], Y2)), slice(max(bx1, X1), min(bx1 + mk.shape[1], X2))
        if ys.start < ys.stop and xs.start < xs.stop:
            full[ys.start - Y1:ys.stop - Y1, xs.start - X1:xs.stop - X1] = \
                mk[ys.start - by1:ys.stop - by1, xs.start - bx1:xs.stop - bx1]
        return full
    size = (max(1, int((X2 - X1) * k)), max(1, int((Y2 - Y1) * k)))
    small = lambda a: np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize(size, Image.BILINEAR)) > 127
    M = small(paste(o))
    D = np.zeros_like(M)
    for ob in others:
        if ob is not o and ob.get("mask") is not None and ob.get("box"):
            bx = ob["box"]
            if bx[0] < X2 and bx[2] > X1 and bx[1] < Y2 and bx[3] > Y1:
                D |= small(paste(ob))
    D &= ~M
    # image-border pixels outside the frame are already cut; pixels at the frame edge count as don't-care
    return M, D, (X1, Y1, k, size)


def mask_fit(cam, o: dict, w: int, h: int, others=(), init=None, fit_yaw: bool = True):
    """Fit (x, y, log scale[, yaw]) so the projected primitive silhouette matches the SAM mask (IoU, occluders
    = don't care).  Returns (xy, scale, yaw, iou)."""
    shape = _dims(o)
    M, D, (X1, Y1, k, size) = _roi_mask(o, w, h, others)
    care = ~D
    from PIL import ImageDraw as _D

    def sil(x, y, ls, yaw):
        uv = (cam.project(_solid_points(shape, x, y, np.exp(ls), yaw)) - [X1, Y1]) * k
        hull = _hull_uv(uv)
        im = Image.new("L", size, 0)
        _D.Draw(im).polygon([tuple(p) for p in hull], fill=255)
        return np.asarray(im) > 127

    def loss(p):
        x, y, ls = p[:3]
        yaw = p[3] if len(p) > 3 else 0.0
        P = sil(x, y, ls, yaw)
        inter = np.count_nonzero(P & M & care)
        union = np.count_nonzero((P | M) & care)
        return 1 - inter / max(union, 1) + 0.02 * ls * ls

    if init is None:
        init, s0 = fit_position(cam, shape, o["box"], w, h)
    else:
        init, s0 = init
    best = None
    is_box = shape["type"] == "box" and fit_yaw
    starts = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4] if is_box else [None]
    for y0 in starts:
        x0 = [init[0], init[1], np.log(s0)] + ([y0] if y0 is not None else [])
        st = [0.02, 0.02, 0.1] + ([0.3] if y0 is not None else [])
        p, v = _nelder_mead(loss, x0, st)
        if best is None or v < best[1]:
            best = (p, v)
    p, v = best
    yaw = float(((p[3] + np.pi / 2) % np.pi) - np.pi / 2) if len(p) > 3 else 0.0
    return p[:2], float(np.exp(np.clip(p[2], np.log(0.4), np.log(2.5)))), yaw, float(1 - v)


def _hull_uv(pts):
    from .calib import _hull
    return _hull(np.asarray(pts, float))


GUARD = {"guarded": (0.85, 0.04), "guarded_iou": (0.85, 9.0), "guarded_dist": (0.0, 0.04),
         "guarded_loose": (0.80, 0.06)}


def place_object(o: dict, cam, H, method: str, w: int, h: int, others=()):
    """-> (xy, scale, yaw, extra) for one listed object with a SAM box."""
    if method.startswith("mask_fit") or method.startswith("guarded"):
        bxy, bs = fit_position(cam, _dims(o), o["box"], w, h)
        if o.get("mask") is None:
            return bxy, bs, 0.0, {"mask_iou": None, "placement": "box_fit"}
        xy, s, yaw, iou = mask_fit(cam, o, w, h, others, init=(bxy, bs), fit_yaw=(method != "mask_fit_noyaw"))
        extra = {"mask_iou": round(iou, 3), "placement": "mask_fit"}
        if method.startswith("guarded"):
            # Fall back to the 2D-box fit when the primitive cannot explain the mask (lying / non-convex objects)
            # or the two fits disagree a lot - measured to remove mask_fit's gross failures on real photos.
            thr_iou, thr_d = GUARD.get(method, GUARD["guarded"])
            dd = float(np.hypot(*(np.asarray(xy) - bxy)))
            extra["mask_vs_box_cm"] = round(dd * 100, 2)
            if iou < thr_iou or dd > thr_d:
                return bxy, bs, 0.0, {**extra, "placement": "box_fit"}
        return xy, s, yaw, extra
    xy, s = _place_box(o, cam, H, method, w, h)
    return xy, s, 0.0, {}


def _place_box(o: dict, cam, H, method: str, w: int, h: int):
    x1, y1, x2, y2 = o["box"]
    shape = _dims(o)
    r = shape["radius"] if shape["type"] != "box" else min(shape["size"][:2]) / 2
    if method == "bottom_center":      # original MVP: bottom-center of the box, shifted +x by r
        return apply_h(H, [((x1 + x2) / 2, y2)])[0] + np.array([r, 0.0]), 1.0
    if method == "bottom_ray":         # bottom-center, shifted by r away from the camera (horizontally)
        xy = apply_h(H, [((x1 + x2) / 2, y2)])[0]
        d = xy - cam.center[:2]
        return xy + r * d / (np.linalg.norm(d) + 1e-9), 1.0
    if method == "ray_mid":            # ray through the box center hits the plane z = h/2
        return cam.intersect_z((x1 + x2) / 2, (y1 + y2) / 2, _height(shape) / 2), 1.0
    if method == "box_fit_fixed":      # 2D-box fit, Muse Spark dims
        return fit_position(cam, shape, o["box"], w, h, fit_scale=False)
    if method == "box_fit":            # 2D-box fit, overall scale fitted
        return fit_position(cam, shape, o["box"], w, h, fit_scale=True)
    raise ValueError(method)


def mask_outline(o: dict, max_pts: int = 48):
    """Simplified outline polygon [[u, v], ...] of the SAM mask in image pixels (row extremes, so holes and
    concavities along rows are filled), or None."""
    mk = o.get("mask")
    if mk is None or not mk.any():
        return None
    x1, y1 = o["box"][0], o["box"][1]
    rows = np.flatnonzero(mk.any(1))
    left = [(int(np.argmax(mk[r])) + x1, r + y1) for r in rows]
    right = [(mk.shape[1] - 1 - int(np.argmax(mk[r][::-1])) + x1 + 1, r + y1) for r in rows]
    poly = np.array(left + right[::-1], float)
    from .calib import _simplify_closed
    eps = 1.0
    out = poly
    while len(out) > max_pts and eps < 50:
        out = _simplify_closed(poly, eps)
        eps *= 1.6
    return [[round(float(u), 1), round(float(v), 1)] for u, v in out]


def _mean_color(img: Image.Image | None, box) -> list[float]:
    if img is None:
        return [0.7, 0.7, 0.7, 1.0]
    x1, y1, x2, y2 = box
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 0.15, (y2 - y1) * 0.15
    a = np.asarray(img.crop((int(cx - hw), int(cy - hh), int(cx + hw) + 1, int(cy + hh) + 1)).convert("RGB"), float)
    c = np.median(a.reshape(-1, 3), axis=0) / 255 if a.size else np.array([0.7, 0.7, 0.7])
    return [round(float(v), 3) for v in c] + [1.0]


def _footprint_r(shape) -> float:
    if shape["type"] in ("cylinder", "bowl") and shape.get("axis", "z") == "z":
        return shape["radius"]
    return 0.5 * float(np.hypot(*shape_bbox(shape)[:2]))


def tidy(objects: list[dict], table: dict, margin: float = 0.01, robot_r: float = 0.12):
    """Clamp objects onto the table, out of the robot base, and push overlapping footprints apart.
    Flags every change in obj["perception"]["adjusted"] (list of reasons)."""
    cx, cy = table["center"]
    sx, sy = table["size"]
    for _ in range(50):
        moved = False
        for i, a in enumerate(objects):
            ra = _footprint_r(a["shape"])
            for b in objects[i + 1:]:
                rb = _footprint_r(b["shape"])
                d = np.subtract(b["pos"], a["pos"])
                dist = float(np.linalg.norm(d))
                if dist < ra + rb + 0.002:
                    n = d / dist if dist > 1e-6 else np.array([0.0, 1.0])
                    push = (ra + rb + 0.004 - dist) / 2
                    a["pos"] = list(np.subtract(a["pos"], n * push)); b["pos"] = list(np.add(b["pos"], n * push))
                    for o in (a, b):
                        o["perception"].setdefault("adjusted", [])
                        if "overlap" not in o["perception"]["adjusted"]:
                            o["perception"]["adjusted"].append("overlap")
                    moved = True
        for o in objects:
            r = _footprint_r(o["shape"])
            p = np.array(o["pos"], float)
            q = np.clip(p, [cx - sx / 2 + r + margin, cy - sy / 2 + r + margin],
                        [cx + sx / 2 - r - margin, cy + sy / 2 - r - margin])
            if np.linalg.norm(q) < robot_r + r:
                q = q / (np.linalg.norm(q) + 1e-9) * (robot_r + r)
            if np.linalg.norm(q - p) > 1e-6:
                o["pos"] = list(q)
                o["perception"].setdefault("adjusted", [])
                if "table_bounds" not in o["perception"]["adjusted"]:
                    o["perception"]["adjusted"].append("table_bounds")
                moved = True
        if not moved:
            break
    for o in objects:
        o["pos"] = [round(float(o["pos"][0]), 4), round(float(o["pos"][1]), 4)]


# ------------------------------------------------------------------- fidelity: colour, orientation, shape
# Everything here only uses the photo, the SAM mask and the camera (never dataset ground truth); it is measured
# by scripts/fidelity_eval.py (silhouette IoU and Lab dE from the photo's own camera).

def srgb_to_lab(rgb) -> np.ndarray:
    """sRGB in [0, 1], shape (..., 3) -> CIE Lab (D65)."""
    c = np.asarray(rgb, float)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_srgb(lab) -> np.ndarray:
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    Mi = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                   [0.0556434, -0.2040259, 1.0572252]])
    lin = np.clip(xyz @ Mi.T, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def _mask_full(o: dict, w: int, h: int, erode: int = 2) -> np.ndarray | None:
    """SAM mask in full-image coordinates, eroded by `erode` px so edge pixels (background, neighbours) drop out."""
    mk = o.get("mask")
    if mk is None or not mk.any():
        return None
    full = np.zeros((h, w), bool)
    x1, y1 = int(o["box"][0]), int(o["box"][1])
    mk = mk.astype(bool)
    y2, x2 = min(h, y1 + mk.shape[0]), min(w, x1 + mk.shape[1])
    full[y1:y2, x1:x2] = mk[:y2 - y1, :x2 - x1]
    er = full.copy()
    for dy in range(-erode, erode + 1):
        for dx in range(-erode, erode + 1):
            er &= np.roll(np.roll(full, dy, 0), dx, 1)
    return er if er.sum() >= 30 else full


LIGHT_PCT = 60   # "lightness correction": shading/shadows pull the median L below the lit albedo


def _lab_colour(lab_px: np.ndarray) -> list[float]:
    """Robust colour of a pixel set: median a, b and the LIGHT_PCT-th percentile L -> rgba."""
    L = float(np.percentile(lab_px[:, 0], LIGHT_PCT))
    ab = np.median(lab_px[:, 1:], axis=0)
    rgb = lab_to_srgb([L, ab[0], ab[1]])
    return [round(float(v), 3) for v in rgb] + [1.0]


def mask_colour(img_lab: np.ndarray, o: dict) -> list[float] | None:
    M = _mask_full(o, img_lab.shape[1], img_lab.shape[0])
    return None if M is None else _lab_colour(img_lab[M])


def _two_tone(lab_px: np.ndarray, iters: int = 8):
    """2-means in Lab -> (dominant Lab, secondary Lab, share of the secondary, dE between them)."""
    c = np.array([np.percentile(lab_px, 25, axis=0), np.percentile(lab_px, 75, axis=0)])
    for _ in range(iters):
        d = np.linalg.norm(lab_px[:, None] - c[None], axis=2)
        lab_ = d.argmin(1)
        for k in range(2):
            if (lab_ == k).any():
                c[k] = np.median(lab_px[lab_ == k], axis=0)
    n1 = float((lab_ == 1).mean())
    dom, sec = (1, 0) if n1 > 0.5 else (0, 1)
    return c[dom], c[sec], min(n1, 1 - n1), float(np.linalg.norm(c[0] - c[1]))


BAND_MIN_DE, BAND_MIN_SHARE = 14.0, 0.10


def colour_bands(img_lab: np.ndarray, o: dict, along: str = "rows") -> list[dict] | None:
    """Split the object's visible colour into 2-3 bands along image rows (standing objects: top of the image =
    top of the object) or columns (lying objects).  Per-line median colours; the split minimising the
    count-weighted within-band Lab variance wins, and only if adjacent bands differ by dE >= BAND_MIN_DE and each
    band holds >= BAND_MIN_SHARE of the lines.  Returns [{"frac", "rgba"}, ...] from the first line on, or None."""
    M = _mask_full(o, img_lab.shape[1], img_lab.shape[0])
    if M is None:
        return None
    if along == "cols":
        M, lab = M.T, np.transpose(img_lab, (1, 0, 2))
    else:
        lab = img_lab
    lines = [lab[i][M[i]] for i in range(M.shape[0]) if M[i].any()]
    lines = [l for l in lines if len(l) >= 3]
    n = len(lines)
    if n < 8:
        return None
    med = np.array([np.median(l, axis=0) for l in lines])
    cnt = np.array([len(l) for l in lines], float)
    # cumulative sums for O(1) weighted variance of any line range
    cw, cs, cq = np.cumsum(np.r_[0, cnt]), np.cumsum(np.r_[np.zeros((1, 3)), med * cnt[:, None]], 0), \
        np.cumsum(np.r_[0, (med ** 2).sum(1) * cnt])

    def sse(a, b):  # lines a..b-1
        wgt = cw[b] - cw[a]
        if wgt <= 0:
            return 0.0
        mu = (cs[b] - cs[a]) / wgt
        return float(cq[b] - cq[a] - wgt * (mu @ mu))

    def mean(a, b):
        return (cs[b] - cs[a]) / (cw[b] - cw[a])

    m = max(4, int(np.ceil(BAND_MIN_SHARE * n)))
    best = (sse(0, n), [(0, n)])
    for i in range(m, n - m + 1):            # two bands
        v = sse(0, i) + sse(i, n)
        if v < best[0] * 0.75:
            best = (v, [(0, i), (i, n)])
    for i in range(m, n - 2 * m + 1):        # three bands
        for j in range(i + m, n - m + 1):
            v = sse(0, i) + sse(i, j) + sse(j, n)
            if v < best[0] * 0.75:
                best = (v, [(0, i), (i, j), (j, n)])
    segs = best[1]
    if len(segs) == 1:
        return None
    cols = [mean(a, b) for a, b in segs]
    if any(np.linalg.norm(cols[k] - cols[k + 1]) < BAND_MIN_DE for k in range(len(cols) - 1)):
        return None
    out = []
    for (a, b) in segs:
        px = np.concatenate(lines[a:b])
        out.append({"frac": round((b - a) / n, 3), "rgba": _lab_colour(px)})
    return out


# --- category templates: a few recognisable compound shapes, scaled to the fitted bounding box
def _rgba(lab):
    return [round(float(v), 3) for v in lab_to_srgb(lab)] + [1.0]


def template_shape(category: str, label: str, shape: dict, primary, secondary=None) -> dict | None:
    """Compound shape for a recognisable category, in the object frame (origin = centre of the bbox).
    primary / secondary: rgba.  Returns None when there is no template for the category."""
    cat = f"{category} {label}".lower()
    sec = secondary or primary
    if "drill" in cat:
        L, W, H = shape_bbox(shape)
        W = min(W, 0.45 * L)
        parts = [
            {"type": "box", "size": [0.70 * L, 0.80 * W, 0.36 * H], "pos": [0.08 * L, 0, 0.32 * H], "rgba": primary,
             "name": "body"},
            {"type": "cylinder", "radius": 0.30 * W, "height": 0.14 * L, "axis": "x",
             "pos": [0.43 * L, 0, 0.32 * H], "rgba": sec, "name": "chuck"},
            {"type": "box", "size": [0.26 * L, 0.70 * W, 0.46 * H], "pos": [-0.17 * L, 0, -0.05 * H], "rgba": sec,
             "name": "handle"},
            {"type": "box", "size": [0.45 * L, 1.0 * W, 0.22 * H], "pos": [-0.12 * L, 0, -0.39 * H], "rgba": sec,
             "name": "battery"}]
        return {"type": "compound", "parts": parts, "bbox": [L, W, H], "template": "drill"}
    if shape["type"] == "cylinder" and ("mug" in cat or "cup" in cat) and "bottle" not in cat:
        r, h = shape["radius"], shape["height"]
        parts = [{"type": "cylinder", "radius": r, "height": h, "rgba": primary, "name": "body"},
                 {"type": "box", "size": [0.45 * r, 0.25 * r, 0.55 * h], "pos": [1.25 * r, 0, 0], "rgba": primary,
                  "name": "handle"},
                 {"type": "box", "size": [0.55 * r, 0.25 * r, 0.12 * h], "pos": [1.0 * r, 0, 0.28 * h], "rgba": primary},
                 {"type": "box", "size": [0.55 * r, 0.25 * r, 0.12 * h], "pos": [1.0 * r, 0, -0.28 * h], "rgba": primary}]
        return {"type": "compound", "parts": parts, "bbox": [2 * r, 2 * r, h], "template": "mug"}
    if shape["type"] == "cylinder" and "bottle" in cat:
        r, h = shape["radius"], shape["height"]
        parts = [{"type": "cylinder", "radius": r, "height": 0.72 * h, "pos": [0, 0, -0.14 * h], "rgba": primary,
                  "name": "body"},
                 {"type": "cylinder", "radius": 0.62 * r, "height": 0.12 * h, "pos": [0, 0, 0.28 * h], "rgba": primary},
                 {"type": "cylinder", "radius": 0.40 * r, "height": 0.16 * h, "pos": [0, 0, 0.42 * h], "rgba": sec,
                  "name": "cap"}]
        return {"type": "compound", "parts": parts, "bbox": [2 * r, 2 * r, h], "template": "bottle"}
    return None


# --- orientation: yaw, standing vs lying, which box side is up - by silhouette IoU against the SAM mask
def _rot_z(yaw):
    c, si = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -si, 0], [si, c, 0], [0, 0, 1]])


def _prim_points(p: dict) -> np.ndarray:
    """Points (object frame) whose convex hull is one primitive, centred at the origin."""
    if p["type"] == "box":
        return np.array([[sx * p["size"][0] / 2, sy * p["size"][1] / 2, sz * p["size"][2] / 2]
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    r, hgt, n = p["radius"], p["height"], 16
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[r * np.cos(a), r * np.sin(a)]
    pts = np.r_[np.c_[ring, np.full(n, -hgt / 2)], np.c_[ring, np.full(n, hgt / 2)]]
    axis = p.get("axis", "z")
    if axis == "x":
        pts = pts[:, [2, 1, 0]] * [1, 1, -1]
    elif axis == "y":
        pts = pts[:, [0, 2, 1]] * [1, 1, -1]
    return pts


def shape_point_sets(shape: dict, x: float, y: float, yaw: float) -> list[np.ndarray]:
    """World-frame point sets (one convex part each) of a shape resting on the table at (x, y) with `yaw`."""
    bb = shape_bbox(shape)
    base = np.array([x, y, shape.get("z0", 0.0) + bb[2] / 2])
    R = _rot_z(yaw)
    parts = shape["parts"] if shape["type"] == "compound" else [shape]
    out = []
    for p in parts:
        pts = _prim_points(p) + np.asarray(p.get("pos", [0, 0, 0]), float)
        out.append(pts @ R.T + base)
    return out


def _sil_union(cam, point_sets, X1, Y1, k, size) -> np.ndarray:
    from PIL import ImageDraw as _D
    im = Image.new("L", size, 0)
    d = _D.Draw(im)
    for pts in point_sets:
        uv = (cam.project(pts) - [X1, Y1]) * k
        hull = _hull_uv(uv)
        if len(hull) >= 3:
            d.polygon([tuple(p) for p in hull], fill=255)
    return np.asarray(im) > 127


def _box_variants(shape: dict) -> list[tuple[str, dict]]:
    """Which side of a box is up: as listed (standing), lying on its long side, standing on its end."""
    L, W, H = shape["size"]
    out = [("standing", shape)]
    if max(L, W, H) > 1.6 * min(L, W, H):
        out += [("on_side", {**shape, "size": [L, H, W]}), ("on_end", {**shape, "size": [H, W, L]})]
    return out


def refine_orientation(cam, o: dict, shape: dict, xy, yaw: float, w: int, h: int, others=(),
                       switch_margin: float = 0.08):
    """Yaw (and standing vs lying) that best explains the SAM mask, position (nearly) fixed.
    Candidates: the shape as is; a cylinder on its side when it is elongated; a box with another side up.
    Switching away from the standing pose needs an IoU gain >= switch_margin.  Within a candidate the position
    may shift a little (quadratic penalty, 2 cm ~ 0.03 IoU); a pose change re-fits xy freely.
    Returns (shape, xy, yaw, info)."""
    M, D, (X1, Y1, k, size) = _roi_mask(o, w, h, others)
    care = ~D
    xy0 = np.asarray(xy, float)

    def iou(shp, x, y, yw):
        P = _sil_union(cam, shape_point_sets(shp, x, y, yw), X1, Y1, k, size)
        inter = np.count_nonzero(P & M & care)
        union = np.count_nonzero((P | M) & care)
        return inter / max(union, 1)

    def fit(shp, free_xy: bool):
        lam = 0.03 / 0.02 ** 2 * (0.1 if free_xy else 1.0)
        loss = lambda p: 1 - iou(shp, p[1], p[2], p[0]) + lam * float(np.sum((p[1:] - xy0) ** 2))
        best = None
        sym = np.pi if shp["type"] == "box" else 2 * np.pi   # a compound (drill) is not symmetric end to end
        if shp["type"] == "cylinder" and shp.get("axis", "z") == "z":
            grid = [0.0]
        else:
            grid = list(np.linspace(0, sym, 12, endpoint=False))
        for yw in grid:
            v = loss(np.array([yw, *xy0]))
            if best is None or v < best[1]:
                best = (np.array([yw, *xy0]), v)
        if len(grid) > 1 or free_xy:
            p, v = _nelder_mead(loss, best[0], [0.15, 0.015, 0.015], iters=120)
            best = (p, v)
        p, v = best
        if not free_xy:  # keep the baseline yaw unless the refinement clearly explains the mask better
            v0 = loss(np.array([yaw, *xy0]))
            if v0 - v < 0.02:
                p = np.array([yaw, *xy0])
        return p, iou(shp, p[1], p[2], p[0])

    if shape["type"] == "cylinder" and shape.get("axis", "z") == "z" and shape["height"] > 2.2 * 2 * shape["radius"]:
        cands = [("standing", shape), ("lying", {**shape, "axis": "x"})]
    elif shape["type"] == "box":
        cands = _box_variants(shape)
    else:
        cands = [("standing", shape)]
    results = []
    for name, shp in cands:
        p, v = fit(shp, free_xy=(name != "standing"))
        results.append((name, shp, p, v))
    base = results[0]
    best = base
    for r in results[1:]:
        if r[3] > best[3] + (switch_margin if best is base else 0.0):
            best = r
    name, shp, p, v = best
    yaw_out = float(p[0])
    if shp["type"] == "box":
        yaw_out = float(((yaw_out + np.pi / 2) % np.pi) - np.pi / 2)
    elif shp["type"] == "compound":
        yaw_out = float(((yaw_out + np.pi) % (2 * np.pi)) - np.pi)
    elif shp.get("axis") == "x":
        yaw_out = float(((yaw_out + np.pi / 2) % np.pi) - np.pi / 2)   # a lying cylinder is symmetric end to end
    return shp, [float(p[1]), float(p[2])], yaw_out, {"pose": name, "iou_before": round(base[3] if name != "standing" else iou(shape, xy0[0], xy0[1], yaw), 3),
                                                       "iou_after": round(v, 3), "shift_cm": round(float(np.linalg.norm(p[1:] - xy0)) * 100, 2)}


def _axis_end_columns(cam, shape, xy, yaw) -> bool:
    """For a lying cylinder: True if its +axis end projects to the right (larger u) of its -axis end."""
    L = shape["height"]
    d = np.array([np.cos(yaw), np.sin(yaw)]) * L / 2
    z = shape_bbox(shape)[2] / 2
    uv = cam.project([[xy[0] + d[0], xy[1] + d[1], z], [xy[0] - d[0], xy[1] - d[1], z]])
    return bool(uv[0, 0] > uv[1, 0])


def apply_fidelity(objects: list[dict], raw_objs: list[dict], cam, img: Image.Image | None, w: int, h: int):
    """Photo-faithful appearance and orientation for every object with a SAM mask (in place):
    1. colour: robust Lab median of the mask pixels; 2-3 colour bands for labelled cans/boxes;
    2. shape: category template (drill / mug / bottle) with dominant + secondary colour;
    3. orientation: yaw, standing vs lying, which box side is up - by silhouette IoU vs the mask."""
    if img is None:
        return
    lab = srgb_to_lab(np.asarray(img.convert("RGB"), float) / 255.0)
    cxy = cam.center[:2]
    dist = {id(o): float(np.linalg.norm(np.asarray(obj["pos"]) - cxy)) for obj, o in zip(objects, raw_objs)}
    for obj, o in zip(objects, raw_objs):
        if o.get("mask") is None:
            continue
        # Only objects nearer the camera may hide this one (don't-care); a farther object's mask must stay
        # uncovered, otherwise the silhouette grows over its neighbours (the drill body over the Spam can).
        nearer = [r for r in raw_objs if r is not o and dist.get(id(r), 0.0) < dist[id(o)]]
        M = _mask_full(o, w, h)
        if M is None:
            continue
        px = lab[M]
        dom, sec, share, de = _two_tone(px)
        primary = _lab_colour(px)
        secondary = _rgba(sec) if (de >= BAND_MIN_DE and share >= 0.15) else None
        obj["rgba"] = primary
        fid = {"colour": "mask_median", "two_tone_dE": round(de, 1), "secondary_share": round(share, 2)}
        shape = obj["shape"]
        tpl = template_shape(obj.get("category", ""), obj.get("label", ""), shape, primary, secondary)
        if tpl is not None:
            # the darker cluster is the handle/battery/cap on these objects
            if secondary is not None and sec[0] > dom[0]:
                for part in tpl["parts"]:
                    part["rgba"] = primary if part["rgba"] == secondary else secondary
            shape = tpl
            fid["template"] = tpl["template"]
        shape, xy, yaw, info = refine_orientation(cam, o, shape, obj["pos"], obj.get("yaw", 0.0), w, h, nearer)
        fid.update(info)
        obj["shape"], obj["pos"], obj["yaw"] = shape, [round(xy[0], 4), round(xy[1], 4)], round(yaw, 4)
        if shape["type"] in ("box", "cylinder"):
            lying = shape.get("axis", "z") != "z"
            bands = colour_bands(lab, o, "cols" if lying else "rows")
            if bands:
                if lying:
                    if _axis_end_columns(cam, shape, xy, yaw):   # bands run from the -axis end to the +axis end
                        pass
                    else:
                        bands = bands[::-1]
                else:
                    bands = bands[::-1]                         # image top = object top; bands are bottom-up
                shape["bands"] = bands
                fid["bands"] = len(bands)
        obj["perception"]["fidelity"] = fid


def estimate_scene(raw: dict, paper_px, paper_xy=DEFAULT_PAPER_XY, method: str = "guarded",
                   paper_size=A4, table: dict | None = None, clamp: bool = True, privacy: str = "private",
                   name: str | None = None, photo_file: str | None = None, f: float | None = None,
                   camera=None, fidelity: bool = True) -> dict:
    """Pure geometry (no API): raw perception + paper corners -> scene JSON.
    camera: optional calib.Camera already in the robot/table frame (e.g. known K + pose); then paper_px may be None."""
    w, h = raw["width"], raw["height"]
    if camera is not None:
        cam, cam_info = camera, {"f_px": float(camera.K[0, 0]), "f_source": "given_camera",
                                 "cam_height_m": round(float(camera.center[2]), 3)}
        G = cam.K @ np.c_[cam.R[:, 0], cam.R[:, 1], cam.t]
        H = np.linalg.inv(G); H = H / H[2, 2]
        paper_px = np.zeros((0, 2)) if paper_px is None else np.asarray(paper_px, float)
    else:
        paper_px = np.asarray(paper_px, float)
        H = fit_homography(paper_px, paper_world(paper_xy, paper_size))
        cam, cam_info = camera_from_homography(H, w, h, f=f)
    table = table or DEFAULT_TABLE
    try:
        img = Image.open(raw["file"]).convert("RGB")
    except Exception:
        img = None
    objects, ids, placed_raw = [], set(), []
    for o in raw["objects"]:
        if not o.get("box"):
            continue
        placed_raw.append(o)
        xy, s, yaw, extra = place_object(o, cam, H, method, w, h, raw["objects"])
        shape = _scaled(_dims(o), s)
        oid = re.sub(r"[^a-z0-9_]", "_", o["id"].lower()) or "obj"
        while oid in ids:
            oid += "_b"
        ids.add(oid)
        sig, conf = sigma_for(o["box_source"], o.get("sam_conf"), extra.get("mask_iou"))
        m, fr = sorted(o["mass_kg_range"][:2]), sorted(o["friction_range"][:2])
        objects.append({
            "id": oid, "label": o["label"], "category": o["category"], "shape": shape,
            "pos": [float(xy[0]), float(xy[1])], "yaw": round(float(yaw), 4),
            "mass_kg": {"nominal": round(float(np.mean(m)), 3), "range": m},
            "friction": {"nominal": round(float(np.mean(fr)), 3), "range": fr},
            "fragile": bool(o["fragile"]), "rgba": _mean_color(img, o["box"]),
            "perception": {"box_px": [int(v) for v in o["box"]], "source": o["box_source"], "confidence": conf,
                           "sigma_m": sig, "method": method, "size_scale": round(s, 3),
                           "dims_muse": o["dims_m"], "center_norm_muse": o["center_norm"],
                           "sam_confidence": None if o.get("sam_conf") is None else round(o["sam_conf"], 4),
                           "mask_px": mask_outline(o), **extra}})
    if fidelity:
        apply_fidelity(objects, placed_raw, cam, img, w, h)
    if clamp:
        tidy(objects, table)
    dropped = [o["id"] for o in raw["objects"] if not o.get("box")]
    stem = Path(photo_file or raw["file"]).stem
    return {"name": name or stem, "source": f"perception ({method}) from {Path(raw['file']).name}",
            "privacy": privacy, "table": table, "objects": objects,
            "photo": {"file": photo_file or raw["file"], "width": w, "height": h,
                      "homography_px_to_table": np.round(H, 9).tolist(),
                      "paper_corners_px": np.round(paper_px, 2).tolist(), "paper_xy": list(paper_xy),
                      "paper_size": list(paper_size), "camera": cam.to_json(), "camera_info": cam_info},
            "perception_notes": {"not_segmented": dropped}}


def perceive(photo: Path, paper_px=None, paper_xy=DEFAULT_PAPER_XY, data: str = "private",
             method: str = "guarded", robot_side: str = "bottom", **kw) -> dict:
    """Photo -> scene. paper_px: 4 pixel corners (near-left, near-right, far-right, far-left; see calib.py),
    or None to auto-detect the sheet."""
    photo = Path(photo)
    w, h = Image.open(photo).size
    if paper_px is None:
        det = detect_paper(photo)
        if det is None:
            raise ValueError("could not find the A4 sheet; pass its 4 corners (near-left, near-right, far-right, far-left)")
        paper_px = order_corners(det["corners_px"], w, h, robot_side)
    raw = perceive_raw(photo, data=data)
    return estimate_scene(raw, paper_px, paper_xy, method=method, privacy=data, **kw)


# ------------------------------------------------------------------------------------------- overlay
def draw_overlay(photo, scene: dict) -> Image.Image:
    """Photo with SAM boxes + ids (solid green = concept / high confidence, dashed orange = point prompt /
    low confidence), each object's estimated footprint (projected back from the table), and the A4 outline."""
    im = (Image.open(photo) if not isinstance(photo, Image.Image) else photo).convert("RGB").copy()
    d = ImageDraw.Draw(im)
    W = im.size[0]
    lw = max(2, W // 400)
    try:
        font = ImageFont.load_default(size=max(14, W // 70))
    except TypeError:
        font = ImageFont.load_default()
    ph = scene.get("photo", {})
    if ph.get("paper_corners_px"):
        pc = [tuple(p) for p in ph["paper_corners_px"]]
        d.line(pc + [pc[0]], fill=(40, 140, 255), width=lw)
        for name, p in zip(("NL", "NR", "FR", "FL"), pc):
            d.ellipse([p[0] - 2 * lw, p[1] - 2 * lw, p[0] + 2 * lw, p[1] + 2 * lw], fill=(40, 140, 255))
            d.text((p[0] + 3 * lw, p[1] - 6 * lw), name, fill=(40, 140, 255), font=font)
    Hinv = np.linalg.inv(np.asarray(ph["homography_px_to_table"])) if ph.get("homography_px_to_table") else None
    for o in scene["objects"]:
        pr = o.get("perception") or {}
        box = pr.get("box_px")
        if not box:
            continue
        low = pr.get("confidence") == "low"
        col = (255, 150, 0) if low else (40, 220, 90)
        x1, y1, x2, y2 = box
        if low:  # dashed rectangle
            seg = 10
            for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
                L = np.hypot(b[0] - a[0], b[1] - a[1])
                for k in range(0, int(L), 2 * seg):
                    t0, t1 = k / L, min(1, (k + seg) / L)
                    d.line([(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                            (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)], fill=col, width=lw)
        else:
            d.rectangle(box, outline=col, width=lw)
        if pr.get("mask_px"):
            mp = [tuple(p) for p in pr["mask_px"]]
            if len(mp) >= 3:
                ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
                ImageDraw.Draw(ov).polygon(mp, fill=col + (60,))
                im.paste(Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB"))
                d = ImageDraw.Draw(im)
                d.line(mp + [mp[0]], fill=col, width=max(1, lw // 2 + 1))
        tag = o["id"] + (" (low)" if low else "")
        tb = d.textbbox((x1, y1), tag, font=font)
        ty = y1 - (tb[3] - tb[1]) - 4 if y1 > 20 else y2 + 2
        d.rectangle([x1, ty, x1 + tb[2] - tb[0] + 6, ty + tb[3] - tb[1] + 4], fill=col)
        d.text((x1 + 3, ty), tag, fill=(0, 0, 0), font=font)
        if Hinv is not None:  # footprint on the table (rotated rectangle for boxes, circle otherwise)
            ring = apply_h(Hinv, footprint_polygon(o, n=40))
            d.line([tuple(p) for p in ring] + [tuple(ring[0])], fill=(255, 255, 255), width=max(1, lw // 2))
            cpt = apply_h(Hinv, [o["pos"]])[0]
            d.ellipse([cpt[0] - 2 * lw, cpt[1] - 2 * lw, cpt[0] + 2 * lw, cpt[1] + 2 * lw], fill=col, outline=(0, 0, 0))
    return im


def detections_only(photo, data: str = "private") -> dict:
    """No paper -> no metric placement.  Muse Spark listing + SAM boxes only (for photos without an A4 sheet);
    returns a scene-shaped dict whose objects carry `perception` but no `pos`, usable with draw_overlay."""
    raw = perceive_raw(Path(photo), data=data)
    objs = [{"id": o["id"], "label": o["label"], "category": o["category"], "shape": _dims(o),
             "perception": {"box_px": [int(v) for v in o["box"]], "source": o["box_source"],
                            "confidence": sigma_for(o["box_source"], o.get("sam_conf"), None)[1],
                            "sam_confidence": o.get("sam_conf"), "mask_px": mask_outline(o)}}
            for o in raw["objects"] if o.get("box")]
    return {"name": Path(photo).stem, "privacy": data, "objects": objs,
            "photo": {"file": str(photo), "width": raw["width"], "height": raw["height"]},
            "perception_notes": {"not_segmented": [o["id"] for o in raw["objects"] if not o.get("box")],
                                 "listed": len(raw["objects"])}}


def perceive_photo(path, privacy: str = "private", corners=None, robot_side: str = "bottom",
                   paper_xy=DEFAULT_PAPER_XY, out_dir=None, name: str | None = None, method: str = "guarded"):
    """Backend entry point for an uploaded photo.

    privacy: "private" for user uploads (Standard mode), "public" only for synthetic/CC photos.
    corners: optional [[u, v] x 4] near-left, near-right, far-right, far-left; None -> auto-detect.
    Returns (scene dict, overlay PNG path).  Raises ValueError if the sheet is not found and no corners given.
    """
    path = Path(path)
    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = perceive(path, None if corners is None else np.asarray(corners, float), paper_xy, data=privacy,
                     method=method, robot_side=robot_side, name=name,
                     photo_file=str(path.relative_to(ROOT)) if path.is_absolute() and ROOT in path.parents else str(path))
    scene["photo"]["corners_source"] = "manual" if corners is not None else "auto"
    ov = out_dir / f"{name or path.stem}.overlay.png"
    draw_overlay(path, scene).save(ov)
    return scene, ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", type=Path)
    ap.add_argument("--paper-corners", default=None,
                    help='pixel corners "u,v u,v u,v u,v": near-left, near-right, far-right, far-left '
                         '(default: auto-detect the sheet)')
    ap.add_argument("--robot-side", default="bottom", choices=["bottom", "top", "left", "right"],
                    help="where the robot base is in the photo (for auto corner ordering)")
    ap.add_argument("--paper-xy", default=f"{DEFAULT_PAPER_XY[0]},{DEFAULT_PAPER_XY[1]}",
                    help="robot-frame xy of the paper's near-left corner")
    ap.add_argument("--privacy", default="private", choices=["private", "public"])
    ap.add_argument("--method", default="guarded")
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    px = None if a.paper_corners is None else np.array([[float(t) for t in c.split(",")] for c in a.paper_corners.split()])
    xy = tuple(float(t) for t in a.paper_xy.split(","))
    scene, ov = perceive_photo(a.photo, a.privacy, px, a.robot_side, xy, out_dir=a.out.parent, name=a.out.stem,
                               method=a.method)
    a.out.write_text(json.dumps(scene, ensure_ascii=False, indent=1))
    print(f"wrote {a.out} with {len(scene['objects'])} objects; overlay {ov}")


if __name__ == "__main__":
    main()
