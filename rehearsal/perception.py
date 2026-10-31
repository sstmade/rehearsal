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

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAPER_XY = (0.30, 0.10)     # robot-frame xy of the paper's near-left corner (see calib.py)
DEFAULT_TABLE = {"center": [0.45, 0.0], "size": [1.3, 1.2]}
# Per-axis position sigma (m) for the stress test, from the synthetic evaluation (docs/perception.md):
# sigma = median radial error / 1.18 (Rayleigh), rounded up.
SIGMA_M = {"high": 0.010, "low": 0.020}

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


def _sam(client, img: str, prompt: str) -> dict[int, tuple[int, int, int, int]]:
    """One SAM 3.1 call; returns {object id: (x1, y1, x2, y2)} in source pixels.
    (The `<|mask;...|>` token after each box is a length-prefixed base-85 binary blob in an undocumented,
    compressed format — not decoded; see docs/perception.md.)"""
    from .meta_client import sam
    text = sam(prompt, img, caller="perception.sam")
    return {int(m.group(1)): tuple(map(int, m.groups()[1:5])) for m in BOX_RE.finditer(text)}


def sam_boxes(client, img: str, phrase: str) -> list[tuple[int, int, int, int]]:
    """Concept prompt: every instance of a short noun phrase."""
    return list(_sam(client, img, phrase).values())


def sam_points(client, img: str, points: dict[int, tuple[int, int]], w: int, h: int):
    """Point prompts: one positive point per object id, all objects in one request."""
    prompt = "<0f>" + ",".join(f"{i}<|point;x={x};y={y};w={w};h={h}|>" for i, (x, y) in points.items())
    return _sam(client, img, prompt)


def segment_objects(client, img: str, objs: list[dict], w: int, h: int, pool) -> list[dict]:
    """Assign a SAM box to each object Muse Spark listed.

    1. One concept request per category returns every instance of that kind.
    2. Each object takes the unclaimed instance box nearest to its Muse Spark center
       (box containing the point preferred).
    3. Objects still without a box are segmented by a point prompt at their center.
    Adds "box" (pixels) and "box_source" ("concept" | "point" | None) to each object.
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
        members, boxes = [o for o in objs if o["category"] == cat], list(found[cat])
        pairs = sorted((cost(o, b), i, j) for i, o in enumerate(members) for j, b in enumerate(boxes))
        used_o, used_b = set(), set()
        for c, i, j in pairs:
            if i in used_o or j in used_b or c > 0.15:
                continue
            members[i]["box"], members[i]["box_source"] = boxes[j], "concept"
            used_o.add(i); used_b.add(j)

    missing = [o for o in objs if "box" not in o]
    if missing:
        pts = {k: tuple(int(v) for v in centers[o["id"]]) for k, o in enumerate(missing, 1)}
        got = sam_points(client, img, pts, w, h)
        for k, o in enumerate(missing, 1):
            o["box"], o["box_source"] = got.get(k), "point" if k in got else None
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
        return {"type": "box", "size": [float(np.clip(d[0], 0.02, 0.5)), float(np.clip(d[1], 0.02, 0.5)),
                                        float(np.clip(d[2], 0.005, 0.4))]}
    return {"type": o["shape"], "radius": float(np.clip(d[0], 0.01, 0.2)), "height": float(np.clip(d[1], 0.01, 0.4))}


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
    return np.r_[np.c_[ring, np.zeros(n)], np.c_[ring, np.full(n, hgt)]]


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
                                                                     _height(shape) / 2)), 0.0])
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


def place_object(o: dict, cam, H, method: str, w: int, h: int):
    """-> (xy, scale) for one listed object with a SAM box."""
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


def _mean_color(img: Image.Image | None, box) -> list[float]:
    if img is None:
        return [0.7, 0.7, 0.7, 1.0]
    x1, y1, x2, y2 = box
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 0.15, (y2 - y1) * 0.15
    a = np.asarray(img.crop((int(cx - hw), int(cy - hh), int(cx + hw) + 1, int(cy + hh) + 1)).convert("RGB"), float)
    c = np.median(a.reshape(-1, 3), axis=0) / 255 if a.size else np.array([0.7, 0.7, 0.7])
    return [round(float(v), 3) for v in c] + [1.0]


def _footprint_r(shape) -> float:
    if shape["type"] == "box":
        return 0.5 * float(np.hypot(*shape["size"][:2]))
    return shape["radius"]


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


def estimate_scene(raw: dict, paper_px, paper_xy=DEFAULT_PAPER_XY, method: str = "box_fit",
                   paper_size=A4, table: dict | None = None, clamp: bool = True, privacy: str = "private",
                   name: str | None = None, photo_file: str | None = None, f: float | None = None) -> dict:
    """Pure geometry (no API): raw perception + paper corners -> scene JSON."""
    w, h = raw["width"], raw["height"]
    paper_px = np.asarray(paper_px, float)
    H = fit_homography(paper_px, paper_world(paper_xy, paper_size))
    cam, cam_info = camera_from_homography(H, w, h, f=f)
    table = table or DEFAULT_TABLE
    try:
        img = Image.open(raw["file"]).convert("RGB")
    except Exception:
        img = None
    objects, ids = [], set()
    for o in raw["objects"]:
        if not o.get("box"):
            continue
        xy, s = place_object(o, cam, H, method, w, h)
        shape = _scaled(_dims(o), s)
        oid = re.sub(r"[^a-z0-9_]", "_", o["id"].lower()) or "obj"
        while oid in ids:
            oid += "_b"
        ids.add(oid)
        conf = "high" if o["box_source"] == "concept" else "low"
        m, fr = sorted(o["mass_kg_range"][:2]), sorted(o["friction_range"][:2])
        objects.append({
            "id": oid, "label": o["label"], "category": o["category"], "shape": shape,
            "pos": [float(xy[0]), float(xy[1])], "yaw": 0.0,
            "mass_kg": {"nominal": round(float(np.mean(m)), 3), "range": m},
            "friction": {"nominal": round(float(np.mean(fr)), 3), "range": fr},
            "fragile": bool(o["fragile"]), "rgba": _mean_color(img, o["box"]),
            "perception": {"box_px": [int(v) for v in o["box"]], "source": o["box_source"], "confidence": conf,
                           "sigma_m": SIGMA_M[conf], "method": method, "size_scale": round(s, 3),
                           "dims_muse": o["dims_m"], "center_norm_muse": o["center_norm"]}})
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
             method: str = "box_fit", robot_side: str = "bottom", **kw) -> dict:
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
        tag = o["id"] + (" (low)" if low else "")
        tb = d.textbbox((x1, y1), tag, font=font)
        ty = y1 - (tb[3] - tb[1]) - 4 if y1 > 20 else y2 + 2
        d.rectangle([x1, ty, x1 + tb[2] - tb[0] + 6, ty + tb[3] - tb[1] + 4], fill=col)
        d.text((x1 + 3, ty), tag, fill=(0, 0, 0), font=font)
        if Hinv is not None:  # footprint circle on the table
            r = _footprint_r(o["shape"]) if o["shape"]["type"] != "box" else min(o["shape"]["size"][:2]) / 2
            a = np.linspace(0, 2 * np.pi, 40)
            ring = apply_h(Hinv, np.c_[o["pos"][0] + r * np.cos(a), o["pos"][1] + r * np.sin(a)])
            d.line([tuple(p) for p in ring] + [tuple(ring[0])], fill=col, width=max(1, lw // 2))
            c = apply_h(Hinv, [o["pos"]])[0]
            d.ellipse([c[0] - 2 * lw, c[1] - 2 * lw, c[0] + 2 * lw, c[1] + 2 * lw], fill=col, outline=(0, 0, 0))
    return im


def detections_only(photo, data: str = "private") -> dict:
    """No paper -> no metric placement.  Muse Spark listing + SAM boxes only (for photos without an A4 sheet);
    returns a scene-shaped dict whose objects carry `perception` but no `pos`, usable with draw_overlay."""
    raw = perceive_raw(Path(photo), data=data)
    objs = [{"id": o["id"], "label": o["label"], "category": o["category"], "shape": _dims(o),
             "perception": {"box_px": [int(v) for v in o["box"]], "source": o["box_source"],
                            "confidence": "high" if o["box_source"] == "concept" else "low"}}
            for o in raw["objects"] if o.get("box")]
    return {"name": Path(photo).stem, "privacy": data, "objects": objs,
            "photo": {"file": str(photo), "width": raw["width"], "height": raw["height"]},
            "perception_notes": {"not_segmented": [o["id"] for o in raw["objects"] if not o.get("box")],
                                 "listed": len(raw["objects"])}}


def perceive_photo(path, privacy: str = "private", corners=None, robot_side: str = "bottom",
                   paper_xy=DEFAULT_PAPER_XY, out_dir=None, name: str | None = None, method: str = "box_fit"):
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
    ap.add_argument("--method", default="box_fit")
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
