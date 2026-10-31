"""Photo -> scene JSON (same schema as scenes/*.json).  UNTESTED: needs MODEL_API_KEY.

1. Muse Spark lists objects, picks one SAM noun phrase per object, and estimates size/mass/friction.
2. SAM 3.1 returns pixel boxes for each noun phrase.
3. Pixel boxes -> table coordinates through a homography fitted on a reference sheet of paper
   (A4 / Letter) lying flat on the table. This assumes objects' footprints are on the table plane;
   the bottom-center of each box is used as the contact point.

    python -m rehearsal.perception photo.jpg --paper-corners "x1,y1 x2,y2 x3,y3 x4,y4" -o scenes/my_desk.json
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
from pathlib import Path

import numpy as np
from openai import OpenAI

BASE_URL = "https://api.meta.ai/v1"
A4 = (0.297, 0.210)

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

BOX_RE = re.compile(r"(?:^|<\d+f>|,)(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)")


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def list_objects(client: OpenAI, img: str, effort: str = "low") -> list[dict]:
    # "low" measured ~2x faster than the default on desk photos with the same object coverage.
    r = client.chat.completions.create(
        model="muse-spark-1.3",
        reasoning_effort=effort,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": img}}]}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "Objects", "schema": OBJECTS_SCHEMA, "strict": True}},
    )
    return json.loads(r.choices[0].message.content)["objects"]


def _sam(client: OpenAI, img: str, prompt: str) -> dict[int, tuple[int, int, int, int]]:
    """One SAM 3.1 call; returns {object id: (x1, y1, x2, y2)} in source pixels."""
    text = ""
    with client.responses.stream(model="sam-3.1", input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": img}]}]) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                text += event.delta
    return {int(m.group(1)): tuple(map(int, m.groups()[1:5])) for m in BOX_RE.finditer(text)}


def sam_boxes(client: OpenAI, img: str, phrase: str) -> list[tuple[int, int, int, int]]:
    """Concept prompt: every instance of a short noun phrase."""
    return list(_sam(client, img, phrase).values())


def sam_points(client: OpenAI, img: str, points: dict[int, tuple[int, int]], w: int, h: int):
    """Point prompts: one positive point per object id, all objects in one request."""
    prompt = "<0f>" + ",".join(f"{i}<|point;x={x};y={y};w={w};h={h}|>" for i, (x, y) in points.items())
    return _sam(client, img, prompt)


def segment_objects(client: OpenAI, img: str, objs: list[dict], w: int, h: int, pool) -> list[dict]:
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


def fit_homography(px: np.ndarray, world: np.ndarray) -> np.ndarray:
    A = []
    for (u, v), (x, y) in zip(px, world):
        A += [[u, v, 1, 0, 0, 0, -x * u, -x * v, -x], [0, 0, 0, u, v, 1, -y * u, -y * v, -y]]
    return np.linalg.svd(np.asarray(A))[2][-1].reshape(3, 3)


def to_table(H: np.ndarray, u: float, v: float) -> np.ndarray:
    p = H @ np.array([u, v, 1.0])
    return p[:2] / p[2]


def perceive(photo: Path, paper_px: np.ndarray, paper_xy: tuple[float, float]) -> dict:
    """paper_px: 4 pixel corners (near-left, near-right, far-right, far-left as seen from the robot).
    paper_xy: where the paper's near-left corner sits in the robot frame."""
    client = OpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"])
    img = data_url(photo)
    x0, y0 = paper_xy
    L, W = A4
    world = np.array([[x0, y0], [x0, y0 - W], [x0 + L, y0 - W], [x0 + L, y0]])
    H = fit_homography(paper_px, world)

    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    w, h = Image.open(photo).size
    with ThreadPoolExecutor(8) as pool:
        listed = segment_objects(client, img, list_objects(client, img), w, h, pool)
    objects = []
    for o in listed:
        if not o["box"]:
            print(f"SAM found nothing for {o['id']} ({o['category']}); skipping")
            continue
        x1, y1, x2, y2 = o["box"]
        xy = to_table(H, (x1 + x2) / 2, y2)  # bottom-center of the box touches the table
        d = o["dims_m"]
        r = d[0] if o["shape"] != "box" else min(d[0], d[1]) / 2
        xy = xy + np.array([r, 0.0])  # shift from the near edge to the footprint center (crude)
        shape = ({"type": "box", "size": d[:3]} if o["shape"] == "box"
                 else {"type": o["shape"], "radius": d[0], "height": d[1]})
        objects.append({
            "id": o["id"], "label": o["label"], "shape": shape, "pos": [float(xy[0]), float(xy[1])],
            "mass_kg": {"nominal": float(np.mean(o["mass_kg_range"])), "range": o["mass_kg_range"]},
            "friction": {"nominal": float(np.mean(o["friction_range"])), "range": o["friction_range"]},
            "fragile": o["fragile"], "sam_box_px": [x1, y1, x2, y2]})
    return {"name": photo.stem, "source": f"perception from {photo.name}",
            "table": {"center": [0.45, 0.0], "size": [1.3, 1.2]}, "objects": objects}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", type=Path)
    ap.add_argument("--paper-corners", required=True,
                    help='pixel corners "u,v u,v u,v u,v": near-left, near-right, far-right, far-left')
    ap.add_argument("--paper-xy", default="0.30,0.10", help="robot-frame xy of the paper's near-left corner")
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    px = np.array([[float(t) for t in c.split(",")] for c in a.paper_corners.split()])
    xy = tuple(float(t) for t in a.paper_xy.split(","))
    scene = perceive(a.photo, px, xy)
    a.out.write_text(json.dumps(scene, ensure_ascii=False, indent=1))
    print(f"wrote {a.out} with {len(scene['objects'])} objects")


if __name__ == "__main__":
    main()
