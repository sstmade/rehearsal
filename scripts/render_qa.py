"""Muse Spark visual QA critic: AI-generated table photo vs our render from the same camera -> per-object fidelity issues.

For each table NN it sends Muse Spark (Contributor mode: the photos are AI-generated, the renders are ours) two images:
  1. the original photo   photos/ai_gen/tables_v6/table_NN_*.png
  2. the latest render through the photo's calibrated camera, first found of
       ../footage/tables_sim/dark/   (dark studio renders, any file whose name starts with NN / t6_NN; *photo* preferred)
       footage/tables_sim/dark/      (same, inside rehearsal-mvp)
       out/tables_v6/render/NN_photo.png   (render_tables.py, photo camera)
       ../footage/tables_sim/cards/NN.png  (card: photo | 3/4 view; last resort)
plus the scene's object list (ids + labels), and asks for a strict-JSON list of fidelity problems, one entry per
object problem.  Background / room style and the robot are ignored on purpose (stylised).

Writes out/tables_v6/qa/t6_NN.json and out/tables_v6/qa/QA_SUMMARY.md.  All calls go through rehearsal.meta_client
(cached, ledgered).

    .venv/bin/python scripts/render_qa.py 17 03 13     # pilot
    .venv/bin/python scripts/render_qa.py              # all 50
    .venv/bin/python scripts/render_qa.py --summary    # rebuild QA_SUMMARY.md only
"""
from __future__ import annotations

import base64
import io
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rehearsal.meta_client import chat_json  # noqa: E402

PH = ROOT / "photos" / "ai_gen" / "tables_v6"
OUT = ROOT / "out" / "tables_v6" / "qa"
ISSUES = ["wrong_colour", "wrong_shape", "missing", "extra", "texture_bleed", "wrong_scale", "wrong_position"]
MAX_SIDE = 1280

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["overall_fidelity", "summary", "issues"],
    "properties": {
        "overall_fidelity": {"type": "integer", "description": "1 (unrecognisable) .. 10 (indistinguishable layout/objects)"},
        "summary": {"type": "string"},
        "issues": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["object_id", "label", "issue", "severity", "fix"],
            "properties": {
                "object_id": {"type": "string", "description": "id from the object list, or 'none' for an object "
                                                               "that is in the render but not in the list"},
                "label": {"type": "string"},
                "issue": {"type": "string", "enum": ISSUES},
                "severity": {"type": "integer", "enum": [1, 2, 3]},
                "fix": {"type": "string"}}}}}}

PROMPT = """You are a strict visual QA critic for a robotics digital twin.
Image 1 is a photo of a real-looking table. Image 2 is our 3D simulation render of the same table, seen from the same camera.
The objects in the twin (id: label, approx. centre in the photo as x%,y%):
{objects}

Compare ONLY the tabletop and the objects on it. Deliberately IGNORE: the background, walls, floor, room and lighting style
(the render is intentionally stylised), any robot arm or gripper, and render noise/aliasing.

List every fidelity problem, one entry per problem per object:
- wrong_colour: object colour/material clearly differs from the photo
- wrong_shape: primitive shape is wrong (e.g. cylinder for a flat cloth, box for a bowl, missing handle/rim)
- missing: an object visible in the photo is absent in the render (use the photo's name; object_id from the list if it is listed, else 'none')
- extra: an object in the render that is not in the photo
- texture_bleed: photo texture of a neighbour, the table, or the background is painted onto the object (wrong patches, streaks, halos)
- wrong_scale: clearly too big/small relative to the photo
- wrong_position: clearly shifted, rotated, floating, sunk into the table, or overlapping something it should not
Severity: 1 = minor, noticeable only on close look; 2 = clearly visible; 3 = breaks recognition of the object.
fix: one short concrete suggestion (e.g. "use a flat textured sheet instead of a box", "sample texture only inside the mug's own mask").
Only report real, visible differences; do not invent problems. If the table itself is wrong (shape/colour/texture), report it with object_id "table".
overall_fidelity: 1..10 for the tabletop + objects only."""


def find_render(nn: str) -> Path | None:
    for d in (ROOT.parent / "footage" / "tables_sim" / "dark", ROOT / "footage" / "tables_sim" / "dark"):
        if d.is_dir():
            c = sorted(p for p in d.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")
                       and (p.name.startswith(nn) or p.name.startswith(f"t6_{nn}")))
            if c:
                photo_cam = [p for p in c if "photo" in p.name]
                return (photo_cam or c)[0]
    for p in (ROOT / "out" / "tables_v6" / "render" / f"{nn}_photo.png",
              ROOT.parent / "footage" / "tables_sim" / "cards" / f"{nn}.png"):
        if p.exists():
            return p
    return None


def jpeg_url(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_scene(nn: str) -> dict:
    for p in (ROOT / "scenes" / f"t6_{nn}.json", ROOT / "out" / "tables_v6" / f"t6_{nn}.json"):
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(nn)


def qa_table(nn: str) -> dict:
    sc = load_scene(nn)
    photo = next(PH.glob(f"table_{nn}_*.png"))
    render = find_render(nn)
    if render is None:
        return {"table": f"t6_{nn}", "error": "no render found"}
    W, H = sc["photo"]["width"], sc["photo"]["height"]
    lines = []
    for o in sc.get("objects", []):
        b = (o.get("perception") or {}).get("box_px")
        pos = f" ({100 * (b[0] + b[2]) / 2 / W:.0f}%,{100 * (b[1] + b[3]) / 2 / H:.0f}%)" if b else ""
        lines.append(f"- {o['id']}: {o.get('label', '')}{pos}")
    msg = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT.format(objects="\n".join(lines) or "(none: empty table)")},
        {"type": "text", "text": "Image 1 (photo):"},
        {"type": "image_url", "image_url": {"url": jpeg_url(photo)}},
        {"type": "text", "text": "Image 2 (render, same camera):"},
        {"type": "image_url", "image_url": {"url": jpeg_url(render)}}]}]
    res = chat_json(msg, SCHEMA, "RenderQA", caller="render_qa", data="public", effort="medium")
    labels = {o["id"]: o.get("label", "") for o in sc.get("objects", [])}
    for it in res["issues"]:
        it["severity"] = int(it["severity"])
        if it["object_id"] in labels and not it["label"]:
            it["label"] = labels[it["object_id"]]
    res["issues"].sort(key=lambda i: -i["severity"])
    rec = {"table": f"t6_{nn}", "photo": str(photo.relative_to(ROOT)),
           "render": str(render.relative_to(ROOT.parent)), "n_objects": len(labels),
           "model": "muse-spark-1.3-contributor", **res}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"t6_{nn}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


def summary():
    recs = [json.loads(p.read_text()) for p in sorted(OUT.glob("t6_*.json"))]
    recs = [r for r in recs if "issues" in r]
    for r in recs:
        r["_score"] = sum(i["severity"] for i in r["issues"])
    kinds = Counter(i["issue"] for r in recs for i in r["issues"])
    kinds_w = Counter()
    for r in recs:
        for i in r["issues"]:
            kinds_w[i["issue"]] += i["severity"]
    sev = Counter(i["severity"] for r in recs for i in r["issues"])
    renders = Counter(str(Path(r["render"]).parent) for r in recs)
    L = ["# Render QA summary (Muse Spark critic)", "",
         f"{len(recs)} tables, {sum(len(r['issues']) for r in recs)} issues "
         f"(severity 3: {sev[3]}, 2: {sev[2]}, 1: {sev[1]}). Generated by `scripts/render_qa.py`.", "",
         "Renders compared: " + ", ".join(f"`{k}` ({v})" for k, v in renders.items()) + ". "
         "Background, room style and robot ignored by prompt.", "",
         "## Most common issue types", "", "| issue | count | severity-weighted |", "|---|---:|---:|"]
    L += [f"| {k} | {kinds[k]} | {kinds_w[k]} |" for k, _ in kinds_w.most_common()]
    L += ["", "## Worst tables (sum of severities)", "",
          "| table | fidelity (1-10) | issues | sev. sum | worst problems |", "|---|---:|---:|---:|---|"]
    for r in sorted(recs, key=lambda r: (-r["_score"], r["overall_fidelity"])):
        top = "; ".join(f"{i['label'] or i['object_id']}: {i['issue']} ({i['severity']})" for i in r["issues"][:3])
        L.append(f"| {r['table']} | {r['overall_fidelity']} | {len(r['issues'])} | {r['_score']} | {top} |")
    L += ["", "## All severity-3 findings", ""]
    for r in recs:
        for i in r["issues"]:
            if i["severity"] == 3:
                L.append(f"- **{r['table']}** `{i['object_id']}` {i['label']}: {i['issue']} — {i['fix']}")
    (OUT / "QA_SUMMARY.md").write_text("\n".join(L) + "\n")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--summary" not in sys.argv:
        nns = args or [f"{i:02d}" for i in range(1, 51)]
        with ThreadPoolExecutor(6) as pool:
            for r in pool.map(lambda n: _safe(n), nns):
                print(r.get("table"), r.get("overall_fidelity"), len(r.get("issues", [])), r.get("error", ""),
                      flush=True)
    summary()


def _safe(nn):
    try:
        return qa_table(nn)
    except Exception as e:
        return {"table": f"t6_{nn}", "error": f"{type(e).__name__}: {str(e)[:200]}"}


if __name__ == "__main__":
    main()
