"""eval_v7: replace the empty table 34 with a new AI-generated table photo, through the SAME pipeline as the 50
(scripts/batch_tables.py: Muse Spark listing + SAM 3.1 masks in Contributor mode, tabletop outline read by eye,
auto camera, typical-size scale, photo texture, Muse Spark compile + intent gate).  Only the output paths differ:

  photo  : photos/ai_gen/tables_v7/table_51_oak-desk-laptop.png  (= photos/ai_gen/desk_oak_laptop.png, an
           AI-generated test photo listed in photos/ai_gen/SOURCES.md as "not used yet")
  scene  : footage/tables_sim/eval_v7/scenes/t6_51.json
  intent : footage/tables_sim/eval_v7/intent/51.json
  texture/overlay: photos/ai_gen/tables_v7/t6_51.*

Run (rehearsal-mvp/):  set -a; source .env; set +a; .venv/bin/python scripts/eval_v7/new_table.py
"""
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import batch_tables as B  # noqa: E402
from rehearsal.perception import draw_overlay, perceive_raw  # noqa: E402

EV = ROOT.parent / "footage" / "tables_sim" / "eval_v7"
NN = "51"
B.PH = ROOT / "photos" / "ai_gen" / "tables_v7"
B.OUT = EV / "intent"
B.TASKS[NN] = "Put the glasses on the notebook."
B.TOPS[NN] = [(1, 76), (99, 76), (86, 33), (14, 33)]     # near-left, near-right, far-right, far-left (by eye, 5% grid)


def main():
    photo = next(B.PH.glob(f"table_{NN}_*.png"))
    rel = str(photo.relative_to(ROOT))
    w, h = Image.open(photo).size
    raw = perceive_raw(photo, data="public")
    f, pitch, roll, kind, q, M = B.calibrate_top(NN, w, h)
    sc, cam = B.build(NN, raw, M, f, pitch, roll, kind, q, w, h, rel)
    tex, mean = B.table_texture(photo, cam, sc["table"], M, raw)
    tex_path = B.PH / f"t6_{NN}.table_tex.png"
    tex.save(tex_path)
    sc["title"] = "Oak desk laptop (AI-generated test photo)"
    sc["provenance"] = "AI-generated test photo · not a real capture"
    sc["ai_generated"] = True
    sc["demo_instruction"] = B.TASKS[NN]
    sc["visual"] = {"table_texture": str(tex_path.relative_to(ROOT)), "table_rgba": mean, "table_top": kind}
    sc["source"] = (f"AI-generated test photo {photo.name} (photos/ai_gen/tables_v7; same image as "
                    "photos/ai_gen/desk_oak_laptop.png). Same pipeline as scripts/batch_tables.py via "
                    "scripts/eval_v7/new_table.py. No hand-edited object positions.")
    (EV / "scenes").mkdir(parents=True, exist_ok=True)
    (EV / "intent").mkdir(parents=True, exist_ok=True)
    (EV / "scenes" / f"t6_{NN}.json").write_text(json.dumps(sc, ensure_ascii=False, indent=1))
    draw_overlay(photo, sc).save(B.PH / f"t6_{NN}.overlay.png")
    res = B.compile_task(sc, B.TASKS[NN])
    (EV / "intent" / f"{NN}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(NN, kind, "objs", [o["id"] for o in sc["objects"]], "dropped", sc["perception_notes"]["dropped_off_table"],
          "status", res["intent"].get("status"), "goals", res["intent"].get("goals"), "gate", res["gate"].get("verdict"))


if __name__ == "__main__":
    main()
