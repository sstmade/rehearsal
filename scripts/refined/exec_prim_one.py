"""Primitive (collision shapes, flat matte) drop-in of one exec_dark clip: same cached trajectory, camera, timing,
frame count and catch flash (exec_tables.render_clip with DRESS_STYLE=primitive, pixel-aligned with dark).
No re-simulation.  Writes hero/exec_dark/NN_primitive.mp4 (+ _labeled, _still).
  DRESS_STYLE=primitive EXEC_HQ=1 .venv/bin/python scripts/refined/exec_prim_one.py NN
"""
import json, os, shutil, sys
from pathlib import Path
os.environ["DRESS_STYLE"] = "primitive"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import exec_tables as ET  # noqa: E402

nn = sys.argv[1]
DARK = ET.OUT.parent / "exec_dark"
WORK = ET.OUT / f"_{nn}_drop_in"
WORK.mkdir(parents=True, exist_ok=True)
rec = json.loads((DARK / f"{nn}.json").read_text())
ET.OUT = WORK
r = ET.render_clip(nn, rec)
for a, b in ((f"{nn}.mp4", f"{nn}_primitive.mp4"), (f"{nn}_labeled.mp4", f"{nn}_primitive_labeled.mp4"),
             (f"{nn}_still.png", f"{nn}_primitive_still.png")):
    shutil.copy2(WORK / a, DARK / b)
print("wrote", DARK / f"{nn}_primitive.mp4", r)
