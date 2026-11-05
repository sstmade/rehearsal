"""After assemble: stills, run JSON copy and README.md for one refined scene directory.

  .venv/bin/python scripts/refined/finish_scene.py DIR KEY INTENT_KEY "notes line" ["notes line" ...]
"""
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

R = Path(__file__).resolve().parents[3] / "footage/tables_sim/hero/refined"
d, key, ik = R / sys.argv[1], sys.argv[2], sys.argv[3]
notes = sys.argv[4:]
exp = d / "_export"
shots = json.loads((exp / f"{key}_shots.json").read_text())["frames"]
meta = json.loads((exp / f"{key}_traj_meta.json").read_text())
run = json.loads((R / "_runs/cache" / f"{key}_run.json").read_text())
it = json.loads((R / "_runs/intent" / f"{ik}.json").read_text())
q = None
if it.get("answer_of"):
    q0 = json.loads((R / "_runs/intent" / f"{it['answer_of']}.json").read_text())
    q = q0["intent"].get("question") or q0["intent"].get("reason")
shutil.copy(R / "_runs/cache" / f"{key}_run.json", d / f"{key}_run.json")
shutil.copy(R / "_runs/intent" / f"{ik}.json", d / f"{ik}_intent.json")
st = d / "stills"
st.mkdir(exist_ok=True)
picked = {}
for i, s in enumerate(shots):                       # middle frame of every shot segment
    picked.setdefault(s["shot"], []).append(i)
n = 0
for shot, idx in picked.items():
    if shot in ("move",):
        continue
    for j in sorted(set([idx[len(idx) // 2]] + ([idx[-1]] if shot in ("hold", "freeze") else []))):
        f = d / "_frames" / f"f_{j:04d}.png"
        if f.exists():
            Image.open(f).convert("RGB").save(st / f"{n:02d}_{shot}_{j:04d}.jpg", quality=92)
            n += 1
ok = bool(run.get("planned") and run.get("success"))
gl = [f"- `{g['pred']}({', '.join(g['args'])})`: {'ok' if g['ok'] else 'FAIL'} ({g.get('why', '')})"
      for g in run.get("goals", [])]
s = (run.get("search") or {})
lines = [f"# {d.name}", "",
         f"**Instruction:** \"{it['instruction']}\"", ""]
if q:
    lines += [f"**Rehearsal asked:** \"{q}\"", "", f"**Answer:** \"{it.get('answer')}\"", ""]
lines += [f"**Verdict:** {'PLAN VERIFIED · SENT' if ok else 'REHEARSAL DOESN’T SEND IT (no safe plan, nothing sent)'}",
          "", f"Search: {s.get('reason') or 'solution found'}; stats {json.dumps(s.get('stats', {}))}", "",
          f"Demo extension (near-placement): {'YES' if run.get('demo_extension') else 'no'}", "",
          "## Goals (checked in physics)", *gl, "",
          "## Files",
          f"- `{d.name}.mp4`: clean clip (photo, then crossfade into the Cycles twin, then the rehearsal)",
          f"- `{d.name}_labeled.mp4`: the same with the HUD chips (instruction / question / answer / verdict)",
          "- `photo_vs_render.jpg`: the AI-generated photo | the Cycles twin from the photo camera | a 50/50 blend",
          "- `stills/`: one still per shot",
          f"- `{key}_run.json`, `{ik}_intent.json`: the run and compiled-intent records", "",
          "## Notes",
          "- Visual only: Cycles re-render of the recorded MuJoCo trajectory (body poses per frame). Re-modelled",
          "  objects are sized to the twin's primitives; physics results are unchanged by the look.",
          *[f"- {x}" for x in notes]]
(d / "README.md").write_text("\n".join(lines) + "\n")
print("finished", d, n, "stills")
