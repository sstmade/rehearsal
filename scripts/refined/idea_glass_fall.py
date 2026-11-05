"""IDEA beat "one wrong move, and a glass is on the floor": a deliberately naive candidate on AI test table t6_19,
simulated with the real executor and physics (no faking), recorded like every refined run.

Candidate (hand-written, NOT from the planner or Muse): carry the TV remote at a low height (hand 0.10 m) straight
through the tall tumbler's position toward the far edge, then set it down.  Checks: the goal near(remote, cloth) plus
the implicit undisturbed(everything else) -> the tumbler's fall is what rejects it.
  .venv/bin/python scripts/refined/idea_glass_fall.py
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "user_desk"))
import run_user_desk as U  # noqa: E402
REF = ROOT.parent / "footage/tables_sim/hero/refined"
U.CACHE = REF / "_runs/cache"
U.INTENT = REF / "_runs/intent"
U.SCENE = ROOT / "scenes/t6_19.json"
KEY = "idea_glass_fall"
U.INTENT.mkdir(parents=True, exist_ok=True)
(U.INTENT / f"{KEY}.json").write_text(json.dumps({
    "key": KEY, "instruction": "Move the remote next to the cloth.",
    "intent": {"status": "goals", "goals": [{"pred": "near", "args": ["remote_control_1", "cloth_1"]}]},
    "note": "hand-written goal for the IDEA beat (no model call); the candidate below is deliberately naive"}, indent=1))
cand = [{"obj": "remote_control_1", "dest": "cloth_1", "relation": "near@0.47,0.02", "transit_height": 0.10,
         "via": [[0.40, -0.22], [0.66, -0.22]]}]
rec = U.plan_and_run(KEY, force=True, replay=cand)
rec["idea_note"] = ("Deliberately naive candidate (hand-written for the film's 'one wrong move' line): the remote is "
                    "carried 0.10 m above the table straight through the tumbler toward the far edge. Real executor + "
                    "MuJoCo physics; nothing scripted about the tumbler.")
(U.CACHE / f"{KEY}_run.json").write_text(json.dumps(rec, indent=1, default=float))
fp = rec.get("final_poses", {}).get("tumbler_1")
print("success", rec["success"], "error", rec.get("execution_error"), "tumbler final", fp, "disturbed", rec.get("disturbed"))
