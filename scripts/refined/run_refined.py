"""Refined hero scenes: real tasks on the AI-generated tables through the real pipeline.

    intent : Muse Spark compile (rehearsal.agents.MuseAgent, data="public" = Contributor mode: the tables are
             AI-generated synthetic images) + rehearsal.intent_gate (physics) -- same calls as run_user_desk.compile_one
    plan   : scripts/user_desk/run_user_desk.plan_and_run (on/inside goals -> the shipped search_planner.search with the
             exec_tables budgets; `near` goals -> the labelled near-placement DEMO EXTENSION of run_user_desk)
    verify : rehearsal.executor.run_plan, goals + implicit undisturbed(everything else)

Outputs: footage/tables_sim/hero/refined/_runs/{intent,cache}/<key>.json (+ _traj.npz, _scene.json).
Nothing under footage/tables_sim/hero/exec/ or scenes/ is written.

Run (rehearsal-mvp/):  set -a; source .env; set +a
  .venv/bin/python scripts/refined/run_refined.py intent KEY [--answer "..."]   # compile (one paid call, cached)
  .venv/bin/python scripts/refined/run_refined.py plan KEY [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "user_desk"))
import run_user_desk as U  # noqa: E402

REF = ROOT.parent / "footage" / "tables_sim" / "hero" / "refined"
RUNS = REF / "_runs"
U.CACHE = RUNS / "cache"
U.INTENT = RUNS / "intent"
FIXED09 = REF / "09_eggs" / "twin" / "t6_09_fixed.json"

# key -> (scene file, instruction).  Keys ending in _a are the follow-up after a clarifying question.
TASKS = {
    "09_eggs_fixed": (FIXED09, "Put the eggs into the mixing bowl."),
}


def load_tasks():
    p = RUNS / "tasks.json"
    if p.exists():
        for k, v in json.loads(p.read_text()).items():
            TASKS[k] = (Path(v[0]) if v[0].startswith("/") else ROOT / v[0], v[1])


def use_scene(key):
    if key not in TASKS and key.endswith("_a"):      # the follow-up after a question runs on the same table
        TASKS[key] = TASKS[key[:-2]]
    U.SCENE = TASKS[key][0]
    return json.loads(U.SCENE.read_text())


def compile_one(key, instruction, sc, data="public", answer_of=None, answer=None):
    from rehearsal import intent_gate
    from rehearsal import meta_client as M
    from rehearsal.agents import MuseAgent, scene_summary
    from rehearsal.executor import reachability_report
    text = instruction
    if answer_of is not None:      # follow-up to a clarifying question: the instruction, the question, the answer
        q = json.loads((U.INTENT / f"{answer_of}.json").read_text())
        text = (f"{q['instruction']}\n(You asked: \"{q['intent']['question']}\" The user answered: \"{answer}\")")
    before = M.status()["spent"]
    summary = scene_summary(sc, reachability_report(sc))
    t0 = time.time()
    intent = MuseAgent(data=data).compile_intent(text, summary)
    wall = time.time() - t0
    gate = intent_gate.check(intent.get("goals") or [], sc, physics=True) if intent["status"] == "goals" else None
    after = M.status()["spent"]
    rec = {"key": key, "instruction": instruction, "compiled_text": text, "intent": intent, "gate": gate,
           "cost_usd": round(after - before, 5), "compile_wall_s": round(wall, 1), "data": data,
           "scene_file": str(U.SCENE)}
    if answer_of is not None:
        rec.update(answer_of=answer_of, answer=answer)
    U.INTENT.mkdir(parents=True, exist_ok=True)
    (U.INTENT / f"{key}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(key, intent["status"], json.dumps(intent.get("goals")), "Q/R:", intent.get("question") or intent.get("reason"),
          "| assumptions:", intent.get("assumptions"), "| gate", (gate or {}).get("verdict"),
          [f["code"] for f in (gate or {}).get("flags", [])], f"${rec['cost_usd']}", flush=True)
    return rec


def main():
    load_tasks()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["intent", "plan", "copy09"])
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--answer", help="intent: answer to the clarifying question of KEY (writes KEY_a)")
    ap.add_argument("--time-budget", type=float, default=900.0)
    a = ap.parse_args()
    if a.cmd == "copy09":          # the shipped compiled goals of table 09 (no model call)
        comp = json.loads((ROOT / "out" / "tables_v6" / "09.json").read_text())
        U.INTENT.mkdir(parents=True, exist_ok=True)
        (U.INTENT / "09_eggs_fixed.json").write_text(json.dumps(
            {"key": "09_eggs_fixed", "instruction": comp["instruction"], "intent": comp["intent"], "gate": comp.get("gate"),
             "cost_usd": 0.0, "copied_from": "out/tables_v6/09.json (the shipped compile; no new model call)",
             "scene_file": str(FIXED09)}, indent=1, default=float))
        return
    for k in a.keys:
        sc = use_scene(k)
        if a.cmd == "intent":
            if a.answer:
                ka = k + "_a"
                TASKS[ka] = TASKS[k]
                compile_one(ka, TASKS[k][1], sc, answer_of=k, answer=a.answer)
            elif (U.INTENT / f"{k}.json").exists() and not a.force:
                print(k, "cached:", json.loads((U.INTENT / f"{k}.json").read_text())["intent"]["status"])
            else:
                compile_one(k, TASKS[k][1], sc)
        else:
            rec = U.plan_and_run(k, a.force, time_budget=a.time_budget)
            rec["scene_file"] = str(U.SCENE)
            rec["demo_extension"] = "near-placement" in str(rec.get("planner", "")) or U.ROTATE
            (U.CACHE / f"{k}_run.json").write_text(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    main()
