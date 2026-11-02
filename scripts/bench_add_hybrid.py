"""Add the shipped hybrid planner's plan to every Bench case: plans/hybrid.json ($0, no API call).

The plan is the final attempt of out/<case>/report.json, the shipped product run (planner "hybrid": Muse Spark's goals
and plan hint, motion found by rehearsal/search_planner.py). Only cases whose report is a hybrid run on the same scene
as the case are updated; the LLM-only plans (muse_first / muse_after_log) are left as they are.

    .venv/bin/python scripts/bench_add_hybrid.py && .venv/bin/python bench/leaderboard.py --n 40
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _objs(spec):
    return [{k: v for k, v in o.items() if k != "perception"} for o in spec["objects"]]


def main() -> int:
    for d in sorted((ROOT / "bench" / "cases").iterdir()):
        cp, rp = d / "case.json", ROOT / "out" / d.name / "report.json"
        if not cp.is_file() or not rp.is_file():
            continue
        case, report = json.loads(cp.read_text()), json.loads(rp.read_text())
        if report.get("planner") != "hybrid" or not report.get("attempts"):
            print(f"{d.name}: skipped (report is not a hybrid run)")
            continue
        if _objs(json.loads((d / "scene.json").read_text())) != _objs(report["scene_spec"]):
            print(f"{d.name}: skipped (report scene differs from the case scene)")
            continue
        last = report["attempts"][-1]
        (d / "plans" / "hybrid.json").write_text(json.dumps(last["plan"], ensure_ascii=False, indent=1))
        case["files"]["plans"]["hybrid"] = "plans/hybrid.json"
        case.setdefault("plan_sources", {})["hybrid"] = {
            "report": f"out/{d.name}/report.json", "attempt": last.get("attempt"), "agent": report.get("agent"),
            "report_success": last.get("success"),
            "report_stress": {k: (report.get("stress_test") or {}).get(k) for k in ("successes", "n")}}
        cp.write_text(json.dumps(case, ensure_ascii=False, indent=1))
        print(f"{d.name}: plans/hybrid.json (attempt {last.get('attempt')}, {report.get('agent')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
