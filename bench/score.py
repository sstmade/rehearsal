"""Score a plan on a Rehearsal Bench case: physics + goal predicates + stress test on fixed seeds.

    .venv/bin/python bench/score.py bench/cases/kitchen_01 my_plan.json            # n = 40, seeds 1000-1039
    .venv/bin/python bench/score.py bench/cases/kitchen_01 my_plan.json --n 10 --json card.json

The plan is our plan format: {"actions": [{"obj", "dest", "relation": "in"|"on", "transit_height", "via"?}]}.
Score card:
  nominal     pass/fail of the plan on the scene exactly as perceived, with the violated predicates
  robustness  k/n runs that pass when friction (x0.6-1.4), mass (x0.7-1.3) and each object's position
              (N(0, sigma) per axis, sigma from the case) are randomized. Seeds seed0..seed0+n-1 with common
              random numbers, so every planner faces identical noise.
  score       robustness k/n if the nominal run passes, else 0
Exit code 0 if nominal passes, 1 if not, 2 on a usage error. Zero Meta API calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import caselib as C  # noqa: E402


def score_plan(case, plan: dict, n: int = C.N_DEFAULT, seed0: int = C.SEED0, ex=None) -> dict:
    """case: a folder path or a load_case() dict. ex: optional process pool for the stress runs."""
    case = C.load_case(case) if not isinstance(case, dict) else case
    scene, goals, sigmas = case["scene"], case["goals"], case["sigmas"]
    nom = C.simulate(scene, plan, goals)
    runs = []
    if n > 0:
        jobs = [(scene, plan, goals, s, sigmas) for s in range(seed0, seed0 + n)]
        if ex is None:
            with C.pool() as p:
                runs = list(p.map(C.run_job, jobs))
        else:
            runs = list(ex.map(C.run_job, jobs))
    k = sum(r["success"] for r in runs)
    counts = Counter()
    for r in runs:
        if r["success"]:
            continue
        counts.update(r["violated"])
        if r["execution_error"]:
            counts[_err_kind(r["execution_error"])] += 1
    return {"case": case["meta"]["id"], "plan_key": C.plan_key(plan) if isinstance(plan, dict) else None,
            "nominal": {"pass": nom["success"], "violated": nom["violated"],
                        "execution_error": nom["execution_error"], "evidence": nom["evidence"][:6]},
            "robustness": {"k": k, "n": n, "seed0": seed0, "failed_seeds": [r["seed"] for r in runs if not r["success"]],
                           "violated_counts": dict(counts.most_common())},
            "score": round(k / n, 4) if (nom["success"] and n) else (1.0 if nom["success"] else 0.0),
            "goals": [C.goal_str(g) for g in goals], "goals_source": case["meta"]["goals_source"],
            "code": C.code_version()}


def _err_kind(msg: str) -> str:
    m = msg.lower()
    if "invalid plan" in m or "simulator error" in m:
        return "execution: invalid plan"
    if "protective" in m:
        return "execution: protective stop"
    return "execution: out of reach (IK)"


def render_card(card: dict) -> str:
    nm, rb = card["nominal"], card["robustness"]
    L = [f"Rehearsal Bench score card: {card['case']}  (plan {card['plan_key']})",
         f"  goals ({card['goals_source']}): " + ", ".join(card["goals"]),
         f"  nominal     {'PASS' if nm['pass'] else 'FAIL'}"]
    if nm["execution_error"]:
        L.append(f"              execution error: {nm['execution_error']}")
    for v in nm["violated"]:
        L.append(f"              violated: {v}")
    L.append(f"  robustness  {rb['k']}/{rb['n']} under noise (seeds {rb['seed0']}-{rb['seed0'] + rb['n'] - 1})")
    for v, c in list(rb["violated_counts"].items())[:6]:
        L.append(f"              {c:>3}x {v}")
    L.append(f"  score       {card['score']:.3f}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", help="case folder, e.g. bench/cases/kitchen_01")
    ap.add_argument("plan", help='plan JSON file ({"actions": [...]}), or "-" for stdin')
    ap.add_argument("--n", type=int, default=C.N_DEFAULT, help="stress runs (default 40)")
    ap.add_argument("--seed0", type=int, default=C.SEED0, help="first seed (default 1000)")
    ap.add_argument("--json", default=None, help="also write the score card as JSON here")
    args = ap.parse_args(argv)
    if not (Path(args.case) / "case.json").is_file():
        print(f"no case.json in {args.case}", file=sys.stderr)
        return 2
    try:
        plan = json.loads(sys.stdin.read() if args.plan == "-" else Path(args.plan).read_text())
    except (OSError, ValueError) as e:
        print(f"cannot read plan: {e}", file=sys.stderr)
        return 2
    card = score_plan(args.case, plan, max(0, args.n), args.seed0)
    print(render_card(card))
    if args.json:
        Path(args.json).write_text(json.dumps(card, indent=1, ensure_ascii=False))
    return 0 if card["nominal"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
