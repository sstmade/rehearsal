"""Experiment B — why 2 cm matters: success vs injected perception error, initial vs revised plan.
Offline, $0.

    .venv/bin/python eval/perception_curve.py [--n 100] [--sigmas 0,0.4,0.8,1.2,1.6,2.0,2.4,3.0]

Plans per scene: the reference naive (initial) and fixed (revised) plans from scenes/*.json, plus the
agents' first and final plans found in eval/results/ for the same code (only the demo / full task).
Every object gets xy error ~ N(0, sigma) per axis; friction x U(0.6,1.4), mass x U(0.7,1.3) as in the
product. Seeds are HELD OUT (1000..1000+n-1, never used to select or revise anything). Common random
numbers: seed s draws the same unit-normal direction for every sigma, so the curve is paired (errors
are scaled, not redrawn). Wilson 95% intervals. Plans that already fail at sigma=0 are swept with
n=20 only (they fail for a geometric reason, not a perception one).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

HOLDOUT = 1000
FULL = {"desk_01": "把杯子放进碗里，然后把书摞整齐"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=str(C.REPO / "scenes"))
    ap.add_argument("--only", default="")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sigmas", default="0,0.4,0.8,1.2,1.6,2.0,2.4,3.0", help="cm")
    args = ap.parse_args()
    sigmas_cm = [float(s) for s in args.sigmas.split(",")]
    only = [s for s in args.only.split(",") if s] or None
    plans = C.reference_plans(Path(args.scenes), only) + C.final_plans(Path(args.scenes), only, include_initial=True)
    plans = [p for p in plans if p["task"]["instruction"] in
             (p["task"]["scene_spec"].get("demo_instruction"), FULL.get(p["task"]["scene"]))]
    print(f"code: {C.code_label()}  plans: {[(p['task']['scene'], p['agent']) for p in plans]}")
    out, t0 = [], time.time()
    with C.pool() as ex:
        for p in plans:
            scene, ids = p["task"]["scene_spec"], [o["id"] for o in p["task"]["scene_spec"]["objects"]]
            nominal = C.stress(scene, p["plan"], p["goals"], [HOLDOUT], [0.0] * len(ids), ex)["successes"]
            n = args.n if nominal else min(args.n, 20)
            curve = []
            for s in sigmas_cm:
                st = C.stress(scene, p["plan"], p["goals"], range(HOLDOUT, HOLDOUT + n), [s / 100] * len(ids), ex)
                lo, hi = C.wilson(st["successes"], st["n"])
                curve.append({"sigma_cm": s, "successes": st["successes"], "n": st["n"],
                              "ci95": [round(lo, 3), round(hi, 3)], "example_failures": st["failed"][:2]})
            print(f"  {p['task']['scene']:<14} {p['agent']:<16} {p['plan_key']}  "
                  + "  ".join(f"{c['sigma_cm']}:{c['successes']}/{c['n']}" for c in curve), flush=True)
            out.append({"scene": p["task"]["scene"], "instruction": p["task"]["instruction"],
                        "agent": p["agent"], "plan_key": p["plan_key"], "plan": p["plan"], "curve": curve})
    rdir = C.EVAL_DIR / "results"
    rdir.mkdir(exist_ok=True)
    path = rdir / f"pc_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"kind": "perception_curve", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "code": C.code_label(), "seeds": f"{HOLDOUT}..", "wall_s": round(time.time() - t0, 1),
                                "plans": out}, ensure_ascii=False, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
