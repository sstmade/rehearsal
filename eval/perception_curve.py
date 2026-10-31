"""Why 2 cm matters: success rate of each final plan vs injected perception error.  Offline, $0.

    .venv/bin/python eval/perception_curve.py [--n 40] [--sigmas 0,0.5,1,1.5,2,2.5,3,4]

Every object gets xy error ~ N(0, sigma) per axis (friction/mass randomised as in the product).
Common random numbers: seed s draws the same unit-normal direction for every sigma, so the curve
is a paired comparison (error vectors are scaled, not redrawn) and is close to monotone.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=str(C.REPO / "scenes"))
    ap.add_argument("--only", default="")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--sigmas", default="0,0.5,1,1.5,2,2.5,3,4", help="cm")
    args = ap.parse_args()
    sigmas_cm = [float(s) for s in args.sigmas.split(",")]
    plans = C.final_plans(Path(args.scenes), [s for s in args.only.split(",") if s] or None)
    print(f"code: {C.code_label()}  plans: {len(plans)}")
    out, t0 = [], time.time()
    with C.pool() as ex:
        for p in plans:
            scene, ids = p["task"]["scene_spec"], [o["id"] for o in p["task"]["scene_spec"]["objects"]]
            curve = []
            for s in sigmas_cm:
                st = C.stress(scene, p["plan"], p["goals"], range(args.n), [s / 100] * len(ids), ex)
                lo, hi = C.wilson(st["successes"], st["n"])
                curve.append({"sigma_cm": s, "successes": st["successes"], "n": st["n"],
                              "ci95": [round(lo, 3), round(hi, 3)], "example_failures": st["failed"][:2]})
            print(f"  {p['task']['scene']:<14} {p['agent']:<8} {p['plan_key']}  "
                  + "  ".join(f"{c['sigma_cm']}cm:{c['successes']}" for c in curve))
            out.append({"scene": p["task"]["scene"], "instruction": p["task"]["instruction"],
                        "agent": p["agent"], "plan_key": p["plan_key"], "plan": p["plan"], "curve": curve})
    rdir = C.EVAL_DIR / "results"
    rdir.mkdir(exist_ok=True)
    tag = "robotics" if "robotics" in C.code_label() else "main"
    path = rdir / f"pc_{time.strftime('%Y%m%d_%H%M%S')}_{tag}.json"
    path.write_text(json.dumps({"kind": "perception_curve", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "code": C.code_label(), "n": args.n, "wall_s": round(time.time() - t0, 1),
                                "plans": out}, ensure_ascii=False, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
