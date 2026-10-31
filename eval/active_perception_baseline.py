"""Active perception: does choosing WHICH object to re-observe by value of information beat
simple baselines?  Offline, $0 (no model calls; plans come from eval/results/).

    .venv/bin/python eval/active_perception_baseline.py [--n 40] [--k 2]

Model of a re-observation (same as the product, rehearse.active_perception): that object's
perception sigma drops from 1.2 cm to 0.4 cm. Strategies, for an observation budget k:
  voi          greedy: pick the object whose refinement maximises stress-test successes, repeat
  random       expected value over ALL size-k subsets (exact average, not a sample)
  touched      re-observe every object the plan picks or places on (budget = #touched)
  all          re-observe every object (budget = #objects)
Selection uses seeds 0..n-1 (what the product does). Every strategy is then SCORED on held-out
seeds 10000..10000+n-1, so VoI gets no credit for fitting the noise draws it selected on; the
in-sample number is reported too, to show the optimism of selecting and scoring on the same seeds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

HOLDOUT = 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=str(C.REPO / "scenes"))
    ap.add_argument("--only", default="")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max-plans", type=int, default=0)
    args = ap.parse_args()
    plans = C.final_plans(Path(args.scenes), [s for s in args.only.split(",") if s] or None)
    # one plan per scene+instruction is enough (prefer muse, else offline); skip sub-tasks dupes
    pick = {}
    for p in plans:
        k = (p["task"]["scene"], p["task"]["instruction"])
        if k not in pick or p["agent"] == "muse":
            pick[k] = p
    plans = list(pick.values())[: args.max_plans or None]
    print(f"code: {C.code_label()}  plans: {len(plans)}")
    S, SR = C.R.SIGMA, C.R.SIGMA_REFINED
    sel_seeds, ho_seeds = range(args.n), range(HOLDOUT, HOLDOUT + args.n)
    out = []
    t0 = time.time()
    with C.pool() as ex:
        for p in plans:
            scene, plan, goals = p["task"]["scene_spec"], p["plan"], p["goals"]
            ids = [o["id"] for o in scene["objects"]]
            memo = {}

            def score(refined: frozenset, seeds):
                key = (refined, seeds.start)
                if key not in memo:
                    sig = [SR if oid in refined else S for oid in ids]
                    memo[key] = C.stress(scene, plan, goals, seeds, sig, ex)["successes"]
                return memo[key]

            print(f"\n== {p['task']['scene']} | {p['task']['instruction']} | {p['agent']} plan {p['plan_key']}")
            base_sel, base_ho = score(frozenset(), sel_seeds), score(frozenset(), ho_seeds)
            # VoI greedy on selection seeds
            chosen, voi_curve = [], []
            for _ in range(args.k):
                opts = {oid: score(frozenset(chosen + [oid]), sel_seeds) for oid in ids if oid not in chosen}
                best = max(opts, key=opts.get)
                chosen.append(best)
                voi_curve.append({"k": len(chosen), "look_at": best, "in_sample": opts[best],
                                  "held_out": score(frozenset(chosen), ho_seeds), "options_in_sample": opts})
                print(f"  voi k={len(chosen)}: {best}  in-sample {opts[best]}/{args.n}  "
                      f"held-out {voi_curve[-1]['held_out']}/{args.n}")
            rand_curve = []
            for k in range(1, args.k + 1):
                vals = [score(frozenset(sub), ho_seeds) for sub in itertools.combinations(ids, k)]
                rand_curve.append({"k": k, "held_out_mean": sum(vals) / len(vals),
                                   "held_out_min": min(vals), "held_out_max": max(vals), "subsets": len(vals)})
                print(f"  random k={k}: mean {rand_curve[-1]['held_out_mean']:.1f}/{args.n} "
                      f"(range {min(vals)}-{max(vals)} over {len(vals)} subsets)")
            touched = sorted({a["obj"] for a in plan["actions"]} | {a["dest"] for a in plan["actions"]})
            touched_ho = score(frozenset(touched), ho_seeds)
            all_ho = score(frozenset(ids), ho_seeds)
            print(f"  base held-out {base_ho}/{args.n}; touched({len(touched)}) {touched_ho}; all({len(ids)}) {all_ho}")
            out.append({"scene": p["task"]["scene"], "instruction": p["task"]["instruction"],
                        "agent": p["agent"], "plan_key": p["plan_key"], "n": args.n, "n_objects": len(ids),
                        "base_in_sample": base_sel, "base_held_out": base_ho,
                        "voi": voi_curve, "random": rand_curve,
                        "touched": {"objects": touched, "k": len(touched), "held_out": touched_ho},
                        "all": {"k": len(ids), "held_out": all_ho},
                        "stress_tests_run": len(memo)})
    rdir = C.EVAL_DIR / "results"
    rdir.mkdir(exist_ok=True)
    res = {"kind": "active_perception", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": C.code_label(),
           "sigma_cm": S * 100, "refined_sigma_cm": SR * 100, "selection_seeds": [0, args.n - 1],
           "held_out_seeds": [HOLDOUT, HOLDOUT + args.n - 1], "wall_s": round(time.time() - t0, 1), "plans": out}
    tag = "robotics" if "robotics" in C.code_label() else "main"
    path = rdir / f"ap_{time.strftime('%Y%m%d_%H%M%S')}_{tag}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\nwrote {path} ({res['wall_s']} s)")


if __name__ == "__main__":
    main()
