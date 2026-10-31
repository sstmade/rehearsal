"""Experiment C — active perception: is the value-of-information pick better than simple baselines,
on HELD-OUT noise draws?  Offline, $0 (no model calls).

    .venv/bin/python eval/active_perception_baseline.py [--n-select 40] [--n-heldout 200]

Plans: the scene's hand-verified reference fixed plan (scenes/*.json `reference.fixed_plan`) and,
where no reference exists, the best passing final plan found in eval/results/ for the same code.
A re-observation is modelled exactly as the product does (rehearse.active_perception): that object's
perception sigma drops from 1.2 cm to 0.4 cm, everything else unchanged, common random numbers.

Protocol (fixes the winner's-curse bias flagged in research docs/claims.md §1.6/§4):
  SELECT  on seeds 0..n_select-1 (what the product does at run time)
  SCORE   every policy on held-out seeds 1000..1000+n_heldout-1, never used for selection
Policies (budget: one extra look):
  voi        argmax over objects of selection-set successes (ties -> first in scene order)
  random     expected held-out success of a uniformly random object (exact mean over all objects)
  fail_pred  heuristic: object that appears most often in failed predicates on the selection set
  near_path  heuristic: non-moved object closest to the plan's straight carry segments
             (the plan's own objects excluded, since they are not obstacles); falls back to voi order
  oracle     best single object on the held-out set (upper bound for one look)
  all        re-observe every object (budget = #objects)
(A "largest sigma" heuristic is degenerate here: all objects share sigma = 1.2 cm.)
Paired statistics vs. no extra look: discordant seeds (fixed / broken) and an exact two-sided sign test.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

HOLDOUT = 1000


def sign_test(up: int, down: int) -> float:
    n = up + down
    if n == 0:
        return 1.0
    k = min(up, down)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def seg_dist(p, a, b) -> float:
    p, a, b = map(np.asarray, (p, a, b))
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def near_path_object(scene, plan) -> list[str]:
    pos = {o["id"]: np.array(o["pos"][:2], float) for o in scene["objects"]}
    moved = {a["obj"] for a in plan["actions"]} | {a["dest"] for a in plan["actions"]}
    segs = []
    for a in plan["actions"]:
        pts = [pos[a["obj"]], *[np.array(v, float) for v in (a.get("via") or [])], pos[a["dest"]]]
        segs += list(zip(pts, pts[1:]))
    cand = [oid for oid in pos if oid not in moved]
    return sorted(cand, key=lambda oid: min(seg_dist(pos[oid], a, b) for a, b in segs)) if segs else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=str(C.REPO / "scenes"))
    ap.add_argument("--only", default="")
    ap.add_argument("--n-select", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=200)
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s] or None
    cands = C.reference_plans(Path(args.scenes), only, ("fixed_plan",)) + C.final_plans(Path(args.scenes), only)
    pick = {}
    rank = {"ref_fixed": 3, "muse": 2, "offline": 1}
    for p in cands:
        if p["task"]["instruction"] != p["task"]["scene_spec"].get("demo_instruction") and p["agent"] != "offline":
            continue
        k = p["task"]["scene"]
        if p["task"]["scene"] == "desk_01" and p["task"]["instruction"] != "把杯子放进碗里，然后把书摞整齐":
            continue
        if k not in pick or rank.get(p["agent"], 0) > rank.get(pick[k]["agent"], 0):
            pick[k] = p
    plans = list(pick.values())
    print(f"code: {C.code_label()}  plans: {[(p['task']['scene'], p['agent']) for p in plans]}")
    S, SR = C.R.SIGMA, C.R.SIGMA_REFINED
    sel, ho = range(args.n_select), range(HOLDOUT, HOLDOUT + args.n_heldout)
    out, t0 = [], time.time()
    with C.pool() as ex:
        for p in plans:
            scene, plan, goals = p["task"]["scene_spec"], p["plan"], p["goals"]
            ids = [o["id"] for o in scene["objects"]]
            memo = {}

            def run(refined: frozenset, seeds):
                key = (refined, seeds.start)
                if key not in memo:
                    memo[key] = C.stress(scene, plan, goals, seeds, [SR if i in refined else S for i in ids], ex)
                return memo[key]

            print(f"\n== {p['task']['scene']} ({p['agent']} plan {p['plan_key']})", flush=True)
            base_sel, base_ho = run(frozenset(), sel), run(frozenset(), ho)
            sel_scores = {oid: run(frozenset([oid]), sel)["successes"] for oid in ids}
            ho_runs = {oid: run(frozenset([oid]), ho) for oid in ids}
            # failed-predicate heuristic: object mentions in failure lines on the selection set
            jobs = [(scene, plan, goals, int(s), [S] * len(ids), False) for s in sel]
            counts = {oid: 0 for oid in ids}
            for r in ex.map(C.R._variant, jobs):
                for line in r["failed"]:
                    for oid in ids:
                        # predicate lines "pred(args): why" -> count args only; execution errors -> whole line
                        head = line.split("):")[0] if line.split("(")[0].isidentifier() and "):" in line else line
                        counts[oid] += oid in head
            voi = max(ids, key=lambda o: sel_scores[o])
            fail_pred = max(ids, key=lambda o: (counts[o], sel_scores[o])) if any(counts.values()) else voi
            near = (near_path_object(scene, plan) or [voi])[0]
            oracle = max(ids, key=lambda o: ho_runs[o]["successes"])
            all_ho = run(frozenset(ids), ho)

            def paired(r):
                b = np.array(base_ho["ok"]); v = np.array(r["ok"])
                up, down = int((~b & v).sum()), int((b & ~v).sum())
                return {"successes": r["successes"], "n": r["n"], "fixed": up, "broken": down,
                        "sign_test_p": round(sign_test(up, down), 5)}

            pol = {"none": {"look_at": None, **paired(base_ho)},
                   "voi": {"look_at": voi, **paired(ho_runs[voi])},
                   "random": {"look_at": "uniform", "successes": float(np.mean([ho_runs[o]["successes"] for o in ids])),
                              "n": args.n_heldout},
                   "fail_pred": {"look_at": fail_pred, **paired(ho_runs[fail_pred])},
                   "near_path": {"look_at": near, **paired(ho_runs[near])},
                   "oracle": {"look_at": oracle, **paired(ho_runs[oracle])},
                   "all": {"look_at": f"all {len(ids)}", **paired(all_ho)}}
            for k, v in pol.items():
                print(f"  {k:<10} {str(v['look_at']):<14} {v['successes']}/{args.n_heldout}"
                      + (f"  +{v['fixed']}/-{v['broken']} p={v['sign_test_p']}" if "fixed" in v else ""), flush=True)
            out.append({"scene": p["task"]["scene"], "instruction": p["task"]["instruction"], "agent": p["agent"],
                        "plan_key": p["plan_key"], "plan": plan, "n_objects": len(ids),
                        "selection": {"base": base_sel["successes"], "n": args.n_select, "by_object": sel_scores},
                        "heldout_by_object": {o: r["successes"] for o, r in ho_runs.items()},
                        "fail_pred_counts": counts, "near_path_order": near_path_object(scene, plan),
                        "policies": pol})
    rdir = C.EVAL_DIR / "results"
    rdir.mkdir(exist_ok=True)
    res = {"kind": "active_perception", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": C.code_label(),
           "sigma_cm": S * 100, "refined_sigma_cm": SR * 100, "selection_seeds": [0, args.n_select - 1],
           "held_out_seeds": [HOLDOUT, HOLDOUT + args.n_heldout - 1], "wall_s": round(time.time() - t0, 1),
           "plans": out}
    path = rdir / f"ap_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\nwrote {path} ({res['wall_s']} s)")


if __name__ == "__main__":
    main()
