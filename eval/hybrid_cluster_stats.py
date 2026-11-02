"""Cluster-level statistics for the hybrid + gate vs log-revise robustness comparison ($0, no simulation, no API).

docs/hybrid.md reports an exact McNemar test over 1,980 (task, seed) pairs on the five hard tables. Those pairs are not
independent: the 99 tasks share 5 scenes and the same 20 held-out seeds, and the search is deterministic given the
goals, so most tasks in a scene carry the *same* hybrid plan (and many revise plans repeat too). This script re-reads the
per-(plan, seed) outcomes already stored in eval/results/hybrid_work.json, rebuilds exactly the pairs the McNemar test
used (eval/hybrid_eval.py, summarize), and tests the same difference at the cluster level:

  1. seed-level exact McNemar (the original test, for reference)
  2. cluster-robust (GEE-style: identity link, independence working correlation, sandwich variance) z-tests on the
     paired difference, clustered by task, by distinct (hybrid plan, revise plan) pair, two-way by task and seed, and
     by scene
  3. cluster bootstrap of the pooled difference: resampling tasks; resampling scenes then tasks within them
  4. exact sign-flip permutation tests on per-cluster mean differences: over distinct plan pairs, and over the 5 scenes

    .venv/bin/python eval/hybrid_cluster_stats.py      # writes eval/results/hybrid_cluster_stats.json
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eval.hybrid_eval import HARD, pkey  # noqa: E402

WORK = REPO / "eval" / "results" / "hybrid_work.json"
OUT = REPO / "eval" / "results" / "hybrid_cluster_stats.json"
B = 20000
RNG = np.random.default_rng(20261102)


def pairs():
    """One row per task both arms solved on the hard tables: scene, repeat, plan keys, the two 20-seed vectors."""
    work = json.loads(WORK.read_text())
    rob, srch = work["robust"], work["search"]
    rows = []
    for t in work["tasks"]:
        s = t["scene"]
        if s not in HARD:
            continue
        r = srch.get(f"hybrid_gate20|{s}|{t['repeat']}")
        if not (r and r["ref_ok"] and t["revise_ok"]):
            continue
        ka, kb = pkey(r["plan"]), pkey(t["final_plan"])
        a, b = rob.get(f"{s}|{ka}"), rob.get(f"{s}|{kb}")
        if a is None or b is None:
            continue
        rows.append({"scene": s, "repeat": t["repeat"], "hyb": ka, "rev": kb,
                     "a": np.array(a, dtype=float), "b": np.array(b, dtype=float)})
    return rows


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def sandwich_z(d: np.ndarray, groups: list[np.ndarray]) -> dict:
    """Mean of d with a cluster-robust variance. groups: one label array per clustering (two arrays = two-way
    clustering, Cameron-Gelbach-Miller: V = V_g1 + V_g2 - V_g1∩g2). Small-sample factor G/(G-1) per clustering."""
    n, mu = len(d), d.mean()
    e = d - mu

    def v_of(lab):
        uniq = np.unique(lab)
        g = len(uniq)
        s = np.array([e[lab == u].sum() for u in uniq])
        return (g / (g - 1) if g > 1 else 1.0) * (s ** 2).sum() / n ** 2, g

    if len(groups) == 1:
        v, g = v_of(groups[0])
        gs = [g]
    else:
        v1, g1 = v_of(groups[0])
        v2, g2 = v_of(groups[1])
        inter = np.array([f"{x}|{y}" for x, y in zip(groups[0], groups[1])])
        v12, _ = v_of(inter)
        v, gs = max(v1 + v2 - v12, max(v1, v2)), [g1, g2]
    se = math.sqrt(v)
    z = mu / se if se > 0 else float("inf")
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"diff": round(mu, 4), "se": round(se, 4), "z": round(z, 2), "p_normal": round(p, 4), "clusters": gs,
            "ci95": [round(mu - 1.96 * se, 4), round(mu + 1.96 * se, 4)]}


def signflip(x: np.ndarray, exact_max: int = 22) -> float:
    """Two-sided exact (or 200k Monte Carlo) sign-flip permutation p for mean(x) = 0."""
    x = x[x != 0] if np.any(x != 0) else x
    obs = abs(x.mean())
    if len(x) <= exact_max:
        signs = np.array(list(itertools.product([1, -1], repeat=len(x))))
        stats = np.abs((signs * x).mean(axis=1))
    else:
        signs = RNG.choice([1, -1], size=(200000, len(x)))
        stats = np.abs((signs * x).mean(axis=1))
    return float((stats >= obs - 1e-12).mean())


def boot(rows, two_stage: bool) -> dict:
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(r)
    scenes = sorted(by_scene)
    da = {id(r): r["a"].mean() for r in rows}
    db = {id(r): r["b"].mean() for r in rows}
    out = np.empty(B)
    for i in range(B):
        if two_stage:
            pick = []
            for s in RNG.choice(scenes, size=len(scenes)):
                grp = by_scene[s]
                pick += [grp[j] for j in RNG.integers(0, len(grp), size=len(grp))]
        else:
            pick = [rows[j] for j in RNG.integers(0, len(rows), size=len(rows))]
        out[i] = np.mean([da[id(r)] - db[id(r)] for r in pick])
    lo, hi = np.percentile(out, [2.5, 97.5])
    p = min(1.0, 2 * min((out <= 0).mean(), (out >= 0).mean()))
    return {"ci95": [round(float(lo), 4), round(float(hi), 4)], "p_two_sided": round(float(p), 4), "B": B}


def main() -> int:
    rows = pairs()
    a = np.concatenate([r["a"] for r in rows])
    b = np.concatenate([r["b"] for r in rows])
    d = a - b
    nseed = len(rows[0]["a"])
    task = np.repeat([f"{r['scene']}|{r['repeat']}" for r in rows], nseed)
    plan_pair = np.repeat([f"{r['scene']}|{r['hyb']}|{r['rev']}" for r in rows], nseed)
    scene = np.repeat([r["scene"] for r in rows], nseed)
    seed = np.tile(np.arange(nseed), len(rows))
    only_h, only_r = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())

    # per distinct plan pair and per scene: mean difference (each cluster counted once)
    pp = defaultdict(list)
    for r in rows:
        pp[f"{r['scene']}|{r['hyb']}|{r['rev']}"].append(r["a"].mean() - r["b"].mean())
    pp_d = np.array([v[0] for v in pp.values()])          # identical plans -> identical vectors -> same value
    sc = defaultdict(list)
    for r in rows:
        sc[r["scene"]].append(r["a"].mean() - r["b"].mean())
    sc_d = np.array([np.mean(v) for v in sc.values()])
    distinct_vec_pairs = len({(r["a"].tobytes(), r["b"].tobytes()) for r in rows})

    res = {
        "source": str(WORK.relative_to(REPO)),
        "n_tasks": len(rows), "n_seed_pairs": int(len(d)), "seeds": nseed,
        "n_scenes": len(sc), "n_distinct_hybrid_plans": len({(r["scene"], r["hyb"]) for r in rows}),
        "n_distinct_revise_plans": len({(r["scene"], r["rev"]) for r in rows}),
        "n_distinct_plan_pairs": len(pp), "n_distinct_outcome_vector_pairs": distinct_vec_pairs,
        "hybrid_rate": round(float(a.mean()), 4), "revise_rate": round(float(b.mean()), 4),
        "seed_level_mcnemar": {"hybrid_only": only_h, "revise_only": only_r, "p_exact": round(mcnemar(only_h, only_r), 5)},
        "gee_sandwich": {
            "by_task": sandwich_z(d, [task]),
            "by_distinct_plan_pair": sandwich_z(d, [plan_pair]),
            "two_way_task_and_seed": sandwich_z(d, [task, seed]),
            "two_way_plan_pair_and_seed": sandwich_z(d, [plan_pair, seed]),
            "by_scene": sandwich_z(d, [scene]),
        },
        "cluster_bootstrap": {"tasks": boot(rows, False), "scenes_then_tasks": boot(rows, True)},
        "signflip_permutation": {
            "distinct_plan_pairs": {"n": len(pp_d), "nonzero": int((pp_d != 0).sum()),
                                    "mean_diff": round(float(pp_d.mean()), 4), "p": round(signflip(pp_d), 4)},
            "scenes": {"n": len(sc_d), "per_scene_diff": {s: round(float(np.mean(v)), 4) for s, v in sc.items()},
                       "p": round(signflip(sc_d), 4), "min_attainable_p": round(2 / 2 ** len(sc_d), 4)},
        },
    }
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
