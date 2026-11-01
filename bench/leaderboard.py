"""Score three planners on every bench case FROM CACHED PLANS ONLY (zero Meta API calls) and write
bench/LEADERBOARD.md + bench/leaderboard.json.

    .venv/bin/python bench/leaderboard.py            # n = 40 stress runs per (case, planner)
    .venv/bin/python bench/leaderboard.py --n 10     # quicker

Planners:
  muse_first        Muse Spark's first plan in the recorded product run (attempt 1 of out/<scene>/report.json)
  muse_after_log    the plan Muse Spark ended with after seeing the simulator's log (last attempt; for agent-loop
                    runs, the plan it submitted after its own simulate / stress-test calls)
  scripted_baseline straight-line pick-place per goal in instruction order at 0.15 m (bench/caselib.py)
Where cached Evaluation v3 plans exist (eval/results round3, 20 repeats per scene, English instruction), their
first and after-log plans are also re-simulated (nominal only) and reported as k/20.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import caselib as C  # noqa: E402
from bench.score import score_plan  # noqa: E402

PLANNERS = [
    ("muse_first", "Muse Spark · first plan", "attempt 1 of the recorded run"),
    ("muse_after_log", "Muse Spark · after log feedback", "last attempt, after reading the simulator's log"),
    ("scripted_baseline", "Scripted baseline", "straight lines in instruction order, 0.15 m carry height"),
]
KIND_LABEL = {"hand_designed": "hand-designed", "real_dataset_photo": "real photo (YCB-Video)",
              "ai_generated_photo": "AI-generated photo", "real_photo": "real photo", "user_photo": "user photo"}


def _nominal_batch(case, plans: list[dict], ex) -> int:
    sims = ex.map(C.run_job, [(case["scene"], p, case["goals"], None, case["sigmas"]) for p in plans])
    return sum(r["success"] for r in sims)


def build(n: int, cases_root: Path = C.CASES, log=print) -> dict:
    rows = []
    with C.pool() as ex:
        for d in C.list_cases(cases_root):
            case = C.load_case(d)
            meta = case["meta"]
            row = {"id": meta["id"], "title": meta["title"], "kind": meta["provenance"]["kind"],
                   "goals_source": meta["goals_source"], "failures": meta["failures"]["count"],
                   "failed_runs": meta["failures"].get("failed_runs_covered", meta["failures"]["count"]),
                   "cells": {}, "eval": None}
            for pid, _, _ in PLANNERS:
                pp = d / meta["files"]["plans"].get(pid, f"plans/{pid}.json")
                if not pp.is_file():
                    row["cells"][pid] = None          # this planner has no plan for this case
                    continue
                plan = json.loads(pp.read_text())
                card = score_plan(case, plan, n, C.SEED0, ex)
                row["cells"][pid] = {"nominal": card["nominal"]["pass"], "k": card["robustness"]["k"], "n": n,
                                     "score": card["score"], "violated": card["nominal"]["violated"],
                                     "execution_error": card["nominal"]["execution_error"],
                                     "top_stress_failure": next(iter(card["robustness"]["violated_counts"]), None),
                                     "plan_key": card["plan_key"]}
                log(f"{meta['id']:<22} {pid:<18} nominal {'PASS' if card['nominal']['pass'] else 'fail'}  "
                    f"robust {card['robustness']['k']}/{n}")
            ev = {}
            for k in ("first", "after_log"):
                rel = meta["files"]["plans"].get(f"eval_round3_{k}")
                if rel:
                    plans = [json.loads(line)["plan"] for line in (d / rel).read_text().splitlines() if line.strip()]
                    ev[k] = {"k": _nominal_batch(case, plans, ex), "n": len(plans)}
            row["eval"] = ev or None
            rows.append(row)
    totals = {}
    for pid, _, _ in PLANNERS:
        cells = [r["cells"][pid] for r in rows if r["cells"].get(pid)]
        totals[pid] = {"nominal": sum(c["nominal"] for c in cells), "cases": len(cells),
                       "k": sum(c["k"] for c in cells), "n": sum(c["n"] for c in cells),
                       "k_if_nominal": sum(c["k"] for c in cells if c["nominal"]),
                       "mean_score": round(sum(c["score"] for c in cells) / max(1, len(cells)), 4)}
    ev_tot = {k: {"k": sum(r["eval"][k]["k"] for r in rows if r["eval"] and k in r["eval"]),
                  "n": sum(r["eval"][k]["n"] for r in rows if r["eval"] and k in r["eval"])}
              for k in ("first", "after_log")}
    return {"generated": time.strftime("%Y-%m-%d %H:%M"), "n": n, "seed0": C.SEED0, "code": C.code_version(),
            "planners": [{"id": p, "label": l, "desc": s} for p, l, s in PLANNERS],
            "cases": rows, "totals": totals, "eval_totals": ev_tot,
            "n_cases": len(rows), "n_failures": sum(r["failures"] for r in rows),
            "n_failed_runs": sum(r["failed_runs"] for r in rows)}


def _cell(c: dict | None) -> str:
    if not c:
        return "—"
    mark = "✅" if c["nominal"] else "❌"
    return f"{mark} {c['k']}/{c['n']}"


def render_md(lb: dict) -> str:
    P = lb["planners"]
    n = lb["n"]
    L = ["# Rehearsal Bench v0 — leaderboard", "",
         f"_Generated {lb['generated']} by `bench/leaderboard.py` (code `{lb['code']['git']}`, "
         f"`code#{lb['code']['code_hash']}`). {lb['n_cases']} cases, {lb['n_failures']} recorded failures "
         f"({lb['n_failed_runs']} failed runs). All plans are cached: **zero Meta API calls**._", "",
         f"Each cell: ✅/❌ = nominal run (scene exactly as perceived) passes every goal predicate; k/{n} = runs that "
         f"pass under randomized friction ×0.6–1.4, mass ×0.7–1.3 and per-object position error N(0, σ) "
         f"(seeds {lb['seed0']}–{lb['seed0'] + n - 1}, identical for every planner). Score = k/{n} if nominal passes, else 0.", "",
         "| case | provenance | goals | " + " | ".join(p["label"] for p in P) + " | recorded failures |",
         "|---|---|---|" + "---|" * len(P) + "---|"]
    for r in lb["cases"]:
        L.append(f"| `{r['id']}` | {KIND_LABEL.get(r['kind'], r['kind'])} | {r['goals_source'].replace('_', ' ')} | "
                 + " | ".join(_cell(r["cells"].get(p["id"])) for p in P) + f" | {r['failures']} |")
    t = lb["totals"]
    L.append("| **total** | | | " + " | ".join(
        f"**{t[p['id']]['nominal']}/{t[p['id']]['cases']}** nominal · **{t[p['id']]['k']}/{t[p['id']]['n']}** robust"
        for p in P) + f" | **{lb['n_failures']}** |")
    L.append("| **mean score** | | | " + " | ".join(f"**{t[p['id']]['mean_score']:.3f}**" for p in P) + " | |")
    L += ["", "Planners:", ""]
    L += [f"- **{p['label']}** (`{p['id']}`): {p['desc']}." for p in P]
    L += ["", "## Cached Evaluation v3 plans (20 repeats per scene, nominal only)", "",
          "Re-simulated on the current code from `eval/results/eval_20261031_173045_round3_main.json` "
          "(English instruction, full-log condition). Graded against the case goals (reference goals + implicit "
          "`undisturbed`).", "",
          "| case | Muse Spark · first plan | Muse Spark · after log feedback (≤ 2 revisions) |", "|---|---|---|"]
    for r in lb["cases"]:
        if r["eval"]:
            e = r["eval"]
            L.append(f"| `{r['id']}` | {e['first']['k']}/{e['first']['n']} | {e['after_log']['k']}/{e['after_log']['n']} |")
    et = lb["eval_totals"]
    L.append(f"| **pooled** | **{et['first']['k']}/{et['first']['n']}** | **{et['after_log']['k']}/{et['after_log']['n']}** |")
    L += ["", "## Notes", "",
          "- Per-case cells score **one** cached plan per planner (the one the shipped product run produced), so a "
          "cell is an anecdote; the Evaluation v3 table above is the n = 20 view for the five scenes that have it.",
          "- `muse_after_log` on agent-loop runs (`kitchen_01`, `real_ycbv59`, `aigen_*`) is the plan Muse Spark "
          "submitted after its own simulate / stress-test tool calls, so its feedback included stress results, "
          "not only the contact log.",
          "- Goals marked *planner compiled* come from the recorded run (no hand-verified reference yet); a planner "
          "that compiled different goals is still scored on these.",
          "- The scripted baseline is deliberately trivial: a floor, not a competitor. A 0.15 m straight carry "
          "is too low on every table here (it knocks the bottle on desk_01, presses on the kettle in both kitchens, "
          "sweeps the lamp), so it passes none; a smarter fixed-height heuristic is an obvious next baseline.",
          "- Sanity check: on every case the `muse_after_log` robustness equals the stress test in that case's "
          "`report.json` (same plan, seeds and noise model), e.g. `real_ycbv59` 7/40, `kitchen_01` 37/40.",
          "- Simulation only (MuJoCo, primitives, suction grasp). Small n. See bench/README.md, Limitations.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=C.N_DEFAULT)
    ap.add_argument("--cases", default=str(C.CASES))
    ap.add_argument("--out", default=str(C.BENCH))
    args = ap.parse_args(argv)
    lb = build(args.n, Path(args.cases))
    out = Path(args.out)
    (out / "leaderboard.json").write_text(json.dumps(lb, indent=1, ensure_ascii=False))
    (out / "LEADERBOARD.md").write_text(render_md(lb), encoding="utf-8")
    print(f"\nwrote {out / 'LEADERBOARD.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
