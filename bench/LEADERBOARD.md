# Rehearsal Bench v0 — leaderboard

_Generated 2026-11-02 03:20 by `bench/leaderboard.py` (code `d51c865`, `code#b98a13633e`). 10 cases, 84 recorded failures (121 failed runs). All plans are cached: **zero Meta API calls**._

Each cell: ✅/❌ = nominal run (scene exactly as perceived) passes every goal predicate; k/40 = runs that pass under randomized friction ×0.6–1.4, mass ×0.7–1.3 and per-object position error N(0, σ) (seeds 1000–1039, identical for every planner). Score = k/40 if nominal passes, else 0.

| case | provenance | goals | Hybrid · shipped | LLM-only · first plan | LLM-only · after log feedback | Scripted baseline | recorded failures |
|---|---|---|---|---|---|---|---|
| `aigen_desk_lamp_mug` | AI-generated photo | planner compiled | ✅ 8/40 | ✅ 8/40 | ✅ 8/40 | ❌ 0/40 | 0 |
| `aigen_desk_walnut` | AI-generated photo | planner compiled | ✅ 9/40 | ❌ 2/40 | ✅ 9/40 | ❌ 2/40 | 1 |
| `aigen_kitchen` | AI-generated photo | planner compiled | ✅ 7/40 | ✅ 7/40 | ✅ 7/40 | ❌ 0/40 | 0 |
| `desk_01` | hand-designed | planner compiled | ✅ 34/40 | ✅ 34/40 | ✅ 34/40 | ❌ 0/40 | 0 |
| `desk_02` | hand-designed | reference | ✅ 26/40 | ❌ 0/40 | ✅ 24/40 | ❌ 0/40 | 19 |
| `kitchen_01` | hand-designed | reference | ✅ 37/40 | ❌ 0/40 | ✅ 37/40 | ❌ 0/40 | 18 |
| `lamp_01` | hand-designed | reference | ✅ 34/40 | ❌ 0/40 | ✅ 34/40 | ❌ 0/40 | 8 |
| `real_ycbv59` | real photo (YCB-Video) | planner compiled | ✅ 7/40 | ✅ 7/40 | ✅ 7/40 | ❌ 0/40 | 0 |
| `storage_01` | hand-designed | reference | ✅ 36/40 | ❌ 0/40 | ✅ 37/40 | ❌ 0/40 | 21 |
| `windowsill_01` | hand-designed | reference | ✅ 26/40 | ❌ 0/40 | ✅ 36/40 | ❌ 0/40 | 17 |
| **total** | | | **10/10** nominal · **224/400** robust | **4/10** nominal · **58/400** robust | **10/10** nominal · **233/400** robust | **0/10** nominal · **2/400** robust | **84** |
| **mean score** | | | **0.560** | **0.140** | **0.583** | **0.000** | |

Planners:

- **Hybrid · shipped** (`hybrid`): Muse Spark states the goals, a search finds the motion (the shipped run, out/<scene>).
- **LLM-only · first plan** (`muse_first`): legacy arm: Muse Spark writes every motion, first try.
- **LLM-only · after log feedback** (`muse_after_log`): legacy arm: Muse Spark's plan after reading the physics log.
- **Scripted baseline** (`scripted_baseline`): straight lines in instruction order, 0.15 m carry height.

## Cached Evaluation v3 plans (20 repeats per scene, nominal only)

Re-simulated on the current code from `eval/results/eval_20261031_173045_round3_main.json` (English instruction, full-log condition). Graded against the case goals (reference goals + implicit `undisturbed`).

| case | LLM-only · first plan | LLM-only · after log feedback (≤ 2 revisions) |
|---|---|---|
| `desk_02` | 1/20 | 19/20 |
| `kitchen_01` | 0/20 | 20/20 |
| `lamp_01` | 12/20 | 20/20 |
| `storage_01` | 0/20 | 20/20 |
| `windowsill_01` | 1/20 | 20/20 |
| **pooled** | **14/100** | **99/100** |

## Notes

- **Hybrid · shipped** is the planner the product runs today. Its plan for each case is the final attempt of the shipped `out/<scene>/report.json` (one attempt on every table), scored here with the same scorer, seeds and noise as the LLM-only columns. Its robust total should equal the sum of the reports' own stress tests. The two LLM-only columns are the legacy arm, kept for comparison.
- Per-case cells score **one** cached plan per planner (the one the shipped product run produced), so a cell is an anecdote; the Evaluation v3 table above is the n = 20 view for the five scenes that have it.
- `muse_after_log` on agent-loop runs (`kitchen_01`, `real_ycbv59`, `aigen_*`) is the plan Muse Spark submitted after its own simulate / stress-test tool calls, so its feedback included stress results, not only the contact log.
- Goals marked *planner compiled* come from the recorded run (no hand-verified reference yet); a planner that compiled different goals is still scored on these.
- The scripted baseline is deliberately trivial: a floor, not a competitor. A 0.15 m straight carry is too low on every table here (it knocks the bottle on desk_01, presses on the kettle in both kitchens, sweeps the lamp), so it passes none; a smarter fixed-height heuristic is an obvious next baseline.
- Sanity check: on every case the `muse_after_log` robustness equals the stress test in that case's `report.json` (same plan, seeds and noise model), e.g. `real_ycbv59` 7/40, `kitchen_01` 37/40.
- Simulation only (MuJoCo, primitives, suction grasp). Small n. See bench/README.md, Limitations.
