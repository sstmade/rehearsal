# Hybrid planning: the LLM states the goal, search finds the motion, physics checks it

_Draft section. Source of every number: `eval/results/hybrid_20261102_011656.json`, produced by
`eval/hybrid_eval.py` (stage cache `eval/results/hybrid_work.json`, logs `log_hybrid.txt` for the live Muse
stage and `log_hybrid4.txt` for the final run). Everything here is simulated in MuJoCo._

**Thesis.** An LLM should not do motion planning. Muse Spark turns the instruction into checkable goals
and decides which object goes where. A deterministic search (`rehearsal/search_planner.py`) picks the
action order, carry heights and detours. The simulator verifies each candidate, then stress-tests the
winner under randomized friction, mass and perception error.

## 1. Headline table

The five hand-designed hard scenes are desk_02, kitchen_01, lamp_01, storage_01 and windowsill_01.

- **Units.** One task is one Muse run on a scene. There are 20 tasks per scene, 100 in total.
- **Scoring.** Every arm is scored on the scene's hand-verified reference goals plus implicit
  `undisturbed`. Only arm (c) is given those goals as input.
- **Robustness.** Every distinct successful final plan is re-run on held-out seeds 1000–1019. The noise
  model is the product's: friction ×U(0.6, 1.4), mass ×U(0.7, 1.3), per-object σ = 1.2 cm. Every arm uses
  the same seeds (common random numbers). The robustness figure is pooled over all (plan, seed) runs.

| arm | solved (95% CI) | sims per task | planner wall time | Muse $ per solved task | held-out robustness |
|---|---|---|---|---|---|
| (a) Muse first plan | 14/100 (9–22%) | 1 | median 46 s LLM latency | $0.0110 | 72% (202/280) |
| (b) Muse + full-log revise (≤ 2), EVAL v3 cached | 99/100 (95–100%) | 2.0 | median 93 s LLM + sims | $0.0028 | 79% (1556/1980) |
| (c) search only, from reference goals, no LLM | 5/5 scenes | 7.6 | 9 s mean, 26 s max | $0 | 52% (52/100) |
| (c+) search + 20-seed robustness gate | 5/5 scenes | 57 + 125 stress | 106 s mean, 318 s max | $0 | 80% (80/100) |
| (d) **hybrid**: Muse goals + plan hint, search fills geometry | **100/100 (96–100%)** | 6.9 | 9 s search + Muse latency | **$0.0015** | 72% (1429/2000) |
| (d+) **hybrid + 20-seed gate** | **100/100 (96–100%)** | 55 + 128 stress | 111 s mean, 328 s max, + Muse latency | **$0.0015** | **80% (1595/2000)** |

**How (b) and (d+) compare on robustness.** The comparison is paired: the same task, both arms solved
it, and the same seed. That gives 1,980 seed pairs. Hybrid + gate succeeded where revise failed on 66 of
them; revise succeeded where hybrid + gate failed on 40. Exact McNemar p = 0.015, in favour of hybrid +
gate. The pooled rates are close: 79.7% against 78.6%.

**Per-scene held-out robustness** (successes over plan × seed runs):

| scene | (b) Muse + revise | (c) search | (c+) search + gate20 | (d) hybrid | (d+) hybrid + gate20 |
|---|---|---|---|---|---|
| desk_02 | 59% | 70% | 70% | 66% | 66% |
| kitchen_01 | 85% | 55% | 85% | 85% | 85% |
| lamp_01 | 74% | 80% | 80% | 80% | 80% |
| storage_01 | 92% | 20% | 90% | 89% | 93% |
| windowsill_01 | **82%** | 35% | 75% | 38% | **76%** |

**The other five scenes** are desk_01, real_ycbv59, aigen_desk_lamp_mug, aigen_desk_walnut and
aigen_kitchen. Muse was called live: 10 repeats per scene, Contributor mode, the product's prompts.

| arm | solved (95% CI) | sims per task | Muse $ per solved task | held-out robustness |
|---|---|---|---|---|
| (a) Muse first plan | 33/50 (52–78%) | 1 | $0.0023 | 39% |
| (b) Muse + full-log revise | 50/50 (93–100%) | 1.3 | $0.0020 | 32% |
| (c) search only | 5/5 scenes | 3.4 | $0 | 32% |
| (d) hybrid | 50/50 (93–100%) | 3.2 | $0.0015 | 32% |
| (d+) hybrid + gate20 | 50/50 (93–100%) | 14.5 + 156 stress | $0.0015 | 32% |

- Muse's first plan failed on aigen_desk_walnut (0/10) and mostly on aigen_desk_lamp_mug (3/10).
  Hybrid solved all 20 of those tasks without a revision.
- Robustness on the four photo scenes is low for every arm: 15–25%. It is identical across arms because
  those scenes carry per-object perception σ of 1.3–3.9 cm. Most failures are suction misses caused by
  perception error, and no carry height or detour can fix those. The (a) figure of 39% is higher only
  because it averages over a different set of tasks: only first-try successes count, which over-weights
  desk_01 and real_ycbv59.
- desk_01 is 85% in every arm.

**Short forms that are safe to quote:**

- "On five hard scenes, Muse's own first plan failed 86 of 100 times. Muse's goals plus a geometric
  search solved 100 of 100, with no revision and no extra model calls, in a median of about 7 simulations.
  The search itself takes about 9 seconds."
- "With the stress-test gate, the hybrid's plans are as robust as the log-revised LLM plans: 80% vs 79%
  held-out (paired p = 0.015 in the hybrid's favour). Muse spend per solved task is half: $0.0015 vs
  $0.0028."

## 2. The first plan the search finds is brittle; the stress gate fixes most of it

Search alone always found a plan that passes the noiseless simulation, but that plan is often brittle.
Cheapest-first search returns the lowest carry height and the nearest detour that works nominally, which
leaves no margin.

| first plan found by search (no gate) vs reference fixed plan | held-out robustness |
|---|---|
| storage_01: search 20% · reference fixed plan 95% (19/20) | 4/20 |
| windowsill_01: search 35% · reference 85% (17/20) | 7/20 |
| kitchen_01: search 55% · reference 85% (17/20) | 11/20 |

Typical failures under noise:

- On windowsill, the forearm presses on the tray while lowering the mug (protective stop), or the frame
  next to the detour is knocked over.
- On kitchen, the hand presses on the kettle.

**The robustness gate.** Each nominal solution is stress-tested on *selection* seeds 0..n−1, which are
disjoint from the held-out seeds 1000+ used for reporting. The search keeps going until a solution reaches
90%, or until it has examined 10 solutions, and then keeps the best one.

- A later candidate is stress-tested only until it can no longer beat the current best plan, so the kept
  plan is always fully evaluated.
- With the gate, held-out robustness on the hard scenes rises from 52% to 80% for search alone, and from
  72% to 80% for the hybrid.

**10 vs 20 selection seeds on windowsill_01** (held-out seeds 1000–1019):

| windowsill_01 | no gate | 10-seed gate | 20-seed gate |
|---|---|---|---|
| search only (1 plan × 20 seeds) | 35% | 55% | 75% |
| hybrid (20 tasks × 20 seeds) | 38% (150/400) | 57% (226/400) | 76% (302/400) |

- With 10 seeds, the first solution with an alternative detour scores 9/10 on the selection seeds, so the
  gate stops early. That 9/10 is selection luck: held out, the same plan scores 55–57%.
- 20 selection seeds remove most of that optimism: held out, the chosen plan scores 75%.
- The extra cost is about 100 more stress simulations per task and about 25 s on windowsill.

**What is still not fixed:**

- windowsill_01 stays below the LLM-revised plans: 76% vs 82%.
- The search's detour candidates come from a coarse grid beside the straight path. The best LLM plans use
  a detour pulled toward the robot base (for example the reference via [0.45, −0.05] at 0.20 m), and the
  grid only approximates that point.

**Cost of the gate.** storage_01 is the expensive case: about 230 prefix simulations plus about 180 stress
simulations, and up to 5.5 minutes on a shared machine with 3 processes. The gate therefore suits offline
rehearsal, not an interactive loop.

The gate was strengthened twice during this experiment, and every run is reported:

| gate version | hybrid robustness on the hard scenes |
|---|---|
| 1. threshold early stop, at most 3 accepted alternatives per search node | 76% |
| 2. stop a candidate once it cannot beat the best plan | 76% (same plans) |
| 3. as 2, with at most 10 alternatives per node when the gate is on (shipped) | 80% |

The tables above use version 3. The intermediate result files from versions 1 and 2 were deleted as
superseded; this table keeps their pooled numbers.

## 3. Where search alone fails and the LLM is needed

Search needs goals, and the instruction is the only source of them. Physics cannot say what the user
meant.

- **Feasibility count.** I ran the search on every ordered "put X on Y" pair of objects in the five hard
  scenes, with a budget of 8 simulations per pair. **74 of 132 pairs are physically achievable**:

  | scene | achievable pairs |
  |---|---|
  | desk_02 | 18/30 |
  | kitchen_01 | 14/42 |
  | lamp_01 | 11/20 |
  | storage_01 | 20/20 |
  | windowsill_01 | 11/20 |

  So in every scene, many goals would "succeed" in simulation. The instruction is the only thing that
  says "the pen cup on the tray", not "the tray on the pen cup".
- **Muse's assignments.** In 150/150 tasks, Muse's object→destination assignment matched the reference
  exactly. This holds both for its plan's actions and for the merged assignment the search used. That
  covers stacking order ("largest at the bottom" → box_medium on box_large, box_small on box_medium) and
  the two-step aigen_desk_lamp_mug (coaster onto the laptop first, then the mug onto the coaster).
- **A gap in Muse's goals.** Its compiled goals did *not* match the reference predicates in most tasks:
  only 8 of 150 contained all of them. Muse states the placements (`on`, `inside`, `stacked`) but usually
  omits `upright` / `stable`. This did not cost a single task, because the suction transport keeps objects
  upright and scoring used the full reference goals. But the search only enforces the goals it is given:
  a constraint the LLM leaves out (say "keep the cup upright" for a tippy object) would not be searched
  for.
- **Implicit goals come from code, not from the LLM.** `undisturbed` for every unmentioned object is added
  by `with_implicit_goals`.

**Division of labour, from the data:**

- The LLM is reliable at intent: 150/150 correct assignments.
- The LLM is unreliable at geometry: 86/100 first plans failed on the hard scenes.
- Search is reliable at geometry: 100/100 given Muse's goals.
- Physics is the judge, both for nominal success and for robustness.
- The only Muse call the hybrid needs beyond goal compilation is the plan hint. It supplies the order to
  try first and the first geometry guess, and it is optional: `--planner search` skips it.

## 4. What was built

- `rehearsal/search_planner.py`. Deterministic, makes no model calls.
  - **Orders:** the hint order first, then the others by Kendall distance. Orders that put something
    onto an object that is moved later are pruned.
  - **Candidates per action:** carry height in {0.15, 0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.45} × (no
    detour or one of about 13 via points), cheapest first. The LLM's own numbers are tried first.
  - **Kinematic dry run before any physics:** IK every 2 cm along the exact `transport` waypoints, seeded
    like the executor, plus a MuJoCo collision query of the arm and the held object against the other
    objects. It starts from the object poses the previous simulation ended in. Out-of-reach candidates
    are pruned, and the remaining candidates are tried in three passes: clear (≥ 3 cm), then near miss,
    then colliding.
  - **Depth-first prefix simulation:** each partial plan is checked against the goal predicates that can
    be decided so far. Failures feed back simple height prunes.
  - **Optional robustness gate:** see section 2.
- `rehearsal/rehearse.py`: `--planner llm|hybrid|search`, which is also the `planner` setting of
  `rehearse()`, and `plan_with_search()`.
  - Muse is asked again (`revise`, with a description of the failed search) only when the search fails.
  - If the second search also fails, the old revise loop takes over.
  - The default `llm` mode is unchanged. Offline mode (`REHEARSAL_OFFLINE=1`) works with all three modes.
- `rehearsal/pipeline.py` and `rehearsal/server.py`: an optional `planner` field on rehearse jobs.
- `rehearsal/executor.py`: `run_plan` also returns `final_poses`.
- `tests/test_search_planner.py`: 12 test cases.

## 5. Caveats

- **Simulation only.** MuJoCo, a suction-weld grasp, primitive or mesh geometry. "Robustness" means this
  project's noise model, not real-world success.
- **The scenes were built to be hard for an LLM's geometry guesses.** They were not built to be hard for a
  grid search. A cluttered table that needs a long detour, or a scene where the coarse via grid has no
  good point, could defeat the search. windowsill_01 already shows the grid's limit on robustness.
- **Search-only arms have n = 1 per scene.** The search is deterministic, so with reference goals there is
  one plan per scene. Its robustness rests on 20 seeds per scene, which makes per-scene figures coarse
  (±20 points). Hybrid arms have 20 tasks per hard scene, but on several scenes the search converges to
  the same plan.
- **Arms (a)/(b) on the hard scenes are cached EVAL v3 runs** (Muse Spark 1.3 Contributor, default effort,
  English instructions, 2026-10-31). They were not re-run.
  - Their dollar figures are that run's ledger spend, attributed by caller tag.
  - The hybrid on those scenes reuses the same cached goals and plan, so it is charged the same
    goals + plan spend.
  - LLM latencies come from that run (12-way parallel). (a)'s median comes from first-try-success cells
    only.
  - The search wall times were measured on this shared machine with 3 processes, so wall-time
    comparisons across arms are rough.
- **Robustness for (b) and (d) covers different plan sets.** Pooled robustness counts each arm's
  successful final plans: 99 tasks for (b), 100 for (d). The paired McNemar test uses only tasks both
  arms solved.
- **The reference goals for real_ycbv59, aigen_desk_lamp_mug, aigen_desk_walnut and aigen_kitchen were
  written by me** (the hybrid engineer) from each scene's `demo_instruction`, in `eval/hybrid_tasks.json`.
  **Nobody else has reviewed them.** desk_01's reference goals are the existing ones from
  `eval/tasks.json`.
- **The photo scenes say little about planners.** Robustness there is set by perception σ, not by the
  plan, so the numbers are identical across arms. They show that the hybrid does no harm on those scenes,
  not that it helps.
- **The gate was tuned while looking at these scenes.** Three iterations, all reported in section 2. The
  held-out seeds were never used for selection.
- **Cost.** New Muse spend for this experiment: **$0.0993** over 111 paid calls, all for the extra scenes
  (ledger callers `hybrid.*`, Contributor mode). The hard-scene numbers used cached responses. The global
  budget counter rose more than that during the run because other agents were spending concurrently.
