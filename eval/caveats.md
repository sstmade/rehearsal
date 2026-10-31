## Method & caveats (read before quoting any number)

- **Everything here is simulated.** Scenes are hand-designed tabletop layouts (desk_01, kitchen_01,
  windowsill_01) built as MuJoCo primitives (cylinders, boxes, a segmented bowl) with a Franka Panda from
  MuJoCo Menagerie. No result in A–C involves a real robot or a real photo. Grasping is a suction weld
  (always succeeds once the cup is on the top face); real grasp physics is out of scope.
- **The noise model is assumed, not measured.** Stress tests draw friction ×U(0.6, 1.4), mass ×U(0.7, 1.3)
  and per-object xy position error N(0, σ) per axis with σ = 1.2 cm (the product's default), independent
  across objects. Real perception error is neither Gaussian nor independent (a bad A4 corner shifts every
  object together). Section B shows how strongly every robustness number depends on σ: treat "x/40" as
  "at σ = 1.2 cm", not as a real-world success rate. The only measured perception numbers (D) come from
  synthetic renders.
- **Seeds.** Seed s fixes friction, mass and the unit error direction (common random numbers), so two plans
  or two σ settings are compared on identical draws (paired). Section A robustness uses seeds 0–39 (as the
  product). Sections B and C score on **held-out seeds from 1000** that were never used to select a plan or a
  re-observation. Selecting and scoring on the same 40 seeds inflated the old desk_01 claim (35/40 → 40/40);
  on held-out seeds it is ~89% → ~96% (research docs/claims.md §4; reproduced in C).
- **Small n.** Muse Spark matrix: n = 5–10 repeats per (task, condition); offline agent is deterministic
  (n = 1 per task). Quote counts with their n; Wilson 95% intervals are given in B. With n = 10 a difference
  of 2–3 successes is not significant.
- **Repeats and caching.** Muse responses are cached by exact request; repeat k > 0 appends an inert
  "[eval trial k]" line to each prompt to obtain an independent sample (repeat 0 is the product's prompt).
  Re-running the matrix is free and returns identical results; it does not re-sample the model.
- **Paired ablation.** Within one repeat, goal compilation, the first plan and attempt 1 are shared by all
  three feedback conditions; only the revise input differs. When attempt 1 succeeds the conditions are
  identical by construction (counted once per condition).
- **Two success criteria.** "≤2 rev" stops on the agent's own compiled goals (what the product does);
  "ref" re-checks the same simulation against hand-written reference goals (eval/tasks.json or the
  scene's `reference.goals`), which guards against an agent compiling easy goals.
- **Data mode / cost.** All three scenes are `"privacy": "public"` (hand-designed, no user data), so Muse
  Spark ran in Contributor mode (muse-spark-1.3-contributor, ~20× cheaper). Costs come from the shared
  ledger entries tagged `eval.*`.
- **Code under test.** Each result file records the code it ran on (`code#<hash of rehearsal/*.py>`).
  The `muse_kitchen` / `muse_desk_windowsill` runs used a read-only snapshot of the robotics branch
  (979d772, pre-merge; same scenes/executor as main after the merge). Its ACTION_SCHEMA has no `via`
  field, so on windowsill_01 Muse could not express the reference detour; it instead found a two-hop relay
  (set the mug down on an intermediate reachable spot, then carry it to the tray at a lower height) using the
  reach limits quoted in the IK evidence. `muse_agentsnap` runs on a snapshot of the agent branch (via in the
  schema, compact revise history) and is the closest to what will ship.
- **Compute budget.** Run on a shared, heavily loaded laptop: Experiment B uses n = 60 held-out seeds for
  plans that pass at σ = 0 (n = 20 for plans that already fail at σ = 0) and six σ levels; Experiment C uses
  200 held-out seeds (1000–1199). Scripts take `--n` / `--n-heldout` to rerun larger.
- **Not measured yet:** diagnosis correctness (does the written diagnosis name the true colliding pair /
  limit), and the agent engineer's tool-loop agent (`--agent-loop`), which is evaluated on their side.
