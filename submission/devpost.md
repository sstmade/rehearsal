# Submission form text (English)

Every number below traces to `docs/goalbench.md`, `docs/hybrid.md`, `docs/real_robot.md`, `docs/perception.md`,
`EVAL.md` or `docs/intent.md`. `TODO(user)` items are URLs and team. Do not submit with any left.

---

## Project name

Rehearsal

## Tagline (≤ 60 chars)

Words don't collide. The real world does.

Alternative: *Models propose. Physics decides.*

## Summary (one paragraph)

Rehearsal is a pre-flight check for robot instructions, measured in simulation. Before a robot touches your things,
it confirms what you meant, tries the plan on a physics copy of your table, and tells you what it assumed, what it
checked, and what it can't promise. We started out letting Meta's Muse Spark plan the motion and fix
it from a physics log. Our own data took that story apart: a plain search with no model calls solved every hard
table in seconds. So each part now does the one job it is good at. Muse Spark turns your words into goals a
simulator can check, lists its assumptions, and may ask or refuse. A search finds the motion. MuJoCo checks every
candidate and stress-tests the winner. In simulation, on five hard tables,
Muse's first plan solved 14 of 100 runs (5 tasks × 20); Muse's goals plus search solved 100 of 100. And we measured a
blind spot no simulator can close by itself: of 55 wrong goal compilations, physics certified 17 as success, all of
them guesses on ambiguous instructions where the model should have asked. In one hand-picked case, "put the soup next
to the mouse" became "keep the soup upright". The robot did nothing, and physics said ✓. We also checked the physics
copy against 23 real Franka episodes from 7 labs. It is not good enough yet: from one photo, objects are off by a
median 5.5 cm at the grasp, more than the gripper's 4.25 cm margin, so the replay grasps air. That is why Rehearsal
states its uncertainty and asks for a closer look, and why we don't claim real-arm readiness.

## Inspiration

A green tick on the wrong goal. We asked Muse Spark to "put the tomato soup next to the mouse". It wrote goals that
said nothing about the mouse. The planner did nothing. The simulator checked the goals, and they held. Success. The
soup was still 22 cm away.

That was the second thing that surprised us. The first was worse for our pitch: our headline result ("the model fixes
85 of 86 failed plans when it reads the physics log") was real, but a dumb grid search solved the same tables in
seconds. The model was mostly guessing carry heights. So we rebuilt the project around what each part is actually
good at, and we publish the results that disproved us.

## What it does

1. **Photo → physics copy.** Take one phone photo with an A4 sheet on the table. Muse Spark lists the objects and
   estimates size, mass and friction ranges. SAM 3.1 segments each one. The sheet fixes the scale. The result is a
   simulation-ready approximation in MuJoCo with a Franka arm and a position uncertainty per object.
2. **Words → checkable goals.** Muse Spark compiles the instruction into predicates (`on`, `inside`, `stacked`,
   `upright`, `stable`, `undisturbed`, `near`, `clear`). It must list its assumptions for you to confirm. It may ask
   one question ("Which book should I put in the bowl: the large hardcover, medium paperback, or small notebook?") or
   refuse with a reason ("The robot can only suction-pick and transport and has no cutting tool"). A no-LLM gate
   refuses empty goals and unknown objects, and asks when the goals already hold before the robot moves.
3. **Search → motion.** A deterministic planner tries orders, carry heights and detours, with an IK and collision dry
   run before any physics.
4. **Physics → verdict.** Every candidate is simulated and checked by code. The winner is re-run under randomized
   friction, mass and position error, and kept only if it survives.
5. **Report.** What physics checked, what was estimated, what was assumed, and which object is worth a second look.
6. **Reality check (a limitation we measured).** We replayed 23 real Franka episodes from DROID (7 labs) in the physics
   copy and compared the predicted outcome with the filmed one. The copy is not accurate enough yet; see below.

## How we built it

- **Meta Muse Spark 1.3**: intent (goals, assumptions, clarify / refuse; structured output), an optional plan hint for
  the search, a revise step only when the search fails, perception (object list, size, mass, friction), and the
  legacy agent loop where it drives `simulate` / `stress_test` / `look_again` / `submit` as function tools.
- **Meta SAM 3.1**: concept segmentation per noun phrase with a point-prompt fallback; lossless masks fitted by a 3D
  shape's silhouette (median placement error 0.41 cm on synthetic renders, 1.71 cm on 83 real objects with ground
  truth).
- **Meta SAM 3.1 video**: tracks every object through each real DROID clip to measure where it actually ended up.
- **Meta SAM 3D Objects** (run locally via the community MLX port, 8-bit SAM-License weights): look-alike meshes for
  display; physics uses simple shapes.
- **MuJoCo** with the Franka Panda from MuJoCo Menagerie; a search planner in plain Python; three.js for the replay.
- One Meta API client with an on-disk cache, a hard budget and a ledger of every call.

## Challenges we ran into

- **Our headline fell over.** A zero-LLM search solved the five hard tables that we had built to show off the model's
  repair loop. We moved geometry to search and measured the split instead of arguing for it.
- **The search's first plan was brittle.** It picks the lowest height that works once. On storage_01 that plan
  survived 20% of noisy runs, against 95% for the reference. A stress gate fixed most of it; we report all three
  versions of the gate.
- **The verifier blind spot.** Physics can only check the goal it is given. We had to label intent by hand (60
  cases), then get a blind hold-out written by someone who had seen nothing.
- **Calibrating against a real robot.** Stereo estimates of DROID's table height disagreed by several
  centimetres across recordings of the same unchanged scene. We bounded it with the robot itself: fingertips can never go below the table.

## Accomplishments that we're proud of

- **Intent, measured.** With the old prompt, 40 of 60 instructions compiled to what the user meant (67%, CI 54–77%),
  and 0 of 24 infeasible trials were refused. With assumptions and permission to ask or refuse: 60 of 60 on the dev
  set (an upper bound) and 28/30 (CI 79–98%) on a 30-case blind hold-out written by one other person (old prompt
  21/30; paired p = 0.065). C's two misses look like label errors but count as misses under rules fixed beforehand.
- **Geometry, handed to search.** On five hard tables in simulation: Muse's first plan 14/100 (CI 9–22%), hybrid
  100/100 (CI 96–100%), about 7 simulations and 9 s of search per task. With the stress gate, held-out robustness
  80% vs 79% for log-revised LLM plans (paired McNemar p = 0.015), at half the Muse spend per solved task ($0.0015 vs
  $0.0028).
- **A real-robot check that told us the truth.** On 17 DROID episodes from 7 labs, scored by rules committed before
  any result, the physics replay lost to "the object ends where the gripper let go" (+5.0 cm mean error, CI +0.9 to
  +9.7, p = 0.007; moved/not-moved 37/51 vs 46/51) and invented motion that "nothing moves" avoids. The diagnosis is
  specific: the simulated gripper closed on air in 16 of 17 episodes, because the copy's object sat a median 5.5 cm
  from the real grasp. Our first 6 episodes (5 of 6 matched) had looked fine; trivial rules matched those too.
- **Honesty as a feature.** We publish the results that disproved us, twice: the log-repair headline and the
  real-robot check. Also the places where we still lose (windowsill robustness 76% vs 82%).

## What we learned

- Language models are good at intent: Muse Spark chose the right object and destination in 150 of 150 runs (10 instructions). They are
  poor at geometry: 86 of 100 first plans failed on the hard tables.
- Physics can't pick intent: 74 of 132 "put X on Y" pairs on those tables are physically achievable. Only the
  instruction says which one you meant.
- A verifier is only as good as its goal. A model that can say "I'm not sure" or "I can't" closes more of that gap than
  a no-LLM check. Our deterministic gate flags 0 of 55 wrong compilations; it only catches the "robot does nothing"
  cases (6 trials turned into questions), with 0 false flags on 173 correct ones.
- Real footage keeps you honest. The weak link between a photo and a real grasp is not contact physics; it is a few
  centimetres of position error, which decides whether the grasp catches anything at all.

## What's next

- **Real-arm transfer, with a number to hit:** first-frame object positions inside the gripper's margin (median error
  well under 4.25 cm at the grasp; about 1–2 cm to beat "where the gripper let go"), re-scored on the same 17 DROID
  episodes and the same pre-registered baselines.
- A sampling-planner baseline (RRT-Connect, random restarts) at equal budget.
- Calibrating the noise model from real residuals instead of assuming it.
- Running Rehearsal's own plans on a real arm.
- GoalBench as an installable eval, with a second labeller.

## Built with

Meta Muse Spark 1.3 · Meta SAM 3.1 (image and video) · Meta SAM 3D Objects · MuJoCo · MuJoCo Menagerie (Franka Panda)
· DROID dataset (CC BY 4.0) · Python · three.js

## Meta tools used, and how

| Tool | What it does in Rehearsal |
|---|---|
| Muse Spark 1.3 | Intent compiler with assumptions, clarify and refuse; plan hint; revise-on-search-failure; object listing; agent loop with function tools; structured output throughout; Contributor mode for public scenes, Standard for uploads |
| SAM 3.1 | Per-object masks from noun phrases, fitted into 3D shapes |
| SAM 3.1 video | Measures where real objects ended up in DROID clips |
| SAM 3D Objects | Display meshes for the physics copy |

Total Meta API spend: under $6 for the whole project ($4.43 in `budget/ledger.jsonl` on Nov 2, including a conservative $1.00 placeholder for a short lost-ledger window).

## Honest limits

Everything that plans was evaluated in simulation. No robot ran our plans; the DROID check replays recorded human
teleoperation (23 episodes, 7 labs), and there the physics copy does not yet beat trivial baselines
because single-photo position error is too large. GoalBench labels are ours; the hold-out is 30 cases from one author. One model, one
action type, a suction grasp, simple shapes. The tables were built to be hard for a model's geometry, not for search.

## Credits

Robot footage: DROID (Khazatsky et al., RSS 2024), CC BY 4.0. YCB-Video frame: MIT (PoseCNN, Xiang et al., RSS 2018).

## Links

- Live demo: TODO(user): live URL
- Video: TODO(user): video URL
- Code: TODO(user): GitHub URL

## Team

TODO(user): names, roles, contact.
