# Submission form text (English)

All numbers below are from EVAL.md, docs/agent.md, docs/perception.md and `experiments/agent/`.
Remaining `TODO(user)` items are the user's: URLs and team. Do not submit with any TODO left.

---

## Project name

Rehearsal

## Tagline (≤ 60 chars)

Models propose, physics decides.

Alternative: *Rehearse a robot task on your own desk, from one photo.*

## Summary (one paragraph)

Rehearsal turns one phone photo of a desk into a physics twin, a simulation-ready approximation of the table
in MuJoCo with a Franka Panda arm, then makes a language model's plan prove itself before anyone trusts it
on real objects. Muse Spark lists the objects and estimates their size, mass and friction; SAM 3.1 segments
them; an A4 sheet in the photo gives metric scale. Muse Spark compiles your instruction into predicates the
simulator can check (inside, on, stacked, upright, stable; code adds "don't disturb anything else"), plans
the moves, and, when physics says no, reads the simulator's log and fixes the plan. That log is what makes
the difference: with it, Muse Spark repaired 17/17 failed first plans on four hard scenes; told only "it
failed", 7/17. The final plan is replayed in 40 parallel universes with perception noise, and Rehearsal tells
you which object is worth a second look: on our demo desk, re-observing the mug raises held-out success from
178/200 to 191/200 and breaks nothing.

## Inspiration

Home robots are evaluated in labs and deployed in homes, and every home is different. The research answer
is "test in simulation first", but building a sim of a real room still takes scanners, CAD and an engineer.
We wanted the opposite end: a photo and a sheet of paper.

We were also struck by how confidently language models plan physical tasks they cannot check. On our
windowsill scene Muse Spark's first plan asked the arm to carry a mug at 0.50 m above a tray 0.79 m from the
base, where the Panda can reach only 0.26 m; it failed first try in 15 of 15 runs. A simulator will not let
it. So we split the job: models propose, physics decides.

The "look again" step comes from Adam W. Harley and colleagues' work on active perception
(*Move to See Better*, BMVC 2021) and embodied rearrangement (*TIDEE*, ECCV 2022): an agent should choose
what to observe based on what it needs to do. We apply that to a twin: which measurement would most improve
the chance the task succeeds?

## What it does

1. **Photo → twin.** Upload a photo with an A4 sheet on the table. Muse Spark names each movable object and
   estimates dimensions, a mass range and a friction range. SAM 3.1 segments each one from a short noun
   phrase, falling back to point prompts for anything it misses (flagged low confidence). A homography on
   the A4 corners maps pixels to table metres. You get a MuJoCo scene with a Franka Panda.
2. **Instruction → checkable goals.** "Put the mug in the bowl, then stack the books" becomes
   `inside(mug, bowl) ∧ upright(mug) ∧ stacked(…) ∧ stable(…)`; Rehearsal's code then adds `undisturbed`
   for every object the instruction did not mention, so the model cannot forget it. Code evaluates all of
   them on simulator ground truth.
3. **Plan, fail, read the log, fix.** Muse Spark plans transport actions (with waypoints). If a predicate
   fails, it receives the evidence: timestamped contacts with link roles ("t = 6.138 s gripper body ↔
   kettle_1, 62 N, protective stop"), IK reach limits found by bisection, measured offsets. It writes a
   diagnosis and a new plan, up to three attempts.
4. **Or the model drives.** In agent-loop mode Muse Spark holds four function tools, `simulate`,
   `stress_test`, `look_again` and `submit`, and chooses the next step itself; `submit` re-simulates and
   rejects a failing plan.
5. **40 parallel universes.** The final plan is re-run with per-object position noise (σ = 1.2 cm, or the
   photo pipeline's per-object σ), friction ×0.6–1.4 and mass ×0.7–1.3, using common random numbers so every
   comparison is paired.
6. **Look again.** For each object, Rehearsal re-runs the universes as if that object had been re-observed
   (σ 1.2 → 0.4 cm) and ranks objects by how much robustness improves, scored on held-out seeds.
7. **Report and 3D replay.** Verified by physics, estimated under uncertainty, and assumed, kept separate.

## How we built it

- **Physics:** MuJoCo via `MjSpec`, Franka Panda from MuJoCo Menagerie, damped-least-squares IK, Cartesian
  transport primitive, suction modelled as a weld, contact events logged with timestamps and body names.
- **Predicates:** six small geometric checks (`inside`, `on`, `stacked`, `upright`, `stable`, `undisturbed`),
  each a few lines over simulator state.
- **Agent:** Muse Spark 1.3 with JSON-schema structured output for three roles (compile goals, plan,
  diagnose/revise), plus an opt-in tool-calling loop (`simulate` / `stress_test` / `look_again` / `submit`)
  where the model decides what to run next. The system prompt states the one action, the frame, and that
  success is checked by a simulator, not the model.
- **Perception:** Muse Spark for object listing and physical estimates; SAM 3.1 concept prompts in parallel,
  matched to Muse Spark's object centres, with point-prompt fallback; A4 homography for scale; SAM's lossless
  mask fitted by a 3D primitive's silhouette; a per-object σ derived from mask fit quality.
- **Robustness:** process-pool stress test with paired seeds; greedy value-of-information over objects,
  selected on seeds 0–39 and scored on held-out seeds.
- **Web app:** Python HTTP server with photo upload and job polling, and a three.js replay of recorded 50 Hz
  body poses with collision highlights, the tool path and stress-test trails; a demo mode for the video.
- **Budget safety:** every Meta API call goes through one client that caches identical requests, reserves a
  worst-case cost under a file lock against a hard cap, and logs real usage. Re-runs and UI work cost $0.

## Use of Meta AI

| Model | What it does in Rehearsal |
|---|---|
| **Muse Spark 1.3** | Perception: object list, SAM noun phrases, dimensions, mass and friction ranges |
| **Muse Spark 1.3** | Compiles the instruction into simulator-checkable predicates (strict JSON schema) |
| **Muse Spark 1.3** | Plans transport actions with waypoints |
| **Muse Spark 1.3** | Reads the simulation log, diagnoses failures, revises the plan |
| **Muse Spark 1.3**, function tools | **Agent loop:** drives `simulate` / `stress_test` / `look_again` / `submit`; `submit` is physically gated |
| **SAM 3.1** | Concept segmentation from noun phrases; point-prompt fallback; lossless masks fitted by primitive silhouettes |
| **SAM 3.1 video** | Reality check: tracks a phone clip after execution and compares final positions with the simulation's prediction |

**Does the model actually use the log?** We tested it. Same failed first plan, three revise inputs, four hard
scenes, five repeats (docs/agent.md, run 2): full simulator log **17/17** repaired within two revisions;
failed-predicate names only 8/17; "it failed, try again" 7/17. On the kitchen scene, where the fix is to move
the kettle *before* the cup, it was 5/5 with the log and 0/5 without. An independent eval on the windowsill
scene: 15/15 vs 5/15 (Fisher p = 0.0002).

**The agent loop, one real run** (windowsill, `experiments/agent/loop_windowsill_01.json`, $0.0028, 7 tool
calls, 79% of prompt tokens from the prompt cache): the model's first `simulate` fails on reach ("the tool tip
can reach at most z = 0.40 m"); it lowers the carry and routes around the vase, two `simulate` successes,
`stress_test` 18/20, tries lower still (17/20), and `submit`s the better plan, which physics accepts. On the
kitchen scene it moved the kettle first unprompted and chose `look_again(cup_1)`, 17/20 → 20/20, the same
object the exhaustive value-of-information search picks.

Contributor mode was used only for synthetic renders and public Wikimedia photos; user photos use Standard
mode. All calls go through one client with a disk cache, a hard budget cap and a ledger: the whole project
cost $2.41 in Meta API calls.

## Challenges we ran into

- **Making failure honest.** On our first desk scene Muse Spark planned correctly on the first try, which is
  good for Muse Spark and useless as a demo of self-correction. Scripted failures would be dishonest, so we
  built scenes where the failure comes from physics the model cannot see in a summary: the elbow sweeping a
  neighbouring object, targets at the edge of reach, stacks that look fine and topple, paths that need a
  detour. Five shipped: `kitchen_01` (the gripper body lands on the kettle unless the kettle moves first),
  `windowsill_01` (tray at the edge of reach, tall vase in the straight path), `lamp_01` (the wrist sweeps a
  0.48 m lamp), `storage_01` (a stack built first blocks the cup's path), `desk_02` (reach limit at the far
  stapler). Muse Spark's first plan failed on windowsill 15/15 times, kitchen 12/20, lamp 2/5 and storage 5/5
  (desk_02 was not measured with Muse Spark).
- **Scale from one photo.** Muse Spark's size estimates are plausible but not metric. The A4 homography
  anchors scale; Muse Spark's estimates fill in height and shape.
- **SAM coverage.** Concept prompts miss some objects in cluttered scenes; point prompts at Muse Spark's
  object centres recover most of them, and we mark those as low confidence so the stress test treats them
  with more suspicion.
- **What "looking again" means without a robot.** We model it as a smaller σ for one object and measure the
  change in paired universes. In the product it becomes a request for one more photo.
- **A $15 night.** Caching and a budget-reserving client let several people iterate on prompts without
  surprise spend.

## Accomplishments that we're proud of

- The model never grades itself. Every "success" is a predicate on simulator state.
- A real log-driven fix, Muse Spark 1.3 verbatim on the kitchen scene: "Evidence shows tool tip at z=0.21m,
  below kettle top 0.25m, and the gripper body above the tip hit the kettle top-edge at [0.64,0.06,0.25] with
  62N. This is a lowering-phase collision, not a transit collision, so higher transit_height or via waypoints
  cannot avoid it because the final vertical descent is the same. The obstruction must be removed first."
- The log ablation: 17/17 repaired with the simulator's evidence vs 7/17 without.
- Active perception that gives a concrete answer and survives held-out seeds: look again at the mug,
  178/200 → 191/200, 13 universes fixed, none broken.
- Numbers: perception error 0.41 cm median / 1.05 cm P90 on renders, 1.71 / 6.18 cm on real YCB-Video and
  HOPE photos (83 objects); first-try success on the hard scenes 0/15 (windowsill), 8/20 (kitchen) → after
  ≤ 2 revisions with the log 15/15 and 20/20; whole evaluation $0.23 of Meta API calls.

## What we learned

- Language models are good at turning instructions into specifications and at reading a log for a cause.
  They are poor judges of their own physical plans. Giving them structured evidence (which bodies touched,
  when, how far off) works better than asking them to imagine. The ablation qualifies this: where the fix is
  a one-parameter search (carry height), blind retries often get there too; where it is structural (reorder
  the actions), the log is the difference between 0/5 and 5/5.
- Uncertainty is more useful when it is actionable. "178/200" is a number; "look again at the mug" is a decision.
- On our demo desk, perception mattered more than dynamics: shrinking only the mug's position error fixed 13
  of the 22 failing held-out universes and broke none, even though friction and mass still varied. Modelling
  error per object is what made the stress test informative. A caveat we publish: a simple heuristic
  (re-observe the object named most often in failed predicates) picks the same object on all three scenes.

## What's next

- **Second look for real:** ask the user for one more photo of the object Rehearsal flags, and fuse it.
- **Meshes instead of primitives:** SAM 3D Objects for appearance, convex decomposition for collision.
- **More robots:** swap the Panda for a mobile manipulator from MuJoCo Menagerie; the scene builder is
  robot-agnostic.
- **Real-robot validation:** run the same plans on a real arm and measure how often "40/40 in sim" holds.
- **A dataset of real homes:** every rehearsal is a (scene, instruction, plan, failure, fix) trace with a free,
  verifiable reward from the simulator, useful for training open planners.

## Links

- GitHub: TODO(user): public repo URL
- Live demo: TODO(user): Hugging Face Spaces URL
- Video: TODO(user): video URL
- X post: TODO(user): X post URL

## Team

TODO(user): names, roles
