# Submission form text (English)

Placeholders: `{{EVAL: …}}` = number from EVAL.md, `{{CONFIRM: …}}` = commander to confirm,
`{{TODO-USER: …}}` = the user fills in. Do not submit with any `{{…}}` left.

---

## Project name

Rehearsal

## Tagline (≤ 60 chars)

Models propose, physics decides.

Alternative: *Rehearse a robot task on your own desk, from one photo.*

## Summary (one paragraph)

Rehearsal turns one phone photo of a desk into a MuJoCo physics twin with a Franka Panda arm, then makes a
language model's plan prove itself before anyone trusts it on real objects. Muse Spark lists the objects
and estimates their size, mass and friction; SAM 3.1 segments them; an A4 sheet in the photo gives metric
scale. Muse Spark compiles your instruction into predicates the simulator can check (inside, on, stacked,
upright, stable, and "don't disturb anything else"), plans the moves, and, when physics says no, reads the
simulator's contact log and fixes the plan. The final plan is replayed in 40 parallel universes with
perception noise, and Rehearsal tells you which object is worth a second look: on our demo desk, re-observing
the mug raises robustness from 35/40 to 40/40.

## Inspiration

Home robots are evaluated in labs and deployed in homes, and every home is different. The research answer
is "test in simulation first", but building a sim of a real room still takes scanners, CAD and an engineer.
We wanted the opposite end: a photo and a sheet of paper.

We were also struck by how confidently language models plan physical tasks they cannot check. A model will
happily carry a mug at 14 cm over a 26 cm bottle. A simulator will not let it. So we split the job: models
propose, physics decides.

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
   `inside(mug, bowl) ∧ upright(mug) ∧ stacked(…) ∧ stable(…)`, plus an implicit `undisturbed` for every
   other object. Code evaluates these on simulator ground truth.
3. **Plan, fail, read the log, fix.** Muse Spark plans transport actions (with optional waypoints). If a
   predicate fails, it receives the evidence: timestamped contacts ("t = 3.026 s bottle_1 ↔ robot_arm"),
   IK errors, measured offsets. It writes a diagnosis and a new plan, up to three attempts.
4. **40 parallel universes.** The final plan is re-run with per-object position noise (σ = 1.2 cm),
   friction ×0.6–1.4 and mass ×0.7–1.3, using common random numbers so every comparison is paired.
5. **Look again.** For each object, Rehearsal re-runs the universes as if that object had been re-observed
   (σ 1.2 → 0.4 cm) and ranks objects by how much robustness improves.
6. **Report and 3D replay.** Verified by physics, estimated under uncertainty, and assumed, kept separate.

## How we built it

- **Physics:** MuJoCo via `MjSpec`, Franka Panda from MuJoCo Menagerie, damped-least-squares IK, Cartesian
  transport primitive, suction modelled as a weld, contact events logged with timestamps and body names.
- **Predicates:** six small geometric checks (`inside`, `on`, `stacked`, `upright`, `stable`, `undisturbed`),
  each a few lines over simulator state.
- **Agent:** Muse Spark 1.3 with JSON-schema structured output for three roles (compile goals, plan,
  diagnose/revise). The system prompt states the one action, the frame, and that success is checked by a
  simulator, not the model.
- **Perception:** Muse Spark for object listing and physical estimates; SAM 3.1 concept prompts in parallel,
  matched to Muse Spark's object centres, with point-prompt fallback; A4 homography for scale.
- **Robustness:** process-pool stress test with paired seeds; greedy value-of-information over objects.
- **Web app:** Python HTTP server and a three.js replay of recorded 50 Hz body poses with collision
  highlights, the tool path and stress-test trails. {{CONFIRM: upload flow, demo mode}}
- **Budget safety:** every Meta API call goes through one client that caches identical requests, reserves a
  worst-case cost under a file lock against a hard cap, and logs real usage. Re-runs and UI work cost $0.

## Meta tools used

| Model | What it does in Rehearsal |
|---|---|
| **Muse Spark 1.3** | Perception: object list, SAM noun phrases, dimensions, mass and friction ranges, fragility |
| **Muse Spark 1.3** | Compiles the instruction into simulator-checkable predicates |
| **Muse Spark 1.3** | Plans transport actions |
| **Muse Spark 1.3** | Reads the simulation log, diagnoses failures, revises the plan |
| **SAM 3.1** | Concept segmentation from noun phrases; point-prompt fallback |

Contributor mode was used only for synthetic renders and public Wikimedia photos; user photos use Standard mode.

## Challenges we ran into

- **Making failure honest.** On our first desk scene Muse Spark planned correctly on the first try, which is
  good for Muse Spark and useless as a demo of self-correction. Scripted failures would be dishonest, so we
  built scenes where the failure comes from physics the model cannot see in a summary: the elbow sweeping a
  neighbouring object, targets at the edge of reach, stacks that look fine and topple, paths that need a
  detour. {{CONFIRM: list the hard scenes that actually shipped}}
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
- A real log-driven fix: "at t = 3.026 s the mug hit bottle_1 (0.26 m tall); the mug hangs 0.10 m below the
  cup, so transit height must exceed 0.36 m; use 0.42 m." {{CONFIRM: swap for the Muse Spark hard-scene example}}
- Active perception that gives a concrete answer: look again at the mug, 35/40 → 40/40.
- Numbers: perception error {{EVAL: median / P90 cm}}, first-try {{EVAL: first-try success rate}} →
  after revision {{EVAL: success after revision}}, log ablation {{EVAL: full log vs no feedback}}.

## What we learned

- Language models are good at turning instructions into specifications and at reading a log for a cause.
  They are poor judges of their own physical plans. Giving them structured evidence (which bodies touched,
  when, how far off) works better than asking them to imagine. {{EVAL: log ablation supports / qualifies this}}
- Uncertainty is more useful when it is actionable. "35/40" is a number; "look again at the mug" is a decision.
- On our demo desk, perception mattered more than dynamics: shrinking only the mug's position error fixed every
  failing universe, even though friction and mass still varied. Modelling error per object is what made the
  stress test informative.

## What's next

- **Second look for real:** ask the user for one more photo of the object Rehearsal flags, and fuse it.
- **Meshes instead of primitives:** SAM 3D Objects for appearance, convex decomposition for collision.
- **More robots:** swap the Panda for a mobile manipulator from MuJoCo Menagerie; the scene builder is
  robot-agnostic.
- **Real-robot validation:** run the same plans on a real arm and measure how often "40/40 in sim" holds.
- **A dataset of real homes:** every rehearsal is a (scene, instruction, plan, failure, fix) trace with a free,
  verifiable reward from the simulator, useful for training open planners.

## Links

- GitHub: {{TODO-USER: public repo URL}}
- Live demo: {{TODO-USER: Hugging Face Spaces URL}}
- Video: {{TODO-USER: video URL}}
- X post: {{TODO-USER: X post URL}}

## Team

{{TODO-USER: names, roles}}
