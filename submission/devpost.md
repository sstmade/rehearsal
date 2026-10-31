# Submission form text (English)

All numbers below are from docs/agent.md (ablation run 2), EVAL.md, docs/real_photo_demo.md, docs/perception.md
and the shipped `out/*/report.json`. Remaining `TODO(user)` items are the user's: URLs and team. Do not submit
with any TODO left.

---

## Project name

Rehearsal

## Tagline (≤ 60 chars)

A flight simulator for robot AI, from one photo.

Alternative: *Models propose, physics decides.*

## Summary (one paragraph)

Pilots rehearse in simulators. A home robot's first touch of your things should not be its first try.
Rehearsal turns one photo of a table into a physics copy where mistakes are free: SAM 3.1 finds every object,
SAM 3D Objects rebuilds how each one looks, Muse Spark names them and plans your task, and a Franka Panda arm
tries the plan in MuJoCo. Code checks the result; the model never grades itself. When physics says no, Muse
Spark reads the physics log and fixes its own plan. That log is the finding: with it, Muse Spark repaired
17 of 17 failed first plans on four hard scenes; told only "it failed", 7 of 17. Then the plan is replayed in
40 slightly different worlds, because on a real photo a plan that passed once succeeded only 2 of 20 times
under small random changes. Rehearsal tells you which object is worth a second photo: on that photo, looking
again at the box took held-out success from 7/40 to 21/40.

## Inspiration

A confident plan that is wrong. We gave Meta's frontier model a tidy desk task, "put the tea tin on the place
mat", and the plan read perfectly: pick up the tin, carry it round the honey jar, set it down. In simulation,
the wrist swept a 48 cm desk lamp off the table. The model had never seen the lamp as an obstacle, because in
language, nothing collides. Words don't collide. The real world does.

Every home is different, and robots are tested in labs. The research answer is "simulate first", but building
a simulation of a real room still takes scanners, CAD and an engineer. Pilots get a simulator before they fly;
a home robot gets nothing. We wanted the cheapest possible rehearsal: a photo and a sheet of paper.

The "look again" step comes from Adam W. Harley and colleagues' work on active perception (*Move to See
Better*, BMVC 2021) and embodied rearrangement (*TIDEE*, ECCV 2022): an agent should choose what to observe
based on what it needs to do.

## What it does

**Act 1: see the table.** Upload a photo (an A4 sheet on the table gives metric scale; a dataset photo can use
its own camera). SAM 3.1 finds every object from a short noun phrase per object, with a point-prompt fallback
flagged low confidence. SAM 3D Objects rebuilds how each object looks from the photo and its mask. Muse Spark
names each object and estimates size, a mass range and a friction range. Out comes a MuJoCo scene with a
Franka Panda and a per-object position uncertainty.

**Act 2: rehearse.** "Put the mug in the bowl, then stack the books" is compiled by Muse Spark into goals a
simulator can check (`inside`, `on`, `stacked`, `upright`, `stable`); our code adds `undisturbed` for every
object you did not mention, so the model cannot forget the lamp. Muse Spark plans the moves. The arm tries
them. Code evaluates every goal on simulator ground truth.

**Act 3: learn from the crash.** If a goal fails, Muse Spark gets the physics log: which link touched what,
when, with how much force; how high the arm can reach at that spot; which goal failed by how much. It writes
a diagnosis and a new plan. In agent-loop mode it drives its own tools, `simulate`, `stress_test`,
`look_again` and `submit`; `submit` re-runs physics and rejects a failing plan, so the only way out is
through physics.

**Act 4: doubt the pass.** The final plan is replayed in 40 parallel worlds with perception noise, friction
and mass varied (common random numbers, so comparisons are paired). Then Rehearsal asks: if I re-observed
only this object, how many worlds would succeed? It tells you which object is worth a second photo, scored on
held-out seeds. The report separates verified by physics, estimated under uncertainty, and assumed, and a
three.js replay shows the run with SAM 3D meshes or the physics shapes (one toggle).

## How we built it

- **Meta chain, plainly:** Muse Spark 1.3 understands the scene, compiles goals, plans, reads the physics log,
  fixes itself and drives its own tools (strict JSON schemas and function tools through the Meta Model API).
  SAM 3.1 segments each object from a noun phrase; SAM 3.1 video tracks a phone clip after a real run for a
  reality check. SAM 3D Objects rebuilds a mesh per object from the photo and its mask; we ran it locally via
  the community MLX port mlx-spatial with 8-bit SAM-License weights (not Meta's official code or checkpoints),
  then oriented and scaled each mesh to the physics primitive through the photo's calibrated camera.
- **Physics:** MuJoCo via `MjSpec`, Franka Panda from MuJoCo Menagerie, damped-least-squares IK, a Cartesian
  transport primitive with waypoints, suction modelled as a weld, contact events logged with timestamps, link
  roles and forces, reach limits found by bisection.
- **Predicates:** six small geometric checks (`inside`, `on`, `stacked`, `upright`, `stable`, `undisturbed`),
  each a few lines over simulator state. The model never decides success.
- **Perception:** SAM's lossless mask fitted by a 3D primitive's silhouette; an A4 homography for scale (or the
  dataset camera); a fidelity pass that picks orientation and colour bands by projecting the twin back into the
  photo (mean silhouette IoU 0.55 → 0.64, colour error ΔE 21.7 → 16.3 on the real photo).
- **Robustness:** process-pool stress test with paired seeds; greedy value-of-information over objects, selected
  on seeds 0–39 and scored on held-out seeds.
- **Web app:** Python HTTP server with upload and job polling; three.js replay of recorded 50 Hz body poses,
  collision highlights, stress trails, and a SAM 3D mesh / physics-shape toggle.
- **Budget safety:** every Meta API call goes through one client that caches identical requests, reserves a
  worst-case cost against a hard cap, and logs real usage. The whole project spent $2.41 on the Meta API.

## Challenges we ran into

- **Making failure honest.** On our first desk scene Muse Spark planned correctly first try, which is good for
  Muse Spark and useless for showing self-correction. Scripted failures would be dishonest, so we hand-designed
  five hard scenes where the failure comes from physics the model cannot see in words: a wrist sweeping a lamp,
  a gripper body landing on a kettle, targets at the edge of reach, a stack that blocks the path. All six demo
  reports are live Muse Spark runs; the frontier model's first plan failed in 5 of 5 hard scenes.
- **Proving the log matters.** Would the model have fixed the plan anyway? We paired the same failed first plan
  with three kinds of feedback and measured (below).
- **A pass that was luck.** On the real photo the first plan passed simulation. Only the stress test, which the
  model ran itself, showed 2 of 20 under small perturbations. Without the 40 worlds we would have shipped a
  plan that works on paper.
- **Running SAM 3D Objects on a laptop.** Meta's official code needs CUDA; the community MLX port's dense mesh
  export needs about 44 GB, more than our 36 GB. We export the Gaussian splat instead and run marching cubes on
  it ourselves (under a second), then align each mesh by rendering it through the photo's camera. 7 of 8
  objects; the keyboard, cut by the photo edge, came back as a slab and falls back to its box.
- **A $15 night.** Caching and a budget-reserving client let several people iterate without surprise spend.

## Accomplishments that we're proud of

- **17 of 17.** With the physics log, Muse Spark repaired every failed first plan within two revisions. Told
  only the names of the failed goals: 8/17. Told only "it failed, try again": 7/17 (docs/agent.md, run 2,
  four hard scenes, five repeats). On the kitchen scene, where the fix is to move the kettle *before* the cup,
  5/5 with the log and 0/5 without. An independent eval on the windowsill scene: 15/15 vs 5/15, Fisher
  p = 0.0002 (EVAL.md).
- **A real diagnosis, in the model's own words** (shipped `out/kitchen_01/report.json`): "Gripper body extends
  ~7.5cm laterally, evidenced by TCP at [0.715,0.061,0.211] while hand pressed kettle top at [0.64,0.06,0.25]
  with 119N. … No via or transit_height can avoid this vertical descent collision. Correction is to clear the
  obstruction first."
- **The real-photo twist.** On a real camera photo (YCB-Video, MIT): SAM 3.1 found all 8 objects, the twin
  placed them within 1.5 cm median of ground truth, SAM 3D Objects rebuilt 7 of 8. Muse Spark's plan passed,
  its own stress test said 2/20, it looked again at the box and the drill, and held-out success rose from 7/40
  to 21/40. About half a cent in model calls.
- **Look again, held out.** On the designed desk scene, re-observing the mug takes 200 held-out worlds from
  178 to 191 successes, 13 fixed, none broken, p = 0.0002.
- **The model drives.** In the agent loop Muse Spark holds `simulate` / `stress_test` / `look_again` / `submit`.
  On the kitchen scene it moved the kettle first unprompted, stress-tested (17/20), chose `look_again(cup_1)`
  (20/20) and submitted; 5 tool calls, $0.002.

## What we learned

**Telling an AI "that failed" barely helps. Showing it what physics saw fixes it.** That is the lesson we would
hand to the field. Same failed plan, same model, three kinds of feedback: with the simulator's evidence 17 of
17 repaired; with only "it failed", 7 of 17. The log mattered most exactly where guessing fails: on the
kitchen, where the fix is structural (reorder the actions), the model never found it without the log, 0 of 5,
and always with it, 5 of 5. Where the fix is a one-knob search (carry height), blind retries sometimes get
there too. Language models are good at turning words into specifications and evidence into causes. They are
poor judges of their own physical plans. Give them a physics log, not a verdict.

Two smaller lessons. A pass can be luck: without the 40 worlds, the real-photo plan would have shipped at
2 of 20. And uncertainty is more useful when it is a decision: "look again at the box" beats "7/40".

## What's next

- **A real second look:** ask for one more photo of the object Rehearsal flags, and fuse it. Today "look again"
  is modelled as a smaller position error, not a new photo.
- **Meshes in the physics:** SAM 3D Objects meshes now carry the appearance; convex decomposition would let them
  carry the collisions too.
- **Real-robot validation:** run the same plans on a real arm and measure how often "40/40 in sim" holds.
- **More robots:** swap the Panda for a mobile manipulator from MuJoCo Menagerie; the scene builder is robot-agnostic.
- **A dataset of real homes:** every rehearsal is a (scene, instruction, plan, failure, fix) trace with a free,
  verifiable reward from the simulator, the kind of data open planners are short of.

## Honest scope

Simulation only; no real robot was used. The six designed scenes are hand-designed. `real_ycbv59` is a public
dataset photo (YCB-Video, MIT), not our own desk. The twin is a simulation-ready approximation (primitives at
measured positions), not a replica. SAM 3D Objects ran through a community port with third-party 8-bit weights.

## Links

- GitHub: TODO(user): public repo URL
- Live demo: TODO(user): Hugging Face Spaces URL
- Video: TODO(user): video URL
- X post: TODO(user): X post URL

## Team

TODO(user): names, roles
