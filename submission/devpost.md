# Submission form text (English)

Every number below traces to `README.md` and its sources (`docs/aigen_photos.md`, `docs/goalbench.md`,
`docs/intent.md`, `docs/hybrid.md`, `docs/real_robot.md`, `out/tables_v6/qa/QA_SUMMARY.md`, `budget/ledger.jsonl`, and
the eval_v7 tallies in `footage/tables_sim/eval_v7/OUTCOMES.md`). Replace `VIDEO_URL`, `GITHUB_URL` and `DEMO_URL`
before pasting.

Where each judging criterion is answered: **Impact** → Inspiration, What it does, What's next. **Use of Meta AI
tooling** → How we built it (the four Muse Spark roles, the critic). **Technical depth** → How we built it,
Challenges, Accomplishments. **Demo** → What it does (the app), Links.

---

## Project name

Rehearsal

## Tagline (≤ 60 chars)

Every action, rehearsed before it happens once.

Alternative: *Your table. Rehearsed first.*

## Summary (one paragraph)

Before a home robot moves, it should already have lived that moment many times, in simulation. Rehearsal turns one
photo of your table into a physics twin, using Meta Muse Spark to name the objects, SAM 3.1 to outline them and
SAM 3D to make them look real. Muse Spark then turns your sentence into goals that physics can check and states what
it assumed; it asks when you were ambiguous and refuses when it shouldn't act. A search plays out many possible futures
in MuJoCo: different orders, heights and routes. Physics rejects the ones that collide, tip or miss. Only a plan that
passed is sent, and if none passes, nothing is sent and Rehearsal says why. Across 50 AI-generated test tables it simulated 5,624
futures and rejected 5,149 of them. It sent 24 verified plans, found 4 goals already met, asked 5 questions, refused
4 requests, declined to send 12 times, and got 1 wrong. That is all in
simulation, and we publish the failures with the wins.

## Inspiration

"Clean this up." Anyone would say it to a home robot. Then I looked at my own messy desk and realised I didn't know
what I'd mean: stack the boxes, group the small things, or throw out the paper? A robot that arrives in a real home
has to answer questions no lab answers for it. What did you mean? What must not move? Will it work *here*? What if
it's wrong?

Language models are good with words and poor with geometry. A physics check is good with geometry but blind to intent.
We found this the hard way. We asked Muse Spark to "put the tomato soup next to the mouse"; it wrote goals about
keeping the soup upright, the robot did nothing, and the simulator said ✓. The soup was still 22 cm away. So we built
the layer that should sit between what you say and what a robot does: understand the intent, rehearse the motion,
and send only what passed.

## What it does

**A working app (`/app`), five steps, each labelled with the Meta model it used and its time and cost:**

1. **Photo.** Pick one of 50 test tables or upload your own photo (uploads need an API key).
2. **Perceive.** Muse Spark names every object, and SAM 3.1 outlines each one to the pixel.
3. **Twin.** One photo becomes a 3D physics twin. A *Physics* view shows the simple shapes the simulator trusts, and a
   *Visual* view shows real-looking SAM 3D objects with the photo's own textures. The app opens in Physics view.
4. **Instruct.** You type what you want. Muse Spark returns goals physics can check (what moves, what stays, what it
   assumed), or one question (for "Make room for my laptop." it asks where and how much space; the answer "The front of the table.
   Just put the orange and the spoon beside the milk carton." becomes goals), or a refusal with a
   reason ("pouring water onto the keyboard would deliberately damage electronics and create a safety hazard, and
   the robot only has a suction transport action with no ability to pour liquid").
5. **Rehearse.** The search plays out futures in MuJoCo, and live counters show how many were simulated and how many
   rejected. A wall of attempts gives the reason for every rejection. The sent plan replays in 3D, followed by a short
   stress test under noise. If nothing passes, you see what a direct attempt would have done, and why it wasn't sent.

**What it found on 50 AI-generated test tables (final run, eval_v7):**

| Outcome | v6 (first pass) | eval_v7 (final) |
|---|---|---|
| Rehearsed, plan sent, goals held | 9 | **24** |
| Goals already held | 4 | 4 |
| No safe plan, nothing sent (with a reason) | 23 | **12** |
| Asked a question | 5 | 5 |
| Refused, with a reason | 4 | 4 |
| We got it wrong | 4 | **1** |
| Empty table (no task) | 1 | 0 |

eval_v7 simulated **5,624 futures and rejected 5,149** across 41 tasks; collision was the most common reason. The
physics checker is unchanged (plus a stricter rim check). The wider search uses **labelled planner extensions**
(placement offsets, yaw turns, a grasp-point offset, short releases above the rim, "next to" placement): **19 of the
24 done tables used them (13 of the 15 newly solved)**, and **3 of the 24 use a twin corrected from its photo**. On
the hardest table, "Put the eggs into the mixing bowl.", 946 rollouts failed and nothing was sent.

**On my own desk** (local only, never uploaded or shipped), one sentence in my own words, "Move the power bank onto the
card wallet.", was rehearsed in physics and passed: the power bank was placed and 17 other objects were left
undisturbed. "Move the water bottle next to the monitor" led it to ask *which* bottle. One candidate carry, 0.22 m
high, dragged the bottle and tipped it over, so it was rejected; after 9 candidates and 8 rejections, a 0.30 m carry
passed. ("Next to" placement uses a beta extension of the planner.) Asked to pour water on the keyboard, it refused.

## How we built it

**Meta models, used beyond a single call: four roles for Muse Spark, plus SAM 3.1 and SAM 3D.**

- **Muse Spark 1.3: scene.** It lists the objects in a photo, gives SAM one noun phrase per object, and estimates size,
  mass and friction ranges. On the AI-generated tables, where there is no ruler in the photo, it also estimates the
  tabletop's size and the camera's pitch.
- **Muse Spark 1.3: intent.** Using structured output, it compiles the instruction into predicates a simulator checks
  (`on`, `inside`, `stacked`, `upright`, `stable`, `undisturbed`, `near`, `clear`), and it **must list its
  assumptions**.
- **Muse Spark 1.3: ask and refuse.** The same call may return one clarifying question or a refusal with a reason, and
  either one stops everything before the robot moves. A deterministic gate underneath refuses empty goals and unknown
  objects, and it asks when the goal already holds.
- **Muse Spark 1.3: critic, an unexpected use.** We gave it each of our twin renders next to its source photo and
  asked what we got wrong. It flagged 827 fidelity issues across 50 tables (wrong shape, colour, position, scale,
  missing objects), and we fixed them or disclosed them. The model checked our own work.
- **SAM 3.1** gives pixel-precise masks from noun phrases, with a point-prompt fallback. Each mask is fitted by a
  simple 3D shape. Median placement error is 0.41 cm on synthetic renders and 1.71 cm on 83 real objects with ground
  truth. **SAM 3.1 video** tracked real objects through DROID robot clips for our reality check.
- **SAM 3D Objects** runs on-device (a community MLX port with 8-bit weights, $0 in API costs). It produces
  real-looking meshes for display only: physics runs on the simple shapes it can trust.

**The rest of the stack:** MuJoCo with a simulated Franka Panda (MuJoCo Menagerie); a search planner in plain Python
that tries action orders, carry heights and detours, with an IK and collision dry run before any physics; a stress
test under randomised friction, mass and position error; and a Python server with a three.js front end. One Meta API
client caches identical requests, enforces a hard budget and logs every call. The whole project spent **$8.61** on
the Meta API, across 4,877 logged calls (as of Nov 4, 02:05 ET, including a $1.00 placeholder for a lost-ledger
window).

## Challenges we ran into

- **Our first headline fell over.** Muse Spark repaired 85 of 86 failed plans by reading the physics log. Then a
  zero-LLM search solved the same tables in seconds, because the model had mostly been guessing carry heights. We moved
  geometry to search and kept the model on intent.
- **Physics certified the wrong goal.** Of 55 wrong goal compilations, physics reported 17 as success; the old prompt
  refused 0 of 24 impossible requests. The fix was to make the model state its assumptions and let it ask or refuse.
- **Photo → physics without a ruler.** The AI-generated tables have no real camera and no A4 sheet. We fitted the
  camera from the tabletop's right angles and Muse Spark's size estimates. Right angles alone drove the pitch to its
  bound on 18 of 50 photos.
- **Honest negatives are hard to film.** Many tables end in "nothing sent". Rather than hide them, the film shows what a
  naive attempt would do: the "glass on the floor" shot is a deliberately naive, hand-written attempt on test table
  t6_19, really simulated and rejected. It is not planner output.

## Accomplishments that we're proud of

- **It says "not yet" instead of guessing.** It runs on 50 tables it had never seen, with every outcome counted: 24
  sent, 4 already done, 12 declined with a reason, 5 questions, 4 refusals, and 1 that we report as our own mistake.
- **Intent, measured blind.** On 30 instructions written by someone who saw nothing of our work, the shipped intent step
  got 27/30 right, against 21/30 for the old prompt. On our 60-case dev set it went from 40/60 to 60/60 (an upper
  bound: we wrote those labels).
- **Geometry, handed to search.** On five hard tables, Muse Spark's own first plans solved 14 of 100 runs. Muse's goals
  plus our search solved 100 of 100, at about half the Muse spend per solved task.
- **A pre-registered real-robot check, with a diagnosis.** On 23 real Franka episodes (DROID, 7 labs), our twin is not
  yet precise enough to predict real grasps: objects are placed a median 5.5 cm off, against a 4.25 cm gripper margin.
  That gives the next step a number to hit.

## What we learned

- Models are good at intent: Muse Spark picked the right object and destination in 150 of 150 runs. They are poor at
  geometry: 86 of 100 first plans failed on the hard tables.
- Physics can't pick intent: 74 of 132 "put X on Y" pairs on those tables are physically achievable, and only the
  sentence says which one you meant.
- A verifier is only as good as its goal. A model allowed to say "I'm not sure" or "I won't" closes more of that gap
  than any rule we could write.
- The compute is the point. Harder tables take more futures, and every rejected future is a mistake that never
  happened in your home.

## What's next

- **Real-arm transfer, with a target:** first-frame object positions inside the gripper's margin (well under 4.25 cm;
  about 1–2 cm to beat the simplest baseline), re-scored on the same DROID episodes.
- More actions than carry-and-place: pushing, pouring, opening.
- **A data blueprint (vision, not a result).** Every rehearsal, passed or rejected, is a labelled physical episode:
  intent, goals, motions and the physics outcome. Labs stage a few setups by hand, while homes are endless. At scale,
  this could become a data foundation for home robots. No policy has been trained on it yet.
- Rehearsal as a layer that home robots, and the apps built on them, call before they act.

## Built with

Meta Muse Spark 1.3 · Meta SAM 3.1 (image and video) · Meta SAM 3D Objects · MuJoCo · MuJoCo Menagerie (Franka Panda)
· DROID dataset (CC BY 4.0) · Python · three.js · Blender (film renders) · ACE-Step 1.5 and VoxCPM2 (film audio, run locally)

## Meta tools used, and how

| Tool | What it does in Rehearsal |
|---|---|
| Muse Spark 1.3 | Scene understanding (objects, sizes, masses, table size, camera pitch); intent → checkable goals with stated assumptions; ask; refuse; **AI critic of our own renders (827 issues across 50 tables)**; optional plan hint and revision. Structured output throughout |
| SAM 3.1 | Pixel-precise masks per object from noun phrases, fitted into 3D shapes |
| SAM 3.1 video | Tracks real objects in DROID robot clips to measure where they really ended up |
| SAM 3D Objects | On-device real-looking 3D objects for the twin (visual only) |

## Honest limits

Everything that plans was evaluated in simulation, and no robot has run our plans. The 50 test tables are
AI-generated photos without ground truth. Against 23 real-robot episodes, a twin built from one photo does not yet
beat trivial baselines, because position error is too large for a real grasp. The robot has one action
(carry-and-place) with an idealised suction grasp, and physics runs on simple shapes, with SAM 3D meshes used for
looks only. The intent results rest on 60 dev cases with our own labels and a 30-case hold-out written by one person.
The "data foundation" is a vision, not a result.

## About the film

- All robot motion is simulated (MuJoCo, simulated Franka Panda). The test tables are AI-generated photos.
- The "glass on the floor" shot is a deliberately naive, hand-written attempt on test table t6_19, really simulated
  and rejected; it is not planner output.
- The rehearsal-variant tiles are real candidates from the search on table t6_02, re-simulated for the film.
- The many-futures sequence is table 50 (141 futures simulated).
- The opening failure clips are third-party: YouTube clips under CC BY 3.0 (credited on screen) and the ViFailback
  dataset (MIT).
- Music: "Your Breath" from Mixkit (Mixkit Free License) for the intro, continued with ACE-Step 1.5 (MIT), generated
  locally. Voice-over: the creator's own voice, cloned locally with VoxCPM2 (Apache-2.0).

## Credits

Robot footage: DROID (Khazatsky et al., RSS 2024), CC BY 4.0. Physics: MuJoCo; the robot model comes from MuJoCo
Menagerie (Apache-2.0). YCB-Video frame: MIT (PoseCNN, Xiang et al., RSS 2018). SAM 3D Objects via the community port
mlx-spatial (SAM License). Film: failure clips from YouTube creators (CC BY 3.0) and ViFailback (MIT); music by Mixkit
("Your Breath", Mixkit Free License) and ACE-Step 1.5 (MIT); voice cloned with VoxCPM2 (Apache-2.0).

## Links

- Video: VIDEO_URL
- Code: GITHUB_URL
- Live demo: DEMO_URL (or run locally: see the README's Quick start)

## Team

Shengtao Shen (solo, Indie track): idea, code, evaluation and film.
