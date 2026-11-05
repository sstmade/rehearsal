# Claims audit

Sources audited: `README.md`, `../Rehearsal-项目方案.md`, `rehearsal/rehearse.py`, `rehearsal/executor.py`,
`rehearsal/agents.py`, `rehearsal/predicates.py`, `rehearsal/sim.py` (branch `research`, commit 15ea573,
which matches `main` for these files), and the only existing run, `out/desk_01/report.json` in the main repo.

Rule of thumb for the pitch: **say what the code does, name the assumption, show the number with its N.**

## 0. Facts about the current evidence (read first)

- The only report on disk (`out/desk_01/report.json`) was produced by **`offline-stand-in`**, the
  rule-based agent, not Muse Spark. Its diagnosis ("t=… 撞到了 … 改为 0.xx m") is hand-written Python
  in `OfflineAgent.revise`. **The "35/40 → 40/40" and the attempt-1-fail → attempt-2-success story
  currently come from the stand-in.** Before any demo or README claim that Muse Spark self-corrects,
  we need a run with `agent: muse-spark-…` in the report.
- In that run the stress test went 35/40 and looking again at `mug_1` gave 40/40. The other candidates scored
  bowl_1 38, and bottle, books and plant 35 each. That selection and the 40/40 were measured **on the same 40
  seeds** (see §1.6, winner's curse). A held-out re-check is in §4. **Held out, it is 89% → 95.5%,
  not 35/40 → 40/40, and the baseline moves from 99% to 61% when σ goes from 0.8 to 1.6 cm.**

## 1. Claim by claim

### 1.1 "Digital twin" / "photo → physics-ready digital twin of your space"
- **As implemented:** objects are primitives (box, cylinder, bowl) placed from a homography of the
  table plane, sized from Muse Spark's estimate plus the A4 sheet. Physical properties are LLM guesses
  with ranges. Grasping is a weld constraint that always succeeds. `perception.py` is marked
  *untested* in the README. `desk_01` is hand-written. SAM 3D meshes are not wired in.
- **Verdict:** overstated. ACDC (CoRL 2024) explicitly contrasts "twins" with approximate "cousins". A
  Meta researcher will hear "digital twin" as "high-fidelity replica".
- **Say instead:** "a simulation-ready approximation of your table: primitive shapes at measured
  positions, with estimated mass and friction ranges." Use "twin" only in the consumer tagline, if at all.
- **Evidence to make it airtight:** the 方案's own metrics. Per-object position error (cm) and size error
  against a tape measure on 3 real photos (median and max), and silhouette IoU of re-projected
  primitives vs. SAM masks. Then **feed the measured position error back as σ** (see 1.5).

### 1.2 "Self-corrects from simulator logs" / "reads the simulator's own logs to diagnose why"
- **As implemented:** `MuseAgent.revise` gets the full attempt history as JSON (plan, per-predicate pass
  and fail with numeric reasons, contact events with timestamps and `while_holding`, IK error message) and must
  output a diagnosis plus a new full plan. At most 3 attempts. The physics re-checks the new plan. The
  mechanism is real. **But the only recorded success is from the rule-based stand-in**, and we have no
  measurement that the log is what causes the fix (it could just be a retry with the goal list).
- **Prior art:** REFLECT (CoRL 2023) explains failures from robot logs. Inner Monologue (CoRL 2022) replans
  from feedback. SIMPACT (CVPR 2026) refines VLM plans from sim rollouts. **Do not claim novelty for log-based repair.**
- **Say instead:** "When a simulated attempt fails, Muse Spark receives a structured log (which predicate
  failed and by how much, timestamped contacts, IK failures), must state a physical cause, and proposes
  a revised plan, which is re-simulated and accepted only if every predicate passes."
- **Evidence:** Experiment A below: success within 3 attempts with a full log vs. failed predicates only
  vs. a bare "it failed, try again", on ≥5 hard scenes. Also report **diagnosis accuracy**: does
  the stated cause name the body pair that actually collided, or the predicate that failed? This can be
  scored automatically against the event log.

### 1.3 "Models propose, physics decides" / "every conclusion is verified by the simulator"
- **As implemented:** true for *execution outcomes*. Predicates are evaluated on MuJoCo ground truth, never by
  the model. Two caveats a researcher will spot:
  1. **Goal compilation is not verified by physics.** If Muse Spark compiles "stack the books" into
     the wrong order or drops an object, the simulator will happily certify the wrong goal. Physics
     decides *whether the predicates hold*, not *whether they mean what the user said*.
  2. Predicate thresholds are hand-set (upright cos > 0.95, `on` gap < 1.5 cm, `stable` < 5 mm in 1 s,
     undisturbed < 2 cm). "Verified" means verified against these tolerances, inside the sim.
- **Also:** the implicit "don't disturb other objects" goal is **added deterministically by code**
  (`with_implicit_goals`), not compiled by Muse Spark. That is a strength (it can't be forgotten), so
  describe it accurately: "Rehearsal automatically adds a no-side-effects constraint for every unmentioned object."
- **Say instead:** "The model proposes goals and plans. Whether the plan achieves those goals is decided by
  physics, not the model. Goals are shown to the user for confirmation."
- **Evidence:** a goal-compilation accuracy table on ~20 instructions (exact-match vs. a hand-labelled
  predicate set). Cheap, and it addresses caveat 1 head-on.

### 1.4 "Common random numbers"
- **As implemented:** correct. `_variant` seeds `default_rng(seed)` and draws friction, mass and a standard-normal
  `(n_objects, 2)` matrix *before* scaling by the per-object σ. So two stress tests that differ only in one
  object's σ share every other draw, and that object's error is the same direction scaled by σ'/σ.
  Seeds are `0..N-1` in every stress test.
- **Nuance:** with CRN, a "re-observed" object's residual error is the *same* error shrunk 3×, not
  a fresh draw. That is a legitimate coupling (it makes the comparison monotone and low-variance), but say so.
- **Say instead:** "Plan variants and re-observation options are compared on identical noise draws
  (common random numbers; Glasserman & Yao 1992), so differences reflect the change, not sampling luck."
- **Evidence:** report the *paired* statistic. Count discordant seeds (fail→success vs. success→fail) and give a
  McNemar/sign-test p-value, plus the variance of the difference with vs. without CRN. A one-line demonstration
  that CRN cut the needed N would make this a technical point, not a buzzword.

### 1.5 "Stress test across 40 parallel universes" / "estimated success rate"
- **As implemented:** domain randomization used for evaluation. Friction ×U(0.6, 1.4), mass ×U(0.7, 1.3),
  and per-object xy position error N(0, 1.2 cm) per axis. The *real* object is displaced and the robot acts on its
  belief (the scene file). Objects the robot placed itself are then known exactly (`placed_by_robot`), which is an
  oracle assumption. The stress test runs only on the final plan, and only if it succeeded nominally.
- **Problems:** (a) σ = 1.2 cm is an **assumed constant**, not measured. The per-object `perception.sigma_m`
  field in the contract is **not used** by `rehearse.py` (it uses `SIGMA` for all objects). (b) 35/40 has
  a 95% Wilson interval of roughly 74–95%. Quote the interval or "about 7 in 8". (c) It is a sim
  estimate, not a real-world success probability (SIMPLER measured sim–real correlation; we have not).
- **Say instead:** "Under our assumed noise model (±1.2 cm position error, friction and mass varied), the plan
  succeeded in 35 of 40 simulated variations." Avoid "parallel universes" with researchers. Call it
  domain randomization (Tobin et al. 2017) for plan evaluation.
- **Evidence:** Experiment B (success vs. σ curve), plus using measured per-object σ from 1.1 once available.

### 1.6 "Active perception = value of information" (35/40 → 40/40)
- **As implemented:** for each object, re-run the stress test with only that object's σ reduced
  1.2 → 0.4 cm (same seeds). Pick the argmax, keep it, repeat up to `--look` rounds. The symbolic plan is
  fixed, but grasp and place coordinates come from the (better) belief at execution time.
  "Looking again" is **simulated by shrinking σ**. No second photo is taken.
- **What it is, precisely:** a Monte-Carlo estimate of the **expected improvement in plan success
  probability from a (simulated) partial observation of one object**, under a Gaussian error model, with
  the symbolic plan held fixed and selection greedy and myopic. That *is* decision-relevant value of information
  in Howard's (1966) sense: value measured in task outcome, not entropy. But it is not full VoI, which would
  let the *plan itself* change after the observation, and it is not belief-space planning.
- **Statistical problem (winner's curse):** the argmax over 7 candidates and the reported "after" number
  are computed **on the same 40 seeds**, so the after-score is optimistically biased. The fix is trivial:
  select on seeds 0–39, report on held-out seeds (e.g. 1000–1199). See §4 for a held-out check we already ran.
- **Say instead:** "Rehearsal ranks which object is worth a second look by how much reducing its position
  uncertainty raises simulated success (a Monte-Carlo value-of-information estimate). On held-out noise draws,
  re-observing the mug raised simulated success from 89% to 95.5% (N = 200, paired)."
- **Evidence:** Experiment C (VoI vs. random vs. heuristic re-observation, held-out seeds). Even better: a real second
  photo of the chosen object, re-run perception, and measure the actual σ reduction.

### 1.7 "Muse Spark infers physical properties" / "fragile"
- **As implemented:** mass and friction nominal values and ranges are in the scene file. The stress test
  multiplies *nominal* values by global scales (it does not sample within each object's own `range`). `fragile` is
  stored but no predicate or event uses it (no impact-force check).
- **Say instead:** "Muse Spark estimates mass and friction. The stress test varies them ±30–40%." Drop
  fragility claims unless a max-contact-force check is added.

### 1.8 "Reachability verified via IK"
- **Accurate:** `reachability_report` runs damped-least-squares IK for a top-down suction pose on the real
  Panda model (MuJoCo Menagerie). Caveat: top-down orientation only, and failure means "no top-down solution
  found", not "unreachable by any grasp".

## 2. Five skeptical questions from a Meta robotics researcher

1. **"How do you know any of this transfers to a real robot? You have no robot."**
   *Honest answer:* We don't, and we don't claim a real-world success rate. The output is a *simulated
   feasibility and robustness report* under an explicit noise model. The failures it catches are ones
   physics alone explains (collisions from a low carry height, unstable stacking order, unreachable
   poses), and those transfer to reality, as Text2Motion and PRoC3S found for geometric feasibility.
   Continuous numbers such as 35/40 are model-dependent.
   *Measure:* even without a robot, (i) measure real perception error on 3 photos and re-run the stress test with
   that σ. (ii) A "sanity transfer" check: physically do the failing configuration by hand (e.g. the big book on the
   small book) and film whether it topples as the sim says. Report agreement on k cases.

2. **"Is the LLM actually using the log, or does it just retry?"**
   *Answer:* Show Experiment A: full log vs. predicates-only vs. no feedback, and automatically scored
   diagnosis correctness. If no-feedback does as well, say so and reframe around verification.
   *Measure:* success within ≤3 attempts, mean attempts, diagnosis-names-correct-collision rate.

3. **"Your 40/40 after looking again is selected and evaluated on the same seeds, and your σ is made up. Isn't VoI just
   picking the object nearest the failure?"**
   *Answer:* Held-out evaluation (§4) removes the selection bias. Compare against a simple heuristic baseline
   (re-observe the object involved in most failed predicates, or the smallest object, or the one with least
   clearance). If the heuristic matches VoI on easy scenes, VoI's value is that it needs no hand-coded rule and
   generalises to side-effect failures (e.g. it picks a bystander bottle that gets bumped).
   *Measure:* Experiment C on ≥5 scenes, success after 1 look, VoI vs. random vs. heuristic, held-out seeds.

4. **"How is this different from SIMPACT / PRoC3S / REFLECT?"**
   *Answer:* See related-work.md. In one line: same "simulate before you act" family. Our additions are
   predicate-checked success with automatic side-effect constraints, paired robustness estimates under perception
   uncertainty, and task-driven selection of what to re-observe. They have real-robot validation and richer
   geometry. We do not.

5. **"A suction weld that always succeeds and primitive shapes: aren't you hiding the hard part of manipulation?"**
   *Answer:* Yes, deliberately. Rehearsal targets *task-level* failures (ordering, clearance, stability,
   reachability, side effects), not grasp synthesis. That is stated in every report under "Assumed (not verified)".
   *Measure:* break failures down by type over all scenes, to show which categories the tool catches.
   Optionally run a sensitivity check: cap the weld force at the suction cup's rated load (e.g. a 20 N
   weld limit) and see whether conclusions change.

## 3. Recommended experiments (each ≤ 1 h with existing code, $0 unless noted)

**A. Log ablation for self-correction** (agent/eval team, Muse Spark; ~3 conditions × 5 scenes × 2 runs ×
≤4 calls ≈ 120 calls). Use the response cache, and note that the eval allowance is $3.
Conditions: `revise` gets (1) the full history (current), (2) only failed predicates with `why`, (3) only
"attempt failed". Use hard scenes where attempt 1 fails for a physical reason (low transit height over a tall
object, a big book on a small one, a bystander in the carry path). Report success within 3 attempts, attempts used,
and diagnosis correctness (does the diagnosis mention the colliding pair from `events`?). This is the
single most important experiment for the "Muse Spark reads the log" claim, and **the current evidence for that
claim is from the rule-based stand-in.**

**B. Perception-error tolerance curve** (offline, ~10 min). For the final plan of each scene, sweep σ ∈ {0, 0.4, 0.8,
1.2, 1.6, 2.0, 3.0} cm with N = 200 held-out seeds (CRN across σ). Plot success with Wilson CIs. This turns
"35/40" into a curve ("this plan tolerates ~1 cm. Beyond that the mug misses the bowl"). Overlay the
measured perception error from 1.1 to show where we actually sit. Also compare the initial vs. revised
plan on the same curve: revision should shift it right.

**C. VoI vs. baselines with held-out evaluation** (offline, ~20 min). Select on seeds 0–39, evaluate on
1000–1199. Policies: VoI argmax, random object (average over all objects = the expected random policy),
and a heuristic (the object in most failed predicates in the selection set). Metric: held-out success after 1 look.
Report McNemar discordant pairs vs. baseline. Run on ≥5 scenes, or on desk_01 with 5 random perturbations of
object positions, so it is more than one anecdote.

(Cheapest add-on to C: the CRN variance demonstration from 1.4, in the same script.)

## 4. Held-out re-check of the desk_01 VoI result (run by research, offline, $0)

Setup: the final plan of `out/desk_01/report.json` (offline agent, transit height already revised),
the `_variant` function as-is, **seeds 1000–1199 (N = 200, never used for selection)**, CRN across all
rows. Script: `heldout_voi.py`, a scratch script (not committed: research owns only docs.
Eval can copy it into `eval/`). Wall time 15 min for 3,000 rollouts on the laptop.

**Re-observe one object (σ 1.2 → 0.4 cm for that object only):**

| Option | Held-out success | Paired vs. baseline (seeds fixed / broken) |
|---|---|---|
| baseline (no second look) | 178/200 (89.0%) | — |
| **mug_1** (VoI pick on seeds 0–39) | **191/200 (95.5%)** | +13 / −0 (exact sign test p ≈ 2×10⁻⁴) |
| bowl_1 | 185/200 (92.5%) | +7 / −0 |
| book_small | 185/200 (92.5%) | +7 / −0 (scored *0 gain* on the 40 selection seeds) |
| bottle_1, book_large, book_medium, plant_pot | 178/200 each | +0 / −0 |
| random object (expected over 7) | 181.9/200 (90.9%) | — |

**Global σ sweep (all objects, same seeds):**

| σ per axis | 0 | 0.4 cm | 0.8 cm | **1.2 cm (assumed)** | 1.6 cm | 2.0 cm | 3.0 cm |
|---|---|---|---|---|---|---|---|
| success /200 | 200 | 200 | 198 | **178** | 121 | 70 | 6 |

**What this means for the claims:**
- The VoI *ranking* survives held-out evaluation: the mug is still the best object to look at, and the
  effect is significant on paired seeds. The **"40/40" headline does not survive**: held out, it is
  **89% → 95.5%**. Say "about 89% → 96% on 200 fresh noise draws". Do not say "35/40 → 40/40".
- The selection set is noisy at N = 40: book_small showed 0 gain there but +7 held out. Select with N ≥ 100,
  or report the held-out number.
- VoI beats the random-object policy by ~4.6 points here. That is one scene. Experiment C must repeat this on more scenes.
- **The robustness number is dominated by the assumed σ.** Success falls off a cliff between 0.8 cm (99%) and
  1.6 cm (60%). So "89%" means almost nothing until σ is measured from real photos (1.1/1.5). The defensible
  framing is the curve: "this plan tolerates up to ~1 cm of position error".
- With this CRN coupling, reducing one object's σ never broke a seed (−0 everywhere). That is expected
  (the same error shrunk), and it is why the paired test is so sensitive. Mention it if asked.
- Caveat: all of this uses the offline agent's plan. Re-run with a Muse Spark plan once one exists.
