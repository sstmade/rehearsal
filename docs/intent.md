# The intent step: from an instruction to goals the user has confirmed

_Product branch, 2026-11-02. Every number below comes from a script in this repo; commands are at the end._

## Headline: the blind hold-out (30 cases, run once, prompts frozen)

`bench/holdout/cases.json` was written by someone who had not seen the dev cases, the prompts or any results
(blindness statement in `bench/holdout/README.md`). We ran it once on the shipped compile step (arm C) and on the
old product prompt (arm A): 3 repeats each with the inert `[eval trial k]` tag, paired by (case, repeat), Muse Spark
`muse-spark-1.3` in Contributor mode, 180 calls, **$0.118**. The scoring rules (`bench/holdout/run.py`) were
committed before any output was seen, and nothing in the prompt was changed afterwards. Both arms are scored on the
goal set the *current* product verifies (compiled goals + the fixed implicit `undisturbed`, see below). A case
counts as correct when at least 2 of its 3 repeats are. Brackets are Wilson 95% intervals.

| | cases correct (n = 30) | trials (n = 90) |
|---|---|---|
| A. old product prompt (5 predicates, no way to say no) | 21/30 = 70% [52, 83] | 64/90 = 71% [61, 79] |
| A, scored with the old implicit protection (the product as it actually was) | 18/30 = 60% [42, 75] | |
| A + deterministic gate (no LLM) | 22/30 = 73% [56, 86] | 67/90 = 74% [65, 82] |
| **C. shipped compile step (assumptions + clarify / refuse)** | **28/30 = 93% [79, 98]** | **82/90 = 91% [83, 95]** |
| C + deterministic gate (the full shipped intent step) | 28/30 = 93% [79, 98] | 82/90 = 91% [83, 95] |

**Paired C vs A:** C right and A wrong on 9 cases, the reverse on 2 (exact sign test p = 0.065, n = 30). Per trial,
25 vs 7 (p = 0.002), but repeats of a case are correlated, so the case-level p is the honest one.

| category (cases) | A | C |
|---|---|---|
| explicit (5) | 4/5 = 80% [38, 96] | 5/5 = 100% [57, 100] |
| referential (5) | 5/5 = 100% [57, 100] | 4/5 = 80% [38, 96] |
| implicit (7) | 5/7 = 71% [36, 92] | 7/7 = 100% [65, 100] |
| ordering (4) | 4/4 = 100% [51, 100] | 3/4 = 75% [30, 95] |
| ambiguous, should ask (4) | 0/4 = 0% [0, 49] | 4/4 = 100% [51, 100] |
| infeasible, should refuse (5) | 3/5 = 60% [23, 88] | 5/5 = 100% [57, 100] |
| **by language:** English (24) | 16/24 = 67% [47, 82] | 22/24 = 92% [74, 98] |
| Chinese (6) | 5/6 = 83% [44, 97] | 6/6 = 100% [61, 100] |

What C fixes: it asks on all 4 ambiguous cases ("Which can should I put on the box: soup, Spam, or tuna?"), where A
guessed every time; it refuses "put the kettle inside the coffee cup" (size) and "hand the mug to me, I'm on the
other side of the room", which A compiled into `inside(kettle, cup)` and into a goal set that holds already; and it
has `near`/`clear`, which A lacks ("beside the photo frame", "next to the laptop but nothing goes ON the laptop").
A's 3 "correct" refusals are empty goal lists, which only count because the product now refuses an empty goal set.

**C's two misses are both refusals of a case labelled "goals", and both look like label errors.** We report them
as misses (the rules were fixed in advance), but a reader should know:
- ho09 "Whatever is sitting furthest from the robot arm, put it on the pencil case." The label picks the stapler
  (largest x, 0.79 m). By distance from the robot base the laptop is further (0.82 m vs 0.80 m), and the product's
  reachability check marks the laptop unreachable. C refused for exactly that reason, 3/3.
- ho19 "kettle on the trivet first, after that the apple on the cutting board". The scene's cutting board is marked
  unreachable by the product's own IK check (it sits at x = 0.80 m), and C refused 3/3 saying so. The label assumes
  it is reachable.
With both adjudicated in C's favour, C would be 30/30; we do not use that number.

C's other errors: on ho11 ("filled to the brim, don't tip it") one repeat omitted `stable(cup)`, and on ho12 (hot tea,
Chinese) one repeat refused, arguing that a suction cup on the open top of a mug of hot tea is unsafe (a fair point about this robot model). **Dev set vs hold-out:** C scored 60/60 cases (177/180 trials) on the 60 dev cases it
was written against; on cases it never saw it scores 28/30 (82/90). Treat **60/60 as a dev-set upper bound and
28/30 as the number to quote.** The hold-out is small (n = 30; the interval is 79-98%).


## What shipped

Rehearsal's division of labour is: **the LLM turns intent into verifiable goals, and must confirm its assumptions,
ask, or refuse; search solves the geometry; physics verifies and gates robustness.** The intent step is the first
part, and it now runs in this order (`rehearsal/rehearse.py: compile_step`):

1. **Compile (Muse Spark, one call).** The request is byte-identical to GoalBench arm C
   (`rehearsal/agents.py: COMPILE_PROMPT`, `INTENT_SCHEMA`; `bench/goals/run.py` imports the same objects and a test
   checks it). The vocabulary adds `undisturbed`, `near` and `clear`. The model must list its assumptions and may
   answer `clarify` (one question) or `refuse` (a reason) instead of goals.
   `REHEARSAL_COMPILE=legacy` restores the old prompt (arm A) for eval reproducibility.
2. **Deterministic intent gate (no LLM, `rehearsal/intent_gate.py`).** On any goal set the model returns:
   - refuse if it is **empty** (nothing asks for a change), names an **unknown object**, or is **malformed**;
   - ask if it is **vacuous** (every goal already holds after a 1.5 s physics settle with no robot motion, i.e.
     "success" would mean the robot does nothing) or **contradictory** (one object sent to two places, `clear(b)`
     while something goes onto b, `undisturbed(x)` while x is moved).
   It also builds the **effect preview**: what will move where, what must not move or tip, and which extra
   conditions are checked, as ready sentences.
3. **Stop or run.** A model refusal / question, or a gate flag, stops the rehearsal before any planning: the report
   gets `status: "refused"` or `"needs_clarification"`, a `message`, no attempts and no stress test, and the API's
   `success` is false. **An empty or refused goal set is no longer a success.** (Before, `[]` became "nothing may
   move", which an empty plan passes, so "put the cup in the fridge" was certified.) A user who has read the
   assumptions and preview can re-run with `confirmed: true`, which overrides gate *questions* but never refusals.
4. **Implicit protection fix (`rehearsal/predicates.py: with_implicit_goals`).** Every object the goals do not
   *move* (first argument of on/inside/near, upper objects of stacked) now gets the implicit `undisturbed`,
   including destinations and objects mentioned only in `upright`/`stable`/`clear`. Before, any mention removed the
   protection, so "don't knock over the vase" compiled to `upright(vase)` made the check *weaker*
   (`dropped_protection`). Rescoring the stored dev runs with the fix: arm A 65% → 72% of trials, arm B 77% → 80%,
   arm C unchanged at 98%; `dropped_protection` goes from 10 / 3 / 0 to 0 in every arm. The published GoalBench
   tables keep scoring the old behaviour (`bench/goals/score.py: PROTECT_MENTIONED = False`).
5. **Planner default is `hybrid`** with the 20-seed robustness gate (`docs/hybrid.md`); `llm` and `search` stay
   available (`planner` in the API, `--planner` on the CLI, `REHEARSAL_ROBUST_N` to change the gate size).

The JSON fields the UI renders (`status`, `message`, `intent.assumptions`, `intent.question`, `intent.reason`,
`intent.gate.preview`, ...) are specified in `docs/api.md` ("Report: intent fields"). `POST /api/intent` runs steps
1-2 only, for a confirmation screen before the rehearsal.

## The dumb baseline for C: a deterministic gate

Rule for every claim: could a trivial check do this? The gate in step 2 is that check. It is cheap and has no model in
it, so we scored it as a baseline for arm C on stored data ($0, `bench/goals/gate_eval.py`,
`bench/goals/results/gate_eval.json`) and on the hold-out above.

**On the 55 stored wrong compilations** that `bench.goals.consequence --auto` replays (arms A and B), the gate flags
**0 of 55**, including **0 of the 17** that physics certified as success while the user's intent did not hold. Arm C
gets all 55 cases right (majority of repeats). This is not a bug in the gate; it is what those 55 are:
- the 17 certified-wrong ones are guesses that ask for a *real* change: "tidy up the desk" → stack the books,
  "make some room" → book onto the laptop, "put the box on the other box" → small onto large. No structural check
  can tell these from the right answer;
- 26 are compilations of impossible requests (bowl inside the mug, the unreachable laptop) that look well formed
  and fail in physics; the gate does not reason about sizes or reach;
- 9 are `dropped_protection`, which the protection fix (step 4) removes structurally, without the gate;
- the auto-selection excludes `vocab_gap` trials, which is exactly where the gate works (next item).

**Where the gate does catch something:** the "robot does nothing" compilations. With the old prompt, "put the soup
next to the mouse" and "put the mug next to the vase" compiled to `upright(x) ∧ stable(x)` (3/3 repeats each).
The gate's physics settle shows these already hold and asks instead of certifying success: 6 of arm A's 63 silently
wrong dev trials become questions (silent wrong 63 → 57). The fridge case's `[]` is now refused instead of passed.
On arm C's dev trials it converts the one contradictory compilation (o04: `stacked(L, M, S)` plus `stacked(S, M)`)
into a question. **It flags 0 of 173 distinct correct compilations** (no false alarms). On the hold-out it lifts A
from 21 to 22 cases (23 if any flag counts) and changes nothing for C.

**Conclusion.** A trivial check closes the "nothing happens" and "empty / unknown object" holes, cheaply and
without false alarms, and the product keeps it as a safety net under C. It does not come close to C: the errors that
matter (guessing on ambiguous requests, compiling impossible ones) need the model to say "I'm not sure" or "I can't".

## Self-consistency as an ask trigger (KnowNo-style, no conformal calibration)

Trigger: ask whenever the 3 stored repeats of a case disagree. Scored on stored data, $0
(`bench/goals/gate_eval.py`, `bench/holdout/consistency.py`). "Agree" means the same verified goal set (stacked
expanded, implicit `undisturbed` added, order ignored) and the same status.

| arm, data | asks | should have asked (ambiguous or infeasible): precision / recall | majority answer wrong: precision / recall |
|---|---|---|---|
| A, dev (60) | 16 | 6/16 = 0.38 / 6/19 = 0.32 | 5/16 = 0.31 / 5/20 = 0.25 |
| B, dev (60) | 27 | 0.22 / 0.32 | 0.22 / 0.43 |
| A, hold-out (30) | 8 | 2/8 = 0.25 / 2/9 = 0.22 | 4/8 = 0.50 / 4/9 = 0.44 |
| A, dev, placements only | **0** | – / 0 | – / 0 |
| A, hold-out, placements only | 1 | 1/1 / 1/9 | 1/1 / 1/9 |

Disagreement is almost entirely about auxiliary atoms (an extra `upright` or `stable`). Restricted to *what goes
where*, Muse's 3 repeats agree on **every** dev case in arms A and B and on 29 of 30 hold-out cases, including every
confident wrong guess ("put the box on the other box" → small onto large, 3/3). **Self-consistency cannot detect a
confident wrong guess at this temperature, so it is not a substitute for letting the model ask.** On arm C it mostly
fires on harmless variation (precision 0 for wrong answers on both sets). We did not ship it.

## Real-robot check, for completeness

The DROID replay agreed with reality on 5 of 6 episodes. A trivial rule that reads only the trajectory (fingers
stopped short of fully closed, then the end effector rose) also gets 5/6, with identical predictions, so the twin's
success verdict adds nothing measurable over it on these six episodes (`docs/real_robot.md`,
`scripts/droid_trivial_baseline.py`). The twin's added value has to be argued from object outcomes (11/14
moved/not-moved agreement, 2.5 cm median final error) and from more episodes.


## Caveats

- **Small n.** 30 hold-out cases, 4-7 per category; per-category intervals are wide. The case-level C vs A
  difference has p = 0.065.
- **Two hold-out labels look wrong** (ho09, ho19, above), both against C. We kept the pre-registered scoring.
- **One author per set.** The hold-out was written blind but by a single person; the dev set and C's prompt were
  written by the same people. C's refusal paragraph names the failure classes the dev set probes (sizes,
  reachability, missing objects); the hold-out is the check that this generalises, and it mostly does.
- **The scorer's choices.** Extra `upright`/`stable`/`undisturbed` atoms are harmless; a required `upright`/`stable`
  (full cup) must be stated. `near` counts only in the labelled direction unless the label lists the swap.
- **What the gate cannot see.** Wrong but real changes, impossible geometry, unreachable objects. Those are left to
  the model (asks/refusals) and to physics (which reports them as plan failures).
- **`near` is verifiable but not yet executable.** The transport action places an object in/on another object;
  there is no "put it on the table next to X" action, so a `near` goal compiles and is checked correctly but the
  planner cannot achieve it, and the rehearsal reports a failure.
- **Muse settings:** `muse-spark-1.3`, Contributor mode, default effort, one sample per repeat. The agent-loop mode
  (`--agent-loop`) and `eval/hybrid_eval.py` still use the legacy compile call for reproducibility.


## Reproduce

```
.venv/bin/python -m bench.holdout.consistency            # self-consistency on the hold-out, $0
.venv/bin/python -m bench.holdout.run --budget 0.30     # hold-out arms A and C (cached: free on re-run)
.venv/bin/python -m bench.holdout.run --score-only
.venv/bin/python -m bench.goals.gate_eval               # gate + self-consistency on stored dev data, $0
.venv/bin/python scripts/droid_trivial_baseline.py      # DROID trivial baseline, $0
.venv/bin/python -m pytest tests/test_intent.py
```
