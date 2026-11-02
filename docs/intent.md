# The intent step: from an instruction to goals the user has confirmed

_Product branch, 2026-11-02. Every number below comes from a script in this repo; commands are at the end._

HOLDOUT_SECTION

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

GATE_SECTION

DEV_CAVEATS

## Reproduce

```
.venv/bin/python -m bench.holdout.run --budget 0.30     # hold-out arms A and C (cached: free on re-run)
.venv/bin/python -m bench.holdout.run --score-only
.venv/bin/python -m bench.goals.gate_eval               # gate + self-consistency on stored dev data, $0
.venv/bin/python scripts/droid_trivial_baseline.py      # DROID trivial baseline, $0
.venv/bin/python -m pytest tests/test_intent.py
```
