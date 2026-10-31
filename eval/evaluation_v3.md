## Evaluation v3 (definitive) — supersedes every earlier headline number

Run on 2026-10-31 (16:43–17:31 ET) with the current `main` code (`58a842f`, `code#189b1a39c1`), the shipped
English `demo_instruction` of each scene, Muse Spark 1.3 in Contributor mode (all five scenes are public,
hand-designed), default reasoning effort, one harness (`eval/run_eval.py --demo-only`) and one analysis
(`eval/round3_analyze.py`). Results: `eval/results/eval_20261031_173045_round3_main.json` (main),
`eval_20261031_170047_round3_kitchen_zh.json` and `round3_kitchen_refgoals.json` (kitchen factor arms; the reference-goals arm was stopped at 14 of 20 repeats for budget),
summary with CIs in `eval/results/round3_summary.json`. Cost of all v3 runs: **$1.12** (ledger, `eval.*`).

**Protocol.** Each repeat is the product pipeline: Muse compiles goals from the instruction, writes a plan, and
the plan is simulated once (attempt 1). Scoring is always against the scene's hand-verified `reference.goals`
plus implicit `undisturbed` (the model never sees them). If attempt 1 fails, the *same* failed attempt is revised
under three feedback conditions (paired; only the revise input differs): **full log** (product: failed goals with
numbers, IK/reach error, evidence sentences), **failed predicates only** (names + "aborted" flag), **"it failed"
only** (plan + success=false). Up to 2 revisions, stopping when the agent's own goals pass. Repeats are made
independent by an inert `[eval trial k]` line (repeat 0 is the product's exact prompt). Intervals are Wilson 95%;
the paired test is exact McNemar (two-sided binomial on discordant pairs). In no case did the agent's own goals
pass while the reference goals failed, so the two scorings agree everywhere.

### 1. First try (n = 20 per scene)

| scene | first plan succeeded | first plan failed | how it failed (first attempts) |
|---|---|---|---|
| desk_02 | 1/20 (1–24%) | 19/20 | out of reach at the carry height (IK) 19 |
| kitchen_01 | 0/20 (0–16%) | 20/20 | gripper body pressed on the kettle (protective stop) 19, IK 1 |
| lamp_01 | 12/20 (39–78%) | 8/20 | hit the lamp / disturbed it, IK, protective stop (mixed) |
| storage_01 | 0/20 (0–16%) | 20/20 | out of reach at the carry height (IK) 20 |
| windowsill_01 | 1/20 (1–24%) | 19/20 | out of reach at the carry height (IK) 19 |
| **all five** | **14/100 (8–22%)** | **86/100 (78–91%)** | |

### 2. Log ablation on the 86 failed first plans (paired, recovered within 2 revisions)

| scene | failed first plans | full log | failed predicates only | "it failed" only | full vs none: discordant (exact p) |
|---|---|---|---|---|---|
| desk_02 | 19 | 18/19 (75–99%) | 13/19 (46–85%) | 8/19 (23–64%) | 10:0 (p = 0.002) |
| kitchen_01 | 20 | **20/20 (84–100%)** | **0/20 (0–16%)** | **1/20 (1–24%)** | 19:0 (p = 4×10⁻⁶) |
| lamp_01 | 8 | 8/8 (68–100%) | 5/8 (31–86%) | 5/8 (31–86%) | 3:0 (p = 0.25) |
| storage_01 | 20 | 20/20 (84–100%) | 15/20 (53–89%) | 11/20 (34–74%) | 9:0 (p = 0.004) |
| windowsill_01 | 19 | 19/19 (83–100%) | 16/19 (62–94%) | 13/19 (46–85%) | 6:0 (p = 0.03) |
| **pooled** | **86** | **85/86 (94–100%)** | **49/86 (46–67%)** | **38/86 (34–55%)** | **47:0 (p = 1.4×10⁻¹⁴)** |

With only **one** revision the gap is larger: full log 70/86 (72–88%), failed predicates 22/86 (18–36%), "it
failed" 11/86 (7–21%). Full log vs failed-predicates-only: 36:0 discordant (p = 3×10⁻¹¹). There is not a single
failed plan that a blind or predicates-only retry fixed and the full log did not. lamp_01 is the one scene where
the effect is not significant (only 8 failures; blind retries fix 5 of 8).

### 3. Conflict (a), kitchen "0/5 without the log" vs "20/20 without the log": explained

The two numbers were measured with **different instruction languages**, and the difference is real and reproducible:

| arm (same harness, same code, same scene) | instruction | goals given to Muse | first try OK | blind ("it failed") recovered | full log recovered |
|---|---|---|---|---|---|
| v3 main | English "Put the cup on the coaster, then put the kettle on the trivet." | compiled by Muse (product) | 0/20 (0–16%) | **1/20** (1–24%) | 20/20 |
| v3 kitchen_en_refgoals (run-2 setup) | English | reference goals handed in | 0/14 (0–22%) | **1/14** (1–32%) | 14/14 |
| v3 kitchen_zh (EVAL v1 setup) | Chinese "把杯子放到杯垫上，再把水壶放到隔热垫上" | compiled by Muse | 6/20 (14–52%) | **13/14** (68–99%) | 14/14 |

- **Language is the factor.** With the English instruction Muse never moved the kettle first on its own (0 of 20
  first plans) and, told only "it failed", reordered in 2 of 20 revisions; instead it kept the instructed order
  and tried `via` detours and other carry heights, which cannot clear the gripper body. With the Chinese
  instruction it put the kettle first in 6 of 20 *first* plans (exactly the 6 first-try successes) and reordered
  blind in 13 of 14 failed cases. English "then" is read as a hard ordering constraint; Chinese 再 ("and also /
  then") apparently is not.
- **Not the cause:** the harness (both arms above use `run_eval.py`, and `experiments/agent/ablation.py` run 2
  matches the English arms), how goals reach the model (compiled vs reference: 1/20 vs 1/14), the scene version
  or executor (the Chinese arm runs on current main and reproduces EVAL v1), the system prompt (identical in all
  v3 arms), reasoning effort (default everywhere).
- **Denominator, too.** EVAL v1's "full log 20/20, none 20/20" counted all 20 repeats, including the 8 whose first
  plan already succeeded; among its failed first plans, blind recovery was 12/12. Run 2's "0/5" was over failed
  first plans only. v3 reports failed first plans only.
- So both old numbers were measured correctly, but neither was a general claim. The shipped product prompt is
  English; on it, the kitchen effect is the strongest in the evaluation (20/20 vs 1/20, p = 4×10⁻⁶).

### 4. Old claims, replaced

| old claim | status | v3 replacement |
|---|---|---|
| "5 of 5 hard scenes failed first try" (n = 1 per scene, from the shipped runs) | true but anecdotal | 86 of 100 first tries failed across the five hard scenes (lamp_01 is the easy one: 8/20) |
| "17/17 vs 8/17 vs 7/17" (run 2, 4 scenes × 5 repeats) | direction confirmed, n too small | 85/86 vs 49/86 vs 38/86 |
| kitchen "0/5 without the log" (run 2) | confirmed for English | 1/20 blind, 0/20 predicates-only, 20/20 full log |
| kitchen "every condition recovered, none 20/20" (EVAL v1) | true for the Chinese instruction only; denominator included first-try successes | Chinese: 13/14 blind; English (shipped): 1/20 |
| windowsill "15/15, 12/15, 5/15" (EVAL v1, Chinese) | superseded | English, n = 19 failed: 19/19, 16/19, 13/19 (p = 0.03); the gap is mostly at one revision: 17/19 vs 6/19 vs 2/19 |

### 5. Recommended headline sentences (true and defensible; use these verbatim)

1. **First try:** "Across five hard tabletop scenes, Muse Spark's first plan failed in simulation in 86 of 100
   tries (20 per scene)."
2. **The log matters:** "Given the physics log, it repaired 85 of the 86 failed plans within two revisions. Told
   only which goals failed, it repaired 49; told only that the plan failed, 38 (paired, p < 10⁻¹³)."
3. **Kitchen, with its condition:** "On the kitchen counter, where the fix is to move the kettle first, the log
   was the difference: 20 of 20 repaired with it, 1 of 20 without (English instruction). Asked in Chinese, the
   model often reorders on its own, so this scene's gap depends on the prompt language."
4. Short forms for tight surfaces: "86/100 first plans failed · 85/86 repaired with the physics log vs 38/86
   without". Always say "in simulation" and "within two revisions". Do not use "5 of 5", "17/17" or "0/5" any more.

**Caveats that travel with these numbers.** Everything is simulated (MuJoCo, hand-designed scenes, suction grasp);
the five scenes were designed to be hard, so 86/100 is a property of these scenes, not of everyday tables; one
model (Muse Spark 1.3 Contributor), default effort; lamp_01 contributes only 8 failed plans and its effect is not
significant on its own; the kitchen result depends on instruction language (section 3).
