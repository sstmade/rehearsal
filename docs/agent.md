# Agent: Muse Spark drives the rehearsal

Owner: agent role. Code: `rehearsal/agents.py`, `rehearsal/agent_loop.py`, `experiments/agent/`.

## 1. Two ways the model is used

| Mode | Who decides the next step | Entry point |
|---|---|---|
| Fixed pipeline (default) | our code: compile goals -> plan -> simulate -> revise (<= 3 attempts) | `rehearse(...)` |
| **Agent loop** (opt-in) | **Muse Spark**, through function tools | `rehearse(..., agent_loop=True)`, `--agent-loop`, `python -m rehearsal.agent_loop scene.json "instr"` |

## 2. The agent loop

Goals are compiled once (`MuseAgent.compile_goals`, strict JSON schema) — or passed in (`goals=`). Then one
Chat Completions conversation with four strict function tools (`chat_tools`, `tool_choice` is always "auto"):

| Tool | What physics does | What the model gets back |
|---|---|---|
| `simulate(actions)` | `executor.run_plan` once, nominal parameters | success, failed/passed predicates with numbers, `execution_error` (IK reach diagnostics: max reachable tool height here / at segment end), `evidence` sentences (which link hit what, when, during which phase), arm contact forces |
| `stress_test(actions, n)` | `rehearse.stress_test`, n in [8, 40] universes (friction, mass, per-object perception sigma) | successes/n, top failure reasons, objects whose perception error is larger in failing runs than in passing ones |
| `look_again(object_id)` | shrinks that object's sigma to 0.4 cm, reruns the last stress test **with the same seeds** | before/after successes |
| `submit(actions, rationale)` | re-simulates the plan; **a failing plan is rejected** and the loop continues | accepted / rejected + physics report |

"Models propose, physics decides" is literal here: the model can only finish through `submit`, and `submit`
is a physics check. If the model answers without a tool call it is nudged once; if it never submits, the last
simulated plan is reported as unsubmitted.

Caps: 8 tool calls, $0.10, 12 model turns per run. Prompt caching: system rules + tool schemas + scene summary
form a stable prefix, `prompt_cache_key = rehearsal-loop-<scene>`; measured 77% of input tokens cached
(windowsill run: 13,877 of 18,092).

Every action now has `via` ([[x, y], ...], required, may be empty): the carry passes through these waypoints at
`transit_height` (robotics executor). `ACTION_ITEM` in `agents.py` is shared by `ACTION_SCHEMA` and the tools.

### Report

`rehearse(..., agent_loop=True)` keeps the CONTRACTS §2 shape: each `simulate`/`submit` call becomes an
attempt (re-rendered to `attempt_N.gif`/`.traj.json`, plus `via_tool`), the last one (the submitted plan) is
the final attempt, and the usual `stress_test`/`active_perception` run on it. New optional key `agent_trace`:

```jsonc
{"model", "data", "feedback", "tools", "caps", "stopped": "submitted"|"tool-call cap"|"cost cap"|...,
 "tool_calls", "cost_usd", "goal_compile_cost", "prompt_tokens", "cached_tokens", "completion_tokens",
 "wall_time_s", "submitted": {"plan", "rationale", "verified"},
 "story": ["simulate(...) -> FAILED (aborted) | IK failed ...", ...],      // one line per tool call
 "turns": [{"turn", "text" /* the model's own narration */, "prompt_tokens", "cached_tokens",
            "completion_tokens", "cost", "cache_hit",
            "tool_calls": [{"name", "args", "result", "seconds"}]}],
 "agent_stress_runs": [{"plan", "look_at", "successes", "n", "sigma_cm"}], "agent_looks": [...]}
```

### Real runs (Contributor mode, public scenes)

windowsill_01 "把马克杯放到托盘上" (`experiments/agent/loop_windowsill_01.json`), $0.0023, 6 tool calls:

```
[model] 避开高花瓶走南侧绕行，我先仿真验证高空搬运方案。
simulate(mug_1->tray_1@0.5 via [[0.55,-0.15]]) -> FAILED | IK failed during 'carry mug_1 to tray_1': at xy [0.709,-0.101] the tool tip can reach at most z=0.47 m ...
[model] 上次高度超出了远端可达范围，这次我改用贴近极限的低空南线绕过花瓶再测一次。
simulate(mug_1->tray_1@0.24 via [[0.55,-0.15]]) -> SUCCESS
simulate(mug_1->tray_1@0.24 via [[0.35,-0.15],[0.65,-0.2]]) -> SUCCESS
stress_test(..., n=20) -> 19/20
[model] ...压力测试有一例因感知偏差导致放置失败，我复测马克杯来缩小误差。
look_again(mug_1) -> 19/20 -> 20/20
submit(...) -> ACCEPTED
```

kitchen_01 via `rehearse --agent-loop` ($0.0018): the model moved the kettle first on its own, simulate
SUCCESS, stress 17/20, chose `look_again(cup_1)` -> 20/20. The brute-force value-of-information search in
`active_perception` (all 7 objects) independently picked the same object (cup_1 20/20, coaster_1 19/20, rest 17/20).

## 3. Log-feedback ablation

Harness: `experiments/agent/ablation.py` (real Muse Spark 1.3 Contributor; scored on each scene's verified
`reference.goals` + implicit `undisturbed`, never on goals the model compiled). Two experiments:

- **pipeline** — the product loop (plan -> simulate -> revise, max 3 attempts). Paired: the first plan is written
  and simulated once per repeat; only if it fails, revisions run from that same failure under each condition:
  `full_log` (compact: failed goals with numbers, execution_error, evidence sentences — product default),
  `full_log_raw` (old raw history with all contact events), `outcome_only` (failed predicate names, aborted flag),
  `none` (success=false only).
- **loop** — the agent loop with tools simulate + submit only (max 6 calls); `feedback` sets what those tools return.

### Run 3 / Evaluation v3 (supersedes runs 1–2 for any quoted number) — see `EVAL.md` "Evaluation v3"

Current main, shipped English `demo_instruction`, product pipeline (Muse compiles goals), n = 20 per scene over
five hard scenes, harness `eval/run_eval.py --demo-only`, scored on `reference.goals`, $1.12. First plan failed
86/100 (desk_02 19, kitchen_01 20, lamp_01 8, storage_01 20, windowsill_01 19 of 20). On those 86 failed plans,
recovered within 2 revisions: full log **85/86**, failed predicates only 49/86, "it failed" only 38/86
(47:0 discordant, exact McNemar p = 1.4e-14). kitchen_01: 20/20 vs 0/20 vs 1/20. Run 2's kitchen "0/5" is
confirmed for English; EVAL v1's "none 20/20" was the Chinese instruction (v3 Chinese arm: blind 13/14 — the
model reorders on its own when asked in Chinese, never when asked in English "then"). Goals compiled vs handed in
made no difference (blind 1/20 vs 1/14).

### Run 2 — `ablation_20261031_031322.json` (English instructions + English SYSTEM, robotics round-2 scenes), 5 repeats

Pipeline, revisions from the same failed first plan (fixed by attempt 2 / by attempt 3, over repeats whose first plan failed):

| scene | first plan failed | full_log | full_log_raw | outcome_only | none |
|---|---|---|---|---|---|
| windowsill_01 | 5/5 | 4 / **5** of 5 | 3 / 5 of 5 | 2 / 3 of 5 | 2 / 3 of 5 |
| kitchen_01 | 5/5 | 4 / **5** of 5 | 4 / 5 of 5 | 0 / **0** of 5 | 0 / **0** of 5 |
| lamp_01 | 2/5 | 1 / 2 of 2 | 1 / 2 of 2 | 1 / 1 of 2 | 1 / 2 of 2 |
| storage_01 | 5/5 | 5 / **5** of 5 | 5 / 5 of 5 | 3 / 4 of 5 | 0 / 2 of 5 |
| **pooled (17 failed first plans)** | | **14 / 17 of 17** | 13 / 17 | 6 / 8 | 3 / 7 |

With the simulator's evidence, Muse Spark repaired **17/17** failed first plans within two revisions (14/17 with
one); with only the names of the failed predicates 8/17, with "it failed, try again" 7/17. kitchen_01 (the fix is
to *reorder* the actions because the gripper body lands on the kettle) is never solved without the log: 0/5 and
0/5 vs 5/5. Compact vs raw history: 17/17 vs 17/17 (14 vs 13 at attempt 2), so compacting costs nothing.
desk_02 and the run-2 loop experiment did not complete: from 02:40 ET the Meta API stopped answering (every open
request hung in an SSL read for 30+ min, no caller in the shared ledger logged a call), so those runs were
killed; raw per-trial records are in `experiments/agent/partial_r2_*.jsonl`.

### Run 1 — `ablation_20261031_005257.json` (Chinese instructions, pre-English prompt), 5 repeats, $0.097

Pipeline (counts over repeats whose first plan failed):

| scene | first plan failed | full_log fixed @2 / @3 | full_log_raw | outcome_only | none |
|---|---|---|---|---|---|
| windowsill_01 | 5/5 | **5/5 / 5/5** | 5/5 / 5/5 | 0/5 / 3/5 | 2/5 / 4/5 |
| kitchen_01 | 2/5 | 2/2 / 2/2 | 1/2 / 2/2 | 2/2 / 2/2 | 1/2 / 2/2 |

Loop (n=5 each, all conditions 5/5 verified success in the end; first simulate failed 5/5 on windowsill):
simulations until success on windowsill — full_log 5,5,5,3,4; outcome_only 4,4,4,5,4; none 6,4,5,6,5.
kitchen_01: the loop's first simulate failed only 2/15 (the model usually moved the kettle first by itself).

Reading: with one revision, the evidence matters a lot (windowsill fixed on the first revision 5/5 with the
log vs 0/5 outcome-only, 2/5 blind). Given more tries, blind search over one parameter (transit height) often
gets there too, and inside the agent loop — where the model can call simulate repeatedly — the three
conditions all reach 5/5; blind runs just need more simulations. Compact vs raw history: no loss (5/5 vs 5/5
on windowsill, 2/2 vs 1/2 by attempt 2 on kitchen) at 43% fewer input tokens (mean prompt 1331 vs 2352 tokens per revise call, 7 vs 8 calls; these scenes have the robotics `evidence`, so the raw history is still modest).

## 4. Other changes in agents.py

- `MuseAgent.revise` sends `compact_attempt()` records (plan, failed goals with numbers, `execution_error`
  without the echoed action, `evidence` sentences) instead of raw histories with every contact event.
  `compact_history=False` restores the old behaviour (measured in the ablation as `full_log_raw`).
- `OfflineAgent(scene)`: if the scene has a `reference` block, replays goals / `naive_plan` / `fixed_plan`, so
  offline demos of windowsill_01 / kitchen_01 show a real fail -> fix. `make_agent(data, scene=None)`.

## 5. Known issues

- The loop's `look_again` is a model of re-observation (sigma 1.2 -> 0.4 cm), not a new photo / SAM call.
- Wall time is dominated by model latency (about 10-20 s per turn with default reasoning effort); a run takes
  1.5-3 min including stress tests.
- `chat_tools`/`chat_json` have no wall-clock guard: during the 02:40 API stall, requests hung far past the
  client's 300 s timeout, and their reservations stay in `budget/state.json` "reserved" (about $0.15).
  The ablation now saves per trial (`--partial`) and runs one process per scene.
