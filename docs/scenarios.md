# Scenarios: where a reasonable first plan fails in physics

All numbers below were measured in this repo (MuJoCo, Franka Panda, suction tool) on 2026-10-31.
"Stress" = `rehearse.stress_test` (friction x0.6-1.4, mass x0.7-1.3, sigma = 1.2 cm perception error per object).
"Muse" = real Muse Spark 1.3 (Contributor mode, these scenes are synthetic / `privacy: public`), driven by
`MuseAgent.compile_goals/plan/revise` with the evidence `run_plan` returns. Total robotics API spend: < $0.01.

## What the simulator now reports (so a text model can diagnose)

`run_plan` returns, besides the old keys:

- `execution_error`: `{action_index, action, error, phase, ...}`. The error is a sentence. For IK failures it includes
  `target_xyz`, `residual_cm`, `segment_from/to`, `progress`, `horizontal_dist_from_base`,
  `max_reachable_tool_height_here` and `max_reachable_tool_height_at_segment_end`, found by bisecting IK.
  Example: *"IK failed during 'carry mug_1 to tray_1': at xy [0.741, -0.043] the tool tip can reach at most z=0.42 m,
  the move asked for z=0.45 m; above the move's end point [0.79, -0.08] the max is z=0.26 m"*.
- **Protective stop** (new physics rule): if a robot link presses on an object it is not handling with more
  than 40 N (normal force), the plan aborts there, the way a real Franka's collision reflex does.
  Normal plans peak at about 15 N on non-handled objects, so this only triggers on real collisions.
- `events[kind=contact]` now carry `link` (`hand`, `link7`, ...), `link_role` (`gripper body`, `wrist flange`,
  `forearm`, ...), `phase` (`"lower cup_1 on coaster_1"`), `at_xyz`. `between` still says `robot_arm`.
  The suction tip touching the object it is about to pick no longer shows up as a contact.
- `disturbed`: objects that ended up where the plan did not put them: `moved_cm`, `tilt_deg`, `final_xyz`,
  `fell_off_table`, `released_mid_air_after_error`, and `first_hit` (who hit it, which link, which phase).
- `arm_contacts`: peak normal force per (object, robot link).
- `evidence`: the above as plain sentences. This is the part to show a language model.

`transport` takes an optional `via: [[x, y], ...]`: the carry passes through these points at `transit_height`
between lifting the object and arriving above the destination. A missing or empty list means a straight carry.

## desk_01 (unchanged): mug into bowl, stack books

- Instruction: `把杯子放进碗里，然后把书摞整齐`
- Offline agent at 0.15 m: the **wrist flange (link7)** hits `bottle_1` at z=0.25 m during
  `carry mug_1 to bowl_1`; the bottle falls off the table (moved 158 cm, 90 deg). Revised to 0.42 m: success.
  Stress on the fixed plan: 35/40.
- Muse plans 0.45 m and succeeds on the first try, so this scene shows no correction with the real model.
  That is why the next two scenes exist.

## windowsill_01: reach limit against clearance (needs `via`)

- Instruction: `把马克杯放到托盘上` (mug_1 at [0.35, 0.25] -> tray_1 at [0.79, -0.08], r = 0.80 m from the base).
  A 0.32 m vase sits on the straight line between them.
- Why a careful plan fails: clearing the vase with the mug hanging under the tool needs transit >= 0.32 + 0.10 = 0.42 m.
  The top-down workspace shrinks with distance: above the tray the tool tip can reach **at most ~0.26 m**.
  The scene summary does not show this; only IK reveals it. Straight carries, measured:

  | transit | result |
  |---|---|
  | 0.20 | wrist flange (link7) hits the vase; the vase falls off the table |
  | 0.30 | protective stop: gripper body pressed on vase_1 with 42 N |
  | 0.40-0.50 | IK failure mid-carry; above the tray the max is 0.26 m |

- The fix: a detour on the robot side of the vase, low enough to reach the tray. `transit 0.20, via [[0.45,-0.05]]`:
  success (sim 7.5 s), stress 37/40.
- **Muse** (with `via` added to the schema in a scratch driver, see the interface request):
  attempt 1 already detours, but at 0.50 m. IK fails: *"...at most z=0.48 m... above the move's end point
  [0.79,-0.08] the max is z=0.26 m"*. Attempt 2's diagnosis: *"IK/workspace limit, not collision ... lower transit to
  0.22 m (<0.26 m limit)"*, keeping the detour. Success (sim 8.2 s); stress on that plan 38/40.
- Offline agent: cannot solve it (it only raises transit height after collisions). Expected.
- Without `via` in ACTION_SCHEMA, no straight plan can succeed in this scene.

## kitchen_01: order matters because of the gripper's size

- Instruction: `把杯子放到杯垫上，再把水壶放到隔热垫上`. coaster_1 [0.72, 0.06] sits right behind kettle_1
  ([0.565, 0.06], r 0.075, h 0.25) as seen from the robot. There is a 3 cm gap between the coaster edge and the kettle.
- Why the natural plan (instruction order) fails: the cup fits on the coaster, but the Panda hand is about 20 cm long
  along the reach direction. While lowering the cup, the **gripper body (hand)** comes down on the kettle top.
  Raising transit_height does not help, because the collision happens in the vertical lowering phase.
- Evidence (Muse attempt 1, transit 0.45): *"protective stop during 'lower cup_1 on coaster_1': the robot's gripper body
  (hand) pressed on kettle_1 with 62 N at [0.64, 0.06, 0.25] (tool tip at [0.716, 0.06, 0.21])"*.
  That run used the earlier 60 N threshold. With the current 40 N threshold the wrong order fails 0/24 under
  stress at transit 0.40 and 0.50, and 1/24 at 0.45.
- Fix: move the kettle first. **Muse** attempt 2: *"Fix ordering: move kettle to isolated trivet first to clear coaster
  approach, then place cup."* Success (sim 14.5 s). Stress on kettle-first at 0.45 m: 36/40.
- Offline agent: gets the goals wrong (it only knows the desk vocabulary). Its revise rule would raise transit, which
  does not fix this scene. Expected.
- Distractors (plate, apple, cutting board) are checked by the implicit `undisturbed` goal.

## Performance

A/B on desk_01, the same 3-action plan, base commit vs this branch, run back-to-back. The machine was shared and
loaded, so read the minimum times:

| plan | base min | new min |
|---|---|---|
| 0.15 m | 2.86-3.67 s | 2.27-2.89 s |
| 0.45 m | 5.88-6.04 s | 2.20-2.79 s |

The new code is faster because geom-to-body names are cached instead of looked up on every contact on every step.
The contact force is computed only for robot contacts with non-handled objects.
IK-failure diagnostics add up to about 15 extra IK solves, and only on a failing run.

## Known limits

- `via` applies only to the loaded carry. The empty approach to the next object still goes straight at that
  action's transit height.
- `max_reachable_tool_height_*` comes from a seeded numerical IK, so it can differ by about 1 cm between calls.
- When a plan aborts, the held object is released where it is. It can land on the goal by luck, and the
  predicates then pass, but `success` is still false because of `execution_error`.

## `reference` block in the hard scenes

`kitchen_01.json` and `windowsill_01.json` each carry an optional `reference: {goals, naive_plan, fixed_plan}`.
Each one was verified by `run_plan`: naive fails (kitchen: protective stop; windowsill: IK) and fixed succeeds
(sim 14.5 s / 7.5 s). The OfflineAgent only knows desk_01's vocabulary. On these scenes it compiles no goals
and "succeeds" trivially with an empty plan. An offline demo of these scenes should replay `reference` instead.
