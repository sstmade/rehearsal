# Rehearsal report — droid_duck_success

**Instruction:** Put the orange rubber duck in the pot.  
**Agent:** muse-spark-1.3-contributor

## Verified by simulation

### Reachability (IK, top-down suction)

- ✅ `pot_1`
- ✅ `duck_1`

### Attempts

**Attempt 1: success**

- ✅ `inside(duck_1, pot_1)` — xy offset 0.2 cm (limit 3.9), bottom z 0.8 cm

![attempt 1](attempt_1.gif)

## Estimated under uncertainty

Final plan re-run 20 times with friction ×0.6–1.4, mass ×0.7–1.3 and per-object perception error (σ from the scene, default 1.2 cm), seeds 1000+: **0/20 succeeded**.

Failures:
- seed 1000: max perception error 9.9 cm, friction ×1.02 — inside(duck_1, pot_1): xy offset 30.0 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 6.5 cm, height offset -0.0 cm)
- seed 1001: max perception error 7.6 cm, friction ×1.09 — inside(duck_1, pot_1): xy offset 21.3 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 7.1 cm, height offset -0.0 cm)
- seed 1002: max perception error 6.6 cm, friction ×0.90 — inside(duck_1, pot_1): xy offset 12.9 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 7.7 cm, height offset -0.0 cm)
- seed 1003: max perception error 7.4 cm, friction ×0.75 — inside(duck_1, pot_1): xy offset 10.1 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 7.4 cm, height offset -0.0 cm)
- seed 1004: max perception error 7.0 cm, friction ×0.60 — inside(duck_1, pot_1): xy offset 9.7 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 9.6 cm, height offset -0.0 cm)
- seed 1005: max perception error 5.1 cm, friction ×0.67 — inside(duck_1, pot_1): xy offset 18.9 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 6.6 cm, height offset -0.0 cm)
- seed 1006: max perception error 9.0 cm, friction ×0.87 — inside(duck_1, pot_1): xy offset 15.2 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 8.3 cm, height offset -0.0 cm)
- seed 1007: max perception error 4.1 cm, friction ×0.66 — inside(duck_1, pot_1): xy offset 18.6 cm (limit 3.9), bottom z 1.2 cm; action 0: suction cup missed duck_1's top face (xy offset 3.6 cm, height offset 0.2 cm)
- seed 1008: max perception error 11.8 cm, friction ×0.73 — inside(duck_1, pot_1): xy offset 26.8 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 12.1 cm, height offset -0.0 cm)
- seed 1009: max perception error 5.8 cm, friction ×0.65 — inside(duck_1, pot_1): xy offset 7.7 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 7.1 cm, height offset -0.0 cm)
- seed 1010: max perception error 7.5 cm, friction ×0.68 — inside(duck_1, pot_1): xy offset 16.7 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 8.3 cm, height offset -0.0 cm)
- seed 1011: max perception error 8.9 cm, friction ×0.83 — inside(duck_1, pot_1): xy offset 22.8 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 8.6 cm, height offset -0.0 cm)
- seed 1012: max perception error 12.7 cm, friction ×0.85 — inside(duck_1, pot_1): xy offset 20.2 cm (limit 3.9), bottom z -0.0 cm; action 0: protective stop during 'descend to duck_1': the robot's suction tip/finger (left_finger) pressed on pot_1 with 40 N at [0.607, -0.037, 0.064] (tool tip at [0.598, -0.042, 0.073])
- seed 1013: max perception error 8.4 cm, friction ×1.24 — inside(duck_1, pot_1): xy offset 17.3 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 7.0 cm, height offset -0.0 cm)
- seed 1014: max perception error 5.8 cm, friction ×0.95 — inside(duck_1, pot_1): xy offset 17.0 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 6.1 cm, height offset -0.0 cm)
- seed 1015: max perception error 5.6 cm, friction ×0.89 — inside(duck_1, pot_1): xy offset 7.5 cm (limit 3.9), bottom z 0.5 cm
- seed 1016: max perception error 5.2 cm, friction ×0.78 — inside(duck_1, pot_1): xy offset 23.0 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 5.1 cm, height offset -0.0 cm)
- seed 1017: max perception error 5.4 cm, friction ×1.33 — inside(duck_1, pot_1): xy offset 18.3 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 5.8 cm, height offset -0.0 cm)
- seed 1018: max perception error 5.5 cm, friction ×1.08 — inside(duck_1, pot_1): xy offset 22.5 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 5.6 cm, height offset -0.0 cm)
- seed 1019: max perception error 5.8 cm, friction ×0.77 — inside(duck_1, pot_1): xy offset 15.9 cm (limit 3.9), bottom z -0.0 cm; action 0: suction cup missed duck_1's top face (xy offset 3.9 cm, height offset -0.0 cm)

## Assumed (not verified)

- Grasping always succeeds (suction weld). Real grasp physics is out of scope.
- Object geometry is approximated by primitives; perception is taken as given.
