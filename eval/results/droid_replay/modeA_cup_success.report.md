# Rehearsal report — droid_cup_success

**Instruction:** Put the orange cup in the black bowl.  
**Agent:** muse-spark-1.3-contributor

## Verified by simulation

### Reachability (IK, top-down suction)

- ✅ `bowl_1`
- ✅ `cup_1`
- ✅ `mallet_1`

### Attempts

**Attempt 1: success**

- ✅ `inside(cup_1, bowl_1)` — xy offset 0.3 cm (limit 7.3), bottom z 0.8 cm
- ✅ `undisturbed(mallet_1)` — ok

![attempt 1](attempt_1.gif)

## Estimated under uncertainty

Final plan re-run 20 times with friction ×0.6–1.4, mass ×0.7–1.3 and per-object perception error (σ from the scene, default 1.2 cm), seeds 1000+: **1/20 succeeded**.

Failures:
- seed 1000: max perception error 9.9 cm, friction ×1.02 — inside(cup_1, bowl_1): xy offset 13.3 cm (limit 7.3), bottom z 0.1 cm; action 0: suction cup missed cup_1's top face (xy offset 3.5 cm, height offset 0.2 cm)
- seed 1001: max perception error 6.0 cm, friction ×1.09 — inside(cup_1, bowl_1): xy offset 28.1 cm (limit 7.3), bottom z 1.7 cm; action 0: suction cup missed cup_1's top face (xy offset 3.9 cm, height offset 0.1 cm)
- seed 1002: max perception error 6.6 cm, friction ×0.90 — inside(cup_1, bowl_1): xy offset 28.5 cm (limit 7.3), bottom z 1.1 cm; action 0: suction cup missed cup_1's top face (xy offset 4.5 cm, height offset 0.1 cm)
- seed 1003: max perception error 7.1 cm, friction ×0.75 — inside(cup_1, bowl_1): xy offset 32.5 cm (limit 7.3), bottom z 0.4 cm; action 0: suction cup missed cup_1's top face (xy offset 4.4 cm, height offset 0.2 cm)
- seed 1004: max perception error 3.9 cm, friction ×0.60 — inside(cup_1, bowl_1): xy offset 28.9 cm (limit 7.3), bottom z -0.0 cm; action 0: suction cup missed cup_1's top face (xy offset 5.9 cm, height offset -0.1 cm)
- seed 1005: max perception error 7.5 cm, friction ×0.67 — inside(cup_1, bowl_1): xy offset 24.8 cm (limit 7.3), bottom z 0.2 cm; action 0: suction cup missed cup_1's top face (xy offset 3.7 cm, height offset 0.3 cm)
- seed 1006: max perception error 9.0 cm, friction ×0.87 — inside(cup_1, bowl_1): xy offset 20.0 cm (limit 7.3), bottom z 0.6 cm; action 0: suction cup missed cup_1's top face (xy offset 4.8 cm, height offset 0.2 cm)
- seed 1008: max perception error 11.4 cm, friction ×0.73 — inside(cup_1, bowl_1): xy offset 24.7 cm (limit 7.3), bottom z -0.0 cm; action 0: suction cup missed cup_1's top face (xy offset 6.7 cm, height offset -0.2 cm)
- seed 1009: max perception error 9.0 cm, friction ×0.65 — inside(cup_1, bowl_1): xy offset 28.9 cm (limit 7.3), bottom z 0.3 cm; action 0: suction cup missed cup_1's top face (xy offset 4.2 cm, height offset 0.2 cm)
- seed 1010: max perception error 5.5 cm, friction ×0.68 — inside(cup_1, bowl_1): xy offset 25.3 cm (limit 7.3), bottom z 0.8 cm; action 0: suction cup missed cup_1's top face (xy offset 4.6 cm, height offset 0.2 cm)
- seed 1011: max perception error 8.9 cm, friction ×0.83 — inside(cup_1, bowl_1): xy offset 29.5 cm (limit 7.3), bottom z 0.1 cm; action 0: suction cup missed cup_1's top face (xy offset 4.9 cm, height offset 0.1 cm)
- seed 1012: max perception error 11.8 cm, friction ×0.85 — inside(cup_1, bowl_1): xy offset 39.9 cm (limit 7.3), bottom z -0.0 cm; action 0: suction cup missed cup_1's top face (xy offset 6.8 cm, height offset -0.2 cm)
- seed 1013: max perception error 8.4 cm, friction ×1.24 — inside(cup_1, bowl_1): xy offset 20.7 cm (limit 7.3), bottom z 0.5 cm; action 0: suction cup missed cup_1's top face (xy offset 4.1 cm, height offset 0.2 cm)
- seed 1014: max perception error 5.7 cm, friction ×0.95 — inside(cup_1, bowl_1): xy offset 24.0 cm (limit 7.3), bottom z 0.1 cm; action 0: suction cup missed cup_1's top face (xy offset 3.3 cm, height offset 0.2 cm)
- seed 1015: max perception error 8.2 cm, friction ×0.89 — inside(cup_1, bowl_1): xy offset 5.5 cm (limit 7.3), bottom z 5.6 cm
- seed 1016: max perception error 6.7 cm, friction ×0.78 — inside(cup_1, bowl_1): xy offset 8.0 cm (limit 7.3), bottom z 5.4 cm
- seed 1017: max perception error 5.4 cm, friction ×1.33 — inside(cup_1, bowl_1): xy offset 23.6 cm (limit 7.3), bottom z 0.3 cm; action 0: suction cup missed cup_1's top face (xy offset 3.4 cm, height offset 0.3 cm)
- seed 1018: max perception error 5.5 cm, friction ×1.08 — inside(cup_1, bowl_1): xy offset 19.1 cm (limit 7.3), bottom z 0.1 cm; action 0: suction cup missed cup_1's top face (xy offset 3.0 cm, height offset 0.2 cm)
- seed 1019: max perception error 5.8 cm, friction ×0.77 — inside(cup_1, bowl_1): xy offset 6.5 cm (limit 7.3), bottom z 5.4 cm

## Assumed (not verified)

- Grasping always succeeds (suction weld). Real grasp physics is out of scope.
- Object geometry is approximated by primitives; perception is taken as given.
