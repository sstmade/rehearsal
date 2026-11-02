# DROID v2: pre-registered evaluation protocol

Written and committed **before** any v2 scene was built, any replay was run or any real final position was measured.
Episodes were chosen from contact sheets of raw frames and from trajectory statistics (measured finger plateau,
EE lift) only; no replay or tracking had been run on them. Anything changed after results are seen is reported as
a deviation in `docs/real_robot.md` (v2 section), with both numbers.

## Question

Does the physics replay predict where objects end up **better than trivial baselines**, in particular for
collateral effects (objects the gripper did not grasp that were nudged, knocked over, slid or fell)?

## Episodes (17, 7 labs; list in `datasets/droid/episodes_v2.json`)

Selection rules (applied before any replay):
- DROID 1.0.1 raw, in `KarlP/droid` `cam2base_extrinsic_superset.json` (improved extrinsics) with per-episode
  intrinsics, both exterior cameras' calibration quality metric >= 0.85.
- Successes: language annotation of the form "put/place/move X in/on/into Y", rigid objects (no cloth, paper,
  drawers, doors, appliances), random sample (seed 0) of up to 5 per lab, then kept if the measured finger
  plateaued below 0.9 while closed and the EE lifted >= 4.5 cm in that hold (so an object actually moved).
- Failures: failure episodes from the same lab, operator and day as a kept success, kept if they also hold and lift
  (so an object moved even though the task failed).
- Kept only if, in an 8-frame contact sheet of each exterior camera, objects are visible at frame 0, the scene is a
  table/counter in front of the robot, and the chosen camera shows **no person**. The packaged clip is then checked
  again on a denser frame sheet; an episode is dropped (not replaced) if a person appears.
- Camera: ext1 for all 17 (it is the people-free view in every kept episode).

| lab | successes | failures |
|---|---|---|
| AUTOLab | 1 | 0 |
| CLVR | 2 | 2 |
| GuptaLab | 3 | 1 |
| ILIAD | 1 | 2 |
| IRIS | 2 | 0 |
| PennPAL | 1 | 0 |
| RPL | 1 | 1 |

## Pipeline (unchanged from v1 except where stated)

1. Scene from frame 0 (`pre.png`) with the episode's calibrated camera: Muse Spark listing + SAM 3.1 masks +
   `estimate_scene(method="guarded")`, open containers, de-overlap; goals compiled by Muse Spark from the
   instruction (failures use the instruction of the same-scene success). No hand edits.
2. **Table height, fixed rule (v2):** z_table (robot base frame) = min over the episode of the recorded EE (flange)
   z − 0.185 m. This is the v1 fingertip bound taken at its midpoint (v1: tip 0.17–0.19 m below the flange; v1's two
   scenes measured min EE − 0.198 and − 0.174). No stereo sweep in v2.
3. Replay recorded joints, operator gripper command, bracket-grasp weld — code unchanged from v1.
4. Real positions: one SAM 3.1 video request per episode on the clip **subsampled to every 2nd frame (7.5 fps,
   last frame always kept)** to fit the API budget, seeded with the twin's frame-0 boxes.
   **Measurement is predictor-neutral:** every object's real start and final xy are fitted with its bottom on the
   table (z0 = 0), not at the replay's predicted height (v1 used the replay's height; reported as secondary).

## Predictors of each object's final table xy

- **(a) replay**: the physics replay's final xy.
- **(b) stay-put**: the object's frame-0 twin xy (`scene_xy`).
- **(c) release point**: reads only the trajectory and the frame-0 twin xy. Close segments = maximal runs of
  gripper command > 0.5 (v1 `droid_trivial_baseline.close_segments`). A segment *holds* if its measured finger
  plateau (max over the segment after 8 settling steps) is < 0.90. For each holding segment, the grasped object is
  the object whose current predicted xy is nearest to the EE (flange) xy at the segment's first step, if within
  10 cm; its predicted xy becomes the EE xy at the segment's last step (release, or end of episode). Other objects
  stay put. Segments are processed in order.
- **(d) success: gripper rule** (v1 finger variant, unchanged): success iff some close segment holds (plateau
  < 0.90) and the EE then rises >= 5 cm while closed. Also reported: always-success, always-failure.

## Metrics

Over measurable objects (SAM track present at the clip's end, as in v1):
- per-object final-position error |pred − real final xy| for (a), (b), (c);
- subsets: **all**, **moved** (real displacement > 3 cm, `reality.GAP_TOL_M`), **collateral** (moved, and not
  the object that baseline (c) assigns to any holding segment, i.e. not the grasped object);
- moved / not-moved agreement at 3 cm for (a), (b) (always "not moved"), (c);
- success agreement with the DROID label for replay predicates, (d), always-success, always-failure.

Statistics:
- Wilcoxon signed-rank test (two-sided, scipy default `zero_method="wilcox"`) on paired per-object errors,
  replay vs (b) and replay vs (c), on each subset with >= 6 pairs;
- episode-clustered bootstrap (resample episodes with replacement, 10,000 resamples, seed 0, percentile 95% CI)
  for each predictor's median error and for the mean paired difference (replay − baseline) per subset.

## Decision rule

The replay **beats** a baseline on a subset iff the 95% CI of the mean paired difference (replay − baseline) lies
entirely below 0 **and** the Wilcoxon p < 0.05. It is **worse** iff the CI lies entirely above 0. Otherwise:
"not distinguishable on this sample". The headline question is answered by the **moved** and **collateral**
subsets against baseline (c). If the collateral subset has fewer than 6 objects, we say it is too small to answer.

## Exclusions and budget

- Objects SAM loses before the end of the clip are excluded from position metrics (counted and reported).
- An episode whose scene build or replay crashes is reported and excluded; no episode is swapped after results.
- Meta API cap for v2: $0.50 (Contributor mode; DROID is public). If the cap would be exceeded, the remaining
  episodes are dropped in the listed order from the end, and this is reported.

## Amendment 1 (made after packaging, before any scene, replay or tracking result)

The per-episode table rule put the table 7–8 cm too high in `iliad_marker_mug`: its gripper only grasped the marker
upright and never went down to the table, while two failures of the same scene, filmed minutes apart from an
unchanged camera, went 7–8 cm lower. Changed rule: episodes of one scene (same lab, operator id and day, and camera
position within 5 cm in the robot base frame) share one table height = min EE z over all of them − 0.185 m
(`package_v2.py --pool`). This affects ILIAD (3 episodes) and GuptaLab (4 episodes, ≤ 1.2 cm change). The other
episodes keep their per-episode value. The same rule was v1's (min over each scene).
