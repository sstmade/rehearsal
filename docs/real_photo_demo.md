# Real photo, end to end: YCB-Video frame → SAM 3.1 → twin → Muse Spark rehearsal

Until now every demo scene in `out/` was hand-authored, and perception was only measured on its own. This is one run
of the whole chain on a **real photo**: a cluttered office desk from YCB-Video. The objects come from Muse Spark and
SAM 3.1, nobody placed them by hand, and Muse Spark then rehearsed a task in the resulting twin. Every number here is
from that run. Nothing was re-run to pick a better outcome.

## Photo and camera

- **Photo:** YCB-Video, BOP'19 test split, scene 000059, frame 001152 (640×480, PrimeSense/Xtion). Copied unmodified to
  `photos/real/real_ycbv59.png`. **License: MIT**, (c) 2017 Robotics and State Estimation Lab, University of
  Washington. Cite PoseCNN (Xiang et al., RSS 2018) and BOP. Source and credit are recorded in `photos/SOURCES.md`.
- **Camera:** this frame has no A4 sheet, so the dataset's own camera is used: K has fx = 1066.8 px, and the table
  plane comes from `datasets/tierA_ycbv/*.json`, the same setup as the real-photo eval in `docs/perception.md`. The
  camera sits 0.52 m above the desk, about 0.7 m back from the objects, at a grazing tilt.
- **Robot frame:** the robot stands on the camera's side of the desk and faces away from it. Robot +x is the dataset
  table's +y, robot +y is its −x, and everything is shifted 0.62 m in x. That one rigid transform is stored in
  `scene.photo.camera_info.robot_from_dataset_table`.
- **Privacy:** `"public"` (Contributor mode), because this is a public, MIT-licensed dataset photo.
- **Reproduce:** `.venv/bin/python scripts/real_photo_scene.py datasets/tierA_ycbv/ycbv_test_000059_001152.json --name
  real_ycbv59 --title ... --instruction ...`. The Meta calls are cached on disk, so a rerun is free.

## Perception: what SAM 3.1 found

Muse Spark listed 8 objects. SAM 3.1 segmented all 8 with concept prompts, so `not_segmented` is empty. All 6
ground-truth objects were found. The keyboard and mouse were also found, but the dataset does not annotate them, so
they count as extras, not errors. Placement uses `estimate_scene(camera=<dataset camera>, method="guarded")`.

The error is the xy distance between each estimated position and the footprint of the ground-truth 3D bounding-box
centre, in the robot frame:

| object | SAM | fit | σ used by stress | raw error | error in final scene |
|---|---|---|---|---|---|
| tomato soup can | concept, mask IoU 0.98 | mask_fit | 1.2 cm | **0.30 cm** | 0.30 cm |
| tuna can | concept, 0.97 | mask_fit | 1.2 cm | **0.48 cm** | 0.48 cm |
| Spam (potted meat) | concept, 0.86 | mask_fit | 2.8 cm | 1.73 cm | 2.04 cm |
| power drill | concept, 0.84 (low) | box_fit | 3.8 cm | 1.78 cm | 1.76 cm |
| Cheez-It box | concept, 0.94 | box_fit | 2.8 cm | 4.61 cm | 4.97 cm |
| dry-erase marker | concept, 0.38 (low) | box_fit | 3.8 cm | 4.91 cm | 4.91 cm |

Median error is **1.8 cm raw and 1.9 cm in the final scene**, with a max of 5.0 cm. That matches the 83-object
real-photo eval: YCB-V `guarded` gave a median of 2.50 cm. The per-object σ ranked the objects correctly. The two
worst objects carry the largest σ or the "low" label, and the two best carry the smallest σ.

**What it got wrong:**

- **The marker is lying on its side.** The twin models it as a 9 cm upright cylinder. This is the known
  "primitives stand upright" limit. SAM's mask IoU of 0.38 flagged it, and it got the low label and the widest σ.
- **The Cheez-It box has yaw 0.** In the photo it is visibly turned. Its centre is 4.6 cm off, mostly because the
  rotated box is wider in the image than the fitted primitive.
- **The drill is a 18×6×20 cm box.** The size is right, but the shape is crude.

All three are visible in `photos/real/real_ycbv59.twin_compare.png`, which shows the SAM overlay next to the MuJoCo
twin rendered from the dataset camera. Every object rests on the desk in the right arrangement.

**De-overlap (one deviation from the default path):** `perception.tidy` pushes overlapping objects apart. It treats
every footprint as its circumscribed circle. On this desk the drill stands against the Cheez-It box and the Spam can
touches the soup can, so tidy moved them 4–10 cm and roughly doubled the median error to 4.9 cm. The script instead
calls `estimate_scene(clamp=False)` and then separates footprints with exact rotated-rectangle and circle shapes
(SAT, in `scripts/real_photo_scene.py`). That moved the Spam can 0.8 cm, the cracker box 1.9 cm and the drill 1.1 cm. Both errors are in the table above. **No position was hand-edited.** Interface request for perception:
make `tidy` use exact footprints.

## Rehearsal (live Muse Spark, `--agent-loop`)

```
.venv/bin/python -m rehearsal.rehearse scenes/real_ycbv59.json "Put the tomato soup can on top of the Cheez-It box." --agent-loop --out out
```

I picked this task before the run because the photo forces it to be hard. The Cheez-It box is 25 cm tall with a
6.2 cm-wide top, which is about the soup can's diameter. The 20 cm drill stands right against it, between the soup can
and the box. An offline screen with a 0.15 m carry height knocked over the drill and the Spam can. Muse Spark was never
shown that screen.

**Muse Spark did not fail nominally.** It chose a 0.38 m carry height on its first try, and the simulation passed.
The failure it found was a **robustness** failure, which it uncovered itself with the stress test:

| # | tool call | result |
|---|---|---|
| 1 | `simulate(soup→cracker_box @0.38)` | SUCCESS |
| 2 | `stress_test(same, n=20)` | **3/20** |
| 3 | `look_again(cracker_box_1)` | 3/20 → 14/20 |
| 4 | `simulate(@0.45 via [[0.55, −0.25]])` | SUCCESS |
| 5 | `stress_test(same, n=20)` | 14/20 |
| 6 | `look_again(power_drill_1)` | 14/20 → 17/20 |
| 7 | `look_again(spam_can_1)` | 17/20 → 17/20 |
| 8 | `submit(@0.45 via [[0.55, −0.25]])` | ACCEPTED |

Muse Spark's narration while it worked, verbatim:

- "The narrow box top makes placement sensitive to perception error, so I'll get a closer look at the box."
- "The drill sits almost touching the box, so I'll re-observe it to pin down its position before the final check."

Muse Spark's verbatim diagnosis on the submitted plan:

> "Direct carry passed nominal simulate but stress 3/20 with placement misses on narrow Cheez-It top (17.7x6.2cm).
> Re-observed cracker_box and power_drill/spam to shrink sigmas, raised transit to 0.45m so can bottom (~0.35m)
> clears 0.248m box and drill, and routed via [0.55,-0.25] to avoid straight-line clip of clutter. Nominal simulate
> passes, stress 17/20."

**Held-out check** (seeds 1000+, n = 40, with the submitted plan):

- **Stress test before looking again: 7/40.** Failure counts overlap, because one run can fail several checks:
  - 32 × `on(soup, cracker_box)`: the can lands off the narrow top.
  - 8 × `undisturbed`.
  - 5 × the suction cup missing the can's top by 2.7–3.2 cm.
- **The look-again selection** uses its own seeds, 0 and up. It picked `cracker_box_1` first, which raised the
  success count from 8/40 to 22/40. It picked `power_drill_1` second, which raised it from 22 to 33/40. This is the
  same order Muse Spark chose inside the loop.
- **Held-out after looking again: 27/40**, up from 7/40.

**Cost and time:**

- **Meta API:** $0.0055 for perception (Muse listing plus SAM 3.1) and $0.0046 for the rehearsal (8 tool calls,
  27.5 k prompt tokens of which 22.3 k were cached). The total is about **$0.010**, from ledger deltas.
- **Wall time:**
  - Perception: 15 s.
  - Rehearsal: 249 s. Most of it went to the stress tests and look-again rollouts in the simulator, not to the model.

## UI check

`REHEARSAL_PORT=8779 REHEARSAL_OFFLINE=1 python -m rehearsal.server` was opened at `/?scene=real_ycbv59`, 1440×900,
with Playwright. The screenshot is `docs/real_photo_demo_ui.png`.

- **What showed up:** the photo panel shows the SAM 3.1 masks: "8 boxes · 8 masks · 2 low-confidence · public data".
  Also present were the top-down twin, the inventory with per-object σ, all three attempts, the Muse Spark and
  SAM 3.1 badges, and "27/40, 7/40 before looking again".
- **Console:** no errors. The only warnings were Chrome's "GPU stall due to ReadPixels" WebGL performance warnings.

Two UI copy issues for the UI owner:

1. The photo caption says "camera calibrated from the A4 sheet". Here the camera came from the dataset, since
   `camera_info.f_source == "given_camera"`.
2. The outcome tile says "3 attempts · 2 self-corrections". All three were nominal successes, and the corrections
   were for robustness.

## Honest limitations

- **The camera is not phone-calibrated.** It is the dataset's known K and table pose, not the A4-sheet calibration.
  The synthetic auto-vs-exact gap adds about 0.25 cm of median error (see `docs/perception.md`).
- **The robot's position relative to the desk is a chosen transform.** It is not measured.
- **One photo and one live run.** Muse Spark's goal compilation dropped `upright(soup_can_1)`: the goals were `on` plus
  the implicit `undisturbed`. The can stayed upright in every simulated success, but that was not checked as a
  predicate.
- **"Look again" is simulated.** It shrinks that object's σ to 0.4 cm. It does not take a second real photo.
- **The modelling errors above stay in the twin.** The marker stands upright and the Cheez-It box has the wrong yaw.
  The plan never touches the marker. The box yaw error affects where the can lands on the narrow top, and that is
  exactly the uncertainty the stress test and look-again target.
- **Run artifacts are gitignored.** `out/real_ycbv59/` holds the report, GIFs, trajectories and stress trails.
