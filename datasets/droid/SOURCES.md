# DROID episodes (real Franka robot footage)

Six teleoperated episodes from **DROID 1.0.1** (raw release, `gs://gresearch/robotics/droid_raw/1.0.1`), all from
one lab (AUTOLab, UC Berkeley), one robot (FR3 + Robotiq 2F-85), two table scenes.

**License: CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). Camera intrinsics and the improved
camera-to-base extrinsics come from the DROID annotation release `huggingface.co/datasets/KarlP/droid` (CC BY 4.0).

Cite: Alexander Khazatsky, Karl Pertsch, Suraj Nair, et al., *DROID: A Large-Scale In-The-Wild Robot Manipulation
Dataset*, Robotics: Science and Systems (RSS) 2024, arXiv:2403.12945.

Video credit line: "Robot footage: DROID (Khazatsky et al., RSS 2024), CC BY 4.0".

| folder | DROID episode id | outcome (DROID label) | camera | task (DROID language annotation) |
|---|---|---|---|---|
| `duck_success/` | AUTOLab+84bd5053+2023-07-14-16h-44m-09s | success | ext2 (24400334) | Put the orange rubber duck in the pot |
| `duck_missed_grasp/` | AUTOLab+84bd5053+2023-07-14-16h-48m-57s | failure | ext2 (24400334) | operator task: "put duck in pot" |
| `duck_fail_nudge/` | AUTOLab+84bd5053+2023-07-14-17h-02m-13s | failure | ext2 (24400334) | operator task: "put duck in pot" |
| `bowl_fail/` | AUTOLab+84bd5053+2023-07-14-16h-17m-55s | failure | ext2 (24400334) | operator task: "stack two bowls on top of each other" |
| `cup_success/` | AUTOLab+84bd5053+2023-07-27-09h-41m-48s | success | ext1 (22008760) | Put the orange cup in the black bowl |
| `cup_fail/` | AUTOLab+84bd5053+2023-07-27-09h-41m-09s | failure | ext1 (22008760) | operator task: "stack cup on bowl" (failures carry no language annotation) |

What each folder holds (all derived from the DROID files by `realrobot/scripts/extract.py` and
`_tools/package_episode.py`; modifications: frame extraction, H.264 re-encode at 960x540 15 fps, JSON repackaging):

- `pre.png`, `post.png`: first and last frame of the exterior camera above, 1280x720, unmodified pixels.
- `clip.mp4`: the whole episode from that camera, 960x540 at the 15 Hz control rate (scale K by 0.75).
- `camera.json`: K, base<->camera extrinsics, and `rehearsal_camera` in the `rehearsal.calib.Camera` convention
  (world = robot base frame shifted so the table top is z = 0). The table height comes from
  `table_height.json` (`_tools/table_sweep.py`, `_tools/table_height.py`).
- `trajectory.json`: the recorded robot state at every control step (joint positions, Robotiq gripper position
  0 = open .. 1 = closed, EE pose = Panda flange in the base frame), plus gripper open/close events.
- `label.json`: DROID's success flag and language annotations. **The success flag is ground truth for evaluation
  only; no Rehearsal predictor reads it.**

**Privacy.** Raw DROID videos are not face-blurred. Only exterior views without people are vendored here; the other
exterior camera, the wrist camera, the stereo videos, contact sheets and `metadata.json` (which names the
operator) are deliberately not copied. Check any new episode frame by frame before adding it.
