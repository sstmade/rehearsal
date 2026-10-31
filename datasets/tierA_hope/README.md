# Tier A — NVIDIA HOPE-Image (BOP val split), 3 real frames with 6-DoF GT

Real RGB-D frames (RealSense D415, 1920x1080) of 15-21 toy grocery items (Melissa & Doug cans, boxes, condiment
bottles, milk/juice cartons) on a black office console table in front of window blinds (strong backlight).
Dense clutter, some objects inside plastic bags (scene 7). Good stress test for counting + SAM segmentation.

| file | objects | resting on fitted plane | camera tilt from vertical |
|---|---|---|---|
| hope_val_000006_000000 | 20 | 18 | ~51 deg |
| hope_val_000007_000000 | 17 | 17 | ~60 deg |
| hope_val_000008_000000 | 21 | 21 | ~49 deg |

(The val split's other 7 scenes are on a couch / inside a cardboard box / in a drawer and were skipped.)
`*_000000` is the best-lit of the 5 lighting variants per scene.

Source: `val/0000NN/rgb|depth/000000.png` + `scene_camera.json`, `scene_gt.json`, `scene_gt_info.json` from
`https://huggingface.co/datasets/bop-benchmark/hope/resolve/main/hope_val_realsense.zip` (range-request extraction,
~42 MB fetched). Object names from `hope_base.zip/hope/dataset_info.md`, sizes from `hope_models.zip/models/models_info.json`.
License: **CC BY-NC-SA 4.0**, Copyright (C) 2020 NVIDIA Corporation (stated in dataset_info.md; the HF card says
cc-by-sa-4.0 — we follow the stricter NC-SA: fine for a non-commercial demo, derivatives must stay NC-SA, attribute).
Cite: Tyree et al., "6-DoF Pose Estimation of Household Objects for Robotic Manipulation: An Accessible Dataset and
Benchmark", IROS 2022.

Depth: `<stem>_depth.png` uint16, **value = mm** (depth_scale 1.0; already corrected by NVIDIA's 0.98042517 factor).
HOPE model origins are the bbox centers, so `center_cam_m` == `t_model_to_cam_m`.

JSON format, frames, table-plane derivation and evaluation recipe: identical to `../tierA_ycbv/README.md`.
