# Tier A — YCB-Video (BOP'19 test subset), 6 real frames with 6-DoF GT

Real RGB-D frames (PrimeSense/Xtion, 640x480) of YCB objects (cans, boxes, mustard bottle, mug, bowl-like
pitcher, drill, marker, clamps, banana) standing on tables/desks, camera 50-75 deg from vertical.

| file | scene | objects | notes |
|---|---|---|---|
| ycbv_test_000048_001024 | desk top with a box on it | 5 | objects sit on the box top, plane = box top |
| ycbv_test_000050_000640 | checkerboard on wood table | 5 | checkerboard = visual reference (square size not given by the dataset) |
| ycbv_test_000052_000001 | lecture-chair writing tablet | 3 | small curved surface |
| ycbv_test_000055_000128 | wood table | 6 | bleach bottle on wood block (not on table) |
| ycbv_test_000057_000001 | grey office desk w/ keyboard | 5 | cleanest scene |
| ycbv_test_000059_001152 | grey office desk w/ keyboard | 6 | cleanest scene |

Source: `test/<scene>/rgb|depth/<im>.png` + `scene_camera.json`, `scene_gt.json`, `scene_gt_info.json` from
`https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_test_bop19.zip` (extracted with HTTP range
requests by `../_tools/remotezip.py`; only ~11 MB fetched). Object sizes from `models/models_info.json`
(`ycbv_models.zip`). License: **MIT**, Copyright (c) 2017 Robotics and State Estimation Lab, University of Washington.
Cite: Xiang et al., "PoseCNN", RSS 2018; Hodan et al., BOP challenge.

## Files per frame
- `<stem>.png` — original RGB, untouched. `<stem>_depth.png` — uint16, **value x 0.1 = mm** (registered to RGB).
- `<stem>.json` — produced by `../_tools/bop_to_json.py` (format below, shared with `../tierA_hope`).

## JSON format (frames, units)
- `camera.K` — 3x3 pinhole intrinsics (px). No distortion (BOP convention).
- GT frame: **camera frame, OpenCV convention** (x right, y down, z forward), **meters**. `R_model_to_cam`,
  `t_model_to_cam_m` are the dataset's `cam_R_m2c`, `cam_t_m2c/1000` (model origin -> camera).
  `center_cam_m` = 3D bounding-box center of the model in camera coords (YCB model origins are not the bbox center,
  so use this, not `t`). `size_m_model_axes` = model AABB size (m) along the model's own x,y,z;
  `model_axis_closest_to_table_normal` tells which of those is "height".
- `visib_fract`, `bbox_obj_px`, `bbox_visib_px` (x, y, w, h) — straight from `scene_gt_info.json`.
- `table_plane` — **derived by us, not dataset GT**: RANSAC on the GT depth around the objects, hypotheses ranked by how
  many GT objects rest on the plane, LSQ refit. `normal_cam . X + d_m = 0`, normal points to the camera.
  Sanity fields per object: `bottom_height_above_plane_m` (lowest bbox corner; ~0 => resting on the plane, usually
  -1..+1 cm), `top_height_above_plane_m`, `on_table` (|bottom| < 2 cm).
- **Table frame** (meters, right-handed): origin = where the optical axis hits the plane; +z = plane normal (up,
  towards camera); +x = image-right projected on the plane; +y = z x x (away from the camera, image-up).
  `T_table_from_cam` (4x4): `X_table = T[:3,:3] @ X_cam + T[:3,3]`. Each object has `center_table_m` and
  `table_xy_m` (= its first two components = footprint of the bbox center on the table).
- `homography_px_to_table` (3x3): pixel (u,v,1) -> table (x,y,1) for points **on the plane** — same convention as
  `scene.photo.homography_px_to_table` in CONTRACTS.md, so a perception result can be compared directly.

## How to evaluate perception against it
```python
import json, numpy as np
gt = json.load(open("datasets/tierA_ycbv/ycbv_test_000059_001152.json"))
H = np.array(gt["table_plane"]["homography_px_to_table"])
def px_to_table(u, v):
    p = H @ [u, v, 1.0]; return p[:2] / p[2]
# For a detected object: use the pixel where it touches the table (e.g. bottom-center of its mask), map with H,
# compare to obj["table_xy_m"]. The table origin/axes differ from Rehearsal's A4-anchored frame, so compare
# pairwise distances between objects, or fit a 2D rigid transform (Procrustes) before computing errors.
```
Note: the bbox-center footprint is not exactly the contact-point footprint for objects lying on their side; expect
a few mm–cm of slack. There is no A4 sheet in these frames: to test the A4 path, use `K` + `T_table_from_cam`
instead (true metric scale).
