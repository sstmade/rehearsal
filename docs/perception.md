# Perception: photo → digital twin

`rehearsal/perception.py`, `rehearsal/calib.py`, `rehearsal/synth.py`.

## Pipeline

1. **Muse Spark** (`muse-spark-1.3`, strict JSON schema, `reasoning_effort="low"`) lists the movable objects: id, label,
   a short SAM noun phrase (`category`), rough image center, primitive shape, dimensions, mass and friction ranges.
2. **SAM 3.1** runs one concept request per category, which returns every instance. Each listed object takes the
   nearest unclaimed box. Objects left without a box get a point prompt at Muse Spark's center
   (`source: "point"`, `confidence: "low"`).
3. **Calibration from an A4 sheet** (`calib.py`). The paper's 4 corners give the pixel→table homography. From it we recover
   a full pinhole camera (square pixels, principal point at the image center, focal length from the two rotation
   constraints r1·r2 = 0 and |r1| = |r2|, with a fallback to a 26 mm-equivalent phone lens).
4. **Placement (`box_fit`)**. Each object is a 3D primitive (cylinder or bowl; a box is treated as a cylinder with radius
   √(l·w)/2 because its yaw is unknown). We solve for (x, y, overall scale) by least squares so that the primitive's
   projected 2D bounding box matches SAM's box. Box edges that touch the image border are ignored. The proportions come
   from Muse Spark. The scale has a weak prior at 1 and is clamped to [0.4, 2.5], and the scaled dimensions are written
   to `shape`.
5. **Tidy**: objects are clamped onto the table and kept out of the robot base (r = 12 cm). Overlapping footprints
   are pushed apart. Every change is listed in `perception.adjusted`.

## Corner convention

Pixel corners are always given as **near-left, near-right, far-right, far-left**. "Near" means closer to the robot
base. "Left" means the robot's left (+y) when the robot looks along +x. The paper's **long side runs along +x**.
`paper_xy` is the robot-frame position of the near-left corner, default `(0.30, 0.10)`, so the sheet covers
x ∈ [0.30, 0.597] and y ∈ [−0.11, 0.10].

`detect_paper` (numpy/PIL only, no OpenCV) finds the sheet:
1. Scores each pixel as white and unsaturated, then labels connected components at 480 px width.
2. Keeps the component with the highest score among the solid, quadrilateral ones.
3. Re-extracts its boundary at full resolution and takes the convex hull.
4. Simplifies the hull to straight edges. While there are more than 4 edges, it drops the shortest one by extending
   its two neighbours. This recovers a corner hidden behind an object.
5. Fits a line to each side.

`order_corners` makes two decisions:
- **Which edges are the long ones.** It picks the hypothesis whose two focal-length constraints agree, and also checks
  orientation: seen from above, NL→NR→FR→FL has a negative shoelace area in (u right, v down).
- **Which short edge is near.** The paper alone cannot show this, so the caller passes `robot_side` (where the robot
  base is in the photo). The default is `"bottom"`, meaning the photo was taken from behind the robot.

## Measured accuracy (synthetic ground truth)

`python -m rehearsal.synth render` uses MuJoCo's offscreen renderer to draw scenes built by `scene.build_model`, with
the robot hidden. Each render is 1280×960 at 55° vertical FOV, with a shadowed key light, vignetting and sensor noise.
An A4 sheet sits at a known pose, and the exact K, R, t are read back from MuJoCo. `python -m rehearsal.synth eval
[--corners gt|auto]` runs the real Muse Spark + SAM pipeline on each render (`data="public"`) and compares each
object's estimated xy with ground truth.

The set is 3 scenes × 4 camera poses = 12 images, 76 objects. The scenes are desk_01 plus two random layouts with 6
objects each. Camera height above the table is 0.74–0.82 m. The poses:
- **rear50**: behind the robot, 50° down
- **side40**: robot's right side, 40° down
- **front55**: across the table, 55° down
- **oblq45**: rear-right corner, 45° down

All 76 objects were listed and matched; there were no misses and no extras. Raw rows are in
`photos/synth/eval_{gt,auto}.json`.

Placement error is the xy distance in cm, shown as median / P90:

| method | exact corners | auto-detected corners |
|---|---|---|
| `bottom_center` (original MVP: box bottom-center, +r along x) | 4.95 / 11.81 | 5.21 / 11.88 |
| `bottom_ray` (bottom-center, +r away from camera) | 2.37 / 5.08 | 2.60 / 5.58 |
| `ray_mid` (box-center ray ∩ plane z = h/2) | 0.85 / 2.41 | 0.87 / 3.56 |
| `box_fit_fixed` (2D-box fit, Muse Spark dims) | 0.83 / 2.31 | 0.97 / 3.16 |
| **`box_fit`** (2D-box fit + scale; the default) | **0.54 / 1.30** | **0.79 / 2.80** |

`box_fit` broken down by camera pose, median / P90 in cm (19 objects per pose):

| pose | exact corners | auto corners | original `bottom_center`, exact corners |
|---|---|---|---|
| rear50 | 0.54 / 1.20 | 1.02 / 3.42 | 1.50 / 4.13 |
| side40 | 0.70 / 1.36 | 0.88 / 1.29 | 6.34 / 9.96 |
| front55 | 0.49 / 1.10 | 0.48 / 1.32 | 9.33 / 16.55 |
| oblq45 | 0.54 / 1.18 | 0.87 / 5.03 | 4.58 / 7.06 |

`box_fit` broken down by SAM source, median / P90 in cm:
- **concept:** 0.58 / 1.33 with exact corners (n = 55), 0.76 / 2.44 with auto corners
- **point:** 0.52 / 1.14 with exact corners (n = 21), 0.93 / 4.01 with auto corners

On renders the point prompts hit the right object. On real photos they do not always: about half are right, and see
`pill_bottle_1` below. That is why point results stay `confidence: "low"`.

**Paper detection:** all 12 sheets were found. The maximum corner error was ≤ 1.9 px in 11 of 12 images and 4.4 px in
one (rand_2/oblq45, an edge partly occluded). Before the clipped-corner recovery, two images where a book covered a
corner had 10–14 px errors, which gave 15–23 cm object errors. Auto-corner error is dominated by corner accuracy.
Exact corners with the estimated focal length do as well as the true focal length (0.54 / 1.30 both). Forcing the
default 26 mm prior instead gives 0.83 / 1.64.

**Stress-test sigma:** `sigma_m` per axis is 0.010 m for high confidence and 0.020 m for low. The synthetic data
suggests about 0.007 m per axis (median radial error ÷ 1.18). The values are rounded up because real photos are
harder: textures, clutter, lens distortion, no exact A4 pose.

## Real photos (qualitative, no A4)

The Wikimedia photos contain no paper, so there is no metric placement. `detections_only()` returns the listing plus
SAM boxes. The overlays are in `photos/real_overlays/`.
- **messy_desk:** 14 objects listed and all boxed (13 concept, 1 point). The point-prompt `pill_bottle_1` is wrong: its
  box covers the tape dispenser region.
- **gfp_office_desk:** 10 objects listed and all boxed (9 concept, 1 point). The bottles, bowls, mouse, pen, book and
  paper stack are correct.

## Limits

- **SAM masks are not used.** Each box is followed by `<|mask;x=..;y=..;data=H,W,~...|>`, where H×W is the box size + 1
  and the payload is `~` + base-85 text. The alphabet is ASCII 33–126 without `",;<>\|}~`, and it decodes to a 4-byte
  length plus a high-entropy body. That body is not zlib, gzip, lzma, bz2, brotli or zstd, nor a plain or LEB RLE.
  It looks like a custom entropy coder, so placement uses boxes only.
- **Yaw is not estimated** (always 0). Boxes are placed by a yaw-averaged silhouette.
- **Occlusion biases the result.** A partly hidden object gets a truncated box. Edges at the image border are ignored,
  but edges at occluders are not.
- **The sheet must lie flat** at `paper_xy` relative to the robot. Any error in where the sheet sits relative to the
  robot shifts every object by the same amount.
- **There is no lens distortion model.** Phone main cameras are mostly corrected already.
- **The renders are clean primitives**, so real-photo error will be larger than in these tables.

## How to take the photo

- Put an **A4 sheet flat on the table** with its long side pointing away from the robot and its near-left corner at the
  agreed spot (default 30 cm in front of the robot base and 10 cm to its left). Keep all four corners visible.
  The detector can rebuild one corner hidden behind an object, but not more.
- Hold the phone **0.6–0.9 m above the table, tilted 40–60° down**, with the sheet and all objects in frame. Avoid
  shooting straight down: the focal length cannot be recovered from a fronto-parallel sheet, so a 26 mm prior is used.
  Also avoid grazing angles below about 30°.
- Default `robot_side="bottom"`: stand behind the robot, so the robot is at the bottom of the photo. Otherwise pass
  where the robot is, or click the corners by hand in NL, NR, FR, FL order.
- Use even light and avoid strong glare on the paper. A white paper on a white table will not be detected; pass the
  corners by hand in that case.

## API

```python
from rehearsal.perception import perceive_photo, draw_overlay, detections_only, estimate_scene, perceive_raw
scene, overlay_png = perceive_photo(path, privacy="private", corners=None, robot_side="bottom",
                                    paper_xy=(0.30, 0.10), out_dir=..., name=...)
```

- `corners` is `[[u, v] × 4]` in NL, NR, FR, FL order, or `None` to auto-detect.
- `perceive_photo` raises `ValueError` if the sheet cannot be found and no corners were given.
- Each object in `scene` carries `category` and `perception {box_px, source, confidence, sigma_m, method, size_scale,
  adjusted?}`.
- `scene` also carries `photo {file, width, height, homography_px_to_table, paper_corners_px, paper_xy, camera,
  camera_info, corners_source}`, `privacy`, and `perception_notes.not_segmented`.

CLI: `python -m rehearsal.perception photo.jpg [--paper-corners "u,v u,v u,v u,v"] [--robot-side bottom] -o scenes/x.json`.
