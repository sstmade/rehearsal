# AI-generated test photos, end to end (no A4 sheet)

**Every photo on this page is an AI-generated test photo (image model), not a real capture.** None of them is the
owner's desk or any real room. They test whether the chain photo → Muse Spark + SAM 3.1 → MuJoCo twin → Muse Spark
agent loop → stress test → look again holds up on more than one photo-like table. Provenance:
`photos/ai_gen/SOURCES.md`. Because the images are synthetic, all Meta calls used Contributor mode (`data="public"`).

The real-photo reference run (YCB-Video frame, known camera) is `docs/real_photo_demo.md`. These runs use the same
code path: `estimate_scene(camera=..., method="guarded", clamp=False)`, exact-footprint de-overlap
(`separate` from `scripts/real_photo_scene.py`), then `python -m rehearsal.rehearse <scene> "<task>" --agent-loop`.
The only new piece is where the camera comes from.

## The problem: no A4 sheet

The metric pipeline (`rehearsal/calib.py`) assumes an A4 sheet on the table. None of these photos has one, and a
generated image has no real focal length. `scripts/aigen_photo_scene.py` builds the camera from another reference of
known size that lies flat on the table, and propagates its uncertainty:

- **`rectfit` (used for the two desks).** Muse Spark lists the objects and SAM 3.1 masks them. The corners of each flat
  rectangle come from its mask (hull → quad → straight-edge refinement, the `calib.py` helpers). One rectangle of
  known size gives the scale: a phone, 146.6 × 70.6 × 7.8 mm. It is iPhone-class but the model is unknown, so we
  assume 3 % (1σ). Focal length, tilt and roll are solved so that **every** rectangle has right angles: the phone,
  the closed laptop or tablet, and the desk top. The desk corners are extrapolated from its straight edges, given
  by hand as 4 pixel points. The phone's aspect ratio is matched too. This conditions the fit far better than
  the phone alone. Phone-only homography gave a ±7–17 cm Monte Carlo position spread on `desk_lamp_mug`; the joint
  fit gives 0.6–1.7 cm.
- **`circle` (used for the kitchen).** No rectangle lies flat on the counter, so the reference is the cork coaster
  (10 cm diameter, 8 mm thick, 8 % 1σ). One circle cannot fix the focal length, so f = 0.75 × image width is
  assumed (the pipeline's default phone FOV, 20 % 1σ). The tilt comes from the ellipse's minor/major ratio, after
  removing the visible rim. The distance comes from f × D / major. The roll comes from the counter/backsplash joint.
- **Robot frame.** The robot stands at the near (bottom-of-photo) edge, facing away from the camera. +y runs along
  the far table edge, the near edge is at x = 0.15 m (kitchen 0.20 m), and the table is centred on y = 0. This
  placement is a choice, not a measurement, the same as for `real_ycbv59`.
- **Uncertainty.** We run 30 Monte Carlo draws. Mask corners are jittered by 1.5 px, the hand-given table corners
  by 4 px, the reference size by its σ, and f by its σ (circle only). The whole geometric fit is re-run each time
  (no API). Each object's robust spread (1.4826 × MAD) is added in quadrature to its detector σ. That sum is the
  `sigma_m` the stress test uses. All of it is stored in `scene.perception_notes.scale_reference`.

There is no ground truth for a generated image, so "position sanity" means **cross-checks**: objects of roughly
known size measured on the calibrated table plane.

## Results

| scene | photo | scale reference | cross-checks (measured on the fitted plane) | MC position spread | task | first try | agent stress (own seeds, n=20) | after its look-agains | held-out n=40 before → after look-again | Meta cost |
|---|---|---|---|---|---|---|---|---|---|---|
| `aigen_kitchen` | `tile_kitchen.png` (collage crop, 705×525, not upscaled) | cork coaster ⌀10 cm, f assumed 529 px ±20 %; camera 0.19 m above the counter, view 25.6° above the plane | coaster fits ⌀9.1 cm (ref 10); trivet ⌀12.4 cm (a ~15 cm trivet by pixel ratio: fitted shape is small); kettle ⌀13.8 cm, mug ⌀6.9 cm, apple ⌀6.6 cm, all plausible | 1.4–3.2 cm | "Put the mug on the trivet." (kettle stands between them) | **SUCCESS**: planned a 0.32 m carry with a detour west of the kettle from the start | **6/20** | mug 6→15, trivet 15→19; wider detour 19/20, submitted | **7/40 → 40/40** (look at mug, then trivet) | $0.0052 |
| `aigen_desk_lamp_mug` | `desk_lamp_mug.png` 1448×1086 | phone 146.6×70.6 mm ±3 % + right angles of phone, laptop, desk; f 1550 px, pitch 42.8°, roll 2.5°, camera 0.78 m up | laptop 33×22 cm (13–14″ class); paperback 21.6×12.5 cm; coaster ⌀8.6 cm; desk 57×93 cm, corner angles 88–91° | 0.7–1.7 cm | "Put the cork coaster on the closed laptop, then put the mug on the coaster." (lamp between the mug/coaster and the laptop) | **SUCCESS**: 0.45/0.5 m carries routed via (0.2, −0.15), around and over the lamp | **2/20** | laptop 2→2, mug 2→10, coaster 10→10; raised carries 10/20, submitted | **8/40 → 27/40** (look at mug, then the lamp) | $0.0073 (+ $0.007 for a run that crashed, see below) |
| `aigen_desk_walnut` | `desk_walnut.png` 1448×1086 | phone ±3 % + right angles of phone, tablet, desk; f 1546 px, pitch 55°, roll 4.2°, camera 0.87 m up | closed tablet 24.5–25.6 × 17.3 cm (an 11″ iPad is 24.8 × 17.9); hardcover 16×23.5 cm; coaster ⌀9.9 cm; desk 49×84 cm, angles 88–91° | 0.5–1.6 cm | "Put the mug on the cork coaster, then put the phone on the book." | **FAILED**: mug first, then phone at 0.18 m; the arm, carrying the phone, hit the mug it had just placed. Swapped order (phone first) → SUCCESS | **5/20** | mug 5→15, coaster 15→15, phone 15→18, book 18→19; submitted | **9/40 → 32/40** (look at mug, then phone) | $0.0038 (+ $0.0066 for the first placement, see below) |

Overlays (SAM 3.1 masks, fitted footprints) and twin comparisons (overlay | MuJoCo render from the fitted camera):

- `photos/ai_gen/aigen_kitchen.overlay.png`, `photos/ai_gen/aigen_kitchen.twin_compare.png` (mean silhouette IoU 0.88)
- `photos/ai_gen/aigen_desk_lamp_mug.overlay.png`, `photos/ai_gen/aigen_desk_lamp_mug.twin_compare.png` (mean IoU 0.82;
  the lamp is a 38 cm cylinder, IoU 0.43)
- `photos/ai_gen/aigen_desk_walnut.overlay.png`, `photos/ai_gen/aigen_desk_walnut.twin_compare.png` (mean IoU 0.79;
  scissors 0.27, lamp 0.51)

Run artifacts (report.json, GIFs, trails) are in `out/aigen_<name>/`, which is gitignored. Reproduce:

```
.venv/bin/python scripts/aigen_photo_scene.py photos/ai_gen/tile_kitchen.png --name aigen_kitchen \
  --ref circle:coaster_1:0.10:529:0.008 --f-rel 0.2 --size-rel 0.08 --horizon-px "0,222 705,224" \
  --table-px "0,520 705,520 705,224 0,222" --near-x 0.2 --table-size 0.75,1.0 --check cutting_board_1
.venv/bin/python scripts/aigen_photo_scene.py photos/ai_gen/desk_lamp_mug.png --name aigen_desk_lamp_mug \
  --ref "rectfit:smartphone_1:0.1466x0.0706x0.0078:laptop_1" --table-px "-43,920 1516,902 1255,339 165,345" \
  --check laptop_1,book_1,smartphone_1 --size-rel 0.03
.venv/bin/python scripts/aigen_photo_scene.py photos/ai_gen/desk_walnut.png --name aigen_desk_walnut \
  --ref "rectfit:smartphone_1:0.1466x0.0706x0.0078:tablet_1" --table-px "15,943 1453,940 1273,321 157,321" \
  --check tablet_1,book_1,smartphone_1 --size-rel 0.03
.venv/bin/python scripts/fidelity_eval.py scenes/aigen_kitchen.json --out photos/ai_gen/aigen_kitchen
.venv/bin/python -m rehearsal.rehearse scenes/aigen_kitchen.json "Put the mug on the trivet." --agent-loop --out out
```

(The `--title`/`--instruction`/`--ref-note` flags used for the committed scenes are omitted here for brevity.) The
Meta calls are cached on disk, so a rerun of the perception step is free.

## Skipped or not attempted

- **`desk_ruler.png`: skipped, cannot be calibrated sensibly.** With the steel ruler as the reference (320 mm overall,
  scaled by its length only) plus the notebook and tray, the right-angle fit stays poor. Its residual is 0.090,
  against 0.004–0.008 on the two desks used. It needs an 8.7° roll, and the notebook's opposite edges come out 19.2
  vs 17.6 cm. The generated desk top is not a rectangle (its edges flare and curve), so it cannot help. Also, the
  three **stacked** books are listed as three table-level boxes: the pipeline has no stacking, and de-overlap would
  spread them 33 cm apart. The ruler was also fitted as a 43 cm standing box. The scene was built once for this
  check and not committed.
- **`desk_oak_laptop.png`, `tile_desk`, `tile_coffee`, `tile_windowsill`, `tile_bedside`**: cropped (tiles) but not run,
  for lack of time. Notes: `tile_desk` has the lamp between mug and coaster, but its only flat reference is the
  coaster and the laptop sits partly off the table edge. The bedside tile has a phone, so it could use `rectfit`.

Collage tile boxes in `collage_src.png` (x1, y1, x2, y2): desk (10, 9, 724, 534), kitchen (736, 9, 1441, 534),
coffee (10, 540, 481, 1075), windowsill (488, 540, 966, 1075), bedside (973, 540, 1443, 1075). They were found from
the white gutters.

## Honest limitations

- **There is no ground truth.** The image model drew these scenes, so position error cannot be measured. The
  cross-checks only show that the calibrated sizes are plausible: the tablet within ~2 % of an 11″ iPad, the
  coasters 8.6–9.9 cm against a nominal 10 cm. The "true" phone size of a generated phone is undefined; the 3 % σ
  stands for that.
- **The kitchen focal length is assumed.** Objects at the coaster's depth are robust to it (σ 1.4–1.8 cm). The
  apple and the cutting board, further back, are not (2.5–3.2 cm). The generated counter is not even
  self-consistent: the backsplash joint implies a counter about 1.15 m deep, while the board leaning on the wall
  stands 0.6 m away. The counter is therefore given a fixed 0.75 × 1.0 m extent (`--table-size`), and its
  "corner angles" in the notes are meaningless: the side edges are the image borders. The board leans on a wall
  that is not modelled, so in the twin it stands free.
- **Hand input.** Four table-edge points per desk photo (and two backsplash points) were read off a pixel grid. The
  reference corners come from SAM masks. No object position was hand-edited.
- **Primitives.** The black lamp is a solid cylinder (r 8.6 cm, 38 cm), a keep-out volume rather than its real arm
  and shade. The kettle has no handle, the mug handle comes from the mug template, and the scissors are a box
  (IoU 0.27).
- **Goal compilation** dropped `upright(mug_1)` on the lamp desk (goals: two `on` plus implicit `undisturbed`), as on
  `real_ycbv59`. The kitchen goals kept `upright` and `stable`.
- **"Look again" is simulated.** It shrinks that object's σ to 0.4 cm; it does not take a new photo.
- **Robot placement is chosen, and on the walnut desk the first choice was wrong.** With the near table edge at
  x = 0.15 m, the phone sat 0.23 m from the robot base. That is inside the arm's dead zone: "no top-down pose
  reaches xy [0.228, 0.026] at any height". The agent loop spent all 8 tool calls on IK failures, and once a
  protective stop on the mug. It submitted a plan that was REJECTED ($0.0066; artifacts kept in
  `out/aigen_desk_walnut_near015/`). That is a scene-setup error, not a perception or planning result, so the scene
  was rebuilt with the near edge at x = 0.28 m: the same camera and objects, shifted 13 cm. It was then run once.
  The table row is that run. Both runs are reported; nothing else was re-run.

## A bug found on the way (fixed on this branch, needs robotics review)

The first `aigen_desk_lamp_mug` run **crashed**. In one stress-test variant the arm was still pressing on the mug
during the settle phase after the plan. The resulting `ProtectiveStop` was raised outside the try block in
`executor.run_plan`, escaped the process pool, and aborted the whole rehearsal after the agent loop had finished
(about $0.007 spent). The agent's own `stress_test(n=20)` call had hit the same error earlier. Fix (commit
`98d20f7`): the settle phase now records a protective stop as that run's `execution_error`, so the run counts as
failed instead of crashing. Tests: 42 passed. `rehearsal/executor.py` is robotics-owned, so this needs their review.
The committed lamp run is the rerun on the fixed code.

## Cost

Perception (Muse listing + SAM 3.1) on 4 photos: ≈ $0.07. Rehearsals: kitchen $0.0052, lamp desk $0.0073 + $0.007
(crashed run), walnut $0.0038 + $0.0066 (first robot placement). The ledger went from $3.552 to $3.655 (≈ $0.10 for this whole task, against a $1.50 allowance).
