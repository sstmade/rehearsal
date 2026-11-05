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

## Look-alike objects (Meta SAM 3D Objects, visual only) and the robot stand

The three scenes now carry SAM 3D meshes the same way `real_ycbv59` does (`docs/real_photo_demo.md`, "Look-alike
objects"): SAM 3.1 mask → SAM 3D Objects splat (community MLX port `mlx-spatial`, 8-bit weights, on device, no API
cost) → `experiments/sam3d/splat2mesh.py` → 18k faces → orientation fitted against the photo through the calibrated
camera → `assets/sam3d/aigen_<name>/<obj>.glb`. Physics still uses the primitives; the viewer's **Real look /
Simple shapes** toggle switches between them. The SAM License note in `THIRD_PARTY_NOTICES.md` covers these assets.

New in `scripts/sam3d_assets.py`: an upright cylinder proxy has no yaw of its own, so the best axis-aligned
orientations are refined with free yaw in 15° steps. That turned the kettle's spout and handle and the black lamp's
arm the way the photo shows them (kettle IoU 0.82 → 0.86, lamp 0.50 → 0.55).

| scene | object | SAM 3D time | peak memory | Gaussians | photo IoU of the oriented mesh | shape mismatch | orientation |
|---|---|---|---|---|---|---|---|
| `aigen_kitchen` | apple | 333 s | 10.4 GB | 700k | 0.74 | 0.07 | yaw 150° |
| `aigen_kitchen` | cutting board | 276 s | 10.4 GB | 444k | 0.83 | 0.28 | axis-aligned |
| `aigen_kitchen` | electric kettle | 348 s | 10.4 GB | 725k | 0.86 | 0.09 | yaw 300° |
| `aigen_kitchen` | mug | 306 s | 10.4 GB | 548k | 0.72 | 0.16 | axis-aligned |
| `aigen_kitchen` | trivet | 242 s | 10.4 GB | 161k | 0.82 | 0.20 | yaw 240° |
| `aigen_desk_walnut` | book | 252 s | 15.9 GB | 450k | 0.95 | 0.10 | axis-aligned |
| `aigen_desk_walnut` | desk lamp | 261 s | 15.8 GB | 309k | 0.59 | 0.11 | axis-aligned |
| `aigen_desk_walnut` | mug | 399 s | 15.9 GB | 932k | 0.76 | 0.17 | axis-aligned |
| `aigen_desk_walnut` | potted plant | 417 s | 15.9 GB | 893k | 0.71 | 0.03 | yaw 330° |
| `aigen_desk_walnut` | tablet | 310 s | 15.9 GB | 410k | 0.84 | 0.21 | axis-aligned |
| `aigen_desk_lamp_mug` | book | 258 s | 15.9 GB | 431k | 0.86 | 0.17 | axis-aligned |
| `aigen_desk_lamp_mug` | black desk lamp | 254 s | 15.9 GB | 258k | 0.55 | 0.26 | yaw 345° |
| `aigen_desk_lamp_mug` | laptop | 322 s | 15.9 GB | 456k | 0.81 | 0.32 | axis-aligned |
| `aigen_desk_lamp_mug` | mug | 353 s | 15.9 GB | 741k | 0.71 | 0.13 | axis-aligned |
| `aigen_desk_lamp_mug` | potted plant | 418 s | 15.9 GB | 966k | 0.82 | 0.11 | yaw 165° |

"Shape mismatch" is the mean |log| ratio between the scaled mesh's extents and the fitted primitive's. About 80 min
of machine time for 15 objects, one at a time. On the 1448 × 1086 desk photos the peak is 15.9 GB, not the 10–11 GB
seen on the smaller frames. Each GLB is 361 KB, 1.8 MB per scene. **Kept as primitives** (a mesh adds nothing at
this size or they are flat): the coasters, the phones and the scissors.

Compare images (photo | physics shapes | meshes, all through the photo's camera, rendered by the web viewer itself
with `scripts/sam3d_compare.py`): `photos/ai_gen/aigen_kitchen.sam3d_compare.png`,
`photos/ai_gen/aigen_desk_walnut.sam3d_compare.png`, `photos/ai_gen/aigen_desk_lamp_mug.sam3d_compare.png`.
Viewer screenshots: `docs/aigen_sam3d_ui.jpg`.

**Limits.** The mesh is fitted to the twin pose, so where the twin is off, the mesh is too. Example: the walnut desk
lamp sits a few centimetres right of the photo. The thin lamps score lowest (IoU 0.55–0.59): a pole and a shade are
hard to cover with one uniform scale. The lamp_mug mug's handle points away from the camera. Its orientation comes
from the 24 axis-aligned candidates, because the mug proxy is a compound with its own handle direction. The
cutting board leans on the wall in the photo but stands upright in the twin.

**Robot stand.** These small tables put the robot base off the near table edge, and before this change nothing
was under it. `rehearsal/scene.py` `support_boxes()` now adds a stand from the floor to the base whenever the table
top does not cover the base footprint (here and on `real_ycbv59`). It also adds four visual-only table legs on every
scene. These are static geoms on the world body, so no body indices change. Replaying every recorded attempt of
the four affected scenes gives bit-identical trajectories, and the GIFs were re-rendered from them. The web viewer
mirrors this geometry. Check: `docs/robot_stand_check.jpg`, all 10 scenes.

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

## 50-table batch: photo → twin → task (`scripts/batch_tables.py`, `scripts/render_tables.py`)

50 AI-generated test photos (not real captures; `photos/ai_gen/tables_v6/`), each with a different everyday task
(`TASKS` in `batch_tables.py`). Each becomes `scenes/t6_NN.json` (34, the empty table, is kept in `out/tables_v6/`:
an empty scene is not a valid Rehearsal scene) and its compiled intent `out/tables_v6/NN.json`.

**Perception and geometry.**
- Muse Spark lists the objects and SAM 3.1 masks them (cached).
- The tabletop outline is hand-read on a 5% grid (`TOPS`).
- Camera:
  - Roll is fixed at 0.
  - Pitch and focal length are fitted from right angles at the corners that are really in view, plus Muse Spark's
    estimate of the tabletop's width/depth and camera pitch (`table_dims`, ~$0.04 for all 50).
  - Right angles alone drove the pitch to its bound on 18 of 50 photos: AI-generated images often have too little
    perspective convergence.
- Scale: the camera height is set so the median fitted object size matches Muse Spark's typical sizes.
- Flat and soft objects (paper, books, phones, towels) take their footprint from the SAM mask projected onto the
  table. Cables and leashes become thin strips.
- An object is kept, and nudged onto the top, if its base lies inside the tabletop outline in the photo. Before this
  fix the lamp in 05 was dropped, and Muse Spark then correctly said "there is no lamp".
- Robot stand-off is chosen per scene so the nearest object is at least 0.30 m from the base. The top-down tool
  cannot reach closer, and the first execution pass failed on 6 scenes for that reason.

**Rendering (visual only; physics and planning use the shipped primitives).**
- Tabletop texture: the photo rectified onto the table plane. Object masks are dilated by 41 px and filled from a
  clean patch of the same surface.
- Photo skin (`scripts/photo_skin.py`):
  - Every box face, cylinder top and visible cylinder side that the photo's camera saw gets a massless,
    non-colliding mesh textured with the photo pixels at its 3D points.
  - Faces seen nearly edge-on (cos < 0.3) are skipped; they would smear.
- **Disclosure.** Through the photo's own camera the twin looks almost exactly like the photo *because those pixels
  were projected from that camera*. Geometry is judged from other viewpoints.
- Cloth: towels, clothes, socks and knitwear are drawn as a MuJoCo flex sheet (≤16×16 vertices). It falls onto a
  hidden rigid core the size of the object and drapes over it. In the scene JSON and in planning, cloth stays a thin
  rigid pad.
- Round and oval tables are drawn as elliptical slabs; their collision box is unchanged.
- Hero tables: Meta SAM 3D Objects meshes replace the photo skin on containers and task objects (next section).

**Intent** (Muse Spark compile step plus deterministic gate, shipped prompt): 41 goals / 5 asks / 4 refusals on the
final scenes. Earlier passes had up to 11 refusals. Several came from our pipeline, not the model: dropped objects,
and objects placed 0.18–0.27 m from the base. Those were fixed in the geometry, not by changing the prompt.

**Rehearsal on the 40 goal scenes** (`scripts/exec_tables.py`, search planner plus MuJoCo; no Meta API calls; log in
`footage/tables_sim/hero/exec/exec_log.json`). Table 34 is empty and its goal is "nothing on the table", so it is not
counted here.

- **14:** search found a plan and the physics replay held every goal.
  - In 9 of them the robot really moves (06, 29, 31, 35, 41, 43, 44, 45, 50).
  - In 5 the physics check says the goals already hold, so the plan is "no motion" (02, 22, 25, 26, 30). We count
    only 4 as passes: in 30 the photo's fallen bottle was rebuilt upright, so it is a reconstruction miss.
- **3 (05, 14, 32):** search returned a plan with no motion, but the goals do not hold. These are counted as
  failures of ours, together with 30: **4 missed** in all.

**Canonical tally:** 9 done · 4 already done · 23 no safe plan, not sent · 4 missed.
- **23:** search found no verified plan, so Rehearsal would not send anything. To show *why*, the script then
  replays a naive direct attempt.
  - **22 of these fail in simulation:** 10 collisions, 4 suction misses, 2 out of reach, and 6 with objects ending
    in the wrong place.
  - Scene 20's replay ends with objects below the table. That is a simulation artifact, not a robot mistake, and it
    is not shown anywhere.
  - One (38) succeeds by luck; it is still not counted as verified.

**Honest reading:** on these AI-generated tables the planner finds a verified plan for 14 of 40 tasks. The rest are
correctly *not sent*, rather than solved. The usual causes are a top-down suction tool next to tall rims (bowls,
buckets) and objects near the reach limit. The catch clips in the video are these direct attempts, and they are
labelled "No safe plan found. Here is what a direct attempt would do". We do not present them as plans that failed
in rehearsal.

## 50-table batch: SAM 3D hero tables

Meta SAM 3D Objects meshes for the hero tables of the 50-table batch (`scenes/t6_NN.json`, photos in
`photos/ai_gen/tables_v6/`). This is **visual only: physics, planning and the shipped scene JSON keep the
primitives.** In the MuJoCo renders the primitives stay in the model (they collide and carry mass) but are drawn
with alpha 0, and the mesh is a non-colliding, massless geom on the same body.

Pipeline: `scripts/sam3d_tables.py` (hero object list per scene) drives `scripts/sam3d_assets.py` one object at a
time, on device (MLX, 8-bit weights, no API cost) → `assets/sam3d/t6_NN/<obj>.glb` + `.json`. Splats are cached in
`.scratch/sam3d/t6_NN/outputs/<obj>/gaussians.ply`. The splat depends only on the photo and the SAM 3.1 mask; the
orientation step depends on the calibration and the twin poses. Re-run it after a scene rebuild, in ~10–40 s per
scene with no reconstruction:

    SAM3D_WEIGHTS=weights/sam-3d-objects-mlx-8bit sam3d_env/bin/python scripts/sam3d_tables.py --realign

New in `sam3d_assets.py` tonight:
- **Open containers.** SAM 3D closed the pet basket with a lid that carried the floor's colours (all interior
  Gaussians sat in the top 10 % of the height). For basket/bowl/bucket/tray/holder labels, `carve_lid` now moves
  the top-band Gaussians over the wall-enclosed interior down to the floor and re-meshes (`--no-carve` turns this
  off). It is a no-op when the walls enclose nothing or a floor already exists.
- **Mug yaw.** Mug/cup/jug compounds also get the free-yaw refinement.
- **Proportions check.** A steep penalty applies when the mesh's proportions disagree with the primitive's
  (`2·max(0, shape_err − 0.25)`). Without it, the laundry basket was stood on its end to fill a mask that the twin
  box does not cover.
- **Colour gain.** A per-channel `rgb_gain` (photo / reprojected mesh inside the mask) is stored for renderers.

MuJoCo (`scripts/sam3d_mj.py`, `add_sam3d_visuals(spec, scene, hide_primitive=True)`): MuJoCo has no vertex
colours, so each mesh's colours are baked into a per-face texture atlas. Each face gets a 2×2-texel cell, and the
faces are ordered by Morton code so that mipmaps blend only neighbouring faces. The result is one geom and one
texture per object, with smooth normals. Placement is `rehearsal/export3d.visual_meshes` itself, so the web viewer
and MuJoCo agree. Colour is × `rgb_gain` (clipped 0.7–1.4) with material emission 0.3, because the splat colours
already contain the photo's shading. Compare images: `footage/tables_sim/hero/sam3d/NN_compare.png`. Top row:
photo | primitives | SAM 3D (3/4 view). Bottom row: overlay | primitives | SAM 3D (photo camera).

| scene | object | SAM 3D time | peak memory | photo IoU (oriented mesh) | shape mismatch | lid carved |
|---|---|---|---|---|---|---|
| `t6_43` | basket | 363 s | 15.9 GB | 0.82 | 0.03 | yes (158k Gaussians) |
| `t6_43` | pet brush (slicker) | 335 s | 15.9 GB | 0.51 | 0.10 | – |
| `t6_43` | pet brush (paddle) | 313 s | 16.2 GB | 0.78 | 0.04 | – |
| `t6_03` | laundry basket | 416 s | 16.0 GB | 0.48 | 0.15 | – (splat already open) |
| `t6_01` | mug (cream) | 440 s | 15.9 GB | 0.73 | 0.16 | – |
| `t6_01` | mug (green) | 479 s | 15.8 GB | 0.78 | 0.10 | – |
| `t6_01` | pen holder | 404 s | 15.9 GB | 0.72 | 0.05 | – |
| `t6_06` | spice tin | 470 s | 15.9 GB | 0.65 | 0.05 | – |
| `t6_06` | mixing bowl | 338 s | 15.9 GB | 0.82 | 0.06 | – |

Still running in the background when this was written (same driver, priority order): the 06 onions, then 09, 46,
28 and 27. Their sidecar JSONs hold the same fields. IoU here is measured after the realign against the scenes as
rebuilt at 00:20–00:31.

**Looks great:** the two pet brushes (bristles, red and blue handles; the primitives are slabs), the open woven pet
basket, the green mug with its rust rim, and the laundry basket's shape and grey handles.

**Weak:**
- The cream mug's handle faces the wrong side. With the handle on the right, IoU drops 0.73 → 0.58, which points
  to a twin-pose offset; re-check after the realign.
- The laundry basket is visibly smaller than and offset from the photo's (IoU 0.48). The twin primitive has the
  same error. The mesh follows the twin, so where the twin is off, the mesh is too.
- The pen holder's pens are lost in a dark blob.
- The basket floor reads a little too white under gain + emission.

**Cost:** ~5.5–8 min and ~16 GB peak per object, one at a time, on a 38 GB Mac with other renders running.

**Set dressing (3 Nov):** the renders (exec hero clips and the 50 poster/card stills) add room walls, floors, a
reshaped table top with legs, a clean procedural top texture (marble / wood / laminate / steel instead of the
inpainted photo texture) and simple shaped stand-ins for thin tools, bottles, bags and yarn. This is visual only,
added for realism: nothing added collides or has mass, and the physics, plans and outcomes are unchanged
(`scripts/hero_dress.py`, `scripts/dress_tex.py`; `HERO_DRESS=0` turns it off).
