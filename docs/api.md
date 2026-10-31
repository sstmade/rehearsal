# Rehearsal HTTP API

Server: `python -m rehearsal.server` (stdlib only). Binds `127.0.0.1:8765` by default;
`REHEARSAL_HOST` / `REHEARSAL_PORT` override (container: `0.0.0.0:7860`).
All API responses are JSON (`Cache-Control: no-store`). Errors are always `{"error": "message"}` with
a 4xx/5xx status. Scene names and object ids match `^[A-Za-z0-9_-]{1,64}$`; anything else is a 400.

Long work runs as **jobs** (in memory, lost on restart). At most **1 rehearsal** runs at a time (others
queue); up to **2 perceptions** run concurrently. Poll `GET /api/job?id=` every ~500 ms.

## Scenes

### `GET /api/scenes`
```json
["desk_01", "photo_3f2a9c1b7d4e"]
```

### `GET /api/gallery`
```json
[{"scene": "desk_01", "title": "desk_01", "thumb": "/out/desk_01/thumb.png?v=1730419200",
  "has_report": true, "demo_instruction": "把杯子放进碗里…", "privacy": "private", "n_objects": 7}]
```
`thumb` is the first frame of `out/<scene>/attempt_1.gif` if a rehearsal exists, else the scene's
`photo.file`, else a generated top-down sketch (`/out/_thumbs/<scene>.png`, robot at the bottom).
`thumb` may be `null`. `title` = scene `"title"` field if present, else the name.
Extra keys beyond CONTRACTS (`demo_instruction`, `privacy`, `n_objects`) are additive.

### `GET /api/scene?name=desk_01`
The scene JSON (CONTRACTS §2). 404 if missing, 400 if the name is invalid.

### `POST /api/scene`
Body: full scene JSON (user-edited). `name` decides the file (`scenes/<name>.json`, overwrites).
```json
→ 200 {"name": "my_desk"}
→ 400 {"error": "invalid scene: object mug_1.pos [0.9, 0.9] is not on the table; duplicate object id book_1"}
```
Validation: `name`; `table.center`/`table.size` (0.1–5 m); 1–40 objects; unique non-reserved ids
(`table`, `hand`, `link0…7`, … are reserved); `shape.type` in box/cylinder/bowl with dimensions
0.005–1 m (radius ≤ 0.5); `pos` inside the table rectangle; `mass_kg.nominal` 0.001–50;
optional `friction.nominal` in (0, 3], `rgba` 4×[0,1], `privacy` public/private. Unknown extra keys are kept.

## Perception

### `POST /api/perceive` (multipart/form-data, ≤ 15 MB)
| field | required | value |
|---|---|---|
| `photo` | yes | JPEG / PNG / WebP file |
| `paper_corners` | no | JSON `[[u,v],[u,v],[u,v],[u,v]]` pixels: near-left, near-right, far-right, far-left |
| `privacy` | no | `"private"` (default; anything the user uploads) or `"public"` |

```js
const fd = new FormData();
fd.append("photo", file);
fd.append("paper_corners", JSON.stringify(corners));   // optional
const { job, upload } = await (await fetch("/api/perceive", { method: "POST", body: fd })).json();
```
```json
→ 200 {"job": "3f2a9c1b7d4e", "upload": "/uploads/3f2a9c1b7d4e.jpg"}
→ 400 {"error": "photo is not a readable image"}     413 if over 15 MB
```
The job result when done:
```json
{"scene": "photo_3f2a9c1b7d4e", "scene_spec": {…scene JSON…}, "overlay": "/uploads/3f2a9c1b7d4e_overlay.png",
 "valid": true, "issues": [], "saved": true}
```
If the perceived scene fails validation (`valid: false`, `issues: [...]`), it is **not saved**;
show `scene_spec` in the editor and `POST /api/scene` the corrected version (keep or change `name`).
If perception isn't available in this build, the job ends in `state: "error"` with a message naming
`perceive_photo`.

## Rehearsal

### `POST /api/rehearse`
```json
{"scene": "desk_01", "instruction": "把杯子放进碗里，然后把书摞整齐", "stress": 40, "look": 2}
```
`stress` 0–200 (default 40; 0 skips the stress test), `look` 0–5 active-perception rounds (default 2).
```json
→ 200 {"job": "a1b2c3d4e5f6", "started": true}
→ 400 / 404 {"error": "..."}
```
Jobs are queued, never rejected for being busy. The agent's data mode follows the scene's `privacy`.

### `GET /api/job?id=a1b2c3d4e5f6`
```json
{"id": "a1b2c3d4e5f6", "kind": "rehearse", "state": "running",
 "log": ["agent: offline-stand-in (data=private)", "reachable: {...}", "..."],
 "result": null, "error": null,
 "progress": {"phase": "stress", "fraction": 0.52},
 "params": {"scene": "desk_01", "instruction": "…", "stress": 40, "look": 2},
 "queue_ahead": 0, "created": 1730419200.1, "started": 1730419200.2, "finished": null}
```
`state`: `queued` → `running` → `done` | `error`. `progress.phase` for rehearse: `queued, starting,
setup, attempt, stress, look, report, done`; `fraction` is overall 0–1 (monotonic within a job).
`queue_ahead` = jobs of the same kind ahead of this one (only while queued). On error, `error` is
`"ExceptionType: message"` and the traceback is appended to `log`.

Rehearse result:
```json
{"scene": "desk_01", "report": "/api/report?scene=desk_01", "success": true, "attempts": 2,
 "agent": "offline-stand-in", "stress": {"successes": 37, "n": 40}, "wall_time_s": 61.3}
```
`GET /api/jobs` lists recent jobs (same fields minus `log`, newest first).

### `GET /api/report?scene=desk_01`
`out/<scene>/report.json` (CONTRACTS §2) or `null`. Files beside it are served under
`/out/<scene>/…` (`attempt_N.gif`, `attempt_N.traj.json`, `scene3d.json`, `meshes.bin`,
`stress_trails.json`, `stress_trails_after.json`, `report.md`, `thumb.png`). Add `?v=<timestamp>` to bust caches.

### `GET /api/status` (legacy)
Reflects the most recent rehearse job:
```json
{"running": true, "log": ["..."], "error": null, "job": "a1b2c3d4e5f6", "state": "running",
 "progress": {"phase": "attempt", "fraction": 0.17}}
```
`running` is true while that job is queued or running.

## Budget
### `GET /api/budget`
```json
{"spent": 1.2345, "reserved": 0.0, "cap": 15.0, "remaining": 13.7655}
```

## Static files
`/` → `web/index.html`; `/web/…`, `/out/…`, `/photos/…`, `/uploads/…`. Paths with `..`, hidden
files, or anything resolving outside those folders → 404.

## Report: uncertainty fields (round 2)

- Perception σ is per object: `obj.perception.sigma_m` if the scene has it, else 1.2 cm.
- `stress_test`: the headline is measured on **held-out seeds** `1000..1000+n-1` (`seed0: 1000`).
  `sigma_cm` is `{object_id: cm}` and `sigma_cm_mean` is a number.
- `active_perception`: `sigma_cm` is now also `{object_id: cm}` (the initial σ, same shape as
  `stress_test.sigma_cm`). `sigma_cm_mean` is a number, `default_sigma_cm` is 1.2, `refined_sigma_cm` is 0.4.
  The object to look at is chosen on **selection seeds** `0..n-1`. `steps[].before/after/value_by_object`
  use those seeds, so they are optimistic.
  - `selection`: `{"seed0": 0, "n", "before", "after"}`
  - `heldout`: `{"seed0": 1000, "n", "before", "after"}`. **Show this one as the gain.** `before` equals `stress_test.successes`.
  - `final`: the held-out after-look stress test, with `runs` (`final.successes == heldout.after`).
    `selection_final` holds the selection-seed version.
  - `stress_trails.json` and `stress_trails_after.json` are both on held-out seeds, so they pair by seed.
