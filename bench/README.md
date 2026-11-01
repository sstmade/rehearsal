# Rehearsal Bench v0

**Every home becomes a test track.** Each table someone rehearses in Rehearsal can become a shareable, scored
test case: the photo, its physics copy, the task, the goals as predicates the simulator checks, and every plan
that already failed on it, with the contact log that shows why. Any planner can then be scored on the same
tables under the same random noise. The unit is the **failure case**, not just the scene.

v0 has 10 cases (6 hand-designed, 1 real dataset photo, 3 AI-generated test photos), seeded from the outputs
Rehearsal had already produced. Building the bench cost **$0** in Meta API calls: every plan is cached. Results:
[LEADERBOARD.md](LEADERBOARD.md).

## What a case is

`bench/cases/<id>/`

| file | what it holds |
|---|---|
| `case.json` | the case card ([schema.json](schema.json), `schema_version` 0.1): instruction (EN, plus ZH where one exists), goals as predicates (`on`, `inside`, `stacked`, `upright`, `stable`, `undisturbed`) and where they come from (hand-verified `reference` or `planner_compiled`), per-object perception σ, the stress protocol, provenance (`hand_designed`, `real_dataset_photo`, `ai_generated_photo`, `real_photo`, `user_photo`), privacy, consent and licence, a summary of the recorded run, and the code version |
| `scene.json` | the physics copy in Rehearsal's scene format (CONTRACTS.md §2), exactly as rehearsed; `photo.file` points to `photo.jpg` |
| `photo.jpg` | the photo, at most 1024 px and 300 KB, EXIF stripped. For hand-designed scenes it is a MuJoCo render (`provenance.photo_is_render`) |
| `failures.jsonl` | one line per distinct failed plan: the plan, which predicates failed, the execution error (out of reach, protective stop), plain-sentence evidence, the contact log (what touched what, when, during which phase), the disturbed objects and how many recorded runs produced that plan. From `out/<scene>/report.json`, plus the failed first plans in Evaluation v3's cached runs (re-simulated locally to recover their contact logs; `reproduces` says whether they still fail on the current code) |
| `plans/` | cached plans: `muse_first.json`, `muse_after_log.json`, `scripted_baseline.json` and, where Evaluation v3 has them, `eval_round3_first.jsonl` / `eval_round3_after_log.jsonl` (20 repeats) |

A plan is our plan format: `{"actions": [{"obj", "dest", "relation": "in"|"on", "transit_height": m, "via": [[x, y], ...]}]}`
(robot frame, metres; suction pick from the top, carried at `transit_height`, optional waypoints).

## Add your table

1. Rehearse it in the app: upload a photo, fix the copy if needed, press **Rehearse** (or
   `.venv/bin/python -m rehearsal.rehearse scenes/<name>.json "<instruction>"`).
2. Export it, only if the photo's owner agrees to publish it as a public test case:
   ```
   .venv/bin/python scripts/bench_export.py photo_<job-id> --owner-consent
   ```
   Uploads are refused without `--owner-consent`. Uploads saved as private (Standard mode, the default) are
   refused even with it, unless `--allow-private` is also passed. Bundled public scenes need no flag
   (`scripts/bench_export.py --all` re-exports all ten). Nothing is uploaded anywhere: the case is a local folder,
   and the web page offers it as a download (`/api/bench/case?id=<id>`).
3. Check `case.json`: the licence (`--licence`, default CC BY 4.0 for contributed photos) and whether the photo
   shows people or private details. If it does, do not publish it.

## Score your planner (3 commands)

```
# 1. write your plan for a case (our plan format), e.g. from bench/cases/kitchen_01/scene.json + case.json
cat bench/cases/kitchen_01/case.json | python -m json.tool | head -40
# 2. score it: nominal run + 40 noisy runs on fixed seeds 1000-1039
.venv/bin/python bench/score.py bench/cases/kitchen_01 my_plan.json --json card.json
# 3. re-run the leaderboard (3 cached planners on every case; add yours to PLANNERS in bench/leaderboard.py)
.venv/bin/python bench/leaderboard.py
```

The score card gives **nominal** (does the plan pass every goal predicate on the scene as perceived; which
predicates it violated), **robustness k/n** (runs that pass with friction ×0.6–1.4, mass ×0.7–1.3 and each object's
position off by N(0, σ) per axis, σ from the case), the most common violations under noise, and **score** = k/n if the
nominal run passes, else 0. The seeds use common random numbers, so every planner faces identical noise. The noise
draw is the product's own stress test (`rehearse._variant`), so scoring the plan a report shipped reproduces that
report's number: `real_ycbv59`'s final plan scores 7/40, as in its report.

## Licences

- **Hand-designed scenes and renders**: made by the Rehearsal team; the project's licence is not chosen yet (see
  THIRD_PARTY_NOTICES.md). Renders use the Franka Panda model from MuJoCo Menagerie (Apache-2.0).
- **`real_ycbv59`**: a YCB-Video frame (BOP'19 test), MIT, © 2017 University of Washington RSE Lab
  (Xiang et al., PoseCNN, RSS 2018; BOP benchmark). Resized; scene geometry derived by the Rehearsal team.
- **`aigen_*`**: AI-generated test photos made by the project owner with a text-to-image model. They are **not real
  captures** and show no real desk, place or person (photos/ai_gen/SOURCES.md). Each case says so in
  `provenance.kind` and `privacy.attribution`.
- **Meshes**: cases carry no meshes; physics uses primitives. The look-alike GLBs in `assets/sam3d/` were made with
  Meta SAM 3D Objects and are under the **SAM License** (copy in `assets/sam3d/SAM_LICENSE.txt`). If you add
  them to a case, keep that licence and the credit.
- **Contributed photos**: whatever the contributor agreed to (`privacy.licence`), only with `owner_consent: true`.

## Limitations (honest)

- **Small n.** 10 cases, and each leaderboard cell scores one cached plan per planner. Only the five scenes from
  Evaluation v3 have 20 repeats. Read the cells as anecdotes and the pooled numbers as a first look.
- **Simulation only.** MuJoCo, a Franka Panda with a suction grasp that always holds. No real robot moved, and
  there is no sim-to-real study, so a high score does not mean a real robot would succeed.
- **Primitives for physics.** Objects are boxes, cylinders and bowls (some compounds); the SAM 3D meshes are
  visual only. Perception error is modelled as a Gaussian position error per object, not re-detected from pixels.
- **Goals.** Five cases have hand-verified reference goals. The others use the goals Muse Spark compiled in the
  recorded run, so they encode that run's reading of the instruction.
- **Hard by design.** The hand-designed scenes were built to make first plans fail, so failure rates describe these
  scenes, not everyday tables. No member of the public has contributed a case yet.
- **Code drift.** Cases record the code version (`code.code_hash`). Scores are only comparable on the same version;
  re-run `bench/leaderboard.py` after simulator changes.
