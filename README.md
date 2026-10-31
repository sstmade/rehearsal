# Rehearsal MVP

Feasibility check for the simulation core: scene → goals → plan → MuJoCo → predicate check → diagnose + revise → stress test → report.

## Run

```bash
uv venv .venv && uv pip install --python .venv/bin/python mujoco numpy pillow imageio openai
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
(cd third_party/mujoco_menagerie && git sparse-checkout set franka_emika_panda)

.venv/bin/python -m rehearsal.rehearse scenes/desk_01.json "把杯子放进碗里，然后把书摞整齐" --stress 40
```

Output: `out/<scene>/report.md`, `report.json`, and one GIF per attempt.

Web UI: `.venv/bin/python -m rehearsal.server`, then open http://localhost:8765.

### Active perception ("where to look again")

After the stress test, each object is scored by the value of observing it again: the stress test is
rerun with only that object's perception error reduced from σ=1.2 cm to σ=0.4 cm, on the same seeds
(paired comparison). The best object is "looked at" and the process repeats (`--look 2` rounds).
In the MVP the second look is simulated by the smaller σ; with real perception it is a prompt to
take another photo of that object. Cost: objects × `--stress` extra runs per round.

Without `MODEL_API_KEY`, the agent is `OfflineAgent`, a rule-based stand-in that only understands the demo instruction. With the key set, `MuseAgent` uses Muse Spark 1.3 for goal compilation, planning and diagnosis.

## Layout

| File | Role |
|---|---|
| `rehearsal/scene.py` | Scene JSON → MuJoCo model (MjSpec) + Panda + one suction weld per object |
| `rehearsal/sim.py` | IK (damped least squares), cartesian moves, suction grasp, contact event log, rendering |
| `rehearsal/executor.py` | `transport` action, perception error as belief-vs-reality offset, run a plan |
| `rehearsal/predicates.py` | `inside / on / stacked / upright / stable / undisturbed`, checked on sim ground truth |
| `rehearsal/agents.py` | `MuseAgent` (Muse Spark, structured output) and `OfflineAgent` |
| `rehearsal/rehearse.py` | Attempt loop (≤3), parallel stress test, report |
| `rehearsal/export3d.py` | Visual geometry (incl. Panda meshes + normals) → `scene3d.json` + `meshes.bin` for the browser |
| `rehearsal/server.py` | Local web UI and API (`/api/rehearse`, `/api/status`, `/api/report`) |
| `web/replay.js` | Three.js replay: recorded 50 Hz body poses, collision highlights, TCP path, stress-test trails |
| `rehearsal/perception.py` | Photo → scene JSON via Muse Spark + SAM 3.1 + paper-sheet homography (**untested**) |

## Assumptions

- Grasping is a suction weld that always succeeds once the tool is on the object's top face.
- Objects are primitives (box, cylinder, bowl). SAM 3D meshes are not wired in yet.
- `scenes/desk_01.json` is hand-written in the format perception produces.

Third party: Franka Panda model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0); [three.js](https://threejs.org) r160 vendored in `web/vendor/` (MIT).
