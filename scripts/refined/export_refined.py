"""MuJoCo run -> per-frame poses for the Blender re-render (generalised scripts/blender06/export_traj.py).

Non-invasive like export_traj: exec_tables.render_clip is run on a cached run record (no planning, no physics
decisions: the recorded qpos is replayed, the cloth is stepped with the rigid bodies pinned, exactly as the shipped
clips do it), with OUT pointed at a scratch dir and two hooks (Shooter.render, caption) that snapshot every frame.

  .venv/bin/python scripts/refined/export_refined.py KEY CACHE_DIR OUT.npz
CACHE_DIR holds KEY_run.json, KEY_traj.npz, KEY_scene.json (run_refined.py / exec_tables.py layout).
Unlike export_traj, a caught failure's frozen tail is recorded too (frames after the freeze repeat the freeze state).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "blender06"))
import export_traj as E  # noqa: E402  (patches exec_tables.Shooter / caption)

os.environ.pop("EXEC_HQ", None)                    # 1x MuJoCo frames are enough: only the poses are used
X, S = E.X, E.S
_orig = E._orig_caption


def rec_caption(img, text, sub=None, alpha=1.0, color=(255, 255, 255), note=None):
    i = len(S["frames"])
    if S["pending"] is not None:
        S["frames"].append(S["pending"])
        S["pending"] = None
        S["task_seen"] = True
    elif S.get("task_seen"):                       # caught failure: frozen frame (no new render)
        s = dict(S["frames"][-1])
        s["frozen"] = True
        S["frames"].append(s)
    else:
        s = dict(S["init"])
        s["arm"] = {False}
        s["photo"] = True
        S["frames"].append(s)
    if i % 60 == 0:
        S["refs"][i] = np.asarray(img).copy()
    S["frames"][-1] = dict(S["frames"][-1], text=(text, sub, float(alpha), note))
    return _orig(img, text, sub, alpha, color, note)


X.caption = rec_caption


def main():
    key, cache, outp = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    outp.parent.mkdir(parents=True, exist_ok=True)
    X.CACHE = cache
    X.OUT = outp.parent / "_mj_clip"
    X.OUT.mkdir(parents=True, exist_ok=True)
    rec = json.loads((cache / f"{key}_run.json").read_text())
    assert rec.get("executed"), "nothing executed: no clip to export"
    rend = X.render_clip(key, rec)
    F = S["frames"]
    assert len(F) == rend["frames"], (len(F), rend["frames"])
    arm = np.array([[False in f["arm"], True in f["arm"]] for f in F])
    np.savez_compressed(
        outp, fps=X.FPS, W=X.W, H=X.H,
        xpos=np.stack([f["xpos"] for f in F]).astype(np.float32),
        xquat=np.stack([f["xquat"] for f in F]).astype(np.float32),
        gpos=np.stack([f["gpos"] for f in F]).astype(np.float32),
        gmat=np.stack([f["gmat"] for f in F]).astype(np.float32),
        flex=np.stack([f["flex"] for f in F]).astype(np.float32),
        cam=np.stack([f["cam"] for f in F]),
        arm_off=arm[:, 0], arm_on=arm[:, 1],
        photo=np.array([bool(f.get("photo")) for f in F]),
        frozen=np.array([bool(f.get("frozen")) for f in F]),
        text=np.array([json.dumps(f["text"]) for f in F]))
    mdl = E.export_model(S["m"], Path(str(outp).replace(".npz", "_tex")))
    np.savez_compressed(str(outp).replace(".npz", "_model.npz"), **mdl)
    rdir = outp.parent / "mj_ref"
    rdir.mkdir(exist_ok=True)
    from PIL import Image
    for i, img in S["refs"].items():
        Image.fromarray(img).save(rdir / f"mj_{i:04d}.png")
    meta = {"key": key, "frames": len(F), "fps": X.FPS, "render": rend, "hero_cam": rec.get("hero_cam"),
            "photo": str(ROOT / S["sc"]["photo"]["file"]), "scene_name": S["sc"]["name"],
            "catch": rec.get("catch"), "success": rec.get("success"), "planned": rec.get("planned"),
            "events": rec.get("events"), "sim_time_s": rec.get("sim_time_s"), "dt": rec.get("dt"),
            "actions": rec["plan"]["actions"], "instruction": rec.get("instruction"),
            "run_json": str(cache / f"{key}_run.json"), "n_objects": len(S["sc"]["objects"]),
            "nbody": int(S["m"].nbody), "ngeom": int(S["m"].ngeom), "nflexvert": int(S["m"].nflexvert)}
    Path(str(outp).replace(".npz", "_meta.json")).write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps({k: meta[k] for k in ("frames", "render", "catch")}, default=float))


if __name__ == "__main__":
    main()
