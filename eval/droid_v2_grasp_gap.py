"""DROID v2 diagnosis: how far is the twin's object from the real gripper when the real gripper closes?  $0, offline.

    .venv/bin/python eval/droid_v2_grasp_gap.py      # -> eval/results/droid_v2/grasp_gap.json

Per episode, at the FIRST close command (first step with gripper command > 0.5, eval/droid_v2.close_segments):
  gap_nearest_cm  = xy distance from the recorded EE (Panda flange; the gripper points down, so its xy is the
                    fingertip xy) to the NEAREST object of the first-frame twin (scenes/droid/<slug>.json)
  gap_target_cm   = same, to the object the instruction's first goal moves (scene droid.goals[0].args[0]), if any
  first_sim_event = the replay's first grasp-related event (eval/results/droid_v2/<slug>.json): "closed_on_air"
                    or "grasp" (with the object it welded)
Inputs: datasets/droid/<slug>/trajectory.json, scenes/droid/<slug>.json, eval/results/droid_v2/<slug>.json.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("droid_v2", ROOT / "eval" / "droid_v2.py")
dv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dv)

HALF_STROKE_CM = 4.25          # Robotiq 2F-85: 85 mm stroke -> an object more than ~4.25 cm off-centre is missed


def episode_gap(traj: dict, scene: dict, events: list[dict]) -> dict:
    segs = dv.close_segments(traj["gripper_command"])
    a = segs[0][0]
    ee = np.asarray(traj["cartesian_position"][a][:2], float)
    d = {o["id"]: float(np.linalg.norm(np.asarray(o["pos"][:2], float) - ee)) * 100 for o in scene["objects"]}
    nearest = min(d, key=d.get)
    goals = (scene.get("droid") or {}).get("goals") or []
    target = goals[0]["args"][0] if goals else None
    ev = next((e for e in events if e["kind"] in ("grasp", "closed_on_air")), None)
    return {"first_close_step": a, "ee_xy": ee.round(4).tolist(), "nearest_object": nearest,
            "gap_nearest_cm": round(d[nearest], 1), "target_object": target,
            "gap_target_cm": None if target is None else round(d[target], 1),
            "first_sim_event": None if ev is None else ev["kind"],
            "first_sim_event_obj": None if ev is None else ev.get("obj")}


def summarize(rows: dict[str, dict]) -> dict:
    g = [r["gap_nearest_cm"] for r in rows.values()]
    gt = [r["gap_target_cm"] for r in rows.values() if r["gap_target_cm"] is not None]
    return {"n_episodes": len(rows),
            "gap_nearest_cm": {"median": round(float(np.median(g)), 1), "min": min(g), "max": max(g),
                               "n_above_half_stroke": sum(x > HALF_STROKE_CM for x in g)},
            "gap_target_cm": {"n": len(gt), "median": round(float(np.median(gt)), 1), "min": min(gt), "max": max(gt)},
            "first_sim_event_closed_on_air": sum(r["first_sim_event"] == "closed_on_air" for r in rows.values()),
            "first_sim_event_grasp": [f"{k}: {r['first_sim_event_obj']}" for k, r in rows.items()
                                      if r["first_sim_event"] == "grasp"],
            "half_stroke_cm": HALF_STROKE_CM}


def main() -> dict:
    rows = {}
    for slug in dv.v2_slugs():
        traj = json.loads((ROOT / "datasets" / "droid" / slug / "trajectory.json").read_text())
        scene = json.loads((ROOT / "scenes" / "droid" / f"{slug}.json").read_text())
        events = json.loads((dv.RES / f"{slug}.json").read_text())["events"]
        rows[slug] = episode_gap(traj, scene, events)
    out = {"method": __doc__.strip().splitlines()[0], "summary": summarize(rows), "episodes": rows}
    (dv.RES / "grasp_gap.json").write_text(json.dumps(out, indent=1))
    for k, r in rows.items():
        print(f"{k:22s} nearest {r['nearest_object']:15s} {r['gap_nearest_cm']:5.1f} cm  target {r['target_object']} "
              f"{r['gap_target_cm']}  sim: {r['first_sim_event']} {r['first_sim_event_obj'] or ''}")
    print(json.dumps(out["summary"], indent=1))
    return out


if __name__ == "__main__":
    main()
