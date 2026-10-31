"""Run a high-level plan in a fresh simulation and return verifiable results."""
from __future__ import annotations

import numpy as np

from . import predicates
from .scene import build_model, object_height
from .sim import GAP, IKFailure, Sim


def transport(sim: Sim, obj: str, dest: str, relation: str, transit_height: float):
    """Suction-pick `obj` and put it in/on `dest`, travelling with the TCP at `transit_height`."""
    p = believed_pos(sim, obj)
    h_obj = object_height(sim.objs[obj])
    above = lambda xy, z: np.array([xy[0], xy[1], z])

    sim.move_to(above(p, transit_height), f"approach {obj}")
    sim.move_to(above(p, p[2] + h_obj / 2 + GAP), f"descend to {obj}")
    sim.grasp(obj)
    sim.move_to(above(p, transit_height), f"lift {obj}")

    q = believed_pos(sim, dest)
    sim.move_to(above(q, transit_height), f"carry {obj} to {dest}")
    if relation == "in":
        floor = q[2] - object_height(sim.objs[dest]) / 2 + 0.008
    else:
        floor = q[2] + object_height(sim.objs[dest]) / 2
    sim.move_to(above(q, floor + h_obj + GAP + 0.006), f"lower {obj} {relation} {dest}")
    sim.release()
    sim.wait(0.3)
    sim.move_to(above(q, transit_height), f"retreat from {dest}")
    sim.placed_by_robot.add(obj)


def believed_pos(sim: Sim, oid: str) -> np.ndarray:
    """Where perception thinks the object is. Objects the robot has placed itself are known exactly."""
    p = sim.obj_state(oid)["pos"].copy()
    if oid not in sim.placed_by_robot:
        p[:2] -= sim.perception_error.get(oid, 0.0)
    return p


def run_plan(scene: dict, plan: dict, goals: list[dict], render=False,
             friction_scale=1.0, mass_scale=1.0, perception_error=None, record=None) -> dict:
    """perception_error: (n_objects, 2) array; the real object sits that far from where the
    robot believes it is (the scene file is the robot's belief)."""
    model = build_model(scene, friction_scale, mass_scale, pos_noise=perception_error)
    sim = Sim(model, scene, render=render)
    if perception_error is not None:
        sim.perception_error = {o["id"]: np.asarray(e) for o, e in zip(scene["objects"], perception_error)}
    if record:
        sim.start_recording(record)
    error = None
    for i, a in enumerate(plan["actions"]):
        try:
            transport(sim, a["obj"], a["dest"], a["relation"], a["transit_height"])
        except IKFailure as e:
            error = {"action_index": i, "action": a, "error": str(e)}
            sim.release()
            break

    sim.wait(0.5)
    before = {oid: sim.obj_state(oid)["pos"].copy() for oid in sim.objs}
    sim.wait(1.0)
    sim.settle_motion = {oid: float(np.linalg.norm(sim.obj_state(oid)["pos"] - before[oid]))
                         for oid in sim.objs}

    results = predicates.evaluate(sim, goals)
    return {
        "success": error is None and all(r["ok"] for r in results),
        "goals": results,
        "execution_error": error,
        "events": [e.to_dict() for e in sim.events],
        "frames": sim.frames,
        "trajectory": sim.traj,
        "sim_time": round(sim.d.time, 2),
    }


def reachability_report(scene: dict) -> dict:
    sim = Sim(build_model(scene), scene)
    return {oid: sim.reachable(oid) for oid in sim.objs}
