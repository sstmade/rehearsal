"""Run a high-level plan in a fresh simulation and return verifiable results."""
from __future__ import annotations

import numpy as np

from . import predicates
from .scene import build_model, object_height
from .sim import GAP, IKFailure, Sim


def transport(sim: Sim, obj: str, dest: str, relation: str, transit_height: float, via=None):
    """Suction-pick `obj` and put it in/on `dest`, travelling with the TCP at `transit_height`.

    via: optional list of [x, y] waypoints the carry passes through (at transit_height) between
    lifting `obj` and arriving above `dest`; after placing, the empty arm returns through them in
    reverse before the next action's approach. Empty / None = straight lines.
    """
    for name in (obj, dest):
        if name not in sim.objs:
            raise IKFailure(f"unknown object id {name!r}", {"phase": "plan check"})
    p = believed_pos(sim, obj)
    h_obj = object_height(sim.objs[obj])
    above = lambda xy, z: np.array([float(xy[0]), float(xy[1]), z])

    sim.move_to(above(p, transit_height), f"approach {obj}")
    sim.just_released = None
    sim.expect_contact = obj
    sim.move_to(above(p, p[2] + h_obj / 2 + GAP), f"descend to {obj}")
    sim.grasp(obj)
    sim.expect_contact = None
    sim.move_to(above(p, transit_height), f"lift {obj}")

    for k, v in enumerate(via or []):
        sim.move_to(above(v, transit_height), f"carry {obj} via waypoint {k} {list(v)}")
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
    # Go back out along the same detour, so the empty arm does not cut across what the detour avoided.
    for k, v in reversed(list(enumerate(via or []))):
        sim.move_to(above(v, transit_height), f"return via waypoint {k} {list(v)}")


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
            transport(sim, a["obj"], a["dest"], a["relation"], a["transit_height"], a.get("via"))
        except IKFailure as e:
            error = {"action_index": i, "action": a, "error": f"action {i}: {e}", **e.info}
            sim.expect_contact = None
            sim.release()
            break

    sim.wait(0.5)
    before = {oid: sim.obj_state(oid)["pos"].copy() for oid in sim.objs}
    sim.wait(1.0)
    sim.settle_motion = {oid: float(np.linalg.norm(sim.obj_state(oid)["pos"] - before[oid]))
                         for oid in sim.objs}

    results = predicates.evaluate(sim, goals)
    events = [e.to_dict() for e in sim.events]
    disturbed = disturbances(sim, events)
    return {
        "success": error is None and all(r["ok"] for r in results),
        "goals": results,
        "execution_error": error,
        "disturbed": disturbed,
        "arm_contacts": [{"obj": o, "link": l, "peak_force_n": round(f, 1), "t": t, "phase": ph}
                         for (o, l), (f, t, ph) in sim.robot_contact_peak.items() if f > 2.0],
        "evidence": evidence(error, disturbed, events),
        "events": events,
        "frames": sim.frames,
        "trajectory": sim.traj,
        "sim_time": round(sim.d.time, 2),
    }


def _tilt_deg(sim: Sim, oid: str) -> float:
    c = float(np.clip(sim.obj_state(oid)["zaxis"][2], -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def disturbances(sim: Sim, events: list[dict]) -> list[dict]:
    """Objects that ended up somewhere the plan did not put them, with the first contact that
    touched them. Robot-placed objects only count if they tipped over (> 20 deg)."""
    out = []
    for oid in sim.objs:
        pos = sim.obj_state(oid)["pos"]
        moved = float(np.linalg.norm(pos[:2] - sim.start_pos[oid][:2]))
        tilt = _tilt_deg(sim, oid)
        placed = oid in sim.placed_by_robot
        if placed and tilt < 20:
            continue
        if not placed and moved < 0.01 and tilt < 10:
            continue
        cause = next((e for e in events if e["kind"] == "contact" and oid in e["between"]
                      and not {"table", "floor"} & set(e["between"])
                      and e.get("while_holding") != oid                        # its own carry
                      and not (str(e.get("phase", "")).startswith("lower")      # intended placement contact
                               and str(e.get("phase", "")).endswith(" " + oid))
                      and (e.get("link") or e.get("while_holding") in e["between"])), None)
        d = {"obj": oid, "moved_cm": round(moved * 100, 1), "tilt_deg": round(tilt, 0),
             "final_xyz": [round(float(v), 3) for v in pos], "placed_by_robot": placed}
        if pos[2] < -0.05:
            d["fell_off_table"] = True
        if oid in sim.grasped and not placed:
            d["released_mid_air_after_error"] = True
        if cause:
            other = next(x for x in cause["between"] if x != oid)
            d["first_hit"] = {"t": cause["t"], "by": other, "link": cause.get("link"),
                              "link_role": cause.get("link_role"), "phase": cause.get("phase"),
                              "tcp_z": cause.get("tcp_z"), "at_xyz": cause.get("at_xyz")}
        out.append(d)
    return out


def evidence(error, disturbed, events) -> list[str]:
    """Plain-language facts from the simulation, for a language model to diagnose from."""
    lines = []
    if error:
        lines.append(error["error"])
    for e in events:
        if e["kind"] == "contact" and e.get("link") and not {"table", "floor"} & set(e["between"]):
            other = next(x for x in e["between"] if x != "robot_arm")
            lines.append(f"t={e['t']}s the robot's {e['link_role']} ({e['link']}) touched {other} "
                         f"at {e['at_xyz']} during '{e['phase']}' (tool tip z={e['tcp_z']} m"
                         + (f", holding {e['while_holding']})" if e["while_holding"] else ")"))
        elif (e["kind"] == "contact" and e.get("while_holding") in e["between"]
              and not {"table", "floor"} & set(e["between"])):
            other = next(x for x in e["between"] if x != e["while_holding"])
            if str(e.get("phase", "")).startswith("lower") and e["phase"].endswith(" " + other):
                continue  # the held object touching its own destination is the intended placement
            lines.append(f"t={e['t']}s the held {e['while_holding']} touched {other} at {e['at_xyz']} "
                         f"during '{e['phase']}' (tool tip z={e['tcp_z']} m)")
    for d in disturbed:
        s = f"{d['obj']} ended {d['moved_cm']} cm from its start, tilted {d['tilt_deg']:.0f} deg"
        if d.get("fell_off_table"):
            s += ", fell off the table"
        if d.get("released_mid_air_after_error"):
            s += " (it was being carried when the plan aborted and was dropped)"
        if d.get("first_hit"):
            h = d["first_hit"]
            who = f"robot {h['link_role']} ({h['link']})" if h["link"] else h["by"]
            s += f"; first hit by {who} at t={h['t']}s during '{h['phase']}'"
        lines.append(s)
    return lines


def reachability_report(scene: dict) -> dict:
    sim = Sim(build_model(scene), scene)
    return {oid: sim.reachable(oid) for oid in sim.objs}
