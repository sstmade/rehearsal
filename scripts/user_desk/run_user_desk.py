"""Rehearsal on the user's own desk (PRIVATE scene `user_desk`; not part of the 50-table set).

    intent  : Muse Spark compile (rehearsal.agents.MuseAgent, data="private") + intent_gate (with physics)
    plan    : on/inside goals -> the shipped rehearsal.search_planner.search (unchanged);
              near goals     -> the shipped planner has no action for them (docs/intent.md: "`near` is verifiable but
                                not yet executable").  This wrapper adds a geometric spot search (place the object on the
                                table at a free spot next to the reference) and verifies every candidate with the stock
                                executor + stock predicates.  It lives only here; rehearsal/ and the 50 tables are untouched.
    verify  : rehearsal.executor.run_plan (physics), goals + implicit undisturbed(everything else)
    render  : scripts/exec_tables.render_clip (photo -> twin -> 3/4 hero, HQ), pointed at footage/user_desk_sim/

Run:  cd rehearsal-mvp; set -a; source .env; set +a
      .venv/bin/python scripts/user_desk/run_user_desk.py intent [KEY ...]
      .venv/bin/python scripts/user_desk/run_user_desk.py plan [KEY ...]
      .venv/bin/python scripts/user_desk/run_user_desk.py render [--hq] [--export] [KEY ...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
V2 = os.environ.get("USER_DESK_V2") == "1"          # v2: the mesh twin (scripts/user_desk/twin_v2.py)
ROTATE = os.environ.get("USER_DESK_ROTATE") == "1"  # demo extension: the near-placement may turn the object (yaw)
OUT = ROOT.parent / "footage" / "user_desk_sim" / ("v2" if V2 else "")
CACHE = OUT / "_cache"
INTENT = OUT / "intent"
SCENE = OUT / "twin" / ("user_desk_v2_scene.json" if V2 else "user_desk_scene.json")
if V2:
    sys.path.insert(0, str(ROOT / "scripts" / "user_desk"))
    import twin_v2
    twin_v2.install()
REC_EVERY = 2

# key -> instruction (as given, object names adjusted to the perceived labels where the user's word differs)
TASKS = {
    "t1_pen_mouse": "Put the pen next to the mouse.",
    "t2_powerbank_wallet_asgiven": "Move the power bank onto the card wallet.",
    "t2_powerbank_pouch": "Move the power bank onto the blue pouch.",
    "t3_remote_keyboard": "Put the remote next to the keyboard.",
    "t4a_tidy": "Tidy up my desk.",
    "t5_pour": "Pour the water bottle onto the keyboard.",
    "t6_bottle_monitor": "Move the water bottle next to the monitor.",
}
TIDY_ANSWER = ("Line up the small items neatly in a row along the front edge of the mat: the pen, the remote, the power "
               "bank, the blue pouch and the white round case, all aligned and parallel.")


def scene():
    return json.loads(SCENE.read_text())


def instruction_of(key):
    if key in TASKS:
        return TASKS[key]
    p = INTENT / f"{key}.json"
    return json.loads(p.read_text())["instruction"]


# ====================================================================================== intent
def compile_one(key, instruction, sc):
    from rehearsal import intent_gate
    from rehearsal import meta_client as M
    from rehearsal.agents import MuseAgent, scene_summary
    from rehearsal.executor import reachability_report
    before = M.status()["spent"]
    summary = scene_summary(sc, reachability_report(sc))
    t0 = time.time()
    intent = MuseAgent(data="private").compile_intent(instruction, summary)
    wall = time.time() - t0
    gate = intent_gate.check(intent.get("goals") or [], sc, physics=True) if intent["status"] == "goals" else None
    after = M.status()["spent"]
    rec = {"key": key, "instruction": instruction, "intent": intent, "gate": gate,
           "cost_usd": round(after - before, 5), "compile_wall_s": round(wall, 1), "data": "private"}
    INTENT.mkdir(parents=True, exist_ok=True)
    (INTENT / f"{key}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(key, intent["status"], json.dumps(intent.get("goals")), intent.get("question") or intent.get("reason"),
          "gate", (gate or {}).get("verdict"), [f["code"] for f in (gate or {}).get("flags", [])],
          f"${rec['cost_usd']}", flush=True)
    return rec


# ====================================================================================== near-placement extension
def _poly(o, pos=None, yaw=None):
    """Full horizontal outline (bounding box of the whole shape, incl. overhanging compound parts; circle for
    upright cylinders) at pose (pos, yaw)."""
    from rehearsal.scene import shape_bbox
    s = o["shape"]
    pos = np.asarray(o["pos"] if pos is None else pos, float)[:2]
    yaw = o.get("yaw", 0.0) if yaw is None else yaw
    if s["type"] in ("cylinder", "bowl") and s.get("axis", "z") == "z":
        a = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        return np.c_[np.cos(a), np.sin(a)] * s["radius"] + pos
    L, W = shape_bbox(s)[:2]
    c, si = np.cos(yaw), np.sin(yaw)
    loc = np.array([[L / 2, W / 2], [-L / 2, W / 2], [-L / 2, -W / 2], [L / 2, -W / 2]])
    return loc @ np.array([[c, si], [-si, c]]) + pos


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def _inside(p, P):
    s = None
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cr) < 1e-12:
            continue
        if s is None:
            s = cr > 0
        elif (cr > 0) != s:
            return False
    return True


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_x(a, b, c, d):
    return (_cross(a, b, c) > 0) != (_cross(a, b, d) > 0) and (_cross(c, d, a) > 0) != (_cross(c, d, b) > 0)


def poly_dist(P, Q):
    """Distance between two convex polygons (0 if they overlap: a vertex inside, or two edges crossing)."""
    if any(_inside(p, Q) for p in P) or any(_inside(q, P) for q in Q):
        return 0.0
    if any(_seg_x(P[i], P[(i + 1) % len(P)], Q[j], Q[(j + 1) % len(Q)])
           for i in range(len(P)) for j in range(len(Q))):
        return 0.0
    d = min(_seg_dist(p, Q[i], Q[(i + 1) % len(Q)]) for p in P for i in range(len(Q)))
    return min(d, min(_seg_dist(q, P[i], P[(i + 1) % len(P)]) for q in Q for i in range(len(P))))


def _yaw_of(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rot2(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_], [s_, c]])


def footprints(o, pos, quat):
    """World 2D outlines of an object at a pose: its convex hulls' footprints (v2 mesh twin) or the v1 outline."""
    if "mesh" not in o:
        return [_poly(o, pos, _yaw_of(quat))]
    R = _rot2(_yaw_of(quat))
    return [np.asarray(fp["poly"]) @ R.T + np.asarray(pos[:2]) for fp in o["mesh"]["hull_footprints"]]


def long_axis(P):
    """Angle of the long axis of a 2D point set."""
    c = P - P.mean(0)
    w, V = np.linalg.eigh(c.T @ c)
    return math.atan2(V[1, 1], V[0, 1])


def _wrap_half(a):
    """Angle to (-pi/2, pi/2]: objects here look the same turned by 180 degrees."""
    a = (a + math.pi / 2) % math.pi - math.pi / 2
    return a


def near_spots(sc, poses, obj, ref, clearance=0.015, max_n=8):
    """Free table spots for `obj` next to `ref` (near() would hold with 1.5 cm to spare), nearest-carry first.
    -> [[x, y, dyaw], ...]; dyaw is 0 unless USER_DESK_ROTATE=1 (demo extension: turn it parallel / square to ref)."""
    from scipy.spatial import ConvexHull
    from rehearsal.predicates import NEAR_GAP
    from rehearsal.scene import footprint_radius
    objs = {o["id"]: o for o in sc["objects"]}
    o, r = objs[obj], objs[ref]
    cur = np.asarray(poses[obj][:2])
    rp = np.asarray(poses[ref][:2])
    ra, rb = footprint_radius(o), footprint_radius(r)
    dmax = NEAR_GAP - 0.015 + ra + rb
    own = np.vstack(footprints(o, poses[obj][:2], poses[obj][3:7])) - cur
    own = own[ConvexHull(own).vertices]                    # its outline relative to its centre, as it lies now
    others = [Q for k, v in objs.items() if k != obj for Q in footprints(v, poses[k][:2], poses[k][3:7])]
    from shapely import prepared
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    blocked = prepared.prep(unary_union([Polygon(Q).buffer(clearance, 8) if len(Q) >= 3 else
                                         Polygon(Q).buffer(clearance) for Q in others]))
    yaws = [0.0]
    if ROTATE:
        rpts = np.vstack(footprints(r, poses[ref][:2], poses[ref][3:7]))
        a0, ar = long_axis(own), long_axis(rpts)
        yaws += sorted({round(_wrap_half(ar - a0), 3), round(_wrap_half(ar + math.pi / 2 - a0), 3)} - {0.0},
                       key=abs)
    (cx, cy), (sx, sy) = sc["table"]["center"], sc["table"]["size"]
    cand = []
    for dyaw in yaws:
        P0 = own @ _rot2(dyaw).T
        for dx in np.arange(-dmax, dmax + 1e-9, 0.01):
            for dy in np.arange(-dmax, dmax + 1e-9, 0.01):
                if math.hypot(dx, dy) > dmax:
                    continue
                p = rp + [dx, dy]
                R = math.hypot(*p)
                if R < 0.30 or R > 0.80:
                    continue
                P = P0 + p
                if (P[:, 0].min() < cx - sx / 2 + 0.03 or P[:, 0].max() > cx + sx / 2 - 0.03
                        or P[:, 1].min() < cy - sy / 2 + 0.03 or P[:, 1].max() > cy + sy / 2 - 0.03):
                    continue
                if blocked.intersects(Polygon(P)):           # == min poly_dist(P, others) < clearance
                    continue
                cand.append((abs(dyaw) > 0, float(np.linalg.norm(p - cur)), p, dyaw))
    cand.sort(key=lambda c: (c[0], c[1]))
    out = []
    for _, _, p, dyaw in cand:
        if all(np.linalg.norm(p - q[:2]) > 0.04 or abs(dyaw - q[2]) > 1e-6 for q in out):
            out.append([p[0], p[1], dyaw])
        if len(out) >= max_n:
            break
    return [[round(float(v), 4) for v in q] for q in out]


PLACE_HEIGHTS = (0.15, 0.22, 0.30)


def near_now(sc, poses, a, b, margin=0.01):
    """predicates.near on settled poses (same formula), with a 1 cm margin."""
    from rehearsal.predicates import NEAR_GAP
    from rehearsal.scene import footprint_radius, object_height
    objs = {o["id"]: o for o in sc["objects"]}
    pa, pb = np.asarray(poses[a][:3]), np.asarray(poses[b][:3])
    gap = np.linalg.norm(pa[:2] - pb[:2]) - footprint_radius(objs[a]) - footprint_radius(objs[b])
    level = abs((pa[2] - object_height(objs[a]) / 2) - (pb[2] - object_height(objs[b]) / 2)) < 0.015
    return gap < NEAR_GAP - margin and level


def transport_ext(sim, obj, dest, relation, transit_height, via=None):
    """Stock transport for in/on.  relation 'near@x,y': pick obj, carry it, lower it onto the table at (x, y)."""
    from rehearsal import executor
    from rehearsal.scene import object_height
    from rehearsal.sim import GAP, IKFailure
    if not relation.startswith("near@"):
        return executor._stock_transport(sim, obj, dest, relation, transit_height, via)
    vals = [float(v) for v in relation[5:].split(",")]
    xy, dyaw = vals[:2], (vals[2] if len(vals) > 2 else 0.0)
    if obj not in sim.objs or dest not in sim.objs:
        raise IKFailure(f"unknown object id {obj!r}/{dest!r}", {"phase": "plan check"})
    p = executor.believed_pos(sim, obj)
    h_obj = object_height(sim.objs[obj])
    above = lambda q, z: np.array([float(q[0]), float(q[1]), z])
    sim.move_to(above(p, transit_height), f"approach {obj}")
    sim.just_released = None
    sim.expect_contact = obj
    sim.move_to(above(p, p[2] + h_obj / 2 + GAP), f"descend to {obj}")
    sim.grasp(obj)
    sim.expect_contact = None
    sim.move_to(above(p, transit_height), f"lift {obj}")
    q_down = sim.down_quat.copy()
    if abs(dyaw) > 1e-6:          # demo extension: turn the held object about the vertical, 0.1 rad per IK step
        import mujoco
        n = max(1, int(math.ceil(abs(dyaw) / 0.1)))
        for k in range(1, n + 1):
            a = dyaw * k / n
            qz = np.array([math.cos(a / 2), 0, 0, math.sin(a / 2)])
            mujoco.mju_mulQuat(sim.down_quat, qz, q_down)
            sim.move_to(above(p, transit_height), f"turn {obj}")
    for k, v in enumerate(via or []):
        sim.move_to(above(v, transit_height), f"carry {obj} via waypoint {k} {list(v)}")
    sim.move_to(above(xy, transit_height), f"carry {obj} to {dest}")
    sim.move_to(above(xy, h_obj + GAP + 0.006), f"lower {obj} next {dest}")
    sim.release()
    sim.wait(0.3)
    sim.move_to(above(xy, transit_height), f"retreat from {dest}")
    if abs(dyaw) > 1e-6:          # turn the empty hand back up at the carry height, not down by the objects
        sim.down_quat[:] = q_down
        sim.move_to(above(xy, transit_height), "turn the hand back")
    sim.placed_by_robot.add(obj)
    for k, v in reversed(list(enumerate(via or []))):
        sim.move_to(above(v, transit_height), f"return via waypoint {k} {list(v)}")


def install_ext():
    from rehearsal import executor
    if not hasattr(executor, "_stock_transport"):
        executor._stock_transport = executor.transport
        executor.transport = transport_ext


def search_ext(sc, goals, time_budget_s=900.0, sim_budget=120, log=print):
    """Sequential DFS over the moves the goals ask for (on/inside: stock carry heights; near: free spots x heights);
    every candidate prefix is simulated by the stock executor and kept only if the decidable goals hold."""
    from rehearsal.executor import run_plan
    from rehearsal.search_planner import HEIGHTS, prefix_goals
    install_ext()
    moves = []
    for g in goals:
        a = g["args"]
        if g["pred"] in ("on", "inside") and len(a) == 2:
            moves.append((a[0], a[1], "on" if g["pred"] == "on" else "in"))
        elif g["pred"] == "stacked":
            moves += [(up, lo, "on") for lo, up in zip(a, a[1:])]
        elif g["pred"] == "near" and len(a) == 2:
            moves.append((a[0], a[1], "near"))
    t0 = time.time()
    st = {"sims": 0, "spots_tried": 0, "wall_s": 0.0}
    first = []          # the first (best-ranked) candidate of every move the search reached: the "direct attempt"
    base = run_plan(sc, {"actions": []}, [g for g in goals if not g.get("implicit")])
    held = sorted(f"{g['pred']}({','.join(g['args'])})" for g in base["goals"] if g["ok"])
    # order: a move whose reference/destination is itself moved comes after that move (near chains)
    todo, rest = [], list(moves)
    while rest:
        nxt = next((m for m in rest if not any(o[0] == m[1] for o in rest if o is not m)), rest[0])
        todo.append(nxt)
        rest.remove(nxt)
    log(f"moves in order {todo}; holding at start: {held}")
    to_move = {m[0] for m in todo}

    def sim(actions, gl):
        if st["sims"] >= sim_budget or time.time() - t0 > time_budget_s:
            raise TimeoutError
        st["sims"] += 1
        return run_plan(sc, {"actions": actions}, gl)

    def dfs(k, prefix, poses):
        if k == len(todo):
            yield list(prefix)
            return
        obj, dest, rel = todo[k]
        moved = {m[0] for m in todo[:k + 1]}
        gl = goals if k == len(todo) - 1 else prefix_goals(goals, moved, to_move)
        if rel == "near" and near_now(sc, poses, obj, dest):
            log(f"  {obj} is already next to {dest} here: no move")
            r = sim(prefix, gl) if prefix else run_plan(sc, {"actions": []}, gl)
            if r["success"]:
                yield from dfs(k + 1, prefix, r["final_poses"])
            return
        if rel == "near":
            # clearance levels, like the shipped planner's (clear first, then near misses; physics decides)
            spots = near_spots(sc, poses, obj, dest, clearance=0.015)
            if not spots:
                spots = near_spots(sc, poses, obj, dest, clearance=0.005)
                log(f"  no spot with 1.5 cm clearance; with 0.5 cm: {len(spots)}")
            log(f"  {obj} next to {dest}: {len(spots)} free spots {spots}")
            cands = [(f"near@{x},{y}" + (f",{t}" if t else ""), h) for (x, y, t) in spots for h in PLACE_HEIGHTS]
            cands.sort(key=lambda c: (PLACE_HEIGHTS.index(c[1]) > 0, ))   # every spot at the lowest carry first
        else:
            cands = [(rel, h) for h in HEIGHTS[:5]]
        accepted = 0
        if cands and all(f["obj"] != obj for f in first):
            first.append({"obj": obj, "dest": dest, "relation": cands[0][0], "transit_height": 0.3, "via": []})
        for relation, h in cands:
            act = {"obj": obj, "dest": dest, "relation": relation, "transit_height": h, "via": []}
            r = sim(prefix + [act], gl)
            st["spots_tried"] += 1
            bad = [f"{g['pred']}({','.join(g['args'][:3])}): {g['why']}" for g in r["goals"] if not g["ok"]]
            log(f"    try {relation} h={h}: {'OK' if r['success'] else 'fail'} "
                f"{(r['execution_error'] or {}).get('error', '')[:120]} {bad[:2]}")
            if r["success"]:
                accepted += 1
                yield from dfs(k + 1, prefix + [act], r["final_poses"])
                if accepted >= 2:
                    return

    plan, reason = None, None
    try:
        for sol in dfs(0, [], base["final_poses"]):
            plan = {"diagnosis": "" if sol else "every move the goals ask for already holds", "actions": sol}
            break
        if plan is None:
            reason = "search space exhausted (no candidate verified in physics)"
    except TimeoutError:
        reason = "search budget exhausted"
    st["wall_s"] = round(time.time() - t0, 1)
    return {"success": plan is not None, "plan": plan, "reason": reason, "stats": st, "moves": todo,
            "direct_attempt": first}


# ====================================================================================== plan + execute (recorded)
def plan_and_run(key, force=False, time_budget=900.0, replay=None):
    """replay: an action list to execute instead of planning (a candidate the search tried and rejected), recorded
    with planned=False and replay_of_rejected_candidate=True, so its clip shows what Rehearsal caught."""
    from rehearsal import executor
    from rehearsal.predicates import with_implicit_goals
    from rehearsal.search_planner import assignments_from_goals, search
    from rehearsal.sim import Sim
    CACHE.mkdir(parents=True, exist_ok=True)
    cj, cz = CACHE / f"{key}_run.json", CACHE / f"{key}_traj.npz"
    if cj.exists() and cz.exists() and not force:
        return json.loads(cj.read_text())
    sc = scene()
    (CACHE / f"{key}_scene.json").write_text(json.dumps(sc))
    comp = json.loads((INTENT / f"{key}.json").read_text())
    rec = {"scene": "user_desk", "key": key, "instruction": comp["instruction"], "intent_status": comp["intent"]["status"]}
    if comp["intent"]["status"] != "goals":
        rec.update(planned=False, executed=False, note="intent is not 'goals' (clarify/refuse): nothing planned or sent")
        cj.write_text(json.dumps(rec, indent=1))
        return rec
    goals = with_implicit_goals(comp["intent"]["goals"], [o["id"] for o in sc["objects"]])
    rec["goals_compiled"] = comp["intent"]["goals"]
    has_near = any(g["pred"] == "near" for g in goals)
    t0 = time.time()
    if replay is not None:
        rec["planner"] = "replay of one candidate the search simulated and rejected (no planning in this run)"
        rec["replay_of_rejected_candidate"] = True
        rec["search"] = {"success": False, "reason": "replay", "stats": {}}
        plan, default = None, replay
    elif not has_near:
        assigns = assignments_from_goals(goals)
        res = search(sc, goals, assigns, sim_budget=300, time_budget_s=time_budget)
        rec["planner"] = "rehearsal.search_planner.search (shipped)"
        rec["search"] = {"success": res["success"], "reason": res.get("reason"), "stats": res.get("stats"),
                         "wall_s": round(time.time() - t0, 1)}
        plan = res["plan"]
        default = [dict(a, transit_height=0.3, via=[]) for a in assigns]
    else:
        # what the shipped planner alone does with a near goal (it has no action for it)
        assigns = assignments_from_goals(goals)
        stock = search(sc, goals, assigns, sim_budget=10, time_budget_s=60)
        rec["shipped_planner_alone"] = {"plan_actions": len((stock["plan"] or {}).get("actions", [])),
                                        "success": stock["success"], "reason": stock.get("reason")}
        res = search_ext(sc, goals, time_budget_s=time_budget)
        rec["planner"] = "user_desk near-placement extension (scripts/user_desk/run_user_desk.py) + stock executor"
        rec["search"] = {"success": res["success"], "reason": res.get("reason"), "stats": res["stats"],
                         "moves": res["moves"], "wall_s": round(time.time() - t0, 1)}
        plan = res["plan"]
        # no plan: the direct attempt is the first candidate the search tried for each move (carry at 0.3 m),
        # executed anyway so the clip shows what would have happened (as exec_tables does with its default plan)
        default = res["direct_attempt"]
    rec["planned"] = plan is not None
    if plan is None:
        plan = {"diagnosis": "rejected candidate, replayed" if replay is not None else
                "search found no plan; straight default plan", "actions": default}
    install_ext()

    class RecSim(Sim):
        last = None

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.qrec = [self.d.qpos.copy()]
            self.nstep = 0
            RecSim.last = self

        def _step(self):
            try:
                super()._step()
            finally:
                self.nstep += 1
                if self.nstep % REC_EVERY == 0:
                    self.qrec.append(self.d.qpos.copy())

    orig = executor.Sim
    executor.Sim = RecSim
    t1 = time.time()
    try:
        r = executor.run_plan(sc, plan, goals)
    finally:
        executor.Sim = orig
    sim = RecSim.last
    rec["plan_exec"] = plan
    # display copy: relation 'near@x,y' -> 'next to' (captions), place_xy kept
    disp = []
    for a in plan["actions"]:
        b = dict(a)
        if a["relation"].startswith("near@"):
            b["relation"] = "next to"
            vals = [float(v) for v in a["relation"][5:].split(",")]
            b["place_xy"] = vals[:2]
            if len(vals) > 2:
                b["turn_rad"] = vals[2]
        disp.append(b)
    rec["plan"] = {"diagnosis": plan.get("diagnosis", ""), "actions": disp}
    rec["sim_wall_s"] = round(time.time() - t1, 1)
    rec["sim_time_s"] = r["sim_time"]
    rec["executed"] = True
    rec["success"] = bool(r["success"])
    rec["execution_error"] = (r["execution_error"] or {}).get("error")
    rec["goals"] = [{"pred": g["pred"], "args": g["args"], "ok": g["ok"], "why": g["why"],
                     "implicit": bool(g.get("implicit"))} for g in r["goals"]]
    explicit = [g for g in rec["goals"] if not g["implicit"]]
    rec["goals_held"] = f"{sum(g['ok'] for g in explicit)}/{len(explicit)}"
    rec["undisturbed_ok"] = all(g["ok"] for g in rec["goals"] if g["implicit"])
    rec["disturbed"] = r["disturbed"]
    rec["events"] = [e for e in r["events"] if e["kind"] in ("grasp", "release", "protective_stop", "ik_failure",
                                                            "grasp_miss")]
    rec["final_poses"] = r["final_poses"]
    rec["dt"] = float(sim.m.opt.timestep) * REC_EVERY
    rec["nq_rigid"] = int(sim.m.nq)
    np.savez_compressed(cz, q=np.asarray(sim.qrec, np.float32))
    cj.write_text(json.dumps(rec, indent=1, default=float))
    print(key, "planned", rec["planned"], "success", rec["success"], rec["goals_held"], rec.get("execution_error"),
          rec["search"], flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["intent", "plan"])
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--instruction", help="intent: compile this text under the (single) key")
    ap.add_argument("--replay", help="plan: JSON action list to execute (a rejected candidate) instead of planning")
    ap.add_argument("--intent-from", help="plan: reuse this key's compiled intent (copied, no model call)")
    a = ap.parse_args()
    sc = scene()
    if a.cmd == "intent":
        if a.instruction:
            compile_one(a.keys[0], a.instruction, sc)
            return
        for k in a.keys or list(TASKS):
            if (INTENT / f"{k}.json").exists() and not a.force:
                print(k, "cached")
                continue
            compile_one(k, TASKS[k], sc)
    elif a.cmd == "plan":
        for k in a.keys:
            if a.intent_from and not (INTENT / f"{k}.json").exists():
                d = json.loads((INTENT / f"{a.intent_from}.json").read_text())
                d.update(key=k, copied_from=a.intent_from, cost_usd=0.0)
                (INTENT / f"{k}.json").write_text(json.dumps(d, indent=1))
            plan_and_run(k, a.force, replay=json.loads(a.replay) if a.replay else None)


if __name__ == "__main__":
    main()
