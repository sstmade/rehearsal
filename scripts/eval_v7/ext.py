"""eval_v7: the shipped geometric search with a larger budget and extra action families (PLANNER EXTENSION).

Nothing in rehearsal/ is edited.  The verifier is unchanged: every candidate is executed by the stock
rehearsal.executor.run_plan (MuJoCo, same Sim, same protective stop at 40 N, same IK, same grasp check) and judged by
the stock rehearsal.predicates (same tolerances), goals + implicit undisturbed(everything else).

What is added (all of it is "how to move", never "what counts as done"):
  * more carry heights (0.50, 0.55 m) on top of the shipped 0.15-0.45 m;
  * a placement offset (dx, dy) inside / on the destination instead of always its centre: several items into one
    container no longer all land on the same spot, and the hand can come down on the side away from a neighbour;
  * a yaw turn of the held object about the vertical (the same rotation as the labelled near-placement demo
    extension in scripts/user_desk/run_user_desk.py), to line a long object up with a long container;
  * release above the rim ("drop"): for non-fragile objects going *into* a container, the object is let go with its
    bottom 1.5 cm above the rim instead of lowering the hand into the container (the hand is wider than most
    openings).  Never for eggs, glass, cups, mugs, bowls, plates, beakers, bottles, jars, pots, phones;
  * near-goals use run_user_desk.search_ext (the labelled near-placement demo extension) -- see run_v7.py;
  * a bigger simulation / time budget and more simulations per search node.

Every extended action is encoded in its relation string, "in@dx,dy,dyaw,drop" (base relation "in"/"on"), so
plans stay JSON and the stock run_plan passes it to the (wrapped) transport unchanged.
"""
from __future__ import annotations

import json
import math
import time

import mujoco
import numpy as np

from rehearsal import executor
from rehearsal import search_planner as SP
from rehearsal.scene import footprint_radius, object_height, shape_bbox
from rehearsal.sim import GAP, IKFailure

EXTRA_HEIGHTS = (0.50, 0.55)
FRAGILE = ("egg", "glass", "cup", "mug", "bowl", "plate", "dish", "beaker", "tube", "bottle", "jar", "pot",
           "vase", "phone", "teapot", "carafe", "lamp", "laptop")
DROP_CLEAR = 0.015          # object bottom this far above the rim at release


# ------------------------------------------------------------------------------------------------ relation codec
def parse_rel(rel: str):
    """'in' -> ('in', 0, 0, 0, False); 'in@0.03,-0.02,0.785,1' -> ('in', 0.03, -0.02, 0.785, True)."""
    if "@" not in rel:
        return rel, 0.0, 0.0, 0.0, False
    base, v = rel.split("@", 1)
    x = [float(t) for t in v.split(",")]
    x += [0.0] * (6 - len(x))
    return base, x[0], x[1], x[2], bool(x[3])


def parse_grip(rel: str) -> np.ndarray:
    """Suction point offset (gx, gy) from the object's centre (world frame, at pick time); 0 = centre (stock)."""
    if "@" not in rel:
        return np.zeros(2)
    x = [float(t) for t in rel.split("@", 1)[1].split(",")] + [0.0] * 6
    return np.array([x[4], x[5]])


def _rot(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_], [s_, c]])


def make_rel(base, dx=0.0, dy=0.0, dyaw=0.0, drop=False, gx=0.0, gy=0.0) -> str:
    if max(abs(dx), abs(dy), abs(dyaw), abs(gx), abs(gy)) < 1e-9 and not drop:
        return base
    r = f"{base}@{dx:.3f},{dy:.3f},{dyaw:.3f},{int(drop)}"
    return r + (f",{gx:.3f},{gy:.3f}" if abs(gx) > 1e-9 or abs(gy) > 1e-9 else "")


def _qz(a):
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


# ------------------------------------------------------------------------------------------------ executor wrapper
def transport_v7(sim, obj, dest, relation, transit_height, via=None):
    """Stock transport for plain 'in'/'on'; the extended relation adds offset / yaw / drop (same motions otherwise,
    same phase labels, so logs and captions read the same)."""
    base, dx, dy, dyaw, drop = parse_rel(relation)
    if (base, dx, dy, dyaw, drop) == (relation, 0.0, 0.0, 0.0, False):
        return executor._stock_transport(sim, obj, dest, relation, transit_height, via)
    g = parse_grip(relation)
    for name in (obj, dest):
        if name not in sim.objs:
            raise IKFailure(f"unknown object id {name!r}", {"phase": "plan check"})
    p = executor.believed_pos(sim, obj).copy()
    p[:2] += g                       # suction point (stock: the centre)
    h_obj = object_height(sim.objs[obj])
    above = lambda xy, z: np.array([float(xy[0]), float(xy[1]), z])
    sim.move_to(above(p, transit_height), f"approach {obj}")
    sim.just_released = None
    sim.expect_contact = obj
    sim.move_to(above(p, p[2] + h_obj / 2 + GAP), f"descend to {obj}")
    sim.grasp(obj)
    sim.expect_contact = None
    sim.move_to(above(p, transit_height), f"lift {obj}")
    q_down = sim.down_quat.copy()
    if abs(dyaw) > 1e-6:      # turn the held object about the vertical, 0.1 rad per step (run_user_desk's rotation)
        n = max(1, int(math.ceil(abs(dyaw) / 0.1)))
        for k in range(1, n + 1):
            mujoco.mju_mulQuat(sim.down_quat, _qz(dyaw * k / n), q_down)
            sim.move_to(above(p, transit_height), f"turn {obj}")
    for k, v in enumerate(via or []):
        sim.move_to(above(v, transit_height), f"carry {obj} via waypoint {k} {list(v)}")
    q = executor.believed_pos(sim, dest)
    tgt = q[:2] + np.array([dx, dy]) + _rot(dyaw) @ g      # tool point that puts the object's centre at the spot
    sim.move_to(above(tgt, transit_height), f"carry {obj} to {dest}")
    hd = object_height(sim.objs[dest])
    if drop:
        z = q[2] + hd / 2 + DROP_CLEAR + h_obj + GAP
    else:
        floor = q[2] - hd / 2 + 0.008 if base == "in" else q[2] + hd / 2
        z = floor + h_obj + GAP + 0.006
    sim.move_to(above(tgt, z), f"lower {obj} {base} {dest}")
    sim.release()
    sim.wait(0.3)
    sim.move_to(above(tgt, transit_height), f"retreat from {dest}")
    if abs(dyaw) > 1e-6:
        sim.down_quat[:] = q_down
        sim.move_to(above(tgt, transit_height), "turn the hand back")
    sim.placed_by_robot.add(obj)
    for k, v in reversed(list(enumerate(via or []))):
        sim.move_to(above(v, transit_height), f"return via waypoint {k} {list(v)}")


def install():
    if not hasattr(executor, "_stock_transport"):
        executor._stock_transport = executor.transport
    executor.transport = transport_v7


# ------------------------------------------------------------------------------------------------ kinematic dry run
class KinV7(SP.Kinematics):
    """The shipped dry run (IK every 2 cm + collision query of arm and held object), aware of offset/yaw/drop."""

    def check(self, act, poses, start_tcp, start_q):
        base, dx, dy, dyaw, drop = parse_rel(act["relation"])
        g = parse_grip(act["relation"])
        if (dx, dy, dyaw, drop) == (0.0, 0.0, 0.0, False) and not g.any():
            return super().check(dict(act, relation=base), poses, start_tcp, start_q)
        self.checks += 1
        obj, dest, h = act["obj"], act["dest"], float(act["transit_height"])
        vias = [np.asarray(v, float) for v in (act.get("via") or [])]
        p, q = np.asarray(poses[obj][:3]), np.asarray(poses[dest][:3])
        p = p.copy()
        p[:2] += g
        tq = q[:2] + np.array([dx, dy]) + _rot(dyaw) @ g
        ho, hd = self.h[obj], self.h[dest]
        if drop:
            zl = q[2] + hd / 2 + DROP_CLEAR + ho + GAP
        else:
            floor = q[2] - hd / 2 + 0.008 if base == "in" else q[2] + hd / 2
            zl = floor + ho + GAP + 0.006
        up = lambda xy, z: np.array([xy[0], xy[1], z])
        supports = self._touching(obj, poses)
        segs = [("approach", up(p, h), None, set(), 0.0),
                ("descend", up(p, p[2] + ho / 2 + GAP), None, {obj}, 0.0),
                ("lift", up(p, h), obj, supports, 0.0)]
        if abs(dyaw) > 1e-6:
            n = max(1, int(math.ceil(abs(dyaw) / 0.1)))
            segs += [("turn", up(p, h), obj, set(), dyaw * k / n) for k in range(1, n + 1)]
        segs += [(f"via{k}", up(v, h), obj, set(), dyaw) for k, v in enumerate(vias)]
        segs += [("carry", up(tq, h), obj, set(), dyaw),
                 ("lower", up(tq, zl), obj, {dest} | self._inside_of(dest, poses), dyaw),
                 ("retreat", up(tq, h), None, {obj, dest} | self._inside_of(dest, poses), dyaw)]
        if abs(dyaw) > 1e-6:
            segs += [("turnback", up(tq, h), None, set(), 0.0)]
        segs += [(f"return{k}", up(v, h), None, set(), 0.0) for k, v in reversed(list(enumerate(vias)))]
        self.set_poses(poses)
        q0 = self.sim.down_quat.copy()
        oq = np.asarray(poses[obj][3:7], float)
        tcp, qc = np.asarray(start_tcp, float), np.asarray(start_q, float)
        hits, near = [], []
        placed = False
        try:
            for phase, target, held, ignore, yaw in segs:
                mujoco.mju_mulQuat(self.sim.down_quat, _qz(yaw), q0)
                if phase == "retreat" and not placed:
                    placed = True
                    a = self.qadr[obj]
                    zf = (q[2] - hd / 2 + 0.008 if base == "in" else q[2] + hd / 2) + ho / 2
                    self.d.qpos[a:a + 3] = [q[0] + dx, q[1] + dy, zf]
                n = max(1, int(np.linalg.norm(target - tcp) / SP.STEP))
                for k in range(1, n + 1):
                    wp = tcp + (target - tcp) * k / n
                    try:
                        qc = self.sim.ik(wp, qc)
                    except IKFailure:
                        return {"ik_ok": False, "ik_phase": phase, "ik_at": wp.round(3).tolist(),
                                "hits": hits, "near": near}
                    if held:
                        a = self.qadr[held]
                        c = wp[:2] - _rot(yaw) @ g
                        self.d.qpos[a:a + 3] = [c[0], c[1], wp[2] - GAP - ho / 2]
                        qq = np.zeros(4)
                        mujoco.mju_mulQuat(qq, _qz(yaw), oq)
                        self.d.qpos[a + 3:a + 7] = qq
                    for who, other, dist in self._contacts(qc, held, ignore):
                        (hits if dist < 0 else near).append((phase, who, other, round(dist, 4)))
                tcp = target
        finally:
            self.sim.down_quat[:] = q0
        return {"ik_ok": True, "hits": hits, "near": near, "end_tcp": tcp, "end_q": qc}

    def _inside_of(self, dest, poses):
        """Objects already sitting in/on dest (contact with them while placing is expected)."""
        q = np.asarray(poses[dest][:3])
        r = footprint_radius(self.objs[dest]) + 0.02
        hd = self.h[dest]
        return {k for k, p in poses.items() if k != dest and np.linalg.norm(np.asarray(p[:2]) - q[:2]) < r
                and p[2] > q[2] - hd / 2}


# ------------------------------------------------------------------------------------------------ candidates
def _yaw_of(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap_half(a):
    return (a + math.pi / 2) % math.pi - math.pi / 2


def long_axis(o, quat):
    """World angle of the object's long horizontal axis, and its elongation (long / short side)."""
    L, W = shape_bbox(o["shape"])[:2]
    a = _yaw_of(quat)
    return (a if L >= W else a + math.pi / 2), max(L, W) / max(min(L, W), 1e-6)


def fragile(o) -> bool:
    t = (o["id"] + " " + o.get("category", "") + " " + o.get("label", "")).lower()
    return any(k in t for k in FRAGILE)


def place_spots(assign, poses, kin, max_n=6):
    """Offsets (dx, dy) from the destination's centre where the stock predicate would still hold with 1 cm to spare,
    best first: furthest from what already sits in the destination and from the objects around it."""
    obj, dest, rel = assign["obj"], assign["dest"], assign["relation"]
    o, d = kin.objs[obj], kin.objs[dest]
    q = np.asarray(poses[dest][:3])
    if rel == "in":
        lim = footprint_radius(d) - 0.008 - footprint_radius(o) * 0.5 - 0.01
    else:
        lim = footprint_radius(d) - 0.01
    if lim <= 0.005:
        return [(0.0, 0.0)]
    # the object's own reach (half its long side) must also stay inside the destination's outline
    Ld, Wd = shape_bbox(d["shape"])[:2]
    others = [(np.asarray(p[:2]), footprint_radius(kin.objs[k]))
              for k, p in poses.items() if k not in (obj, dest)]
    pts = []
    step = max(0.01, lim / 4)
    for x in np.arange(-lim, lim + 1e-9, step):
        for y in np.arange(-lim, lim + 1e-9, step):
            if math.hypot(x, y) > lim:
                continue
            c = q[:2] + np.array([x, y])
            clear = min((np.linalg.norm(c - pc) - rc for pc, rc in others), default=1.0)
            pts.append((clear, -math.hypot(x, y), x, y))
    pts.sort(reverse=True)
    out = [(0.0, 0.0)]
    for _, _, x, y in pts:
        if all(math.hypot(x - a, y - b) > max(0.025, lim / 2.5) for a, b in out):
            out.append((round(float(x), 3), round(float(y), 3)))
        if len(out) >= max_n:
            break
    return out


def yaw_options(assign, poses, kin):
    obj, dest = assign["obj"], assign["dest"]
    o, d = kin.objs[obj], kin.objs[dest]
    ao, eo = long_axis(o, poses[obj][3:7])
    ad, ed = long_axis(d, poses[dest][3:7])
    if eo < 1.4:
        return [0.0]
    out = [0.0]
    for target in ((ad, ad + math.pi / 2) if ed >= 1.15 else (ad, ad + math.pi / 4, ad - math.pi / 4)):
        dy = round(_wrap_half(target - ao), 3)
        if abs(dy) > 0.05 and all(abs(dy - v) > 0.05 for v in out):
            out.append(dy)
    return out


def candidates_v7(assign, poses, kin, hint, use_via=True):
    """(cost, relation, transit_height, via): the shipped candidates first (straight carries low to high), then the
    extra heights, then the extension families (offset x yaw x drop at three heights), then the shipped detours."""
    stock = SP.candidates(dict(assign, relation=assign["relation"]), poses, kin, hint, use_via)
    rel = assign["relation"]
    out = [(c, rel, h, v) for c, h, v in stock if not v]
    vias = [(c + 40.0, rel, h, v) for c, h, v in stock if v]
    hs = sorted({h for _, h, v in stock if not v})
    hmin = hs[0] if hs else 0.3
    out += [(8.0 + i, rel, h, []) for i, h in enumerate(EXTRA_HEIGHTS) if h >= hmin]
    o, d = kin.objs[assign["obj"]], kin.objs[assign["dest"]]
    drops = [False]
    if rel == "in" and not fragile(o) and kin.h[assign["dest"]] >= 0.05:
        drops.append(True)
    spots = place_spots(assign, poses, kin)
    yaws = yaw_options(assign, poses, kin)
    hx = sorted({hmin, max(hmin, 0.30), max(hmin, 0.40)})
    j = 0
    for drop in drops:
        for dyaw in yaws:
            for si, (dx, dy) in enumerate(spots):
                if not drop and dyaw == 0.0 and si == 0:
                    continue
                for hi, h in enumerate(hx):
                    j += 1
                    cost = 10.0 + 0.6 * si + 0.8 * (dyaw != 0.0) + 1.5 * drop + 2.0 * hi + 0.001 * j
                    out.append((cost, make_rel(rel, dx, dy, dyaw, drop), h, []))
    # suction point off the centre (away from a tall neighbour): the stock grasp check accepts up to 0.8 x the
    # footprint radius; we use 0.6 x, in 4 directions, with the first spot / no yaw
    gr = 0.6 * 0.8 * footprint_radius(o)
    if gr >= 0.015:
        for gi, (gx, gy) in enumerate(((gr, 0.0), (-gr, 0.0), (0.0, gr), (0.0, -gr))):
            for drop in drops:
                for hi, h in enumerate(hx):
                    for dyaw in yaws[:2]:
                        cost = 20.0 + 0.3 * gi + 1.5 * drop + 2.0 * hi + 0.8 * (dyaw != 0.0)
                        out.append((cost, make_rel(rel, 0.0, 0.0, dyaw, drop, round(gx, 3), round(gy, 3)), h, []))
    out += vias
    seen, uniq = set(), []
    for c in sorted(out, key=lambda c: c[0]):
        key = (c[1], round(c[2], 3), tuple(map(tuple, c[3])))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


# ------------------------------------------------------------------------------------------------ rejection reasons
def classify(r) -> str:
    """One reason per rejected rollout: collision / reach / grasp_miss / drop / tip / disturbance / placement."""
    err = r.get("execution_error")
    if err:
        e = err.get("error", "")
        if "protective stop" in e:
            return "collision"
        if "suction cup missed" in e:
            return "grasp_miss"
        if "IK" in e or "reach" in e:
            return "reach"
        return "other_error"
    dist = r.get("disturbed") or []
    bad = [g for g in r["goals"] if not g["ok"]]
    if any(x.get("fell_off_table") for x in dist):
        return "drop"
    if any(g["pred"] == "upright" for g in bad) or any(x.get("tilt_deg", 0) > 30 for x in dist):
        return "tip"
    if any(g["pred"] == "undisturbed" for g in bad):
        return "disturbance"
    if any(g["pred"] in ("inside", "on", "stacked", "near") for g in bad):
        return "placement"
    if any(g["pred"] == "stable" for g in bad):
        return "unstable"
    return "other"


# ------------------------------------------------------------------------------------------------ rim audit
RIM_MARGIN = 0.01


def rim_audit(scene, r, goals):
    """Extra check on top of the stock inside() (stricter, never looser): an object whose bottom ends within 1 cm of
    the container's top edge is lying across / on the rim, not in it (stock inside() only asks bottom < top).
    -> '' if fine, else a reason string."""
    objs = {o["id"]: o for o in scene["objects"]}
    fp = r["final_poses"]
    bad = []
    for g in goals:
        if g["pred"] != "inside" or len(g["args"]) != 2:
            continue
        a, b = g["args"]
        if a not in fp or b not in fp:
            continue
        za = fp[a][2] - object_height(objs[a]) / 2
        top = fp[b][2] + object_height(objs[b]) / 2
        if za > top - RIM_MARGIN:
            bad.append(f"rim audit: {a} bottom {za*100:.1f} cm, {b} top {top*100:.1f} cm (rests on the rim, not in it)")
    return "; ".join(bad)


# ------------------------------------------------------------------------------------------------ search
def search_v7(scene, goals, assignments, *, sim_budget=1500, time_budget_s=2400.0, branch=3, per_node=60,
              log=None, trace=None):
    """The shipped search_planner.search loop (same orders, same depth-first prefix verification, same clearance
    levels, same evidence prunes) over candidates_v7, with the effort trace recorded.  trace: list to append one
    dict per physics rollout."""
    install()
    t0 = time.time()
    c0 = time.process_time()       # the budget is CPU time of this search (the machine is shared with renders)
    log = log or (lambda *_: None)
    kin = KinV7(scene)
    st = {"sims": 0, "sim_seconds": 0.0, "kin_checks": 0, "kin_ik_pruned": 0, "kin_deferred": 0,
          "evidence_pruned": 0, "orders_tried": 0, "solutions": 0, "found_at_sim": None, "families": {}}
    to_move = {a["obj"] for a in assignments}
    trace = trace if trace is not None else []

    def simulate(actions, gl):
        if st["sims"] >= sim_budget or time.process_time() - c0 > time_budget_s:
            raise SP._Budget()
        st["sims"] += 1
        r = executor.run_plan(scene, {"actions": actions}, gl)
        st["sim_seconds"] += r["sim_time"]
        if r["success"]:
            rim = rim_audit(scene, r, gl)
            if rim:                # stricter than the stock predicate, never looser
                r = dict(r, success=False, rim_audit=rim)
                st["rim_rejected"] = st.get("rim_rejected", 0) + 1
        trace.append({"n": st["sims"], "depth": len(actions), "full": gl is goals, "ok": bool(r["success"]),
                      "why": None if r["success"] else ("placement" if r.get("rim_audit") else classify(r)),
                      "detail": None if r["success"] else (r.get("rim_audit") or (r["execution_error"] or {}).get("error")
                                                           or "; ".join(f"{g['pred']}({','.join(g['args'][:2])}): {g['why']}"
                                                                        for g in r["goals"] if not g["ok"])[:300]),
                      "last": {k: actions[-1][k] for k in ("obj", "dest", "relation", "transit_height", "via")},
                      "t": round(time.time() - t0, 1)})
        return r

    base = executor.run_plan(scene, {"actions": []}, [g for _, g in SP._atoms(goals) if g["pred"] in ("on", "inside")])
    already = {g["args"][0] for g in base["goals"] if g["ok"]}
    todo = [a for a in assignments if a["obj"] not in already]
    covered = {a["obj"] for a in todo} | already
    missing = [g for s, g in SP._atoms(goals) if g["pred"] in ("on", "inside") and s not in covered]
    if missing:
        return _res(None, st, t0, kin, f"no assignment moves the subject of {missing}")
    kin_cache, sim_cache = {}, {}
    seen = []

    def kin_check(prefix, act, poses, tcp, q):
        key = json.dumps([prefix, act], sort_keys=True)
        if key not in kin_cache:
            kin_cache[key] = kin.check(act, poses, tcp, q)
            if not kin_cache[key]["ik_ok"]:
                st["kin_ik_pruned"] += 1
        return kin_cache[key]

    def sim_cached(actions, gl):
        key = json.dumps([actions, gl], sort_keys=True)
        if key not in sim_cache:
            sim_cache[key] = simulate(actions, gl)
        return sim_cache[key]

    def dfs(order, k, prefix, poses, tcp, q, level):
        if k == len(order):
            yield list(prefix)
            return
        a = order[k]
        moved = {x["obj"] for x in order[:k + 1]}
        gl = goals if k == len(order) - 1 else SP.prefix_goals(goals, moved, to_move)
        banned = []
        accepted, tried = 0, 0
        for c in candidates_v7(a, poses, kin, None):
            act = dict(a, relation=c[1], transit_height=c[2], via=c[3])
            res = kin_check(prefix, act, poses, tcp, q)
            if not res["ik_ok"]:
                continue
            grade = 2 if res["hits"] else 1 if res["near"] else 0
            if grade > level:
                st["kin_deferred"] += 1
                continue
            vkey = (c[1], tuple(map(tuple, act["via"])))
            if any(v == vkey and lo <= act["transit_height"] <= hi for v, lo, hi in banned):
                st["evidence_pruned"] += 1
                continue
            if tried >= per_node or accepted >= branch:
                return
            tried += 1
            r = sim_cached(prefix + [act], gl)
            if r["success"]:
                accepted += 1
                nxt = {kk: np.asarray(v) for kk, v in r["final_poses"].items()}
                yield from dfs(order, k + 1, prefix + [act], nxt, res["end_tcp"], res["end_q"], level)
                continue
            err = r["execution_error"]
            if err and err.get("action_index") == k and "IK" in err.get("error", ""):
                banned.append((vkey, act["transit_height"], 9.0))
            hit = next((dd["first_hit"] for dd in r["disturbed"] if dd.get("first_hit")
                        and dd["first_hit"].get("by") == a["obj"]
                        and str(dd["first_hit"].get("phase", "")).startswith(("carry", "lift"))), None)
            if hit:
                banned.append((vkey, 0.0, act["transit_height"]))

    plan = None
    try:
        for level in (0, 1, 2):
            # with 5+ moves the shipped 120 orders x 3 clearance levels never reach level 1/2: cap orders per level
            for order in SP.orders(todo, max_orders=120 if len(todo) <= 4 else 12):
                st["orders_tried"] += 1
                log(f"level {level} order {[x['obj'] for x in order]} sims {st['sims']}")
                for sol in dfs(order, 0, [], dict(kin.settled), kin.home_tcp, kin.home_q, level):
                    if sol in seen:
                        continue
                    seen.append(sol)
                    st["solutions"] += 1
                    st["clearance_level"] = level
                    st["found_at_sim"] = st["sims"]
                    plan = {"diagnosis": "", "actions": sol}
                    raise StopIteration
    except (SP._Budget, StopIteration):
        pass
    if plan:
        for a in plan["actions"]:
            b, dx, dy, dyaw, drop = parse_rel(a["relation"])
            fam = []
            if dx or dy:
                fam.append("offset")
            if parse_grip(a["relation"]).any():
                fam.append("grip_offset")
            if dyaw:
                fam.append("yaw")
            if drop:
                fam.append("drop")
            if a["transit_height"] in EXTRA_HEIGHTS:
                fam.append("extra_height")
            if a.get("via"):
                fam.append("via")
            for f in fam or ["stock"]:
                st["families"][f] = st["families"].get(f, 0) + 1
        st["cpu_s"] = round(time.process_time() - c0, 1)
        return _res(plan, st, t0, kin, None)
    st["cpu_s"] = round(time.process_time() - c0, 1)
    over = st["sims"] >= sim_budget or time.process_time() - c0 > time_budget_s
    return _res(None, st, t0, kin, "simulation budget exhausted" if over else "search space exhausted")


def _res(plan, st, t0, kin, reason):
    st = dict(st, kin_checks=kin.checks, wall_s=round(time.time() - t0, 1), sim_seconds=round(st["sim_seconds"], 1))
    if plan:
        plan = dict(plan, diagnosis=(f"eval_v7 extended search: {st['sims']} simulations, {kin.checks} kinematic "
                                     f"checks, {st['orders_tried']} order(s), {st['wall_s']} s"))
    return {"success": plan is not None, "plan": plan, "stats": st, "reason": reason}
