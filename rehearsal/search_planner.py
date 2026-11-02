"""Deterministic geometric planner over the transport action schema.

The language model decides *what* should happen (goal predicates, which object goes on/in which);
this module decides *how*: action order, carry height per action and at most one detour waypoint,
and the physics simulator verifies every candidate. No model calls happen here.

    from rehearsal.search_planner import search, assignments_from_goals
    res = search(scene, goals, assignments_from_goals(goals))
    res["plan"]        # {"diagnosis": ..., "actions": [...]} or None
    res["stats"]       # sims, kinematic checks, wall time, ...

How it searches (cheapest first, bounded by a simulation budget):

1. Orders: permutations of the assignments, the hint order first, then by Kendall distance from it.
   Orders that put something on an object that is moved later are dropped (suction picks from the top).
2. Depth-first over actions in that order. For action k the candidates are (carry height, via) pairs,
   cost-sorted: the hint's own numbers first, then straight carries from low to high, then detours.
3. Every candidate first gets a kinematic dry run (no dynamics): the exact waypoints `transport`
   will visit, IK every ~2 cm seeded like the executor, and a collision query of the arm and the held
   object against the other objects at each waypoint, from the object poses the previous simulation
   actually ended in. IK failures are pruned; colliding candidates are kept but tried last.
4. Survivors are simulated as the prefix plan (actions 0..k) and accepted only if the goal predicates
   that are decidable so far hold (objects placed so far are where the goals want them, nothing that
   must stay put has moved). Failures feed back simple monotone prunes (an IK failure at height h
   rules out higher carries with the same via; the held object hitting something in transit rules out
   lower ones).
5. Optional robustness gate: each full solution is stress-tested (rehearse.stress_test, selection
   seeds 0..n-1) and the search continues to the next solution if it is below a threshold.
"""
from __future__ import annotations

import itertools
import json
import time

import mujoco
import numpy as np

from .executor import run_plan
from .scene import build_model, footprint_radius, object_height
from .sim import GAP, HOME_Q, IKFailure, Sim

HEIGHTS = (0.15, 0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.45)
KIN_MARGIN = 0.03          # clearance below this (about 2.5 sigma of perception error) is a near miss
STEP = 0.02                # same waypoint spacing as Sim.move_to
TOOL_LINKS = {"hand", "left_finger", "right_finger"}


class _Budget(Exception):
    pass


class SerialPool:
    """Stand-in for a process pool (runs stress variants in this process), for callers that are
    themselves pool workers."""

    @staticmethod
    def map(fn, jobs):
        return map(fn, jobs)


# ----------------------------------------------------------------------------- assignments
def assignments_from_goals(goals: list[dict]) -> list[dict]:
    """on(a,b) -> a on b; inside(a,b) -> a in b; stacked(x1..xn) -> x2 on x1, ..., xn on x(n-1)."""
    out, seen = [], set()
    for g in goals:
        a = g.get("args", [])
        if g["pred"] in ("on", "inside") and len(a) == 2:
            pairs = [(a[0], a[1], "on" if g["pred"] == "on" else "in")]
        elif g["pred"] == "stacked":
            pairs = [(upper, lower, "on") for lower, upper in zip(a, a[1:])]
        else:
            continue
        for obj, dest, rel in pairs:
            if obj not in seen:
                seen.add(obj)
                out.append({"obj": obj, "dest": dest, "relation": rel})
    return out


def assignments_from_plan(plan: dict | None) -> list[dict]:
    out = []
    for a in (plan or {}).get("actions", []):
        out.append({"obj": a["obj"], "dest": a["dest"], "relation": a.get("relation", "on")})
    return out


def merge_assignments(goals: list[dict], plan: dict | None) -> list[dict]:
    """The model's plan when it moves every object the goals need moved (it may know about an
    intermediate step the goals do not spell out), else the goal-derived list in the plan's order."""
    need = assignments_from_goals(goals)
    have = assignments_from_plan(plan)
    if have and {(a["obj"], a["dest"]) for a in need} <= {(a["obj"], a["dest"]) for a in have}:
        return have
    rank = {a["obj"]: i for i, a in enumerate(have)}
    return sorted(need, key=lambda a: rank.get(a["obj"], len(rank)))


# ----------------------------------------------------------------------------- goal atoms
def _atoms(goals: list[dict]) -> list[tuple[str, dict]]:
    """Split goals into (subject, single-predicate goal) atoms."""
    out = []
    for g in goals:
        a = g["args"]
        if g["pred"] == "stacked":
            out += [(up, {"pred": "on", "args": [up, lo]}) for lo, up in zip(a, a[1:])]
        elif g["pred"] == "undisturbed":
            out += [(x, {"pred": "undisturbed", "args": [x]}) for x in a]
        elif a:
            out.append((a[0], {"pred": g["pred"], "args": list(a)}))
    return out


def prefix_goals(goals: list[dict], moved: set[str], to_move: set[str]) -> list[dict]:
    """Atoms decidable after only `moved` have been transported: those about moved objects or
    about objects the plan never moves."""
    return [g for s, g in _atoms(goals)
            if (s in moved or s not in to_move) and not (g["pred"] == "undisturbed" and s in to_move)]


# ----------------------------------------------------------------------------- kinematic dry run
class Kinematics:
    """IK + collision queries on a copy of the scene model, without stepping dynamics."""

    def __init__(self, scene: dict):
        self.scene = scene
        self.sim = Sim(build_model(scene), scene)          # used for its IK only
        m = build_model(scene)
        m.geom_margin[:] = KIN_MARGIN
        self.m, self.d = m, mujoco.MjData(m)
        self.d.qpos[:] = self.sim.d.qpos
        self.objs = {o["id"]: o for o in scene["objects"]}
        self.h = {k: object_height(o) for k, o in self.objs.items()}
        self.qadr = {k: m.jnt_qposadr[m.body_jntadr[m.body(k).id]] for k in self.objs}
        self.root = [self.sim._geom_root[g] for g in range(m.ngeom)]
        self.link = [self.sim._geom_link[g] for g in range(m.ngeom)]
        self.home_tcp = self.sim.tcp().copy()
        self.home_q = HOME_Q.copy()
        self.settled = {k: np.r_[self.sim.d.xpos[b], self.sim.d.xquat[b]] for k, b in self.sim.body_of.items()}
        self.checks = 0

    def set_poses(self, poses: dict):
        for k, p in poses.items():
            a = self.qadr[k]
            self.d.qpos[a:a + 7] = p[:7]

    def _contacts(self, q, held, ignore) -> list[tuple]:
        d = self.d
        d.qpos[:7] = q
        d.qpos[7:9] = 0.0
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_collision(self.m, d)
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            ra, rb = self.root[c.geom1], self.root[c.geom2]
            for mover, other, g in ((ra, rb, c.geom1), (rb, ra, c.geom2)):
                if mover not in ("robot_arm", held) or other in ("robot_arm", "table", "floor", held) \
                        or mover == other:
                    continue
                who = self.link[g] if mover == "robot_arm" else held
                # Expected contacts (the tool on what it picks, the held object on its destination)
                # are only expected for the tool and the held object, never for the rest of the arm.
                if other in ignore and (who in TOOL_LINKS or who == held):
                    continue
                out.append((who, other, float(c.dist)))
        return out

    def check(self, act: dict, poses: dict, start_tcp, start_q) -> dict:
        """Dry-run one transport from the arm state (start_tcp, start_q) with objects at `poses`."""
        self.checks += 1
        obj, dest, rel, h = act["obj"], act["dest"], act["relation"], float(act["transit_height"])
        vias = [np.asarray(v, float) for v in (act.get("via") or [])]
        p, q = np.asarray(poses[obj][:3]), np.asarray(poses[dest][:3])
        ho, hd = self.h[obj], self.h[dest]
        floor = q[2] - hd / 2 + 0.008 if rel == "in" else q[2] + hd / 2
        up = lambda xy, z: np.array([xy[0], xy[1], z])
        supports = self._touching(obj, poses)
        segs = [("approach", up(p, h), None, set()),
                ("descend", up(p, p[2] + ho / 2 + GAP), None, {obj}),
                ("lift", up(p, h), obj, supports)]
        segs += [(f"via{k}", up(v, h), obj, set()) for k, v in enumerate(vias)]
        segs += [("carry", up(q, h), obj, set()),
                 ("lower", up(q, floor + ho + GAP + 0.006), obj, {dest}),
                 ("retreat", up(q, h), None, {obj, dest})]
        segs += [(f"return{k}", up(v, h), None, set()) for k, v in reversed(list(enumerate(vias)))]
        self.set_poses(poses)
        tcp, qc = np.asarray(start_tcp, float), np.asarray(start_q, float)
        hits, near = [], []
        placed = False
        for phase, target, held, ignore in segs:
            if phase == "retreat" and not placed:       # released: obj now rests on dest
                placed = True
                pz = floor + ho / 2
                self.d.qpos[self.qadr[obj]:self.qadr[obj] + 3] = [q[0], q[1], pz]
            n = max(1, int(np.linalg.norm(target - tcp) / STEP))
            for k in range(1, n + 1):
                wp = tcp + (target - tcp) * k / n
                try:
                    qc = self.sim.ik(wp, qc)
                except IKFailure:
                    return {"ik_ok": False, "ik_phase": phase, "ik_at": wp.round(3).tolist(),
                            "hits": hits, "near": near}
                if held:
                    a = self.qadr[held]
                    self.d.qpos[a:a + 3] = [wp[0], wp[1], wp[2] - GAP - ho / 2]
                for who, other, dist in self._contacts(qc, held, ignore):
                    (hits if dist < 0 else near).append((phase, who, other, round(dist, 4)))
            tcp = target
        return {"ik_ok": True, "hits": hits, "near": near, "end_tcp": tcp, "end_q": qc}

    def _touching(self, obj: str, poses: dict) -> set[str]:
        """Objects in contact with `obj` (its supports) in the given state."""
        self.set_poses(poses)
        self.d.qpos[:7] = self.home_q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        out = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            ra, rb = self.root[c.geom1], self.root[c.geom2]
            if obj in (ra, rb) and c.dist < 0.002:
                out.add(rb if ra == obj else ra)
        return out - {"table", "floor", "robot_arm", obj}


# ----------------------------------------------------------------------------- candidates
def via_candidates(p, q, table: dict) -> list[list[float]]:
    """At most one detour point: beside the straight path's midpoint (both sides, three offsets),
    beside its thirds, and pulled toward the robot base (helps at the edge of reach)."""
    p, q = np.asarray(p[:2], float), np.asarray(q[:2], float)
    d = q - p
    L = np.linalg.norm(d) + 1e-9
    nrm = np.array([-d[1], d[0]]) / L
    mid = (p + q) / 2
    pts = [mid + k * nrm for k in (0.12, -0.12, 0.2, -0.2, 0.28, -0.28)]
    pts += [p + d * t + k * nrm for t in (1 / 3, 2 / 3) for k in (0.18, -0.18)]
    pts += [mid * 0.6 + k * nrm for k in (0.0, 0.15, -0.15)]
    (cx, cy), (sx, sy) = table["center"], table["size"]
    ok = [v for v in pts if abs(v[0] - cx) < sx / 2 - 0.03 and abs(v[1] - cy) < sy / 2 - 0.03
          and np.linalg.norm(v) > 0.25]
    return [[round(float(v[0]), 3), round(float(v[1]), 3)] for v in ok]


def candidates(assign: dict, poses: dict, kin: Kinematics, hint: dict | None, use_via: bool = True):
    """(cost, transit_height, via) for one assignment, cheapest first."""
    obj, dest = assign["obj"], assign["dest"]
    p, q = poses[obj], poses[dest]
    ho, hd = kin.h[obj], kin.h[dest]
    # The carry must at least lift the object off its support and lower it onto the destination.
    hmin = max(p[2] + ho / 2 + GAP, q[2] + hd / 2 + ho + GAP + 0.006) + 0.01
    out = []
    if hint and hint.get("obj") == obj and hint.get("dest") == dest:
        out.append((-1.0, float(hint["transit_height"]), [list(map(float, v)) for v in hint.get("via") or []]))
    hs = [h for h in HEIGHTS if h >= hmin] or [round(hmin + 0.02, 3)]
    out += [(float(i), h, []) for i, h in enumerate(hs)]
    if use_via:
        for j, v in enumerate(via_candidates(p, q, kin.scene["table"])):
            out += [(10.0 + i + 0.01 * j, h, [v]) for i, h in enumerate(hs)]
    seen, uniq = set(), []
    for c in sorted(out, key=lambda c: c[0]):
        key = (round(c[1], 3), tuple(map(tuple, c[2])))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def orders(assigns: list[dict], max_orders: int = 120) -> list[list[dict]]:
    """Permutations, given order first then by Kendall distance; drop orders that place something
    on an object that is moved later."""
    n = len(assigns)
    perms = itertools.permutations(range(n)) if n <= 6 else iter([tuple(range(n))])

    def valid(pm):
        for i, j in itertools.combinations(range(n), 2):
            if assigns[pm[i]]["dest"] == assigns[pm[j]]["obj"]:
                return False
        return True

    def kendall(pm):
        return sum(1 for i, j in itertools.combinations(range(n), 2) if pm[i] > pm[j])

    good = sorted((pm for pm in perms if valid(pm)), key=kendall)[:max_orders]
    return [[assigns[i] for i in pm] for pm in good]


# ----------------------------------------------------------------------------- search
def _compact(r: dict) -> dict:
    return {"success": r["success"], "execution_error": r["execution_error"] and r["execution_error"]["error"],
            "failed": [f"{g['pred']}({', '.join(g['args'])}): {g['why']}" for g in r["goals"] if not g["ok"]],
            "evidence": r["evidence"][:6]}


def search(scene: dict, goals: list[dict], assignments: list[dict], *, hint_plan: dict | None = None,
           sim_budget: int = 400, time_budget_s: float = 600.0, branch: int | None = None, per_node: int = 30,
           use_via: bool = True, robust_n: int = 0, robust_threshold: float = 0.9,  # robust_n: 20 recommended
           max_solutions: int = 4, prefer_clearance: bool = True, pool=None, log=None) -> dict:
    """Find a plan over `assignments` that satisfies `goals` (pass them with implicit goals added).

    hint_plan: the model's plan; its order is tried first and its per-action numbers are each
    action's first candidate. branch: accepted children explored per search node before giving up
    on that node (default 3, or 10 with the robustness gate); per_node: simulations per node. robust_n > 0 stress-tests each solution on
    selection seeds 0..robust_n-1 and keeps searching while success < robust_threshold.
    prefer_clearance: try candidates whose dry run keeps KIN_MARGIN clearance first (False = pure
    cost order, like a naive grid search).
    """
    if branch is None:   # the robustness gate needs alternatives to choose from
        branch = 10 if robust_n else 3
    t0 = time.time()
    log = log or (lambda *_: None)
    kin = Kinematics(scene)
    st = {"sims": 0, "sim_seconds": 0.0, "kin_checks": 0, "kin_ik_pruned": 0, "kin_deferred": 0,
          "evidence_pruned": 0, "orders_tried": 0, "solutions": 0, "stress_sims": 0}
    to_move = {a["obj"] for a in assignments}
    hints = {a["obj"]: a for a in (hint_plan or {}).get("actions", [])}

    def simulate(actions, gl):
        if st["sims"] >= sim_budget or time.time() - t0 > time_budget_s:
            raise _Budget()
        st["sims"] += 1
        r = run_plan(scene, {"actions": actions}, gl)
        st["sim_seconds"] += r["sim_time"]
        return r

    # Assignments the initial scene already satisfies need no action.
    base = run_plan(scene, {"actions": []}, [g for _, g in _atoms(goals) if g["pred"] in ("on", "inside")])
    already = {g["args"][0] for g in base["goals"] if g["ok"]}
    todo = [a for a in assignments if a["obj"] not in already]
    if hints:
        rank = {o: i for i, o in enumerate(hints)}
        todo.sort(key=lambda a: rank.get(a["obj"], len(rank)))
    covered = {a["obj"] for a in todo} | already
    missing = [g for s, g in _atoms(goals) if g["pred"] in ("on", "inside") and s not in covered]
    if missing:
        return _result(None, None, st, t0, kin, reason=f"no assignment moves the subject of {missing}")

    solutions = []

    kin_cache, sim_cache = {}, {}
    seen = []                             # solutions already yielded (later clearance levels revisit them)

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
        """level 0: only candidates whose dry run keeps KIN_MARGIN clearance; 1: also near misses;
        2: also kinematic collisions (physics may accept a light touch)."""
        if k == len(order):
            yield list(prefix)
            return
        a = order[k]
        moved = {x["obj"] for x in order[:k + 1]}
        gl = goals if k == len(order) - 1 else prefix_goals(goals, moved, to_move)
        banned = []                       # (via, h_lo, h_hi): heights ruled out by evidence
        accepted, tried = 0, 0
        for c in candidates(a, poses, kin, hints.get(a["obj"]), use_via):
            act = dict(a, transit_height=c[1], via=c[2])
            res = kin_check(prefix, act, poses, tcp, q)
            if not res["ik_ok"]:
                continue
            grade = 2 if res["hits"] else 1 if (res["near"] and prefer_clearance) else 0
            if grade > level:
                st["kin_deferred"] += 1
                continue
            vkey = tuple(map(tuple, act["via"]))
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
            hit = next((d["first_hit"] for d in r["disturbed"] if d.get("first_hit")
                        and d["first_hit"].get("by") == a["obj"]
                        and str(d["first_hit"].get("phase", "")).startswith(("carry", "lift"))), None)
            if hit:
                banned.append((vkey, 0.0, act["transit_height"]))

    try:
        for level in ((0, 1, 2) if prefer_clearance else (2,)):
            for order in orders(todo):
                st["orders_tried"] += 1
                log(f"level {level} order {[x['obj'] for x in order]}")
                for sol in dfs(order, 0, [], dict(kin.settled), kin.home_tcp, kin.home_q, level):
                    if sol in seen:
                        continue
                    seen.append(sol)
                    st["solutions"] += 1
                    st["clearance_level"] = level
                    plan = {"diagnosis": "", "actions": sol}
                    if not robust_n:
                        return _result(plan, None, st, t0, kin)
                    # A later solution is stress-tested only until it can no longer beat the best one so
                    # far, so the kept plan is always fully evaluated and comparisons are exact.
                    best_fails = min((robust_n - r["successes"] for _, _, r in solutions), default=robust_n)
                    rob = gate_stress(scene, plan, goals, robust_n, max_fails=best_fails - 1 if solutions else robust_n)
                    st["stress_sims"] += rob["evaluated"]
                    full = rob["evaluated"] == robust_n
                    frac = rob["successes"] / robust_n
                    log(f"solution {st['solutions']}: robustness {rob['successes']}/{robust_n}"
                        + ("" if full else f" (stopped after {rob['evaluated']}: cannot beat the best)"))
                    if full:
                        solutions.append((frac, plan, {"successes": rob["successes"], "n": robust_n, "seed0": 0,
                                                       "solutions_tested": st["solutions"]}))
                    frac = max(x[0] for x in solutions)
                    if frac >= robust_threshold or st["solutions"] >= max_solutions:
                        raise StopIteration
    except (_Budget, StopIteration):
        pass
    if solutions:
        frac, plan, rob = max(solutions, key=lambda s: s[0])   # first-found wins ties
        return _result(plan, rob, st, t0, kin)
    over = st["sims"] >= sim_budget or time.time() - t0 > time_budget_s
    return _result(None, None, st, t0, kin,
                   reason="simulation budget exhausted" if over else "search space exhausted")


def gate_stress(scene, plan, goals, n, max_fails, seed0=0) -> dict:
    """Stress test on selection seeds seed0..seed0+n-1 (the product noise model, rehearse._variant),
    run in this process and stopped once more than max_fails runs failed. Selection seeds must stay
    disjoint from the held-out seeds (rehearse.HELDOUT_SEED0 = 1000) used to report results."""
    from .rehearse import _variant, object_sigmas
    sig = object_sigmas(scene)
    ok = fails = 0
    for s in range(seed0, seed0 + n):
        if _variant((scene, plan, goals, s, sig, False))["success"]:
            ok += 1
        else:
            fails += 1
            if fails > max_fails:
                break
    return {"successes": ok, "evaluated": ok + fails}


def _result(plan, robustness, st, t0, kin, reason=None) -> dict:
    st = dict(st, kin_checks=kin.checks, wall_s=round(time.time() - t0, 2), sim_seconds=round(st["sim_seconds"], 1))
    if plan:
        plan = dict(plan, diagnosis=(f"geometric search: {st['sims']} simulations, {kin.checks} kinematic "
                                     f"checks, {st['orders_tried']} order(s), {st['wall_s']} s"))
    return {"success": plan is not None, "plan": plan, "stats": st, "robustness": robustness, "reason": reason}


def describe_failure(res: dict, assignments: list[dict]) -> str:
    """One paragraph for the language model when the search found nothing."""
    s = res["stats"]
    return (f"A geometric search over action order, carry heights {HEIGHTS[0]}-{HEIGHTS[-1]} m and one detour "
            f"waypoint ({s['sims']} simulations, {s['kin_checks']} kinematic checks, {s['kin_ik_pruned']} "
            f"candidates out of reach) found no plan that moves "
            + ", ".join(f"{a['obj']} {a['relation']} {a['dest']}" for a in assignments)
            + f" and satisfies the goals ({res['reason']}). The goals or the choice of which object goes "
              "where may be infeasible; consider a different assignment or an intermediate placement.")
