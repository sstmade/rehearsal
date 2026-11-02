"""Goal predicates, evaluated on simulator ground truth (never by a model)."""
from __future__ import annotations

import numpy as np

from .scene import footprint_radius, object_height
from .sim import Sim

UPRIGHT_COS = 0.95
STACK_Z_TOL = 0.015
MOVE_TOL = 0.02
NEAR_GAP = 0.10       # near(a, b): footprints at most 10 cm apart


def _bottom(sim: Sim, oid: str) -> float:
    return sim.obj_state(oid)["pos"][2] - object_height(sim.objs[oid]) / 2


def _top(sim: Sim, oid: str) -> float:
    return sim.obj_state(oid)["pos"][2] + object_height(sim.objs[oid]) / 2


def _xy_dist(sim: Sim, a: str, b: str) -> float:
    return float(np.linalg.norm(sim.obj_state(a)["pos"][:2] - sim.obj_state(b)["pos"][:2]))


def upright(sim, a):
    c = float(sim.obj_state(a)["up_cos"])
    return c > UPRIGHT_COS, f"tilt cos={c:.2f}"


def inside(sim, a, b):
    inner = footprint_radius(sim.objs[b]) - 0.008
    dxy = _xy_dist(sim, a, b)
    za, zb0, zb1 = _bottom(sim, a), _bottom(sim, b), _top(sim, b)
    ok = dxy + footprint_radius(sim.objs[a]) * 0.5 < inner and zb0 - 0.005 < za < zb1
    return ok, f"xy offset {dxy*100:.1f} cm (limit {inner*100:.1f}), bottom z {za*100:.1f} cm"


def on(sim, a, b):
    dz = _bottom(sim, a) - _top(sim, b)
    dxy = _xy_dist(sim, a, b)
    ok = abs(dz) < STACK_Z_TOL and dxy < footprint_radius(sim.objs[b])
    return ok, f"gap {dz*100:.1f} cm, xy offset {dxy*100:.1f} cm"


def stacked(sim, *ids):
    for lower, upper in zip(ids, ids[1:]):
        ok, why = on(sim, upper, lower)
        if not ok:
            return False, f"{upper} not on {lower}: {why}"
    return True, "ok"


def stable(sim, a):
    moved = sim.settle_motion.get(a, 0.0)
    return moved < 0.005, f"moved {moved*100:.1f} cm during 1 s settle"


def undisturbed(sim, *ids):
    bad = []
    for oid in ids:
        st = sim.obj_state(oid)
        moved = float(np.linalg.norm(st["pos"][:2] - sim.start_pos[oid][:2]))
        if moved > MOVE_TOL or st["up_cos"] < UPRIGHT_COS:
            bad.append(f"{oid} moved {moved*100:.1f} cm, tilt cos={st['up_cos']:.2f}")
    return not bad, "; ".join(bad) or "ok"


def near(sim, a, b):
    """a ends up beside b on the same surface ("next to"): footprints (approximated by their inscribed
    circles, so a box's gap is over- not under-estimated) at most NEAR_GAP apart, and neither rests on the other."""
    gap = _xy_dist(sim, a, b) - footprint_radius(sim.objs[a]) - footprint_radius(sim.objs[b])
    stacked_ = on(sim, a, b)[0] or on(sim, b, a)[0] or inside(sim, a, b)[0]
    level = abs(_bottom(sim, a) - _bottom(sim, b)) < STACK_Z_TOL
    ok = gap < NEAR_GAP and not stacked_ and level
    return ok, f"footprint gap {gap*100:.1f} cm (limit {NEAR_GAP*100:.0f}), bottoms {'level' if level else 'not level'}"


def clear(sim, b):
    """Nothing ends up resting on top of b ("don't put anything on the laptop"). Uses the outer half-extent of
    b's footprint, so it is stricter than `on`."""
    from .scene import contact_extent
    reach = 0.5 * max(contact_extent(sim.objs[b]["shape"])[1])
    top = _top(sim, b)
    on_top = [o for o in sim.objs if o != b and -STACK_Z_TOL < _bottom(sim, o) - top < 0.03
              and _xy_dist(sim, o, b) < reach]
    return not on_top, ("nothing on top" if not on_top else f"{', '.join(on_top)} resting on {b}")


PREDICATES = {"inside": inside, "on": on, "upright": upright, "stacked": stacked,
              "stable": stable, "undisturbed": undisturbed, "near": near, "clear": clear}


def evaluate(sim: Sim, goals: list[dict]) -> list[dict]:
    out = []
    for g in goals:
        fn = PREDICATES[g["pred"]]
        ok, why = fn(sim, *g["args"])
        out.append({**g, "ok": bool(ok), "why": why})
    return out


def with_implicit_goals(goals: list[dict], all_ids: list[str]) -> list[dict]:
    """Objects the task does not mention must stay where they are."""
    mentioned = {a for g in goals for a in g["args"]}
    others = [o for o in all_ids if o not in mentioned]
    return goals + ([{"pred": "undisturbed", "args": others, "implicit": True}] if others else [])
