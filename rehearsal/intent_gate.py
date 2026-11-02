"""Deterministic intent gate: checks a compiled goal set before anything is planned. No LLM.

A physics verifier checks whether the goals hold, never whether they are the right goals. This gate catches the
cheap, structural ways a compiled goal set can be wrong (docs/intent.md scores it on GoalBench):

  empty           no goal asks for any change (only undisturbed/clear, or nothing at all)       -> refuse
  unknown_object  a goal names an id that is not in the scene                                  -> refuse
  malformed       unknown predicate or wrong number of arguments                               -> refuse
  contradiction   one object sent to two places, clear(b) while something goes onto b,
                  undisturbed(x) while x is moved                                               -> clarify
  vacuous         every compiled goal already holds in the scene as it is (one 1.5 s physics
                  settle, no robot motion): "success" would mean the robot does nothing          -> clarify

It also builds an effect preview for the confirmation screen: what will move where, what must stay put, and
which extra conditions are checked. It cannot catch a wrong goal that does ask for a real change
(e.g. "make room" compiled to "book onto the laptop").
"""
from __future__ import annotations

from .predicates import MOVE_PREDS, PREDICATES, moved_objects

ARITY = {"inside": (2, 2), "on": (2, 2), "near": (2, 2), "upright": (1, 1), "stable": (1, 1), "clear": (1, 1),
         "stacked": (2, 99), "undisturbed": (1, 99)}
CHANGE_PREDS = set(MOVE_PREDS) | {"stacked", "upright", "stable"}
REL_TEXT = {"on": "onto", "inside": "into", "near": "next to"}


def _label(scene: dict, oid: str) -> str:
    o = next((o for o in scene["objects"] if o["id"] == oid), None)
    return f"{o.get('label') or oid} ({oid})" if o else oid


def placements(goals: list[dict]) -> list[tuple[str, str, str]]:
    """(obj, relation, dest) for every move the goals ask for; stacked() expanded bottom-up."""
    out = []
    for g in goals:
        a = g.get("args") or []
        if g.get("pred") in MOVE_PREDS and len(a) == 2:
            out.append((a[0], g["pred"], a[1]))
        elif g.get("pred") == "stacked":
            out += [(u, "on", lo) for lo, u in zip(a, a[1:])]
    return list(dict.fromkeys(out))


def effect_preview(goals: list[dict], scene: dict) -> dict:
    """What the compiled goals mean physically, in plain words, before anything runs."""
    ids = [o["id"] for o in scene["objects"]]
    compiled = [g for g in goals if not g.get("implicit")]
    moves = placements(compiled)
    moved = moved_objects(compiled)
    stays = [i for i in ids if i not in moved]
    conds = []
    for g in compiled:
        a = g.get("args") or []
        if g.get("pred") == "upright" and a:
            conds.append(f"{_label(scene, a[0])} ends upright")
        elif g.get("pred") == "stable" and a:
            conds.append(f"{_label(scene, a[0])} is at rest at the end")
        elif g.get("pred") == "clear" and a:
            conds.append(f"nothing ends up on top of {_label(scene, a[0])}")
    return {"moves": [{"obj": o, "relation": r, "dest": d,
                       "text": f"{_label(scene, o)} goes {REL_TEXT[r]} {_label(scene, d)}"} for o, r, d in moves],
            "stays": stays,
            "stays_text": (f"must not move or tip: {', '.join(_label(scene, i) for i in stays)}" if stays else ""),
            "conditions": conds}


def _structural(goals: list[dict], ids: set[str]) -> list[dict]:
    flags = []
    unknown = sorted({a for g in goals for a in g.get("args") or [] if a not in ids})
    if unknown:
        flags.append({"code": "unknown_object", "message": f"the goals name {', '.join(unknown)}, which "
                      f"{'is' if len(unknown) == 1 else 'are'} not in the scene"})
    bad = [g for g in goals if g.get("pred") not in PREDICATES or g.get("pred") not in ARITY
           or not ARITY[g["pred"]][0] <= len(g.get("args") or []) <= ARITY[g["pred"]][1]]
    if bad:
        flags.append({"code": "malformed", "message": "malformed goal(s): " + ", ".join(
            f"{g.get('pred')}({', '.join(g.get('args') or [])})" for g in bad)})
    if not [g for g in goals if g.get("pred") in CHANGE_PREDS]:
        flags.append({"code": "empty", "message": "the compiled goals ask for nothing to change"
                      + (" (no goals at all)" if not goals else "")})
    return flags


def _contradictions(goals: list[dict]) -> list[dict]:
    flags, dest = [], {}
    for o, r, d in placements(goals):
        dest.setdefault(o, set()).add((r, d))
    for o, ds in dest.items():
        places = {d for r, d in ds if r != "near"}
        if len(places) > 1:
            flags.append({"code": "contradiction", "message": f"{o} is sent to {len(places)} places: "
                          + ", ".join(sorted(places))})
    targets = {d for o, r, d in placements(goals) if r in ("on", "inside")}
    for g in goals:
        a = g.get("args") or []
        if g.get("pred") == "clear" and a and a[0] in targets:
            flags.append({"code": "contradiction", "message": f"clear({a[0]}) but something is placed on/in {a[0]}"})
        if g.get("pred") == "undisturbed" and not g.get("implicit"):
            both = sorted(set(a) & moved_objects(goals))
            if both:
                flags.append({"code": "contradiction", "message": f"{', '.join(both)} must stay put and also move"})
    return flags


def vacuous(goals: list[dict], scene: dict) -> tuple[bool, list[dict]]:
    """True when every compiled goal already holds after a 1.5 s settle with no robot motion."""
    from .executor import run_plan
    compiled = [g for g in goals if not g.get("implicit")]
    if not compiled:
        return True, []
    r = run_plan(scene, {"actions": []}, compiled)
    return all(x["ok"] for x in r["goals"]) and r["execution_error"] is None, r["goals"]


def check(goals: list[dict], scene: dict, physics: bool = True) -> dict:
    """Gate a compiled goal set. Returns
    {"verdict": "pass"|"clarify"|"refuse", "flags": [{"code", "message"}], "preview": effect_preview(...),
     "holds_now": [goal results at t=0] | None}. physics=False skips the vacuity simulation."""
    ids = {o["id"] for o in scene["objects"]}
    compiled = [g for g in goals if not g.get("implicit")]
    flags = _structural(compiled, ids)
    holds = None
    if not flags:
        flags += _contradictions(compiled)
        if physics:
            vac, holds = vacuous(compiled, scene)
            if vac:
                flags.append({"code": "vacuous", "message": "every compiled goal already holds right now, so "
                              "\"success\" would mean the robot does nothing"})
    refuse = {"empty", "unknown_object", "malformed"}
    verdict = ("refuse" if any(f["code"] in refuse for f in flags) else "clarify" if flags else "pass")
    known = [g for g in compiled if set(g.get("args") or []) <= ids]
    return {"verdict": verdict, "flags": flags, "preview": effect_preview(known, scene), "holds_now": holds}


def summarize(gate: dict) -> str:
    """One sentence for a refusal/clarification shown to the user."""
    return "; ".join(f["message"] for f in gate["flags"])
