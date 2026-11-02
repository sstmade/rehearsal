""""Physics certified the wrong goal": take goals Muse Spark actually compiled (from results/raw_<arm>.jsonl),
plan for them with a simple deterministic planner (no LLM, no API calls), simulate in MuJoCo, then judge the SAME
simulation twice: against the compiled goals (what the product checks and reports) and against the hand-labelled
reference goals (what the user meant).

    .venv/bin/python -m bench.goals.consequence A:i06:0 A:f02:0 ...
    .venv/bin/python -m bench.goals.consequence --auto      # every distinct wrong compilation, first repeat
Writes results/consequence.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.goals import score as S  # noqa: E402
from bench.goals.cases import CASES  # noqa: E402
from bench.goals.run import RESULTS, load_raw, load_scenes  # noqa: E402
from rehearsal.executor import run_plan  # noqa: E402
from rehearsal.predicates import with_implicit_goals  # noqa: E402
from rehearsal.scene import object_height  # noqa: E402

BY_ID = {c["id"]: c for c in CASES}
HEIGHTS = (0.32, 0.40, 0.48, 0.25, 0.20, 0.16)


def plan_for(goals: list[dict], scene: dict, transit: float) -> dict:
    """One transport per on/inside atom (stacked expanded bottom-up), lower placements first."""
    moves = []
    for g in goals:
        if g["pred"] in ("on", "inside") and len(g["args"]) == 2:
            moves.append((g["args"][0], g["args"][1], "in" if g["pred"] == "inside" else "on"))
        elif g["pred"] == "stacked":
            moves += [(u, l, "on") for l, u in zip(g["args"], g["args"][1:])]
    moves = list(dict.fromkeys(moves))      # on(a,b) + stacked(..., b, a) name the same move once
    ordered, placed = [], set()
    while moves:                 # place x on y only after y itself has been placed (if y moves at all)
        movers = {m[0] for m in moves}
        nxt = next((m for m in moves if m[1] not in movers or m[1] in placed), moves[0])
        moves.remove(nxt)
        ordered.append(nxt)
        placed.add(nxt[0])
    return {"actions": [{"obj": o, "dest": d, "relation": r, "transit_height": transit, "via": []}
                        for o, d, r in ordered]}


def reference_goals(case: dict, ids: list[str], ref_index: int = 0) -> list[dict]:
    if not case["refs"]:
        return []
    ref = case["refs"][ref_index]
    goals = [{"pred": p, "args": list(a)} for s in ref["required"] + ref["implicit"] for p, a in [S.parse_alts(s)[0]]]
    keep = sorted(set(ids) - S._mentioned(ref))
    return goals + ([{"pred": "undisturbed", "args": keep}] if keep else [])


def demo(arm: str, cid: str, rep: int) -> dict:
    case = BY_ID[cid]
    scene = load_scenes()[case["scene"]]
    ids = [o["id"] for o in scene["objects"]]
    out = load_raw(arm)[(cid, rep)]["output"]
    compiled = [g for g in out.get("goals", []) if set(g["args"]) <= set(ids)]
    if out.get("status", "ok") != "ok":
        compiled = []
    product_goals = with_implicit_goals([dict(g) for g in compiled], ids)
    ref = reference_goals(case, ids, S.score_trial(case, out, ids).get("ref", 0) or 0)
    best = None
    for h in HEIGHTS:
        plan = plan_for(compiled, scene, h)
        r = run_plan(scene, plan, product_goals + ref)
        n = len(product_goals)
        prod, refres = r["goals"][:n], r["goals"][n:]
        ok_exec = r["execution_error"] is None
        res = {"transit_height": h, "plan": plan, "execution_error": (r["execution_error"] or {}).get("error"),
               "product_success": ok_exec and all(x["ok"] for x in prod),
               # a request that should have been refused / clarified has no correct "success"
               "intent_success": (ok_exec and all(x["ok"] for x in refres)) if ref else
               (False if case["expect"] == "flag" else None),
               "product_goals": [_short(x) for x in prod], "reference_goals": [_short(x) for x in refres],
               "disturbed": r["disturbed"]}
        if best is None or (res["product_success"] and not best["product_success"]):
            best = res
        if res["product_success"]:
            break
    return {"arm": arm, "case": cid, "rep": rep, "scene": case["scene"], "instruction": case["instruction"],
            "expect": case["expect"], "model_output": out, "score": S.score_trial(case, out, ids), **best}


def _short(x):
    return {"goal": f"{x['pred']}({', '.join(x['args'])})", "ok": x["ok"], "why": x["why"]}


def auto_selection() -> list[tuple[str, str, int]]:
    """First repeat of each distinct wrong compilation per (arm, case), arms A and B."""
    sel, seen = [], set()
    scored = [json.loads(line) for line in (RESULTS / "scored.jsonl").read_text().splitlines()]
    for r in scored:
        if r["arm"] not in ("A", "B") or r["score"]["pass"] or r["score"]["primary"] in ("call_failed", "vocab_gap"):
            continue
        key = (r["arm"], r["case"], json.dumps(sorted(json.dumps(g, sort_keys=True) for g in r["output"].get("goals", []))))
        if key in seen:
            continue
        seen.add(key)
        sel.append((r["arm"], r["case"], r["rep"]))
    return sel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="*", help="arm:case:rep")
    ap.add_argument("--auto", action="store_true")
    args = ap.parse_args()
    items = auto_selection() if args.auto else [(a, c, int(r)) for a, c, r in (i.split(":") for i in args.items)]
    results = []
    for a, c, r in items:
        d = demo(a, c, r)
        results.append(d)
        flag = "CERTIFIED WRONG" if d["product_success"] and d["intent_success"] is False else ""
        print(f"{a}:{c}:{r} product={d['product_success']} intent={d['intent_success']} {flag}", flush=True)
    (RESULTS / ("consequence_auto.json" if args.auto else "consequence.json")).write_text(
        json.dumps(results, indent=1, ensure_ascii=False))
