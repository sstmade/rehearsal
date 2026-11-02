"""Score the deterministic intent gate (rehearsal/intent_gate.py, no LLM) on stored GoalBench data ($0), and the
self-consistency "ask if the 3 repeats disagree" trigger.

1. The 55 distinct wrong compilations replayed by `bench.goals.consequence --auto` (arms A and B): which does the
   gate flag, in particular the 17 that physics certified as success while the user's intent did not hold?
2. False flags: the gate on every distinct correct compilation (all arms, cases whose answer is goals).
3. Every stored trial with the gate appended ("A+gate", "B+gate", "C+gate"), scored with the GoalBench scorer.
   A+gate is the dumb baseline for arm C: same legacy prompt, plus a deterministic check.
4. Self-consistency trigger on arms A and B: ask iff the 3 repeats' goal sets differ. Precision/recall against
   "the case should be flagged" (expect flag/either) and against "the majority compilation is wrong".

    .venv/bin/python -m bench.goals.gate_eval      # writes results/gate_eval.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.goals import score as S  # noqa: E402
from bench.goals.cases import CASES  # noqa: E402
from bench.goals.run import ARMS, RESULTS, load_raw, load_scenes  # noqa: E402
from rehearsal import intent_gate  # noqa: E402

BY_ID = {c["id"]: c for c in CASES}
_gate_cache: dict = {}


def gate(goals, scene_name, scenes) -> dict:
    key = (scene_name, json.dumps(goals, sort_keys=True))
    if key not in _gate_cache:
        g = intent_gate.check(goals, scenes[scene_name])
        _gate_cache[key] = {"verdict": g["verdict"], "codes": sorted({f["code"] for f in g["flags"]})}
    return _gate_cache[key]


def gated_output(out, scene_name, scenes) -> dict:
    if "error" in out or out.get("status", "ok") != "ok":
        return out
    g = gate(out.get("goals") or [], scene_name, scenes)
    return out if g["verdict"] == "pass" else dict(out, status=g["verdict"])


def canon(out, ids=None, placements_only=False) -> str:
    """The goal set as the product would verify it (stacked expanded, implicit undisturbed added, order ignored),
    plus the status. Two repeats "agree" iff these are equal."""
    if "error" in out:
        return "error"
    if out.get("status", "ok") != "ok":
        return out["status"]
    atoms = S.atoms_of(out.get("goals") or [], ids) if ids is not None else \
        {json.dumps(g, sort_keys=True) for g in out.get("goals") or []}
    if placements_only:
        atoms = {a for a in atoms if a[0] in ("on", "inside", "near")}
    return json.dumps(sorted(map(str, atoms)))


def main() -> dict:
    scenes = load_scenes()
    res = {}

    # 1. the 55 wrong compilations
    cons = json.loads((RESULTS / "consequence_auto.json").read_text())
    rows = []
    for d in cons:
        g = gate(d["model_output"].get("goals") or [], d["scene"], scenes)
        rows.append({"arm": d["arm"], "case": d["case"], "rep": d["rep"], "expect": d["expect"],
                     "certified_wrong": bool(d["product_success"] and d["intent_success"] is False),
                     "physics_failed": not d["product_success"], "primary": d["score"]["primary"],
                     "gate": g["verdict"], "codes": g["codes"]})
    cw = [r for r in rows if r["certified_wrong"]]
    res["wrong_compilations"] = {
        "n": len(rows), "flagged": sum(r["gate"] != "pass" for r in rows),
        "certified_wrong": len(cw), "certified_wrong_flagged": sum(r["gate"] != "pass" for r in cw),
        "certified_wrong_missed": [f"{r['arm']}:{r['case']} ({r['primary']})" for r in cw if r["gate"] == "pass"],
        "certified_wrong_caught": [f"{r['arm']}:{r['case']} {r['codes']}" for r in cw if r["gate"] != "pass"],
        "physics_failed": sum(r["physics_failed"] for r in rows),
        "physics_failed_flagged": sum(r["physics_failed"] and r["gate"] != "pass" for r in rows),
        "rows": rows}
    # what arm C did on the same (case) set, all 3 repeats
    scored = [json.loads(line) for line in (RESULTS / "scored.jsonl").read_text().splitlines()]
    c_by_case = defaultdict(list)
    for r in scored:
        if r["arm"] == "C":
            c_by_case[r["case"]].append(r["score"]["pass"])
    res["wrong_compilations"]["arm_C_on_same_cases"] = {
        "certified_wrong": [sum(sum(c_by_case[r["case"]]) * 2 > len(c_by_case[r["case"]]) for r in cw), len(cw)],
        "all": [sum(sum(c_by_case[r["case"]]) * 2 > len(c_by_case[r["case"]]) for r in rows), len(rows)]}

    # 2. false flags on distinct correct compilations
    seen, ff = set(), []
    for r in scored:
        c = BY_ID[r["case"]]
        out = r["output"]
        if not r["score"]["pass"] or "error" in out or out.get("status", "ok") != "ok" or S.is_flagged(out):
            continue
        key = (r["case"], canon(out))
        if key in seen:
            continue
        seen.add(key)
        g = gate(out.get("goals") or [], c["scene"], scenes)
        ff.append({"arm": r["arm"], "case": r["case"], "gate": g["verdict"], "codes": g["codes"]})
    res["false_flags"] = {"n_correct_distinct": len(ff), "flagged": sum(x["gate"] != "pass" for x in ff),
                          "which": [x for x in ff if x["gate"] != "pass"]}

    # 3. every trial, with the gate appended
    ids_of = {s: [o["id"] for o in sc["objects"]] for s, sc in scenes.items()}
    res["systems"] = {}
    for arm in ("A", "B", "C"):
        raw = load_raw(arm)
        for name, use_gate in ((arm, False), (arm + "+gate", True)):
            trows = []
            for (cid, rep), r in sorted(raw.items()):
                c = BY_ID[cid]
                out = gated_output(r["output"], c["scene"], scenes) if use_gate else r["output"]
                trows.append({"case": cid, "rep": rep, "category": c["category"],
                              "score": S.score_trial(c, out, ids_of[c["scene"]], ARMS[arm][3])})
            maj = defaultdict(list)
            for t in trows:
                maj[t["case"]].append(t["score"]["pass"])
            cases_ok = {k: sum(v) * 2 > len(v) for k, v in maj.items()}
            res["systems"][name] = {
                # silent_wrong: the system went ahead (no clarify/refuse) with a wrong goal set -- what physics
                # would then certify or fail without telling the user the goal was wrong.
                "silent_wrong": sum(not t["score"]["pass"] and not t["score"]["flagged"] for t in trows),
                "flagged": sum(t["score"]["flagged"] for t in trows),
                "gate_flagged": ([f"{t['case']}.r{t['rep']}" for t, (k, r) in zip(trows, sorted(raw.items()))
                                  if use_gate and t["score"]["flagged"] and not S.is_flagged(r["output"])]),
                "trials": [sum(t["score"]["pass"] for t in trows), len(trows)],
                "cases": [sum(cases_ok.values()), len(cases_ok)],
                "by_category": S.aggregate(trows, key=lambda t: t["category"]),
                "primary_errors": dict(S.error_counts(trows))}

    # 4. self-consistency trigger
    res["self_consistency"] = {}
    for arm, placements_only in (("A", False), ("B", False), ("C", False),
                                 ("A/placements", True), ("B/placements", True), ("C/placements", True)):
        arm_name, arm = arm, arm.split("/")[0]
        raw = load_raw(arm)
        by = defaultdict(list)
        for (cid, rep), r in raw.items():
            by[cid].append(r["output"])
        sc_rows = []
        for cid, outs in by.items():
            c = BY_ID[cid]
            ids = ids_of[c["scene"]]
            disagree = len({canon(o, ids, placements_only) for o in outs}) > 1
            passes = [S.score_trial(c, o, ids, ARMS[arm][3])["pass"] for o in outs]
            sc_rows.append({"case": cid, "category": c["category"], "expect": c["expect"], "ask": disagree,
                            "should_flag": c["expect"] in ("flag", "either"),
                            "majority_wrong": sum(passes) * 2 < len(passes), "any_wrong": not all(passes)})
        out = {}
        for label in ("should_flag", "majority_wrong", "any_wrong"):
            tp = sum(r["ask"] and r[label] for r in sc_rows)
            fp = sum(r["ask"] and not r[label] for r in sc_rows)
            fn = sum(not r["ask"] and r[label] for r in sc_rows)
            out[label] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                          "recall": round(tp / (tp + fn), 3) if tp + fn else None}
        out["asks"] = sum(r["ask"] for r in sc_rows)
        out["n_cases"] = len(sc_rows)
        out["asked_cases"] = sorted(r["case"] for r in sc_rows if r["ask"])
        # complementary trigger: ask on disagreement OR gate flag on the majority output (A/B only)
        res["self_consistency"][arm_name] = out
    (RESULTS / "gate_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def print_summary(res):
    w = res["wrong_compilations"]
    print(f"wrong compilations: gate flags {w['flagged']}/{w['n']}; certified-wrong flagged "
          f"{w['certified_wrong_flagged']}/{w['certified_wrong']}; physics-failed flagged "
          f"{w['physics_failed_flagged']}/{w['physics_failed']}")
    print("  caught:", w["certified_wrong_caught"])
    print("  missed:", w["certified_wrong_missed"])
    print("  arm C on the same cases (majority):", w["arm_C_on_same_cases"])
    f = res["false_flags"]
    print(f"false flags on distinct correct compilations: {f['flagged']}/{f['n_correct_distinct']}", f["which"])
    for k, v in res["systems"].items():
        print(f"{k:7s} trials {S.fmt_rate(*v['trials'])} cases {S.fmt_rate(*v['cases'])} silent_wrong {v['silent_wrong']}"
              f" flagged {v['flagged']} errors {v['primary_errors']}")
        if v["gate_flagged"]:
            print("        gate flagged:", v["gate_flagged"])
    for arm, v in res["self_consistency"].items():
        print(f"self-consistency {arm}: asks {v['asks']}/{v['n_cases']} {v['asked_cases']}")
        for k in ("should_flag", "majority_wrong", "any_wrong"):
            print(f"   vs {k}: {v[k]}")


if __name__ == "__main__":
    print_summary(main())
