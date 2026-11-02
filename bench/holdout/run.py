"""Run and score the blind GoalBench hold-out (bench/holdout/cases.json, 30 cases, authored without seeing the dev
cases or the prompts) on arm A (legacy compile prompt) and arm C (the shipped compile step), plus the deterministic
intent gate on top of each.

Requests are byte-identical to the product's (rehearsal/agents.py COMPILE_PROMPT_LEGACY / COMPILE_PROMPT, the
product's scene summary), with the inert "[eval trial k]" tag on repeats k > 0, exactly as bench/goals/run.py does.
Callers: holdout.<arm>.<case>.r<k>, Contributor mode (all scenes are public or synthetic).

The scoring rules below were written before any hold-out output was seen. Prompts are frozen: nothing in
rehearsal/agents.py was changed after the hold-out run.

    .venv/bin/python -m bench.holdout.run --budget 0.30          # calls (cached on re-run)
    .venv/bin/python -m bench.holdout.run --score-only
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.goals import score as S  # noqa: E402
from bench.goals.run import load_scenes  # noqa: E402
from rehearsal import agents as A  # noqa: E402
from rehearsal import meta_client as M  # noqa: E402
from rehearsal.predicates import with_implicit_goals  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CASES = json.loads((HERE / "cases.json").read_text())
CALLER = "holdout."
ARMS = {"A": (A.COMPILE_PROMPT_LEGACY, A.GOAL_SCHEMA, "Goals"),
        "C": (A.COMPILE_PROMPT, A.INTENT_SCHEMA, "GoalsConfirm")}
PLACE = ("on", "inside", "near")
CHANGE = set(PLACE) | {"stacked", "upright", "stable"}


# ------------------------------------------------------------------------------------------------ calls
def summaries(scenes):
    from rehearsal.executor import reachability_report
    cache = RESULTS / "reach.json"
    reach = json.loads(cache.read_text()) if cache.exists() else {}
    for s in sorted({c["scene"] for c in CASES} - set(reach)):
        reach[s] = reachability_report(scenes[s])
    RESULTS.mkdir(exist_ok=True)
    cache.write_text(json.dumps(reach, indent=1))
    return {s: A.scene_summary(scenes[s], reach[s]) for s in {c["scene"] for c in CASES}}


def user_prompt(arm, case, summary, rep):
    return ARMS[arm][0].format(summary=json.dumps(summary), instruction=case["instruction"]) + (
        f"\n\n[eval trial {rep}]" if rep else "")


def spend() -> float:
    p = M.budget_dir() / "ledger.jsonl"
    tot = 0.0
    for line in (p.read_text().splitlines() if p.exists() else []):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(e.get("caller", "")).startswith(CALLER):
            tot += e.get("cost", 0.0)
    return tot


def raw_path(arm):
    return RESULTS / f"raw_{arm}.jsonl"


def load_raw(arm):
    out = {}
    if raw_path(arm).exists():
        for line in raw_path(arm).read_text().splitlines():
            r = json.loads(line)
            out[(r["case"], r["rep"])] = r
    return out


_lock = threading.Lock()


def run(arms, reps, budget, workers=3):
    scenes = load_scenes()
    summ = summaries(scenes)
    todo = [(arm, c, rep) for arm in arms for rep in range(reps) for c in CASES
            if (c["id"], rep) not in {k for k, v in load_raw(arm).items() if "error" not in v["output"]}]
    print(f"{len(todo)} calls; holdout spend so far ${spend():.4f}", flush=True)

    def work(item):
        arm, c, rep = item
        if spend() >= budget:
            return "stop: budget"
        _, schema, name = ARMS[arm]
        msgs = [{"role": "system", "content": A.SYSTEM},
                {"role": "user", "content": user_prompt(arm, c, summ[c["scene"]], rep)}]
        try:
            out = M.chat_json(msgs, schema, name, caller=f"{CALLER}{arm}.{c['id']}.r{rep}", data="public")
        except M.BudgetExceeded as e:
            return f"stop: {e}"
        except Exception as e:  # noqa: BLE001
            out = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        with _lock, open(raw_path(arm), "a") as f:
            f.write(json.dumps({"arm": arm, "case": c["id"], "rep": rep, "output": out}, ensure_ascii=False) + "\n")
        return "ok"

    RESULTS.mkdir(exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(work, t) for t in todo])):
            r = f.result()
            if r != "ok":
                print(r, flush=True)
            if i % 20 == 0:
                print(f"{i}/{len(todo)} ${spend():.4f}", flush=True)
    print(f"done; holdout spend ${spend():.4f}")


# ------------------------------------------------------------------------------------------------ scoring
def expand(goals) -> set[tuple]:
    out = set()
    for g in goals:
        p, a = g["pred"], tuple(g.get("args") or [])
        if p == "stacked":
            out |= {("on", (u, lo)) for lo, u in zip(a, a[1:])}
        elif p == "undisturbed":
            out |= {("undisturbed", (x,)) for x in a}
        else:
            out.add((p, a))
    return out


def placement_order(goals) -> list[str]:
    seq = []
    for g in goals:
        a = g.get("args") or []
        if g["pred"] in PLACE and a:
            seq.append(a[0])
        elif g["pred"] == "stacked":
            seq += a[1:]
    return list(dict.fromkeys(seq))


def system_status(arm_output: dict) -> str:
    """goals | clarify | refuse. Arm A has no status: a goal set that asks for no change counts as a refusal
    (that is what the product's intent gate now does with it)."""
    if "error" in arm_output:
        return "error"
    st = arm_output.get("status", "ok")
    if st != "ok":
        return st
    return "goals" if [g for g in arm_output.get("goals") or [] if g.get("pred") in CHANGE] else "refuse"


def score_goals(case, goals, ids, protect_mentioned=True) -> tuple[bool, list[str]]:
    """Pass iff, for some acceptable reading: every required atom is in the verified goal set (compiled goals +
    the product's implicit undisturbed); no forbidden atom; no object moved that the reading does not move; and,
    when order_matters, the reading's moves appear in its order."""
    halluc = sorted({a for g in goals for a in g.get("args") or [] if a not in ids})
    if halluc:
        return False, ["hallucinated_object"]
    verified = expand(with_implicit_goals([dict(g) for g in goals], ids, protect_mentioned=protect_mentioned))
    forbidden = expand(case.get("forbidden", []))
    best = None
    for reading in [case["required"]] + case.get("acceptable_alternatives", []):
        errs = []
        req = expand(reading)
        miss = sorted(req - verified)
        for p, a in miss:
            errs.append("dropped_protection" if p == "undisturbed" else
                        "missed_implicit" if p in ("upright", "stable", "clear") else "missed_goal")
        if verified & forbidden:
            errs.append("forbidden_goal")
        moved_ref = {a[0] for p, a in req if p in PLACE}
        extra = sorted({a[0] for p, a in verified if p in PLACE and a and a[0] not in moved_ref})
        if extra:
            errs.append("extra_action")
        if case.get("order_matters") and not errs:
            want = placement_order(reading)
            got = [x for x in placement_order(goals) if x in want]
            if got != want:
                errs.append("wrong_order")
        if best is None or len(errs) < len(best):
            best = errs
    return not best, best


def score_trial(case, output, ids, protect_mentioned=True) -> dict:
    st = system_status(output)
    exp = case["expected_status"]
    if st == "error":
        return {"status": st, "pass": False, "flag_ok": False, "errors": ["call_failed"]}
    if exp == "goals":
        if st != "goals":
            return {"status": st, "pass": False, "flag_ok": False, "errors": ["unnecessary_flag"]}
        ok, errs = score_goals(case, output.get("goals") or [], ids, protect_mentioned)
        return {"status": st, "pass": ok, "flag_ok": ok, "errors": errs}
    ok = st == exp
    flagged = st in ("clarify", "refuse")
    return {"status": st, "pass": ok, "flag_ok": flagged,
            "errors": [] if ok else (["wrong_flag_kind"] if flagged else ["failed_to_flag"])}


def with_gate(output, scene) -> dict:
    """The product's pipeline: model output, then the deterministic gate on any goal set it returns."""
    from rehearsal import intent_gate
    if "error" in output or output.get("status", "ok") != "ok":
        return output
    gate = intent_gate.check(output.get("goals") or [], scene)
    if gate["verdict"] == "pass":
        return output
    return dict(output, status=gate["verdict"], question=intent_gate.summarize(gate), gate_flags=gate["flags"])


def gate_only(output, scene) -> dict:
    """Gate verdict alone on the model's goals, for a gate-vs-model comparison."""
    from rehearsal import intent_gate
    gate = intent_gate.check(output.get("goals") or [], scene)
    return {"verdict": gate["verdict"], "codes": [f["code"] for f in gate["flags"]]}


def wilson_fmt(k, n):
    return S.fmt_rate(k, n)


def report(arms=("A", "C")) -> dict:
    scenes = load_scenes()
    byid = {c["id"]: c for c in CASES}
    systems = {}
    gate_cache = {}
    for arm in arms:
        raw = load_raw(arm)
        for sysname, gated in ((arm, False), (arm + "+gate", True)):
            rows = []
            for (cid, rep), r in sorted(raw.items()):
                c = byid[cid]
                ids = [o["id"] for o in scenes[c["scene"]]["objects"]]
                out = r["output"]
                if gated:
                    key = (arm, cid, rep)
                    if key not in gate_cache:
                        gate_cache[key] = with_gate(out, scenes[c["scene"]])
                    out = gate_cache[key]
                rows.append({"case": cid, "rep": rep, "category": c["category"], "lang": c["lang"],
                             "expected": c["expected_status"], "score": score_trial(c, out, ids),
                             "score_legacy_protection": score_trial(c, out, ids, protect_mentioned=False)})
            systems[sysname] = rows
    summary = {"n_cases": len(CASES), "systems": {}}
    for name, rows in systems.items():
        summary["systems"][name] = summarize(rows)
    if "A" in systems and "C" in systems:
        summary["paired_C_vs_A"] = paired_cases(systems["C"], systems["A"])
        summary["paired_C_vs_A_trials"] = paired_trials(systems["C"], systems["A"])
        summary["paired_Cgate_vs_C"] = paired_cases(systems["C+gate"], systems["C"])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    with open(RESULTS / "scored.jsonl", "w") as f:
        for name, rows in systems.items():
            for r in rows:
                f.write(json.dumps({"system": name, **r}, ensure_ascii=False) + "\n")
    return summary


def case_majority(rows, field="pass", legacy=False) -> dict:
    by = defaultdict(list)
    for r in rows:
        s = r["score_legacy_protection"] if legacy else r["score"]
        by[r["case"]].append(s[field])
    return {c: sum(v) * 2 > len(v) for c, v in by.items()}


def summarize(rows) -> dict:
    byid = {c["id"]: c for c in CASES}
    maj = case_majority(rows)
    out = {"trials": [sum(r["score"]["pass"] for r in rows), len(rows)],
           "cases": [sum(maj.values()), len(maj)],
           "cases_legacy_protection": [sum(case_majority(rows, legacy=True).values()), len(maj)],
           "cases_flag_any": [sum(case_majority(rows, "flag_ok").values()), len(maj)],
           "by_category": {}, "by_lang": {}, "by_expected": {}, "errors": defaultdict(int), "status_counts": {}}
    for key, field in (("by_category", "category"), ("by_lang", "lang"), ("by_expected", "expected_status")):
        acc = defaultdict(lambda: [0, 0])
        for cid, ok in maj.items():
            acc[byid[cid][field]][0] += ok
            acc[byid[cid][field]][1] += 1
        out[key] = {k: v for k, v in sorted(acc.items())}
    for r in rows:
        for e in r["score"]["errors"]:
            out["errors"][e] += 1
        k = f"{r['expected']}->{r['score']['status']}"
        out["status_counts"][k] = out["status_counts"].get(k, 0) + 1
    out["errors"] = dict(out["errors"])
    out["per_case"] = {cid: [r["score"]["pass"] for r in rows if r["case"] == cid] for cid in maj}
    return out


def paired_cases(x_rows, y_rows) -> dict:
    x, y = case_majority(x_rows), case_majority(y_rows)
    common = sorted(set(x) & set(y))
    xo = sum(x[c] and not y[c] for c in common)
    yo = sum(y[c] and not x[c] for c in common)
    return {"n": len(common), "x_only": xo, "y_only": yo, "both": sum(x[c] and y[c] for c in common),
            "neither": sum(not x[c] and not y[c] for c in common), "p": S.sign_test(xo, yo)}


def paired_trials(x_rows, y_rows) -> dict:
    y = {(r["case"], r["rep"]): r["score"]["pass"] for r in y_rows}
    pairs = [(r["score"]["pass"], y[(r["case"], r["rep"])]) for r in x_rows if (r["case"], r["rep"]) in y]
    xo = sum(a and not b for a, b in pairs)
    yo = sum(b and not a for a, b in pairs)
    return {"n": len(pairs), "x_only": xo, "y_only": yo, "p": S.sign_test(xo, yo)}


def print_report(s):
    for name, d in s["systems"].items():
        print(f"\n== {name}: cases {wilson_fmt(*d['cases'])} | trials {wilson_fmt(*d['trials'])} | "
              f"legacy-protection cases {wilson_fmt(*d['cases_legacy_protection'])} | "
              f"flag-any cases {wilson_fmt(*d['cases_flag_any'])}")
        for k in ("by_category", "by_lang", "by_expected"):
            print("  ", k, {a: wilson_fmt(*b) for a, b in d[k].items()})
        print("   errors", d["errors"])
        print("   status", d["status_counts"])
    for k in ("paired_C_vs_A", "paired_C_vs_A_trials", "paired_Cgate_vs_C"):
        if k in s:
            print(k, s[k])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A", "C"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--budget", type=float, default=0.30)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--workers", type=int, default=3, help="concurrent API requests (threads)")
    args = ap.parse_args()
    if not args.score_only:
        from rehearsal import load_env
        load_env()
        run(args.arms, args.reps, args.budget, args.workers)
    print_report(report(args.arms))
