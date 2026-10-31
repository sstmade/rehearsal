"""Evaluation v3 analysis: first-try success and the paired log ablation, from run_eval result files.

    .venv/bin/python eval/round3_analyze.py --arm main=eval/results/eval_X_round3_main.json \
        --arm kitchen_zh=... --out eval/results/round3_summary.json

Definitions (all scored on the scene's hand-verified reference goals + implicit `undisturbed`):
- first try succeeded  = attempt 1 ran without an execution error and every reference goal holds.
- failed first plan    = anything else. The SAME attempt 1 is then revised under each feedback condition.
- recovered            = the last attempt (<= 2 revisions; the loop stops when the agent's own goals pass)
                         satisfies the reference goals.
Paired test: exact two-sided binomial (McNemar) on discordant pairs, full_log vs none.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import wilson  # noqa: E402

CONDS = ("full_log", "outcome_only", "none")


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def groups(cells):
    g = {}
    for c in cells:
        if "condition" not in c or c.get("status", "ok") != "ok" and not c.get("attempts"):
            continue
        g.setdefault((c["scene"], c["instruction"], c["repeat"]), {})[c["condition"]] = c
    return g


def ci(k, n):
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci95": [round(lo, 3), round(hi, 3)]}


def analyze(cells):
    out = {"per_scene": {}, "pooled": {}}
    pooled = {"first_ok": 0, "n": 0, "failed": 0, **{c: 0 for c in CONDS}, **{c + "_at2": 0 for c in CONDS},
              "pairs": {"full_only": 0, "none_only": 0, "oc_full_only": 0, "oc_oc_only": 0}, "incomplete": 0}
    for (scene, ins, rep), cs in sorted(groups(cells).items()):
        d = out["per_scene"].setdefault(scene, {"instruction": ins, "n": 0, "first_ok": 0, "failed": 0,
                                                **{c: 0 for c in CONDS}, **{c + "_at2": 0 for c in CONDS},
                                                "pairs": {"full_only": 0, "none_only": 0,
                                                          "oc_full_only": 0, "oc_oc_only": 0},
                                                "incomplete": 0, "agent_goal_pass_ref_fail_first": 0,
                                                "first_failure_types": {}})
        any_c = next(iter(cs.values()))
        d["n"] += 1
        pooled["n"] += 1
        first_ok = bool(any_c["first_try_ref_success"])
        if first_ok:
            d["first_ok"] += 1
            pooled["first_ok"] += 1
            continue
        if not all(c in cs and cs[c].get("status") == "ok" for c in CONDS):
            d["incomplete"] += 1
            pooled["incomplete"] += 1
            continue
        d["failed"] += 1
        pooled["failed"] += 1
        if any_c["first_try_success"]:
            d["agent_goal_pass_ref_fail_first"] += 1
        for t in any_c["failure_types_first"]:
            d["first_failure_types"][t] = d["first_failure_types"].get(t, 0) + 1
        rec = {c: bool(cs[c]["ref_success_within_2_revisions"]) for c in CONDS}
        at2 = {c: len(cs[c]["attempts"]) >= 2 and bool(cs[c]["attempts"][1]["ref_success"]) for c in CONDS}
        for c in CONDS:
            for tgt in (d, pooled):
                tgt[c] += rec[c]
                tgt[c + "_at2"] += at2[c]
        for tgt in (d["pairs"], pooled["pairs"]):
            tgt["full_only"] += rec["full_log"] and not rec["none"]
            tgt["none_only"] += rec["none"] and not rec["full_log"]
            tgt["oc_full_only"] += rec["full_log"] and not rec["outcome_only"]
            tgt["oc_oc_only"] += rec["outcome_only"] and not rec["full_log"]

    def fin(d):
        r = {"n_first_plans": d["n"], "first_try_success": ci(d["first_ok"], d["n"]),
             "first_try_failed": ci(d["n"] - d["first_ok"], d["n"]),
             "failed_first_plans_revised": d["failed"], "incomplete": d["incomplete"]}
        for c in CONDS:
            r[c] = ci(d[c], d["failed"])
            r[c + "_after_1_revision"] = ci(d[c + "_at2"], d["failed"])
        p = d["pairs"]
        r["full_vs_none"] = {"full_only": p["full_only"], "none_only": p["none_only"],
                             "p_exact_mcnemar": round(mcnemar_exact(p["full_only"], p["none_only"]), 6)}
        r["full_vs_outcome_only"] = {"full_only": p["oc_full_only"], "outcome_only_only": p["oc_oc_only"],
                                     "p_exact_mcnemar": round(mcnemar_exact(p["oc_full_only"], p["oc_oc_only"]), 6)}
        for k in ("instruction", "agent_goal_pass_ref_fail_first", "first_failure_types"):
            if k in d:
                r[k] = d[k]
        return r

    out["per_scene"] = {s: fin(d) for s, d in out["per_scene"].items()}
    out["pooled"] = fin(pooled)
    return out


def fmt(c):
    return f"{c['k']}/{c['n']} ({c['ci95'][0]*100:.0f}–{c['ci95'][1]*100:.0f}%)" if c["n"] else "–"


def table(res):
    lines = ["| scene | first try OK | failed first plans | full log | failed predicates only | \"it failed\" only | full vs none: discordant (p) |",
             "|---|---|---|---|---|---|---|"]
    for s, r in list(res["per_scene"].items()) + [("**pooled**", res["pooled"])]:
        fv = r["full_vs_none"]
        lines.append(f"| {s} | {fmt(r['first_try_success'])} | {r['failed_first_plans_revised']} | {fmt(r['full_log'])} | "
                     f"{fmt(r['outcome_only'])} | {fmt(r['none'])} | {fv['full_only']}:{fv['none_only']} (p={fv['p_exact_mcnemar']:.2g}) |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=file.json[,file2.json|partial.jsonl]")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = {"definitions": __doc__, "arms": {}}
    for a in args.arm:
        name, files = a.split("=", 1)
        cells, srcs = [], []
        for f in files.split(","):
            srcs.append(f)
            if f.endswith(".jsonl"):
                for line in Path(f).read_text().splitlines():
                    cells += json.loads(line)
            else:
                d = json.loads(Path(f).read_text())
                cells += d["cells"]
        r = analyze(cells)
        r["sources"] = srcs
        res["arms"][name] = r
        print(f"\n### {name}\n" + table(r))
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
