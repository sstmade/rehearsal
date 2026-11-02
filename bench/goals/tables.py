"""Markdown tables for docs/goalbench.md from results/raw_*.jsonl (no API calls).

    .venv/bin/python -m bench.goals.tables > bench/goals/results/tables.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.goals import score as S  # noqa: E402
from bench.goals.cases import CASES  # noqa: E402
from bench.goals.run import report, scored_rows  # noqa: E402

ARMS = ["A", "B", "C"]
NAMES = {"A": "A product prompt", "B": "B + vocabulary", "C": "C + assumptions/clarify"}


def main():
    out = report(ARMS)
    rows = {a: scored_rows(a) for a in ARMS}
    p = print
    p("### Overall (n = cases x repeats; Wilson 95% CI)\n")
    p("| arm | intent match | ignoring unexpressible intents | exact match |")
    p("|---|---|---|---|")
    for a in ARMS:
        d = out["arms"][a]
        p(f"| {NAMES[a]} | {S.fmt_rate(*d['overall'])} | {S.fmt_rate(*d['overall_expressible'])} | {S.fmt_rate(*d['exact'])} |")
    p("\nRepeats of one case are correlated (A and B are near-deterministic per case), so the CIs above are too")
    p("narrow. Case-level view (a case counts as passed when the majority of its repeats pass; n = 60 cases):\n")
    p("| arm | cases passed (majority of repeats) |")
    p("|---|---|")
    for a in ARMS:
        by = {}
        for r in rows[a]:
            by.setdefault(r["case"], []).append(r["score"]["pass"])
        k = sum(sum(v) * 2 > len(v) for v in by.values())
        p(f"| {NAMES[a]} | {S.fmt_rate(k, len(by))} |")
    cats = sorted({c["category"] for c in CASES})
    ncase = {cat: sum(c["category"] == cat for c in CASES) for cat in cats}
    p("\n### By category\n")
    p("| category (cases) | " + " | ".join(NAMES[a] for a in ARMS) + " |")
    p("|---|" + "---|" * len(ARMS))
    for cat in cats:
        p(f"| {cat} ({ncase[cat]}) | " + " | ".join(S.fmt_rate(*out["arms"][a]["by_category"].get(cat, (0, 0))) for a in ARMS) + " |")
    p("\n### By language (only the 13 cases that exist in both English and Chinese)\n")
    p("| language | " + " | ".join(NAMES[a] for a in ARMS) + " |")
    p("|---|" + "---|" * len(ARMS))
    for lang in ("en", "zh"):
        p(f"| {lang} | " + " | ".join(S.fmt_rate(*out["arms"][a]["paired_lang"].get(lang, (0, 0))) for a in ARMS) + " |")
    p("\n### Paired comparisons (same case, same repeat)\n")
    p("| comparison | both pass | only first | only second | neither | n | sign-test p |")
    p("|---|---|---|---|---|---|---|")
    for k, v in out["paired"].items():
        if k.endswith("category"):
            continue
        x, _, y = k.split("_")
        p(f"| {x} vs {y} | {v['both']} | {v['x_only']} | {v['y_only']} | {v['neither']} | {v['n']} | {v['p']:.2g} |")
    p("\n### C vs B by category (paired)\n")
    p("| category | C only | B only | p |")
    p("|---|---|---|---|")
    for cat, v in out["paired"].get("C_vs_B_by_category", {}).items():
        p(f"| {cat} | {v['x_only']} | {v['y_only']} | {v['p']:.2g} |")
    p("\n### Primary error per failed trial\n")
    errs = sorted({e for a in ARMS for e in out["arms"][a]["primary_errors"]}, key=S.ERROR_PRIORITY.index)
    p("| error | " + " | ".join(NAMES[a] for a in ARMS) + " |")
    p("|---|" + "---|" * len(ARMS))
    for e in errs:
        p(f"| {e} | " + " | ".join(str(out["arms"][a]["primary_errors"].get(e, 0)) for a in ARMS) + " |")
    p("\n### Per case (passes out of repeats)\n")
    p("| case | category | instruction | " + " | ".join(ARMS) + " |")
    p("|---|---|---|" + "---|" * len(ARMS))
    for c in CASES:
        cells = []
        for a in ARMS:
            rs = [r for r in rows[a] if r["case"] == c["id"]]
            k = sum(r["score"]["pass"] for r in rs)
            prim = sorted({r["score"]["primary"] for r in rs if r["score"]["primary"]})
            cells.append(f"{k}/{len(rs)}" + (f" {','.join(prim)}" if prim else ""))
        p(f"| {c['id']} | {c['category']} | {c['instruction']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
