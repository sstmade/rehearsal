"""Fill the {{V7_*}} slots in README.md and the submission text from the eval_v7 results.

eval_v7 writes its results outside the repo, next to the film footage (footage/tables_sim/eval_v7/). First run
scripts/eval_v7/aggregate.py (it writes effort.json + outcomes.json there), then:

    .venv/bin/python scripts/fill_v7_slots.py                 # dry run: print every slot value, change nothing
    .venv/bin/python scripts/fill_v7_slots.py --write         # fill README.md, submission/devpost.md, devpost.zh.md
    .venv/bin/python scripts/fill_v7_slots.py --eval-dir DIR  # results somewhere else

It refuses to write while any of the 50 tables is still "not_run" or "error" (pass --allow-partial to override,
and then say so in the text). Light: reads two JSON files, no simulation, no Meta API calls.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = ROOT.parent / "footage" / "tables_sim" / "eval_v7"
TARGETS = ["README.md", "submission/devpost.md", "submission/devpost.zh.md"]
SLOT = re.compile(r"\{\{(V7_[A-Z0-9_]+)\}\}")


def _n(x) -> str:
    return f"{x:,}" if isinstance(x, int) else str(x)


def slot_values(eval_dir: Path) -> tuple[dict[str, str], list[str]]:
    eff = json.loads((eval_dir / "effort.json").read_text())
    rows = json.loads((eval_dir / "outcomes.json").read_text())
    s = eff["summary"]
    c7 = collections.Counter(r["v7"] for r in rows)
    c6 = collections.Counter(r["v6"] for r in rows)
    pending = [r["slot"] for r in rows if r["v7"] in ("not_run", "error", "?")]
    reasons = s.get("rejected_by_reason") or {}
    reason_rows = "\n".join(f"| {k.replace('_', ' ')} | {_n(v)} |"
                            for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))
    top = max(reasons.items(), key=lambda kv: kv[1])[0].replace("_", " ") if reasons else "n/a"
    vals = {
        "V7_N_TABLES": len(rows),
        "V7_DONE": c7["done"],
        "V7_ALREADY": c7["already_done"],
        "V7_NOSAFE": c7["no_safe_plan"],
        "V7_MISSED": c7["missed"],
        "V7_ASKED": c7["asked"],
        "V7_REFUSED": c7["refused"],
        "V7_EMPTY": c7["empty"],
        "V7_PENDING": len(pending),
        "V7_DONE_DELTA": f"{c7['done'] - c6['done']:+d}",
        "V7_TASKS_REHEARSED": s["tasks_rehearsed"],
        "V7_ROLLOUTS": s["futures_simulated"],
        "V7_KIN_CHECKS": s["candidates_screened_kinematically"],
        "V7_REJECTED": s["rollouts_rejected"],
        "V7_REJECTED_REACH": s["rejected_in_dry_run_out_of_reach"],
        "V7_SEARCH_HOURS": s["search_wall_hours"],
        "V7_SIM_MINUTES": s["simulated_robot_minutes"],
        "V7_TOP_REASON": top,
        "V7_REASON_ROWS": reason_rows or "| (none recorded) | 0 |",
    }
    return {k: _n(v) for k, v in vals.items()}, pending


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--write", action="store_true", help="replace the slots in place (default: dry run)")
    ap.add_argument("--allow-partial", action="store_true", help="write even if some tables are not run yet")
    a = ap.parse_args()
    if not (a.eval_dir / "effort.json").exists():
        print(f"no {a.eval_dir}/effort.json yet: run `.venv/bin/python scripts/eval_v7/aggregate.py` first")
        return 2
    vals, pending = slot_values(a.eval_dir)
    for k, v in vals.items():
        print(f"{{{{{k}}}}} = {v if k != 'V7_REASON_ROWS' else chr(10) + v}")
    if pending:
        print(f"\n{len(pending)} table(s) not finished: {' '.join(pending)}")
        if a.write and not a.allow_partial:
            print("refusing to write partial results (use --allow-partial and disclose it)")
            return 3
    missing = set()
    for t in TARGETS:
        p = ROOT / t
        if not p.exists():
            continue
        text = p.read_text()
        used = set(SLOT.findall(text))
        missing |= used - vals.keys()
        if a.write and used:
            new = SLOT.sub(lambda m: vals.get(m.group(1), m.group(0)), text)
            p.write_text(new)
            print(f"filled {len(used & vals.keys())} slot kinds in {t}")
        elif used:
            print(f"{t}: {len(used)} slot kinds would be filled")
    if missing:
        print(f"unknown slots (left as is): {sorted(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
