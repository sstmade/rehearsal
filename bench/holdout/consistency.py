"""Self-consistency "ask if the 3 repeats disagree" trigger on the stored hold-out runs ($0).
    .venv/bin/python -m bench.holdout.consistency"""
from __future__ import annotations

import json
from collections import defaultdict

from bench.goals.gate_eval import canon
from bench.goals.run import load_scenes
from bench.holdout.run import CASES, RESULTS, load_raw


def main():
    scenes = load_scenes()
    summ = json.loads((RESULTS / "summary.json").read_text())
    out = {}
    for arm in ("A", "C"):
        for placements_only in (False, True):
            by = defaultdict(list)
            for (cid, rep), r in load_raw(arm).items():
                by[cid].append(r["output"])
            rows = []
            for c in CASES:
                ids = [o["id"] for o in scenes[c["scene"]]["objects"]]
                ask = len({canon(o, ids, placements_only) for o in by[c["id"]]}) > 1
                passes = summ["systems"][arm]["per_case"][c["id"]]
                rows.append({"case": c["id"], "ask": ask, "should_flag": c["expected_status"] != "goals",
                             "majority_wrong": sum(passes) * 2 < len(passes)})
            res = {"asks": sorted(r["case"] for r in rows if r["ask"])}
            for label in ("should_flag", "majority_wrong"):
                tp = sum(r["ask"] and r[label] for r in rows)
                fp = sum(r["ask"] and not r[label] for r in rows)
                fn = sum(not r["ask"] and r[label] for r in rows)
                res[label] = {"tp": tp, "fp": fp, "fn": fn}
            out[f"{arm}{'/placements' if placements_only else ''}"] = res
    (RESULTS / "consistency.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
