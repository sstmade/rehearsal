"""Print where each eval_v7 search got stuck: deepest verified prefix, rejection reasons at the next action, and the
most common physics evidence there (used to write eval_v7/DIAGNOSIS.md)."""
import collections
import json
import re
import sys
from pathlib import Path

EV = Path(__file__).resolve().parents[3] / "footage" / "tables_sim" / "eval_v7"


def norm(d):
    d = re.sub(r"-?\d+\.\d+", "#", d or "")
    return re.sub(r"\[[^\]]*\]", "[..]", d)[:170]


def main(keys):
    for k in keys or sorted(p.stem for p in (EV / "runs").glob("*.json")):
        r = json.loads((EV / "runs" / f"{k}.json").read_text())
        tf = EV / "trace" / f"{k}.json"
        tr = json.loads(tf.read_text()) if tf.exists() else []
        s = r.get("search") or {}
        print(f"===== {k} {r.get('outcome')} | {r.get('instruction')} | sims {len(tr)} | {s.get('reason')} | "
              f"wall {s.get('wall_s')}")
        if r.get("error"):
            print("   ERROR", r["error"])
        okd = [t["depth"] for t in tr if t["ok"]]
        print("   deepest verified prefix:", max(okd) if okd else 0, "of", len({(t['last'] or {}).get('obj') for t in tr}))
        by = collections.defaultdict(collections.Counter)
        ev = collections.defaultdict(collections.Counter)
        for t in tr:
            if not t["ok"]:
                by[t["depth"]][t["why"]] += 1
                ev[t["depth"]][f"{(t['last'] or {}).get('obj')}: " + norm(t["detail"])] += 1
        for dpt in sorted(by):
            print(f"   depth {dpt}: {dict(by[dpt])}")
            for e, n in ev[dpt].most_common(3):
                print(f"      {n}x {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
