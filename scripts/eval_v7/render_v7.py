"""eval_v7 execution clips, same look as footage/tables_sim/hero/exec_dark/ (reuses scripts/exec_tables.render_clip
unchanged; visual only).  Reads the recorded v7 run from footage/tables_sim/eval_v7/_cache/, writes
footage/tables_sim/eval_v7/exec_dark/NN{.mp4,_labeled.mp4,_still.png,.json}.

Run (rehearsal-mvp/):  .venv/bin/python scripts/eval_v7/render_v7.py [--hq] NN ...
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
EV = ROOT.parent / "footage" / "tables_sim" / "eval_v7"


def main():
    args = sys.argv[1:]
    if "--hq" in args:
        os.environ["EXEC_HQ"] = "1"
        args.remove("--hq")
    os.environ.setdefault("DRESS_STYLE", "dark")
    import exec_tables as X
    X.OUT = EV / ("exec_" + os.environ["DRESS_STYLE"])
    X.CACHE = EV / "_cache"
    X.OUT.mkdir(parents=True, exist_ok=True)
    for nn in args:
        rec = json.loads((X.CACHE / f"{nn}_run.json").read_text())
        if not rec.get("planned") or not rec.get("success") or not rec["plan"]["actions"]:
            print(nn, "skip: not a verified plan with actions")
            continue
        t0 = time.time()
        rec["render"] = X.render_clip(nn, rec)
        (X.OUT / f"{nn}.json").write_text(json.dumps(rec, indent=1, default=float))
        print(nn, "rendered", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
