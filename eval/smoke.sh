#!/usr/bin/env bash
# Regression smoke test (< 2 min, $0): pytest + a tiny offline eval on one task, no stress test.
set -euo pipefail
cd "$(dirname "$0")/.."
export REHEARSAL_OFFLINE=1
t0=$(date +%s)
.venv/bin/python -m pytest -q -x "$@"
.venv/bin/python eval/run_eval.py --agents offline --only desk_01 --max-tasks 1 --conditions full_log,none \
    --stress 4 --tag smoke | tail -4
# the closed loop must still recover on desk_01 with the full log
f=$(ls -t eval/results/eval_*_smoke.json | head -1)
.venv/bin/python - "$f" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
row = {a["condition"]: a for a in r["aggregate"]}
assert row["full_log"]["within_2_rev"] == 1, row["full_log"]
assert row["full_log"]["first_try"] == 0, "desk_01 offline should fail first (transit too low)"
print("smoke ok:", {k: (v["first_try"], v["within_2_rev"]) for k, v in row.items()})
PY
rm -f eval/results/eval_*_smoke.json eval/results/eval_*_smoke.csv
echo "smoke total $(( $(date +%s) - t0 )) s"
