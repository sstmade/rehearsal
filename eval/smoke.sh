#!/usr/bin/env bash
# Regression smoke test (~2 min, $0): pytest (minus the slow precompute test) + a tiny offline eval on one
# task (2 feedback conditions, 4-seed stress test) with assertions on the closed-loop outcome.
set -euo pipefail
cd "$(dirname "$0")/.."
export REHEARSAL_OFFLINE=1
export REHEARSAL_WORKERS="${REHEARSAL_WORKERS:-3}"
t0=$(date +%s)
# test_precompute runs a full rehearsal (slow on a loaded machine); FULL=1 includes it.
# A pytest failure does not stop the eval part; the script still exits non-zero at the end.
set +e
if [ "${FULL:-0}" = "1" ]; then .venv/bin/python -m pytest -q "$@"
else .venv/bin/python -m pytest -q --deselect tests/test_precompute.py::test_precompute_runs_only_demo_scenes "$@"; fi
pytest_rc=$?
set -e
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
echo "smoke total $(( $(date +%s) - t0 )) s (pytest exit $pytest_rc)"
exit $pytest_rc
