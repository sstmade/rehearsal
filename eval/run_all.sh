#!/usr/bin/env bash
# Whole evaluation in one command. Offline parts cost $0. The Muse Spark matrix is cached by request
# hash, so re-running it is free unless prompts/code changed; it stops at EVAL_BUDGET (default $2,
# counted over all ledger entries tagged eval.*).  Skip it with MUSE=0.
set -euo pipefail
cd "$(dirname "$0")/.."
MUSE="${MUSE:-1}"
REHEARSAL_WORKERS="${REHEARSAL_WORKERS:-3}" REHEARSAL_OFFLINE=1 .venv/bin/python eval/run_eval.py --agents offline --tag offline_main
if [ "$MUSE" = "1" ]; then
  REHEARSAL_WORKERS="${REHEARSAL_WORKERS:-3}" .venv/bin/python eval/run_eval.py --agents muse --only kitchen_01,windowsill_01 --repeats 10 --parallel 2 --tag muse_main
  REHEARSAL_WORKERS="${REHEARSAL_WORKERS:-3}" .venv/bin/python eval/run_eval.py --agents muse --only desk_01 --repeats 5 --parallel 2 --tag muse_main_desk
fi
REHEARSAL_WORKERS="${REHEARSAL_WORKERS:-3}" REHEARSAL_OFFLINE=1 .venv/bin/python eval/perception_curve.py
REHEARSAL_WORKERS="${REHEARSAL_WORKERS:-3}" REHEARSAL_OFFLINE=1 .venv/bin/python eval/active_perception_baseline.py
.venv/bin/python eval/render_eval.py
.venv/bin/python -m rehearsal.meta_client
