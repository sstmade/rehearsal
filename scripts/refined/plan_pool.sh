#!/bin/bash
# Run refined plans with at most 3 workers (one heavy.sh slot). Args: "key[:budget]" ...
cd "$(dirname "$0")/../.."
L=../footage/tables_sim/hero/refined/_runs/logs
run() { k=${1%%:*}; b=${1#*:}; [ "$b" = "$1" ] && b=900
  env OMP_NUM_THREADS=1 .venv/bin/python scripts/refined/run_refined.py plan $k --force --time-budget $b > $L/${k}_plan.log 2>&1; }
for job in "$@"; do
  while [ $(jobs -rp | wc -l) -ge 3 ]; do sleep 5; done
  run "$job" &
done
wait
