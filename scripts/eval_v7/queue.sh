#!/bin/bash
# eval_v7 queue: keep at most $JOBS run_v7 workers alive (shared machine); args = table ids
cd "$(dirname "$0")/../.."
EV="../footage/tables_sim/eval_v7"; mkdir -p "$EV/logs"
JOBS=${JOBS:-5}; BUDGET=${BUDGET:-1500}
for nn in "$@"; do
  while [ "$(pgrep -f 'run_v7.py --one' | wc -l)" -ge "$JOBS" ]; do sleep 10; done
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nohup .venv/bin/python scripts/eval_v7/run_v7.py --one --budget "$BUDGET" --sims 1500 "$nn" > "$EV/logs/$nn.log" 2>&1 &
  sleep 2
done
wait
