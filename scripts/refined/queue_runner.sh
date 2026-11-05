#!/bin/bash
# The refined-heroes agent's ONE heavy slot: run tools/heavy.sh scripts/refined/queue_runner.sh QDIR
# Executes the job files dropped into QDIR (*.job, bash, name order) strictly one after another; idles while the
# queue is empty; exits when QDIR/STOP exists.  Never more than one job at a time.
Q=$1; mkdir -p "$Q/done"
while [ ! -f "$Q/STOP" ]; do
  j=$(ls "$Q"/*.job 2>/dev/null | head -1)
  if [ -z "$j" ]; then sleep 15; continue; fi
  echo "$(date +%T) START $(basename "$j")"
  bash "$j" > "${j%.job}.log" 2>&1; echo "$(date +%T) END $(basename "$j") rc=$?"
  mv "$j" "${j%.job}.log" "$Q/done/" 2>/dev/null
done
