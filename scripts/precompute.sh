#!/usr/bin/env bash
# Rehearse every bundled scene with its demo instruction, offline (OfflineAgent, no API calls, $0),
# writing out/<scene>/ so the web app opens with results instead of an empty gallery.
#   scripts/precompute.sh                 # all scenes
#   scripts/precompute.sh desk_01,lamp_01 # some scenes
# Deterministic: seeded stress tests, so re-running gives the same numbers.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python
args=(--precompute)
[ -n "${1:-}" ] && args+=(--only "$1")
exec "$PY" -m rehearsal.rehearse "${args[@]}"
