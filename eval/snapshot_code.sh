#!/usr/bin/env bash
# Read-only snapshot of another branch's code + scenes, for evaluating work that is not merged yet.
#   eval/snapshot_code.sh robotics      -> eval/.snap/robotics/ (rehearsal/, scenes/, SNAPSHOT_REF)
# Then:  REHEARSAL_CODE=eval/.snap/robotics .venv/bin/python eval/run_eval.py --scenes eval/.snap/robotics/scenes
# The snapshot shares third_party/.env and (via REHEARSAL_BUDGET_DIR, set by eval/common.py) the budget ledger.
set -euo pipefail
ref="${1:?usage: snapshot_code.sh <git-ref>}"
here="$(cd "$(dirname "$0")/.." && pwd)"
dst="$here/eval/.snap/$ref"
rm -rf "$dst"; mkdir -p "$dst"
git -C "$here" archive "$ref" rehearsal scenes | tar -x -C "$dst"
ln -s "$here/third_party" "$dst/third_party"
[ -e "$here/.env" ] && ln -s "$here/.env" "$dst/.env"
echo "$ref $(git -C "$here" rev-parse --short "$ref")" > "$dst/SNAPSHOT_REF"
mkdir -p "$here/eval/scenes_snapshot/$ref"
cp "$dst"/scenes/*.json "$here/eval/scenes_snapshot/$ref/"
cp "$dst/SNAPSHOT_REF" "$here/eval/scenes_snapshot/$ref/"
echo "snapshot: $dst ($(cat "$dst/SNAPSHOT_REF"))"
