#!/usr/bin/env bash
# Assemble the Hugging Face Space folder locally (no network): only what the Docker build needs,
# taken from the committed tree (git archive), with deploy/hf_space_README.md as the Space README
# (HF reads its YAML header: sdk docker, app_port 7860).
#   scripts/make_space.sh [git-ref] [dest]     # defaults: HEAD, out_space/  (gitignored)
# Then upload it, see deploy/DEPLOY.md.
set -euo pipefail
cd "$(dirname "$0")/.."
REF="${1:-HEAD}"
DEST="${2:-out_space}"
rm -rf "$DEST" && mkdir -p "$DEST"
git archive "$REF" Dockerfile .dockerignore requirements.txt rehearsal scenes web photos \
  scripts/entrypoint.sh scripts/gl_check.sh scripts/precompute.sh THIRD_PARTY_NOTICES.md \
  | tar -x -C "$DEST"
git show "$REF:deploy/hf_space_README.md" > "$DEST/README.md"
# Ship the local out/ (live Muse Spark reports) instead of baking offline replays at build time.
# LIVE_OUT=0 keeps the build-time offline precompute.
if [ "${LIVE_OUT:-1}" = "1" ] && ls out/*/report.json >/dev/null 2>&1; then
  cp -R out "$DEST/out"
  sed -i.bak '/^out$/d' "$DEST/.dockerignore" && rm "$DEST/.dockerignore.bak"
  sed -i.bak -e 's|^RUN mkdir -p out uploads budget|COPY --chown=app out out\nRUN mkdir -p out uploads budget|' \
             -e 's|^ARG PRECOMPUTE=1|ARG PRECOMPUTE=0|' "$DEST/Dockerfile" && rm "$DEST/Dockerfile.bak"
  echo "Included out/ ($(ls out | wc -l | tr -d ' ') scenes); build-time precompute off"
fi
echo "Space folder: $DEST ($(du -sh "$DEST" | cut -f1), $(find "$DEST" -type f | wc -l | tr -d ' ') files)"
if grep -q "TODO-USER" "$DEST/README.md"; then
  echo "NOTE: $DEST/README.md still has a {{TODO-USER}} placeholder (GitHub URL); edit deploy/hf_space_README.md"
fi
