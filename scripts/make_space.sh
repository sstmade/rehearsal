#!/usr/bin/env bash
# Assemble the Hugging Face Space folder locally (no network): only what the Docker build needs,
# taken from the committed tree (git archive), with deploy/hf_space_README.md as the Space README
# (HF reads its YAML header: sdk docker, app_port 7860).
#   scripts/make_space.sh [git-ref] [dest]     # defaults: HEAD, out_space/  (gitignored)
#   OUT_SRC=/path/to/out scripts/make_space.sh # take the shipped reports from another checkout (default: ./out)
#   (also copies out_llm_v3/ next to OUT_SRC, but only if rehearsal/server.py has a route for it: it is ~230 MB)
# Then upload it, see deploy/DEPLOY.md.
set -euo pipefail
cd "$(dirname "$0")/.."
REF="${1:-HEAD}"
DEST="${2:-out_space}"
rm -rf "$DEST" && mkdir -p "$DEST"
git archive "$REF" Dockerfile .dockerignore requirements.txt rehearsal scenes web photos bench assets/sam3d \
  scripts/entrypoint.sh scripts/gl_check.sh scripts/precompute.sh THIRD_PARTY_NOTICES.md \
  | tar -x -C "$DEST"
git show "$REF:deploy/hf_space_README.md" > "$DEST/README.md"
# Ship the local out/ (live Muse Spark / hybrid reports) instead of baking offline replays at build time.
# LIVE_OUT=0 keeps the build-time offline precompute.
OUT_SRC="${OUT_SRC:-out}"
if [ "${LIVE_OUT:-1}" = "1" ] && ls "$OUT_SRC"/*/report.json >/dev/null 2>&1; then
  # only the scene reports the gallery serves (folders with report.json); not tables_v6/ renders (~430 MB),
  # droid_v2/ (not served) or app/ (the /app twin cache, rebuilt at runtime)
  mkdir -p "$DEST/out"
  for d in "$OUT_SRC"/*/; do [ -f "$d/report.json" ] && cp -R "$d" "$DEST/out/"; done
  # /app reads the 50 test tables' task instructions from out/tables_v6/NN.json (not the renders next to them)
  if ls "$OUT_SRC"/tables_v6/[0-9]*.json >/dev/null 2>&1; then
    mkdir -p "$DEST/out/tables_v6" && cp "$OUT_SRC"/tables_v6/[0-9]*.json "$DEST/out/tables_v6/"
  fi
  sed -i.bak '/^out$/d' "$DEST/.dockerignore" && rm "$DEST/.dockerignore.bak"
  sed -i.bak -e 's|^RUN mkdir -p out uploads budget|COPY --chown=app out out\nRUN mkdir -p out uploads budget|' \
             -e 's|^ARG PRECOMPUTE=1|ARG PRECOMPUTE=0|' "$DEST/Dockerfile" && rm "$DEST/Dockerfile.bak"
  echo "Included out/ ($(ls "$OUT_SRC" | wc -l | tr -d ' ') scenes); build-time precompute off"
  LLM_SRC="$(dirname "$OUT_SRC")/out_llm_v3"
  if [ -d "$LLM_SRC" ] && grep -q out_llm_v3 "$DEST/rehearsal/server.py"; then
    cp -R "$LLM_SRC" "$DEST/out_llm_v3"
    sed -i.bak '/^out_\*$/d' "$DEST/.dockerignore" && rm "$DEST/.dockerignore.bak"
    sed -i.bak 's|^COPY --chown=app out out$|COPY --chown=app out out\nCOPY --chown=app out_llm_v3 out_llm_v3|' "$DEST/Dockerfile" && rm "$DEST/Dockerfile.bak"
    echo "Included out_llm_v3/ (history of LLM-only runs)"
  fi
fi
echo "Space folder: $DEST ($(du -sh "$DEST" | cut -f1), $(find "$DEST" -type f | wc -l | tr -d ' ') files)"
if grep -q "TODO-USER" "$DEST/README.md"; then
  echo "NOTE: $DEST/README.md still has a {{TODO-USER}} placeholder (GitHub URL); edit deploy/hf_space_README.md"
fi
