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
echo "Space folder: $DEST ($(du -sh "$DEST" | cut -f1), $(find "$DEST" -type f | wc -l | tr -d ' ') files)"
if grep -q "TODO-USER" "$DEST/README.md"; then
  echo "NOTE: $DEST/README.md still has a {{TODO-USER}} placeholder (GitHub URL); edit deploy/hf_space_README.md"
fi
