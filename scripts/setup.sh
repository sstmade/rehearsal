#!/usr/bin/env bash
# One-shot local setup: Python 3.12 venv in .venv, pinned dependencies, and the Franka Panda model
# from MuJoCo Menagerie (sparse clone, pinned commit). Safe to re-run.
#   scripts/setup.sh            # runtime + test deps
#   NO_DEV=1 scripts/setup.sh   # runtime deps only
# Needs: git, and uv (https://docs.astral.sh/uv/) or python3.12 with venv/pip.
set -euo pipefail
cd "$(dirname "$0")/.."

MENAGERIE_COMMIT="${MENAGERIE_COMMIT:-a662de603224c47123b0783821c58ad4402ad3e3}"
MENAGERIE_DIR=third_party/mujoco_menagerie
REQ=requirements-dev.txt
[ -n "${NO_DEV:-}" ] && REQ=requirements.txt

echo "==> Python venv (.venv) + $REQ"
if command -v uv >/dev/null 2>&1; then
  [ -x .venv/bin/python ] || uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r "$REQ"
else
  [ -x .venv/bin/python ] || python3.12 -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r "$REQ"
fi
# Linux only: MuJoCo's EGL/OSMesa offscreen backends go through PyOpenGL (macOS uses CGL).
if [ "$(uname -s)" = "Linux" ]; then
  if command -v uv >/dev/null 2>&1; then uv pip install --python .venv/bin/python PyOpenGL==3.1.10
  else .venv/bin/python -m pip install -q PyOpenGL==3.1.10; fi
fi

echo "==> MuJoCo Menagerie franka_emika_panda @ ${MENAGERIE_COMMIT:0:12}"
if [ ! -f "$MENAGERIE_DIR/franka_emika_panda/panda.xml" ]; then
  rm -rf "$MENAGERIE_DIR"
  git clone -q --filter=blob:none --sparse --no-checkout \
    https://github.com/google-deepmind/mujoco_menagerie.git "$MENAGERIE_DIR"
  git -C "$MENAGERIE_DIR" sparse-checkout set franka_emika_panda
  git -C "$MENAGERIE_DIR" checkout -q "$MENAGERIE_COMMIT"
else
  echo "    already present, skipping"
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> wrote .env from .env.example (no API key: the app runs offline until you add MODEL_API_KEY)"
fi

.venv/bin/python -c "import mujoco, rehearsal.sim; print('ok: mujoco', mujoco.__version__)"
echo
echo "Next:"
echo "  scripts/precompute.sh                      # demo results for every bundled scene (offline, free)"
echo "  .venv/bin/python -m rehearsal.server       # http://localhost:8765"
