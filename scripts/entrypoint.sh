#!/bin/sh
# Container entrypoint. The image defaults to REHEARSAL_OFFLINE=1 (no Meta API spend). The code treats
# any non-empty value as "offline", and HF Spaces cannot unset a variable, so REHEARSAL_OFFLINE=0 (or
# false/no) set as a Space variable is translated to "unset" here to enable live Muse Spark.
cd "$(dirname "$0")/.."
# Headless GL backend chosen at build time by scripts/gl_check.sh (egl or osmesa).
if [ -z "${MUJOCO_GL:-}" ] && [ -f .mujoco_gl ]; then
  MUJOCO_GL="$(cat .mujoco_gl)"; PYOPENGL_PLATFORM="$MUJOCO_GL"; export MUJOCO_GL PYOPENGL_PLATFORM
fi
case "${REHEARSAL_OFFLINE:-}" in
  0|false|False|FALSE|no|off) unset REHEARSAL_OFFLINE ;;
esac
if [ -n "${REHEARSAL_OFFLINE:-}" ] || [ -z "${MODEL_API_KEY:-}" ]; then
  echo "rehearsal: offline mode (OfflineAgent, no Meta API calls)"
else
  echo "rehearsal: LIVE mode, Muse Spark calls spend money (cap: \$${REHEARSAL_BUDGET_CAP:-default})"
fi
exec python -m rehearsal.server "$@"
