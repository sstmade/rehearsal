#!/bin/sh
# Find a headless OpenGL backend MuJoCo can render with (Linux containers: egl, then osmesa) and
# write its name to .mujoco_gl for scripts/entrypoint.sh. Exits non-zero if none works.
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
for gl in ${MUJOCO_GL_CANDIDATES:-egl osmesa}; do
  if MUJOCO_GL=$gl PYOPENGL_PLATFORM=$gl "$PY" -c "
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"box\" size=\".1 .1 .1\"/></worldbody></mujoco>')
with mujoco.Renderer(m, 64, 64) as r:
    d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); img = r.render()
assert img.shape == (64, 64, 3) and img.any(), 'blank frame'
" 2>/tmp/gl_check.err; then
    echo "$gl" > .mujoco_gl
    echo "gl_check: MUJOCO_GL=$gl works"
    exit 0
  fi
  echo "gl_check: MUJOCO_GL=$gl failed: $(tail -1 /tmp/gl_check.err)"
done
exit 1
