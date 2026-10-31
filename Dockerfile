# syntax=docker/dockerfile:1
# Rehearsal web app as a Hugging Face Docker Space (port 7860). Building pushes nothing anywhere.
#   docker build -t rehearsal .
#   docker run --rm -p 7860:7860 rehearsal                                              # offline, $0
#   docker run --rm -p 7860:7860 -e MODEL_API_KEY=... -e REHEARSAL_OFFLINE=0 rehearsal  # live Muse Spark ($)
#
# The demo results (out/, gitignored) are computed here at build time with the offline agent:
# deterministic, no API key, ~35 CPU-minutes total (about 4 min on 14 cores, longer on small builders).
# Build with --build-arg PRECOMPUTE=0 to skip it (the gallery then starts empty).
# See deploy/DEPLOY.md.

FROM python:3.12-slim-bookworm

# git: Menagerie sparse clone. Mesa: headless MuJoCo rendering on the CPU (llvmpipe), through EGL
# (libegl1 = glvnd loader, libegl-mesa0 = Mesa vendor, libgl1-mesa-dri = llvmpipe driver) or OSMesa
# (libosmesa6) as a fallback; scripts/gl_check.sh picks whichever works.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
        libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libgles2 libglvnd0 libosmesa6 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    REHEARSAL_HOST=0.0.0.0 \
    REHEARSAL_PORT=7860 \
    REHEARSAL_OFFLINE=1 \
    REHEARSAL_BUDGET_CAP=2 \
    REHEARSAL_BUDGET_DIR=/app/budget

# HF Spaces run the container as uid 1000.
RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements.txt .
# PyOpenGL: MuJoCo's EGL/OSMesa backends on Linux (macOS uses CGL, so it is not in requirements.txt).
RUN pip install -r requirements.txt PyOpenGL==3.1.10

# Franka Emika Panda from MuJoCo Menagerie (Apache-2.0), pinned; third_party/ is gitignored.
ARG MENAGERIE_COMMIT=a662de603224c47123b0783821c58ad4402ad3e3
RUN git clone -q --filter=blob:none --sparse --no-checkout \
        https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie \
 && git -C /tmp/menagerie sparse-checkout set franka_emika_panda \
 && git -C /tmp/menagerie checkout -q "$MENAGERIE_COMMIT" \
 && mkdir -p third_party/mujoco_menagerie \
 && mv /tmp/menagerie/franka_emika_panda third_party/mujoco_menagerie/ \
 && rm -rf /tmp/menagerie

COPY --chown=app rehearsal rehearsal
COPY --chown=app scenes scenes
COPY --chown=app web web
COPY --chown=app photos photos
COPY --chown=app scripts/entrypoint.sh scripts/gl_check.sh scripts/precompute.sh scripts/
RUN mkdir -p out uploads budget && chown -R app /app
USER app

# Pick the working offscreen GL backend (written to /app/.mujoco_gl), then bake the demo results.
ARG PRECOMPUTE=1
RUN sh scripts/gl_check.sh \
 && if [ "$PRECOMPUTE" = "1" ]; then \
      MUJOCO_GL="$(cat .mujoco_gl)" PYOPENGL_PLATFORM="$(cat .mujoco_gl)" PYTHON=python \
        bash scripts/precompute.sh \
      || echo "WARNING: precompute reported errors, see the summary above"; \
    fi \
 && rm -rf budget/* && du -sh out

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"REHEARSAL_PORT\"]}/api/scenes')"
CMD ["sh", "scripts/entrypoint.sh"]
