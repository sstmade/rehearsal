# syntax=docker/dockerfile:1
# Rehearsal web app (HF Spaces style: port 7860). Build only; nothing here pushes anywhere.
#   docker build -t rehearsal .
#   docker run --rm -p 7860:7860 rehearsal            # offline agent (no key)
#   docker run --rm -p 7860:7860 -e MODEL_API_KEY=... -e REHEARSAL_OFFLINE= rehearsal
#
# third_party/ is gitignored, so the Franka Panda model is fetched here from MuJoCo Menagerie
# (sparse checkout, pinned to the commit the project was developed against).
# Offline alternative: put a checkout at third_party/mujoco_menagerie and build with
#   --build-arg MENAGERIE_FROM=local    (see the "local" stage below)

ARG MENAGERIE_FROM=git

# ---------------------------------------------------------------- menagerie (git)
FROM python:3.12-slim AS menagerie-git
ARG MENAGERIE_COMMIT=f3b6013629c7b94f8ee93fd977b161fa8350f9bc
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none --sparse --no-checkout \
        https://github.com/google-deepmind/mujoco_menagerie.git /menagerie \
 && cd /menagerie && git sparse-checkout set franka_emika_panda \
 && (git checkout -q "$MENAGERIE_COMMIT" || git checkout -q HEAD) \
 && rm -rf .git

# ---------------------------------------------------------------- menagerie (local copy)
FROM python:3.12-slim AS menagerie-local
COPY third_party/mujoco_menagerie/franka_emika_panda /menagerie/franka_emika_panda

FROM menagerie-${MENAGERIE_FROM} AS menagerie

# ---------------------------------------------------------------- app
FROM python:3.12-slim
# Headless offscreen rendering for the attempt GIFs: EGL on Mesa (llvmpipe, CPU only).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libgles2 libglvnd0 \
 && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    REHEARSAL_HOST=0.0.0.0 \
    REHEARSAL_PORT=7860 \
    REHEARSAL_OFFLINE=1 \
    REHEARSAL_BUDGET_DIR=/app/budget

WORKDIR /app
COPY requirements.txt .
# PyOpenGL is what mujoco's EGL/OSMesa backends use on Linux (macOS uses CGL, so it is not in requirements.txt).
RUN pip install --no-cache-dir -r requirements.txt PyOpenGL==3.1.10

COPY --from=menagerie /menagerie/franka_emika_panda third_party/mujoco_menagerie/franka_emika_panda
COPY rehearsal rehearsal
COPY scenes scenes
COPY web web
COPY photos photos
COPY README.md CONTRACTS.md ./
# Precomputed demo results are baked in if out/ exists in the build context (it is gitignored:
# run `python -m rehearsal.rehearse --precompute` before building). Needs BuildKit (default).
RUN --mount=type=bind,source=.,target=/ctx \
    if [ -d /ctx/out ]; then cp -r /ctx/out ./out; fi \
 && mkdir -p out uploads budget \
 && useradd -m -u 1000 app && chown -R app /app
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"REHEARSAL_PORT\"]}/api/scenes')"
CMD ["python", "-m", "rehearsal.server"]
