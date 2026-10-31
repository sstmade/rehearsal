# Deploying the live demo (Hugging Face Space, Docker)

What judges get: the web app on port 7860 with all six bundled scenes already rehearsed. The results are
computed while the image builds, with the deterministic offline agent, so the Space needs **no API key
and spends $0**. Nothing in this repo pushes or deploys by itself; these are the manual steps.

## 0. Before you start

- Commit everything you want in the Space (the folder is built from the committed tree).
- Put the GitHub URL into `deploy/hf_space_README.md` (replace `{{TODO-USER: GitHub URL}}`) and commit.
- A Hugging Face account and an access token with **write** scope: https://huggingface.co/settings/tokens

## 1. Create the Space (browser)

https://huggingface.co/new-space
- Owner/name: e.g. `<you>/rehearsal`
- SDK: **Docker**, template **Blank**
- Hardware: **CPU basic** (free: 2 vCPU, 16 GB RAM) is enough to serve the demo
- Visibility: **Public** (the submission needs a link judges can open)

Do not add any secret yet.

## 2. Assemble and upload the files (terminal, repo root)

```bash
scripts/make_space.sh                     # -> out_space/ (gitignored, ~14 MB, no network)
uvx --from huggingface_hub hf auth login  # paste the write token
uvx --from huggingface_hub hf upload <you>/rehearsal out_space . --repo-type space \
    --commit-message "Rehearsal demo"
```

`out_space/` holds `Dockerfile`, `.dockerignore`, `requirements.txt`, `rehearsal/`, `scenes/`, `web/`,
`photos/`, the three runtime scripts, `THIRD_PARTY_NOTICES.md`, and `README.md` = `deploy/hf_space_README.md`
(its YAML header tells HF: `sdk: docker`, `app_port: 7860`). `hf upload` stores the images through HF's
large-file storage automatically; a plain `git push` to the Space would be rejected for the binary files
(.jpg/.png/.gif/.mp4) unless they go through Git LFS/Xet, so prefer `hf upload`.
`datasets/` (HOPE is CC BY-NC-SA) is deliberately not part of the Space.

To update later: commit, re-run `scripts/make_space.sh`, run the same `hf upload` (it uploads a new commit;
add `--delete "*"` to also remove files that no longer exist).

## 3. Watch the build

Space page -> **Logs** -> *Build*. Expected sequence:
1. apt (Mesa EGL/OSMesa) and pip install, ~1-2 min
2. Menagerie sparse clone, seconds
3. `gl_check: MUJOCO_GL=egl works` (or `osmesa`)
4. `===== precompute <scene>` x 6 and a summary with `"success": true` for each, then `du -sh out` (~116 MB)

Step 4 is ~35 CPU-minutes of simulation: measured 250 s wall on a 14-core Mac (desk_01 alone 90 s).
On a builder with few cores it can take 15-20 min. If the build times out or step 3 prints no working
backend: in **Settings -> Variables and secrets** add the variable `PRECOMPUTE` = `0` (HF passes Space
variables to the Docker build as build args) and rebuild; the app then starts with an empty gallery and
someone has to press Rehearse, so treat that as a fallback only.

Then open the **App** tab: the gallery shows the six scenes. By default `scripts/make_space.sh` ships
your local `out/` (the live Muse Spark runs, agent `muse-spark-1.3-contributor`) and turns the build-time
precompute off, so the Space shows real model diagnoses: e.g. kitchen_01 fails with a protective stop,
Muse Spark reorders the moves, and the Rehearse panel shows its tool-call trace. The badges read
"Muse Spark · Contributor". With `LIVE_OUT=0 scripts/make_space.sh` the build bakes offline replays
instead and the badges read "Offline reference replay · no model call".

## 4. Optional: live Muse Spark on the Space (spends real money)

The image defaults to `REHEARSAL_OFFLINE=1`. To go live, in **Settings -> Variables and secrets**:
- **Secret** `MODEL_API_KEY` = your Meta Model API key (secrets are never shown in logs or to visitors)
- **Variable** `REHEARSAL_OFFLINE` = `0` (the entrypoint treats 0/false/no/off as "live")
- **Variable** `REHEARSAL_BUDGET_CAP` = a small number of USD, e.g. `2` (the image default is 2)

Warnings, read before doing this:
- **Anyone on the internet can then spend your money.** The public app accepts photo uploads
  (Muse Spark + SAM 3.1 calls) and new rehearsals. There is no login.
- **The budget cap is per container, not global.** The ledger lives in the container (`/app/budget`) and
  is wiped on every restart/rebuild, so the cap resets each time the Space restarts.
- The image sets `REHEARSAL_PUBLIC_DEMO=1`: when the server supports it, uploads are refused (403),
  bundled scenes and their reports are read-only, and only one rehearsal runs at a time. Check that
  `rehearsal/server.py` contains `PUBLIC_DEMO` before relying on it; if it does not, keep the public Space
  offline. For a live demo with uploads, run locally (`scripts/setup.sh`, put the key in `.env`) or
  duplicate the Space as **private**, add the secret there and set `REHEARSAL_PUBLIC_DEMO` = `0`.
- To stop spending immediately: delete the `MODEL_API_KEY` secret (the Space restarts offline).

## 5. Run the same image locally (if Docker is available)

```bash
docker build -t rehearsal .
docker run --rm -p 7860:7860 rehearsal                     # http://localhost:7860, offline
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build log: `gl_check: MUJOCO_GL=egl failed` then `osmesa works` | fine, OSMesa is used |
| Both backends fail | rendering GIFs is impossible: set `PRECOMPUTE=0`, report the error |
| App tab: "Starting" forever | check *Container* logs; the server must listen on 0.0.0.0:7860 (set in the Dockerfile) |
| Gallery empty | the precompute step was skipped or failed; see the build log summary |
| `hf upload` rejected | token needs write scope; Space name must match `<you>/<space>` |
