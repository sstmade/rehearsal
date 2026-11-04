"""API-level check of /app in public-demo mode with an EMPTY budget dir (what a clean clone / the HF Space sees).

    .venv/bin/python scripts/app_public_check.py [--rehearse t6_41]

Starts its own server (REHEARSAL_PUBLIC_DEMO=1, REHEARSAL_OFFLINE=1, no key, fresh temp budget dir), then
for every test table compiles each suggested instruction (must come from the shipped replay file, never
"no_recording"), and rehearses one table with a short budget. No browser, $0. Exit 0 = all good.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearse", default="t6_41")
    ap.add_argument("--budget", type=int, default=30)
    a = ap.parse_args()
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    bd = tempfile.mkdtemp(prefix="rehearsal_budget_")
    env = dict(os.environ, REHEARSAL_PORT=str(port), REHEARSAL_PUBLIC_DEMO="1", REHEARSAL_OFFLINE="1",
               MODEL_API_KEY="", REHEARSAL_BUDGET_DIR=bd)
    srv = subprocess.Popen([sys.executable, "-m", "rehearsal.server"], cwd=ROOT, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    B = f"http://127.0.0.1:{port}"

    def get(p):
        return json.load(urllib.request.urlopen(B + p, timeout=30))

    def post(p, body):
        r = urllib.request.Request(B + p, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(r, timeout=60))

    def wait(j, t=600):
        t0 = time.time()
        while time.time() - t0 < t:
            jb = get(f"/api/job?id={j}")
            if jb["state"] in ("done", "error"):
                if jb["state"] == "error":
                    raise RuntimeError(jb.get("error"))
                return jb["result"]
            time.sleep(0.4)
        raise TimeoutError(j)

    bad = []
    try:
        for _ in range(60):
            try:
                cfg = get("/api/app/config"); break
            except OSError:
                time.sleep(0.5)
        print("config:", {k: cfg[k] for k in ("live", "public_demo", "uploads", "my_desk", "sam3d")})
        if cfg["live"] or cfg["uploads"] or cfg["my_desk"] or cfg["sam3d"] or not cfg["public_demo"]:
            bad.append("public mode flags wrong")
        tables = [x for x in get("/api/app/samples") if x.get("group") == "tables"] if isinstance(get("/api/app/samples"), list) \
            else get("/api/app/samples").get("tables", [])
        n = ok = 0
        stat = {}
        for t in tables:
            info = get(f"/api/app/scene?id={t['id']}")
            if not info.get("photo"):
                bad.append(f"{t['id']}: no photo URL")
            for ins in info.get("suggestions", []):
                n += 1
                r = wait(post("/api/app/intent", {"scene": t["id"], "instruction": ins})["job"])
                stat[r["status"]] = stat.get(r["status"], 0) + 1
                if r["status"] in ("no_recording", "api_error"):
                    bad.append(f"{t['id']}: {r['status']} for {ins!r}")
                else:
                    ok += 1
        print(f"tables: {len(tables)}; suggested instructions: {n}; answered from replay: {ok}; statuses: {stat}")
        tw = post("/api/app/twin", {"scene": a.rehearse})
        r = wait(post("/api/app/intent", {"scene": a.rehearse, "instruction": get(f"/api/app/scene?id={a.rehearse}")["suggestions"][0]})["job"])
        if r["status"] == "ready":
            t0 = time.time()
            res = wait(post("/api/app/rehearse", {"scene": a.rehearse, "instruction": r["instruction"], "goals": r["intent"]["goals"],
                                                   "budget_s": a.budget})["job"])
            ev = get(f"/api/app/events?job=&since=0") if False else None
            print(f"rehearse {a.rehearse}: verdict {res['verdict']} in {time.time() - t0:.0f} s; sims {res['budget'].get('used_sims')}; "
                  f"twin bodies {len(tw['objects'])}")
        else:
            bad.append(f"rehearse table intent status {r['status']}")
        spent = json.loads((Path(bd) / "state.json").read_text()).get("spent", 0) if (Path(bd) / "state.json").exists() else 0
        print(f"Meta spend: ${spent}")
    finally:
        srv.terminate()
    for b in bad:
        print("FAIL", b)
    print("PUBLIC CHECK", "PASS" if not bad else f"FAIL ({len(bad)})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
