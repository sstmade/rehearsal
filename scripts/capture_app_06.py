"""Record the full /app flow on table 06 with LIVE Meta calls, and save the measured per-model timings.

    .venv/bin/python scripts/capture_app_06.py --url http://127.0.0.1:8777 [--out ../footage/cut/app_capture]

Uploads the table 06 photo through the real upload step (contributor mode; the downscaled upload has different
bytes from the batch run, so Muse Spark and SAM 3.1 are called live, not replayed), builds the twin, compiles
"Put the onions into the steel bowl." (a new scene summary -> a live Muse Spark call), rehearses, and opens the
attempt wall. Writes <out>/timings_06.json and a 1920x1080 screen recording <out>/app_06_full.webm.
Costs roughly $0.05-0.15 of Meta API budget. Needs a live server (API key, not PUBLIC_DEMO).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHOTO = ROOT / "photos" / "ai_gen" / "tables_v6" / "table_06_indian-kitchen.png"
INSTR = "Put the onions into the steel bowl."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default=str(ROOT.parent / "footage" / "cut" / "app_capture"))
    ap.add_argument("--budget", type=int, default=900)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(urllib.request.urlopen(a.url + "/api/app/config"))
    if not cfg.get("uploads"):
        raise SystemExit("this server does not accept uploads (no API key or PUBLIC_DEMO)")
    from playwright.sync_api import sync_playwright
    marks, t0 = {}, time.time()
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1920, "height": 1080},
                            record_video_dir=str(out / "_raw"), record_video_size={"width": 1920, "height": 1080})
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: print("PAGEERROR", e))
        pg.goto(a.url + "/app"); pg.wait_for_selector(".card", timeout=30000); pg.wait_for_timeout(1500)
        marks["upload_start"] = time.time() - t0
        pg.set_input_files("#fileIn", str(PHOTO))
        pg.wait_for_selector("#buildBtn", timeout=300000)
        marks["perceived"] = time.time() - t0
        pg.wait_for_timeout(5000)                                  # the HUD reveal
        pg.click("#buildBtn"); pg.wait_for_selector("#instr", timeout=120000)
        marks["twin"] = time.time() - t0
        pg.wait_for_timeout(6000)                                  # primitives -> refined morph
        pg.click("#instr"); pg.keyboard.type(INSTR, delay=45); pg.click("#sendBtn")
        pg.wait_for_selector("#rehearseBtn, .bub.no, #replyBtn", timeout=180000)
        marks["intent"] = time.time() - t0
        pg.wait_for_timeout(2500)
        if not pg.query_selector("#rehearseBtn"):
            raise SystemExit("intent did not compile to goals: " + pg.inner_text("#intentOut")[:300])
        pg.evaluate(f"""(() => {{ const seg = document.querySelector('#budgetSeg');
            let x = seg.querySelector('button[data-b="{a.budget}"]');
            if (!x) {{ x = document.createElement('button'); x.dataset.b = '{a.budget}'; x.textContent = '{a.budget} s'; seg.appendChild(x); }}
            seg.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b === x)); }})()""")
        pg.click("#rehearseBtn")
        pg.wait_for_selector("#verdictCard .vcard", timeout=600000)
        marks["verdict"] = time.time() - t0
        pg.wait_for_timeout(9000)                                  # the verified-plan replay
        tiles = pg.query_selector_all(".tile")
        for el in tiles[1:4]:
            el.click(); pg.wait_for_timeout(4500)
        if pg.query_selector("#wallExpand"):
            pg.click("#wallExpand"); pg.wait_for_timeout(4000); pg.click("#wallExpand")
        state = pg.evaluate("""() => { const S = window.__rehearsal; const run = S.runs[0] || {};
            return { scene: S.sid, models: (S.info || {}).models, n_objects: (S.info || {}).objects?.length,
                     intent: S.intent && { wall_s: S.intent.wall_s, cost_usd: S.intent.cost_usd, cached: S.intent.cached,
                                           status: S.intent.status, model: S.intent.model, goals: S.intent.intent?.goals },
                     verdict: run.verdict, budget: run.result?.budget, stats: run.result?.search?.stats,
                     stress: run.result?.stress, via_direct: !!run.result?.via_direct, attempts: (run.cands || []).length }; }""")
        video = pg.video.path()
        ctx.close(); b.close()
    shutil.move(video, out / "app_06_full.webm")
    shutil.rmtree(out / "_raw", ignore_errors=True)
    rec = {"table": "t6_06", "photo": str(PHOTO.relative_to(ROOT)), "instruction": INSTR,
           "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "data_mode": cfg.get("upload_data"),
           "steps": [
               *[{"model": m["model"], "step": m["role"], "latency_s": m["s"], "cost_usd": m["usd"], "paid_calls": m["calls"]}
                 for m in (state.get("models") or [])],
               {"model": "Muse Spark", "step": "intent compile", "latency_s": (state["intent"] or {}).get("wall_s"),
                "cost_usd": (state["intent"] or {}).get("cost_usd"), "cached": (state["intent"] or {}).get("cached")},
               {"model": "MuJoCo (search + physics)", "step": "rehearse", "latency_s": (state.get("budget") or {}).get("used_s"),
                "cost_usd": 0.0, "sims": (state.get("budget") or {}).get("used_sims")}],
           "verdict": state.get("verdict"),
           "verdict_path": ("direct_carry" if state.get("via_direct") else "search") if state.get("verdict") == "sent" else None,
           "stress": state.get("stress") and {"successes": state["stress"].get("successes"), "n": state["stress"].get("n")}, "attempts_recorded": state.get("attempts"), "wall_clock_marks_s": {k: round(v, 1) for k, v in marks.items()}, "app_state": state,
           "video": "app_06_full.webm"}
    (out / "timings_06.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec["steps"], indent=1), "\nverdict:", rec["verdict"])


if __name__ == "__main__":
    main()
