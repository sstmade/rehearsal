"""Record the full /app flow on table 06 with LIVE Meta calls, and save the measured per-model timings.

    .venv/bin/python scripts/capture_app_06.py --url http://127.0.0.1:8777 [--out ../footage/cut/app_capture]

Uploads the table 06 photo through the real upload step (contributor mode; the downscaled upload has different
bytes from the batch run, so Muse Spark and SAM 3.1 are called live, not replayed), builds the twin, compiles
"Put the onions into the steel bowl." (a new scene summary -> a live Muse Spark call), rehearses, and opens the
attempt wall. Writes <out>/timings_06.json and a 1920x1080 screen recording <out>/app_06_full.mp4 (60 fps CDP screencast, GPU-rendered, drawn cursor; see smooth_ui.py).
Costs roughly $0.05-0.15 of Meta API budget. Needs a live server (API key, not PUBLIC_DEMO).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
PHOTO = ROOT / "photos" / "ai_gen" / "tables_v6" / "table_06_indian-kitchen.png"
INSTR = "Put the onions into the steel bowl."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default=str(ROOT.parent / "footage" / "cut" / "app_capture"))
    ap.add_argument("--budget", type=int, default=900)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(urllib.request.urlopen(a.url + "/api/app/config"))
    if not cfg.get("uploads"):
        raise SystemExit("this server does not accept uploads (no API key or PUBLIC_DEMO)")
    from playwright.sync_api import sync_playwright
    marks, t0 = {}, time.time()
    with sync_playwright() as p:
        from smooth_ui import GPU_ARGS, Recorder60
        b = p.chromium.launch(args=GPU_ARGS)
        ctx = b.new_context(viewport={"width": 1920, "height": 1080})
        pg = ctx.new_page()
        from smooth_ui import Driver
        d = Driver(pg, start=(1100, 760))
        pg.on("pageerror", lambda e: print("PAGEERROR", e))
        pg.goto(a.url + "/app"); pg.wait_for_selector(".card", timeout=30000)
        rec60 = Recorder60(pg, out / f"app_06_full{a.suffix}.mp4")
        pg.mouse.move(d.x, d.y); pg.wait_for_timeout(900)
        d.to("#drop"); pg.wait_for_timeout(350)
        try:
            with pg.expect_file_chooser(timeout=4000) as fc:
                pg.mouse.down(); pg.wait_for_timeout(70); pg.mouse.up()
            fc.value.set_files(str(PHOTO))
        except Exception:  # noqa: BLE001  the drop zone did not open a chooser: feed the input directly
            pg.set_input_files("#fileIn", str(PHOTO))
        marks["upload_start"] = time.time() - t0
        d.park(1660, 700)
        pg.wait_for_selector("#buildBtn", timeout=300000)
        marks["perceived"] = time.time() - t0
        pg.wait_for_timeout(3600)                                  # the HUD reveal
        d.click("#buildBtn"); pg.wait_for_selector("#instr", timeout=120000)
        marks["twin"] = time.time() - t0
        d.park(1660, 640)
        pg.wait_for_timeout(7000)                                  # photo -> twin morph, lands on physics
        d.click("#instr"); d.type(INSTR); pg.wait_for_timeout(250); d.click("#sendBtn")
        pg.wait_for_selector("#rehearseBtn, .bub.no, #replyBtn", timeout=180000)
        marks["intent"] = time.time() - t0
        pg.wait_for_timeout(1000)
        if not pg.query_selector("#rehearseBtn"):
            raise SystemExit("intent did not compile to goals: " + pg.inner_text("#intentOut")[:300])
        pg.evaluate(f"""(() => {{ const seg = document.querySelector('#budgetSeg');
            let x = seg.querySelector('button[data-b="{a.budget}"]');
            if (!x) {{ x = document.createElement('button'); x.dataset.b = '{a.budget}'; x.textContent = '{a.budget} s'; seg.appendChild(x); }}
                x.onclick = () => seg.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b === x)); }})()""")
        d.click(f'#budgetSeg button[data-b="{a.budget}"]'); pg.wait_for_timeout(300)
        d.click("#rehearseBtn")
        d.park(1180, 560)
        pg.wait_for_selector("#verdictCard .vcard", timeout=900000)
        marks["verdict"] = time.time() - t0
        pg.wait_for_timeout(8000)                                  # verdict beat + the verified-plan replay
        tiles = pg.query_selector_all(".tile")
        for k in (1, 2, 3):
            if k < len(tiles):
                d.click(f".tile >> nth={k}"); pg.wait_for_timeout(3800)
        if pg.query_selector("#wallExpand"):
            d.click("#wallExpand"); pg.wait_for_timeout(3200); d.click("#wallExpand"); pg.wait_for_timeout(800)
        state = pg.evaluate("""() => { const S = window.__rehearsal; const run = S.runs[0] || {};
            return { scene: S.sid, models: (S.info || {}).models, n_objects: (S.info || {}).objects?.length,
                     intent: S.intent && { wall_s: S.intent.wall_s, cost_usd: S.intent.cost_usd, cached: S.intent.cached,
                                           status: S.intent.status, model: S.intent.model, goals: S.intent.intent?.goals },
                     verdict: run.verdict, budget: run.result?.budget, stats: run.result?.search?.stats,
                     stress: run.result?.stress, via_direct: !!run.result?.via_direct, attempts: (run.cands || []).length }; }""")
        vstats = rec60.stop()
        ctx.close(); b.close()
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
           "video": f"app_06_full{a.suffix}.mp4", "video_stats": vstats}
    (out / f"timings_06{a.suffix}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec["steps"], indent=1), "\nverdict:", rec["verdict"])


if __name__ == "__main__":
    main()
