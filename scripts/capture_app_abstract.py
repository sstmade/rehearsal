"""Record /app at 1920x1080 on the refined-scenes tables with abstract instructions (and one refusal).

    .venv/bin/python scripts/capture_app_abstract.py --url http://127.0.0.1:8777 [--out ../footage/cut/app_capture]

For each take: open the table, build the twin, type the instruction, compile with Muse Spark (recorded responses are
replayed at $0; anything new is a live call), click the first suggested answer when Muse Spark asks, then rehearse in
real search + MuJoCo physics and hold on the verdict and its replay. Every verdict is whatever physics returns - takes
are never retried for a nicer outcome. Writes app_<name>.mp4 per take (60 fps, see smooth_ui.py) and abstract_takes.json with the outcomes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
# (take, table, instruction, search budget s). The 31 / 02 / 43 takes replay the chains the refined-scenes run
# verified (footage/tables_sim/hero/refined/RESULTS.md): same instruction, same answer (the first suggested
# answer), same twin (t6_31_fixed is the disclosed photo-corrected twin).
TAKES = [
    ("clean_27", "t6_27", "Clean this up.", 60),
    ("laptop_32", "t6_32", "Make room for my laptop.", 60),
    ("dinner_02", "t6_02", "Get the table ready for dinner.", 60),
    ("refuse_02", "t6_02", "Knock the juice glass off the table.", 60),
    ("dinner_31f", "t6_31_fixed", "Get the table ready for dinner.", 90),
    ("laptop_02", "t6_02", "Make room for my laptop.", 120),
    ("laptop_02_v2", "t6_02", "Make room for my laptop.", 120),   # answer v2: orange + spoon beside the carton
    ("tidy_43", "t6_43", "Tidy up the pet things.", 240),
    ("tidy_43_600s", "t6_43", "Tidy up the pet things.", 600),   # the verified run had 900 s and parallel sims
]


def take(pw, url, out, name, sid, instr, budget):
    from smooth_ui import Driver
    from smooth_ui import GPU_ARGS, Recorder60
    b = pw.chromium.launch(args=GPU_ARGS)
    ctx = b.new_context(viewport={"width": 1920, "height": 1080})
    pg = ctx.new_page()
    d = Driver(pg, start=(1700, 760))
    pg.on("pageerror", lambda e: print("PAGEERROR", e))
    rec = {"take": name, "scene": sid, "instruction": instr, "steps": []}
    t0 = time.time()
    try:
        pg.goto(f"{url}/app?scene={sid}"); pg.wait_for_selector("#buildBtn", timeout=60000)
        rec60 = Recorder60(pg, out / f"app_{name}.mp4")
        pg.mouse.move(d.x, d.y)
        pg.wait_for_timeout(2600)                                     # the HUD reveal
        d.click("#buildBtn"); pg.wait_for_selector("#instr", timeout=120000)
        d.park(1660, 640)
        pg.wait_for_timeout(7000)                                     # photo -> twin morph, lands on physics
        d.click("#instr"); d.type(instr); pg.wait_for_timeout(250); d.click("#sendBtn")
        for _ in range(3):
            pg.wait_for_selector("#rehearseBtn, .bub.no, #replyBtn", timeout=180000)
            pg.wait_for_timeout(900)
            it = pg.evaluate("() => { const i = window.__rehearsal.intent || {}; return {status: i.status, cached: i.cached, "
                             "cost: i.cost_usd, q: i.intent && (i.intent.question || i.intent.reason)}; }")
            rec["steps"].append(it)
            if pg.query_selector("#askSugg button"):
                ans = pg.inner_text("#askSugg button >> nth=0")
                rec["steps"][-1]["answer"] = ans
                d.to("#askSugg button >> nth=0"); pg.wait_for_timeout(450)
                d.click("#askSugg button >> nth=0", ms=250, pause=120)
                pg.wait_for_function("() => !document.querySelector('#askSugg')", timeout=30000)
                continue
            if pg.query_selector("#confirmBtn") and not pg.query_selector("#rehearseBtn"):
                d.click("#confirmBtn"); pg.wait_for_timeout(900)
            break
        if pg.query_selector("#rehearseBtn"):
            # a budget the UI has no button for (local server only; the public demo caps at 240 s) gets one added
            pg.evaluate(f"""(() => {{ const seg = document.querySelector('#budgetSeg');
                let x = seg.querySelector('button[data-b="{budget}"]');
                if (!x) {{ x = document.createElement('button'); x.dataset.b = '{budget}'; x.textContent = '{budget} s'; seg.appendChild(x); }}
                x.onclick = () => seg.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b === x)); }})()""")
            d.click(f'#budgetSeg button[data-b="{budget}"]'); pg.wait_for_timeout(300)
            d.click("#rehearseBtn")
            d.park(1180, 560)
            pg.wait_for_selector("#verdictCard .vcard", timeout=900000)
            pg.wait_for_timeout(8000)                                 # verdict beat + replay
            tiles = pg.query_selector_all(".tile")
            for k in (1, 2):
                if k < len(tiles):
                    d.click(f".tile >> nth={k}"); pg.wait_for_timeout(3800)
            run = pg.evaluate("() => { const r = window.__rehearsal.runs[0] || {}; return {verdict: r.verdict, "
                              "attempts: (r.cands || []).length, budget: r.result && r.result.budget}; }")
            rec.update(run)
        else:
            d.park(1180, 560)
            pg.wait_for_timeout(4000)
            rec["verdict"] = rec["steps"][-1].get("status") if rec["steps"] else None
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    rec["wall_s"] = round(time.time() - t0, 1)
    try:
        rec["video_stats"] = rec60.stop()
    except NameError:
        pass
    ctx.close(); b.close()
    rec["video"] = f"app_{name}.mp4"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default=str(ROOT.parent / "footage" / "cut" / "app_capture"))
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--only")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright
    recs = []
    with sync_playwright() as pw:
        for name, sid, instr, bud in TAKES:
            if a.only and name not in a.only.split(","):
                continue
            r = take(pw, a.url, out, name + a.suffix, sid, instr, bud)
            print(json.dumps(r), flush=True)
            recs.append(r)
    shutil.rmtree(out / "_raw", ignore_errors=True)
    f = out / "abstract_takes.json"
    old = json.loads(f.read_text())["takes"] if f.exists() else []
    keep = [r for r in old if r["take"] not in {x["take"] for x in recs}]
    f.write_text(json.dumps({"measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "takes": keep + recs}, indent=1))


if __name__ == "__main__":
    main()
