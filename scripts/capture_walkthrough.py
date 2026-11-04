"""One continuous /app walkthrough for the film: start screen -> choose a table -> perception HUD (Muse Spark labels,
SAM 3.1 masks) -> twin morph landing in Physics -> type the instruction -> ASK, click the suggested answer ->
rehearse (the futures wall filling, nothing trimmed) -> verdict + replay of the verified plan.

    .venv/bin/python scripts/capture_walkthrough.py --scene t6_02 --instr "Make room for my laptop." --name 02 --budget 120

Writes <out>/app_full_walkthrough_<name>_phys.mp4 (60 fps, see smooth_ui.py) and walkthrough_<name>_timings.json
(start time of each step in seconds from the first video frame). The private "My desk" card is hidden in the recording.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

HIDE_PRIVATE = "() => { for (const c of document.querySelectorAll('.card[data-id=\"my_desk\"]')) c.remove(); }"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default=str(ROOT.parent / "footage" / "cut" / "app_capture"))
    ap.add_argument("--scene", required=True)
    ap.add_argument("--instr", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--budget", type=int, default=120)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright
    from smooth_ui import GPU_ARGS, Driver, Recorder60
    marks, rec = {}, {"scene": a.scene, "instruction": a.instr, "steps": []}
    with sync_playwright() as p:
        b = p.chromium.launch(args=GPU_ARGS)
        ctx = b.new_context(viewport={"width": 1920, "height": 1080})
        pg = ctx.new_page()
        d = Driver(pg, start=(1500, 820))
        pg.on("pageerror", lambda e: print("PAGEERROR", e))
        # the private desk card must never reach a recording: drop it before the first frame and whenever the grid re-renders
        pg.add_init_script("new MutationObserver(() => { for (const c of document.querySelectorAll('.card[data-id=\"my_desk\"]')) c.remove(); })"
                           ".observe(document, {childList: true, subtree: true});")
        pg.goto(a.url + "/app"); pg.wait_for_selector(f'.card[data-id="{a.scene}"]', timeout=30000)
        pg.evaluate(HIDE_PRIVATE)
        assert not pg.query_selector('.card[data-id="my_desk"]')
        vid = out / f"app_full_walkthrough_{a.name}_phys.mp4"
        r60 = Recorder60(pg, vid)
        t0 = time.time()
        mark = lambda k: marks.__setitem__(k, round(time.time() - t0, 2))  # noqa: E731
        mark("1_start_screen")
        pg.mouse.move(d.x, d.y); pg.wait_for_timeout(1400)
        d.click(f'.card[data-id="{a.scene}"]', ms=650, pause=260)
        mark("2_table_chosen")
        d.park(1660, 760)
        pg.wait_for_selector("#buildBtn", timeout=60000)
        mark("3_4_perception_hud")                                    # Muse Spark labels + SAM 3.1 masks draw in
        pg.wait_for_timeout(4200)
        d.click("#buildBtn"); mark("5_twin_build")
        pg.wait_for_selector("#instr", timeout=120000)
        d.park(1660, 640)
        pg.wait_for_timeout(7000)                                     # photo -> twin morph, lands on physics
        mark("6_type_instruction")
        d.click("#instr"); d.type(a.instr); pg.wait_for_timeout(250); d.click("#sendBtn")
        for _ in range(3):
            pg.wait_for_selector("#rehearseBtn, .bub.no, #replyBtn", timeout=180000)
            it = pg.evaluate("() => { const i = window.__rehearsal.intent || {}; return {status: i.status, cached: i.cached, "
                             "cost: i.cost_usd, q: i.intent && (i.intent.question || i.intent.reason)}; }")
            rec["steps"].append(it)
            if pg.query_selector("#askSugg button"):
                mark("7_ask_question")
                pg.wait_for_timeout(1300)                             # let the question be read
                rec["steps"][-1]["answer"] = pg.inner_text("#askSugg button >> nth=0")
                d.to("#askSugg button >> nth=0"); pg.wait_for_timeout(450)
                d.click("#askSugg button >> nth=0", ms=250, pause=120)
                mark("7b_answer_clicked")
                pg.wait_for_function("() => !document.querySelector('#askSugg')", timeout=30000)
                continue
            if pg.query_selector("#confirmBtn") and not pg.query_selector("#rehearseBtn"):
                d.click("#confirmBtn")
            break
        if not pg.query_selector("#rehearseBtn"):
            raise SystemExit("no rehearse button: " + json.dumps(rec["steps"]))
        pg.wait_for_timeout(700)
        pg.evaluate(f"""(() => {{ const seg = document.querySelector('#budgetSeg');
            let x = seg.querySelector('button[data-b="{a.budget}"]');
            if (!x) {{ x = document.createElement('button'); x.dataset.b = '{a.budget}'; x.textContent = '{a.budget} s'; seg.appendChild(x); }}
            x.onclick = () => seg.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b === x)); }})()""")
        d.click(f'#budgetSeg button[data-b="{a.budget}"]'); pg.wait_for_timeout(300)
        d.click("#rehearseBtn"); mark("8_rehearse_futures_wall")
        d.park(1180, 600)
        pg.wait_for_selector("#verdictCard .vcard", timeout=900000)   # every attempt shown as it lands; nothing trimmed
        mark("9_verdict")
        pg.wait_for_timeout(9000)                                     # verdict beat + replay of the verified plan
        if pg.query_selector("#wallExpand"):
            mark("9b_all_futures_grid")
            d.click("#wallExpand"); pg.wait_for_timeout(5000); d.click("#wallExpand"); pg.wait_for_timeout(2500)
        mark("end")
        st = pg.evaluate("""() => { const S = window.__rehearsal; const run = S.runs[0] || {};
            return { verdict: run.verdict, attempts: (run.cands || []).length, budget: run.result?.budget,
                     via_direct: !!run.result?.via_direct, stress: run.result?.stress && {successes: run.result.stress.successes, n: run.result.stress.n},
                     models: (S.info || {}).models || null }; }""")
        vs = r60.stop()
        ctx.close(); b.close()
    rec.update(st); rec["verdict_path"] = ("direct_carry" if st["via_direct"] else "search") if st["verdict"] == "sent" else None
    rec.update({"video": vid.name, "video_stats": vs, "step_start_s": marks, "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    (out / f"walkthrough_{a.name}_timings.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("verdict", "verdict_path", "attempts", "budget", "stress", "step_start_s", "video_stats")}))


if __name__ == "__main__":
    main()
