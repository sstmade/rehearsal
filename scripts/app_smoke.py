"""Headless smoke test of the /app product flow (Playwright, Chromium).

    .venv/bin/python scripts/app_smoke.py [--url http://127.0.0.1:8765] [--runs 3] [--shots DIR] [--budget 60]

Flow on a public sample table, with recorded (cached, $0) Muse Spark answers:
  pick sample -> perception HUD -> build twin -> compile instruction -> rehearse -> verdict + attempt wall + replay,
then the ASK path on a second table (clarifying question shown, reply box present).
Exits 0 only if every run passes. Without --url it starts its own server on a free port. Wrap it in tools/heavy.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ("t6_41", "Put the stones into the bowl.")
ASK = ("t6_11", "Move the brush jar away from the sketchbook.")


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def start_server():
    port = free_port()
    env = dict(os.environ, REHEARSAL_PORT=str(port))
    p = subprocess.Popen([sys.executable, "-m", "rehearsal.server"], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(url + "/api/app/config", timeout=2); return p, url
        except OSError:
            time.sleep(0.5)
    p.kill(); raise SystemExit("server did not start")


def spent(url) -> float:
    return json.load(urllib.request.urlopen(url + "/api/app/config"))["budget"]["spent"]


W, H = 1920, 1080


def run_once(pw, url, k, shots, budget) -> list[str]:
    errs: list[str] = []
    b = pw.chromium.launch()
    pg = b.new_page(viewport={"width": W, "height": H})
    pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))

    def shot(name):
        if shots:
            pg.screenshot(path=str(Path(shots) / f"smoke{k}_{name}.png"))

    def step(name, fn):
        t = time.time()
        fn()
        print(f"  [{k}] {name:<34} {time.time() - t:6.1f} s", flush=True)

    try:
        sid, instr = MAIN
        step("samples grid", lambda: pg.goto(url + "/app") or pg.wait_for_selector(f'.card[data-id="{sid}"]', timeout=30000))
        shot("1_pick")
        step("perception HUD", lambda: (pg.click(f'.card[data-id="{sid}"]'), pg.wait_for_selector("#buildBtn", timeout=30000),
                                        pg.wait_for_function("document.querySelector('#percSub').textContent.match(/(\\d+)\\/\\1/)", timeout=30000)))
        shot("2_perceive")
        step("3D twin", lambda: (pg.click("#buildBtn"), pg.wait_for_selector("#instr", timeout=90000)))
        pg.wait_for_timeout(1500); shot("3_twin")
        step("intent compile", lambda: (pg.fill("#instr", instr), pg.click("#sendBtn"),
                                        pg.wait_for_selector("#rehearseBtn, .bub.no, #replyBtn", timeout=120000)))
        shot("4_goals")
        if not pg.query_selector("#rehearseBtn"):
            raise AssertionError("intent did not produce goals: " + pg.inner_text("#intentOut")[:200])
        if budget:
            pg.evaluate(f"(() => {{ const b = document.querySelector('#budgetSeg button[data-b=\"{budget}\"]'); b && b.click(); }})()")
        step("rehearse (search + physics)", lambda: (pg.click("#rehearseBtn"),
                                                     pg.wait_for_selector("#verdictCard .vcard", timeout=420000)))
        pg.wait_for_timeout(700); shot("5a_bigverdict")
        pg.wait_for_timeout(2500); shot("5_verdict")
        tiles = pg.eval_on_selector_all(".tile", "els => els.length")
        verdict = pg.inner_text("#verdict")
        print(f"  [{k}] verdict: {verdict.splitlines()[0] if verdict else '?'} | attempt tiles: {tiles}")
        if tiles < 1:
            raise AssertionError("attempt wall is empty")
        if not pg.is_visible("#replayBar"):
            raise AssertionError("replay bar not shown")
        pg.click(".tile >> nth=-1"); pg.wait_for_timeout(1500); shot("6_attempt")

        sid, instr = ASK
        step("ASK path", lambda: (pg.goto(f"{url}/app?scene={sid}"), pg.wait_for_selector("#buildBtn", timeout=30000),
                                  pg.click("#buildBtn"), pg.wait_for_selector("#instr", timeout=90000),
                                  pg.fill("#instr", instr), pg.click("#sendBtn"),
                                  pg.wait_for_selector("#replyBtn, #rehearseBtn, .bub.no", timeout=120000)))
        shot("7_ask")
        if not pg.query_selector("#replyBtn"):
            raise AssertionError("expected a clarifying question")
        if not pg.query_selector("#askSugg button"):
            raise AssertionError("no suggested answer offered for the clarifying question")
        step("ASK suggested answer", lambda: (pg.click("#askSugg button >> nth=0"),
                                              pg.wait_for_selector("#rehearseBtn, .bub.no", timeout=120000)))
        shot("8_answered")
        if not pg.query_selector("#rehearseBtn"):
            raise AssertionError("suggested answer did not compile to goals: " + pg.inner_text("#intentOut")[:200])
    except Exception as e:  # noqa: BLE001
        errs.append(f"{type(e).__name__}: {str(e).splitlines()[0][:300]}")
        try:
            shot("FAIL")
        except Exception:  # noqa: BLE001
            pass
    finally:
        b.close()
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url"); ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--shots"); ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--width", type=int, default=1920); ap.add_argument("--height", type=int, default=1080)
    a = ap.parse_args()
    global W, H
    W, H = a.width, a.height
    from playwright.sync_api import sync_playwright
    srv = None
    url = a.url
    if not url:
        srv, url = start_server()
    if a.shots:
        Path(a.shots).mkdir(parents=True, exist_ok=True)
    s0 = spent(url)
    fails = 0
    try:
        with sync_playwright() as pw:
            for k in range(1, a.runs + 1):
                t = time.time()
                errs = run_once(pw, url, k, a.shots, a.budget)
                print(f"run {k}: {'PASS' if not errs else 'FAIL'} in {time.time() - t:.0f} s" + "".join(f"\n    {e}" for e in errs))
                fails += bool(errs)
    finally:
        print(f"Meta spend during smoke test: ${spent(url) - s0:.4f}")
        if srv:
            srv.terminate()
    print(f"{a.runs - fails}/{a.runs} runs passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
