"""Local web UI: python -m rehearsal.server  ->  http://localhost:8765"""
from __future__ import annotations

import json
import threading
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .rehearse import rehearse

ROOT = Path(__file__).resolve().parent.parent
PORT = 8765

state = {"running": False, "log": [], "error": None}
lock = threading.Lock()


def _run(scene: str, instruction: str, stress: int):
    try:
        rehearse(str(ROOT / "scenes" / f"{scene}.json"), instruction, stress, str(ROOT / "out"),
                 log=lambda s: state["log"].append(s))
    except Exception:
        state["error"] = traceback.format_exc()
    finally:
        state["running"] = False


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            self.path = "/web/index.html"
            return super().do_GET()
        if url.path == "/api/scenes":
            return self._json(sorted(p.stem for p in (ROOT / "scenes").glob("*.json")))
        if url.path == "/api/status":
            return self._json(state)
        if url.path == "/api/report":
            p = ROOT / "out" / q.get("scene", [""])[0] / "report.json"
            return self._json(json.loads(p.read_text()) if p.exists() else None)
        if url.path.startswith(("/out/", "/web/")):
            return super().do_GET()
        self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/rehearse":
            return self.send_error(404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with lock:
            if state["running"]:
                return self._json({"error": "a rehearsal is already running"}, 409)
            state.update(running=True, log=[], error=None)
        threading.Thread(target=_run, args=(body["scene"], body["instruction"], int(body.get("stress", 40))),
                         daemon=True).start()
        self._json({"started": True})


def main():
    print(f"Rehearsal UI on http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
