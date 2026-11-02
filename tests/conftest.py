import http.client
import json
import os
import shutil
import threading
import uuid
from pathlib import Path

import pytest

os.environ["REHEARSAL_OFFLINE"] = "1"          # never call Muse Spark from tests
os.environ["REHEARSAL_WORKERS"] = "3"   # shared, overloaded machine
os.environ.setdefault("REHEARSAL_ROBUST_N", "0")   # hybrid is the default planner; its 20-seed gate is slow

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
DESK_FIXTURE = FIXTURES / "desk_01.json"   # frozen; tests never depend on the live scenes/


@pytest.fixture
def root(tmp_path):
    """A throwaway project root: copies of scenes/ and web/index.html, empty out/."""
    (tmp_path / "scenes").mkdir()
    shutil.copy(DESK_FIXTURE, tmp_path / "scenes" / "desk_01.json")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "index.html").write_text("<!doctype html><title>t</title>")
    (tmp_path / "photos").mkdir()
    (tmp_path / "out").mkdir()
    (tmp_path / "secret.txt").write_text("top secret")
    return tmp_path


@pytest.fixture
def budget_dir(tmp_path, monkeypatch):
    d = tmp_path / "budget"
    monkeypatch.setenv("REHEARSAL_BUDGET_DIR", str(d))
    return d


class Client:
    def __init__(self, port):
        self.port = port

    def raw(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, r.getheader("Content-Type", ""), data

    def get(self, path):
        s, ct, d = self.raw("GET", path)
        return s, (json.loads(d) if ct.startswith("application/json") else d)

    def post(self, path, obj):
        s, ct, d = self.raw("POST", path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
        return s, json.loads(d)

    def multipart(self, path, fields):
        """fields: {name: str | (filename, bytes, ctype)}"""
        b = uuid.uuid4().hex
        parts = []
        for k, v in fields.items():
            if isinstance(v, tuple):
                fn, data, ct = v
                parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{fn}"\r\n'
                             f"Content-Type: {ct}\r\n\r\n".encode() + data + b"\r\n")
            else:
                parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        body = b"".join(parts) + f"--{b}--\r\n".encode()
        s, ct, d = self.raw("POST", path, body, {"Content-Type": f"multipart/form-data; boundary={b}"})
        return s, json.loads(d)


@pytest.fixture
def server(root, budget_dir):
    from rehearsal.pipeline import JobManager
    from rehearsal.server import make_server
    mgr = JobManager()
    srv = make_server("127.0.0.1", 0, root=root, mgr=mgr)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield Client(srv.server_address[1]), srv
    srv.shutdown()
    srv.server_close()
