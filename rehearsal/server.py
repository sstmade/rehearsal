"""Local web UI + JSON API (stdlib only):  python -m rehearsal.server  ->  http://localhost:8765

Env: REHEARSAL_HOST (default 127.0.0.1; the container sets 0.0.0.0), REHEARSAL_PORT (default 8765).
API reference: docs/api.md. Long work (rehearse, perceive) runs as jobs, see rehearsal/pipeline.py.
"""
from __future__ import annotations

import json
import mimetypes
mimetypes.add_type("model/gltf-binary", ".glb")
import os
import shutil
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import pipeline
from .pipeline import valid_name, validate_scene
from .scene import shape_bbox

ROOT = Path(__file__).resolve().parent.parent
HOST = os.environ.get("REHEARSAL_HOST", "127.0.0.1")
PORT = int(os.environ.get("REHEARSAL_PORT", 8765))
MAX_UPLOAD = 15 * 1024 * 1024
MAX_VIDEO = 30 * 1024 * 1024
MAX_JSON = 1024 * 1024
STATIC = ("out", "web", "photos", "uploads", "assets")
IMAGE_EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
MAX_STRESS, MAX_LOOK = 200, 5
# Public demo (HF Space): no uploads (they would call the Meta API), bundled scenes and their
# precomputed reports are read-only, one rehearsal queued at a time, smaller stress tests.
PUBLIC_DEMO = os.environ.get("REHEARSAL_PUBLIC_DEMO", "").strip().lower() not in ("", "0", "false", "no", "off")
PUBLIC_MAX_STRESS = 40


# scenes shipped with the app (read-only when PUBLIC_DEMO)
BUNDLED = frozenset(p.stem for p in (ROOT / "scenes").glob("*.json"))


class ApiError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg


# ----------------------------------------------------------------------------- helpers
def safe_join(base: Path, rel: str) -> Path | None:
    """base/rel if it stays inside base (no '..', no absolute paths, no hidden files)."""
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or p.startswith(".") or "\\" in p or "\x00" in p for p in parts):
        return None
    base = base.resolve()
    p = base.joinpath(*parts).resolve()
    return p if p.is_relative_to(base) and p.is_file() else None


def scene_path(root: Path, name) -> Path:
    if not valid_name(name):
        raise ApiError(400, "scene name must be 1-64 characters of [A-Za-z0-9_-]")
    return root / "scenes" / f"{name}.json"


def list_scenes(root: Path) -> list[str]:
    return sorted(p.stem for p in (root / "scenes").glob("*.json") if valid_name(p.stem))


def _thumb(root: Path, name: str, spec: dict) -> str | None:
    """URL of a thumbnail: first frame of attempt_1.gif, else the scene photo, else a top-down sketch."""
    out = root / "out" / name
    gif, png = out / "attempt_1.gif", out / "thumb.png"
    try:
        from PIL import Image, ImageDraw
        if gif.exists():
            if not png.exists() or png.stat().st_mtime < gif.stat().st_mtime:
                with Image.open(gif) as im:
                    im.seek(0)
                    im.convert("RGB").save(png)
            return f"/out/{name}/thumb.png?v={int(png.stat().st_mtime)}"
        photo = (spec.get("photo") or {}).get("file")
        if isinstance(photo, str) and photo.startswith(("photos/", "uploads/")) and \
                safe_join(root / photo.split("/")[0], photo.split("/", 1)[1]):
            return "/" + photo
        sk = root / "out" / "_thumbs" / f"{name}.png"
        src = root / "scenes" / f"{name}.json"
        if not sk.exists() or sk.stat().st_mtime < src.stat().st_mtime:
            sk.parent.mkdir(parents=True, exist_ok=True)
            W, H, pad = 320, 240, 12
            (cx, cy), (sx, sy) = spec["table"]["center"], spec["table"]["size"]
            k = min((W - 2 * pad) / sy, (H - 2 * pad) / sx)
            # Robot view: +x (away from robot) is up, +y is left.
            def px(x, y):
                return W / 2 - (y - cy) * k, H / 2 - (x - cx) * k
            im = Image.new("RGB", (W, H), (38, 38, 44))
            d = ImageDraw.Draw(im)
            x0, y0 = px(cx - sx / 2, cy + sy / 2)
            x1, y1 = px(cx + sx / 2, cy - sy / 2)
            d.rectangle([x0, y1, x1, y0], fill=(158, 128, 97))
            for o in spec["objects"]:
                u, v = px(*o["pos"])
                col = tuple(int(255 * c) for c in (o.get("rgba") or [0.7, 0.7, 0.7, 1])[:3])
                s = o["shape"]
                if s["type"] in ("cylinder", "bowl") and s.get("axis", "z") == "z":
                    r = s["radius"] * k
                    d.ellipse([u - r, v - r, u + r, v + r], fill=col, outline=(20, 20, 20))
                else:
                    bb = shape_bbox(s)
                    hx, hy = bb[1] * k / 2, bb[0] * k / 2
                    d.rectangle([u - hx, v - hy, u + hx, v + hy], fill=col, outline=(20, 20, 20))
            bx, by = px(0, 0)
            d.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill=(230, 230, 230))
            im.save(sk)
        return f"/out/_thumbs/{name}.png?v={int(sk.stat().st_mtime)}"
    except Exception:  # noqa: BLE001 - a broken thumbnail must not break the gallery
        return None


def gallery(root: Path) -> list[dict]:
    items = []
    for name in list_scenes(root):
        try:
            spec = json.loads((root / "scenes" / f"{name}.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        items.append({"scene": name, "title": spec.get("title") or name,
                      "thumb": _thumb(root, name, spec),
                      "has_report": (root / "out" / name / "report.json").exists(),
                      "demo_instruction": spec.get("demo_instruction"),
                      "privacy": spec.get("privacy", "private"),
                      "n_objects": len(spec.get("objects", []))})
    return items


def bench_summary(root: Path) -> dict:
    """Rehearsal Bench (bench/): the exported cases and the cached-plan leaderboard, for the landing."""
    cases = []
    for f in sorted((root / "bench" / "cases").glob("*/case.json")):
        try:
            m = json.loads(f.read_text())
        except Exception:  # noqa: BLE001 - one broken case must not hide the rest
            continue
        if not valid_name(m.get("id")) or m["id"] != f.parent.name:
            continue
        cases.append({"id": m["id"], "title": m.get("title", m["id"]),
                      "kind": (m.get("provenance") or {}).get("kind"),
                      "failures": (m.get("failures") or {}).get("count", 0),
                      "failed_runs": (m.get("failures") or {}).get("failed_runs_covered", 0),
                      "download": f"/api/bench/case?id={m['id']}"})
    lb = None
    p = root / "bench" / "leaderboard.json"
    if p.is_file():
        try:
            full = json.loads(p.read_text())
            lb = {k: full.get(k) for k in ("generated", "n", "seed0", "planners", "totals", "eval_totals")}
            lb["cases"] = [{"id": r["id"], "cells": {k: {"nominal": c["nominal"], "k": c["k"], "n": c["n"]}
                                                     for k, c in r["cells"].items() if c}} for r in full.get("cases", [])]
        except Exception:  # noqa: BLE001
            lb = None
    return {"n_cases": len(cases), "n_failures": sum(c["failures"] for c in cases),
            "n_failed_runs": sum(c["failed_runs"] for c in cases), "cases": cases, "leaderboard": lb}


def bench_zip(root: Path, case_id) -> bytes:
    """One case folder as a zip (<id>/case.json, scene.json, photo.jpg, failures.jsonl, plans/...)."""
    import io
    import zipfile
    if not valid_name(case_id):
        raise ApiError(400, "case id must be 1-64 characters of [A-Za-z0-9_-]")
    d = root / "bench" / "cases" / case_id
    if not (d / "case.json").is_file():
        raise ApiError(404, f"no bench case {case_id}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(d.rglob("*")):
            if f.is_file() and not any(part.startswith(".") for part in f.relative_to(d).parts):
                z.write(f, f"{case_id}/{f.relative_to(d).as_posix()}")
        schema = root / "bench" / "schema.json"
        if schema.is_file():
            z.write(schema, f"{case_id}/schema.json")
    return buf.getvalue()


def parse_multipart(ctype: str, body: bytes) -> dict:
    """{field: (filename | None, bytes)} using the stdlib email parser (cgi is gone in 3.13)."""
    msg = BytesParser(policy=HTTP).parsebytes(
        b"MIME-Version: 1.0\r\nContent-Type: " + ctype.encode("latin-1") + b"\r\n\r\n" + body)
    if not msg.is_multipart():
        raise ApiError(400, "expected multipart/form-data")
    fields = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            fields[name] = (part.get_filename(), part.get_payload(decode=True) or b"")
    return fields


def legacy_status(mgr: pipeline.JobManager) -> dict:
    j = mgr.latest("rehearse")
    if j is None:
        return {"running": False, "log": [], "error": None, "job": None, "state": None}
    return {"running": j["state"] in ("queued", "running"), "log": j["log"],
            "error": j["error"] and "\n".join([j["error"], *j["log"][-20:]]),
            "job": j["id"], "state": j["state"], "progress": j["progress"]}


# ----------------------------------------------------------------------------- handler
class Handler(BaseHTTPRequestHandler):
    server_version = "Rehearsal/1.0"

    @property
    def root(self) -> Path:
        return self.server.root

    @property
    def mgr(self) -> pipeline.JobManager:
        return self.server.mgr

    def log_message(self, *a):
        if os.environ.get("REHEARSAL_HTTP_LOG"):
            super().log_message(*a)

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _zip(self, body: bytes, filename: str):
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _err(self, code, msg):
        self._json({"error": msg}, code)

    def _body(self, limit: int) -> bytes:
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise ApiError(411, "Content-Length required")
        if n < 0:
            raise ApiError(400, "bad Content-Length")
        if n > limit:
            raise ApiError(413, f"body too large (limit {limit // (1024 * 1024)} MB)")
        return self.rfile.read(n)

    def _json_body(self):
        try:
            return json.loads(self._body(MAX_JSON) or b"null")
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, "body must be JSON")

    def _file(self, path: Path):
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in (".js", ".mjs"):
            ctype = "text/javascript"
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            with open(path, "rb") as f:
                shutil.copyfileobj(f, self.wfile)

    def _dispatch(self, fn):
        try:
            fn()
        except ApiError as e:
            self._err(e.code, e.msg)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            self._err(500, f"{type(e).__name__}: {e}")

    def do_GET(self):
        self._dispatch(self._get)

    def do_HEAD(self):
        self._dispatch(self._get)

    def do_POST(self):
        self._dispatch(self._post)

    # ------------------------------------------------------------------ GET
    def _get(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        path = url.path
        if path == "/":
            p = safe_join(self.root / "web", "index.html")
            return self._file(p) if p else self._err(404, "web/index.html missing")
        if path == "/api/bench/case":
            return self._zip(bench_zip(self.root, q.get("id")), f"rehearsal-bench-{q.get('id')}.zip")
        if path.startswith("/api/"):
            route = {"/api/bench": lambda: bench_summary(self.root),
                     "/api/scenes": lambda: list_scenes(self.root),
                     "/api/gallery": lambda: gallery(self.root),
                     "/api/scene": lambda: self._get_scene(q.get("name")),
                     "/api/job": lambda: self._get_job(q.get("id")),
                     "/api/jobs": self.mgr.list,
                     "/api/report": lambda: self._get_report(q.get("scene")),
                     "/api/budget": self._budget,
                     "/api/status": lambda: legacy_status(self.mgr)}.get(path)
            if route is None:
                return self._err(404, f"no such endpoint {path}")
            return self._json(route())
        top, _, rest = unquote(path).lstrip("/").partition("/")
        if top in STATIC:
            p = safe_join(self.root / top, rest)
            return self._file(p) if p else self._err(404, "not found")
        self._err(404, "not found")

    def _get_scene(self, name):
        p = scene_path(self.root, name)
        if not p.exists():
            raise ApiError(404, f"no scene {name}")
        return json.loads(p.read_text())

    def _get_job(self, jid):
        j = self.mgr.get(jid or "")
        if j is None:
            raise ApiError(404, f"no job {jid}")
        return j

    def _get_report(self, name):
        if not valid_name(name):
            raise ApiError(400, "scene name must be 1-64 characters of [A-Za-z0-9_-]")
        p = self.root / "out" / name / "report.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _budget(self):
        from .meta_client import status
        return status()

    # ------------------------------------------------------------------ POST
    def _post(self):
        path = urlparse(self.path).path
        route = {"/api/rehearse": self._rehearse, "/api/scene": self._save_scene,
                 "/api/perceive": self._perceive, "/api/reality": self._reality}.get(path)
        if route is None:
            return self._err(404, f"no such endpoint {path}")
        if PUBLIC_DEMO and path in ("/api/perceive", "/api/reality"):
            raise ApiError(403, "uploads are disabled on the public demo; run Rehearsal locally to try your own photo")
        self._json(route())

    def _rehearse(self):
        b = self._json_body()
        if not isinstance(b, dict):
            raise ApiError(400, "body must be a JSON object")
        scene = b.get("scene")
        if not scene_path(self.root, scene).exists():
            raise ApiError(404, f"no scene {scene}")
        instr = b.get("instruction")
        if not isinstance(instr, str) or not instr.strip() or len(instr) > 1000:
            raise ApiError(400, "instruction must be a non-empty string (max 1000 chars)")
        try:
            stress, look = int(b.get("stress", 40)), int(b.get("look", 2))
        except (TypeError, ValueError):
            raise ApiError(400, "stress and look must be integers")
        if not (0 <= stress <= MAX_STRESS and 0 <= look <= MAX_LOOK):
            raise ApiError(400, f"stress must be 0-{MAX_STRESS}, look 0-{MAX_LOOK}")
        if PUBLIC_DEMO:
            if (self.root / "out" / scene / "report.json").exists() and scene_path(self.root, scene).exists() \
                    and scene in BUNDLED:
                raise ApiError(403, f"{scene} is a precomputed demo scene and read-only on the public demo")
            if stress > PUBLIC_MAX_STRESS:
                raise ApiError(400, f"stress is limited to {PUBLIC_MAX_STRESS} on the public demo")
            with self.mgr.lock:
                busy = sum(1 for j in self.mgr.jobs.values() if j.state in ("queued", "running"))
            if busy:
                raise ApiError(429, "the public demo runs one rehearsal at a time; try again in a minute")
        params = {"scene": scene, "instruction": instr.strip(), "stress": stress, "look": look}
        jid = self.mgr.submit("rehearse", pipeline.rehearse_job(self.root, scene, instr.strip(), stress, look),
                              params)
        return {"job": jid, "started": True}

    def _save_scene(self):
        spec = self._json_body()
        problems = validate_scene(spec)
        if problems:
            raise ApiError(400, "invalid scene: " + "; ".join(problems))
        if PUBLIC_DEMO and spec["name"] in BUNDLED:
            raise ApiError(403, f"{spec['name']} is a bundled demo scene; save your edit under a new name")
        (self.root / "scenes").mkdir(exist_ok=True)
        scene_path(self.root, spec["name"]).write_text(json.dumps(spec, ensure_ascii=False, indent=1))
        return {"name": spec["name"]}

    def _perceive(self):
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            raise ApiError(400, "expected multipart/form-data with a 'photo' field")
        fields = parse_multipart(ctype, self._body(MAX_UPLOAD))
        if "photo" not in fields or not fields["photo"][1]:
            raise ApiError(400, "missing 'photo' file")
        data = fields["photo"][1]
        try:
            import io
            from PIL import Image
            with Image.open(io.BytesIO(data)) as im:
                fmt = im.format
                im.verify()
        except Exception:  # noqa: BLE001
            raise ApiError(400, "photo is not a readable image")
        if fmt not in IMAGE_EXT:
            raise ApiError(400, f"unsupported image format {fmt}; use JPEG, PNG or WebP")
        privacy = (fields.get("privacy", (None, b"private"))[1].decode(errors="replace").strip() or "private")
        if privacy not in ("public", "private"):
            raise ApiError(400, 'privacy must be "public" or "private"')
        corners = None
        if fields.get("paper_corners") and fields["paper_corners"][1].strip():
            try:
                corners = json.loads(fields["paper_corners"][1])
            except ValueError:
                raise ApiError(400, "paper_corners must be JSON [[u,v] x4]")
            if not (isinstance(corners, list) and len(corners) == 4
                    and all(pipeline._vec(c, 2) for c in corners)):
                raise ApiError(400, "paper_corners must be JSON [[u,v] x4]")
        robot_side = (fields.get("robot_side", (None, b"bottom"))[1].decode(errors="replace").strip() or "bottom")
        if robot_side not in ("bottom", "top", "left", "right"):
            raise ApiError(400, 'robot_side must be one of "bottom", "top", "left", "right"')
        up = self.root / "uploads"
        up.mkdir(exist_ok=True)
        holder = {}

        def work(job):
            return pipeline.perceive_job(self.root, holder["path"], privacy, corners, robot_side)(job)

        # Name the file after the job id: submit first, then write before the worker can need it.
        # The worker reads holder["path"]; guard with the manager lock so it is set before it runs.
        with self.mgr.lock:
            jid = self.mgr.submit("perceive", work, {"privacy": privacy, "corners": corners,
                                                      "robot_side": robot_side})
            holder["path"] = up / f"{jid}{IMAGE_EXT[fmt]}"
            holder["path"].write_bytes(data)
        return {"job": jid, "upload": f"/uploads/{holder['path'].name}"}

    def _reality(self):
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            raise ApiError(400, "expected multipart/form-data with 'video' and 'scene'")
        fields = parse_multipart(ctype, self._body(MAX_VIDEO))
        text = {k: v[1].decode(errors="replace").strip() for k, v in fields.items() if k != "video"}
        scene = text.get("scene")
        scene_path(self.root, scene)  # validates the name
        if not (self.root / "out" / scene / "report.json").exists():
            raise ApiError(404, f"no rehearsal report for {scene}; rehearse it first")
        attempt = None
        if text.get("attempt"):
            if not text["attempt"].isdigit() or int(text["attempt"]) < 1:
                raise ApiError(400, "attempt must be a positive integer")
            attempt = int(text["attempt"])
        try:
            pipeline.pick_attempt(json.loads((self.root / "out" / scene / "report.json").read_text()), attempt)
        except ValueError as e:
            raise ApiError(400, str(e))
        twin = text.get("twin") or None
        if twin is not None and not scene_path(self.root, twin).exists():
            raise ApiError(404, f"no scene {twin}")
        privacy = text.get("privacy") or "private"
        if privacy not in ("public", "private"):
            raise ApiError(400, 'privacy must be "public" or "private"')
        if "video" not in fields or not fields["video"][1]:
            raise ApiError(400, "missing 'video' file")
        data = fields["video"][1]
        ext = video_ext(data)
        if ext is None:
            raise ApiError(400, "video must be MP4, MOV or WebM")
        up = self.root / "uploads"
        up.mkdir(exist_ok=True)
        holder = {}

        def work(job):
            return pipeline.reality_job(self.root, scene, holder["path"], privacy, attempt, twin)(job)

        with self.mgr.lock:
            jid = self.mgr.submit("reality", work, {"scene": scene, "attempt": attempt, "twin": twin,
                                                    "privacy": privacy})
            holder["path"] = up / f"{jid}{ext}"
            holder["path"].write_bytes(data)
        return {"job": jid, "upload": f"/uploads/{holder['path'].name}"}


def video_ext(data: bytes) -> str | None:
    """.mp4 / .mov (ISO BMFF 'ftyp' box) or .webm (EBML magic); None if neither."""
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return ".mov" if data[8:12] == b"qt  " else ".mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    return None


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, root: Path = ROOT, mgr: pipeline.JobManager | None = None):
        super().__init__(addr, Handler)
        self.root = Path(root)
        self.mgr = mgr or pipeline.manager()


def make_server(host: str = HOST, port: int = PORT, root: Path = ROOT, mgr=None) -> Server:
    return Server((host, port), root, mgr)


def main():
    srv = make_server()
    h, p = srv.server_address[:2]
    print(f"Rehearsal UI on http://{'localhost' if h in ('127.0.0.1', '0.0.0.0') else h}:{p}  (bound to {h})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
