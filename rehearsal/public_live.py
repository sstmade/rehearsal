"""Live Meta API calls on the public demo, for visitor UPLOADS only.

Enabled by REHEARSAL_PUBLIC_LIVE=1 together with REHEARSAL_PUBLIC_DEMO=1 (and a MODEL_API_KEY). The sample tables
keep their recorded perception and replayed Muse Spark answers; only a photo the visitor uploads (and the instructions
given on that photo's twin) reach the Meta API. Guard rails:

- hard cap: REHEARSAL_BUDGET_CAP (meta_client checks the ledger before every paid call); new live work is refused
  once less than one request ceiling is left, and the app says the live budget is used up
- the ledger lives in /data/rehearsal_budget when the host has persistent storage (HF Spaces /data), so a restart
  does not reset the spend; otherwise in REHEARSAL_BUDGET_DIR / budget/ (resets with the container)
- per request ceiling (REHEARSAL_REQUEST_CEILING, default $0.25): a call that would push one request past it is
  refused before it is sent
- per-IP limit (in memory): 3 uploads per hour, 10 per day; at most REHEARSAL_LIVE_CONCURRENCY (default 1) live
  photo pipelines at a time; uploads up to 8 MB, downscaled to <= 1600 px before any call
- uploaded photos and the scenes made from them are deleted after one hour (the original upload right after the
  downscale)
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


PER_HOUR = int(_num("REHEARSAL_UPLOADS_PER_HOUR", 3))
PER_DAY = int(_num("REHEARSAL_UPLOADS_PER_DAY", 10))
CONCURRENCY = max(1, int(_num("REHEARSAL_LIVE_CONCURRENCY", 1)))
MAX_UPLOAD = 8 * 1024 * 1024
RETENTION_S = 3600

BUDGET_OUT_MSG = ("The live Meta API budget for this demo is used up, so it is showing recorded demos now. "
                  "Every sample table below still runs live planning and physics.")


def request_ceiling() -> float:
    return _num("REHEARSAL_REQUEST_CEILING", 0.25)


def enabled() -> bool:
    return _flag("REHEARSAL_PUBLIC_DEMO") and _flag("REHEARSAL_PUBLIC_LIVE")


def is_upload_scene(sid: str) -> bool:
    return isinstance(sid, str) and sid.startswith("photo_")


# ---------------------------------------------------------------------------------------------- budget
def use_persistent_ledger(data_root: str = "/data") -> str | None:
    """Point the ledger at persistent storage when the host has it (HF Spaces mounts /data). Leaves an explicit
    REHEARSAL_BUDGET_DIR alone. Returns the directory used, or None (ledger resets with the container)."""
    if os.environ.get("REHEARSAL_BUDGET_DIR"):
        return os.environ["REHEARSAL_BUDGET_DIR"]
    root = Path(data_root)
    if root.is_dir() and os.access(root, os.W_OK):
        d = root / "rehearsal_budget"
        d.mkdir(exist_ok=True)
        os.environ["REHEARSAL_BUDGET_DIR"] = str(d)
        return str(d)
    return None


def budget_remaining() -> float | None:
    try:
        from .meta_client import status
        return status()["remaining"]
    except Exception:  # noqa: BLE001
        return None


def budget_ok() -> bool:
    """Room for at least one more worst-case request under the hard cap."""
    r = budget_remaining()
    return r is not None and r >= request_ceiling()


# ---------------------------------------------------------------------------------------------- abuse limits
class RateLimiter:
    def __init__(self, per_hour: int = PER_HOUR, per_day: int = PER_DAY):
        self.per_hour, self.per_day = per_hour, per_day
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, ip: str, now: float | None = None) -> str | None:
        """Record an upload from ip; returns None when allowed, else the reason it is refused."""
        now = time.time() if now is None else now
        with self.lock:
            q = self.hits[ip]
            while q and now - q[0] > 86400:
                q.popleft()
            if len(q) >= self.per_day:
                return f"This demo allows {self.per_day} photo uploads per visitor per day. Try a sample table, or come back tomorrow."
            if sum(1 for t in q if now - t <= 3600) >= self.per_hour:
                return f"This demo allows {self.per_hour} photo uploads per visitor per hour. Try a sample table meanwhile."
            q.append(now)
            return None


LIMITER = RateLimiter()
_slots = threading.BoundedSemaphore(CONCURRENCY)


def try_slot() -> bool:
    return _slots.acquire(blocking=False)


def release_slot():
    try:
        _slots.release()
    except ValueError:
        pass


def client_ip(handler) -> str:
    fwd = handler.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else "") or handler.client_address[0]


# ---------------------------------------------------------------------------------------------- privacy
def purge_uploads(root: Path, max_age: float = RETENTION_S, now: float | None = None) -> int:
    """Delete uploaded photos (and everything derived from them) older than max_age seconds."""
    now = time.time() if now is None else now
    n = 0
    targets = [p for p in (root / "uploads").glob("*") if p.is_file()] if (root / "uploads").is_dir() else []
    targets += list((root / "scenes").glob("photo_*.json")) + list((root / "scenes" / "_models").glob("photo_*.json"))
    for p in targets:
        try:
            if now - p.stat().st_mtime > max_age:
                p.unlink(); n += 1
        except FileNotFoundError:
            pass
    return n


def start_purger(root: Path, every_s: float = 300):
    def loop():
        while True:
            try:
                purge_uploads(root)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(every_s)
    threading.Thread(target=loop, name="upload-purger", daemon=True).start()
