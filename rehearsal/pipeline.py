"""In-process job system and the work the web server hands to it.

Jobs are kept in memory (lost on restart). Each kind has its own worker pool:
    rehearse  -> 1 worker  (CPU heavy: the stress test already fans out over all cores)
    perceive  -> 2 workers (mostly waiting on the network)
Anything else submitted under an unknown kind gets a 1-worker pool of its own.

A job function receives a `JobHandle` and returns a JSON-serialisable result:

    def work(job):
        job.log("hello")
        job.progress("simulate", 0.5)
        return {"ok": True}
    jid = manager().submit("rehearse", work)
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
POOL_SIZES = {"rehearse": 1, "perceive": 2}
MAX_JOBS = 200          # finished jobs beyond this are forgotten, oldest first
MAX_LOG_LINES = 5000


def valid_name(name) -> bool:
    return isinstance(name, str) and bool(NAME_RE.match(name))


# ------------------------------------------------------------------------------------- jobs
class Job:
    def __init__(self, kind: str, params: dict | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.params = params or {}
        self.state = "queued"
        self.log: list[str] = []
        self.result = None
        self.error: str | None = None
        self.progress = {"phase": "queued", "fraction": 0.0}
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None


class JobHandle:
    """What a job function sees: append log lines and report progress."""

    def __init__(self, mgr: "JobManager", job: Job):
        self._mgr, self._job = mgr, job
        self.id, self.params = job.id, job.params

    def log(self, msg) -> None:
        with self._mgr.lock:
            lines = str(msg).split("\n")
            self._job.log.extend(lines)
            if len(self._job.log) > MAX_LOG_LINES:
                del self._job.log[: len(self._job.log) - MAX_LOG_LINES]

    def progress(self, phase: str, fraction: float) -> None:
        with self._mgr.lock:
            self._job.progress = {"phase": str(phase), "fraction": round(min(max(float(fraction), 0.0), 1.0), 3)}


class JobManager:
    def __init__(self, pool_sizes: dict | None = None):
        self.lock = threading.RLock()
        self.jobs: "OrderedDict[str, Job]" = OrderedDict()
        self.sizes = dict(POOL_SIZES, **(pool_sizes or {}))
        self.pools: dict[str, ThreadPoolExecutor] = {}

    def _pool(self, kind: str) -> ThreadPoolExecutor:
        if kind not in self.pools:
            self.pools[kind] = ThreadPoolExecutor(self.sizes.get(kind, 1), thread_name_prefix=f"job-{kind}")
        return self.pools[kind]

    def submit(self, kind: str, fn, params: dict | None = None) -> str:
        job = Job(kind, params)
        with self.lock:
            self.jobs[job.id] = job
            self._prune()
            self._pool(kind).submit(self._run, job, fn)
        return job.id

    def _run(self, job: Job, fn):
        with self.lock:
            job.state, job.started = "running", time.time()
            job.progress = {"phase": "starting", "fraction": 0.0}
        try:
            result = fn(JobHandle(self, job))
            with self.lock:
                job.result, job.state = result, "done"
                job.progress = {"phase": "done", "fraction": 1.0}
        except Exception as e:  # noqa: BLE001 - every failure must land in the job
            tb = traceback.format_exc()
            with self.lock:
                job.error = f"{type(e).__name__}: {e}"
                job.log.extend(tb.rstrip().split("\n"))
                job.state = "error"
        finally:
            with self.lock:
                job.finished = time.time()

    def _prune(self):
        done = [j for j in self.jobs.values() if j.state in ("done", "error")]
        for j in done[: max(0, len(self.jobs) - MAX_JOBS)]:
            self.jobs.pop(j.id, None)

    def get(self, jid: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(jid)
            return None if job is None else self._view(job)

    def _view(self, job: Job) -> dict:
        ahead = 0
        if job.state == "queued":
            ahead = sum(1 for j in self.jobs.values() if j.kind == job.kind and j is not job
                        and j.state in ("queued", "running") and j.created <= job.created)
        return {"id": job.id, "kind": job.kind, "state": job.state, "log": list(job.log),
                "result": job.result, "error": job.error, "progress": dict(job.progress),
                "params": job.params, "queue_ahead": ahead, "created": job.created,
                "started": job.started, "finished": job.finished}

    def latest(self, kind: str) -> dict | None:
        with self.lock:
            for job in reversed(self.jobs.values()):
                if job.kind == kind:
                    return self._view(job)
        return None

    def list(self) -> list[dict]:
        with self.lock:
            return [{k: v for k, v in self._view(j).items() if k != "log"} for j in reversed(self.jobs.values())]

    def wait(self, jid: str, timeout: float = 60.0) -> dict | None:
        """Block until a job finishes (tests and CLI helpers)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            v = self.get(jid)
            if v is None or v["state"] in ("done", "error"):
                return v
            time.sleep(0.05)
        return self.get(jid)


_MANAGER: JobManager | None = None
_MANAGER_LOCK = threading.Lock()


def manager() -> JobManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = JobManager()
        return _MANAGER


# --------------------------------------------------------------------------- scene validation
SHAPES = {"box", "cylinder", "bowl", "compound"}


def _shape_errors(s, where: str, allowed=("box", "cylinder", "bowl")) -> list[str]:
    """Validation of one primitive (box / cylinder / bowl; cylinders may lie on their side via "axis")."""
    if not isinstance(s, dict) or s.get("type") not in allowed:
        return [f"{where}.type must be one of {sorted(allowed)}"]
    if s["type"] == "box":
        if not _vec(s.get("size"), 3) or not all(0.005 <= v <= MAX_DIM_M for v in s["size"]):
            return [f"{where}.size must be [x, y, z], each 0.005-{MAX_DIM_M} m"]
        return []
    if not (_num(s.get("radius")) and _num(s.get("height"))
            and 0.005 <= s["radius"] <= MAX_DIM_M / 2 and 0.005 <= s["height"] <= MAX_DIM_M):
        return [f"{where} needs radius 0.005-{MAX_DIM_M / 2} m and height 0.005-{MAX_DIM_M} m"]
    if s.get("axis", "z") not in ("x", "y", "z"):
        return [f"{where}.axis must be x, y or z"]
    return []
RESERVED_IDS = {"world", "table", "floor", "robot_arm", "hand", "left_finger", "right_finger",
                *(f"link{i}" for i in range(8))}
MAX_OBJECTS = 40
MAX_DIM_M = 1.0


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _vec(x, n) -> bool:
    return isinstance(x, list) and len(x) == n and all(_num(v) for v in x)


def validate_scene(scene) -> list[str]:
    """Return a list of human-readable problems; empty means the scene can be simulated."""
    errs: list[str] = []
    if not isinstance(scene, dict):
        return ["scene must be a JSON object"]
    if not valid_name(scene.get("name")):
        errs.append("name must be 1-64 characters of [A-Za-z0-9_-]")
    if "privacy" in scene and scene["privacy"] not in ("public", "private"):
        errs.append('privacy must be "public" or "private"')
    t = scene.get("table")
    tb = None
    if not isinstance(t, dict) or not _vec(t.get("center"), 2) or not _vec(t.get("size"), 2):
        errs.append("table must have center [x, y] and size [sx, sy] (numbers, meters)")
    elif not all(0.1 <= s <= 5.0 for s in t["size"]):
        errs.append("table size must be between 0.1 and 5 m per side")
    elif not all(abs(c) <= 5.0 for c in t["center"]):
        errs.append("table center must be within 5 m of the robot")
    else:
        (cx, cy), (sx, sy) = t["center"], t["size"]
        tb = (cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2)

    objs = scene.get("objects")
    if not isinstance(objs, list) or not objs:
        return errs + ["objects must be a non-empty list"]
    if len(objs) > MAX_OBJECTS:
        errs.append(f"at most {MAX_OBJECTS} objects")
    seen = set()
    for i, o in enumerate(objs):
        where = f"objects[{i}]"
        if not isinstance(o, dict):
            errs.append(f"{where} must be an object")
            continue
        oid = o.get("id")
        if not valid_name(oid):
            errs.append(f"{where}.id must be 1-64 characters of [A-Za-z0-9_-]")
        else:
            where = f"object {oid}"
            if oid in seen:
                errs.append(f"duplicate object id {oid}")
            if oid in RESERVED_IDS:
                errs.append(f"object id {oid} is reserved")
            seen.add(oid)
        if "label" in o and not isinstance(o["label"], str):
            errs.append(f"{where}.label must be a string")
        s = o.get("shape")
        if not isinstance(s, dict) or s.get("type") not in SHAPES:
            errs.append(f"{where}.shape.type must be one of {sorted(SHAPES)}")
        elif s["type"] == "compound":
            parts = s.get("parts")
            if not isinstance(parts, list) or not parts or len(parts) > 16:
                errs.append(f"{where}.shape.parts must be a list of 1-16 box/cylinder parts")
            else:
                for k, part in enumerate(parts):
                    e = _shape_errors(part, f"{where}.shape.parts[{k}]", ("box", "cylinder"))
                    if not e and part.get("pos") is not None and not _vec(part["pos"], 3):
                        e = [f"{where}.shape.parts[{k}].pos must be [x, y, z]"]
                    errs += e
            if s.get("bbox") is not None and not (_vec(s["bbox"], 3) and all(0.005 <= v <= MAX_DIM_M for v in s["bbox"])):
                errs.append(f"{where}.shape.bbox must be [x, y, z], each 0.005-{MAX_DIM_M} m")
        else:
            errs += _shape_errors(s, f"{where}.shape", ("box", "cylinder", "bowl"))
        if not _vec(o.get("pos"), 2):
            errs.append(f"{where}.pos must be [x, y]")
        elif tb and not (tb[0] <= o["pos"][0] <= tb[1] and tb[2] <= o["pos"][1] <= tb[3]):
            errs.append(f"{where}.pos {o['pos']} is not on the table")
        if "yaw" in o and not _num(o["yaw"]):
            errs.append(f"{where}.yaw must be a number")
        m = o.get("mass_kg")
        if not isinstance(m, dict) or not _num(m.get("nominal")) or not 0.001 <= m["nominal"] <= 50:
            errs.append(f"{where}.mass_kg.nominal must be 0.001-50 kg")
        f = o.get("friction")
        if f is not None and (not isinstance(f, dict) or not _num(f.get("nominal")) or not 0 < f["nominal"] <= 3):
            errs.append(f"{where}.friction.nominal must be in (0, 3]")
        if "rgba" in o and not (_vec(o["rgba"], 4) and all(0 <= v <= 1 for v in o["rgba"])):
            errs.append(f"{where}.rgba must be 4 numbers in [0, 1]")
    return errs


# ---------------------------------------------------------------------------- job functions
def rehearse_job(root: Path, scene: str, instruction: str, stress: int, look: int, planner: str = "hybrid",
                 confirmed: bool = False):
    """Returns a job function that runs one rehearsal into <root>/out/<scene>/."""

    def work(job: JobHandle):
        from .rehearse import rehearse
        rep = rehearse(str(root / "scenes" / f"{scene}.json"), instruction, stress, str(root / "out"),
                       log=job.log, look_rounds=look, progress=job.progress, planner=planner, confirmed=confirmed)
        st = rep.get("stress_test")
        it = rep.get("intent") or {}
        return {"scene": rep["scene"], "report": f"/api/report?scene={rep['scene']}",
                "status": rep.get("status", "rehearsed"), "message": rep.get("message"),
                "success": bool(rep.get("status", "rehearsed") == "rehearsed"
                                and rep["attempts"] and rep["attempts"][-1]["success"]),
                "attempts": len(rep["attempts"]), "agent": rep["agent"], "planner": rep.get("planner", planner),
                "assumptions": it.get("assumptions", []), "question": it.get("question", ""),
                "reason": it.get("reason", ""),
                "stress": None if not st else {"successes": st["successes"], "n": st["n"]},
                "wall_time_s": rep["wall_time_s"]}

    return work


def intent_job(root: Path, scene: str, instruction: str):
    """Compile the instruction and run the deterministic intent gate; nothing is planned. Runs on the 2-worker
    "perceive" pool (it mostly waits on the model; the gate's vacuity check is one 1.5 s settle)."""

    def work(job: JobHandle):
        from .agents import make_agent, scene_summary
        from .executor import reachability_report
        from .rehearse import compile_step
        from .scene import load_scene
        spec = load_scene(str(root / "scenes" / f"{scene}.json"))
        agent = make_agent(data="public" if spec.get("privacy") == "public" else "private", scene=spec)
        job.progress("compile", 0.2)
        intent = compile_step(agent, spec, instruction, scene_summary(spec, reachability_report(spec)), log=job.log)
        outcome = intent.pop("outcome")
        return {"scene": scene, "instruction": instruction, "agent": agent.name,
                "status": "ready" if outcome == "rehearsed" else outcome, "intent": intent}

    return work


def _perceive_fn():
    """Lazy import so the server starts even while perception is being rewritten."""
    try:
        from . import perception
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"perception module failed to import: {type(e).__name__}: {e}") from e
    fn = getattr(perception, "perceive_photo", None)
    if fn is None:
        raise RuntimeError("perception.perceive_photo(path, privacy, corners=None) is not available "
                           "yet in this build; photo upload is disabled")
    return fn


def perceive_job(root: Path, photo: Path, privacy: str, corners, robot_side: str = "bottom"):
    def work(job: JobHandle):
        fn = _perceive_fn()
        job.log(f"perceiving {photo.name} (data={privacy})")
        job.progress("perceive", 0.1)
        try:
            scene, overlay = fn(str(photo), privacy, corners=corners, robot_side=robot_side)
        except TypeError:  # test doubles with the original 3-argument signature
            scene, overlay = fn(str(photo), privacy, corners=corners)
        job.progress("validate", 0.9)
        if not isinstance(scene, dict):
            raise RuntimeError("perceive_photo returned no scene")
        name = f"photo_{job.id}"
        scene = dict(scene, name=name)
        scene["privacy"] = privacy  # the uploader decides, never the perception output
        overlay_url = None
        if overlay and Path(overlay).is_file():
            dst = root / "uploads" / f"{job.id}_overlay{Path(overlay).suffix or '.png'}"
            if Path(overlay).resolve() != dst.resolve():
                shutil.copyfile(overlay, dst)
            overlay_url = f"/uploads/{dst.name}"
        problems = validate_scene(scene)
        if not problems:
            (root / "scenes" / f"{name}.json").write_text(json.dumps(scene, ensure_ascii=False, indent=1))
            job.log(f"saved scenes/{name}.json with {len(scene['objects'])} objects")
        else:
            job.log("scene needs edits before it can be rehearsed (not saved):")
            for p in problems:
                job.log(f"  - {p}")
        return {"scene": name, "scene_spec": scene, "overlay": overlay_url,
                "valid": not problems, "issues": problems, "saved": not problems}

    return work


def worker_count() -> int | None:
    """Process-pool size for the stress test; REHEARSAL_WORKERS overrides os.cpu_count()."""
    v = os.environ.get("REHEARSAL_WORKERS")
    return int(v) if v and v.isdigit() and int(v) > 0 else None


# ------------------------------------------------------------------------------ reality check
REPORT_LOCK = threading.Lock()   # serialises read-modify-write of report.json


def pick_attempt(report: dict, attempt: int | None) -> int:
    """attempt if it exists in the report, else (None) the last successful attempt."""
    nums = [a["attempt"] for a in report.get("attempts", [])]
    if attempt is not None:
        if attempt not in nums:
            raise ValueError(f"attempt {attempt} not in report (have {nums})")
        return attempt
    ok = [a["attempt"] for a in report.get("attempts", []) if a.get("success")]
    if not ok:
        raise ValueError("no successful attempt in the report; pass attempt explicitly")
    return ok[-1]


def sim_final_from_traj(spec: dict, traj: list) -> dict:
    """{object id: {"xy": [x, y], "z_base": bottom height above the table}} from the last row of a
    'full' attempt_N.traj.json (row = [t, then xyz+quat for every body 1..nbody-1])."""
    import mujoco
    from .scene import build_model, object_height
    m = build_model(spec)
    last = traj[-1]
    out = {}
    for o in spec["objects"]:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["id"])
        i = 1 + 7 * (b - 1)
        x, y, z = last[i:i + 3]
        out[o["id"]] = {"xy": [float(x), float(y)], "z_base": round(max(0.0, float(z) - object_height(o) / 2), 4)}
    return out


def map_ids(sim_spec: dict, twin_spec: dict, max_m: float = 0.08) -> dict:
    """twin id -> sim id by nearest initial position (greedy, unique, within max_m). Identity when
    the twin is the simulated scene itself."""
    if twin_spec is sim_spec or twin_spec.get("name") == sim_spec.get("name"):
        return {o["id"]: o["id"] for o in sim_spec["objects"]}
    pairs = sorted((math.dist(t["pos"], s["pos"]), t["id"], s["id"])
                   for t in twin_spec["objects"] for s in sim_spec["objects"])
    used_t, used_s, out = set(), set(), {}
    for d, t, s in pairs:
        if d <= max_m and t not in used_t and s not in used_s:
            out[t] = s
            used_t.add(t)
            used_s.add(s)
    return out


def _reality_fn():
    try:
        from .reality import reality_check
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"rehearsal.reality is not available: {type(e).__name__}: {e}") from e
    return reality_check


def reality_job(root: Path, scene: str, video: Path, privacy: str, attempt: int | None = None,
                twin: str | None = None):
    """Compare a phone clip of the real execution with out/<scene>/attempt_N's final state.
    twin: scene whose perception.box_px seeds tracking (default: the simulated scene itself)."""

    def work(job: JobHandle):
        fn = _reality_fn()
        out = root / "out" / scene
        report = json.loads((out / "report.json").read_text())
        n = pick_attempt(report, attempt)
        spec = report["scene_spec"]
        job.log(f"reality check: {video.name} vs {scene} attempt {n} (data={privacy})")
        job.progress("sim_final", 0.05)
        sim_final = sim_final_from_traj(spec, json.loads((out / f"attempt_{n}.traj.json").read_text()))
        twin_spec = spec if not twin else json.loads((root / "scenes" / f"{twin}.json").read_text())
        ids = map_ids(spec, twin_spec)
        if twin:
            job.log(f"twin {twin}: matched {len(ids)}/{len(twin_spec['objects'])} objects by position: {ids}")
        if not any((o.get("perception") or {}).get("box_px") for o in twin_spec["objects"]):
            raise RuntimeError(f"scene {twin or scene} has no perception.box_px (not built from a photo); "
                               "pass twin=<the perceived scene> to seed the video tracker")
        twin_final = {t: sim_final[s] for t, s in ids.items()}
        job.progress("track", 0.1)
        res = fn(str(video), twin_spec, twin_final, data=privacy, out_dir=str(out / "reality"))
        job.progress("save", 0.95)
        overlay_url = None
        if res.get("overlay") and Path(res["overlay"]).is_file():
            dst = root / "uploads" / f"{job.id}_reality.gif"
            shutil.copyfile(res["overlay"], dst)
            overlay_url = f"/uploads/{dst.name}"
        result = {"scene": scene, "attempt": n, "twin": twin or scene, "id_map": ids,
                  "per_object": res.get("per_object", []), "verdict": res.get("verdict"),
                  "overlay": overlay_url, "tol_cm": res.get("tol_cm"), "frames": res.get("frames"),
                  "camera": res.get("camera"), "video": f"/uploads/{video.name}", "privacy": privacy,
                  "job": job.id, "created": time.time()}
        with REPORT_LOCK:
            rp = out / "report.json"
            rep = json.loads(rp.read_text())
            rep["reality"] = result
            rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
        job.log(res.get("verdict") or "")
        return result

    return work
