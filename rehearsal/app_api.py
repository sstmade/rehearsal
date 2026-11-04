"""Backend of the product app (/app): photo -> perceive -> 3D twin -> instruct -> rehearse -> replay.

Everything here drives the shipped pipeline; nothing is simulated for show:
  - perception: cached SAM 3.1 masks / Muse Spark labels stored in the scene file (samples), or a live
    perceive_raw() call on an uploaded photo (upload_scene below);
  - intent: rehearsal.rehearse.compile_step (Muse Spark + the deterministic intent gate), served from the
    recorded-response cache when the identical request was made before (free), live otherwise;
  - planning: rehearsal.search_planner.search, every candidate verified by rehearsal.executor.run_plan (MuJoCo);
  - verdict: the found plan re-simulated with a full recording (the replay) plus a short held-out stress test,
    or, when nothing passes within the budget, the direct attempt (cheapest straight carry) simulated and shown.

Run outputs live in out/app/<scene>/ (or, for the user's own desk, outside the repo next to its footage), served
read-only by server.py under /app-out/<scene>/....
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_OUT = ROOT / "out" / "app"
# The user's own desk: local only, never bundled (outside the repo), cached perception, no new Meta calls.
MY_DESK_ID = "my_desk"
MY_DESK_SCENE = ROOT.parent / "footage" / "user_desk_sim" / "twin" / "user_desk_scene.json"
MY_DESK_OUT = ROOT.parent / "footage" / "user_desk_sim" / "app_out"

DEFAULT_BUDGET_S = 90
DEFAULT_SIMS = 120
STRESS_N = 12

_lock = threading.RLock()
EVENTS: dict[str, list] = {}          # job id -> structured progress events (polled by the app)
SAM3D_JOBS: dict[str, dict] = {}


def public_demo() -> bool:
    return os.environ.get("REHEARSAL_PUBLIC_DEMO", "").strip().lower() not in ("", "0", "false", "no", "off")


def upload_data_mode() -> str:
    """Meta API data mode for photos uploaded in the app: REHEARSAL_UPLOAD_DATA=public (default: Contributor
    mode, cheaper; the upload step tells the visitor) or private (Standard mode)."""
    v = os.environ.get("REHEARSAL_UPLOAD_DATA", "public").strip().lower()
    return v if v in ("public", "private") else "public"


def live_models() -> bool:
    """True when this server may call the Meta API (key present, not the public demo, not forced offline)."""
    from .agents import offline_forced
    from . import load_env
    try:
        load_env()
    except Exception:  # noqa: BLE001
        pass
    return bool(os.environ.get("MODEL_API_KEY")) and not public_demo() and not offline_forced()


def my_desk_available() -> bool:
    return not public_demo() and MY_DESK_SCENE.is_file()


def sam3d_available() -> bool:
    return (not public_demo() and (ROOT / "sam3d_env" / "bin" / "python").exists()
            and (ROOT / "weights" / "sam-3d-objects-mlx-8bit").is_dir()
            and (ROOT / "scripts" / "sam3d_assets.py").is_file())


def spent() -> float:
    from .meta_client import status
    try:
        return float(status()["spent"])
    except Exception:  # noqa: BLE001
        return 0.0


# ------------------------------------------------------------------------------------------- scenes
def scene_file(sid: str) -> Path:
    from .pipeline import valid_name
    if sid == MY_DESK_ID:
        if not my_desk_available():
            raise FileNotFoundError("my desk is not available on this server")
        return MY_DESK_SCENE
    if not valid_name(sid):
        raise ValueError("bad scene id")
    p = ROOT / "scenes" / f"{sid}.json"
    if not p.is_file():
        raise FileNotFoundError(f"no scene {sid}")
    return p


def load(sid: str) -> dict:
    from .scene import load_scene
    return load_scene(str(scene_file(sid)))


def out_dir(sid: str) -> Path:
    scene_file(sid)        # validates the id (and that the scene exists) before anything is created
    d = (MY_DESK_OUT if sid == MY_DESK_ID else APP_OUT / sid)
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_mode(spec: dict) -> str:
    return "public" if spec.get("privacy") == "public" else "private"


def photo_url(sid: str, spec: dict) -> str | None:
    if sid == MY_DESK_ID:
        return "/api/app/mydesk/photo"
    f = (spec.get("photo") or {}).get("file")
    return _served(f, ("photos/", "uploads/"))


def _served(f, prefixes) -> str | None:
    """URL of a photo-derived file: its JPEG sibling <name>.web.jpg when present (that is what ships; the PNG
    sources stay local, see scripts/build_web_photos.py), else the file itself."""
    if not (isinstance(f, str) and f.startswith(prefixes)):
        return None
    web = (ROOT / f).with_suffix(".web.jpg")
    if web.is_file():
        return "/" + str(web.relative_to(ROOT))
    return "/" + f if (ROOT / f).is_file() else None


def short_label(o: dict) -> str:
    lab = (o.get("label") or o["id"]).strip()
    return lab


def samples() -> list[dict]:
    """Shipped scenes with a photo (the 50 AI-generated test tables, the AI-generated desks, the YCB-V photo),
    then the hand-authored scenes, then the user's desk when this is a local server."""
    out = []
    if my_desk_available():
        spec = json.loads(MY_DESK_SCENE.read_text())
        out.append({"id": MY_DESK_ID, "title": "My desk", "group": "local", "thumb": "/api/app/mydesk/photo",
                    "n_objects": len(spec["objects"]), "local_only": True, "has_photo": True,
                    "demo_instruction": "Move the power bank onto the card wallet.",
                    "note": "Local only · cached perception · never uploaded"})
    for p in sorted((ROOT / "scenes").glob("*.json")):
        if p.stem.startswith("photo_"):
            continue
        try:
            spec = json.loads(p.read_text())
        except ValueError:
            continue
        ph = photo_url(p.stem, spec)
        title = (spec.get("title") or p.stem).replace(" (AI-generated test photo)", "")
        group = "tables" if p.stem.startswith("t6_") else ("photos" if ph else "authored")
        thumb = ph
        if not ph:
            from .server import _thumb
            thumb = _thumb(ROOT, p.stem, spec)
        out.append({"id": p.stem, "title": title, "group": group, "thumb": thumb, "has_photo": bool(ph),
                    "n_objects": len(spec.get("objects", [])), "demo_instruction": spec.get("demo_instruction"),
                    "ai_generated": bool(spec.get("ai_generated")) or "aigen" in p.stem,
                    "sam3d": (ROOT / "assets" / "sam3d" / p.stem).is_dir()})
    return out


def _tables_intent(sid: str) -> str | None:
    if not sid.startswith("t6_"):
        return None
    p = ROOT / "out" / "tables_v6" / f"{sid[3:]}.json"
    try:
        return json.loads(p.read_text()).get("instruction")
    except (OSError, ValueError):
        return None


def suggestions(sid: str, spec: dict) -> list[str]:
    """Instructions worth offering: the scene's demo instruction, ones with a recorded Muse Spark answer."""
    out = []
    for s in (spec.get("demo_instruction"), _tables_intent(sid)):
        if s and s not in out:
            out.append(s)
    try:   # abstract tasks with recorded answers ("Clean this up.", "Get the table ready for dinner.", ...)
        for c in json.loads((ROOT / "bench" / "goals" / "app_instructions.json").read_text()):
            if c["scene"] == sid and c["instruction"] not in out:
                out.append(c["instruction"])
    except (OSError, ValueError, KeyError):
        pass
    if sid == MY_DESK_ID:
        try:
            sys.path.insert(0, str(ROOT / "scripts" / "user_desk"))
            tasks = ["Move the power bank onto the card wallet.", "Move the power bank onto the blue pouch.",
                     "Put the pen next to the mouse.", "Pour the water bottle onto the keyboard.",
                     "Tidy up my desk.", "Move the water bottle next to the monitor."]
            out += [t for t in tasks if t not in out]
        except Exception:  # noqa: BLE001
            pass
    try:
        from .pipeline import recorded_instructions
        for s in recorded_instructions(spec.get("name", sid)):
            if s not in out and "(You asked:" not in s:
                out.append(s)
    except Exception:  # noqa: BLE001
        pass
    return out[:6]


def scene_info(sid: str) -> dict:
    """What the Perceive step shows: the photo, every object's SAM 3.1 outline + label, and where it came from."""
    spec = load(sid)
    ph = spec.get("photo") or {}
    objs = []
    for i, o in enumerate(spec["objects"]):
        pc = o.get("perception") or {}
        objs.append({"id": o["id"], "label": short_label(o), "category": o.get("category"),
                     "box_px": pc.get("box_px"), "mask_px": pc.get("mask_px"),
                     "conf": pc.get("sam_confidence"), "rgba": o.get("rgba"),
                     "shape": o["shape"]["type"], "sam3d": False})
    sam_dir = ROOT / "assets" / "sam3d" / spec["name"]
    for o in objs:
        o["sam3d"] = (sam_dir / f"{o['id']}.glb").is_file()
    w, h = ph.get("width"), ph.get("height")
    return {"id": sid, "name": spec["name"], "title": (spec.get("title") or sid).replace(" (AI-generated test photo)", ""),
            "privacy": spec.get("privacy", "private"), "photo": photo_url(sid, spec), "width": w, "height": h,
            "objects": objs, "table": spec["table"], "provenance": spec.get("provenance"), "camera": ph.get("camera"),
            "source": spec.get("source"), "suggestions": suggestions(sid, spec),
            "local_only": sid == MY_DESK_ID,
            "table_texture": _table_texture(sid, spec), "models": _models_for(sid),
            "ask_answers": _ask_answers(sid), "twin_note": spec.get("twin_note")}


def _ask_answers(sid: str) -> dict:
    """Suggested replies to Muse Spark's clarifying questions on this table (each one has a recorded answer, so the
    public demo never dead-ends at "ask"): {instruction: [answer, ...]}."""
    try:
        d = json.loads((ROOT / "bench" / "goals" / "app_ask_answers.json").read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for k, v in d.items():
        s, _, ins = k.partition("|")
        if s == sid:
            out[ins] = v
    return out


def _models_for(sid: str):
    f = ROOT / "scenes" / "_models" / f"{sid}.json"     # per-model latency/cost of an upload (kept out of scenes/*.json)
    try:
        return json.loads(f.read_text()) if f.is_file() else None
    except (OSError, ValueError):
        return None


def _table_texture(sid, spec):
    t = (spec.get("visual") or {}).get("table_texture")
    return None if sid == MY_DESK_ID else _served(t, ("photos/",))


# ------------------------------------------------------------------------------------------- twin
def twin(sid: str) -> dict:
    """Export the MuJoCo model for the viewer and let the scene settle (1.5 s of physics, no robot motion):
    the twin the app shows is the state physics actually starts from."""
    import mujoco  # noqa: F401
    from .executor import run_plan
    from .export3d import export_scene3d
    from .scene import build_model
    spec = load(sid)
    d = out_dir(sid)
    src = scene_file(sid)
    stamp = d / "twin.json"
    if stamp.is_file() and stamp.stat().st_mtime >= src.stat().st_mtime and (d / "scene3d.json").is_file():
        sam = ROOT / "assets" / "sam3d" / spec["name"]
        newest = max([p.stat().st_mtime for p in sam.glob("*.glb")], default=0)
        if newest <= stamp.stat().st_mtime:
            return json.loads(stamp.read_text())
    t0 = time.time()
    export_scene3d(build_model(spec), d, spec)
    r = run_plan(spec, {"actions": []}, [], record="full")
    traj = r["trajectory"]
    (d / "settle.traj.json").write_text(json.dumps(traj))
    desc = json.loads((d / "scene3d.json").read_text())
    vis = [b["name"] for b in desc["bodies"] if b.get("visual_mesh")]
    res = {"scene": sid, "base": f"/app-out/{sid}", "settle": f"/app-out/{sid}/settle.traj.json?v={int(time.time())}",
           "home_frame": traj[-1], "visual_meshes": vis, "table": spec["table"],
           "table_texture": _table_texture(sid, spec),
           "objects": [{"id": o["id"], "label": short_label(o)} for o in spec["objects"]],
           "build_s": round(time.time() - t0, 2)}
    stamp.write_text(json.dumps(res))
    return res


# ------------------------------------------------------------------------------------------- intent
def compose_instruction(instruction: str, turns: list[dict] | None) -> str:
    """Multi-turn clarification: the original instruction plus each question and the user's answer."""
    s = instruction.strip()
    for t in turns or []:
        q, a = str(t.get("q", "")).strip().replace("\u2014", "-"), str(t.get("a", "")).strip()
        if a:
            s += f' (You asked: "{q}" My answer: "{a}")'
    return s


def intent_job(sid: str, instruction: str, turns: list[dict] | None):
    def work(job):
        from . import meta_client
        from .agents import MuseAgent, scene_summary
        from .executor import reachability_report
        from .rehearse import compile_step
        spec = load(sid)
        data = data_mode(spec)
        full = compose_instruction(instruction, turns)
        live = live_models()
        before = spent()
        t0 = time.time()
        summary = scene_summary(spec, reachability_report(spec))
        agent = MuseAgent(data=data, cache_only=not live)
        req_key = None
        try:
            from .agents import intent_request_key
            req_key = intent_request_key(summary, full, data)
            cached = meta_client.recorded(req_key) is not None
        except Exception:  # noqa: BLE001
            cached = False
        fallback = None
        try:
            with meta_client.cost_meter() as paid:
                try:
                    intent = compile_step(agent, spec, full, summary, log=job.log)
                except meta_client.CacheMiss:
                    raise
                except Exception as e1:  # noqa: BLE001  live API hiccup: retry once, then the recorded answer
                    if not live:
                        raise
                    job.log(f"live Muse Spark call failed ({type(e1).__name__}: {e1}); retrying once")
                    time.sleep(1.5)
                    try:
                        intent = compile_step(agent, spec, full, summary, log=job.log)
                    except Exception as e2:  # noqa: BLE001
                        if not cached:
                            return {"status": "api_error", "instruction": full, "cost_usd": round(sum(paid), 5),
                                    "cached": False, "suggestions": suggestions(sid, spec),
                                    "message": f"Muse Spark did not answer ({type(e2).__name__}). Nothing was "
                                               "invented. Try again, or pick a suggested instruction (those have "
                                               "recorded answers)."}
                        fallback = f"live call failed ({type(e2).__name__}); showing the recorded answer"
                        intent = compile_step(MuseAgent(data=data, cache_only=True), spec, full, summary, log=job.log)
        except meta_client.CacheMiss:
            return {"status": "no_recording", "instruction": full, "cost_usd": 0.0, "cached": False,
                    "message": "This server makes no model calls and has no recorded Muse Spark answer for this exact "
                               "instruction on this table, so nothing is shown rather than something invented. "
                               "Try one of the suggested instructions.",
                    "suggestions": suggestions(sid, spec)}
        outcome = intent.pop("outcome")
        labels = {o["id"]: short_label(o) for o in spec["objects"]}
        return {"status": {"rehearsed": "ready"}.get(outcome, outcome), "instruction": full,
                "intent": intent, "labels": labels, "model": agent.name, "data": data,
                "cached": cached or bool(fallback), "fallback": fallback, "cost_usd": round(sum(paid), 5),
                "wall_s": round(time.time() - t0, 2)}
    return work


# ------------------------------------------------------------------------------------------- rehearse
def _emit(jid: str, ev: dict):
    with _lock:
        lst = EVENTS.setdefault(jid, [])
        ev = dict(ev, i=len(lst), t=round(time.time(), 3))
        lst.append(ev)
        if len(EVENTS) > 300:
            for k in list(EVENTS)[:50]:
                EVENTS.pop(k, None)


def events(jid: str, since: int = 0) -> list:
    with _lock:
        return list(EVENTS.get(jid, [])[since:])


def _lab(spec, oid):
    o = next((o for o in spec["objects"] if o["id"] == oid), None)
    return short_label(o) if o else oid


def rejection(spec: dict, r: dict) -> dict:
    """Why one simulated candidate failed, in a few words, plus the contacts to highlight."""
    reasons, hits = [], []
    force = {}
    for c in r.get("arm_contacts") or []:
        force[c["obj"]] = max(force.get(c["obj"], 0.0), c["peak_force_n"])
    err = r.get("execution_error")
    if err:
        e = err.get("error", "")
        if "IK" in e or "reach" in e.lower():
            reasons.append("arm cannot reach that pose")
        elif "protective" in e.lower() or "force" in e.lower():
            reasons.append("protective stop: " + e.split(":", 1)[-1].strip()[:60])
        else:
            reasons.append(e[:80])
    for d in r.get("disturbed") or []:
        lab = _lab(spec, d["obj"])
        f = force.get(d["obj"])
        fh = d.get("first_hit") or {}
        if d.get("fell_off_table"):
            txt = f"{lab} fell off the table"
        elif d["tilt_deg"] >= 20:
            txt = f"{lab} tipped ({int(d['tilt_deg'])}°)"
        else:
            txt = f"{lab} pushed {d['moved_cm']} cm"
        if fh.get("by"):
            by = "the arm" if fh.get("link") or fh["by"] == "robot_arm" else _lab(spec, fh["by"])
            txt = (f"{by} hit {lab}" + (f", {f:.0f} N" if f else "")
                   + (f" — {lab} tipped" if d["tilt_deg"] >= 20 else f" — moved {d['moved_cm']} cm"))
        elif f:
            txt += f", {f:.0f} N"
        reasons.append(txt)
        if fh.get("t") is not None:
            hits.append({"t": fh["t"], "obj": d["obj"], "text": txt})
    for c in r.get("arm_contacts") or []:
        if c["peak_force_n"] >= 15 and not any(h["obj"] == c["obj"] for h in hits) and c.get("t") is not None:
            hits.append({"t": c["t"], "obj": c["obj"], "text": f"arm hit {_lab(spec, c['obj'])}, {c['peak_force_n']:.0f} N"})
            if not any(_lab(spec, c["obj"]) in x for x in reasons):
                reasons.append(f"arm touched {_lab(spec, c['obj'])}, {c['peak_force_n']:.0f} N")
    if not reasons:
        for g in r.get("goals") or []:
            if not g["ok"]:
                reasons.append(f"{g['pred']}({', '.join(_lab(spec, a) for a in g['args'])}) failed: {g['why']}"[:120])
                break
    return {"reasons": reasons[:3], "hits": hits}


def _act_text(spec, a):
    rel = {"on": "onto", "in": "into"}.get(a.get("relation"), a.get("relation", ""))
    if str(rel).startswith("near@"):
        rel = "next to"
    return f"{_lab(spec, a['obj'])} {rel} {_lab(spec, a['dest'])}"


def direct_plan(spec, goals, assigns):
    """The direct attempt: the assignments in the order given, each with its cheapest straight carry (the first
    candidate a naive planner would send)."""
    from .search_planner import Kinematics, candidates
    kin = Kinematics(spec)
    poses = dict(kin.settled)
    acts = []
    for a in assigns:
        c = next((c for c in candidates(a, poses, kin, None, use_via=False)), None)
        acts.append(dict(a, transit_height=c[1] if c else 0.15, via=[]))
    return {"diagnosis": "direct attempt: lowest straight carry, no detour", "actions": acts}


def rehearse_job(sid: str, goals: list[dict], instruction: str, budget_s: int, sims: int, near_beta: bool = False):
    def work(job):
        from .executor import run_plan
        from .predicates import with_implicit_goals
        from .rehearse import HELDOUT_SEED0, stress_test
        from .search_planner import assignments_from_goals, search
        jid = job.id
        spec = load(sid)
        ids = [o["id"] for o in spec["objects"]]
        gl = with_implicit_goals(goals, ids)
        assigns = assignments_from_goals(gl)
        has_near = any(g.get("pred") == "near" for g in goals)
        t0 = time.time()
        _emit(jid, {"type": "start", "budget_s": budget_s, "sims": sims, "goals": gl,
                    "assignments": [_act_text(spec, a) for a in assigns]})
        cand = {"n": 0}
        last_order = {"o": None}

        cand_dir = out_dir(sid) / "runs" / jid
        _prune_runs(cand_dir.parent, keep=12)
        cand_dir.mkdir(parents=True, exist_ok=True)

        def on_sim(actions, gls, r, st):
            cand["n"] += 1
            rej = None if r["success"] else rejection(spec, r)
            a = actions[-1] if actions else None
            url = None
            tr = r.pop("trajectory", None)
            if tr:
                (cand_dir / f"cand_{cand['n']:03d}.json").write_text(json.dumps(thin(tr)))
                url = f"/app-out/{sid}/runs/{jid}/cand_{cand['n']:03d}.json"
            _emit(jid, {"type": "candidate", "n": cand["n"], "prefix": len(actions), "traj": url,
                        "hits": (rej or {}).get("hits", []), "plan": [_act_text(spec, x) for x in actions],
                        "of": max(len(assigns), len(actions)), "action": _act_text(spec, a) if a else "",
                        "height": a and a.get("transit_height"), "via": bool(a and a.get("via")),
                        "ok": bool(r["success"]), "reasons": (rej or {}).get("reasons", []),
                        "sims": st["sims"], "wall_s": st["wall_s"], "kin_checks": st.get("kin_checks", 0),
                        "kin_ik_pruned": st.get("kin_ik_pruned", 0)})
            job.progress("search", min(0.95, max(st["sims"] / sims, st["wall_s"] / budget_s)))

        def log(msg):
            if msg.startswith("level") and "order" in msg:
                o = msg.split("order", 1)[1].strip()
                if o != last_order["o"]:
                    last_order["o"] = o
                    lvl = int(msg.split()[1])
                    _emit(jid, {"type": "order", "level": lvl, "order": o})

        result = {"scene": sid, "instruction": instruction, "goals": gl, "budget": {"time_s": budget_s, "sims": sims},
                  "near_beta": False}
        plan, res = None, None
        if has_near and near_beta:      # public demo runs one rehearsal at a time, so the beta's patching is safe
            plan, res = _near_search(spec, gl, budget_s, sims, jid, on_sim, assigns_text=None)
            result["near_beta"] = True
        elif assigns:
            beat = {"t": 0.0}

            from .search_planner import _Budget

            def on_kin(st, act, kr):
                if st["wall_s"] > budget_s:      # kinematic dry runs do not check the clock themselves
                    raise _Budget()
                why = None
                if not kr["ik_ok"]:
                    why = f"out of reach during {kr.get('ik_phase', 'the move')}"
                elif kr["hits"]:
                    ph, who, other, dist = kr["hits"][0]
                    by = "held " + _lab(spec, who) if who == act["obj"] else "arm"
                    why = f"{by} would hit {_lab(spec, other)} ({str(ph).rstrip('0123456789')}, {abs(dist) * 100:.1f} cm overlap)"
                if why:
                    _emit(jid, {"type": "dry", "action": _act_text(spec, act), "height": act["transit_height"],
                                "via": bool(act.get("via")), "why": why, "kin_checks": st["kin_checks"]})
                now = time.time()
                if now - beat["t"] > 0.4:
                    beat["t"] = now
                    _emit(jid, {"type": "kin", "kin_checks": st["kin_checks"], "kin_ik_pruned": st["kin_ik_pruned"],
                                "kin_deferred": st["kin_deferred"], "sims": st["sims"], "wall_s": st["wall_s"]})
                    job.progress("search", min(0.95, max(st["sims"] / sims, st["wall_s"] / budget_s)))
            res = search(spec, gl, assigns, sim_budget=sims, time_budget_s=budget_s, log=log, on_sim=on_sim,
                         on_kin=on_kin, record_sims=True)
            plan = res["plan"] if res["success"] else None
        else:
            res = {"success": False, "reason": ("the planner has no action for near() goals yet (enable the beta "
                                                "near-placement)" if has_near else "the goals ask for no move"),
                   "stats": {"sims": 0, "wall_s": 0.0, "kin_checks": 0}}
        st = res.get("stats", {})
        result["search"] = {"success": bool(plan), "reason": res.get("reason"), "stats": st}
        result["budget"].update(used_s=round(time.time() - t0, 1), used_sims=st.get("sims", 0))
        out = out_dir(sid) / "runs" / jid
        out.mkdir(parents=True, exist_ok=True)
        if plan:
            _emit(jid, {"type": "phase", "phase": "verify", "text": "Replaying the found plan with full recording"})
            r = run_plan(spec, plan, gl, record="full")
            ok = r["success"]
            rej = rejection(spec, r)
            traj = r.pop("trajectory")
            (out / "traj.json").write_text(json.dumps(traj))
            result.update(verdict="sent" if ok else "not_sent", plan=plan,
                          actions=[dict(a, text=_act_text(spec, a)) for a in plan["actions"]],
                          goal_results=r["goals"], hits=rej["hits"], reasons=rej["reasons"],
                          traj=f"/app-out/{sid}/runs/{jid}/traj.json", sim_time=r["sim_time"])
            if ok and not result["near_beta"]:
                _emit(jid, {"type": "phase", "phase": "stress",
                            "text": f"Stress test: {STRESS_N} held-out variants (friction, mass, perception error)"})
                job.progress("stress", 0.96)
                try:
                    stt = stress_test(spec, plan, gl, STRESS_N, seed0=HELDOUT_SEED0,
                                      on_run=lambda k, n: _emit(jid, {"type": "stress", "k": k, "n": n}))
                    result["stress"] = {"successes": stt["successes"], "n": stt["n"],
                                        "fails": [f for x in stt["runs"] if not x["success"] for f in x["failed"][:1]][:3]}
                except Exception as e:  # noqa: BLE001
                    result["stress"] = {"error": f"{type(e).__name__}: {e}"}
        else:
            # Not sent. Show what the direct attempt would have done.
            dp = None
            if assigns and not has_near:
                dp = direct_plan(spec, gl, assigns)
            elif res.get("direct_attempt"):
                dp = {"diagnosis": "direct attempt: first free spot, 0.3 m carry", "actions": res["direct_attempt"]}
            if dp:
                _emit(jid, {"type": "phase", "phase": "direct", "text": "No plan passed. Simulating the direct attempt"})
                r = run_plan(spec, dp, gl, record="full")
                rej = rejection(spec, r)
                (out / "traj.json").write_text(json.dumps(r.pop("trajectory")))
                result.update(direct={"plan": dp, "success": r["success"], "reasons": rej["reasons"],
                                      "actions": [dict(a, text=_act_text(spec, a)) for a in dp["actions"]],
                                      "goal_results": r["goals"]},
                              hits=rej["hits"], traj=f"/app-out/{sid}/runs/{jid}/traj.json")
                if r["success"]:
                    # The search ran out of budget, but the direct carry itself passes every goal in physics:
                    # that is a verified plan like any other.
                    result.update(verdict="sent", plan=dp, actions=result["direct"]["actions"],
                                  goal_results=r["goals"], reasons=[], via_direct=True, sim_time=r["sim_time"])
                    result.pop("direct")
                    _emit(jid, {"type": "phase", "phase": "stress",
                                "text": f"Direct carry passed. Stress test: {STRESS_N} held-out variants"})
                    try:
                        stt = stress_test(spec, dp, gl, STRESS_N, seed0=HELDOUT_SEED0,
                                          on_run=lambda k, n: _emit(jid, {"type": "stress", "k": k, "n": n}))
                        result["stress"] = {"successes": stt["successes"], "n": stt["n"],
                                            "fails": [f for x in stt["runs"] if not x["success"] for f in x["failed"][:1]][:3]}
                    except Exception as e:  # noqa: BLE001
                        result["stress"] = {"error": f"{type(e).__name__}: {e}"}
            if result.get("verdict") != "sent":
                result["verdict"] = "not_sent"
                result["reasons"] = [res.get("reason") or "no plan passed physics"]
        result["wall_s"] = round(time.time() - t0, 1)
        result["cost_usd"] = 0.0          # planning and physics make no model calls
        (out / "result.json").write_text(json.dumps(result, default=float))
        _emit(jid, {"type": "done", "verdict": result["verdict"]})
        return result
    return work


def _prune_runs(runs: Path, keep: int = 12):
    """Keep the newest `keep` run folders of a table (each holds every candidate's trajectory); a public server
    must not fill its disk."""
    try:
        old = sorted((d for d in runs.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)[:-keep or None]
    except OSError:
        return
    for d in old if len(old) and keep else []:
        shutil.rmtree(d, ignore_errors=True)


def thin(traj: list, every: int = 2) -> list:
    """Every 2nd row (50 Hz -> 25 Hz) at 3 decimals: enough for a replay tile, ~4x smaller."""
    rows = traj[::every]
    if traj and rows[-1] is not traj[-1]:
        rows.append(traj[-1])
    return [[round(v, 3) for v in r] for r in rows]


def _near_search(spec, gl, budget_s, sims, jid, on_sim, assigns_text=None):
    """BETA: next-to placement (scripts/user_desk/run_user_desk.py search_ext, the demo extension: free-spot search
    next to the reference object, every candidate simulated by the stock executor)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "scripts" / "user_desk"))
    import run_user_desk as ud
    ud.install_ext()
    _emit(jid, {"type": "phase", "phase": "near", "text": "BETA near-placement: searching free spots"})
    lines = []

    def log(m):
        lines.append(str(m))
        _emit(jid, {"type": "note", "text": str(m)[:160]})
    from . import executor
    stock = executor.run_plan
    count = {"n": 0, "t0": time.time()}

    def recorded(scene, plan, goals, *a, **k):
        if plan.get("actions") and not k.get("record"):
            k["record"] = "full"
            r = stock(scene, plan, goals, *a, **k)
            count["n"] += 1
            on_sim(plan["actions"], goals, r, {"sims": count["n"], "wall_s": round(time.time() - count["t0"], 2),
                                               "kin_checks": 0, "kin_ik_pruned": 0})
            return r
        return stock(scene, plan, goals, *a, **k)
    executor.run_plan = recorded
    try:
        res = ud.search_ext(spec, gl, time_budget_s=budget_s, sim_budget=sims, log=log)
    except Exception as e:  # noqa: BLE001
        return None, {"success": False, "reason": f"near beta failed: {type(e).__name__}: {e}", "stats": {}}
    finally:
        executor.run_plan = stock
    plan = res.get("plan") if isinstance(res, dict) else None
    ok = bool(isinstance(res, dict) and res.get("success") and plan)
    return (plan if ok else None), {"success": ok, "reason": None if ok else (res or {}).get("reason", "no spot passed"),
                                    "stats": (res or {}).get("stats", {}),
                                    "direct_attempt": (res or {}).get("direct_attempt") or []}


# ------------------------------------------------------------------------------------------- uploads
def _ledger_stat(t0: float, prefixes: tuple) -> dict:
    """Number of paid calls and their cost since t0 for these callers (0 calls = answered from the disk cache)."""
    from .meta_client import budget_dir
    n, tot = 0, 0.0
    try:
        for line in (budget_dir() / "ledger.jsonl").read_text().splitlines()[-600:]:
            e = json.loads(line)
            ts = time.mktime(time.strptime(e["ts"], "%Y-%m-%d %H:%M:%S"))
            if ts >= t0 and str(e.get("caller", "")).startswith(prefixes):
                n += 1; tot += float(e.get("cost", 0))
    except (OSError, ValueError, KeyError):
        pass
    return {"calls": n, "usd": round(tot, 5)}


def _ledger_cost(t0: float, prefixes: tuple) -> float:
    """Cost of the ledger entries written since t0 by the given callers (perception runs in a thread pool, so the
    per-thread meter cannot see it; filtering by caller keeps other processes' compile calls out)."""
    from .meta_client import budget_dir
    tot = 0.0
    try:
        for line in (budget_dir() / "ledger.jsonl").read_text().splitlines()[-400:]:
            e = json.loads(line)
            ts = time.mktime(time.strptime(e["ts"], "%Y-%m-%d %H:%M:%S"))
            if ts >= t0 and str(e.get("caller", "")).startswith(prefixes):
                tot += float(e.get("cost", 0))
    except (OSError, ValueError, KeyError):
        pass
    return round(tot, 5)



TABLE_SCHEMA = {"type": "object", "additionalProperties": False,
                "required": ["corners_pct", "tabletop_width_cm", "tabletop_depth_cm", "camera_pitch_deg"],
                "properties": {"corners_pct": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                               "tabletop_width_cm": {"type": "number"}, "tabletop_depth_cm": {"type": "number"},
                               "camera_pitch_deg": {"type": "number"}}}
TABLE_PROMPT = ("Look at the table or desk surface in this photo. Give the four corners of its top surface as "
                "[x, y] percentages of the image width and height (0-100; a corner outside the frame may be <0 or "
                ">100), in this order: near-left, near-right, far-right, far-left. Also estimate the real size of the "
                "tabletop (width left-right, depth front-back, in cm, using the objects on it for scale) and how steeply "
                "the camera looks down at it (degrees below horizontal). Numbers only.")


def upload_job(photo: Path, data: str):
    """Perceive an uploaded photo with the same steps as the 50 test tables (scripts/batch_tables.py): Muse Spark
    lists the objects, SAM 3.1 outlines them, the tabletop outline comes from Muse Spark (the batch read it by eye),
    camera pitch from the outline, scale from typical object sizes."""
    def work(job):
        import numpy as np
        from PIL import Image
        from .meta_client import chat_json
        from .perception import data_url, draw_overlay, perceive_raw
        sys.path.insert(0, str(ROOT / "scripts"))
        import batch_tables as bt
        jid = job.id
        t_start = time.time() - 1
        _emit(jid, {"type": "phase", "phase": "upload", "text": f"Photo received · Meta API data mode: {data}"})
        # downscale large phone photos (the API cost and SAM time do not need 12 MP)
        im = Image.open(photo)
        im = im.convert("RGB")
        if max(im.size) > 1600:
            k = 1600 / max(im.size)
            im = im.resize((round(im.size[0] * k), round(im.size[1] * k)), Image.LANCZOS)
        small = photo.with_name(photo.stem + "_in.jpg")
        im.save(small, quality=92)
        w, h = im.size
        job.progress("muse", 0.1)
        _emit(jid, {"type": "phase", "phase": "muse", "text": "Muse Spark: naming the objects and the tabletop"})
        t0m = time.time()
        dims = chat_json([{"role": "user", "content": [{"type": "text", "text": TABLE_PROMPT},
                                                        {"type": "image_url", "image_url": {"url": data_url(small)}}]}],
                         TABLE_SCHEMA, "TableTop", caller="app.table_top", data=data, effort="low")
        _emit(jid, {"type": "phase", "phase": "sam", "text": "SAM 3.1: outlining every object"})
        job.progress("sam", 0.3)
        from concurrent.futures import ThreadPoolExecutor
        from .perception import list_objects, segment_objects
        img = data_url(small)
        t1 = time.time()
        objs0 = list_objects(None, img, data=data)
        t2 = time.time()
        _emit(jid, {"type": "phase", "phase": "sam", "text": f"SAM 3.1: outlining {len(objs0)} objects"})
        with ThreadPoolExecutor(4) as pool:
            objs1 = segment_objects(None, img, objs0, w, h, pool)
        t3 = time.time()
        raw = {"file": str(small), "width": w, "height": h, "objects": objs1}
        models = [
            {"model": "Muse Spark", "role": "tabletop outline", "s": round(t1 - t0m, 2), **_ledger_stat(t_start, ("app.table_top",))},
            {"model": "Muse Spark", "role": "object list", "s": round(t2 - t1, 2), **_ledger_stat(t_start, ("perception.list_objects",))},
            {"model": "SAM 3.1", "role": "masks", "s": round(t3 - t2, 2), **_ledger_stat(t_start, ("perception.sam",))},
        ]
        _emit(jid, {"type": "objects", "n": len(raw["objects"]),
                    "labels": [o.get("label") or o.get("name") for o in raw["objects"]][:40]})
        q = dims.get("corners_pct") or []
        if len(q) != 4 or not all(isinstance(c, list) and len(c) == 2 for c in q):
            q = [[0, 95], [100, 95], [88, 25], [12, 25]]
        key = f"up_{jid}"
        bt.TOPS[key] = [tuple(float(v) for v in c) for c in q]
        bt.table_dims = lambda nn, _d=dims: _d
        job.progress("fit", 0.7)
        _emit(jid, {"type": "phase", "phase": "fit", "text": "Fitting camera, table and object primitives"})
        f, pitch, roll, kind, qq, M = bt.calibrate_top(key, w, h)
        rel = str(small.relative_to(ROOT))
        sc, cam = bt.build(key, raw, M, f, pitch, roll, kind, qq, w, h, rel)
        name = f"photo_{jid}"
        sc["name"] = name
        sc["privacy"] = data
        sc["title"] = "Your photo"
        sc["provenance"] = f"Uploaded photo · Meta API {'Contributor' if data == 'public' else 'Standard'} mode"
        try:
            tex, mean = bt.table_texture(small, cam, sc["table"], M, raw)
            tp = small.with_name(f"{name}.table_tex.png")
            tex.save(tp)
            sc["visual"] = {"table_texture": str(tp.relative_to(ROOT)), "table_rgba": mean, "table_top": kind}
        except Exception as e:  # noqa: BLE001
            job.log(f"table texture skipped: {e}")
        from .pipeline import validate_scene
        problems = validate_scene(sc)
        if problems:
            raise RuntimeError("perceived scene is not simulatable: " + "; ".join(problems[:4]))
        (ROOT / "scenes" / f"{name}.json").write_text(json.dumps(sc, ensure_ascii=False, indent=1))
        try:
            draw_overlay(small, sc).save(small.with_name(f"{name}.overlay.png"))
        except Exception:  # noqa: BLE001
            pass
        cost = _ledger_cost(t_start, ("app.", "perception", "sam", "list", "segment"))
        _emit(jid, {"type": "done", "scene": name, "cost_usd": cost, "models": models})
        sc_models = ROOT / "scenes" / "_models" / f"{name}.json"
        sc_models.parent.mkdir(exist_ok=True); sc_models.write_text(json.dumps(models))
        return {"scene": name, "n_objects": len(sc["objects"]), "cost_usd": cost, "data": data, "models": models,
                "dropped": sc["perception_notes"].get("dropped_off_table", [])}
    return work


# ------------------------------------------------------------------------------------------- SAM 3D refine
def sam3d_start(sid: str, obj: str) -> dict:
    if not sam3d_available():
        raise PermissionError("SAM 3D refine runs only on a local machine with sam3d_env and weights")
    spec = load(sid)
    if obj not in {o["id"] for o in spec["objects"]}:
        raise ValueError(f"no object {obj}")
    key = f"{sid}:{obj}"
    with _lock:
        cur = SAM3D_JOBS.get(key)
        if cur and cur["state"] == "running":
            return cur
    work = ROOT / ".scratch" / "sam3d" / spec["name"]
    work.mkdir(parents=True, exist_ok=True)
    logf = work / f"{obj}.app.log"
    env = dict(os.environ, SAM3D_WEIGHTS=str(ROOT / "weights" / "sam-3d-objects-mlx-8bit"))
    p = subprocess.Popen([str(ROOT / "sam3d_env" / "bin" / "python"), str(ROOT / "scripts" / "sam3d_assets.py"),
                          str(scene_file(sid)), "--work", str(work), "--only", obj],
                         cwd=str(ROOT), env=env, stdout=open(logf, "w"), stderr=subprocess.STDOUT)
    rec = {"scene": sid, "obj": obj, "state": "running", "started": time.time(), "pid": p.pid, "log": str(logf)}
    with _lock:
        SAM3D_JOBS[key] = rec

    def wait():
        rc = p.wait()
        with _lock:
            rec["state"] = "done" if rc == 0 and (ROOT / "assets" / "sam3d" / spec["name"] / f"{obj}.glb").is_file() \
                else "error"
            rec["finished"] = time.time()
    threading.Thread(target=wait, daemon=True).start()
    return rec


def sam3d_status(sid: str) -> list[dict]:
    with _lock:
        out = []
        for k, v in SAM3D_JOBS.items():
            if v["scene"] == sid:
                d = {kk: vv for kk, vv in v.items() if kk not in ("pid", "log")}
                d["elapsed_s"] = round((v.get("finished") or time.time()) - v["started"], 0)
                out.append(d)
        return out
