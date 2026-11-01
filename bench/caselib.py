"""Shared helpers for Rehearsal Bench: loading cases, the stress-test noise model, plan checks,
the scripted baseline, failure classification and a small JSON-schema validator (no extra deps)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "bench"
CASES = BENCH / "cases"
SCHEMA_PATH = BENCH / "schema.json"
SCHEMA_ID = "rehearsal-bench-case"
SCHEMA_VERSION = "0.1"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# spawn-started pool workers (macOS default) must find the same packages
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(REPO), *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != str(REPO)]])

import numpy as np  # noqa: E402

from rehearsal.executor import run_plan  # noqa: E402
from rehearsal.predicates import with_implicit_goals  # noqa: E402
from rehearsal.rehearse import HELDOUT_SEED0, SIGMA, object_sigmas  # noqa: E402

# Stress protocol. Identical to the product's stress test (rehearse._variant), so a bench score on
# the plan a report shipped reproduces that report's k/n on the same seeds and code.
SEED0 = HELDOUT_SEED0          # 1000
N_DEFAULT = 40
FRICTION_RANGE = (0.6, 1.4)
MASS_RANGE = (0.7, 1.3)
BASELINE_TRANSIT = 0.15        # m, OfflineAgent.DEFAULT_TRANSIT

PROTOCOL = {"seed0": SEED0, "n_default": N_DEFAULT, "friction_scale": list(FRICTION_RANGE),
            "mass_scale": list(MASS_RANGE),
            "perception_error": "per object, xy ~ N(0, sigma_m) per axis (sigma from case.json perception)",
            "rng": "numpy default_rng(seed): friction, then mass, then normal(n_objects, 2) (common random numbers)",
            "grasp": "suction weld on the top face (assumed, not simulated)"}


# ------------------------------------------------------------------ loading
def load_case(case_dir: str | Path) -> dict:
    """{dir, meta (case.json), scene, goals, sigmas} for a case folder."""
    d = Path(case_dir)
    meta = json.loads((d / "case.json").read_text())
    scene = json.loads((d / meta["files"]["scene"]).read_text())
    sig = meta["perception"]["sigma_m"]
    sigmas = [float(sig.get(o["id"], meta["perception"].get("default_sigma_m", SIGMA))) for o in scene["objects"]]
    return {"dir": d, "meta": meta, "scene": scene, "goals": meta["goals"], "sigmas": sigmas}


def list_cases(root: Path = CASES) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("*/case.json"))


def case_goals(scene: dict, report_goals: list[dict] | None) -> tuple[list[dict], str]:
    """Reference goals (hand-verified) plus implicit `undisturbed` when the scene has them, else the
    goals the planner compiled in the recorded run (which already include the implicit ones)."""
    ids = [o["id"] for o in scene["objects"]]
    ref = (scene.get("reference") or {}).get("goals")
    if ref:
        return with_implicit_goals([dict(g) for g in ref], ids), "reference"
    if report_goals:
        return [dict(g) for g in report_goals], "planner_compiled"
    raise ValueError("no goals: the scene has no reference block and no report goals were given")


def sigma_map(scene: dict) -> tuple[dict, str]:
    sig = object_sigmas(scene)
    has = any((o.get("perception") or {}).get("sigma_m") for o in scene["objects"])
    src = ("per-object perception sigma_m from the photo fit" if has
           else f"default {SIGMA * 100:.1f} cm for every object (hand-designed scene, no perception)")
    return {o["id"]: round(float(s), 4) for o, s in zip(scene["objects"], sig)}, src


# ------------------------------------------------------------------ plans
def plan_key(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan.get("actions", []), sort_keys=True).encode()).hexdigest()[:12]


def plan_problems(plan, scene: dict) -> list[str]:
    """Format checks on a plan ({"actions": [{obj, dest, relation, transit_height, via?}]})."""
    if not isinstance(plan, dict) or not isinstance(plan.get("actions"), list):
        return ['plan must be an object with an "actions" list']
    ids = {o["id"] for o in scene["objects"]}
    errs = []
    for i, a in enumerate(plan["actions"]):
        if not isinstance(a, dict):
            errs.append(f"action {i} is not an object")
            continue
        for k in ("obj", "dest"):
            if a.get(k) not in ids:
                errs.append(f"action {i}: unknown {k} {a.get(k)!r}")
        if a.get("relation") not in ("in", "on"):
            errs.append(f'action {i}: relation must be "in" or "on"')
        th = a.get("transit_height")
        if not isinstance(th, (int, float)) or isinstance(th, bool) or not 0 < th < 1.5:
            errs.append(f"action {i}: transit_height must be a number in (0, 1.5) m")
        via = a.get("via") or []
        if not isinstance(via, list) or any(not isinstance(v, list) or len(v) != 2 for v in via):
            errs.append(f"action {i}: via must be a list of [x, y]")
    return errs


def scripted_baseline(goals: list[dict], transit: float = BASELINE_TRANSIT) -> dict:
    """Trivial planner: one straight-line pick-place per goal, in the order the goals are listed
    (instruction order), all at one fixed carry height. stacked(a, b, c) -> b on a, then c on b."""
    acts = []
    for g in goals:
        if g.get("implicit"):
            continue
        a = g["args"]
        if g["pred"] in ("on", "inside"):
            acts.append({"obj": a[0], "dest": a[1], "relation": "in" if g["pred"] == "inside" else "on",
                         "transit_height": transit, "via": []})
        elif g["pred"] == "stacked":
            acts += [{"obj": up, "dest": lo, "relation": "on", "transit_height": transit, "via": []}
                     for lo, up in zip(a, a[1:])]
    return {"diagnosis": f"scripted baseline: straight lines in instruction order at {transit} m", "actions": acts}


# ------------------------------------------------------------------ simulation
def goal_str(g: dict) -> str:
    return f"{g['pred']}({', '.join(g['args'])})"


def noise(seed: int, sigmas) -> dict:
    """The product's noise draw for one seed (same order as rehearse._variant)."""
    rng = np.random.default_rng(seed)
    n = len(sigmas)
    return dict(friction_scale=float(rng.uniform(*FRICTION_RANGE)), mass_scale=float(rng.uniform(*MASS_RANGE)),
                perception_error=rng.normal(0, 1, (n, 2)) * np.asarray(sigmas)[:, None])


def simulate(scene: dict, plan: dict, goals: list[dict], seed: int | None = None, sigmas=None) -> dict:
    """One run. seed=None is the nominal run (scene exactly as believed). Never raises on a bad plan."""
    probs = plan_problems(plan, scene)
    if probs:
        return {"seed": seed, "success": False, "violated": [], "execution_error": "invalid plan: " + "; ".join(probs),
                "evidence": [], "events": [], "disturbed": [], "goals": []}
    var = noise(seed, sigmas) if seed is not None else {}
    try:
        r = run_plan(scene, plan, goals, **var)
    except Exception as e:  # noqa: BLE001 - a planner's plan must never crash the scorer
        return {"seed": seed, "success": False, "violated": [], "execution_error": f"simulator error: {type(e).__name__}: {e}",
                "evidence": [], "events": [], "disturbed": [], "goals": []}
    err = r["execution_error"]
    return {"seed": seed, "success": bool(r["success"]),
            "violated": [goal_str(g) for g in r["goals"] if not g["ok"]],
            "goals": [{"pred": g["pred"], "args": g["args"], "ok": g["ok"], "why": g["why"]} for g in r["goals"]],
            "execution_error": err["error"] if err else None,
            "execution_error_detail": {k: v for k, v in err.items() if k != "action"} if err else None,
            "evidence": r["evidence"], "events": r["events"], "disturbed": r["disturbed"]}


def run_job(args):
    """Pool entry point: (scene, plan, goals, seed, sigmas) -> compact result."""
    scene, plan, goals, seed, sigmas = args
    r = simulate(scene, plan, goals, seed, sigmas)
    return {k: r[k] for k in ("seed", "success", "violated", "execution_error")}


def contact_log(events: list[dict]) -> list[dict]:
    """Events worth keeping in a failure record: every contact that is not with the table/floor,
    protective stops, grasps and releases (the timeline of what went wrong)."""
    return [e for e in events if e.get("kind") != "contact" or not {"table", "floor"} & set(e.get("between", []))]


def failure_types(sim: dict, plan: dict) -> list[str]:
    """Same categories as eval/run_eval.py: ik_reach, protective_stop, invalid_plan, collision, pred:*."""
    types = []
    msg = str(sim.get("execution_error") or "").lower()
    if msg:
        if "invalid plan" in msg or "unknown object" in msg or "simulator error" in msg:
            types.append("invalid_plan")
        elif "protective" in msg:
            types.append("protective_stop")
        elif "missed" in msg:
            types.append("grasp_miss")
        else:
            types.append("ik_reach")
    touched = {a.get("obj") for a in plan.get("actions", [])} | {a.get("dest") for a in plan.get("actions", [])}
    for e in sim.get("events", []):
        if e.get("kind") != "contact":
            continue
        pair = set(e.get("between", []))
        if pair & {"table", "floor"}:
            continue
        actor = "robot_arm" in pair or (e.get("while_holding") in pair)
        if actor and (pair - {"robot_arm", e.get("while_holding")}) - touched:
            types.append("collision")
            break
    types += [f"pred:{v.split('(')[0]}" for v in sim.get("violated", [])]
    return sorted(set(types))


def pool(workers: int | None = None):
    from concurrent.futures import ProcessPoolExecutor
    from rehearsal.pipeline import worker_count
    return ProcessPoolExecutor(max_workers=workers or worker_count())


# ------------------------------------------------------------------ provenance
def code_version() -> dict:
    h = hashlib.sha256()
    for f in sorted((REPO / "rehearsal").glob("*.py")):
        h.update(f.name.encode() + f.read_bytes())
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        sha = None
    return {"git": sha, "code_hash": h.hexdigest()[:10]}


# ------------------------------------------------------------------ tiny JSON-schema validator
def validate(obj, schema: dict | None = None, path: str = "$", ref: str | None = None) -> list[str]:
    """Validate against the subset of JSON Schema used by bench/schema.json: type, required,
    properties, additionalProperties (bool or schema), items, enum, const, minimum, pattern, $ref
    (local #/$defs/...), anyOf. Returns a list of problems (empty = valid)."""
    import re
    root = schema if schema is not None else json.loads(SCHEMA_PATH.read_text())
    return _val(obj, {"$ref": ref} if ref else root, root, path, re)


def _val(x, s, root, path, re) -> list[str]:
    if "$ref" in s:
        node = root
        for part in s["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return _val(x, node, root, path, re)
    if "anyOf" in s:
        if not any(not _val(x, sub, root, path, re) for sub in s["anyOf"]):
            return [f"{path}: matches none of anyOf"]
    errs = []
    t = s.get("type")
    if t:
        types = t if isinstance(t, list) else [t]
        ok = any(_is(x, tt) for tt in types)
        if not ok:
            return [f"{path}: expected {t}, got {type(x).__name__}"]
    if "const" in s and x != s["const"]:
        errs.append(f"{path}: must be {s['const']!r}")
    if "enum" in s and x not in s["enum"]:
        errs.append(f"{path}: {x!r} not in {s['enum']}")
    if "minimum" in s and isinstance(x, (int, float)) and x < s["minimum"]:
        errs.append(f"{path}: {x} < {s['minimum']}")
    if "pattern" in s and isinstance(x, str) and not re.search(s["pattern"], x):
        errs.append(f"{path}: {x!r} does not match {s['pattern']}")
    if isinstance(x, dict):
        for k in s.get("required", []):
            if k not in x:
                errs.append(f"{path}: missing {k!r}")
        props = s.get("properties", {})
        extra = s.get("additionalProperties", True)
        for k, v in x.items():
            if k in props:
                errs += _val(v, props[k], root, f"{path}.{k}", re)
            elif extra is False:
                errs.append(f"{path}: unexpected key {k!r}")
            elif isinstance(extra, dict):
                errs += _val(v, extra, root, f"{path}.{k}", re)
    if isinstance(x, list) and "items" in s:
        for i, v in enumerate(x):
            errs += _val(v, s["items"], root, f"{path}[{i}]", re)
    return errs


def _is(x, t: str) -> bool:
    return {"object": isinstance(x, dict), "array": isinstance(x, list), "string": isinstance(x, str),
            "boolean": isinstance(x, bool), "null": x is None,
            "integer": isinstance(x, int) and not isinstance(x, bool),
            "number": isinstance(x, (int, float)) and not isinstance(x, bool)}[t]
