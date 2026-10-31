"""Shared helpers for the eval harness (owned by the eval role).

Code under test is imported from $REHEARSAL_CODE (default: this worktree). That lets the same
harness evaluate a read-only snapshot of an unmerged branch (see eval/snapshot_code.sh) while the
Meta API budget ledger and response cache stay the shared ones in <worktree>/budget.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "eval"
CODE = Path(os.environ.get("REHEARSAL_CODE", REPO)).resolve()
os.environ.setdefault("REHEARSAL_BUDGET_DIR", str((REPO / "budget").resolve()))
os.environ["REHEARSAL_CODE"] = str(CODE)
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
# spawn-started pool workers need to find the same package
os.environ["PYTHONPATH"] = os.pathsep.join([str(CODE), *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]])

import numpy as np  # noqa: E402

from rehearsal import rehearse as R  # noqa: E402
from rehearsal.predicates import with_implicit_goals  # noqa: E402


def code_hash() -> str:
    """Content hash of the rehearsal/ package under test (identifies the code, not the commit)."""
    h = hashlib.sha256()
    for f in sorted((CODE / "rehearsal").glob("*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:10]


def code_label() -> str:
    return f"{_code_origin()} code#{code_hash()}"


def _code_origin() -> str:
    ref = CODE / "SNAPSHOT_REF"
    if ref.exists():
        return "snapshot:" + ref.read_text().strip()
    try:
        import subprocess
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        br = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
        return f"worktree:{br}@{sha}"
    except Exception:  # noqa: BLE001
        return "worktree"


def load_tasks(scenes_dir: Path, only: list[str] | None = None) -> list[dict]:
    """Auto-discover tasks: every scenes/*.json with a demo_instruction, plus eval/tasks.json
    entries (scene -> list of {instruction, reference_goals?}). Returns one dict per task."""
    spec = json.loads((EVAL_DIR / "tasks.json").read_text()) if (EVAL_DIR / "tasks.json").exists() else {}
    tasks = []
    for path in sorted(Path(scenes_dir).glob("*.json")):
        scene = json.loads(path.read_text())
        name = scene.get("name", path.stem)
        if only and name not in only:
            continue
        entries = list(spec.get(name, []))
        if scene.get("demo_instruction") and not any(e["instruction"] == scene["demo_instruction"] for e in entries):
            entries.insert(0, {"instruction": scene["demo_instruction"]})
        sref = scene.get("reference") or {}
        for e in entries:
            demo = e["instruction"] == scene.get("demo_instruction")
            tasks.append({"scene": name, "scene_path": str(path), "scene_spec": scene,
                          "instruction": e["instruction"],
                          # tasks.json wins; else the scene's own verified reference block (demo task only)
                          "reference_goals": e.get("reference_goals") or (sref.get("goals") if demo else None),
                          "reference_plans": {k: sref[k] for k in ("naive_plan", "fixed_plan") if demo and k in sref},
                          "tags": e.get("tags", [])})
    return tasks


def plan_key(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan.get("actions", []), sort_keys=True).encode()).hexdigest()[:12]


def pool(workers: int | None = None) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers or int(os.environ.get("REHEARSAL_WORKERS", 0)) or None)


def stress(scene: dict, plan: dict, goals: list[dict], seeds, sigmas, ex: ProcessPoolExecutor) -> dict:
    """Paired (common-random-number) stress test on explicit seeds. Uses rehearse._variant, so the
    noise model is exactly the product's: friction x U(0.6,1.4), mass x U(0.7,1.3), per-object
    N(0, sigma) xy perception error."""
    sigmas = list(sigmas)
    jobs = [(scene, plan, goals, int(s), sigmas, False) for s in seeds]
    runs = list(ex.map(R._variant, jobs))
    ok = [bool(r["success"]) for r in runs]
    return {"n": len(ok), "successes": int(sum(ok)), "ok": ok,
            "failed": [r["failed"] for r in runs if not r["success"]][:5]}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def final_goals(task: dict, agent_goals: list[dict]) -> list[dict]:
    """Goals used to judge robustness: the task's reference goals if given, else the agent's own."""
    ids = [o["id"] for o in task["scene_spec"]["objects"]]
    if task.get("reference_goals"):
        return with_implicit_goals([dict(g) for g in task["reference_goals"]], ids)
    return agent_goals


def latest_results() -> list[dict]:
    """All result files in eval/results/, newest last."""
    return [json.loads(p.read_text()) for p in sorted((EVAL_DIR / "results").glob("eval_*.json"))]


def final_plans(scenes_dir: Path, only: list[str] | None = None, agents=("offline", "muse"),
                include_initial: bool = False) -> list[dict]:
    """Distinct passing final plans from eval/results/ produced by THIS code (same code_label),
    one entry per (scene, instruction, plan). Falls back to nothing if no results yet."""
    tasks = {(t["scene"], t["instruction"]): t for t in load_tasks(scenes_dir, only)}
    label = "code#" + code_hash()
    seen, out = set(), []
    for res in latest_results():
        if label not in res.get("code", "") and not _legacy_match(res.get("code", "")):
            continue
        for c in res["cells"]:
            if c.get("agent") not in agents or not c.get("success_within_2_revisions"):
                continue
            t = tasks.get((c["scene"], c["instruction"]))
            if not t:
                continue
            pairs = [("", c["attempts"][-1]["plan"])]
            if include_initial and len(c["attempts"]) > 1:
                pairs.append(("_initial", c["attempts"][0]["plan"]))
            for suffix, plan in pairs:
                key = (c["scene"], c["instruction"], plan_key(plan))
                if key in seen:
                    continue
                seen.add(key)
                out.append({"task": t, "plan": plan, "agent": c["agent"] + suffix, "agent_goals": c["goals"],
                            "goals": final_goals(t, c["goals"]), "plan_key": key[2]})
    return out


def reference_plans(scenes_dir: Path, only: list[str] | None = None, kinds=("fixed_plan", "naive_plan")) -> list[dict]:
    """The robotics engineer's hand-verified naive/fixed plans from each scene's `reference` block."""
    out = []
    for t in load_tasks(scenes_dir, only):
        for k in kinds:
            plan = (t.get("reference_plans") or {}).get(k)
            if plan and t.get("reference_goals"):
                out.append({"task": t, "plan": plan, "agent": "ref_" + k.split("_")[0], "agent_goals": None,
                            "goals": final_goals(t, None), "plan_key": plan_key(plan)})
    return out


def _legacy_match(lbl: str) -> bool:
    """Result files written before code hashes were recorded (first offline runs of 00:16/00:17)."""
    return ("snapshot:robotics 979d772" in lbl and "robotics" in _code_origin() and "979d772" in _code_origin()
            and "code#" not in lbl) or (lbl == "worktree:eval@9f0d78a" and CODE == REPO.resolve())
