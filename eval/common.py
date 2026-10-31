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


def code_label() -> str:
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
        for e in entries:
            tasks.append({"scene": name, "scene_path": str(path), "scene_spec": scene,
                          "instruction": e["instruction"], "reference_goals": e.get("reference_goals"),
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
