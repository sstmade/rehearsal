"""Closed-loop evaluation matrix: (scene, instruction, agent, feedback condition, repeat).

    REHEARSAL_OFFLINE=1 .venv/bin/python eval/run_eval.py --agents offline            # $0
    .venv/bin/python eval/run_eval.py --agents muse --repeats 5 --budget 2.0           # Muse Spark
    REHEARSAL_CODE=eval/.snap/robotics .venv/bin/python eval/run_eval.py --scenes eval/.snap/robotics/scenes

Design
- Per (task, agent, repeat) the agent compiles goals and writes a first plan ONCE; attempt 1 is
  simulated once and shared by all feedback conditions (paired design). Only the revise step differs:
    full_log      the product's history (failed predicates with numbers, contact events, IK errors,
                  and on the robotics branch: evidence lines, disturbances, arm-contact forces)
    outcome_only  per attempt: plan, success flag, names/args of failed predicates, "aborted" flag
    none          per attempt: plan and success=false only ("try again")
- Up to MAX_ATTEMPTS=3 attempts (= first try + 2 revisions), stopping when the agent's own goals pass,
  as in the product. Success is ALSO scored against hand-written reference goals (eval/tasks.json)
  that the agent never sees, so an agent cannot "succeed" by compiling easy goals.
- The final plan (if it passed) is stress-tested on n seeds (common random numbers: seed s gives the
  same friction/mass/perception draw for every plan) against the reference goals.
- Muse repeats > 0 append an inert "[eval trial k]" line to each prompt so the response cache keys
  differ (repeat 0 is byte-identical to the product's prompt). Calls are tagged caller="eval.*" and
  costs are read back from the shared ledger.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (sets up sys.path for the code under test)

from rehearsal import agents as A  # noqa: E402
from rehearsal import meta_client as M  # noqa: E402
from rehearsal.executor import reachability_report, run_plan  # noqa: E402
from rehearsal.predicates import with_implicit_goals  # noqa: E402

MAX_ATTEMPTS = 3
CONDITIONS = ("full_log", "outcome_only", "none")
DROP = ("frames", "trajectory")


# ------------------------------------------------------------------ ledger / budget
def ledger_path() -> Path:
    return M.budget_dir() / "ledger.jsonl"


def eval_spend() -> float:
    p = ledger_path()
    if not p.exists():
        return 0.0
    tot = 0.0
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(e.get("caller", "")).startswith("eval."):
            tot += float(e.get("cost", 0.0))
    return tot


def ledger_since(offset: int, tag: str = "") -> list[dict]:
    p = ledger_path()
    if not p.exists():
        return []
    with open(p, "rb") as f:
        f.seek(offset)
        lines = f.read().decode("utf-8", "replace").splitlines()
    out = []
    for line in lines:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = str(e.get("caller", ""))
        if c.startswith("eval.") and (not tag or c.endswith(tag)):
            out.append(e)
    return out


def ledger_offset() -> int:
    p = ledger_path()
    return p.stat().st_size if p.exists() else 0


class BudgetStop(RuntimeError):
    pass


class EvalMuse(A.MuseAgent):
    """MuseAgent with eval caller tags, a per-repeat cache-busting tag and an eval budget guard.
    Prompts and schemas are the product's (inherited compile_goals/plan/revise)."""

    def __init__(self, data: str, trial: int, budget: float):
        super().__init__(data=data)
        self.trial, self.budget, self.calls = trial, budget, 0
        self.tag = ""  # caller suffix, so concurrent cells can be told apart in the ledger

    def _json(self, user, schema, name):
        if eval_spend() >= self.budget:
            raise BudgetStop(f"eval Meta spend reached ${self.budget:.2f}")
        if self.trial:
            user = user + f"\n\n[eval trial {self.trial}]"
        self.calls += 1
        return M.chat_json([{"role": "system", "content": A.SYSTEM}, {"role": "user", "content": user}],
                           schema, name, caller=f"eval.{name.lower()}{self.tag}", data=self.data,
                           model=self.model, effort=self.effort)


class EvalOffline(A.OfflineAgent):
    calls = 0
    tag = ""


def make(agent: str, data: str, trial: int, budget: float):
    if agent == "offline":
        return EvalOffline()
    if not os.environ.get("MODEL_API_KEY"):
        raise SystemExit("muse agent needs MODEL_API_KEY")
    return EvalMuse(data, trial, budget)


# ------------------------------------------------------------------ history filters
def filter_history(history: list[dict], cond: str, agent_kind: str) -> list[dict]:
    if cond == "full_log":
        return history
    out = []
    for h in history:
        d = {"attempt": h["attempt"], "plan": h["plan"], "success": h["success"]}
        if cond == "outcome_only":
            d["failed_goals"] = [{"pred": g["pred"], "args": g["args"]} for g in h["goals"] if not g["ok"]]
            if h.get("execution_error"):
                d["execution_aborted"] = True
        if agent_kind == "offline":
            d["events"] = []  # OfflineAgent.revise indexes events; give it nothing to read
        out.append(d)
    return out


# ------------------------------------------------------------------ one simulated attempt
def failure_types(r: dict, plan: dict) -> list[str]:
    types = []
    err = r.get("execution_error")
    if err:
        msg = str(err.get("error", "")).lower()
        if "unknown object" in msg or "invalid plan" in msg:
            types.append("invalid_plan")
        elif "protective" in msg:
            types.append("protective_stop")
        elif "missed" in msg:
            types.append("grasp_miss")
        else:
            types.append("ik_reach")
    touched = {a.get("obj") for a in plan.get("actions", [])} | {a.get("dest") for a in plan.get("actions", [])}
    for e in r.get("events", []):
        if e.get("kind") != "contact":
            continue
        pair = set(e.get("between", []))
        if pair & {"table", "floor"}:
            continue
        actor = "robot_arm" in pair or (e.get("while_holding") in pair)
        others = pair - {"robot_arm", e.get("while_holding")}
        if actor and others - touched:
            types.append("collision")
            break
    types += [f"pred:{g['pred']}" for g in r["goals"] if not g["ok"]]
    return sorted(set(types))


def simulate(task: dict, plan: dict, agent_goals: list[dict], ref_goals: list[dict] | None) -> dict:
    t0 = time.time()
    goals = agent_goals + (ref_goals or [])
    try:
        r = run_plan(task["scene_spec"], plan, goals)
    except Exception as e:  # noqa: BLE001 - e.g. a hallucinated object id on main's executor
        r = {"success": False, "goals": [dict(g, ok=False, why="not run") for g in goals],
             "execution_error": {"error": f"invalid plan: {type(e).__name__}: {e}"}, "events": [], "sim_time": 0.0}
    for k in DROP:
        r.pop(k, None)
    res = r["goals"]
    na = len(agent_goals)
    agent_res, ref_res = res[:na], res[na:]
    ok_exec = r.get("execution_error") is None
    agent_view = {k: v for k, v in r.items() if k != "goals"}
    agent_view.update(goals=agent_res, success=ok_exec and all(g["ok"] for g in agent_res))
    return {"agent_view": agent_view,
            "ref_success": (ok_exec and all(g["ok"] for g in ref_res)) if ref_goals else None,
            "failure_types": failure_types(agent_view, plan),
            "sim_wall_s": round(time.time() - t0, 2)}


# ------------------------------------------------------------------ the matrix
def run_cell_group(task, agent_kind, rep, args, reach_cache, ex, stress_cache, log):
    """Returns a list of cells (one per feedback condition) sharing goals/plan/attempt 1."""
    scene = task["scene_spec"]
    data = "public" if scene.get("privacy") == "public" else "private"
    agent = make(agent_kind, data, rep, args.budget)
    if task["scene"] not in reach_cache:
        reach_cache[task["scene"]] = reachability_report(scene)
    summary = A.scene_summary(scene, reach_cache[task["scene"]])
    ids = [o["id"] for o in scene["objects"]]
    ref = C.final_goals(task, None) if task.get("reference_goals") else None
    base = {"scene": task["scene"], "instruction": task["instruction"], "agent": agent_kind,
            "agent_name": agent.name, "data": data, "repeat": rep, "tags": task.get("tags", [])}

    off0 = ledger_offset()
    gid = f".{task['scene']}.{C.hashlib.sha1(task['instruction'].encode()).hexdigest()[:6]}.r{rep}"
    agent.tag = gid + ".plan"
    t_agent = time.time()
    raw_goals = agent.compile_goals(task["instruction"], summary)
    goals = with_implicit_goals(raw_goals, ids)
    if not raw_goals:
        return [dict(base, condition=c, status="unsupported",
                     note="agent compiled no goals (OfflineAgent only knows the desk_01 vocabulary)")
                for c in args.conditions]
    plan0 = agent.plan(task["instruction"], summary, goals)
    agent_s = time.time() - t_agent
    first = simulate(task, plan0, goals, ref)
    shared_paid = ledger_since(off0, agent.tag)
    shared_calls = agent.calls

    cells = []
    for cond in args.conditions:
        off = ledger_offset()
        calls_before = agent.calls
        agent.tag = f"{gid}.{cond}"
        history = [{"attempt": 1, "plan": plan0, **first["agent_view"]}]
        attempts = [{"attempt": 1, "plan": plan0, "success": first["agent_view"]["success"],
                     "ref_success": first["ref_success"], "failure_types": first["failure_types"],
                     "diagnosis": plan0.get("diagnosis", ""), "sim_wall_s": first["sim_wall_s"],
                     "execution_error": (first["agent_view"].get("execution_error") or {}).get("error")}]
        sim_s, rev_s, status = first["sim_wall_s"], 0.0, "ok"
        try:
            while not history[-1]["success"] and len(history) < MAX_ATTEMPTS:
                t = time.time()
                plan = agent.revise(task["instruction"], summary, goals, filter_history(history, cond, agent_kind))
                rev_s += time.time() - t
                s = simulate(task, plan, goals, ref)
                sim_s += s["sim_wall_s"]
                n = len(history) + 1
                history.append({"attempt": n, "plan": plan, **s["agent_view"]})
                attempts.append({"attempt": n, "plan": plan, "success": s["agent_view"]["success"],
                                 "ref_success": s["ref_success"], "failure_types": s["failure_types"],
                                 "diagnosis": plan.get("diagnosis", ""), "sim_wall_s": s["sim_wall_s"],
                                 "execution_error": (s["agent_view"].get("execution_error") or {}).get("error")})
        except BudgetStop as e:
            status = f"budget_stop: {e}"
        except Exception as e:  # noqa: BLE001 - API errors etc.: record, keep the matrix going
            status = f"error: {type(e).__name__}: {e}"
            log(traceback.format_exc())
        paid = ledger_since(off, agent.tag)
        if cond == args.conditions[0]:
            paid = shared_paid + paid
        final = attempts[-1]
        rob = None
        if final["success"] and args.stress:
            sgoals = ref or goals
            key = (task["scene"], task["instruction"], C.plan_key(final["plan"]))
            if key not in stress_cache:
                t = time.time()
                st = C.stress(scene, final["plan"], sgoals, range(args.stress), [R_SIGMA] * len(ids), ex)
                st["wall_s"] = round(time.time() - t, 1)
                stress_cache[key] = st
            st = stress_cache[key]
            rob = {"n": st["n"], "successes": st["successes"], "wall_s": st["wall_s"],
                   "example_failures": st["failed"][:3]}
        ref_ok = [a["ref_success"] for a in attempts]
        cell = dict(base, condition=cond, status=status,
                    goals=goals, goals_match_reference=_goal_match(raw_goals, task.get("reference_goals")),
                    first_try_success=attempts[0]["success"], first_try_ref_success=attempts[0]["ref_success"],
                    success_within_2_revisions=final["success"],
                    ref_success_within_2_revisions=final["ref_success"],
                    attempts_used=len(attempts),
                    revision_needed=not attempts[0]["success"],
                    failure_types_first=attempts[0]["failure_types"],
                    failure_types_all=sorted({t for a in attempts for t in a["failure_types"]}),
                    robustness=rob, attempts=attempts,
                    sim_wall_s=round(sim_s, 2), agent_wall_s=round(agent_s + rev_s, 2),
                    api_calls=(agent.calls - calls_before) + (shared_calls if cond == args.conditions[0] else 0),
                    api_paid_calls=len(paid),
                    prompt_tokens=sum(e.get("prompt_tokens", 0) for e in paid),
                    completion_tokens=sum(e.get("completion_tokens", 0) for e in paid),
                    cost_usd=round(sum(e.get("cost", 0.0) for e in paid), 5),
                    ref_success_by_attempt=ref_ok)
        cells.append(cell)
        log(f"  [{task['scene']}|{agent_kind}|{cond}|r{rep}] first={attempts[0]['success']} "
            f"final={final['success']} ref={final['ref_success']} attempts={len(attempts)} "
            f"types={cell['failure_types_all']} rob={rob and str(rob['successes']) + '/' + str(rob['n'])} "
            f"${cell['cost_usd']} {status if status != 'ok' else ''}")
        if attempts[0]["success"]:
            # first try passed: every other condition is identical by construction
            for c2 in args.conditions[args.conditions.index(cond) + 1:]:
                cells.append(dict(cell, condition=c2, api_calls=0, api_paid_calls=0, prompt_tokens=0,
                                  completion_tokens=0, cost_usd=0.0, shared_first_try=True))
            break
    return cells


R_SIGMA = C.R.SIGMA


def _goal_match(raw: list[dict], ref: list[dict] | None):
    if not ref:
        return None
    norm = lambda gs: sorted((g["pred"], tuple(g["args"])) for g in gs if not g.get("implicit"))
    return norm(raw) == norm(ref)


def aggregate(cells: list[dict]) -> list[dict]:
    groups: dict = {}
    for c in cells:
        if c.get("status") == "unsupported":
            continue
        groups.setdefault((c["scene"], c["instruction"], c["agent"], c["condition"]), []).append(c)
    rows = []
    for (scene, ins, agent, cond), cs in groups.items():
        n = len(cs)
        rob = [c["robustness"] for c in cs if c.get("robustness")]
        ft = {}
        for c in cs:
            for t in c["failure_types_all"]:
                ft[t] = ft.get(t, 0) + 1
        mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
        rows.append({
            "scene": scene, "instruction": ins, "agent": agent, "condition": cond, "n": n,
            "first_try": sum(bool(c["first_try_success"]) for c in cs),
            "first_try_ref": sum(bool(c["first_try_ref_success"]) for c in cs),
            "within_2_rev": sum(bool(c["success_within_2_revisions"]) for c in cs),
            "within_2_rev_ref": sum(bool(c["ref_success_within_2_revisions"]) for c in cs),
            "needed_revision": sum(bool(c["revision_needed"]) for c in cs),
            "mean_attempts": mean([c["attempts_used"] for c in cs]),
            "goals_match_ref": sum(bool(c.get("goals_match_reference")) for c in cs),
            "robustness_mean": mean([r["successes"] / r["n"] for r in rob]),
            "robustness_n": rob[0]["n"] if rob else None,
            "failure_types": ft,
            "sim_wall_s_mean": mean([c["sim_wall_s"] for c in cs]),
            "agent_wall_s_mean": mean([c["agent_wall_s"] for c in cs]),
            "cost_usd": round(sum(c["cost_usd"] for c in cs), 4),
            "tokens": sum(c["prompt_tokens"] + c["completion_tokens"] for c in cs),
            "errors": sum(1 for c in cs if c["status"] != "ok"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=str(C.REPO / "scenes"))
    ap.add_argument("--only", default="", help="comma-separated scene names")
    ap.add_argument("--agents", default="offline", help="offline,muse")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--repeats", type=int, default=1, help="Muse repeats (offline is deterministic: 1)")
    ap.add_argument("--stress", type=int, default=40)
    ap.add_argument("--budget", type=float, default=float(os.environ.get("EVAL_BUDGET", 2.0)),
                    help="stop Muse calls once ledger spend tagged eval.* reaches this ($)")
    ap.add_argument("--tag", default="", help="label stored in the result file name")
    ap.add_argument("--parallel", type=int, default=1, help="concurrent (task, repeat) groups (API-bound)")
    ap.add_argument("--max-tasks", type=int, default=0, help="smoke runs: only the first N tasks")
    args = ap.parse_args()
    args.conditions = [c for c in args.conditions.split(",") if c]
    agents = [a for a in args.agents.split(",") if a]
    if "muse" in agents and os.environ.get("REHEARSAL_OFFLINE"):
        raise SystemExit("REHEARSAL_OFFLINE is set; refusing to run the muse agent")

    tasks = C.load_tasks(Path(args.scenes), [s for s in args.only.split(",") if s] or None)
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    print(f"code: {C.code_label()}  tasks: {len(tasks)}  agents: {agents}  conditions: {args.conditions}")
    budget0 = M.status()
    spend0 = eval_spend()
    t0 = time.time()
    cells, reach_cache, stress_cache = [], {}, {}
    jobs = [(task, agent_kind, rep) for task in tasks for agent_kind in agents
            for rep in range(args.repeats if agent_kind == "muse" else 1)]

    def one(job):
        task, agent_kind, rep = job
        try:
            return run_cell_group(task, agent_kind, rep, args, reach_cache, ex, stress_cache, print)
        except BudgetStop as e:
            print(f"  BUDGET STOP: {e}")
            return [{"scene": task["scene"], "instruction": task["instruction"],
                     "agent": agent_kind, "repeat": rep, "status": f"budget_stop: {e}"}]
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return [{"scene": task["scene"], "instruction": task["instruction"],
                     "agent": agent_kind, "repeat": rep, "status": f"error: {type(e).__name__}: {e}"}]

    from concurrent.futures import ThreadPoolExecutor
    with C.pool() as ex, ThreadPoolExecutor(max(1, args.parallel)) as tp:
        for group in tp.map(one, jobs):
            cells += group
    ok_cells = [c for c in cells if "condition" in c]
    out = {"kind": "run_eval", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": C.code_label(),
           "scenes_dir": os.path.relpath(args.scenes, C.REPO), "args": {k: v for k, v in vars(args).items()},
           "max_attempts": MAX_ATTEMPTS, "sigma_m": R_SIGMA, "wall_s": round(time.time() - t0, 1),
           "budget_before": budget0, "budget_after": M.status(),
           "eval_spend_this_run": round(eval_spend() - spend0, 5),
           "aggregate": aggregate(ok_cells), "cells": cells}
    rdir = C.EVAL_DIR / "results"
    rdir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"eval_{stamp}{'_' + args.tag if args.tag else ''}"
    (rdir / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    cols = ["scene", "instruction", "agent", "condition", "repeat", "status", "first_try_success",
            "first_try_ref_success", "success_within_2_revisions", "ref_success_within_2_revisions",
            "attempts_used", "failure_types_all", "robustness", "sim_wall_s", "agent_wall_s",
            "api_calls", "api_paid_calls", "prompt_tokens", "completion_tokens", "cost_usd"]
    with open(rdir / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in ok_cells:
            row = []
            for k in cols:
                v = c.get(k)
                if k == "robustness" and v:
                    v = f"{v['successes']}/{v['n']}"
                elif isinstance(v, list):
                    v = "|".join(map(str, v))
                row.append(v)
            w.writerow(row)
    print(f"\nwrote {rdir / name}.json/.csv  ({out['wall_s']} s, eval spend this run ${out['eval_spend_this_run']})")
    for r in out["aggregate"]:
        print(f"  {r['scene']:<14} {r['agent']:<8} {r['condition']:<13} n={r['n']} first={r['first_try']} "
              f"≤2rev={r['within_2_rev']} ref={r['within_2_rev_ref']} att={r['mean_attempts']} "
              f"rob={r['robustness_mean']} ${r['cost_usd']}")


if __name__ == "__main__":
    main()
