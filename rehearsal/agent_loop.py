"""Agentic rehearsal: Muse Spark drives the rehearsal itself through function tools.

Instead of our fixed plan -> simulate -> revise pipeline, the model gets four tools and decides
what to do next:

    simulate(actions)            run the plan once in the MuJoCo twin, get the physics evidence
    stress_test(actions, n)      rerun it in n perturbed universes (friction, mass, perception error)
    look_again(object_id)        re-observe one object (its perception sigma shrinks), rerun the stress test
    submit(actions, rationale)   final answer — physics re-checks it; a failing plan is rejected

Hard caps: MAX_TOOL_CALLS tool calls and MAX_COST_USD per run. The full trace (the model's own
narration, every tool call and result, tokens and $ per turn) is returned for the report's optional
`agent_trace` key.

    python -m rehearsal.agent_loop scenes/windowsill_01.json "把马克杯放到托盘上"

`feedback` controls how much of the simulator log the model sees (log-feedback ablation):
    full_log      failed predicates with numbers, IK diagnostics, contact evidence, disturbed objects
    outcome_only  success flag, names of failed predicates, "aborted" flag
    none          success flag only
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import meta_client as M
from .agents import ACTION_ITEM, MuseAgent, SYSTEM, scene_summary
from .executor import reachability_report, run_plan
from .predicates import with_implicit_goals
from .scene import load_scene

MAX_TOOL_CALLS = 8
MAX_COST_USD = 0.10
MAX_TURNS = 12            # model turns (a turn may carry several parallel tool calls)
STRESS_N_DEFAULT = 16
STRESS_N_MAX = 40
FEEDBACKS = ("full_log", "outcome_only", "none")
ALL_TOOLS = ("simulate", "stress_test", "look_again", "submit")

_ACTIONS = {"type": "array", "items": ACTION_ITEM}


def _fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc, "strict": True,
        "parameters": {"type": "object", "properties": props, "required": required,
                       "additionalProperties": False}}}


TOOL_SPECS = {
    "simulate": _fn(
        "simulate",
        "Run the action sequence once in the physics digital twin (nominal friction/mass, objects exactly "
        "where perception put them). Returns whether every goal predicate holds, plus simulator evidence.",
        {"actions": _ACTIONS}, ["actions"]),
    "stress_test": _fn(
        "stress_test",
        "Rerun the action sequence in n randomized universes: friction x0.6-1.4, mass x0.7-1.3 and each "
        "object displaced by its perception error (Gaussian, per-object sigma). Returns successes/n and "
        "which objects' perception error correlates with the failures. n between 8 and 40.",
        {"actions": _ACTIONS, "n": {"type": "integer"}}, ["actions", "n"]),
    "look_again": _fn(
        "look_again",
        "Re-observe one object (a second, closer look), shrinking its perception error, then rerun the last "
        "stress test with the same random seeds. Use it when perception error on a specific object drives "
        "stress-test failures.",
        {"object_id": {"type": "string"}}, ["object_id"]),
    "submit": _fn(
        "submit",
        "Submit the final action sequence. Physics re-checks it; a plan that fails in simulation is rejected "
        "and you may continue.",
        {"actions": _ACTIONS, "rationale": {"type": "string"}}, ["actions", "rationale"]),
}

LOOP_RULES = """You are rehearsing a manipulation task in a physics digital twin before a real robot does it.
You have tools: simulate, stress_test, look_again, submit. Work like an engineer:
propose a plan, simulate it, read the physics evidence, fix the cause, and only submit a plan that
physics has passed. If it is cheap to improve robustness (stress_test, look_again), do it before submitting.
Before every tool call, write one or two short English sentences saying what you are doing and why (this is
shown to the user). You have at most {max_calls} tool calls in total; finish with submit.
Action format: {{"obj", "dest", "relation": "in"|"on", "transit_height": tool-tip height in m above the
table while carrying, "via": list of [x, y] waypoints (m) to pass through at transit_height between lift
and arrival above dest, [] for a straight carry}}."""


def _sigma0(obj: dict) -> float:
    from .rehearse import SIGMA
    return float((obj.get("perception") or {}).get("sigma_m") or SIGMA)


def _fmt_failed(goals):
    return [f"{g['pred']}({', '.join(g['args'])}): {g['why']}" for g in goals if not g["ok"]]


def sim_report(r: dict, feedback: str) -> dict:
    """What the model sees after a simulation, per feedback condition."""
    out = {"success": r["success"]}
    if feedback == "none":
        return out
    if feedback == "outcome_only":
        out["failed_goals"] = [f"{g['pred']}({', '.join(g['args'])})" for g in r["goals"] if not g["ok"]]
        if r.get("execution_error"):
            out["execution_aborted"] = True
        return out
    out["failed_goals"] = _fmt_failed(r["goals"])
    out["passed_goals"] = [f"{g['pred']}({', '.join(g['args'])})" for g in r["goals"] if g["ok"]]
    err = r.get("execution_error")
    if err:
        out["execution_error"] = {k: v for k, v in err.items() if k != "action"}
    if r.get("evidence") is not None:          # robotics executor: plain-language evidence lines
        out["evidence"] = r["evidence"][:12]
        if r.get("arm_contacts"):
            out["arm_contacts"] = r["arm_contacts"][:6]
    else:                                      # older executor: raw contact events
        out["contacts"] = [e for e in r["events"] if e["kind"] in ("contact", "ik_failure", "grasp_miss")
                           and not {"table", "floor"} & set(e.get("between", []))][:12]
    return out


def stress_summary(st: dict, ids: list[str]) -> dict:
    runs = st["runs"]
    fails = [r for r in runs if not r["success"]]
    oks = [r for r in runs if r["success"]]
    out = {"successes": st["successes"], "n": st["n"],
           "sigma_cm": {k: round(v, 2) for k, v in st["sigma_cm"].items()}}
    if fails and oks:
        gap = {oid: float(np.mean([r["perception_error_cm"][oid] for r in fails])
                          - np.mean([r["perception_error_cm"][oid] for r in oks])) for oid in ids}
        top = sorted(gap.items(), key=lambda kv: -kv[1])[:3]
        out["perception_error_driving_failures"] = [
            {"object": k, "mean_error_cm_in_failures_minus_successes": round(v, 2)} for k, v in top if v > 0]
    reasons = {}
    for r in fails:
        for f in r["failed"]:
            key = f.split(":")[0][:80]
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        out["failure_reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:5])
    return out


class _Ledger:
    """Reads back real usage/cost of our own calls from the shared ledger."""

    def __init__(self, caller: str):
        self.caller = caller
        self.p = M.budget_dir() / "ledger.jsonl"
        self.off = self.p.stat().st_size if self.p.exists() else 0

    def new(self) -> list[dict]:
        if not self.p.exists():
            return []
        with open(self.p, "rb") as f:
            f.seek(self.off)
            data = f.read()
        self.off += len(data)
        out = []
        for line in data.decode("utf-8", "replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("caller") == self.caller:
                out.append(e)
        return out


class AgentLoop:
    def __init__(self, scene: dict, instruction: str, *, data: str | None = None, feedback: str = "full_log",
                 tools=ALL_TOOLS, max_calls: int = MAX_TOOL_CALLS, max_cost: float = MAX_COST_USD,
                 goals: list[dict] | None = None, model: str = "muse-spark-1.3", effort: str | None = None,
                 caller: str = "agent.loop", trial: int = 0, log=print, pool=None):
        assert feedback in FEEDBACKS
        self.scene, self.instruction = scene, instruction
        self.data = data or ("public" if scene.get("privacy") == "public" else "private")
        self.feedback, self.tools = feedback, [t for t in ALL_TOOLS if t in tools]
        self.max_calls, self.max_cost = max_calls, max_cost
        self.model, self.effort, self.caller, self.trial, self.log = model, effort, caller, trial, log
        self.ids = [o["id"] for o in scene["objects"]]
        self.sigmas = {o["id"]: _sigma0(o) for o in scene["objects"]}
        self.given_goals = goals
        self.pool = pool
        self.attempts: list[dict] = []     # every simulate/submit execution, CONTRACTS §2 attempt shape
        self.stress_runs: list[dict] = []  # raw stress results (for the report)
        self.looks: list[dict] = []
        self.last_stress = None            # (plan, n, summary)
        self.trace: list[dict] = []
        self.cost = 0.0
        self.n_calls = 0
        self.final = None

    # ------------------------------------------------------------------ tools
    def _plan(self, args, diagnosis=""):
        acts = []
        for a in args.get("actions") or []:
            acts.append({"obj": a.get("obj", ""), "dest": a.get("dest", ""),
                         "relation": a.get("relation", "on"),
                         "transit_height": float(a.get("transit_height", 0.15)),
                         "via": [list(map(float, v[:2])) for v in (a.get("via") or []) if len(v) >= 2]})
        return {"diagnosis": diagnosis, "actions": acts}

    def _simulate(self, plan, kind):
        r = run_plan(self.scene, plan, self.goals)
        for k in ("frames", "trajectory"):
            r.pop(k, None)
        self.attempts.append({"attempt": len(self.attempts) + 1, "plan": plan, "via_tool": kind, **r})
        return r

    def _stress(self, plan, n, sigmas=None):
        from .rehearse import _process_pool, stress_test
        sig = [self.sigmas[i] for i in self.ids] if sigmas is None else sigmas
        if self.pool is None:
            self.pool = _process_pool()
        st = stress_test(self.scene, plan, self.goals, n, sig, self.pool)
        return st

    def call_tool(self, name, args, narration):
        if name == "simulate":
            plan = self._plan(args, narration)
            r = self._simulate(plan, "simulate")
            return sim_report(r, self.feedback)
        if name == "stress_test":
            plan = self._plan(args, narration)
            n = int(min(max(int(args.get("n") or STRESS_N_DEFAULT), 8), STRESS_N_MAX))
            st = self._stress(plan, n)
            s = stress_summary(st, self.ids)
            self.last_stress = (plan, n, s)
            self.stress_runs.append({"plan": plan, **st})
            return s
        if name == "look_again":
            oid = args.get("object_id")
            if oid not in self.sigmas:
                return {"error": f"unknown object id {oid!r}; ids: {self.ids}"}
            if self.last_stress is None:
                return {"error": "run stress_test first; look_again reruns the last stress test"}
            from .rehearse import SIGMA_REFINED
            plan, n, before = self.last_stress
            old = self.sigmas[oid]
            self.sigmas[oid] = min(old, SIGMA_REFINED)
            st = self._stress(plan, n)
            s = stress_summary(st, self.ids)
            self.looks.append({"look_at": oid, "before": before["successes"], "after": s["successes"],
                               "n": n, "sigma_cm_before": round(old * 100, 2),
                               "sigma_cm_after": round(self.sigmas[oid] * 100, 2)})
            self.last_stress = (plan, n, s)
            self.stress_runs.append({"plan": plan, "look_at": oid, **st})
            return {"object": oid, "sigma_cm": [round(old * 100, 2), round(self.sigmas[oid] * 100, 2)],
                    "before": f"{before['successes']}/{n}", "after": f"{s['successes']}/{n}", "stress": s}
        if name == "submit":
            plan = self._plan(args, args.get("rationale", "") or narration)
            r = self._simulate(plan, "submit")
            left = self.max_calls - self.n_calls
            if r["success"] or left <= 0:
                self.final = {"plan": plan, "rationale": args.get("rationale", ""), "verified": r["success"]}
                return {"accepted": r["success"], "physics": sim_report(r, self.feedback)}
            return {"accepted": False, "reason": "physics rejected the submitted plan",
                    "physics": sim_report(r, self.feedback), "tool_calls_left": left}
        return {"error": f"unknown tool {name}"}

    # ------------------------------------------------------------------ loop
    def _system(self, summary) -> str:
        # Stable prefix (system rules + scene) first so prompt caching hits on every later turn.
        return (SYSTEM + "\n\n" + LOOP_RULES.format(max_calls=self.max_calls)
                + f"\n\nScene objects (positions are perception's estimate):\n{json.dumps(summary)}")

    def run(self) -> dict:
        t0 = time.time()
        reach = reachability_report(self.scene)
        summary = scene_summary(self.scene, reach)
        goal_cost = 0.0
        if self.given_goals is None:
            agent = MuseAgent(data=self.data, model=self.model, effort=self.effort)
            led = _Ledger("agent.goals")
            goals = agent.compile_goals(self.instruction, summary)
            goal_cost = sum(e["cost"] for e in led.new())
        else:
            goals = [g for g in self.given_goals if not g.get("implicit")]
        self.goals = with_implicit_goals(goals, self.ids)
        self.cost += goal_cost

        user = (f"Instruction: {self.instruction}\nGoal predicates (checked by the simulator): "
                f"{json.dumps(self.goals)}\n\nRehearse the task and submit a plan that physics passes.")
        if self.trial:
            user += f"\n\n[trial {self.trial}]"    # inert; makes repeat runs bypass the response cache
        messages = [{"role": "system", "content": self._system(summary)}, {"role": "user", "content": user}]
        tools = [TOOL_SPECS[t] for t in self.tools]
        led = _Ledger(self.caller)
        stop = None
        nudges = 0
        for turn in range(MAX_TURNS):
            msg = M.chat_tools(messages, tools, caller=self.caller, data=self.data, model=self.model,
                               effort=self.effort, cache_key=f"rehearsal-loop-{self.scene['name']}")
            usage = led.new()
            tc = msg.get("tool_calls") or []
            step = {"turn": turn + 1, "text": (msg.get("content") or "").strip(), "tool_calls": [],
                    "prompt_tokens": sum(e.get("prompt_tokens", 0) for e in usage),
                    "cached_tokens": sum(e.get("cached_tokens", 0) for e in usage),
                    "completion_tokens": sum(e.get("completion_tokens", 0) for e in usage),
                    "cost": round(sum(e["cost"] for e in usage), 6), "cache_hit": not usage}
            self.cost += step["cost"]
            messages.append({"role": "assistant", "content": msg.get("content"),
                             **({"tool_calls": tc} if tc else {})})
            if step["text"]:
                self.log(f"[model] {step['text']}")
            if not tc:
                self.trace.append(step)
                if nudges >= 1:
                    stop = "model stopped without submit"
                    break
                nudges += 1
                messages.append({"role": "user", "content": "Use the tools: simulate the plan, and finish by calling submit."})
                continue
            for call in tc:
                fn = call["function"]["name"]
                res_time = None
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError as e:
                    args, res = {}, {"error": f"arguments were not valid JSON: {e}"}
                else:
                    if self.n_calls >= self.max_calls:
                        res = {"error": "tool-call budget exhausted"}
                    else:
                        self.n_calls += 1
                        t1 = time.time()
                        try:
                            res = self.call_tool(fn, args, step["text"])
                        except Exception as e:  # noqa: BLE001 - surface tool errors to the model
                            res = {"error": f"{type(e).__name__}: {e}"}
                        res_time = round(time.time() - t1, 1)
                step["tool_calls"].append({"name": fn, "args": args, "result": res,
                                           "seconds": res_time})
                self.log(f"  -> {fn}({_short_args(fn, args)}) = {_short_res(fn, res)}")
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(res, ensure_ascii=False, default=float)})
            self.trace.append(step)
            if self.final is not None:
                stop = "submitted"
                break
            if self.n_calls >= self.max_calls:
                stop = "tool-call cap"
                break
            if self.cost >= self.max_cost:
                stop = "cost cap"
                break
        else:
            stop = "turn cap"
        if self.pool is not None:
            self.pool.shutdown()
            self.pool = None

        if self.final is None and self.attempts:   # fall back to the last simulated plan
            last = self.attempts[-1]
            self.final = {"plan": last["plan"], "rationale": "(not submitted; last simulated plan)",
                          "verified": last["success"]}
        return {
            "goals": self.goals, "reachability": reach, "attempts": self.attempts,
            "final": self.final, "stress_runs": self.stress_runs, "looks": self.looks,
            "final_sigmas": [self.sigmas[i] for i in self.ids],
            "agent_trace": {
                "model": M.muse_model(self.data, self.model), "data": self.data, "feedback": self.feedback,
                "tools": self.tools, "caps": {"tool_calls": self.max_calls, "usd": self.max_cost},
                "stopped": stop, "tool_calls": self.n_calls, "turns": self.trace,
                "goal_compile_cost": round(goal_cost, 6), "cost_usd": round(self.cost, 6),
                "prompt_tokens": sum(t["prompt_tokens"] for t in self.trace),
                "cached_tokens": sum(t["cached_tokens"] for t in self.trace),
                "completion_tokens": sum(t["completion_tokens"] for t in self.trace),
                "submitted": self.final, "wall_time_s": round(time.time() - t0, 1),
                "story": story(self.trace)},
        }


def _short_args(fn, a):
    if fn == "look_again":
        return a.get("object_id", "")
    acts = a.get("actions") or []
    s = "; ".join(f"{x.get('obj')}->{x.get('dest')}@{x.get('transit_height')}"
                  + (f" via {x.get('via')}" if x.get("via") else "") for x in acts)
    return s + (f", n={a.get('n')}" if fn == "stress_test" else "")


def _short_res(fn, r):
    if "error" in r:
        return "error: " + r["error"]
    if fn == "simulate":
        if r["success"]:
            return "SUCCESS"
        why = (["aborted"] if r.get("execution_error") or r.get("execution_aborted") else []) \
            + [g.split(":")[0] for g in (r.get("failed_goals") or [])[:2]]
        return "FAILED" + (f" ({'; '.join(why)})" if why else "")
    if fn == "stress_test":
        return f"{r['successes']}/{r['n']}"
    if fn == "look_again":
        return f"{r['before']} -> {r['after']}"
    if fn == "submit":
        return "ACCEPTED" if r["accepted"] else "REJECTED"
    return ""


def story(trace) -> list[str]:
    """One line per tool call, for the UI: 'simulate -> FAILED (IK ...)'."""
    out = []
    for t in trace:
        for c in t["tool_calls"]:
            r = c["result"]
            line = f"{c['name']}({_short_args(c['name'], c['args'])}) -> {_short_res(c['name'], r)}"
            if c["name"] in ("simulate", "submit"):
                phys = r.get("physics", r)
                ev = (phys.get("execution_error") or {}).get("error") or next(iter(phys.get("evidence") or []), "")
                if ev and not phys.get("success"):
                    line += f" | {ev[:140]}"
            out.append(line)
    return out


def run_loop(scene_path: str, instruction: str, **kw) -> dict:
    return AgentLoop(load_scene(scene_path), instruction, **kw).run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("instruction")
    ap.add_argument("--feedback", default="full_log", choices=FEEDBACKS)
    ap.add_argument("--tools", default=",".join(ALL_TOOLS))
    ap.add_argument("--max-calls", type=int, default=MAX_TOOL_CALLS)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if os.environ.get("REHEARSAL_OFFLINE"):
        raise SystemExit("the agent loop needs Muse Spark (unset REHEARSAL_OFFLINE)")
    res = run_loop(args.scene, args.instruction, feedback=args.feedback, tools=args.tools.split(","),
                   max_calls=args.max_calls, trial=args.trial)
    tr = res["agent_trace"]
    print("\n".join(tr["story"]))
    print(f"stopped: {tr['stopped']}, verified: {res['final'] and res['final']['verified']}, "
          f"${tr['cost_usd']:.4f}, {tr['wall_time_s']} s")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
