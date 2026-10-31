"""Log-feedback ablation for the self-correction claim (real Muse Spark, Contributor mode on public scenes).

    .venv/bin/python experiments/agent/ablation.py --scenes windowsill_01,kitchen_01 --repeats 5 --budget 1.0

Two experiments, both scored on the scene's hand-verified `reference.goals` (+ implicit undisturbed):

A. pipeline  — the product's fixed loop (MuseAgent.plan -> simulate -> MuseAgent.revise, up to 3 attempts).
   Paired design: per (scene, repeat) the first plan is written and simulated ONCE; only if it fails do
   the revisions run under each condition from that same failed attempt:
     full_log      compact history (failed goals with numbers, execution_error, evidence sentences) — product default
     full_log_raw  the raw history (every contact event) instead of the compact one
     outcome_only  plan, success flag, failed predicate names, "aborted" flag
     none          plan and success=false only
B. loop      — the agentic loop (rehearsal/agent_loop.py) with tools simulate+submit only, feedback =
   full_log / outcome_only / none (what the simulate/submit tool returns). Max 6 tool calls.

Writes experiments/agent/ablation_<ts>.json (+ prints a summary table). Every call is tagged
caller="agent.abl.*" so its cost is read back from the shared ledger.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rehearsal import agents as A  # noqa: E402
from rehearsal import meta_client as M  # noqa: E402
from rehearsal.agent_loop import AgentLoop  # noqa: E402
from rehearsal.executor import reachability_report, run_plan  # noqa: E402
from rehearsal.predicates import with_implicit_goals  # noqa: E402
from rehearsal.scene import load_scene  # noqa: E402

PIPE_CONDS = ("full_log", "full_log_raw", "outcome_only", "none")
LOOP_CONDS = ("full_log", "outcome_only", "none")
MAX_ATTEMPTS = 3


def spent(prefix="agent.abl") -> float:
    tot = 0.0
    for line in (M.budget_dir() / "ledger.jsonl").read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(e.get("caller", "")).startswith(prefix):
            tot += e.get("cost", 0.0)
    return tot


def cost_of(caller_prefix: str) -> float:
    return round(spent(caller_prefix), 6)


class TrialMuse(A.MuseAgent):
    def __init__(self, data, trial, tag, compact=True):
        super().__init__(data=data, compact_history=compact)
        self.trial, self.tag = trial, tag

    def _json(self, user, schema, name):
        if self.trial:
            user += f"\n\n[trial {self.trial}]"
        return M.chat_json([{"role": "system", "content": A.SYSTEM}, {"role": "user", "content": user}],
                           schema, name, caller=f"{self.tag}.{name.lower()}", data=self.data,
                           model=self.model, effort=self.effort)


def sim(scene, plan, goals):
    r = run_plan(scene, plan, goals)
    r.pop("frames", None)
    r.pop("trajectory", None)
    return r


def filt(history, cond):
    if cond in ("full_log", "full_log_raw"):
        return history
    out = []
    for h in history:
        d = {"attempt": h["attempt"], "plan": h["plan"], "success": h["success"]}
        if cond == "outcome_only":
            d["failed_goals"] = [{"pred": g["pred"], "args": g["args"]} for g in h["goals"] if not g["ok"]]
            if h.get("execution_error"):
                d["execution_aborted"] = True
        out.append(d)
    return out


def pipeline_trial(scene, instr, goals, summary, k, budget):
    name = scene["name"]
    tag = f"agent.abl.pipe.{name}.{k}"
    agent = TrialMuse("public", k, tag)
    plan = agent.plan(instr, summary, goals)
    r1 = sim(scene, plan, goals)
    first = {"attempt": 1, "plan": plan, **r1}
    rec = {"scene": name, "repeat": k, "first_success": r1["success"],
           "first_plan": plan, "first_evidence": r1.get("evidence"), "conds": {}}
    if r1["success"]:
        rec["cost"] = cost_of(tag)
        return rec
    def run_cond(cond):
        if spent() >= budget:
            return {"skipped": "budget"}
        ag = TrialMuse("public", k, f"{tag}.{cond}", compact=(cond != "full_log_raw"))
        hist = [first]
        while len(hist) < MAX_ATTEMPTS and not hist[-1]["success"]:
            p = ag.revise(instr, summary, goals, filt(hist, cond))
            r = sim(scene, p, goals)
            hist.append({"attempt": len(hist) + 1, "plan": p, **r})
        return {"success": hist[-1]["success"], "attempts": len(hist),
                "fixed_on_attempt": len(hist) if hist[-1]["success"] else None,
                "plans": [h["plan"] for h in hist[1:]],
                "evidence": [h.get("evidence") for h in hist[1:]],
                "cost": cost_of(f"{tag}.{cond}")}

    with ThreadPoolExecutor(len(PIPE_CONDS)) as ex:
        rec["conds"] = dict(zip(PIPE_CONDS, ex.map(run_cond, PIPE_CONDS)))
    rec["cost"] = cost_of(tag)
    return rec


def loop_trial(scene, instr, goals, k, cond, budget):
    if spent() >= budget:
        return {"scene": scene["name"], "repeat": k, "cond": cond, "skipped": "budget"}
    caller = f"agent.abl.loop.{scene['name']}.{cond}.{k}"
    res = AgentLoop(scene, instr, data="public", feedback=cond, tools=("simulate", "submit"),
                    max_calls=6, goals=goals, caller=caller, trial=k + 1, log=lambda *_: None).run()
    tr = res["agent_trace"]
    sims = [a for a in res["attempts"]]
    return {"scene": scene["name"], "repeat": k, "cond": cond,
            "success": bool(res["final"] and res["final"]["verified"]),
            "first_sim_success": sims[0]["success"] if sims else None,
            "n_sims": len(sims), "tool_calls": tr["tool_calls"], "stopped": tr["stopped"],
            "cost": tr["cost_usd"], "story": tr["story"], "wall_time_s": tr["wall_time_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="windowsill_01,kitchen_01")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--budget", type=float, default=1.0, help="stop when agent.abl.* spend reaches this ($)")
    ap.add_argument("--which", default="pipeline,loop")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    which = args.which.split(",")
    t0, spent0 = time.time(), spent()
    out = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "repeats": args.repeats, "pipeline": [], "loop": []}
    for name in args.scenes.split(","):
        scene = load_scene(str(ROOT / "scenes" / f"{name}.json"))
        assert scene.get("privacy") == "public", "ablation uses Contributor mode: public scenes only"
        instr = scene["demo_instruction"]
        goals = with_implicit_goals(scene["reference"]["goals"], [o["id"] for o in scene["objects"]])
        summary = A.scene_summary(scene, reachability_report(scene))
        with ThreadPoolExecutor(args.workers) as ex:
            if "pipeline" in which:
                futs = [ex.submit(pipeline_trial, scene, instr, goals, summary, k, args.budget)
                        for k in range(args.repeats)]
                out["pipeline"] += [f.result() for f in futs]
            if "loop" in which:
                futs = [ex.submit(loop_trial, scene, instr, goals, k, c, args.budget)
                        for k in range(args.repeats) for c in LOOP_CONDS]
                out["loop"] += [f.result() for f in futs]
    out["summary"] = summarize(out)
    out["cost_usd"] = round(spent() - spent0, 6)
    out["wall_time_s"] = round(time.time() - t0, 1)
    p = ROOT / "experiments" / "agent" / f"ablation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(out["summary"], indent=1, ensure_ascii=False))
    print(f"cost ${out['cost_usd']:.4f}, {out['wall_time_s']} s -> {p}")


def summarize(out):
    s = {"pipeline": {}, "loop": {}}
    for rec in out["pipeline"]:
        d = s["pipeline"].setdefault(rec["scene"], {"n": 0, "first_plan_failed": 0,
                                                    **{c: {"fixed": 0, "fixed_by_attempt2": 0, "n": 0}
                                                       for c in PIPE_CONDS}})
        d["n"] += 1
        if not rec["first_success"]:
            d["first_plan_failed"] += 1
            for c, v in rec["conds"].items():
                if "skipped" in v:
                    continue
                d[c]["n"] += 1
                d[c]["fixed"] += int(v["success"])
                d[c]["fixed_by_attempt2"] += int(v["fixed_on_attempt"] == 2)
    for rec in out["loop"]:
        if "skipped" in rec:
            continue
        d = s["loop"].setdefault(rec["scene"], {c: {"n": 0, "success": 0, "first_sim_failed": 0,
                                                    "success_after_first_fail": 0, "mean_tool_calls": 0.0,
                                                    "sims_to_success_after_first_fail": []}
                                                for c in LOOP_CONDS})[rec["cond"]]
        d["n"] += 1
        d["success"] += int(rec["success"])
        d["mean_tool_calls"] += rec["tool_calls"]
        if rec["first_sim_success"] is False:
            d["first_sim_failed"] += 1
            d["success_after_first_fail"] += int(rec["success"])
            if rec["success"]:
                d["sims_to_success_after_first_fail"].append(rec["n_sims"])
    for sc in s["loop"].values():
        for d in sc.values():
            d["mean_tool_calls"] = round(d["mean_tool_calls"] / max(d["n"], 1), 2)
    return s


if __name__ == "__main__":
    main()
