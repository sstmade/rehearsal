"""Hybrid planner experiment: LLM-only vs LLM + log revise vs geometric search vs hybrid.

    .venv/bin/python eval/hybrid_eval.py [--repeats-extra 10] [--seeds 20] [--budget 0.40]

Arms (all scored on reference goals + implicit `undisturbed`; the planners never see them except (c)):
  (a) llm_first   Muse Spark's first plan, simulated once.
  (b) llm_revise  (a) plus up to 2 full-log revisions (product loop).
  (c) search      search_planner from the reference goals, no model at all (deterministic, n = 1).
  (c+) search_gate  (c) with the robustness gate (10 selection seeds 0-9, threshold 9/10, <= 6 solutions).
  (d) hybrid      Muse's compiled goals + Muse's plan (assignment, order, first geometry guess);
                  search_planner fills in the geometry; scored on the reference goals.
  (d+) hybrid_gate  (d) with the same robustness gate.

Hard scenes (desk_02, kitchen_01, lamp_01, storage_01, windowsill_01): Muse's goals and plans are the
cached outputs of Evaluation v3 (eval/results/eval_20261031_173045_round3_main.json, 20 repeats each);
(a) and (b) are those runs' recorded outcomes (not re-run); their $ are read from the ledger by caller tag.
Extra scenes (desk_01, real_ycbv59, aigen_*): Muse is called live (Contributor mode, response cache,
ledger, hard cap = spend at start + --budget), --repeats-extra repeats, same prompts as the product.

Robustness: every distinct successful final plan is stress-tested on held-out seeds 1000..1000+S-1 with
the product noise model (rehearse._variant, per-object sigma from the scene); the same seeds for every
arm (common random numbers), so arm differences are paired.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("REHEARSAL_BUDGET_DIR", str((REPO / "budget").resolve()))
os.environ["PYTHONPATH"] = os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")])

import numpy as np  # noqa: E402

from rehearsal import agents as A  # noqa: E402
from rehearsal import meta_client as M  # noqa: E402
from rehearsal.executor import reachability_report, run_plan  # noqa: E402
from rehearsal.predicates import with_implicit_goals  # noqa: E402
from rehearsal.rehearse import _variant, object_sigmas  # noqa: E402
from rehearsal.search_planner import (SerialPool, assignments_from_goals, merge_assignments,  # noqa: E402
                                      search)

RESULTS = REPO / "eval" / "results"
V3 = RESULTS / "eval_20261031_173045_round3_main.json"
HARD = ["desk_02", "kitchen_01", "lamp_01", "storage_01", "windowsill_01"]
EXTRA = ["desk_01", "real_ycbv59", "aigen_desk_lamp_mug", "aigen_desk_walnut", "aigen_kitchen"]
GATE = dict(robust_n=10, robust_threshold=0.9, max_solutions=6)
HELDOUT0 = 1000


def load_scene(name):
    return json.loads((REPO / "scenes" / f"{name}.json").read_text())


def ref_goals(name, scene):
    ref = (scene.get("reference") or {}).get("goals") or json.loads((REPO / "eval" / "hybrid_tasks.json").read_text())[name]
    return with_implicit_goals([dict(g) for g in ref], [o["id"] for o in scene["objects"]])


def pkey(plan):
    acts = [{k: a[k] for k in ("obj", "dest", "relation", "transit_height", "via")} for a in plan["actions"]]
    return hashlib.sha1(json.dumps(acts, sort_keys=True).encode()).hexdigest()[:12]


def sim_ok(name, plan, goals):
    r = run_plan(load_scene(name), plan, goals)
    return bool(r["success"]), r


def strip_plan(plan):
    return {"diagnosis": plan.get("diagnosis", ""),
            "actions": [{k: a.get(k, [] if k == "via" else None) for k in ("obj", "dest", "relation", "transit_height", "via")}
                        for a in plan["actions"]]}


# ------------------------------------------------------------------ ledger costs for cached v3 runs
def v3_costs():
    """{(scene, repeat): {"first": $ goals+plan, "revise": $ full-log revisions}} from the shared ledger."""
    out = defaultdict(lambda: {"first": 0.0, "revise": 0.0})
    for line in open(Path(os.environ["REHEARSAL_BUDGET_DIR"]) / "ledger.jsonl"):
        e = json.loads(line)
        c = str(e.get("caller", ""))
        if not c.startswith("eval.") or ".refg" in c or not ("2026-10-31 16:4" <= e["ts"] <= "2026-10-31 17:32"):
            continue
        parts = c.split(".")      # eval.<name>.<scene>.<ihash>.r<k>.<suffix>
        if len(parts) < 6 or parts[2] not in HARD:
            continue
        key = (parts[2], int(parts[4][1:]))
        if parts[5] == "plan":
            out[key]["first"] += e.get("cost", 0.0)
        elif parts[5] == "full_log":
            out[key]["revise"] += e.get("cost", 0.0)
    return out


def hard_tasks():
    d = json.loads(V3.read_text())
    costs = v3_costs()
    groups = defaultdict(dict)
    for c in d["cells"]:
        groups[(c["scene"], c["repeat"])][c["condition"]] = c
    tasks = []
    for (scene, rep), g in sorted(groups.items()):
        fl = g["full_log"]
        a0 = fl["attempts"][0]
        tasks.append({"scene": scene, "repeat": rep, "source": "v3_cache", "goals": fl["goals"],
                      "plan0": strip_plan(a0["plan"]), "first_ok": bool(fl["first_try_ref_success"]),
                      "final_plan": strip_plan(fl["attempts"][-1]["plan"]),
                      "revise_ok": bool(fl["ref_success_within_2_revisions"]),
                      "revise_sims": fl["attempts_used"], "llm_wall_s": fl["agent_wall_s"],
                      "cost_first": round(costs[(scene, rep)]["first"], 6),
                      "cost_revise": round(costs[(scene, rep)]["first"] + costs[(scene, rep)]["revise"], 6),
                      "first_only_wall_s": fl["agent_wall_s"] if fl["first_try_ref_success"] else None})
    return tasks


# ------------------------------------------------------------------ live Muse for the extra scenes
class TaggedMuse(A.MuseAgent):
    def __init__(self, trial, tag):
        super().__init__(data="public")
        self.trial, self.tag = trial, tag

    def _json(self, user, schema, name):
        if self.trial:
            user += f"\n\n[eval trial {self.trial}]"
        return M.chat_json([{"role": "system", "content": A.SYSTEM}, {"role": "user", "content": user}],
                           schema, name, caller=f"hybrid.{name.lower()}.{self.tag}", data="public",
                           model=self.model, effort=self.effort)


def ledger_cost(tag_prefix, offset):
    tot = 0.0
    with open(Path(os.environ["REHEARSAL_BUDGET_DIR"]) / "ledger.jsonl") as f:
        f.seek(offset)
        for line in f:
            e = json.loads(line)
            if str(e.get("caller", "")).endswith(tag_prefix):
                tot += e.get("cost", 0.0)
    return tot


def extra_task(args):
    scene_name, rep, reach = args
    scene = load_scene(scene_name)
    instr = scene["demo_instruction"]
    ref = ref_goals(scene_name, scene)
    tag = f"{scene_name}.r{rep}"
    agent = TaggedMuse(rep, tag)
    summary = A.scene_summary(scene, reach)
    off = (Path(os.environ["REHEARSAL_BUDGET_DIR"]) / "ledger.jsonl").stat().st_size
    t = time.time()
    try:
        raw = agent.compile_goals(instr, summary)
        goals = with_implicit_goals(raw, [o["id"] for o in scene["objects"]])
        plan0 = strip_plan(agent.plan(instr, summary, goals))
    except Exception as e:  # noqa: BLE001
        return {"scene": scene_name, "repeat": rep, "source": "live", "error": f"{type(e).__name__}: {e}"}
    wall_first = time.time() - t
    cost_first = ledger_cost(tag, off)
    ok_agent, r = sim_ok(scene_name, plan0, goals)
    first_ok, _ = sim_ok(scene_name, plan0, ref)
    history = [{"attempt": 1, "plan": plan0, **{k: r[k] for k in ("success", "goals", "execution_error", "events", "evidence")}}]
    plan, ok_agent_last, rev_wall = plan0, ok_agent, 0.0
    try:
        while not ok_agent_last and len(history) < 3:
            t2 = time.time()
            plan = strip_plan(agent.revise(instr, summary, goals, history))
            rev_wall += time.time() - t2
            ok_agent_last, r = sim_ok(scene_name, plan, goals)
            history.append({"attempt": len(history) + 1, "plan": plan,
                            **{k: r[k] for k in ("success", "goals", "execution_error", "events", "evidence")}})
    except Exception as e:  # noqa: BLE001
        print(f"revise error {tag}: {e}", flush=True)
    revise_ok, _ = sim_ok(scene_name, plan, ref)
    return {"scene": scene_name, "repeat": rep, "source": "live", "instruction": instr, "goals": goals,
            "plan0": plan0, "first_ok": first_ok, "final_plan": plan, "revise_ok": revise_ok,
            "revise_sims": len(history), "llm_wall_s": round(wall_first + rev_wall, 1),
            "first_only_wall_s": round(wall_first, 1), "cost_first": round(cost_first, 6),
            "cost_revise": round(ledger_cost(tag, off), 6)}


# ------------------------------------------------------------------ search jobs (run in pool workers)
def search_job(job):
    name, arm, goals, assigns, hint, gate, key = job
    scene = load_scene(name)
    t = time.time()
    res = search(scene, goals, assigns, hint_plan=hint, pool=SerialPool, **(GATE if gate else {}))
    out = {"key": key, "arm": arm, "scene": name, "found": res["success"], "reason": res["reason"],
           "stats": res["stats"], "selection_robustness": res["robustness"], "plan": res["plan"],
           "wall_s": round(time.time() - t, 2)}
    if res["plan"]:
        out["ref_ok"], r = sim_ok(name, res["plan"], ref_goals(name, scene))
        out["ref_failed"] = [f"{g['pred']}({', '.join(g['args'])}): {g['why']}" for g in r["goals"] if not g["ok"]]
    else:
        out["ref_ok"] = False
    return out


def stress_job(job):
    name, plan, seed = job
    scene = load_scene(name)
    sig = object_sigmas(scene)
    r = _variant((scene, plan, ref_goals(name, scene), seed, sig, False))
    return bool(r["success"])


def feasible_job(job):
    """Is 'put a on b' physically achievable at all (small search budget)?"""
    name, a, b = job
    scene = load_scene(name)
    goals = with_implicit_goals([{"pred": "on", "args": [a, b]}], [o["id"] for o in scene["objects"]])
    res = search(scene, goals, [{"obj": a, "dest": b, "relation": "on"}], sim_budget=8, time_budget_s=60)
    return {"scene": name, "obj": a, "dest": b, "feasible": res["success"], "sims": res["stats"]["sims"]}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats-extra", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=float, default=0.40)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--work", default=str(RESULTS / "hybrid_work.json"), help="stage cache (resume)")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-feasible", action="store_true")
    args = ap.parse_args()
    work = json.loads(Path(args.work).read_text()) if Path(args.work).exists() else {}

    def save():
        Path(args.work).write_text(json.dumps(work, default=float))

    t0 = time.time()
    b0 = M.status()
    os.environ["REHEARSAL_BUDGET_CAP"] = str(round(b0["spent"] + b0["reserved"] + args.budget, 4))
    print(f"budget at start {b0}; cap for this run {os.environ['REHEARSAL_BUDGET_CAP']}", flush=True)

    # 1. tasks
    if "tasks" not in work:
        tasks = hard_tasks()
        if os.environ.get("MODEL_API_KEY") or (REPO / ".env").exists():
            from rehearsal import load_env
            load_env()
        reach = {n: reachability_report(load_scene(n)) for n in EXTRA}
        jobs = [(n, r, reach[n]) for n in EXTRA for r in range(args.repeats_extra)]
        with ThreadPoolExecutor(6) as ex:
            for t in ex.map(extra_task, jobs):
                print(f"live {t['scene']} r{t['repeat']}: first={t.get('first_ok')} revise={t.get('revise_ok')} "
                      f"${t.get('cost_revise')} {t.get('error', '')}", flush=True)
                tasks.append(t)
        work["tasks"] = tasks
        save()
    tasks = [t for t in work["tasks"] if "error" not in t]
    print(f"{len(tasks)} tasks ({len(work['tasks']) - len(tasks)} live errors)", flush=True)

    # 2. searches
    scenes = HARD + EXTRA
    jobs = []
    for n in scenes:
        g = ref_goals(n, load_scene(n))
        jobs.append((n, "search", g, assignments_from_goals(g), None, False, f"search|{n}"))
        if not args.no_gate:
            jobs.append((n, "search_gate", g, assignments_from_goals(g), None, True, f"search_gate|{n}"))
    for t in tasks:
        assigns = merge_assignments(t["goals"], t["plan0"])
        k = f"{t['scene']}|{t['repeat']}"
        jobs.append((t["scene"], "hybrid", t["goals"], assigns, t["plan0"], False, f"hybrid|{k}"))
        if not args.no_gate:
            jobs.append((t["scene"], "hybrid_gate", t["goals"], assigns, t["plan0"], True, f"hybrid_gate|{k}"))
    done = work.setdefault("search", {})
    todo = [j for j in jobs if j[-1] not in done]
    print(f"search jobs: {len(todo)} to run ({len(done)} cached)", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(search_job, todo)):
            done[r["key"]] = r
            if i % 10 == 0 or not r["ref_ok"]:
                print(f"  [{i + 1}/{len(todo)}] {r['key']}: found={r['found']} ref_ok={r['ref_ok']} "
                      f"sims={r['stats']['sims']} stress={r['stats']['stress_sims']} {r['wall_s']}s "
                      f"{r.get('ref_failed') or r['reason'] or ''}", flush=True)
                save()
    save()

    # 3. held-out stress tests of every distinct successful final plan
    plans = {}
    for t in tasks:
        if t["first_ok"]:
            plans[(t["scene"], pkey(t["plan0"]))] = t["plan0"]
        if t["revise_ok"]:
            plans[(t["scene"], pkey(t["final_plan"]))] = t["final_plan"]
    for r in done.values():
        if r["ref_ok"]:
            plans[(r["scene"], pkey(r["plan"]))] = strip_plan(r["plan"])
    rob = work.setdefault("robust", {})
    seeds = list(range(HELDOUT0, HELDOUT0 + args.seeds))
    need = [(k, p) for k, p in plans.items() if f"{k[0]}|{k[1]}" not in rob]
    print(f"stress: {len(need)} plans x {len(seeds)} seeds ({len(plans) - len(need)} cached)", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for i in range(0, len(need), 12):
            chunk = need[i:i + 12]
            sj = [(k[0], p, s) for k, p in chunk for s in seeds]
            res = list(ex.map(stress_job, sj, chunksize=4))
            for j, (k, _) in enumerate(chunk):
                rob[f"{k[0]}|{k[1]}"] = res[j * len(seeds):(j + 1) * len(seeds)]
            save()
            print(f"  stress {min(i + 12, len(need))}/{len(need)}", flush=True)

    # 4. feasibility of every 'put a on b' (what physics alone cannot decide)
    if not args.no_feasible and "feasible" not in work:
        fj = []
        for n in HARD:
            ids = [o["id"] for o in load_scene(n)["objects"]]
            fj += [(n, a, b) for a in ids for b in ids if a != b]
        with ProcessPoolExecutor(args.workers) as ex:
            work["feasible"] = list(ex.map(feasible_job, fj))
        save()

    out = summarize(work, tasks, seeds)
    b1 = M.status()
    out.update(created=time.strftime("%Y-%m-%d %H:%M:%S"), wall_s=round(time.time() - t0, 1), args=vars(args),
               budget_before=b0, budget_after=b1, new_spend_this_run=round(b1["spent"] - b0["spent"], 4))
    path = RESULTS / f"hybrid_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print_table(out)
    print(f"\nwrote {path}")


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def summarize(work, tasks, seeds):
    rob, srch = work["robust"], work["search"]

    def robv(scene, plan):
        v = rob.get(f"{scene}|{pkey(plan)}")
        return None if v is None else v

    per = defaultdict(lambda: defaultdict(list))   # scene -> arm -> list of task records
    for t in tasks:
        s = t["scene"]
        per[s]["llm_first"].append({"ok": t["first_ok"], "sims": 1, "cost": t["cost_first"],
                                    "wall": t.get("first_only_wall_s"), "rob": robv(s, t["plan0"]) if t["first_ok"] else None})
        per[s]["llm_revise"].append({"ok": t["revise_ok"], "sims": t["revise_sims"], "cost": t["cost_revise"],
                                     "wall": t["llm_wall_s"], "rob": robv(s, t["final_plan"]) if t["revise_ok"] else None})
        for arm in ("hybrid", "hybrid_gate"):
            r = srch.get(f"{arm}|{s}|{t['repeat']}")
            if r is None:
                continue
            per[s][arm].append({"ok": r["ref_ok"], "found": r["found"], "sims": r["stats"]["sims"],
                                "stress_sims": r["stats"]["stress_sims"], "search_wall": r["wall_s"],
                                "cost": t["cost_first"], "rob": robv(s, r["plan"]) if r["ref_ok"] else None,
                                "ref_failed": r.get("ref_failed"), "reason": r["reason"],
                                "goal_ok_but_ref_fail": r["found"] and not r["ref_ok"]})
    for key, r in srch.items():
        arm, s = key.split("|")[:2]
        if arm in ("search", "search_gate"):
            per[s][arm].append({"ok": r["ref_ok"], "found": r["found"], "sims": r["stats"]["sims"],
                                "stress_sims": r["stats"]["stress_sims"], "search_wall": r["wall_s"], "cost": 0.0,
                                "rob": robv(s, r["plan"]) if r["ref_ok"] else None, "plan": r["plan"]})

    def agg(recs):
        n = len(recs)
        k = sum(r["ok"] for r in recs)
        robs = [x for r in recs if r["rob"] for x in r["rob"]]
        cost = sum(r["cost"] or 0 for r in recs)
        d = {"n": n, "success": k, "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n),
             "sims_mean": round(float(np.mean([r["sims"] for r in recs])), 2) if n else None,
             "cost_total": round(cost, 5), "cost_per_solved": round(cost / k, 5) if k else None,
             "robust_heldout": {"k": int(sum(robs)), "n": len(robs),
                                "rate": round(sum(robs) / len(robs), 3) if robs else None}}
        if recs and "stress_sims" in recs[0]:
            d["stress_sims_mean"] = round(float(np.mean([r["stress_sims"] for r in recs])), 1)
            d["search_wall_s_mean"] = round(float(np.mean([r["search_wall"] for r in recs])), 2)
            d["search_wall_s_max"] = round(float(np.max([r["search_wall"] for r in recs])), 2)
        walls = [r["wall"] for r in recs if r.get("wall") is not None]
        if walls:
            d["llm_wall_s_median"] = round(float(np.median(walls)), 1)
        if recs and "goal_ok_but_ref_fail" in recs[0]:
            d["found_but_ref_fail"] = sum(r["goal_ok_but_ref_fail"] for r in recs)
            d["not_found"] = sum(not r["found"] for r in recs)
            d["ref_fail_examples"] = [r["ref_failed"] for r in recs if r["goal_ok_but_ref_fail"]][:4]
        return d

    arms = ["llm_first", "llm_revise", "search", "search_gate", "hybrid", "hybrid_gate"]
    table = {s: {a: agg(per[s][a]) for a in arms if per[s].get(a)} for s in per}
    pooled = {}
    for group, names in (("hard5", HARD), ("extra5", EXTRA), ("all10", HARD + EXTRA)):
        pooled[group] = {a: agg([r for s in names for r in per[s].get(a, [])]) for a in arms}

    # Paired robustness (same held-out seeds): hybrid_gate vs llm_revise, per task where both solved.
    paired = defaultdict(lambda: [0, 0, 0])   # seed-level discordant counts
    for t in tasks:
        s = t["scene"]
        r = srch.get(f"hybrid_gate|{s}|{t['repeat']}")
        if not (r and r["ref_ok"] and t["revise_ok"]):
            continue
        a, b = robv(s, r["plan"]), robv(s, t["final_plan"])
        if a is None or b is None:
            continue
        g = "hard5" if s in HARD else "extra5"
        for x, y in zip(a, b):
            paired[g][0] += x and not y
            paired[g][1] += y and not x
            paired[g][2] += 1
    from math import comb

    def mcnemar(b, c):
        n = b + c
        if n == 0:
            return 1.0
        k = min(b, c)
        return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

    paired_out = {g: {"hybrid_gate_only": v[0], "llm_revise_only": v[1], "seed_pairs": v[2],
                      "p_exact_mcnemar": round(mcnemar(v[0], v[1]), 5)} for g, v in paired.items()}

    feas = work.get("feasible")
    feas_out = None
    if feas:
        feas_out = {}
        for n in HARD:
            rows = [f for f in feas if f["scene"] == n]
            feas_out[n] = {"pairs": len(rows), "feasible": sum(f["feasible"] for f in rows),
                           "examples": [f"{f['obj']} on {f['dest']}" for f in rows if f["feasible"]][:8]}

    goal_match = {}
    for s in per:
        ts = [t for t in tasks if t["scene"] == s]
        ref = ref_goals(s, load_scene(s))
        ref_set = {json.dumps(g, sort_keys=True) for g in ref if not g.get("implicit")}
        goal_match[s] = {"n": len(ts), "exact": sum({json.dumps({k: g[k] for k in ("pred", "args")}, sort_keys=True)
                                                    for g in t["goals"] if not g.get("implicit")} == ref_set for t in ts)}
    return {"kind": "hybrid_eval", "heldout_seeds": [seeds[0], seeds[-1]], "gate": GATE, "per_scene": table,
            "pooled": pooled, "paired_robustness_hybrid_gate_vs_llm_revise": paired_out,
            "feasible_on_pairs": feas_out, "muse_goals_exact_match": goal_match,
            "search_plans": {s: [r.get("plan") for r in per[s].get("search", [])] for s in per}}


def print_table(out):
    arms = ["llm_first", "llm_revise", "search", "search_gate", "hybrid", "hybrid_gate"]
    print("\nscene | " + " | ".join(arms))
    for s, row in list(out["per_scene"].items()) + [(f"[{k}]", v) for k, v in out["pooled"].items()]:
        cells = []
        for a in arms:
            d = row.get(a)
            if not d or not d["n"]:
                cells.append("-")
                continue
            rb = d["robust_heldout"]
            cells.append(f"{d['success']}/{d['n']} sims {d['sims_mean']}"
                         + (f"+{d['stress_sims_mean']}st" if d.get("stress_sims_mean") else "")
                         + (f" rob {rb['rate']:.0%}" if rb["rate"] is not None else ""))
        print(f"{s} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
