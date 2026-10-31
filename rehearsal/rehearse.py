"""End-to-end rehearsal: goals -> plan -> simulate -> verify -> diagnose/revise -> stress test -> report.

    python -m rehearsal.rehearse scenes/desk_01.json "把杯子放进碗里，然后把书摞整齐" --stress 40
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .agents import make_agent, scene_summary
from .executor import reachability_report, run_plan
from .export3d import export_scene3d
from .predicates import with_implicit_goals
from .scene import build_model, load_scene

MAX_ATTEMPTS = 3
SIGMA = 0.012          # perception error (m, per axis) from a single photo
SIGMA_REFINED = 0.004  # after looking at one object again


def _variant(args):
    scene, plan, goals, seed, sigmas, trails = args
    # Same seed -> same random draws for every sigma setting (common random numbers), so two
    # stress tests that differ only in one object's sigma are a paired comparison.
    rng = np.random.default_rng(seed)
    n = len(scene["objects"])
    var = dict(friction_scale=float(rng.uniform(0.6, 1.4)), mass_scale=float(rng.uniform(0.7, 1.3)),
               perception_error=rng.normal(0, 1, (n, 2)) * np.asarray(sigmas)[:, None])
    r = run_plan(scene, plan, goals, record="objects" if trails else None, **var)
    out = {"seed": seed, "success": r["success"],
            "friction_scale": var["friction_scale"], "mass_scale": var["mass_scale"],
            "max_perception_error_cm": float(np.abs(var["perception_error"]).max() * 100),
            "perception_error_cm": {o["id"]: float(np.linalg.norm(e) * 100)
                                    for o, e in zip(scene["objects"], var["perception_error"])},
            "failed": [f"{g['pred']}({', '.join(g['args'])}): {g['why']}" for g in r["goals"] if not g["ok"]]
                      + ([r["execution_error"]["error"]] if r["execution_error"] else [])}
    if trails:
        out["trails"] = _object_trails(scene, r["trajectory"])
    return out


def _object_trails(scene, traj) -> dict:
    """Per-object [x, y, z] samples, keeping only objects that moved more than 1 cm."""
    arr = np.asarray(traj)
    trails = {}
    for i, o in enumerate(scene["objects"]):
        xyz = arr[:, 1 + 3 * i: 4 + 3 * i]
        if np.linalg.norm(xyz - xyz[0], axis=1).max() > 0.01:
            trails[o["id"]] = xyz.round(4).tolist()
    return trails


def stress_test(scene, plan, goals, n: int, sigmas=None, pool=None, trails=False) -> dict:
    sigmas = [SIGMA] * len(scene["objects"]) if sigmas is None else list(sigmas)
    jobs = [(scene, plan, goals, s, sigmas, trails) for s in range(n)]
    if pool is None:
        with ProcessPoolExecutor() as p:
            runs = list(p.map(_variant, jobs))
    else:
        runs = list(pool.map(_variant, jobs))
    fails = [r for r in runs if not r["success"]]
    return {"n": n, "successes": n - len(fails), "runs": runs,
            "sigma_cm": {o["id"]: s * 100 for o, s in zip(scene["objects"], sigmas)}}


def active_perception(scene, plan, goals, n: int, baseline: dict, rounds: int, log=print) -> dict:
    """Value of information: which object, if observed again, raises the stress-test success most?

    For each candidate, rerun the stress test with only that object's perception error reduced
    (same seeds), pick the best, keep its reduced error, and repeat.
    """
    ids = [o["id"] for o in scene["objects"]]
    sigmas = [SIGMA] * len(ids)
    current = baseline
    steps = []
    with ProcessPoolExecutor() as pool:
        for _ in range(rounds):
            if current["successes"] == n:
                break
            options = {}
            for i, oid in enumerate(ids):
                if sigmas[i] <= SIGMA_REFINED:
                    continue
                trial = sigmas.copy()
                trial[i] = SIGMA_REFINED
                options[oid] = stress_test(scene, plan, goals, n, trial, pool)
            best = max(options, key=lambda k: options[k]["successes"])
            gain = options[best]["successes"] - current["successes"]
            log(f"look again: " + ", ".join(f"{k} {v['successes']}/{n}" for k, v in options.items())
                + f" -> {best} (+{gain})")
            if gain <= 0:
                break
            sigmas[ids.index(best)] = SIGMA_REFINED
            steps.append({"look_at": best, "before": current["successes"],
                          "after": options[best]["successes"],
                          "value_by_object": {k: v["successes"] for k, v in options.items()}})
            current = options[best]
    return {"sigma_cm": SIGMA * 100, "refined_sigma_cm": SIGMA_REFINED * 100, "n": n,
            "steps": steps, "final": current, "final_sigmas": sigmas}


def save_gif(frames, path: Path):
    if frames:
        imageio.mimsave(path, frames[::2], duration=0.1, loop=0)


def rehearse(scene_path: str, instruction: str, n_stress: int = 40, out_root: str = "out",
             log=print, look_rounds: int = 2) -> dict:
    scene = load_scene(scene_path)
    out = Path(out_root) / scene["name"]
    out.mkdir(parents=True, exist_ok=True)
    for old in [*out.glob("attempt_*.gif"), *out.glob("attempt_*.traj.json"), *out.glob("stress_trails*.json")]:
        old.unlink()
    export_scene3d(build_model(scene), out)
    agent = make_agent()
    log(f"agent: {agent.name}")

    t0 = time.time()
    reach = reachability_report(scene)
    log(f"reachable: {reach}")
    summary = scene_summary(scene, reach)

    goals = agent.compile_goals(instruction, summary)
    goals = with_implicit_goals(goals, [o["id"] for o in scene["objects"]])
    log("goals: " + json.dumps(goals, ensure_ascii=False))

    history = []
    plan = agent.plan(instruction, summary, goals)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        r = run_plan(scene, plan, goals, render=True, record="full")
        save_gif(r.pop("frames"), out / f"attempt_{attempt}.gif")
        (out / f"attempt_{attempt}.traj.json").write_text(json.dumps(r.pop("trajectory")))
        history.append({"attempt": attempt, "plan": plan, **r})
        log(f"\n--- attempt {attempt}: {'SUCCESS' if r['success'] else 'FAILED'}")
        if plan.get("diagnosis"):
            log("diagnosis: " + plan["diagnosis"])
        for a in plan["actions"]:
            log(f"  transport {a['obj']} {a['relation']} {a['dest']} @ {a['transit_height']} m")
        for g in r["goals"]:
            log(f"  [{'x' if g['ok'] else ' '}] {g['pred']}({', '.join(g['args'])}) — {g['why']}")
        if r["success"]:
            break
        plan = agent.revise(instruction, summary, goals, history)

    final = history[-1]
    stress = stress_test(scene, final["plan"], goals, n_stress, trails=True) if final["success"] and n_stress else None
    if stress:
        (out / "stress_trails.json").write_text(json.dumps(
            [{"seed": r["seed"], "success": r["success"], "trails": r.pop("trails")} for r in stress["runs"]]))
        log(f"\nstress test: {stress['successes']}/{stress['n']} succeeded under randomized "
            "friction, mass and ±1.2 cm perception error")
    looks = None
    if stress and look_rounds:
        log("\nactive perception: which object is worth a second look?")
        looks = active_perception(scene, final["plan"], goals, n_stress, stress, look_rounds, log)
        for st in looks["steps"]:
            log(f"  look at {st['look_at']}: {st['before']}/{n_stress} -> {st['after']}/{n_stress}")
        if looks["steps"]:
            after = stress_test(scene, final["plan"], goals, n_stress, looks.pop("final_sigmas"), trails=True)
            (out / "stress_trails_after.json").write_text(json.dumps(
                [{"seed": r["seed"], "success": r["success"], "trails": r.pop("trails")} for r in after["runs"]]))
        looks.pop("final_sigmas", None)

    report = {"scene": scene["name"], "scene_spec": scene, "instruction": instruction,
              "agent": agent.name, "reachability": reach, "goals": goals, "attempts": history,
              "stress_test": stress, "active_perception": looks, "wall_time_s": round(time.time() - t0, 1)}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    (out / "report.md").write_text(render_report(report), encoding="utf-8")
    log(f"\nreport: {out / 'report.md'}  ({report['wall_time_s']} s)")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("instruction")
    ap.add_argument("--stress", type=int, default=40)
    ap.add_argument("--out", default="out")
    ap.add_argument("--look", type=int, default=2, help="active-perception rounds")
    args = ap.parse_args()
    rehearse(args.scene, args.instruction, args.stress, args.out, look_rounds=args.look)


def render_report(rep: dict) -> str:
    L = [f"# Rehearsal report — {rep['scene']}", "",
         f"**Instruction:** {rep['instruction']}  ", f"**Agent:** {rep['agent']}", "",
         "## Verified by simulation", "", "### Reachability (IK, top-down suction)", ""]
    L += [f"- {'✅' if ok else '❌'} `{oid}`" + ("" if ok else " — outside the arm's workspace")
          for oid, ok in rep["reachability"].items()]
    L += ["", "### Attempts", ""]
    for a in rep["attempts"]:
        L.append(f"**Attempt {a['attempt']}: {'success' if a['success'] else 'failed'}**")
        if a["plan"].get("diagnosis"):
            L.append(f"\n> Diagnosis: {a['plan']['diagnosis']}\n")
        L.append("")
        L += [f"- {'✅' if g['ok'] else '❌'} `{g['pred']}({', '.join(g['args'])})` — {g['why']}" for g in a["goals"]]
        bumps = [e for e in a["events"] if e["kind"] == "contact" and e.get("while_holding")
                 and not {"table", "floor"} & set(e["between"])]
        if bumps:
            L.append("- Unexpected contacts: " + "; ".join(
                f"t={e['t']}s {' ↔ '.join(e['between'])}" for e in bumps))
        L.append(f"\n![attempt {a['attempt']}](attempt_{a['attempt']}.gif)\n")
    st = rep["stress_test"]
    if st:
        L += ["## Estimated under uncertainty", "",
              f"Final plan re-run {st['n']} times with friction ×0.6–1.4, mass ×0.7–1.3 and "
              f"σ=1.2 cm perception error per object: **{st['successes']}/{st['n']} succeeded**.", ""]
        fails = [r for r in st["runs"] if not r["success"]]
        if fails:
            L.append("Failures:")
            L += [f"- seed {r['seed']}: max perception error {r['max_perception_error_cm']:.1f} cm, "
                  f"friction ×{r['friction_scale']:.2f} — {'; '.join(r['failed'])}" for r in fails]
            L.append("")
    ap_ = rep.get("active_perception")
    if ap_ and ap_["steps"]:
        L += ["## Where to look again", "",
              f"Re-observing one object is modelled as shrinking its perception error from "
              f"σ={ap_['sigma_cm']:.1f} cm to σ={ap_['refined_sigma_cm']:.1f} cm (same seeds).", ""]
        for i, st in enumerate(ap_["steps"], 1):
            vals = ", ".join(f"{k} {v}/{ap_['n']}" for k, v in
                             sorted(st["value_by_object"].items(), key=lambda kv: -kv[1]))
            L.append(f"{i}. Look again at **{st['look_at']}**: {st['before']}/{ap_['n']} → "
                     f"{st['after']}/{ap_['n']}  (candidates: {vals})")
        L.append("")
    L += ["## Assumed (not verified)", "",
          "- Grasping always succeeds (suction weld). Real grasp physics is out of scope.",
          "- Object geometry is approximated by primitives; perception is taken as given.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
