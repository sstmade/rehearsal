"""End-to-end rehearsal: goals -> plan -> simulate -> verify -> diagnose/revise -> stress test -> report.

    python -m rehearsal.rehearse scenes/desk_01.json "把杯子放进碗里，然后把书摞整齐" --stress 40

Precomputed demo results (offline, zero API cost):

    python -m rehearsal.rehearse --precompute [--out out] [--only desk_01,...] [--online]

runs every scenes/*.json that has a "demo_instruction" field and writes out/<scene>/. Offline, a
scene with a "reference" block is replayed by OfflineAgent (naive_plan, then fixed_plan); without
one, "demo_instruction_zh" is used when present (the stand-in only parses Chinese keywords). Optional
per-scene fields: "demo_stress" (default 40) and "demo_look" (active-perception rounds, default 2).
--precompute forces REHEARSAL_OFFLINE=1 (OfflineAgent) unless --online is given.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .agents import make_agent, scene_summary
from .executor import reachability_report, run_plan
from .export3d import export_scene3d
from .pipeline import worker_count
from .predicates import with_implicit_goals
from .scene import build_model, load_scene

MAX_ATTEMPTS = 3
SIGMA = 0.012          # perception error (m, per axis) from a single photo
SIGMA_REFINED = 0.004  # after looking at one object again
HELDOUT_SEED0 = 1000   # seeds 0..n-1 choose what to look at; 1000..1000+n-1 report the result


def object_sigmas(scene: dict) -> list[float]:
    """Per-object perception sigma (m): obj["perception"]["sigma_m"] when present, else SIGMA."""
    out = []
    for o in scene["objects"]:
        v = (o.get("perception") or {}).get("sigma_m")
        out.append(float(v) if isinstance(v, (int, float)) and v > 0 else SIGMA)
    return out


def _brief(st: dict) -> dict:
    return {"successes": st["successes"], "n": st["n"], "seed0": st.get("seed0", 0)}


def _process_pool() -> ProcessPoolExecutor:
    """Stress-test workers. On Linux, forkserver avoids forking a threaded web server
    (and any GL context it holds); macOS already defaults to spawn."""
    ctx = mp.get_context("forkserver") if sys.platform.startswith("linux") else None
    return ProcessPoolExecutor(max_workers=worker_count(), mp_context=ctx)


def _noop(*_a, **_k):
    pass


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


def stress_test(scene, plan, goals, n: int, sigmas=None, pool=None, trails=False, on_run=None,
                seed0: int = 0) -> dict:
    """Run seeds seed0..seed0+n-1. sigmas defaults to object_sigmas(scene).
    on_run(done, n) is called after each finished variant (progress reporting)."""
    sigmas = object_sigmas(scene) if sigmas is None else list(sigmas)
    jobs = [(scene, plan, goals, s, sigmas, trails) for s in range(seed0, seed0 + n)]
    on_run = on_run or _noop

    def collect(p):
        runs = []
        for r in p.map(_variant, jobs):
            runs.append(r)
            on_run(len(runs), n)
        return runs

    if pool is None:
        with _process_pool() as p:
            runs = collect(p)
    else:
        runs = collect(pool)
    fails = [r for r in runs if not r["success"]]
    return {"n": n, "successes": n - len(fails), "runs": runs, "seed0": seed0,
            "sigma_cm": {o["id"]: s * 100 for o, s in zip(scene["objects"], sigmas)},
            "sigma_cm_mean": round(float(np.mean(sigmas)) * 100, 4)}


def active_perception(scene, plan, goals, n: int, baseline: dict, rounds: int, log=print,
                      on_option=None, sigmas0=None) -> dict:
    """Value of information: which object, if observed again, raises the stress-test success most?

    For each candidate, rerun the stress test with only that object's perception error reduced
    (same seeds as `baseline`), pick the best, keep its reduced error, and repeat. The numbers in
    `steps` / `final` are selection-set numbers (optimistic: the choice was made on these seeds);
    rehearse() adds an unbiased "heldout" comparison on fresh seeds.
    """
    ids = [o["id"] for o in scene["objects"]]
    sigmas = object_sigmas(scene) if sigmas0 is None else list(sigmas0)
    initial = list(sigmas)
    seed0 = baseline.get("seed0", 0)
    current = baseline
    steps = []
    on_option = on_option or _noop
    with _process_pool() as pool:
        for rnd in range(rounds):
            if current["successes"] == n:
                break
            options = {}
            for i, oid in enumerate(ids):
                if sigmas[i] <= SIGMA_REFINED:
                    continue
                trial = sigmas.copy()
                trial[i] = SIGMA_REFINED
                options[oid] = stress_test(scene, plan, goals, n, trial, pool, seed0=seed0)
                on_option(rnd, len(options), len(ids))
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
    init_cm = {i: round(v * 100, 4) for i, v in zip(ids, initial)}
    # sigma_cm: per-object dict (same shape as stress_test.sigma_cm); default_sigma_cm is SIGMA.
    return {"sigma_cm": init_cm, "sigma_cm_mean": round(float(np.mean(initial)) * 100, 4),
            "default_sigma_cm": SIGMA * 100, "refined_sigma_cm": SIGMA_REFINED * 100, "n": n,
            "initial_sigma_cm": init_cm,
            "steps": steps, "final": current, "final_sigmas": sigmas,
            "selection": {"seed0": seed0, "n": n, "before": baseline["successes"],
                          "after": current["successes"]}}


def save_gif(frames, path: Path):
    if frames:
        imageio.mimsave(path, frames[::2], duration=0.1, loop=0)


def rehearse(scene_path: str, instruction: str, n_stress: int = 40, out_root: str = "out",
             log=print, look_rounds: int = 2, progress=None, agent_loop: bool = False) -> dict:
    """progress(phase, fraction) gets the overall fraction done (0-1) at each step; phases are
    setup, attempt, stress, look, report. The agent's data mode follows scene["privacy"]
    ("public" -> Contributor mode; anything else, including absent, -> private).
    agent_loop=True (opt-in): Muse Spark drives simulate/stress_test/look_again/submit itself
    (rehearsal/agent_loop.py); its trace is stored under the optional report key "agent_trace"."""
    progress = progress or _noop
    progress("setup", 0.0)
    scene = load_scene(scene_path)
    out = Path(out_root) / scene["name"]
    out.mkdir(parents=True, exist_ok=True)
    for old in [*out.glob("attempt_*.gif"), *out.glob("attempt_*.traj.json"), *out.glob("stress_trails*.json")]:
        old.unlink()
    export_scene3d(build_model(scene), out)
    data = "public" if scene.get("privacy") == "public" else "private"
    agent = make_agent(data=data, scene=scene)
    log(f"agent: {agent.name} (data={data})")

    t0 = time.time()
    reach = reachability_report(scene)
    log(f"reachable: {reach}")
    summary = scene_summary(scene, reach)

    trace = None
    if agent_loop:
        history, goals, trace = _agent_loop_attempts(scene, instruction, data, out, log, progress)
    else:
        goals = agent.compile_goals(instruction, summary)
        goals = with_implicit_goals(goals, [o["id"] for o in scene["objects"]])
        log("goals: " + json.dumps(goals, ensure_ascii=False))
        history = []
    plan = agent.plan(instruction, summary, goals) if not agent_loop else None
    for attempt in range(1, MAX_ATTEMPTS + 1 if not agent_loop else 1):
        progress("attempt", 0.05 + 0.35 * (attempt - 1) / MAX_ATTEMPTS)
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
    # Headline stress test runs on the held-out seed block. When active perception runs, it picks
    # objects on a separate selection block (seeds 0..n-1) and its gain is re-measured on the
    # held-out block, paired with this headline (same seeds), so the before/after is not inflated
    # by having chosen the object on the same draws.
    sig0 = object_sigmas(scene)
    stress = None
    if final["success"] and n_stress:
        progress("stress", 0.4)
        stress = stress_test(scene, final["plan"], goals, n_stress, sig0, trails=True, seed0=HELDOUT_SEED0,
                             on_run=lambda k, n: progress("stress", 0.4 + 0.2 * k / n))
    if stress:
        (out / "stress_trails.json").write_text(json.dumps(
            [{"seed": r["seed"], "success": r["success"], "trails": r.pop("trails")} for r in stress["runs"]]))
        sig_txt = (f"±{SIGMA * 100:.1f} cm" if len(set(sig0)) == 1 and sig0[0] == SIGMA
                   else f"per-object {min(sig0) * 100:.1f}-{max(sig0) * 100:.1f} cm")
        log(f"\nstress test: {stress['successes']}/{stress['n']} succeeded under randomized "
            f"friction, mass and {sig_txt} perception error (held-out seeds {HELDOUT_SEED0}+)")
    looks = None
    if stress and look_rounds:
        log("\nactive perception: which object is worth a second look? (selection seeds 0+)")
        progress("look", 0.6)
        sel = stress_test(scene, final["plan"], goals, n_stress, sig0, seed0=0)
        looks = active_perception(
            scene, final["plan"], goals, n_stress, sel, look_rounds, log,
            on_option=lambda rnd, k, m: progress("look", 0.6 + 0.3 * (rnd + k / m) / look_rounds),
            sigmas0=sig0)
        for st in looks["steps"]:
            log(f"  look at {st['look_at']}: {st['before']}/{n_stress} -> {st['after']}/{n_stress} (selection)")
        final_sigmas = looks.pop("final_sigmas")
        after = stress
        if looks["steps"]:
            progress("look", 0.9)
            after = stress_test(scene, final["plan"], goals, n_stress, final_sigmas, trails=True,
                                seed0=HELDOUT_SEED0)
            (out / "stress_trails_after.json").write_text(json.dumps(
                [{"seed": r["seed"], "success": r["success"], "trails": r.pop("trails")} for r in after["runs"]]))
            log(f"  held-out check: {stress['successes']}/{n_stress} -> {after['successes']}/{n_stress}")
        looks["selection_final"] = _brief(looks["final"])
        looks["heldout"] = {"seed0": HELDOUT_SEED0, "n": n_stress, "before": stress["successes"],
                            "after": after["successes"]}
        # `final` (shown by the UI as the after-look number) is the held-out result.
        looks["final"] = after

    progress("report", 0.97)
    report = {"scene": scene["name"], "scene_spec": scene, "instruction": instruction,
              "agent": agent.name, "reachability": reach, "goals": goals, "attempts": history,
              "stress_test": stress, "active_perception": looks, "wall_time_s": round(time.time() - t0, 1)}
    if trace is not None:
        report["agent"] = trace["model"] + " (agent loop)"
        report["agent_trace"] = trace
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float))
    (out / "report.md").write_text(render_report(report), encoding="utf-8")
    log(f"\nreport: {out / 'report.md'}  ({report['wall_time_s']} s)")
    return report


def _agent_loop_attempts(scene, instruction, data, out, log, progress):
    """Run the agentic loop, then re-render each simulate/submit call as attempt_N.gif/.traj.json."""
    from .agent_loop import AgentLoop
    progress("attempt", 0.05)
    res = AgentLoop(scene, instruction, data=data, log=log).run()
    goals = res["goals"]
    log("goals: " + json.dumps(goals, ensure_ascii=False))
    history = []
    for a in res["attempts"]:
        k = a["attempt"]
        r = run_plan(scene, a["plan"], goals, render=True, record="full")
        save_gif(r.pop("frames"), out / f"attempt_{k}.gif")
        (out / f"attempt_{k}.traj.json").write_text(json.dumps(r.pop("trajectory")))
        history.append({"attempt": k, "plan": a["plan"], "via_tool": a["via_tool"], **r})
        log(f"--- attempt {k} ({a['via_tool']}): {'SUCCESS' if r['success'] else 'FAILED'}")
    tr = res["agent_trace"]
    tr["agent_stress_runs"] = [{"plan": s["plan"], "look_at": s.get("look_at"), "successes": s["successes"],
                                "n": s["n"], "sigma_cm": s["sigma_cm"]} for s in res["stress_runs"]]
    tr["agent_looks"] = res["looks"]
    log("\n".join(tr["story"]) + f"\nagent loop: {tr['stopped']}, ${tr['cost_usd']:.4f}")
    if not history:
        raise RuntimeError(f"agent loop produced no simulated plan ({tr['stopped']})")
    return history, goals, tr


def _offline() -> bool:
    from .agents import offline_forced
    return offline_forced() or not os.environ.get("MODEL_API_KEY")


def demo_instruction(spec: dict) -> str:
    """The instruction to precompute with. The offline stand-in only understands the Chinese demo
    vocabulary, so offline it uses "demo_instruction_zh" unless it can replay the scene's
    `reference` block (OfflineAgent(scene) does, then the instruction is display text only)."""
    import inspect
    from .agents import OfflineAgent
    replays = "scene" in inspect.signature(OfflineAgent).parameters
    if _offline() and spec.get("demo_instruction_zh") and not (spec.get("reference") and replays):
        return spec["demo_instruction_zh"]
    return spec["demo_instruction"]


def precompute(scenes_dir: str = "scenes", out_root: str = "out", only: list[str] | None = None,
               log=print) -> list[dict]:
    """Rehearse every scene that carries a "demo_instruction" (see module docstring)."""
    done = []
    for path in sorted(Path(scenes_dir).glob("*.json")):
        spec = json.loads(path.read_text())
        if not spec.get("demo_instruction") or (only and path.stem not in only):
            continue
        instr = demo_instruction(spec)
        log(f"\n===== precompute {path.stem}: {instr}")
        t0 = time.time()
        try:
            rep = rehearse(str(path), instr, int(spec.get("demo_stress", 40)), out_root,
                           log=log, look_rounds=int(spec.get("demo_look", 2)))
            st = rep["stress_test"]
            done.append({"scene": path.stem, "success": rep["attempts"][-1]["success"],
                         "attempts": len(rep["attempts"]),
                         "stress": f"{st['successes']}/{st['n']}" if st else None})
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            done.append({"scene": path.stem, "error": f"{type(e).__name__}: {e}"})
        done[-1]["seconds"] = round(time.time() - t0, 1)
    log("\n===== precompute summary")
    for d in done:
        log("  " + json.dumps(d, ensure_ascii=False))
    if not done:
        log(f"  no scenes with demo_instruction in {scenes_dir}/")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?")
    ap.add_argument("instruction", nargs="?")
    ap.add_argument("--stress", type=int, default=40)
    ap.add_argument("--out", default="out")
    ap.add_argument("--look", type=int, default=2, help="active-perception rounds")
    ap.add_argument("--precompute", action="store_true",
                    help='rehearse every scenes/*.json with a "demo_instruction" (offline)')
    ap.add_argument("--only", default="", help="with --precompute: comma-separated scene names")
    ap.add_argument("--online", action="store_true", help="with --precompute: allow Muse Spark (costs money)")
    ap.add_argument("--agent-loop", action="store_true",
                    help="Muse Spark drives simulate/stress_test/look_again/submit itself (costs money)")
    ap.add_argument("--scenes", default="scenes", help="with --precompute: scenes directory")
    args = ap.parse_args()
    if args.precompute:
        if not args.online:
            os.environ["REHEARSAL_OFFLINE"] = "1"
        res = precompute(args.scenes, args.out, [s for s in args.only.split(",") if s] or None)
        sys.exit(1 if any("error" in r for r in res) else 0)
    if not args.scene or not args.instruction:
        ap.error("scene and instruction are required (or use --precompute)")
    rehearse(args.scene, args.instruction, args.stress, args.out, look_rounds=args.look,
             agent_loop=args.agent_loop)


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
              f"per-object perception error (σ from the scene, default 1.2 cm), seeds {st.get('seed0', 0)}+: "
              f"**{st['successes']}/{st['n']} succeeded**.", ""]
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
              f"its σ to σ={ap_['refined_sigma_cm']:.1f} cm. Objects are chosen on selection seeds "
              f"(numbers in the list below, optimistic); the gain is then re-measured on held-out seeds.", ""]
        for i, st in enumerate(ap_["steps"], 1):
            vals = ", ".join(f"{k} {v}/{ap_['n']}" for k, v in
                             sorted(st["value_by_object"].items(), key=lambda kv: -kv[1]))
            L.append(f"{i}. Look again at **{st['look_at']}**: {st['before']}/{ap_['n']} → "
                     f"{st['after']}/{ap_['n']}  (selection seeds; candidates: {vals})")
        ho = ap_.get("heldout")
        if ho:
            L.append(f"\n**Held-out check** (seeds {ho['seed0']}+, not used for the choice): "
                     f"{ho['before']}/{ho['n']} → {ho['after']}/{ho['n']}")
        L.append("")
    L += ["## Assumed (not verified)", "",
          "- Grasping always succeeds (suction weld). Real grasp physics is out of scope.",
          "- Object geometry is approximated by primitives; perception is taken as given.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
