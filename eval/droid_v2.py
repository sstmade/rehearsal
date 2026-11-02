"""DROID v2: physics replay vs trivial baselines on where objects end up (protocol: eval/droid_v2_protocol.md).

    .venv/bin/python eval/droid_v2.py run            # replay + SAM 3.1 video (cached) for every v2 episode
    .venv/bin/python eval/droid_v2.py score          # baselines + statistics from the saved per-episode files

Predictors of each object's final table xy: (a) replay, (b) stay-put, (c) release point (trajectory only).
Success: replay predicates vs (d) the v1 gripper rule, always-success, always-failure.
Real positions are measured predictor-neutrally (every object's bottom on the table, z0 = 0) from one SAM 3.1 video
request on the clip subsampled to every 2nd frame.  The DROID success label is read only when scoring.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "datasets" / "droid"
RES = ROOT / "eval" / "results" / "droid_v2"
OUT = ROOT / "out" / "droid_v2"
MOVED_CM = 3.0            # reality.GAP_TOL_M
HOLD_MAX = 0.90           # v1 droid_trivial_baseline
SETTLE_STEPS = 8
NEAR_M = 0.10             # release-point baseline: grasped object must be within 10 cm of the EE xy
STRIDE = 2
N_BOOT = 10_000


def v2_slugs() -> list[str]:
    return [e["slug"] for e in json.loads((DATA / "episodes_v2.json").read_text())["episodes"]]


# ------------------------------------------------------------------------------------------- baselines
def close_segments(cmd) -> list[tuple[int, int]]:
    segs, i, n = [], 0, len(cmd)
    while i < n:
        if cmd[i] > 0.5:
            j = i
            while j < n and cmd[j] > 0.5:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def holding(g, a: int, b: int) -> bool:
    settled = g[min(b - 1, a + SETTLE_STEPS):b]
    return bool(len(settled)) and max(settled) < HOLD_MAX


def release_point(traj: dict, start_xy: dict[str, list[float]]) -> tuple[dict, list[str]]:
    """(c): for each holding close segment, the object nearest the EE xy at the segment's first step (<= NEAR_M)
    ends at the EE xy at the segment's last step; everything else stays put.  Returns (final xy, grasped ids)."""
    P = np.asarray(traj["cartesian_position"], float)
    g = traj["gripper_position"]
    cur = {k: np.asarray(v[:2], float) for k, v in start_xy.items()}
    grasped = []
    for a, b in close_segments(traj["gripper_command"]):
        if not holding(g, a, b) or not cur:
            continue
        ee = P[a, :2]
        oid = min(cur, key=lambda k: float(np.linalg.norm(cur[k] - ee)))
        if np.linalg.norm(cur[oid] - ee) > NEAR_M:
            continue
        cur[oid] = P[b - 1, :2].copy()
        grasped.append(oid)
    return {k: v.round(4).tolist() for k, v in cur.items()}, grasped


def gripper_rule(traj: dict, lift_min: float = 0.05) -> bool:
    """(d), v1's finger variant: some close segment holds and the EE then rises >= 5 cm while closed."""
    z = [p[2] for p in traj["cartesian_position"]]
    g = traj["gripper_position"]
    return any(holding(g, a, b) and max(z[a:b]) - z[a] >= lift_min
               for a, b in close_segments(traj["gripper_command"]))


# ---------------------------------------------------------------------------------------------- running
def track_clip(slug: str, stride: int = STRIDE) -> Path:
    """The packaged clip with every `stride`-th frame (last frame always kept), for the SAM 3.1 video request."""
    import imageio.v2 as iio
    out = OUT / f"{slug}.track.mp4"
    if out.exists():
        return out
    rd = iio.get_reader(str(DATA / slug / "clip.mp4"))
    frames = [rd.get_data(i) for i in range(rd.count_frames())]
    rd.close()
    idx = list(range(0, len(frames), stride))
    if idx[-1] != len(frames) - 1:
        idx.append(len(frames) - 1)
    OUT.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(out, [frames[i] for i in idx], fps=15 / stride, codec="libx264", quality=8, macro_block_size=2,
                 ffmpeg_params=["-pix_fmt", "yuv420p"])
    return out


def run_episode(slug: str) -> dict:
    from rehearsal.reality import reality_check
    from rehearsal.replay import clip_camera, load_episode, load_scene, replay
    scene, ep = load_scene(slug), load_episode(slug)
    res = replay(scene, ep)
    clip, cam = track_clip(slug), clip_camera(ep)
    # predictor-neutral measurement: every object's bottom on the table
    neutral = {o["id"]: {"xy": o["pos"][:2], "z_base": 0.0} for o in scene["objects"]}
    rc = reality_check(clip, scene, neutral, data="public", camera=cam)
    # secondary (v1 style): measured at the replay's predicted height (same SAM request -> cached, $0)
    simz = {oid: {"xy": o["final_xy"], "z_base": max(0.0, o["final_z_bottom"])} for oid, o in res["objects"].items()}
    rc_z = reality_check(clip, scene, simz, data="public", camera=cam)
    start_xy = {o["id"]: o["pos"][:2] for o in scene["objects"]}
    rel, grasped = release_point(ep["traj"], start_xy)
    z_final = {r["id"]: r.get("observed_xy") for r in rc_z["per_object"]}
    rows = []
    for r in rc["per_object"]:
        oid = r["id"]
        rows.append({"id": oid, "status": r["status"], "scene_xy": [round(v, 4) for v in start_xy[oid]],
                     "real_start_xy": r.get("observed_start_xy"), "real_final_xy": r.get("observed_xy"),
                     "real_final_xy_at_replay_z": z_final.get(oid),
                     "real_moved_cm": r.get("observed_moved_cm"), "frames_tracked": r.get("frames_tracked"),
                     "pred": {"replay": res["objects"][oid]["final_xy"], "stay_put": start_xy[oid][:2],
                              "release_point": rel[oid]},
                     "replay_displacement_cm": res["objects"][oid]["displacement_cm"],
                     "replay_up_cos": res["objects"][oid]["up_cos"]})
    rec = {"slug": slug, "events": res["events"], "goals": res["goals"], "sim_success": res["success"],
           "gripper_rule_success": gripper_rule(ep["traj"]), "release_point_grasped": grasped,
           "objects": rows, "frames_sent_to_sam": rc["frames"], "table_z_base_m":
               ep["camera"]["table_plane_base"]["z_table_base_m"]}
    # evaluation only, after every prediction above is fixed
    rec["real_success"] = bool(json.loads((DATA / slug / "label.json").read_text())["success"])
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{slug}.json").write_text(json.dumps(rec, indent=1))
    return rec


# ---------------------------------------------------------------------------------------------- scoring
PREDICTORS = ("replay", "stay_put", "release_point")


def object_rows(recs: list[dict], final_key: str = "real_final_xy") -> list[dict]:
    """One row per measurable object: errors (cm) of every predictor, real moved flag, collateral flag."""
    out = []
    for rec in recs:
        for o in rec["objects"]:
            real = o.get(final_key)
            if real is None or o.get("real_moved_cm") is None:
                continue
            real = np.asarray(real, float)
            err = {p: float(np.linalg.norm(np.asarray(o["pred"][p], float) - real)) * 100 for p in PREDICTORS}
            pmoved = {"replay": o["replay_displacement_cm"] > MOVED_CM, "stay_put": False,
                      "release_point": float(np.linalg.norm(np.asarray(o["pred"]["release_point"], float)
                                                            - np.asarray(o["scene_xy"], float))) * 100 > MOVED_CM}
            moved = o["real_moved_cm"] > MOVED_CM
            out.append({"episode": rec["slug"], "id": o["id"], "err": err, "real_moved": moved,
                        "real_moved_cm": o["real_moved_cm"], "pred_moved": pmoved,
                        "collateral": moved and o["id"] not in rec["release_point_grasped"]})
    return out


def cluster_bootstrap(rows: list[dict], stat, n: int = N_BOOT, seed: int = 0):
    """Percentile 95% CI of stat(rows) resampling episodes with replacement."""
    eps = sorted({r["episode"] for r in rows})
    if not eps:
        return None
    by = {e: [r for r in rows if r["episode"] == e] for e in eps}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(eps), len(eps))
        sample = [r for k in pick for r in by[eps[k]]]
        vals.append(stat(sample))
    return [round(float(np.percentile(vals, 2.5)), 2), round(float(np.percentile(vals, 97.5)), 2)]


def wilcoxon_p(d) -> float:
    """Two-sided Wilcoxon signed-rank p-value, zeros dropped (scipy's zero_method="wilcox"), exact conditional
    distribution of the rank sum (average ranks for ties) by dynamic programming.  scipy is not a runtime dep."""
    d = np.asarray([x for x in d if x != 0], float)
    n = len(d)
    if n == 0:
        return 1.0
    a = np.abs(d)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r2 = np.rint(2 * ranks).astype(int)              # integer (doubled) ranks
    t = int(r2[d > 0].sum())
    dist = np.zeros(int(r2.sum()) + 1)
    dist[0] = 1.0
    for r in r2:
        dist[r:] = dist[r:] + dist[:-r].copy() if r else dist
    dist /= dist.sum()
    cdf_lo, cdf_hi = dist[:t + 1].sum(), dist[t:].sum()
    return float(min(1.0, 2 * min(cdf_lo, cdf_hi)))


def compare_subset(rows: list[dict], n_boot: int = N_BOOT) -> dict:
    out = {"n_objects": len(rows), "n_episodes": len({r["episode"] for r in rows})}
    if not rows:
        return out
    for p in PREDICTORS:
        e = [r["err"][p] for r in rows]
        out[p] = {"median_cm": round(float(np.median(e)), 2), "mean_cm": round(float(np.mean(e)), 2),
                  "median_ci": cluster_bootstrap(rows, lambda s, p=p: np.median([r["err"][p] for r in s]), n_boot)}
    for b in ("stay_put", "release_point"):
        d = [r["err"]["replay"] - r["err"][b] for r in rows]
        c = {"mean_diff_cm": round(float(np.mean(d)), 2),
             "mean_diff_ci": cluster_bootstrap(rows, lambda s, b=b: np.mean([r["err"]["replay"] - r["err"][b]
                                                                            for r in s]), n_boot),
             "replay_better": sum(x < 0 for x in d), "baseline_better": sum(x > 0 for x in d),
             "ties": sum(x == 0 for x in d)}
        if len(rows) >= 6 and any(x != 0 for x in d):
            c["wilcoxon_p"] = round(wilcoxon_p(d), 4)
        lo, hi = c["mean_diff_ci"]
        c["verdict"] = ("replay better" if hi < 0 and c.get("wilcoxon_p", 1) < 0.05 else
                        "replay worse" if lo > 0 else "not distinguishable")
        out[f"replay_vs_{b}"] = c
    return out


def score(recs: list[dict], final_key: str = "real_final_xy", n_boot: int = N_BOOT) -> dict:
    rows = object_rows(recs, final_key)
    subsets = {"all": rows, "moved": [r for r in rows if r["real_moved"]],
               "collateral": [r for r in rows if r["collateral"]],
               "not_grasped": [r for r in rows if not any(r["id"] in rec["release_point_grasped"]
                                                          for rec in recs if rec["slug"] == r["episode"])]}
    out = {"final_key": final_key, "subsets": {k: compare_subset(v, n_boot) for k, v in subsets.items()}}
    out["moved_agreement"] = {p: {"agree": sum(r["pred_moved"][p] == r["real_moved"] for r in rows), "n": len(rows)}
                              for p in PREDICTORS}
    out["success_agreement"] = {
        "replay": sum(r["sim_success"] == r["real_success"] for r in recs),
        "gripper_rule": sum(r["gripper_rule_success"] == r["real_success"] for r in recs),
        "always_success": sum(r["real_success"] for r in recs),
        "always_failure": sum(not r["real_success"] for r in recs), "n": len(recs),
        "replay_false_successes": sum(r["sim_success"] and not r["real_success"] for r in recs),
        "gripper_rule_false_successes": sum(r["gripper_rule_success"] and not r["real_success"] for r in recs)}
    out["excluded_objects"] = [f"{rec['slug']}/{o['id']} ({o['status']})" for rec in recs for o in rec["objects"]
                               if o.get(final_key) is None or o.get("real_moved_cm") is None]
    out["collateral_objects"] = [f"{r['episode']}/{r['id']} real {r['real_moved_cm']} cm: "
                                 + ", ".join(f"{p} {r['err'][p]:.1f}" for p in PREDICTORS)
                                 for r in subsets["collateral"]]
    out["rows"] = rows
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    slugs = sys.argv[2:] or v2_slugs()
    if cmd == "run":
        for s in slugs:
            try:
                rec = run_episode(s)
                print(s, "sim", rec["sim_success"], "real", rec["real_success"], "grasped(c)",
                      rec["release_point_grasped"], [(o["id"], o["status"], o["real_moved_cm"]) for o in rec["objects"]],
                      flush=True)
            except Exception as e:  # noqa: BLE001 - protocol: report and exclude, never swap
                print(s, "ERROR", type(e).__name__, e, flush=True)
    recs = [json.loads((RES / f"{s}.json").read_text()) for s in v2_slugs() if (RES / f"{s}.json").exists()]
    sm = {"episodes": [r["slug"] for r in recs], "primary": score(recs),
          "secondary_measured_at_replay_z": score(recs, "real_final_xy_at_replay_z")}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "summary.json").write_text(json.dumps(sm, indent=1))
    p = sm["primary"]
    print(json.dumps({k: v for k, v in p.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
