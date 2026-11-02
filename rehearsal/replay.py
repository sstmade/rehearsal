"""Mode B: replay a real robot's recorded motion in the twin and predict what happens to the objects.

    python -m rehearsal.replay duck_success [...]          # or: all ;  --reality to compare with the clip (SAM 3.1)

Input: a DROID episode (datasets/droid/<slug>/, CC BY 4.0) and its Rehearsal scene (scenes/droid/<slug>.json, built
from the episode's FIRST frame by scripts/droid_scene.py).  The Panda in MuJoCo follows the recorded joint positions
(15 Hz, linearly interpolated to the 500 Hz physics step; the arm is driven kinematically, like the real arm's stiff
position control - contacts do not stop it).  The tool is an approximate Robotiq 2F-85 (scene.ROBOTIQ_*).

Grasp model: the fingers follow the operator's gripper COMMAND (action/target_gripper_position), never the measured
finger position, which already tells whether something was caught.  While the command is "closed", an object is
grasped (welded to the hand at its current relative pose, and the fingers stop squeezing) only when BOTH fingers
touch it in the physics engine - i.e. the closing fingers bracket it.  Closing on air, on one side of an object or
above it grasps nothing; an object pushed by a finger moves by contact physics alone.  Opening the command releases.

Prediction (everything from physics, nothing from the episode's outcome label): every object's final pose, which
objects moved (> 2 cm, as predicates.undisturbed), and the instruction's predicates (scene["droid"]["goals"],
compiled from the language instruction by Muse Spark) -> success / failure.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from . import predicates
from .scene import PANDA_FINGER_GAIN, ROBOTIQ_EXTENSION, ROBOTIQ_HALF_STROKE, build_model, object_height
from .sim import Sim

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "droid"
SCENES = ROOT / "scenes" / "droid"
OUT = ROOT / "out" / "droid"
EPISODES = ["duck_success", "duck_missed_grasp", "duck_fail_nudge", "bowl_fail", "cup_success", "cup_fail"]

HZ = 15.0
CMD_CLOSED = 0.5           # gripper command above this = "close"
CMD_FULL = 0.88            # the teleop's full-close command
FINGER_SPEED = 0.13        # m/s of total opening change (2F-85: 20-150 mm/s; DROID's ~0.13 measured from obs vs cmd)
MOVE_TOL = predicates.MOVE_TOL
MIN_GRASP_GAP = 0.004      # m between the pads: below this the fingers are closed, nothing is between them
BLOCKED = 0.003            # m: the fingers are held this much wider than commanded -> something is in the way


def load_episode(slug: str) -> dict:
    """Trajectory + camera of one vendored episode.  The success label is deliberately NOT loaded here."""
    d = DATA / slug
    return {"slug": slug, "traj": json.loads((d / "trajectory.json").read_text()),
            "camera": json.loads((d / "camera.json").read_text()), "clip": d / "clip.mp4"}


def load_scene(slug: str) -> dict:
    return json.loads((SCENES / f"{slug}.json").read_text())


def replay_scene(scene: dict) -> dict:
    sc = copy.deepcopy(scene)
    sc.setdefault("robot", {})["gripper"] = "robotiq_2f85"
    return sc


def command_width(c: float) -> float:
    """Gripper command (0 open .. ~0.88 closed) -> commanded opening between the finger pads (m)."""
    return 2 * ROBOTIQ_HALF_STROKE * (1.0 - min(max(c, 0.0) / CMD_FULL, 1.0))


class EpisodeCamera:
    """Renders the MuJoCo scene through a real pinhole camera (K, R, t in the table frame), principal point included,
    by overwriting the OpenGL camera's frustum after mjv_updateScene."""

    def __init__(self, model, K, R, t, width: int, height: int):
        self.K, self.R, self.t = np.asarray(K, float), np.asarray(R, float), np.asarray(t, float)
        self.w, self.h = width, height
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        model.vis.quality.offsamples = 4
        model.vis.map.znear = 0.01
        self.renderer = mujoco.Renderer(model, height, width)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

    def render(self, data) -> np.ndarray:
        r = self.renderer
        r.update_scene(data, self.cam, self.opt)
        C = -self.R.T @ self.t
        fwd, up = self.R[2], -self.R[1]              # OpenCV z (forward) and -y (up) in the world
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        n = r.scene.camera[0].frustum_near
        left, right = -cx / fx * n, (self.w - cx) / fx * n
        for gc in r.scene.camera:
            gc.pos[:] = C
            gc.forward[:] = fwd
            gc.up[:] = up
            gc.frustum_center = 0.5 * (left + right)
            gc.frustum_width = 0.5 * (right - left)
            gc.frustum_top = cy / fy * n
            gc.frustum_bottom = -(self.h - cy) / fy * n
        return r.render().copy()


def replay(scene: dict, ep: dict, render: bool = False, render_size=(960, 540), settle_s: float = 1.0) -> dict:
    sc = replay_scene(scene)
    m = build_model(sc)
    sim = Sim(m, sc)                       # objects settle with the arm parked at home
    d = sim.d
    tr = ep["traj"]
    Q = np.asarray(tr["joint_positions"], float)
    cmd = np.asarray(tr["gripper_command"], float)
    n = len(Q)
    T = (n - 1) / HZ
    dt = m.opt.timestep
    lf, rf = m.body("left_finger").id, m.body("right_finger").id
    obj_body = {b: oid for oid, b in sim.body_of.items()}
    root = m.body_rootid

    def set_arm(q, qd):
        d.qpos[:7], d.qvel[:7], d.ctrl[:7] = q, qd, q

    def finger_ctrl(width):
        d.ctrl[7] = (width / 2) / PANDA_FINGER_GAIN

    # start: arm at the first recorded pose, fingers at the first command, objects settle again (0.5 s)
    width = command_width(cmd[0])
    d.qpos[7:9] = width / 2
    for _ in range(int(0.5 / dt)):
        set_arm(Q[0], np.zeros(7))
        finger_ctrl(width)
        mujoco.mj_step(m, d)
    start = {oid: sim.obj_state(oid)["pos"].copy() for oid in sim.objs}
    sim.start_pos = {oid: p.copy() for oid, p in start.items()}

    cam = None
    frames = []
    if render:
        c = ep["camera"]["rehearsal_camera"]
        s = render_size[0] / ep["camera"]["width"]
        cam = EpisodeCamera(m, np.diag([s, s, 1.0]) @ np.asarray(c["K"]), c["R"], c["t"], *render_size)

    held, hold_width = None, None
    events, closed_on_air_logged = [], False
    obj_track = []              # per control step: {id: [x, y, z]}
    steps = int(round(T / dt))
    next_frame = 0             # record / render at the exact 15 Hz frame times (k*dt drifts from i/15 otherwise)
    tq = np.arange(n) / HZ
    for k in range(steps + 1):
        t = k * dt
        q = np.array([np.interp(t, tq, Q[:, j]) for j in range(7)])
        i = min(int(t * HZ), n - 2)
        qd = (Q[i + 1] - Q[i]) * HZ
        c = cmd[min(int(round(t * HZ)), n - 1)]
        closing = c >= CMD_CLOSED
        # finger opening: rate-limited toward the command; a held object keeps them at its width
        target = command_width(c)
        if held is not None:
            target = max(target, hold_width)
        width += np.clip(target - width, -FINGER_SPEED * dt, FINGER_SPEED * dt)
        set_arm(q, qd)
        finger_ctrl(width)
        mujoco.mj_step(m, d)

        if k % 5 == 0:
            if held is None and closing:
                touch = {lf: {}, rf: {}}
                for ci in range(d.ncon):
                    con = d.contact[ci]
                    for ga, gb in ((con.geom1, con.geom2), (con.geom2, con.geom1)):
                        fb = m.geom_bodyid[ga]
                        if fb in touch:
                            ob = obj_body.get(int(root[m.geom_bodyid[gb]]))
                            if ob is not None:
                                touch[fb][ob] = touch[fb].get(ob, 0) + 1
                gap = float(d.qpos[7] + d.qpos[8])
                # bracketing: both fingers touch the same object AND it holds them apart (the measured gap stays
                # wider than the rate-limited command) - two closed fingertips brushing one wall are not a grasp
                both = set(touch[lf]) & set(touch[rf]) if gap > MIN_GRASP_GAP and gap > width + BLOCKED else set()
                if both:
                    oid = max(both, key=lambda o: touch[lf][o] + touch[rf][o])
                    _weld(sim, oid)
                    held = oid
                    hold_width = float(d.qpos[7] + d.qpos[8])
                    events.append({"t": round(t, 3), "step": int(round(t * HZ)), "kind": "grasp", "obj": oid,
                                   "finger_gap_cm": round(hold_width * 100, 1)})
                elif width < 0.006 and not closed_on_air_logged:
                    one_side = sorted(set(touch[lf]) | set(touch[rf]))
                    events.append({"t": round(t, 3), "step": int(round(t * HZ)), "kind": "closed_on_air",
                                   "touching_one_finger": one_side})
                    closed_on_air_logged = True
            elif held is not None and not closing:
                d.eq_active[sim.eq_of[held]] = 0
                events.append({"t": round(t, 3), "step": int(round(t * HZ)), "kind": "release", "obj": held,
                               "at_xyz": np.round(sim.obj_state(held)["pos"], 3).tolist()})
                held, hold_width = None, None
            if not closing:
                closed_on_air_logged = False
        if t + 0.5 * dt >= next_frame / HZ:
            next_frame += 1
            obj_track.append({oid: np.round(sim.obj_state(oid)["pos"], 4).tolist() for oid in sim.objs})
            if cam is not None:
                frames.append(cam.render(d))

    # after the recording: arm holds its last pose, objects settle (as executor.run_plan does)
    if held is not None:
        d.eq_active[sim.eq_of[held]] = 0
        events.append({"t": round(T, 3), "step": n - 1, "kind": "release_at_end", "obj": held})
    before = {oid: sim.obj_state(oid)["pos"].copy() for oid in sim.objs}
    for _ in range(int(settle_s / dt)):
        set_arm(Q[-1], np.zeros(7))
        finger_ctrl(width)
        mujoco.mj_step(m, d)
    sim.settle_motion = {oid: float(np.linalg.norm(sim.obj_state(oid)["pos"] - before[oid])) for oid in sim.objs}

    goals = (scene.get("droid") or {}).get("goals") or []
    results = predicates.evaluate(sim, goals)
    objects = {}
    for oid in sim.objs:
        st = sim.obj_state(oid)
        disp = float(np.linalg.norm(st["pos"][:2] - start[oid][:2]))
        objects[oid] = {"start_xy": np.round(start[oid][:2], 4).tolist(), "final_xy": np.round(st["pos"][:2], 4).tolist(),
                        "final_z_bottom": round(float(st["pos"][2] - object_height(sim.objs[oid]) / 2), 4),
                        "displacement_cm": round(disp * 100, 2), "up_cos": round(float(st["up_cos"]), 3),
                        "moved": bool(disp > MOVE_TOL or st["up_cos"] < predicates.UPRIGHT_COS)}
    return {"slug": ep["slug"], "success": bool(goals) and all(r["ok"] for r in results), "goals": results,
            "objects": objects, "events": events, "object_track_15hz": obj_track, "frames": frames,
            "sim_time": round(float(d.time), 2)}


def _weld(sim: Sim, oid: str):
    """Weld oid to the hand at its current relative pose (same as Sim.grasp, without the suction-cup checks)."""
    m, d = sim.m, sim.d
    eq, hb, ob = sim.eq_of[oid], sim.hand, sim.body_of[oid]
    rel_p = d.xmat[hb].reshape(3, 3).T @ (d.xpos[ob] - d.xpos[hb])
    rel_q = np.zeros(4)
    hq = d.xquat[hb]
    mujoco.mju_mulQuat(rel_q, np.array([hq[0], -hq[1], -hq[2], -hq[3]]), d.xquat[ob])
    m.eq_data[eq, 0:3] = 0
    m.eq_data[eq, 3:6] = rel_p
    m.eq_data[eq, 6:10] = rel_q
    m.eq_data[eq, 10] = 1.0
    d.eq_active[eq] = 1


# ------------------------------------------------------------------------------------------ reality comparison
def clip_camera(ep: dict):
    """The episode camera at the clip's resolution (960x540 = 0.75 x the 1280x720 calibration)."""
    from .calib import Camera
    c = ep["camera"]["rehearsal_camera"]
    s = 960 / ep["camera"]["width"]
    return Camera(np.diag([s, s, 1.0]) @ np.asarray(c["K"]), c["R"], c["t"])


def compare(slug: str, res: dict, scene: dict, out_dir: Path | None = None) -> dict:
    """Track the scene's objects through the real clip with SAM 3.1 video (reality.reality_check, episode camera)
    and compare where they really ended up with the replay's prediction.  Costs one SAM video call (cached)."""
    from .reality import GAP_TOL_M, reality_check
    ep = load_episode(slug)
    sim_final = {oid: {"xy": o["final_xy"], "z_base": max(0.0, o["final_z_bottom"])} for oid, o in res["objects"].items()}
    rc = reality_check(ep["clip"], scene, sim_final, data="public", out_dir=out_dir, camera=clip_camera(ep))
    rows = []
    for r in rc["per_object"]:
        o = res["objects"][r["id"]]
        # moved / not moved is judged at reality.GAP_TOL_M (3 cm) on BOTH sides: the clip measurement of an untouched
        # object wanders up to ~2.7 cm (partial occlusion by the arm), so the predicate's 2 cm would flag noise
        real_moved = None if r.get("observed_moved_cm") is None else r["observed_moved_cm"] > GAP_TOL_M * 100
        rows.append({"id": r["id"], "sim_final_xy": o["final_xy"], "real_final_xy": r["observed_xy"],
                     "final_error_cm": r["delta_cm"], "status": r["status"],
                     "sim_moved": o["displacement_cm"] > GAP_TOL_M * 100, "sim_displacement_cm": o["displacement_cm"],
                     "real_moved": real_moved, "real_displacement_cm": r.get("observed_moved_cm"),
                     "real_start_xy": r.get("observed_start_xy"), "scene_xy": r.get("scene_xy"),
                     "start_error_cm": r.get("start_delta_cm"), "frames_tracked": r.get("frames_tracked"),
                     "last_seen_frame": r.get("last_seen_frame"), "last_seen_xy": r.get("last_seen_xy"),
                     "last_seen_error_cm": r.get("last_seen_delta_cm")})
    return {"slug": slug, "rows": rows, "verdict": rc["verdict"], "overlay": rc.get("overlay"),
            "frames": rc["frames"], "camera": rc["camera"], "moved_tol_cm": GAP_TOL_M * 100}


def save_media(slug: str, res: dict, out_dir: Path, gif: bool = True, gif_width: int = 360):
    """Side by side: real clip | replay rendered from the same camera.  MP4 (1920x540) and optionally a GIF
    (gif_width per half, every 2nd frame)."""
    import imageio.v2 as iio
    from PIL import Image, ImageDraw, ImageFont
    ep = load_episode(slug)
    rd = iio.get_reader(str(ep["clip"]))
    real = [rd.get_data(i) for i in range(rd.count_frames())]
    rd.close()
    sim = res["frames"]
    n = min(len(real), len(sim))
    try:
        font = ImageFont.load_default(size=26)
        small = ImageFont.load_default(size=20)
    except TypeError:
        font = small = ImageFont.load_default()
    events = sorted(res["events"], key=lambda e: e["step"])
    out = []
    for i in range(n):
        a = Image.fromarray(real[i]).convert("RGB")
        b = Image.fromarray(sim[i]).convert("RGB").resize(a.size)
        W = Image.new("RGB", (a.width * 2, a.height))
        W.paste(a, (0, 0))
        W.paste(b, (a.width, 0))
        dr = ImageDraw.Draw(W)
        dr.text((16, 12), "REAL  (DROID, Franka FR3)", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        dr.text((a.width + 16, 12), "REHEARSAL REPLAY  (MuJoCo, same joints)", fill=(255, 255, 255), font=font,
                stroke_width=2, stroke_fill=(0, 0, 0))
        past = [e for e in events if e["step"] <= i and e["kind"] in ("grasp", "closed_on_air", "release")]
        if past:
            e = past[-1]
            txt = {"grasp": f"sim: fingers bracket {e.get('obj')} -> grasp",
                   "closed_on_air": "sim: gripper closed on nothing",
                   "release": f"sim: released {e.get('obj')}"}[e["kind"]]
            dr.text((a.width + 16, a.height - 40), txt, fill=(255, 230, 120), font=small, stroke_width=2,
                    stroke_fill=(0, 0, 0))
        if i == n - 1:
            verdict = "predicted: SUCCESS" if res["success"] else "predicted: FAILURE"
            dr.text((a.width + 16, 46), verdict, fill=(120, 255, 140) if res["success"] else (255, 120, 120),
                    font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        dr.text((16, a.height - 30), "Robot footage: DROID (Khazatsky et al., RSS 2024), CC BY 4.0",
                fill=(230, 230, 230), font=small, stroke_width=2, stroke_fill=(0, 0, 0))
        out.append(np.asarray(W))
    out += [out[-1]] * int(HZ * 1.5)          # hold the end state
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"{slug}.side_by_side.mp4"
    iio.mimwrite(mp4, out, fps=HZ, codec="libx264", quality=5, macro_block_size=2,
                 ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    if not gif:
        return mp4, None
    gif = out_dir / f"{slug}.side_by_side.gif"
    k = gif_width * 2 / out[0].shape[1]
    pics = [Image.fromarray(f).resize((int(f.shape[1] * k), int(f.shape[0] * k)), Image.LANCZOS) for f in out[::2]]
    pal = pics[len(pics) // 2].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    pics = [p.quantize(palette=pal, dither=Image.Dither.NONE) for p in pics]
    pics[0].save(gif, save_all=True, append_images=pics[1:], duration=int(2000 / HZ), loop=0, optimize=True)
    return mp4, gif


def sensitivity(slugs, dz=0.01, dext=0.01) -> dict:
    """Does the success prediction survive the two calibration unknowns?  Robot base height +-dz (= table height
    error) and Robotiq fingertip length +-dext.  Physics only, no API."""
    out = {}
    for slug in slugs:
        scene, ep = load_scene(slug), load_episode(slug)
        row = {}
        for name, key, delta in (("base_z", "base_z", dz), ("tool_extension", "tool_extension", dext)):
            for sgn in (-1, 1):
                sc = copy.deepcopy(scene)
                r = sc.setdefault("robot", {})
                base = float(r.get(key, 0.0 if key == "base_z" else ROBOTIQ_EXTENSION))
                r[key] = base + sgn * delta
                res = replay(sc, ep)
                row[f"{name}{'+' if sgn > 0 else '-'}{delta * 100:.0f}cm"] = {
                    "success": res["success"], "events": [(e["kind"], e.get("obj"), e["step"]) for e in res["events"]]}
        out[slug] = row
    return out


def summarize(results: list[dict]) -> dict:
    tab = {"real_success": {"sim_success": 0, "sim_failure": 0}, "real_failure": {"sim_success": 0, "sim_failure": 0}}
    for r in results:
        tab["real_success" if r["real_success"] else "real_failure"]["sim_success" if r["sim_success"] else "sim_failure"] += 1
    out = {"two_by_two": tab, "episodes": len(results),
           "success_agreement": sum(r["real_success"] == r["sim_success"] for r in results)}
    rows = [dict(row, slug=r["slug"]) for r in results if "reality" in r for row in r["reality"]["rows"]]
    if rows:
        errs = [x["final_error_cm"] for x in rows if x["final_error_cm"] is not None]
        mv = [x for x in rows if x["real_moved"] is not None]
        out["final_error_cm"] = {"n": len(errs), "median": round(float(np.median(errs)), 2),
                                 "mean": round(float(np.mean(errs)), 2), "max": round(float(np.max(errs)), 2),
                                 "untouched_in_sim_median": round(float(np.median(
                                     [x["final_error_cm"] for x in rows if x["final_error_cm"] is not None
                                      and not x["sim_moved"]])), 2)}
        out["moved_agreement"] = {"n": len(mv), "agree": sum(x["sim_moved"] == x["real_moved"] for x in mv),
                                  "disagree": [f"{x['slug']}/{x['id']}: sim {x['sim_displacement_cm']} cm, "
                                               f"real {x['real_displacement_cm']} cm" for x in mv
                                               if x["sim_moved"] != x["real_moved"]],
                                  "not_measurable": [f"{x['slug']}/{x['id']} ({x['status']})" for x in rows
                                                     if x["real_moved"] is None]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="+")
    ap.add_argument("--reality", action="store_true", help="SAM 3.1 video tracking of the clip (Meta API, cached)")
    ap.add_argument("--media", action="store_true", help="render side-by-side MP4s into docs/media/droid/")
    ap.add_argument("--gif", default="duck_success,cup_fail", help="episodes that also get a GIF (with --media)")
    ap.add_argument("--sensitivity", action="store_true", help="re-run with base height / fingertip length +-1 cm")
    a = ap.parse_args()
    slugs = EPISODES if a.slugs == ["all"] else a.slugs
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for slug in slugs:
        scene, ep = load_scene(slug), load_episode(slug)
        res = replay(scene, ep, render=a.media)
        rec = {k: v for k, v in res.items() if k != "frames"}
        if a.media:
            mp4, gif = save_media(slug, res, ROOT / "docs" / "media" / "droid", gif=slug in a.gif.split(","))
            rec["media"] = [str(p.relative_to(ROOT)) for p in (mp4, gif) if p is not None]
        if a.reality:
            rec["reality"] = compare(slug, res, scene, out_dir=OUT / slug)
        # evaluation only: the ground-truth label is read after the prediction is complete
        real = json.loads((DATA / slug / "label.json").read_text())["success"]
        rec["real_success"], rec["sim_success"] = bool(real), res["success"]
        (OUT / f"{slug}.replay.json").write_text(json.dumps(rec, indent=1))
        summary.append(rec)
        print(f"{slug:18s} sim={'SUCCESS' if res['success'] else 'failure':8s} real={'SUCCESS' if real else 'failure':8s}",
              [(e["kind"], e.get("obj"), e["step"]) for e in res["events"]],
              {o: (v["final_xy"], v["displacement_cm"]) for o, v in res["objects"].items()})
        if a.reality:
            for r in rec["reality"]["rows"]:
                print(f"    {r['id']:16s} final err {r['final_error_cm']} cm  sim moved {r['sim_moved']} "
                      f"({r['sim_displacement_cm']} cm)  real moved {r['real_moved']} ({r['real_displacement_cm']} cm)")
    sm = summarize(summary)
    if a.sensitivity:
        sm["sensitivity"] = sensitivity(slugs)
    if a.slugs == ["all"]:
        res_dir = ROOT / "eval" / "results" / "droid_replay"
        res_dir.mkdir(parents=True, exist_ok=True)
        for rec in summary:
            (res_dir / f"{rec['slug']}.json").write_text(json.dumps(rec, indent=1))
        (res_dir / "summary.json").write_text(json.dumps(sm, indent=1))
    print(json.dumps(sm, indent=1))


if __name__ == "__main__":
    main()
