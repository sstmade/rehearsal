"""Camera language + timeline for a refined hero clip (no Blender here: numpy only).

  .venv/bin/python scripts/refined/shots.py EXP_DIR KEY CACHE_DIR [--json overrides]

Reads EXP_DIR/KEY_traj.npz (+ _model.npz, _meta.json) written by export_refined.py and the run's KEY_traj.npz
(for the recording length) and writes EXP_DIR/KEY_shots.json: one entry per OUTPUT frame (30 fps):
  src    export frame whose physics state is shown (timing only: states are never invented or interpolated)
  cam    [x, y, z] camera position, look [x, y, z] target, lens (mm, 36 mm-wide sensor), focus [x, y, z]
  fade   arm/pedestal opacity (0 while the twin is seen from the photo camera, as in the MuJoCo clips)
  shot   shot name (for the README / HUD), speed (playback speed of the physics in this shot)
Shot list (calm, purposeful, no orbits):
  photo   twin through the photo's own camera (static), arm hidden
  move    one eased move from the photo camera to the establishing 3/4 view while the arm fades in
  wide    establishing 3/4: robot base, the first object and its destination in frame; a slow push-in
  grasp   close-up on the first pick (real time)
  place   close-up on the first placement (real time)
  ff      the remaining moves in the wide, fast-forwarded (speed shown on the labelled version)
  last    close-up on the last placement (real time)
  hold    locked-off wide on the result
  catch   (no safe plan) close-up at the contact, then the frozen moment, locked off
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

FPS = 30
T_TASK0 = 1.3 + 1.2 + 0.8 + 4.0           # exec_tables.render_clip: task starts after photo, crossfade, hold, swing
W, H = 1920, 1080
SENSOR_W = 36.0


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def sph(c, d, az, el):
    a, e = math.radians(az), math.radians(el)
    return np.asarray(c, float) + d * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def project(cam, look, lens, P):
    f = np.asarray(look, float) - cam
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    v = np.asarray(P, float) - cam
    z = v @ f
    fx = lens / SENSOR_W * W
    return np.c_[(v @ r) / z * fx + W / 2, -(v @ u) / z * fx + H / 2], z


def fit(points, az, el, lens, look=None, margin=(0.10, 0.12), center=None):
    """Distance (and look-at) so every point is inside the frame with the given margins (fractions)."""
    P = np.asarray(points, float)
    look = np.asarray(look if look is not None else (P.min(0) + P.max(0)) / 2, float)
    d = 1.0
    for _ in range(40):
        cam = sph(look, d, az, el)
        uv, z = project(cam, look, lens, P)
        if (z < 0.05).any():
            d *= 1.4
            continue
        sx = (uv[:, 0].max() - uv[:, 0].min()) / (W * (1 - 2 * margin[0]))
        sy = (uv[:, 1].max() - uv[:, 1].min()) / (H * (1 - 2 * margin[1]))
        s = max(sx, sy)
        if center is None:     # recentre the bbox
            cu, cv = (uv[:, 0].max() + uv[:, 0].min()) / 2 - W / 2, (uv[:, 1].max() + uv[:, 1].min()) / 2 - H / 2
            f = look - cam
            f /= np.linalg.norm(f)
            r = np.cross(f, [0, 0, 1.0])
            r /= np.linalg.norm(r)
            u = np.cross(r, f)
            zm = float(np.median(z))
            fx = lens / SENSOR_W * W
            look = look + r * cu / fx * zm - u * cv / fx * zm
        d *= 1 + 0.8 * (s - 1)
    return sph(look, d, az, el), look


def main():
    exp, key, cache = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    ov = json.loads(sys.argv[sys.argv.index("--json") + 1]) if "--json" in sys.argv else {}
    D = np.load(exp / f"{key}_traj.npz")
    M = np.load(exp / f"{key}_traj_model.npz")
    meta = json.loads((exp / f"{key}_traj_meta.json").read_text())
    q = np.load(cache / f"{key}_traj.npz")["q"]
    NF = D["xpos"].shape[0]
    bname = [str(x) for x in M["body_name"]]
    xpos = D["xpos"].astype(float)
    dt = meta["dt"]
    speed = meta["render"]["playback_speed"]
    sim_T = (len(q) - 1) * dt
    t_rec0 = meta["sim_time_s"] - sim_T
    i_task0 = int(round(T_TASK0 * FPS))
    frozen = D["frozen"].astype(bool) if "frozen" in D else np.zeros(NF, bool)
    i_last = int(np.nonzero(~frozen)[0].max())           # last frame with a fresh physics state

    def src_of(tr):                                      # recording time -> export frame (float)
        return min(i_task0 + tr / speed * FPS, i_last)

    ev = [dict(e, tr=e["t"] - t_rec0) for e in meta["events"]]
    acts = meta["actions"]
    grasps = [e for e in ev if e["kind"] == "grasp"]
    rels = [e for e in ev if e["kind"] == "release"]
    catch = meta.get("catch")
    hand = bname.index("hand")
    bid = lambda n: bname.index(n)
    P0 = xpos[i_task0]
    # ---------------------------------------------------------------- key points (first action)
    a0 = acts[0]
    obj0, dst0 = a0["obj"], a0["dest"]
    p_obj0, p_dst0 = P0[bid(obj0)], P0[bid(dst0)]
    base = np.array([0.0, 0.0, 0.0])
    # arm envelope over the task (so the arm stays in frame in the wide)
    arm_ids = [bid(n) for n in ("link3", "link4", "link5", "link6", "hand")]
    arm_pts = xpos[i_task0:i_last + 1:6][:, arm_ids].reshape(-1, 3)
    objs_moved = sorted({a["obj"] for a in acts} | {a["dest"] for a in acts})
    key_pts = [P0[bid(n)] for n in objs_moved] + [base + [0.1, 0.1, -0.05], base + [-0.1, -0.1, -0.05]]
    sc = json.loads((cache / f"{key}_scene.json").read_text())
    (tcx, tcy), (tsx, tsy) = sc["table"]["center"], sc["table"]["size"]
    key_pts += [[tcx + a * tsx / 2 * 0.92, tcy + b_ * tsy / 2 * 0.92, 0.0] for a in (-1, 1) for b_ in (-1, 1)]
    wide_pts = np.r_[np.asarray(key_pts), arm_pts]
    AZ = ov.get("az", meta.get("hero_cam", {}).get("az", -110.0))
    EL = ov.get("el", 30.0)
    LENS_W = ov.get("lens_wide", 35.0)
    cam_w, look_w = fit(wide_pts, AZ, EL, LENS_W, margin=tuple(ov.get("margin_wide", (0.06, 0.08))))
    push = ov.get("push", 0.06)                           # the wide slowly pushes in by 6 % of its distance
    cam_w_end = look_w + (cam_w - look_w) * (1 - push)
    # photo camera (the exported MuJoCo photo-camera frames: C, f, up, fy, off)
    c0 = D["cam"][0]
    C_ph, f_ph = c0[0:3], c0[3:6]
    look_ph = C_ph + f_ph * 1.0
    lens_ph = c0[9] / H * 24.0 * SENSOR_W / (24.0 * W / H)   # fy px -> mm on a 36 mm-wide sensor
    off_ph = c0[10:12]
    # ---------------------------------------------------------------- close-ups
    LENS_C = ov.get("lens_close", 60.0)
    AZ_C = ov.get("az_close", AZ + 28)
    EL_C = ov.get("el_close", 24.0)

    def close_cam(p, extra=(), az=AZ_C, el=EL_C, span=0.30):
        pts = [p + [dx, dy, dz] for dx in (-span / 2, span / 2) for dy in (-span / 2, span / 2) for dz in (0, span * 0.8)]
        pts += list(extra)
        cam, look = fit(pts, az, el, LENS_C, look=p + [0, 0, 0.06], margin=(0.12, 0.14), center=True)
        return cam, look

    out = []

    def add(n, src, cam, look, lens, focus, fade=1.0, shot="", speed_=1.0, off=(0.0, 0.0)):
        for k in range(n):
            s = src(k) if callable(src) else src
            c = cam(k) if callable(cam) else cam
            lk = look(k) if callable(look) else look
            ln = lens(k) if callable(lens) else lens
            fc = focus(k) if callable(focus) else focus
            fd = fade(k) if callable(fade) else fade
            of = off(k) if callable(off) else off
            out.append({"src": int(round(s)), "cam": [float(v) for v in c], "look": [float(v) for v in lk],
                        "lens": float(ln), "focus": [float(v) for v in fc], "fade": float(fd), "shot": shot,
                        "speed": float(speed_), "off": [float(v) for v in of]})

    hand_at = lambda s: xpos[int(round(min(s, NF - 1))), hand]
    # 1) photo camera, static (the assembly crossfades the photo into this)
    add(int(ov.get("t_photo", 0.6) * FPS), 0, C_ph, look_ph, lens_ph, P0[bid(obj0)], 0.0, "photo", 0, off_ph)
    # 2) eased move photo -> establishing wide, arm fades in over the second half
    nm = int(ov.get("t_move", 3.2) * FPS)
    lerp = lambda a, b, x: np.asarray(a) + (np.asarray(b) - np.asarray(a)) * x
    add(nm, i_task0, lambda k: lerp(C_ph, cam_w, ease(k / (nm - 1))), lambda k: lerp(look_ph, look_w, ease(k / (nm - 1))),
        lambda k: lens_ph + (LENS_W - lens_ph) * ease(k / (nm - 1)), lambda k: lerp(p_obj0, look_w, ease(k / (nm - 1))),
        lambda k: ease((k / (nm - 1) - 0.4) / 0.5), "move", 0, lambda k: np.asarray(off_ph) * (1 - ease(k / (nm - 1))))
    # a beat on the establishing view before the robot moves (static scene, the push continues)
    n_pre = int(ov.get("t_pre", 1.0) * FPS)
    t_w = [0.0]
    total_pre = n_pre
    # timeline of the task in recording time
    T_end_rec = (i_last - i_task0) / FPS * speed
    if catch is None:
        g0 = grasps[0]["tr"] if grasps else 1.0
        r0 = rels[0]["tr"] if rels else g0 + 2
        segs = []                                          # (shot, tr0, tr1, playback speed)
        segs.append(("wide", 0.0, max(0.0, g0 - 0.9), 1.2))
        segs.append(("grasp", max(0.0, g0 - 0.9), g0 + 1.0, 1.0))
        segs.append(("wide", g0 + 1.0, max(g0 + 1.0, r0 - 1.2), 1.6))
        segs.append(("place", max(g0 + 1.0, r0 - 1.2), r0 + 1.0, 1.0))
        if len(rels) > 1:
            rl = rels[-1]["tr"]
            ff = ov.get("ff_speed", 3.0)
            segs.append(("ff", r0 + 1.0, max(r0 + 1.0, rl - 1.2), ff))
            segs.append(("last", max(r0 + 1.0, rl - 1.2), rl + 1.0, 1.0))
            segs.append(("wide", rl + 1.0, T_end_rec, 1.6))
        else:
            segs.append(("wide", r0 + 1.0, T_end_rec, 1.6))
    else:
        tc = catch["t"]
        g0 = grasps[0]["tr"] if grasps else None
        segs = [("wide", 0.0, max(0.0, (g0 - 0.9) if g0 is not None else tc - 2.0), 1.2)]
        if g0 is not None and g0 < tc - 2.5:
            r0 = rels[0]["tr"] if rels and rels[0]["tr"] < tc else None
            segs.append(("grasp", max(0.0, g0 - 0.9), g0 + 1.0, 1.0))
            if r0 is not None:
                segs.append(("wide", g0 + 1.0, max(g0 + 1.0, r0 - 1.2), 1.6))
                segs.append(("place", r0 - 1.2, r0 + 1.0, 1.0))
                segs.append(("wide", r0 + 1.0, max(r0 + 1.0, tc - 2.2), 1.6))
            else:
                segs.append(("wide", g0 + 1.0, tc - 2.2, 1.6))
        segs.append(("catch", segs[-1][2], T_end_rec, 0.8))
    segs = [s for s in segs if s[2] - s[1] > 0.05]
    merged = []                                         # a cut to the wide for < 0.6 s is a jump: stay close
    for sg in segs:
        if merged and sg[0] == "wide" and (sg[2] - sg[1]) / sg[3] < 0.6 and merged[-1][0] != "wide":
            merged[-1] = (merged[-1][0], merged[-1][1], sg[2], merged[-1][3])
        else:
            merged.append(sg)
    segs = merged
    total_wide = total_pre / FPS + sum((s[2] - s[1]) / s[3] for s in segs if s[0] in ("wide", "ff"))

    def wide_cam(k_global):
        x = ease(k_global / max(total_wide * FPS, 1))
        return lerp(cam_w, cam_w_end, x)

    add(n_pre, i_task0, lambda k: wide_cam(k), look_w, LENS_W, look_w, 1.0, "wide", 0)
    t_w[0] += n_pre
    act_of = lambda tr: next((i for i, r in enumerate(rels) if tr < r["tr"] + 0.3), len(rels) - 1)
    for shot, a, b, spd in segs:
        n = max(1, int(round((b - a) / spd * FPS)))
        srcs = [src_of(a + (b - a) * k / n) for k in range(n)]
        if shot in ("wide", "ff"):
            k0 = t_w[0]
            add(n, lambda k: srcs[k], lambda k: wide_cam(k0 + k), look_w, LENS_W,
                lambda k: 0.5 * hand_at(srcs[k]) + 0.5 * look_w, 1.0, shot, spd)
            t_w[0] += n
        elif shot in ("grasp", "place", "last"):
            if shot == "grasp":
                p = P0[bid(obj0)]
                extra = []
            else:
                ia = 0 if shot == "place" else len(rels) - 1
                d = acts[min(ia, len(acts) - 1)]["dest"]
                p = xpos[int(srcs[-1]), bid(d)]
                extra = []
            cam, look = close_cam(p, extra)
            # a slow 4 % push inside the close-up
            add(n, lambda k: srcs[k], lambda k: look + (cam - look) * (1 - 0.04 * ease(k / n)), look, LENS_C,
                lambda k: 0.6 * p + 0.4 * hand_at(srcs[k]) if shot == "grasp" else p + [0, 0, 0.04], 1.0, shot, spd)
        elif shot == "catch":
            pt = np.asarray(catch.get("point") or xpos[int(srcs[-1]), bid(catch["obj"])], float)
            cam, look = close_cam(pt, az=AZ_C, el=EL_C + 6, span=0.34)
            add(n, lambda k: srcs[k], lambda k: look + (cam - look) * (1 - 0.05 * ease(k / n)), look, LENS_C,
                pt, 1.0, "catch", spd)
            # frozen moment, locked off (one render, held by the assembly)
            add(int(ov.get("t_freeze", 4.0) * FPS), srcs[-1], look + (cam - look) * 0.95, look, LENS_C, pt, 1.0,
                "freeze", 0)
    if catch is None:   # locked-off hold on the result
        cam_end = wide_cam(t_w[0])
        add(int(ov.get("t_hold", 2.5) * FPS), out[-1]["src"], cam_end, look_w, LENS_W, look_w, 1.0, "hold", 0)
    info = {"key": key, "frames": len(out), "seconds": round(len(out) / FPS, 2), "speed_clip": speed,
            "segments": [(s, round(a, 2), round(b, 2), sp) for s, a, b, sp in segs], "az": AZ, "el": EL,
            "unique_renders_est": len({(o["src"], tuple(np.round(o["cam"], 4)), o["fade"]) for o in out})}
    (exp / f"{key}_shots.json").write_text(json.dumps({"info": info, "frames": out}, indent=0))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
