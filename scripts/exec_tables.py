"""Hero clips: the Franka arm actually doing the task on the AI-generated table scenes, in the photo-skinned twin.

For each scene NN (scenes/t6_NN.json, compiled goals in out/tables_v6/NN.json; no model calls anywhere):
  1. plan   - rehearsal.search_planner.search on the shipped rigid model (goals + implicit undisturbed()),
              then rehearsal.executor.run_plan with a Sim subclass that records qpos every 2 physics steps.
              Physics decides: the clip shows whatever happened, and the log says whether the goals held.
              If the search finds nothing, the straight default plan is executed anyway (a "caught failure").
  2. render - scripts/photo_skin.dress builds the dressed model (photo decals + flex cloth); each recorded qpos
              is copied into it (same first nq entries) and the cloth is stepped with the rigid bodies pinned.
              Timeline: photo -> crossfade -> twin through the photo's camera (arm hidden) -> camera swings to a
              3/4 view while the arm fades in -> task (slow orbit) -> hold on the result.
Outputs (footage/tables_sim/hero/exec/): NN.mp4 (clean), NN_labeled.mp4 (captions), NN_still.png, NN.json,
exec_log.json, plus _cache/NN_traj.npz.

Run:   .venv/bin/python scripts/exec_tables.py [--jobs 6] [--force] [NN ...]
       (no NN = the default candidate list; --force re-plans even if the cached run matches the scene file)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import dress_style  # noqa: E402

# DRESS_STYLE (scripts/dress_style.py): dark (default) -> hero/exec_dark/, primitive -> hero/exec_primitive/,
# real -> hero/exec/ (as before).  Plans and trajectories always come from (and go to) hero/exec/_cache.
OUT = ROOT.parent / "footage" / "tables_sim" / "hero" / ("exec" + dress_style.suffix())
CACHE = ROOT.parent / "footage" / "tables_sim" / "hero" / "exec" / "_cache"
DEFAULT = ["43", "03", "01", "06", "09", "46", "28", "27", "24", "38", "40"]
W, H, FPS = 1920, 1080, 30
REC_EVERY = 2                      # physics steps per recorded sample (timestep 2 ms -> 250 Hz)


def ffmpeg_exe() -> str:
    from shutil import which
    if which("ffmpeg"):
        return which("ffmpeg")
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def scene_path(nn):
    return ROOT / "scenes" / f"t6_{nn}.json"


def scene_hash(nn) -> str:
    h = hashlib.sha1(scene_path(nn).read_bytes())
    h.update((ROOT / "out" / "tables_v6" / f"{nn}.json").read_bytes())
    return h.hexdigest()[:12]


# ====================================================================== 1. plan + execute (rigid model)
def plan_and_run(nn: str, force=False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cj, cz = CACHE / f"{nn}_run.json", CACHE / f"{nn}_traj.npz"
    hsh = scene_hash(nn)
    if not force and cj.exists() and cz.exists():
        rec = json.loads(cj.read_text())
        if rec.get("scene_hash") == hsh or (os.environ.get("EXEC_USE_CACHE") and (CACHE / f"{nn}_scene.json").exists()):
            return rec

    from rehearsal import executor
    from rehearsal.predicates import with_implicit_goals
    from rehearsal.search_planner import assignments_from_goals, search
    from rehearsal.sim import Sim

    sc = json.loads(scene_path(nn).read_text())
    (CACHE / f"{nn}_scene.json").write_text(json.dumps(sc))      # the exact scene this run used (for re-renders)
    comp = json.loads((ROOT / "out" / "tables_v6" / f"{nn}.json").read_text())
    rec = {"scene": nn, "scene_hash": hsh, "instruction": comp.get("instruction"),
           "intent_status": comp["intent"]["status"]}
    if comp["intent"]["status"] != "goals":
        rec.update(planned=False, executed=False, note="intent is not 'goals' (clarify/refuse): nothing to execute")
        cj.write_text(json.dumps(rec, indent=1))
        return rec
    goals = with_implicit_goals(comp["intent"]["goals"], [o["id"] for o in sc["objects"]])
    assigns = assignments_from_goals(goals)
    t0 = time.time()
    res = search(sc, goals, assigns, sim_budget=300, time_budget_s=float(os.environ.get("EXEC_TIME_BUDGET", 900)))
    rec["search"] = {"success": res["success"], "reason": res.get("reason"), "stats": res.get("stats"),
                     "wall_s": round(time.time() - t0, 1)}
    plan = res["plan"]
    rec["planned"] = plan is not None
    if plan is None:   # execute the straight default plan anyway: the rehearsal shows the failure
        plan = {"diagnosis": "search found no plan; straight default plan",
                "actions": [dict(a, transit_height=0.3, via=[]) for a in assigns]}
    rec["plan"] = plan

    class RecSim(Sim):
        last = None

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.qrec = [self.d.qpos.copy()]
            self.nstep = 0
            RecSim.last = self

        def _step(self):
            try:
                super()._step()
            finally:
                self.nstep += 1
                if self.nstep % REC_EVERY == 0:
                    self.qrec.append(self.d.qpos.copy())

    orig = executor.Sim
    executor.Sim = RecSim
    t1 = time.time()
    try:
        r = executor.run_plan(sc, plan, goals)
    finally:
        executor.Sim = orig
    sim = RecSim.last
    rec["sim_wall_s"] = round(time.time() - t1, 1)
    rec["sim_time_s"] = r["sim_time"]
    rec["executed"] = True
    rec["success"] = bool(r["success"])
    rec["execution_error"] = (r["execution_error"] or {}).get("error")
    rec["goals"] = [{"pred": g["pred"], "args": g["args"], "ok": g["ok"], "why": g["why"],
                     "implicit": bool(g.get("implicit"))} for g in r["goals"]]
    explicit = [g for g in rec["goals"] if not g["implicit"]]
    rec["goals_held"] = f"{sum(g['ok'] for g in explicit)}/{len(explicit)}"
    rec["undisturbed_ok"] = all(g["ok"] for g in rec["goals"] if g["implicit"])
    rec["disturbed"] = r["disturbed"]
    rec["events"] = [e for e in r["events"] if e["kind"] in ("grasp", "release", "protective_stop",
                                                            "ik_failure", "grasp_miss")]
    rec["dt"] = float(sim.m.opt.timestep) * REC_EVERY
    rec["nq_rigid"] = int(sim.m.nq)
    np.savez_compressed(cz, q=np.asarray(sim.qrec, np.float32))
    cj.write_text(json.dumps(rec, indent=1, default=float))
    return rec


# ====================================================================== 2. render
def _ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def _sph(lookat, dist, az, el):
    a, e = np.radians(az), np.radians(el)
    return np.asarray(lookat) + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


class Cam:
    """Camera pose as (lookat, dist, az, el, roll, fy, principal-point offset) so two poses interpolate smoothly."""

    def __init__(self, lookat, dist, az, el, roll=0.0, fy=1500.0, off=(0.0, 0.0)):
        self.lookat, self.dist, self.az, self.el = np.asarray(lookat, float), float(dist), float(az), float(el)
        self.roll, self.fy, self.off = float(roll), float(fy), np.asarray(off, float)

    @staticmethod
    def from_photo(sc):
        c = sc["photo"]["camera"]
        K, R, t = (np.asarray(c[k], float) for k in ("K", "R", "t"))
        C = -R.T @ t
        f = R[2]
        z0 = 0.03
        s = (z0 - C[2]) / f[2]
        look = C + s * f
        v = C - look
        dist = float(np.linalg.norm(v))
        az = float(np.degrees(np.arctan2(v[1], v[0])))
        el = float(np.degrees(np.arcsin(v[2] / dist)))
        up_true = -R[1]
        up0 = np.array([0, 0, 1.0]) - f[2] * f
        up0 /= np.linalg.norm(up0)
        roll = float(np.degrees(np.arctan2(np.dot(np.cross(up0, up_true), f), np.dot(up0, up_true))))
        # the photo scaled to video width and centre-cropped to 16:9
        pw, ph = sc["photo"]["width"], sc["photo"]["height"]
        k = W / pw
        cx, cy = K[0, 2] * k, K[1, 2] * k - (ph * k - H) / 2
        return Cam(look, dist, az, el, roll, K[1, 1] * k, (cx - W / 2, cy - H / 2))

    def lerp(self, o, x):
        daz = (o.az - self.az + 180) % 360 - 180
        return Cam(self.lookat + (o.lookat - self.lookat) * x, self.dist + (o.dist - self.dist) * x,
                   self.az + daz * x, self.el + (o.el - self.el) * x, self.roll + (o.roll - self.roll) * x,
                   self.fy + (o.fy - self.fy) * x, self.off + (o.off - self.off) * x)

    def frame(self):
        C = _sph(self.lookat, self.dist, self.az, self.el)
        f = self.lookat - C
        f /= np.linalg.norm(f)
        up = np.array([0, 0, 1.0]) - f[2] * f
        up /= np.linalg.norm(up)
        r = np.radians(self.roll)
        up = up * np.cos(r) + np.cross(f, up) * np.sin(r)
        return C, f, up


EL = 48.0                          # elevation of the 3/4 views (deg)
VFOV = 38.0                        # vertical field of view of the 3/4 views (deg)
SAFE = (0.04, 0.96, 0.035, 0.85)    # x0, x1, y0, y1 (fractions): the arm + table stay inside; bottom band = captions


def project(cam: Cam, P):
    """World points (N,3) -> pixel coords (N,2) and depth (N,)."""
    C, f, up = cam.frame()
    right = np.cross(f, up)
    v = np.asarray(P, float) - C
    z = v @ f
    x = (v @ right) / z * cam.fy + W / 2 + cam.off[0]
    y = -(v @ up) / z * cam.fy + H / 2 + cam.off[1]
    return np.stack([x, y], 1), z


def fit_cam(pts, az, el, fy=None, safe=SAFE, margin=1.0, nlow=0):
    """Camera at (az, el) that puts every point inside the safe box (centred in it, as large as possible). The last
    `nlow` points (the foot of the robot's stand) may also use the caption band, down to 97% of the height."""
    fy = fy or (H / 2) / np.tan(np.radians(VFOV / 2))
    pts = np.asarray(pts, float)
    look = (pts.min(0) + pts.max(0)) / 2
    ext = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    cam = Cam(look, 1.2 * ext + 0.3, az, el, 0.0, fy)
    x0, x1, y0, y1 = safe[0] * W, safe[1] * W, safe[2] * H, safe[3] * H
    for _ in range(12):
        uv, z = project(cam, pts)
        if (z <= 0.05).any():
            cam.dist *= 1.5
            continue
        if nlow:
            uv = uv.copy()
            uv[-nlow:, 1] -= (0.97 - safe[3]) * H
        bx0, by0 = uv.min(0)
        bx1, by1 = uv.max(0)
        # recentre: move lookat so the bbox centre lands on the safe-box centre
        C, f, up = cam.frame()
        right = np.cross(f, up)
        dz = float(np.median(z))
        du = ((bx0 + bx1) / 2 - (x0 + x1) / 2) / cam.fy * dz
        dv = ((by0 + by1) / 2 - (y0 + y1) / 2) / cam.fy * dz
        cam.lookat = cam.lookat + right * du - up * dv
        # rescale distance to fill the safe box
        s = max((bx1 - bx0) / (x1 - x0), (by1 - by0) / (y1 - y0))
        cam.dist = max(0.3, cam.dist * (1 + 0.85 * (s - 1)))
    cam.dist *= margin
    return cam


def scene_points(sh, d, q, nq1, sc, every=25):
    """Points the 3/4 view must contain: the arm (padded) over the whole recorded run, and the table top."""
    import mujoco
    m = sh.m
    arm = [b for b in range(m.nbody) if m.body(b).name in
           ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger",
            "right_finger")]
    keep = d.qpos.copy()
    P = []
    pad = np.array([[dx, dy, dz] for dx in (-.05, .05) for dy in (-.05, .05) for dz in (-.05, .06)])
    for k in range(0, len(q), every):
        d.qpos[:nq1] = q[k]
        mujoco.mj_kinematics(m, d)
        for b in arm:
            P.extend(d.xpos[b] + pad)
    d.qpos[:] = keep
    mujoco.mj_forward(m, d)
    (cx, cy), (sx, sy) = sc["table"]["center"], sc["table"]["size"]
    P += [[cx + a * sx / 2, cy + b * sy / 2, 0.0] for a in (-1, 1) for b in (-1, 1)]
    P += list(pedestal_foot(sh, 1.0))
    return np.asarray(P)


def pedestal_foot(sh, shrink=0.8):
    """Points around the foot of the robot's stand (where it meets the floor)."""
    m = sh.m
    P = []
    for g in sh.pedestal:
        p, s = m.geom_pos[g], m.geom_size[g]
        z = p[2] - s[2] + 0.004
        P += [[p[0] + a * s[0] * shrink, p[1] + b * s[1] * shrink, z] for a in (-1, 0, 1) for b in (-1, 0, 1)]
    return np.asarray(P).reshape(-1, 3)


# per-scene candidate azimuths for the 3/4 view (the hero kitchen has a back wall at +x: look from the robot side)
HERO_AZS = {"t6_06": (-125,)}


def pick_hero(sh, d, sc, bodies, pts, drift_az=10.0):
    """3/4 views around the table, each framed so the whole arm (over the whole run) and the table fit; keep the one
    that sees most of `bodies` (the objects the task moves, and whatever it hits)."""
    import mujoco
    m = sh.m
    bid = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies} - {-1}
    geoms = np.array([m.geom_bodyid[g] in bid for g in range(m.ngeom)] + [False])
    best, best_s = None, -1
    import hero_dress
    for az in hero_dress.hero_azs(sc) or HERO_AZS.get(sc.get("name"), (-110, -80, -55, -30, -5, 20, 45, 70, 100)):
        # fit for the whole slow orbit (start, middle, end of the drift)
        nlow = len(pedestal_foot(sh, 1.0))
        cams = [fit_cam(pts, az + o, EL, nlow=nlow) for o in (0, drift_az / 2, drift_az)]
        c = max(cams, key=lambda c: c.dist)
        c = Cam(cams[1].lookat, c.dist * 1.03, az, EL, 0.0, c.fy)
        seg = sh.segment(d, c)
        ids = seg[..., 0].astype(int)
        ok = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        score = int(geoms[np.where(ok, ids, m.ngeom)].sum()) * (1.0 - 0.25 * abs(az + 30) / 140)
        # the stand must visibly reach the floor (a stand hidden behind the table looks like it floats)
        foot = pedestal_foot(sh, 0.8)
        if len(foot):
            uv, z = project(c, foot)
            u = np.clip(uv[:, 0].round().astype(int), 0, W - 1)
            v = np.clip(uv[:, 1].round().astype(int), 0, H - 1)
            ped = np.zeros(m.ngeom + 1, bool)
            ped[sh.pedestal] = True
            vis = ok[v, u] & ped[np.where(ok[v, u], ids[v, u], m.ngeom)] & (z > 0)
            score *= 0.25 + 0.75 * float(vis.mean())
        if score > best_s:
            best, best_s = c, score
    return best


class Shooter:
    def __init__(self, m, sc, ss=1):
        import mujoco
        self.mj, self.m, self.sc, self.ss = mujoco, m, sc, ss
        m.vis.global_.offwidth = max(W * ss, m.vis.global_.offwidth)
        m.vis.global_.offheight = max(H * ss, m.vis.global_.offheight)
        m.vis.map.znear = 0.01
        self.r = mujoco.Renderer(m, H * ss, W * ss)
        self.cam = mujoco.MjvCamera()
        self.opt_arm, self.opt_noarm = mujoco.MjvOption(), mujoco.MjvOption()
        self.opt_noarm.geomgroup[2] = 0
        x0 = sc["table"]["center"][0] - sc["table"]["size"][0] / 2
        G = mujoco.mjtGeom
        self.pedestal = [i for i in range(m.ngeom) if m.geom_bodyid[i] == 0 and m.geom_type[i] == G.mjGEOM_BOX
                         and m.geom_pos[i][0] + m.geom_size[i][0] < x0 + 0.05]
        self.ped_alpha = m.geom_rgba[self.pedestal, 3].copy() if self.pedestal else None

    def _raw(self, d, cam: Cam, arm=True):
        m, ss = self.m, self.ss
        if not arm and self.pedestal:
            m.geom_rgba[self.pedestal, 3] = 0
        self.r.update_scene(d, self.cam, self.opt_arm if arm else self.opt_noarm)
        if not arm and self.pedestal:
            m.geom_rgba[self.pedestal, 3] = self.ped_alpha
        C, f, up = cam.frame()
        n = self.r.scene.camera[0].frustum_near
        fx = fy = cam.fy * ss
        Ws, Hs = W * ss, H * ss
        cx, cy = (W / 2 + cam.off[0]) * ss, (H / 2 + cam.off[1]) * ss
        left, right = -cx / fx * n, (Ws - cx) / fx * n
        for gc in self.r.scene.camera:
            gc.pos[:] = C
            gc.forward[:] = f
            gc.up[:] = up
            gc.frustum_center = 0.5 * (left + right)
            gc.frustum_width = 0.5 * (right - left)
            gc.frustum_top = cy / fy * n
            gc.frustum_bottom = -(Hs - cy) / fy * n
        return self.r.render()

    def render(self, d, cam: Cam, arm=True):
        img = self._raw(d, cam, arm)
        if self.ss == 1:
            return dress_style.post(img.copy())
        from PIL import Image
        return dress_style.post(np.asarray(Image.fromarray(img).reduce(self.ss)))

    def segment(self, d, cam: Cam, arm=True):
        self.r.enable_segmentation_rendering()
        try:
            return self._raw(d, cam, arm)[::self.ss, ::self.ss].copy()
        finally:
            self.r.disable_segmentation_rendering()


def _font(size, bold=False):
    from PIL import ImageFont
    for p, idx in (("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0),
                   ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0)):
        try:
            return ImageFont.truetype(p, size, index=idx)
        except Exception:
            pass
    return ImageFont.load_default()


def caption(img, text, sub=None, alpha=1.0, color=(255, 255, 255), note=None):
    """Lower-left caption with a soft dark gradient (+ a small grey note bottom-right); returns a new array."""
    if (not text and not note) or alpha <= 0:
        return img
    from PIL import Image, ImageDraw
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    g = np.zeros((H, W, 4), np.uint8)
    ramp = np.clip((np.arange(H) - (H - 260)) / 260, 0, 1) ** 1.5
    g[..., 3] = (ramp[:, None] * 150 * alpha).astype(np.uint8)
    ov = Image.alpha_composite(ov, Image.fromarray(g))
    dr = ImageDraw.Draw(ov)
    a = int(255 * alpha)
    if text:
        y = H - 100 if sub else H - 80
        dr.text((72, y), text, font=_font(40), fill=(*color, a))
        if sub:
            dr.text((74, y + 52), sub, font=_font(26), fill=(205, 205, 210, int(a * 0.9)))
    if note:
        f = _font(20)
        w = dr.textlength(note, font=f)
        dr.text((W - 72 - w, H - 46), note, font=f, fill=(170, 170, 178, int(a * 0.85)))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def catch_card(img, title, reason, foot, alpha=1.0, side="left"):
    """The 'caught in rehearsal' card: a compact panel at the bottom, on `side`, with title, reason, footnote."""
    from PIL import Image, ImageDraw
    if alpha <= 0:
        return img
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    ft, fr, ff = _font(40, bold=True), _font(28), _font(20)
    tl = _wrap(dr, title, ft, 1100)
    lines = [ln for para in reason.split("\n") for ln in _wrap(dr, para, fr, 860)]
    pw = int(max(*(dr.textlength(x, font=ft) for x in tl + [""]), *(dr.textlength(x, font=fr) for x in lines))
             + 96)
    ph = 40 + 50 * len(tl) + 14 + 38 * len(lines) + 18 + 26 + 34
    x0 = 56 if side == "left" else W - 56 - pw
    y0 = H - 48 - ph
    dr.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=22, fill=(14, 14, 18, int(215 * alpha)),
                         outline=(235, 80, 70, int(230 * alpha)), width=3)
    y = y0 + 40
    for ln in tl:
        dr.text((x0 + 48, y), ln, font=ft, fill=(255, 112, 96, int(255 * alpha)))
        y += 50
    y += 14
    for ln in lines:
        dr.text((x0 + 48, y), ln, font=fr, fill=(240, 240, 244, int(255 * alpha)))
        y += 38
    dr.text((x0 + 48, y + 18), foot, font=ff, fill=(165, 165, 172, int(255 * alpha)))
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def _wrap(dr, text, font, width):
    out, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if dr.textlength(t, font=font) > width and cur:
            out.append(cur)
            cur = w
        else:
            cur = t
    return out + ([cur] if cur else [])


def glow_masks(mask):
    from PIL import Image, ImageFilter
    mk = Image.fromarray((mask * 255).astype(np.uint8))
    dil = mk.filter(ImageFilter.MaxFilter(9))
    edge = (np.asarray(dil) > 0) & ~mask
    soft = np.asarray(dil.filter(ImageFilter.GaussianBlur(14))).astype(np.float32) / 255
    return mask, edge, soft


def glow(img, masks, t, ring=None):
    """Red outline + glow (masks from glow_masks), strength t in [0,1]; optional ring = (x, y) contact point."""
    from PIL import Image, ImageDraw
    mask, edge, soft = masks
    if t <= 0 or (not mask.any() and ring is None):
        return img
    out = img.astype(np.float32)
    red = np.array([255, 60, 50], np.float32)
    out = out * (1 - 0.55 * t * soft[..., None]) + red * 0.55 * t * soft[..., None]
    out[mask] = out[mask] * (1 - 0.28 * t) + red * 0.28 * t
    out[edge] = out[edge] * (1 - 0.95 * t) + red * 0.95 * t
    res = np.clip(out, 0, 255).astype(np.uint8)
    if ring is not None:
        im = Image.fromarray(res)
        dr = ImageDraw.Draw(im)
        x, y = ring
        for r_, w_ in ((30 + 26 * (1 - t), 5), (14, 3)):
            dr.ellipse([x - r_, y - r_, x + r_, y + r_], outline=(255, 70, 60), width=w_)
        res = np.asarray(im)
    return res


# ---------------------------------------------------------------------------------------- plain-language failure
_ROLE = {"gripper body": "gripper", "wrist flange": "wrist", "suction tip/finger": "suction tip"}


_POS = {"left", "right", "middle", "center", "centre", "top", "bottom", "front", "back", "rear", "upper", "lower",
        "first", "second", "third"}
_ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def _plural(w):
    return w + ("es" if w.endswith(("s", "x", "ch", "sh")) else "s")


def plain_names(sc, acts) -> dict:
    """id -> plain noun phrase with its article, for captions and catch cards: 'the second egg' (not 'brown egg
    middle'), 'the blue bucket', 'one of the bolts'. Position words and numbers are dropped; when objects of one
    category are then indistinguishable, the ones the plan moves are named by the order it moves them."""
    import re
    clean = {}
    for o in sc["objects"]:
        t = (o.get("label") or o["id"].replace("_", " ")).strip().lower()
        t = re.sub(r"\s+(on top|under \w+|in \w+)$", "", t) if len(t.split()) > 3 else t
        w = [x for x in t.split() if x not in _POS and not x.isdigit()]
        clean[o["id"]] = " ".join(w) or t
    cat = {o["id"]: (o.get("category") or clean[o["id"]]).strip().lower() for o in sc["objects"]}
    order = []
    for a in acts:
        if a["obj"] not in order:
            order.append(a["obj"])
    out = {}
    for o in sc["objects"]:
        i = o["id"]
        same = [j for j in cat if cat[j] == cat[i]]
        if len(same) == 1 or sum(clean[j] == clean[i] for j in same) == 1:
            out[i] = "the " + clean[i]
            continue
        moved = [j for j in order if j in same]
        if i in moved and len(moved) > 1:
            out[i] = f"the {_ORD[min(moved.index(i), 9)]} {cat[i]}"
        elif i in moved:
            out[i] = f"the {cat[i]}"
        else:
            out[i] = f"one of the {_plural(cat[i])}"
    return out


def _phase_text(phase: str, lab) -> str:
    w = phase.split()
    try:
        if w[0] == "descend":
            return f"reaching down for {lab(w[2])}"
        if w[0] == "lower":
            return f"lowering {lab(w[1])} {'into' if w[2] == 'in' else 'onto'} {lab(w[3])}"
        if w[0] == "carry":
            return f"carrying {lab(w[1])}" + (f" to {lab(w[3])}" if w[2] == "to" else "")
        if w[0] == "lift":
            return f"lifting {lab(w[1])}"
        if w[0] == "approach":
            return f"moving over {lab(w[1])}"
        if w[0] == "retreat":
            return f"pulling back from {lab(w[2])}"
        if w[0] == "grasp":
            return f"picking up {lab(w[1])}"
    except IndexError:
        pass
    return "moving"


def _step_txt(a, names):
    rel = {"in": "into", "inside": "into", "on": "onto", "stacked": "onto"}.get(a["relation"], a["relation"])
    o, dst = names.get(a["obj"], a["obj"]), names.get(a["dest"], a["dest"])
    return f"{o[4:] if o.startswith('the ') else o} {rel} {dst}"


def failure_story(rec, sc, acts, t_rec0, sim_T):
    """-> dict(t, obj, reason, step, point) for a failed run: when it fails (recording time), which object to
    highlight, one plain sentence (no ids, no coordinates), the failing step and an optional 3D contact point."""
    names = plain_names(sc, acts)
    lab = lambda i: names.get(i, "the " + str(i).replace("_", " "))
    ev = next((e for e in rec.get("events", []) if e["kind"] in ("protective_stop", "ik_failure", "grasp_miss")),
              None)
    step = None
    if rec.get("execution_error"):
        import re
        mm = re.match(r"action (\d+)", rec["execution_error"])
        step = int(mm.group(1)) if mm else None
    if ev is not None:
        t = ev["t"] - t_rec0
        if ev["kind"] == "protective_stop":
            role = _ROLE.get(ev.get("link_role", ""), ev.get("link_role", "arm"))
            held = ev.get("while_holding")
            hitter = f"{lab(held)} it carries" if held and ev.get("link") is None else f"the robot's {role}"
            ph_ = _phase_text(ev.get('phase', ''), lab)
            if ph_.endswith(" " + lab(ev["obj"])):          # "... into the bucket while lowering the sponge into it"
                ph_ = ph_[:-len(lab(ev["obj"]))] + "it"
            reason = f"drive {hitter} into {lab(ev['obj'])} ({ev.get('force_n', 0):.0f} N) while {ph_}."
            return {"t": t, "obj": ev["obj"], "reason": reason, "step": step, "point": ev.get("at_xyz")}
        a = acts[step] if step is not None and step < len(acts) else (acts[0] if acts else {})
        if ev["kind"] == "ik_failure":
            obj = a.get("obj")
            r = ev.get("horizontal_dist_from_base")
            phase = ev.get("phase", "")
            target = lab(a.get("dest")) if phase.startswith(("carry", "lower")) else lab(obj)
            why = ("it is too close to the robot's base" if r is not None and r < 0.33 else
                   "it is too far from the robot" if r is not None and r > 0.75 else "no arm pose gets there")
            reason = f"need the arm to reach {target} from above, but {why}."
            return {"t": t, "obj": a.get("dest") if target != lab(obj) else obj, "reason": reason, "step": step,
                    "point": None}
        obj = ev.get("obj") or a.get("obj")
        return {"t": t, "obj": obj, "reason": f"miss the top of {lab(obj)} with the suction cup, so it could not pick it up.",
                "step": step, "point": None}
    # no execution error: the plan ran, but physics says a goal does not hold at the end
    g = next((g for g in rec.get("goals", []) if not g["ok"] and not g.get("implicit")), None) or \
        next((g for g in rec.get("goals", []) if not g["ok"]), None)
    if g is None:
        return {"t": sim_T, "obj": None, "reason": "end with a goal not met.", "step": None, "point": None}
    if g["pred"] in ("inside", "on"):
        o, dst = g["args"][0], g["args"][1]
        import re
        prep = "in" if g["pred"] == "inside" else "on"
        mx = re.search(r"xy offset ([\d.]+) cm \(limit ([\d.]+)\)", g.get("why", ""))
        mz = re.search(r"bottom z (-?[\d.]+) cm", g.get("why", ""))
        if mx and float(mx.group(1)) > float(mx.group(2)):
            reason = f"leave {lab(o)} {float(mx.group(1)):.0f} cm away from {lab(dst)}."
        elif mz and float(mz.group(1)) > 1.5:
            reason = (f"leave {lab(o)} {float(mz.group(1)):.0f} cm too high, resting on something else instead "
                      f"of {'inside' if prep == 'in' else 'on'} {lab(dst)}.")
        else:
            reason = f"not keep {lab(o)} {prep} {lab(dst)}."
        st = next((k for k, a in enumerate(acts) if a["obj"] == o), None)
        return {"t": sim_T, "obj": o, "reason": reason, "step": st, "point": None}
    dist = [x for x in rec.get("disturbed", []) if not x.get("placed_by_robot")]
    o = dist[0]["obj"] if dist else g["args"][0]
    return {"t": sim_T, "obj": o, "reason": f"knock {lab(o)} out of place.", "step": None,
            "point": None}


def with_sky(base_spec):
    """Gradient sky (no hard black wedge where the floor ends)."""
    import mujoco

    def bs(scene):
        spec = base_spec(scene)
        spec.add_texture(name="exec_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.22, 0.23, 0.27],
                         rgb2=[0.03, 0.03, 0.04], width=512, height=3072)
        # a faint tiled floor, so the floor reads as a floor and the robot's stand visibly sits on it
        spec.add_texture(name="exec_floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.265, 0.265, 0.29],
                         rgb2=[0.245, 0.245, 0.27], width=512, height=512)
        mat = spec.add_material(name="exec_floor_mat", texrepeat=[1.0, 1.0], texuniform=True, reflectance=0.0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "exec_floor"
        g = spec.worldbody.first_geom()
        while g is not None:
            if g.type == mujoco.mjtGeom.mjGEOM_PLANE and g.pos[2] < -0.5:
                g.material, g.rgba = "exec_floor_mat", [1, 1, 1, 1]
            g = spec.worldbody.next_geom(g)
        return spec
    return bs


def sam3d_base_spec(sc, base_spec):
    """If scripts/sam3d_mj.py and assets/sam3d/<scene name>/*.glb exist, wrap base_spec so the SAM 3D meshes are
    added (visual only, primitives hidden) to every non-cloth object that has one. -> (base_spec, ids)."""
    if os.environ.get("EXEC_NO_SAM3D"):
        return base_spec, set()
    d = ROOT / "assets" / "sam3d" / sc["name"]
    if not (ROOT / "scripts" / "sam3d_mj.py").exists() or not d.is_dir() or not list(d.glob("*.glb")):
        return base_spec, set()
    try:
        from sam3d_mj import add_sam3d_visuals
    except Exception as e:
        print("sam3d_mj import failed, using photo decals:", e, flush=True)
        return base_spec, set()
    from photo_skin import is_cloth
    sub = dict(sc, objects=[o for o in sc["objects"] if not is_cloth(o)])   # cloth keeps its flex sheet
    ids = set()                     # filled with the ids that really got a mesh (dress calls bs before decals)

    def bs(scene):
        spec = base_spec(scene)
        ids.update(add_sam3d_visuals(spec, sub, hide_primitive=True) or [])
        return spec
    return bs, ids


# SAM 3D textures whose black plastic came out with a warm (gold/olive) cast from the photo's lighting
DARK_PLASTIC = {"t6_43": ("pet_brush_1", "pet_brush_2")}


def neutral_dark_plastic(m, sc):
    """Pull dark, warm-tinted texels of the listed SAM 3D textures to neutral grey (matte black plastic)."""
    import mujoco
    for oid in DARK_PLASTIC.get(sc["name"], ()):
        t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, f"sam3d_{oid}_tex")
        if t < 0:
            continue
        a, h, w, c = m.tex_adr[t], m.tex_height[t], m.tex_width[t], m.tex_nchannel[t]
        px = m.tex_data[a:a + h * w * c].reshape(-1, c)[:, :3].astype(float)
        L = px @ [0.299, 0.587, 0.114]
        warm = (px[:, 0] + px[:, 1]) / 2 - px[:, 2]                 # yellow/olive/gold minus blue
        k = np.clip((150 - L) / 70, 0, 1) * np.clip(warm / 30, 0, 1)  # dark AND warm
        grey = np.repeat(np.minimum(L, 70)[:, None] * 0.8, 3, 1)
        px = px * (1 - k[:, None]) + grey * k[:, None]
        out = m.tex_data[a:a + h * w * c].reshape(-1, c)
        out[:, :3] = np.clip(px, 0, 255).astype(np.uint8)


# photo decals that read as holes or smears in the sim (flat things the photo can't shape): drawn invisible,
# physics unchanged; the table texture underneath still shows them flat
HIDE_VISUAL = {"t6_43": ("dog_collar_1", "leash_1", "hair_tie_1")}


def hide_visual(m, sc):
    import mujoco
    for oid in HIDE_VISUAL.get(sc["name"], ()):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, oid)
        for g in range(m.ngeom):
            if b >= 0 and m.geom_bodyid[g] == b:
                m.geom_rgba[g][3] = 0.0


def flat_fix(m, sc, sam_ids):
    """Thin flat items (phones, tablets) drawn with photo decals: the dark side-face decals merge with the dark top
    into one tall-looking slab. Keep only the top decal and give the edge a lighter frame colour so it reads flat."""
    import re
    import mujoco
    for o in sc["objects"]:
        if o["id"] in sam_ids or not re.search(r"phone|tablet", (o.get("category", "") + " " + o.get("label", ""))
                                                .lower()):
            continue
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["id"])
        if b < 0:
            continue
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_rgba[g][3] == 0:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                m.geom_rgba[g] = [0.48, 0.49, 0.52, 1.0]
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, m.geom_quat[g])
                Vb = V @ R.reshape(3, 3).T
                ext = Vb.max(0) - Vb.min(0)
                if ext[2] > 0.5 * max(ext[0], ext[1]):     # a side face: hide it
                    m.geom_rgba[g][3] = 0.0


def render_clip(nn: str, rec: dict) -> dict:
    import mujoco
    from PIL import Image

    from photo_skin import dress
    from render_tables import base_spec, finish, settle

    hq = bool(os.environ.get("EXEC_HQ"))
    ss = 2 if hq else 1

    def fin(spec):
        m_ = finish(spec)
        # shadow map restricted to the scene (no jagged dark band at the top of the frame)
        m_.stat.extent, m_.vis.map.shadowclip, m_.vis.map.shadowscale = 2.5, 1.5, 0.6
        if hq:
            m_.vis.quality.offsamples = 16
            m_.vis.quality.shadowsize = 8192
        return m_

    snap = CACHE / f"{nn}_scene.json"
    sc = json.loads((snap if snap.exists() else scene_path(nn)).read_text())
    q = np.load(CACHE / f"{nn}_traj.npz")["q"].astype(float)
    nq1, dt = rec["nq_rigid"], rec["dt"]
    t0 = time.time()
    bs, sam_ids = sam3d_base_spec(sc, with_sky(base_spec))
    import hero_dress
    auto = dress_style.studio()      # studio styles: every scene gets the set dressing of the stills (no room)
    bs = hero_dress.wrap(sc, bs, auto=auto)
    bs = dress_style.wrap(sc, bs)
    import photo_skin as _ps
    if hero_dress.cloth_soft(sc):   # soft towel: a thicker sheet the towel's own size, resting on the table
        _ps.CLOTH_SOFT, _ps.CLOTH_RADIUS = True, 0.0049
    _ps.CLOTH_SHRINK = hero_dress.cloth_shrink(sc)
    import photo_skin
    orig_apply, orig_is_cloth = photo_skin.Skin.apply, photo_skin.is_cloth
    acts = rec["plan"]["actions"]
    moved = {a["obj"] for a in acts}
    # a cloth the arm picks up is drawn as a rigid photo-textured pad: a flex sheet lying loose on its hidden core
    # does not follow the suction carry (and stepping it through the carry is very slow)
    rigid = moved | set(hero_dress.not_cloth(sc))       # + set-dressing: e.g. knitting needles are sticks, not cloth
    photo_skin.is_cloth = lambda o: o["id"] not in rigid and orig_is_cloth(o)
    if sam_ids is not None:   # objects with a SAM 3D mesh get no photo decals (cloth and table texture stay)
        def apply(self, spec, hide):
            self.decals = [x for x in self.decals if x[0] not in sam_ids]
            return orig_apply(self, spec, hide)
        photo_skin.Skin.apply = apply
    try:
        m, d = dress(sc, bs, fin, settle)
    finally:
        photo_skin.Skin.apply, photo_skin.is_cloth = orig_apply, orig_is_cloth
    rec["sam3d_objects"] = sorted(sam_ids)
    for k in range(m.nmat):     # SAM 3D meshes: matte (no specular sheen / rainbow highlights)
        nm_ = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, k) or ""
        if nm_.startswith("sam3d_"):
            m.mat_specular[k], m.mat_shininess[k], m.mat_reflectance[k] = 0.0, 0.02, 0.0
    neutral_dark_plastic(m, sc)
    flat_fix(m, sc, sam_ids)
    hide_visual(m, sc)
    hero_dress.model_fix(m, sc, auto=auto)
    prim = dress_style.style() == "primitive"
    if prim:                   # the camera is picked on the dressed look (below), so dark and primitive match
        os.environ["DRESS_STYLE"] = "dark"
    try:
        dress_style.model_fix(m, sc)
    finally:
        if prim:
            os.environ["DRESS_STYLE"] = "primitive"
    assert m.nq >= nq1
    has_cloth = m.nq > nq1
    nv1 = nq1 - 4 * sum(1 for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                        and m.jnt_qposadr[j] < nq1)       # each free joint: 7 qpos, 6 qvel
    sub = max(1, int(round(dt / m.opt.timestep)))
    free = [m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    narm = int(min(free)) if free else 9                  # arm + finger joints come first (qpos == qvel there)

    def pin(k):
        d.qpos[:nq1] = q[k]
        d.qvel[:nv1] = 0
        d.ctrl[:7] = q[k][:7]

    # cloth: settle once more around the recorded start pose
    pin(0)
    mujoco.mj_forward(m, d)
    if has_cloth:
        for _ in range(150):
            mujoco.mj_step(m, d)
            pin(0)
    mujoco.mj_forward(m, d)

    sim_T = (len(q) - 1) * dt
    t_rec0 = rec["sim_time_s"] - sim_T                       # events use absolute sim time
    fail = None if rec.get("success") else failure_story(rec, sc, acts, t_rec0, sim_T)
    if fail:
        if rec.get("planned"):
            card_title, card_body = "Caught in rehearsal.", "The plan would " + fail["reason"]
            card_foot = "Found in a physics simulation of this table, before anything was sent to the robot."
        else:
            card_title = "Rehearsal doesn't send it."
            card_body = "No safe plan found.\nA direct attempt would " + fail["reason"]
            card_foot = "Shown: the direct attempt, in a physics simulation of this table."
        fail["card"] = {"title": card_title, "body": card_body, "foot": card_foot}
        rec["catch"] = fail
    # a caught failure plays only up to the moment it happens (+ a beat), then freezes on it
    play_T = min(sim_T, fail["t"] + 0.45) if fail else sim_T
    n_play = int(round(play_T / dt))

    sh = Shooter(m, sc, ss)
    DRIFT = 10.0
    pts = scene_points(sh, d, q[:n_play + 1], nq1, sc)
    focus = moved | ({fail["obj"]} if fail and fail.get("obj") else set())
    c_hero = pick_hero(sh, d, sc, focus, pts, DRIFT)
    c_photo = Cam.from_photo(sc)
    if prim:
        dress_style.primitive_fix(m, sc)
    rec["hero_cam"] = {"az": c_hero.az, "el": c_hero.el, "dist": round(c_hero.dist, 3)}

    # timeline (seconds)
    T_PHOTO, T_XF, T_HOLD, T_SWING = 1.3, 1.2, 0.8, 4.0
    speed = float(np.clip(play_T / 16.0, 1.0, 3.0))
    T_TASK = play_T / speed
    T_FREEZE, T_CARD = (1.6, 4.4) if fail else (0.0, 0.0)
    T_END = 2.4 if not fail else 0.3
    t_swing0 = T_PHOTO + T_XF + T_HOLD
    t_task0 = t_swing0 + T_SWING
    t_freeze0 = t_task0 + T_TASK
    total = t_freeze0 + T_FREEZE + T_CARD + T_END
    nframes = int(total * FPS)

    photo = Image.open(ROOT / sc["photo"]["file"]).convert("RGB")
    pw, ph = photo.size
    k = W / pw
    photo = np.asarray(photo.resize((W, int(round(ph * k))), Image.LANCZOS))
    y0 = (photo.shape[0] - H) // 2
    photo = photo[y0:y0 + H].copy()
    twin0 = sh.render(d, c_photo, arm=False)

    names = plain_names(sc, acts)
    rel = [e["t"] - t_rec0 for e in rec["events"] if e["kind"] == "release"]
    instr = rec["instruction"]
    n_move = sum(1 for g in rec["goals"] if not g["implicit"] and g["pred"] in ("inside", "on", "near", "stacked"))
    n_other = len({a for g in rec["goals"] if g["pred"] == "undisturbed" for a in g["args"]})
    if not fail:
        verdict = (f"Rehearsed in physics: {n_move} of {n_move} placed" +
                   (f", {n_other} other objects left undisturbed" if rec.get("undisturbed_ok") and n_other else ""))
        verdict_sub = f"“{instr}”"
        vcol = (150, 230, 160)
        if not rec.get("planned"):      # the search found no plan; the straight default plan happened to work
            verdict = "No safe plan found."
            verdict_sub = "A direct attempt happened to work here; Rehearsal still wouldn't send it."
            vcol = (245, 200, 120)
    else:
        rec["fail_why"] = rec.get("execution_error") or next((g["why"] for g in rec["goals"] if not g["ok"]), "")
        vcol = (255, 150, 120)
    NOTE = "MuJoCo physics simulation  ·  suction gripper"

    ff = ffmpeg_exe()
    crf = "14" if hq else "18"

    def enc(path):
        return subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                                 "-preset", "slow" if hq else "medium", "-crf", crf, "-pix_fmt", "yuv420p",
                                 "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    p_clean, p_lab = enc(OUT / f"{nn}.mp4"), enc(OUT / f"{nn}_labeled.mp4")
    if fail:
        still_t = t_freeze0 + T_FREEZE
    else:
        still_t = (rel[-1] - 0.35) / speed + t_task0 if rel else t_task0 + T_TASK * 0.6
    still, k_sim, cloth_s = None, 0, 0.0
    frozen = None                       # (base image, mask, ring) at the failure moment

    def hero_at(tt):
        drift = min(max(tt, 0.0), T_TASK + T_END) / max(T_TASK + T_END, 1e-6)
        return Cam(c_hero.lookat, c_hero.dist, c_hero.az + DRIFT * _ease(drift), c_hero.el, 0.0, c_hero.fy)

    for i in range(nframes):
        t = i / FPS
        lab = sub_txt = note = None
        lab_a = 1.0
        card = None
        if t < t_swing0:                                   # photo -> twin crossfade
            a = _ease((t - T_PHOTO) / T_XF)
            img = (photo * (1 - a) + twin0 * a).astype(np.uint8) if 0 < a < 1 else (photo if a <= 0 else twin0)
            lab = "AI-generated photo" if a < 0.5 else "3D twin, seen from the photo's camera"
            sub_txt = "textures are projected from the photo" if a >= 0.5 else None
            lab_a = min(1.0, abs(a - 0.5) * 4) if 0 < a < 1 else 1.0
        elif fail and t >= t_freeze0:                      # caught: freeze, highlight, card
            if frozen is None:
                cam = hero_at(t_freeze0 - t_task0)
                base = sh.render(d, cam, arm=True)
                mask = np.zeros((H, W), bool)
                if fail.get("obj"):
                    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, fail["obj"])
                    seg = sh.segment(d, cam)
                    ok = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
                    gb = np.append(m.geom_bodyid, -1)
                    mask = ok & (gb[np.where(ok, seg[..., 0], m.ngeom)] == bid)
                ring = None
                if fail.get("point"):
                    uv, z = project(cam, [fail["point"]])
                    if z[0] > 0:
                        ring = tuple(uv[0])
                xs = np.nonzero(mask.any(0))[0]
                cx_ = xs.mean() if len(xs) else (ring[0] if ring else W / 2)
                frozen = (base, glow_masks(mask), ring, "right" if cx_ < W / 2 else "left")
            base, mask, ring, card_side = frozen
            tf = t - t_freeze0
            img = glow(base, mask, _ease(tf / 0.5) * (0.85 + 0.15 * np.cos(tf * 5.0)), ring)
            st = fail.get("step")
            step_txt = (f"step {st + 1} of {len(acts)}: " if st is not None else "") + \
                (_step_txt(acts[st], names) if st is not None and st < len(acts) else "")
            lab, sub_txt, note = f"“{instr}”", step_txt or None, NOTE
            if tf >= T_FREEZE:
                card = _ease((tf - T_FREEZE) / 0.5)
        else:
            if t < t_task0:
                x = (t - t_swing0) / T_SWING
                cam = c_photo.lerp(hero_at(0.0), _ease(x))
                ks = 0
                fade = _ease((x - 0.5) / 0.45)
                lab, sub_txt = "Rehearsal", f"“{instr}”"
            else:
                tt = t - t_task0
                cam = hero_at(tt)
                ks = min(n_play, int(round(tt * speed / dt)))
                fade = 1.0
                note = NOTE
                if tt < T_TASK:
                    j = min(sum(r < tt * speed for r in rel), len(acts) - 1)
                    if fail and fail.get("step") is not None:
                        j = min(j, fail["step"])
                    a_ = acts[j]
                    lab = f"“{instr}”"
                    sub_txt = f"step {j + 1} of {len(acts)}: {_step_txt(a_, names)}"
                else:
                    lab, sub_txt = verdict, verdict_sub
                    lab_a = _ease((tt - T_TASK) / 0.5)
            # advance the sim state to sample ks (cloth stepped with rigid bodies pinned)
            while k_sim < ks:
                k_sim += 1
                if has_cloth and cloth_s > 300:      # safety valve: freeze the cloth rather than render for hours
                    has_cloth, rec["cloth_frozen_at_s"] = False, round(k_sim * dt, 2)
                if has_cloth:
                    tc = time.time()
                    d.qpos[:nq1] = q[k_sim]
                    for _ in range(sub):
                        d.qvel[:narm] = (q[k_sim][:narm] - q[k_sim - 1][:narm]) / dt
                        mujoco.mj_step(m, d)
                        pin(k_sim)
                    cloth_s += time.time() - tc
            pin(k_sim)
            mujoco.mj_forward(m, d)
            if fade >= 1:
                img = sh.render(d, cam, arm=True)
            elif fade <= 0:
                img = sh.render(d, cam, arm=False)
            else:
                img = (sh.render(d, cam, arm=False) * (1 - fade) + sh.render(d, cam, arm=True) * fade).astype(np.uint8)
        if still is None and t >= still_t:
            still = img
        p_clean.stdin.write(np.ascontiguousarray(img).tobytes())
        if card is not None:
            labimg = catch_card(caption(img, None, None, 1.0, note=NOTE), card_title, card_body, card_foot, card,
                                card_side)
        else:
            col = vcol if (lab == verdict if not fail else False) else (255, 255, 255)
            labimg = caption(img, lab, sub_txt, lab_a, col, note=note)
        p_lab.stdin.write(np.ascontiguousarray(labimg).tobytes())
    for p in (p_clean, p_lab):
        p.stdin.close()
        p.wait()
    if still is None:
        still = img
    Image.fromarray(still).save(OUT / f"{nn}_still.png")
    return {"render_wall_s": round(time.time() - t0, 1), "frames": nframes, "duration_s": round(total, 1),
            "playback_speed": round(speed, 2), "cloth": has_cloth, "hq": hq}


SIM_ARTIFACT = {"20": "an object ends on the floor ~75 cm below the table top; flagged in review as a sim "
                      "blow-up, not a robot mistake: excluded from clips and counts"}


def run_one(nn: str, force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        rec = plan_and_run(nn, force)
        if rec.get("executed") and not rec["plan"]["actions"]:
            rec["note"] = "plan has no actions (goals already hold or nothing to move): no clip"
            for f in (f"{nn}.mp4", f"{nn}_labeled.mp4", f"{nn}_still.png"):
                (OUT / f).unlink(missing_ok=True)
        elif nn in SIM_ARTIFACT:
            rec["sim_artifact"] = SIM_ARTIFACT[nn]
            for f in (f"{nn}.mp4", f"{nn}_labeled.mp4", f"{nn}_still.png"):
                (OUT / f).unlink(missing_ok=True)
        elif rec.get("executed") and not os.environ.get("EXEC_PLAN_ONLY"):
            rec["render"] = render_clip(nn, rec)
    except Exception as e:
        rec = {"scene": nn, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
    rec["total_wall_s"] = round(time.time() - t0, 1)
    (OUT / f"{nn}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(nn, "done", rec.get("goals_held"), rec.get("success"), rec.get("error", ""), flush=True)


def summarize(nns):
    """Merge the rows for `nns` into exec_log.json (other scenes' rows are kept)."""
    try:
        log = json.loads((OUT / "exec_log.json").read_text())
    except (FileNotFoundError, ValueError):
        log = {}
    for nn in nns:
        f = OUT / f"{nn}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        log[nn] = {k: r.get(k) for k in ("instruction", "intent_status", "planned", "executed", "success",
                                         "goals_held", "undisturbed_ok", "execution_error", "fail_why", "error",
                                         "sim_time_s", "sim_wall_s", "total_wall_s", "scene_hash")}
        log[nn]["actions"] = len((r.get("plan") or {}).get("actions", []))
        log[nn]["search"] = r.get("search")
        log[nn]["render"] = r.get("render")
        log[nn]["disturbed"] = [x["obj"] for x in r.get("disturbed") or []]
        log[nn]["clip"] = (OUT / f"{nn}.mp4").exists()
        log[nn]["sim_artifact"] = SIM_ARTIFACT.get(nn)
        # counted as done only when the search found a plan and physics verified it (38: lucky direct attempt)
        log[nn]["verified_success"] = bool(r.get("planned") and r.get("success"))
        try:   # the scene or its compiled goals changed after this run: re-run before using the clip
            log[nn]["stale"] = r.get("scene_hash") is not None and scene_hash(nn) != r.get("scene_hash")
        except FileNotFoundError:
            log[nn]["stale"] = True
    log = dict(sorted(log.items()))
    tmp = OUT / "exec_log.json.tmp"
    tmp.write_text(json.dumps(log, indent=1))
    tmp.replace(OUT / "exec_log.json")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--plan-only", action="store_true", help="plan + execute + cache, no rendering")
    ap.add_argument("--time-budget", type=float, default=900, help="search wall-time budget per scene (s)")
    ap.add_argument("--all", action="store_true", help="every scene whose compiled intent is 'goals'")
    ap.add_argument("--hq", action="store_true", help="2x supersampling, 16x MSAA, 8k shadows, crf 14 (slow)")
    a = ap.parse_args()
    if a.hq:
        os.environ["EXEC_HQ"] = "1"
    os.environ["EXEC_TIME_BUDGET"] = str(a.time_budget)
    if a.plan_only:
        os.environ["EXEC_PLAN_ONLY"] = "1"
    if a.all:
        a.scenes = [p.stem for p in sorted((ROOT / "out" / "tables_v6").glob("[0-9][0-9].json"))
                    if json.loads(p.read_text())["intent"]["status"] == "goals" and scene_path(p.stem).exists()]
    nns = a.scenes or DEFAULT
    if a.one:
        for nn in nns:
            run_one(nn, a.force)
        return
    procs, queue = [], list(nns)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    while queue or procs:
        while queue and len(procs) < a.jobs:
            nn = queue.pop(0)
            cmd = [sys.executable, __file__, "--one", nn] + (["--force"] if a.force else [])
            OUT.mkdir(parents=True, exist_ok=True)
            procs.append(subprocess.Popen(cmd, env=env, stdout=open(OUT / f"_{nn}.log", "w"),
                                          stderr=subprocess.STDOUT))
        time.sleep(2)
        procs = [p for p in procs if p.poll() is None]
    log = summarize(nns)
    for nn, r in log.items():
        print(nn, r.get("planned"), r.get("success"), r.get("goals_held"), r.get("error") or "")


if __name__ == "__main__":
    main()
