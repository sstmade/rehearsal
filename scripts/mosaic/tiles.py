"""Render low-res mosaic tile streams with MuJoCo (the same dressed model + studio look as exec_tables.py).

Per scene and style (dark, primitive): every recorded rollout (the shipped run in exec/_cache plus the simulated
candidate rollouts of rollouts.py) from several cameras, NF frames sampled over the whole rollout.

    rehearsal-mvp/.venv/bin/python scripts/mosaic/tiles.py --jobs 4            # all scenes, 128x72
    (hi-res re-renders of chosen streams: see render_streams())

Writes footage/mosaic/_work/tiles/NN_style.npy  (S, NF, h, w, 3) uint8 + NN_style.json (stream specs).
Read-only w.r.t. shipped results.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
CACHE = ROOT.parent / "footage" / "tables_sim" / "hero" / "exec" / "_cache"
DARK = ROOT.parent / "footage" / "tables_sim" / "hero" / "exec_dark"
WORK = ROOT.parent / "footage" / "mosaic" / "_work"
RO = WORK / "rollouts"
TILES = WORK / "tiles"
NF = 120
TW, TH = 128, 72
CAMS_PER = 4

# camera variants: (az offset from the hero az, elevation, margin (<1 = tighter crop), tight-on-table)
CAMSET = [(0, 48, 1.0, False), (35, 40, 0.95, False), (-35, 55, 0.95, False), (70, 32, 1.0, False),
          (-70, 62, 0.9, False), (180, 50, 1.0, False), (15, 72, 0.62, True), (-20, 38, 0.66, True),
          (110, 45, 0.8, False), (-140, 44, 0.85, False), (0, 85, 0.7, True), (50, 58, 0.6, True)]


def rollouts_for(nn):
    """[(rid, q, meta)] : rid 0 = the shipped run, k>0 = simulated candidates."""
    rec = json.loads((CACHE / f"{nn}_run.json").read_text())
    out = [(0, CACHE / f"{nn}_traj.npz", {"kind": "shipped run (exec/_cache)", "success": rec.get("success"),
                                         "nq_rigid": rec["nq_rigid"]})]
    for f in sorted(RO.glob(f"{nn}_c*.npz")):
        meta = json.loads(f.with_suffix(".json").read_text())
        if meta.get("nq_rigid") == rec["nq_rigid"]:
            out.append((int(meta["candidate"]), f, meta))
    return rec, out


def build(nn, rec, style):
    """The dressed MuJoCo model of exec_tables.render_clip (no cloth stepping; tiles are tiny)."""
    os.environ["DRESS_STYLE"] = style          # as exec_tables.render_clip: build in the style itself ...
    import mujoco

    import dress_style
    import exec_tables as ET
    import hero_dress
    import photo_skin
    from photo_skin import dress
    from render_tables import base_spec, settle

    from render_tables import finish

    def fin(spec):
        m_ = finish(spec)
        m_.stat.extent, m_.vis.map.shadowclip, m_.vis.map.shadowscale = 2.5, 1.5, 0.6
        return m_

    # decal textures go to a private per-process folder (never rewrite the shared out/tables_v6/_mj_tex cache)
    photo_skin.TEX = WORK / "mj_tex" / f"{nn}_{style}_{os.getpid()}"
    sc = json.loads((CACHE / f"{nn}_scene.json").read_text())
    bs, sam_ids = ET.sam3d_base_spec(sc, ET.with_sky(base_spec))
    bs = hero_dress.wrap(sc, bs, auto=True)
    bs = dress_style.wrap(sc, bs)
    if hero_dress.cloth_soft(sc):
        photo_skin.CLOTH_SOFT, photo_skin.CLOTH_RADIUS = True, 0.0049
    photo_skin.CLOTH_SHRINK = hero_dress.cloth_shrink(sc)
    orig_apply, orig_is_cloth = photo_skin.Skin.apply, photo_skin.is_cloth
    moved = {a["obj"] for a in rec["plan"]["actions"]}
    rigid = moved | set(hero_dress.not_cloth(sc))
    photo_skin.is_cloth = lambda o: o["id"] not in rigid and orig_is_cloth(o)
    if sam_ids is not None:
        def apply(self, spec, hide):
            self.decals = [x for x in self.decals if x[0] not in sam_ids]
            return orig_apply(self, spec, hide)
        photo_skin.Skin.apply = apply
    try:
        m, d = dress(sc, bs, fin, settle)
    finally:
        photo_skin.Skin.apply, photo_skin.is_cloth = orig_apply, orig_is_cloth
    for k in range(m.nmat):
        nm_ = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, k) or ""
        if nm_.startswith("sam3d_"):
            m.mat_specular[k], m.mat_shininess[k], m.mat_reflectance[k] = 0.0, 0.02, 0.0
    ET.neutral_dark_plastic(m, sc)
    ET.flat_fix(m, sc, sam_ids)
    ET.hide_visual(m, sc)
    hero_dress.model_fix(m, sc, auto=True)
    os.environ["DRESS_STYLE"] = "dark"         # ... model_fix with the dark look (cameras match), then primitive_fix
    try:
        dress_style.model_fix(m, sc)
    finally:
        os.environ["DRESS_STYLE"] = style
    return m, d, sc


class Tiny:
    """exec_tables.Shooter at an arbitrary size (cams are fitted at 1920x1080 and scaled)."""

    def __init__(self, m, w, h):
        import mujoco
        self.mj, self.m, self.w, self.h = mujoco, m, w, h
        m.vis.global_.offwidth = max(w, m.vis.global_.offwidth)
        m.vis.global_.offheight = max(h, m.vis.global_.offheight)
        m.vis.map.znear = 0.01
        self.r = mujoco.Renderer(m, h, w)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

    def render(self, d, cam):
        import dress_style
        k = self.w / 1920.0
        self.r.update_scene(d, self.cam, self.opt)
        C, f, up = cam.frame()
        n = self.r.scene.camera[0].frustum_near
        fx = fy = cam.fy * k
        cx, cy = (960 + cam.off[0]) * k, (540 + cam.off[1]) * k
        left, right = -cx / fx * n, (self.w - cx) / fx * n
        for gc in self.r.scene.camera:
            gc.pos[:] = C
            gc.forward[:] = f
            gc.up[:] = up
            gc.frustum_center = 0.5 * (left + right)
            gc.frustum_width = 0.5 * (right - left)
            gc.frustum_top = cy / fy * n
            gc.frustum_bottom = -(self.h - cy) / fy * n
        return dress_style.post(self.r.render().copy())


def scene_cams(nn, rec, m, d, sc, qs):
    import exec_tables as ET
    hero = None
    jd = DARK / f"{nn}.json"
    if jd.exists():
        hero = json.loads(jd.read_text()).get("hero_cam")
    az0 = hero["az"] if hero else -30.0

    class _Sh:   # scene_points / pedestal_foot only need .m and .pedestal
        pass
    sh = _Sh()
    sh.m = m
    import mujoco
    G = mujoco.mjtGeom
    x0 = sc["table"]["center"][0] - sc["table"]["size"][0] / 2
    sh.pedestal = [i for i in range(m.ngeom) if m.geom_bodyid[i] == 0 and m.geom_type[i] == G.mjGEOM_BOX
                   and m.geom_pos[i][0] + m.geom_size[i][0] < x0 + 0.05]
    nq1 = rec["nq_rigid"]
    pts = ET.scene_points(sh, d, np.concatenate(qs), nq1, sc, every=60)
    nlow = len(ET.pedestal_foot(sh, 1.0))
    cams = []
    for daz, el, margin, tight in CAMSET:
        p = pts
        if tight:   # table top + the arm's hand region only: a closer, brighter view
            p = pts[:-nlow] if nlow else pts
        c = ET.fit_cam(p, az0 + daz, el, margin=margin, nlow=0 if tight else nlow)
        cams.append(c)
    return cams


def sample(q, n=NF):
    return q[np.linspace(0, len(q) - 1, n).round().astype(int)]


def render_scene(job):
    nn, style, w, h, only = job
    tag = f"{nn}_{style}" if only is None else f"{nn}_{style}_{w}"
    dst = TILES / f"{tag}.npy"
    if only is None and dst.with_suffix(".json").exists():
        return tag, "cached"
    t0 = time.time()
    try:
        import mujoco
        rec, ros = rollouts_for(nn)
        m, d, sc = build(nn, rec, style)
        nq1 = rec["nq_rigid"]
        qs = [np.load(p)["q"].astype(float) for _, p, _ in ros]
        cams = scene_cams(nn, rec, m, d, sc, qs)
        if style == "primitive":
            os.environ["DRESS_STYLE"] = "primitive"
            import dress_style
            dress_style.primitive_fix(m, sc)
        m.vis.quality.shadowsize = 1024 if w <= 160 else (2048 if w <= 700 else 4096)
        m.vis.quality.offsamples = 4 if w <= 160 else 8
        mujoco.mj_forward(m, d)
        sh = Tiny(m, w, h)
        flex = getattr(mujoco, "mj_flex", None) if m.nflex else None
        specs = []
        for i, (rid, _, meta) in enumerate(ros):
            for j in range(CAMS_PER):
                ci = (i * CAMS_PER + j + (1 if style == "primitive" else 0)) % len(CAMSET)
                specs.append({"scene": nn, "style": style, "rollout": rid, "cam": ci,
                              "kind": meta.get("kind", "simulated candidate rollout (not shipped)"),
                              "variant": meta.get("variant", "shipped"), "success": meta.get("success")})
        if only is not None:
            specs = [specs[i] for i in only]
        arr = np.lib.format.open_memmap(dst, "w+", np.uint8, (len(specs), NF, h, w, 3))
        rid2q = {rid: q for (rid, _, _), q in zip(ros, qs)}
        for si, s in enumerate(specs):
            qq = sample(rid2q[s["rollout"]])
            for f in range(NF):
                d.qpos[:nq1] = qq[f]
                mujoco.mj_kinematics(m, d)
                mujoco.mj_camlight(m, d)
                if flex:
                    flex(m, d)
                arr[si, f] = sh.render(d, cams[s["cam"]])
        arr.flush()
        del arr
        (TILES / f"{tag}.json").write_text(json.dumps(specs, default=float))
        return tag, f"ok {len(specs)} streams {time.time() - t0:.0f}s"
    except Exception:
        return tag, "FAIL " + traceback.format_exc()[-1500:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--styles", default="dark,primitive")
    a = ap.parse_args()
    TILES.mkdir(parents=True, exist_ok=True)
    nns = a.scenes or [json.loads(f.read_text())["scene"] for f in sorted(CACHE.glob("*_run.json"))
                       if json.loads(f.read_text()).get("executed")]
    jobs = [(nn, st, TW, TH, None) for nn in nns for st in a.styles.split(",")]
    with Pool(a.jobs, maxtasksperchild=1) as p:
        for tag, msg in p.imap_unordered(render_scene, jobs):
            print(tag, msg, flush=True)


if __name__ == "__main__":
    main()
