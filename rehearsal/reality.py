"""Reality check: track the twin's objects through a short phone clip with SAM 3.1 video and compare where they
really ended up with where the simulation said they would.

    clip (3-5 s, same viewpoint as the photo, A4 sheet still on the table)
      -> SAM 3.1 video, box prompts seeded with the twin's perception.box_px (one request, our ids)
      -> last tracked mask of each object -> table xy (same camera / mask fit as perception, on the support height
         the sim predicted)
      -> per object: sim_xy vs observed_xy -> "ok" | "gap" | "lost";  verdict

    python -m rehearsal.reality synth          # synthetic test on out/desk_01 attempt 2 (clean + perturbed clip)

`sam_video` copies meta_client.sam's ledger/cache pattern so the commander can move it into meta_client.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import meta_client as mc
from .calib import Camera, apply_h, camera_from_homography, detect_paper, fit_homography, paper_world
from .perception import REC_RE, _decode_mask, _footprint_r, mask_outline, place_object

ROOT = Path(__file__).resolve().parent.parent
SAM_VIDEO_PER_FRAME = 0.20 / 1000
SAM_VIDEO_MIN_FRAMES = 50
GAP_TOL_M = 0.03   # see docs/perception.md: clean-clip tracking error P90 is well below this


# ---------------------------------------------------------------------------------------------- API
def sam_video(prompt: str, video_url: str, *, caller: str, metadata: dict | None = None,
              n_frames: int | None = None) -> str:
    """SAM 3.1 on a video (concept or box/point prompts on frame 0).  Returns the raw output_text
    (one line per frame: `<Nf>id<|box;...|><|mask;...|>,...`).  Cached by request hash; billed
    max(50, usage.video_frames_processed) x $0.0002, reserving for `n_frames` (or 300) up front."""
    req = {"model": "sam-3.1", "kind": "video", "prompt": prompt,
           "video": hashlib.sha256(video_url.encode()).hexdigest(), "metadata": metadata or {}}
    key = mc._key(req)
    hit = mc._cache_get(key)
    if hit is not None:
        return hit["text"]
    reserve = max(SAM_VIDEO_MIN_FRAMES, n_frames or 300) * SAM_VIDEO_PER_FRAME * 1.2
    mc._reserve(reserve, caller)
    usage = {}

    def call():
        text = ""
        kw = {"model": "sam-3.1", "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_video", "video_url": video_url}]}]}
        if metadata:
            kw["metadata"] = metadata
        with mc.client().responses.stream(**kw) as stream:
            for ev in stream:
                if ev.type == "response.output_text.delta":
                    text += ev.delta
                elif ev.type == "response.completed" and ev.response.usage is not None:
                    usage.update(ev.response.usage.model_dump())
        return text

    entry = {"caller": caller, "model": "sam-3.1", "kind": "sam_video"}
    try:
        text = mc._with_retries(call, caller)
    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        mc._settle(reserve, reserve, entry)
        raise
    frames = int(usage.get("video_frames_processed") or n_frames or SAM_VIDEO_MIN_FRAMES)
    actual = max(SAM_VIDEO_MIN_FRAMES, frames) * SAM_VIDEO_PER_FRAME
    entry["video_frames_processed"] = frames
    mc._settle(reserve, actual, entry)
    mc._cache_put(key, {"text": text, "caller": caller, "usage": usage})
    return text


def parse_video(text: str) -> dict[int, dict[int, dict]]:
    """-> {frame: {track id: {"box": (x1, y1, x2, y2), "conf": c|None, "mask": raster|None}}}"""
    frames = {}
    for line in text.strip().splitlines():
        if not line.startswith("<"):
            continue
        f = int(line[1:line.index("f>")])
        recs = {}
        for m in REC_RE.finditer(line):
            oid, box, _, H, W, enc, payload = m.groups()
            kv = dict(t.split("=", 1) for t in box.split(";") if "=" in t)
            b = tuple(int(float(kv[k])) for k in ("x1", "y1", "x2", "y2"))
            recs[int(oid)] = {"box": b, "conf": float(kv["c"]) if "c" in kv else None,
                              "mask": _decode_mask(int(H), int(W), enc, payload) if H else None}
        frames[f] = recs
    return frames


# ------------------------------------------------------------------------------------------- helpers
def _video_info(path: Path):
    import imageio.v2 as iio2
    rd = iio2.get_reader(str(path))
    n = rd.count_frames()
    last = rd.get_data(n - 1)
    first = rd.get_data(0)
    fps = rd.get_meta_data().get("fps", 20)
    rd.close()
    return n, fps, Image.fromarray(first), Image.fromarray(last)


def _shape_dims(shape: dict) -> tuple[str, list]:
    if shape["type"] == "box":
        return "box", list(shape["size"])
    return shape["type"], [shape["radius"], shape["height"]]


def _camera_for_video(scene: dict, vw: int, vh: int, last: Image.Image, recalibrate: bool):
    """Camera for the clip: re-fit from the A4 sheet in the last frame if found (handheld clip),
    else the photo's camera scaled to the video size (tripod / same viewpoint)."""
    ph = scene["photo"]
    sx, sy = vw / ph["width"], vh / ph["height"]
    old = np.asarray(ph["paper_corners_px"], float) * [sx, sy] if ph.get("paper_corners_px") else None
    if recalibrate and old is not None and len(old) == 4:
        det = detect_paper(last)
        if det is not None:
            q = np.asarray(det["corners_px"], float)
            # order like the photo's corners: nearest-corner assignment (viewpoint roughly unchanged)
            best = min((np.roll(qq, k, axis=0) for qq in (q, q[::-1]) for k in range(4)),
                       key=lambda c: float(np.linalg.norm(c - old, axis=1).sum()))
            if np.linalg.norm(best - old, axis=1).max() < 0.15 * max(vw, vh):
                Hm = fit_homography(best, paper_world(ph["paper_xy"], ph.get("paper_size", (0.297, 0.210))))
                cam, info = camera_from_homography(Hm, vw, vh)
                return cam, Hm, {"source": "paper_in_last_frame", **info, "corners_px": best.tolist()}
    c = ph["camera"]
    K = np.diag([sx, sy, 1.0]) @ np.asarray(c["K"], float)
    cam = Camera(K, c["R"], c["t"])
    G = K @ np.c_[cam.R[:, 0], cam.R[:, 1], cam.t]
    Hm = np.linalg.inv(G); Hm /= Hm[2, 2]
    return cam, Hm, {"source": "photo_camera_scaled"}


def _xy_base(v):
    if isinstance(v, dict):
        return np.asarray(v["xy"], float), float(v.get("z_base", 0.0))
    return np.asarray(v[:2], float), 0.0


# -------------------------------------------------------------------------------------------- check
def reality_check(video_path, scene: dict, sim_final: dict, data: str = "private", out_dir=None,
                  tol_m: float = GAP_TOL_M, recalibrate: bool = True, method: str = "guarded") -> dict:
    """Compare the clip's final object positions with the simulation's final state.

    video_path: MP4/MOV (H.264), same scene and roughly the same viewpoint as the twin's photo, starting before
        anything moves (frame 0 is seeded with the twin's perception.box_px).
    scene: the twin scene JSON (needs objects[*].perception.box_px and photo{camera|paper_corners_px}).
    sim_final: {object id: [x, y] | {"xy": [x, y], "z_base": height of the object's bottom above the table}}.
    Returns {"per_object": [{id, sim_xy, observed_xy, delta_cm, status, mask_iou, frames_tracked}], "verdict",
             "overlay": path to a GIF (or None), "camera": {...}, "frames": n, "tol_cm"}.
    """
    video_path = Path(video_path)
    n, fps, first, last = _video_info(video_path)
    vw, vh = first.size
    ph = scene["photo"]
    sx, sy = vw / ph["width"], vh / ph["height"]
    objs = [o for o in scene["objects"] if (o.get("perception") or {}).get("box_px")][:16]
    track_id = {o["id"]: k for k, o in enumerate(objs, 1)}
    prompt = "<0f>" + ",".join(
        f"{track_id[o['id']]}<|box;x1={int(b[0] * sx)};y1={int(b[1] * sy)};x2={int(b[2] * sx)};y2={int(b[3] * sy)};"
        f"w={vw};h={vh}|>" for o in objs for b in [o["perception"]["box_px"]])
    mime = "video/quicktime" if video_path.suffix.lower() == ".mov" else "video/mp4"
    url = f"data:{mime};base64,{base64.b64encode(video_path.read_bytes()).decode()}"
    assert data in ("public", "private")
    text = sam_video(prompt, url, caller="perception.reality_video", n_frames=n,
                     metadata={"propagation_direction": "forward", "include_confidence": "true"})
    frames = parse_video(text)
    fids = sorted(frames)
    cam, Hm, cam_info = _camera_for_video(scene, vw, vh, last, recalibrate)

    # last frame in which each track appears
    last_rec = {}
    for f in fids:
        for tid, rec in frames[f].items():
            last_rec[tid] = (f, rec)
    final_f = fids[-1] if fids else -1
    pseudo = {}
    for o in objs:
        tid = track_id[o["id"]]
        if tid not in last_rec:
            continue
        f, rec = last_rec[tid]
        shp, dims = _shape_dims(o["shape"])
        xy_sim, zb = _xy_base(sim_final.get(o["id"], o["pos"]))
        pseudo[o["id"]] = {"id": o["id"], "shape": shp, "dims_m": dims, "box": rec["box"], "mask": rec["mask"],
                           "z0": zb, "frame": f, "conf": rec["conf"]}
    rows = []
    others = list(pseudo.values())
    for o in objs:
        xy_sim, zb = _xy_base(sim_final.get(o["id"], o["pos"]))
        p = pseudo.get(o["id"])
        n_tracked = sum(1 for f in fids if track_id[o["id"]] in frames[f])
        if p is None or p["frame"] < final_f - max(2, int(0.1 * len(fids))):
            rows.append({"id": o["id"], "sim_xy": xy_sim.round(4).tolist(), "observed_xy": None, "delta_cm": None,
                         "status": "lost", "frames_tracked": n_tracked})
            continue
        xy, s, yaw, extra = place_object(p, cam, Hm, method, vw, vh, others)
        d = float(np.linalg.norm(np.asarray(xy) - xy_sim))
        rows.append({"id": o["id"], "sim_xy": xy_sim.round(4).tolist(), "observed_xy": np.round(xy, 4).tolist(),
                     "delta_cm": round(d * 100, 2), "status": "gap" if d > tol_m else "ok",
                     "mask_iou": extra.get("mask_iou"), "sam_confidence": p["conf"], "frames_tracked": n_tracked,
                     "box_px": list(p["box"]), "mask_px": mask_outline(p)})
    gaps = [r["id"] for r in rows if r["status"] != "ok"]
    verdict = ("consistent: every object ended within %.0f cm of the simulation" % (tol_m * 100) if not gaps else
               "sim-to-real gap: " + ", ".join(f"{r['id']} {r['status']}"
                                               + (f" ({r['delta_cm']:.1f} cm)" if r["delta_cm"] is not None else "")
                                               for r in rows if r["status"] != "ok"))
    res = {"per_object": rows, "verdict": verdict, "tol_cm": tol_m * 100, "frames": n, "frames_returned": len(fids),
           "camera": cam_info, "video": str(video_path), "overlay": None}
    if out_dir is not None:
        res["overlay"] = str(_overlay_gif(video_path, frames, track_id, rows, Hm, Path(out_dir)))
    return res


def _overlay_gif(video_path: Path, frames, track_id, rows, Hm, out_dir: Path, every: int = 4, width: int = 480):
    """GIF: tracked masks/boxes on the clip; on the last frame, sim position (white ring) vs observed (dot)."""
    import imageio.v2 as iio
    out_dir.mkdir(parents=True, exist_ok=True)
    rd = iio.get_reader(str(video_path))
    inv = {v: k for k, v in track_id.items()}
    status = {r["id"]: r for r in rows}
    Hinv = np.linalg.inv(Hm)
    from PIL import ImageFont
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    imgs = []
    n = rd.count_frames()
    pick = list(range(0, n, every))
    if pick[-1] != n - 1:
        pick.append(n - 1)
    for f in pick:
        im = Image.fromarray(rd.get_data(f)).convert("RGB")
        d = ImageDraw.Draw(im)
        for tid, rec in frames.get(f, {}).items():
            oid = inv.get(tid, str(tid))
            r = status.get(oid, {})
            col = (230, 60, 60) if r.get("status") == "gap" else (40, 220, 90)
            mp = mask_outline({"mask": rec["mask"], "box": rec["box"]}) if rec["mask"] is not None else None
            if mp and len(mp) >= 3:
                d.line([tuple(p) for p in mp] + [tuple(mp[0])], fill=col, width=2)
            else:
                d.rectangle(rec["box"], outline=col, width=2)
            d.text((rec["box"][0] + 2, rec["box"][1] - 24), oid, fill=col, font=font)
        if f == pick[-1]:
            for r in rows:
                sim = apply_h(Hinv, [r["sim_xy"]])[0]
                d.ellipse([sim[0] - 9, sim[1] - 9, sim[0] + 9, sim[1] + 9], outline=(255, 255, 255), width=2)
                if r["observed_xy"]:
                    ob = apply_h(Hinv, [r["observed_xy"]])[0]
                    col = (230, 60, 60) if r["status"] == "gap" else (40, 220, 90)
                    d.line([tuple(sim), tuple(ob)], fill=col, width=2)
                    d.ellipse([ob[0] - 5, ob[1] - 5, ob[0] + 5, ob[1] + 5], fill=col)
                    if r["status"] == "gap":
                        d.text((ob[0] + 10, ob[1] - 10), f"{r['id']}: {r['delta_cm']:.1f} cm off",
                               fill=(255, 80, 80), font=font)
        k = width / im.size[0]
        imgs.append(im.resize((width, int(im.size[1] * k))))
    rd.close()
    imgs += [imgs[-1]] * 15  # hold the verdict frame
    p = out_dir / f"{video_path.stem}.reality.gif"
    iio.mimsave(p, [np.asarray(i) for i in imgs], duration=0.1, loop=0)
    return p


# ------------------------------------------------------------------------------------ synthetic test
def render_clip(scene_spec: dict, traj: list, pose, paper_xy, path: Path, w: int = 960, h: int = 720,
                n_frames: int = 100, fps: int = 20, perturb: dict | None = None, seed: int = 0):
    """Render a recorded attempt (attempt_N.traj.json, 'full' recording: every body's xpos+xquat at 50 Hz) from a
    phone camera, robot hidden (a person is doing the task).  perturb: {object id: (dx, dy)} slides that object
    over the last 20% of the clip, in the video only (the sim does not know).  Returns the GT final positions as
    seen in the video {id: [x, y, z_center]}."""
    import imageio.v2 as iio
    import mujoco
    from .scene import build_model
    from .synth import A4, FOVY, _add_box, paper_corners_world

    m = build_model(scene_spec)
    assert len(traj[0]) == 1 + 7 * (m.nbody - 1), "trajectory does not match this scene's model"
    m.vis.global_.offwidth, m.vis.global_.offheight = max(w, 640), max(h, 480)
    m.vis.global_.fovy = FOVY
    m.vis.quality.shadowsize = 4096
    m.vis.quality.offsamples = 4
    m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    m.vis.headlight.diffuse[:] = [0.45, 0.45, 0.45]
    m.vis.map.znear = 0.01
    for i in range(m.nlight - 1, m.nlight):
        m.light_pos[i] = [0.2, -0.6, 2.2]
        dd = np.array([0.35, 0.45, -2.2]); m.light_dir[i] = dd / np.linalg.norm(dd)
        m.light_castshadow[i] = 1
        m.light_diffuse[i] = [0.75, 0.73, 0.70]
    d = mujoco.MjData(m)
    ids = [o["id"] for o in scene_spec["objects"]]
    bodies = {oid: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, oid) for oid in ids}
    name, lookat, dist, az, el = pose
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    pick = np.linspace(0, len(traj) - 1, n_frames).round().astype(int)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    vig = 1 - 0.18 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2
    out, final = [], {}
    pc = paper_corners_world(*paper_xy)
    with mujoco.Renderer(m, h, w) as r:
        for k, fi in enumerate(pick):
            row = traj[fi]
            for oid, b in bodies.items():
                o = 1 + 7 * (b - 1)
                pos, quat = np.array(row[o:o + 3]), np.array(row[o + 3:o + 7])
                ramp = np.clip((k - 0.8 * n_frames) / (0.15 * n_frames), 0, 1)
                if perturb and oid in perturb:
                    pos[:2] += ramp * np.asarray(perturb[oid])
                j = m.body_jntadr[b]
                adr = m.jnt_qposadr[j]
                d.qpos[adr:adr + 3], d.qpos[adr + 3:adr + 7] = pos, quat
                if k == len(pick) - 1:
                    final[oid] = pos.tolist()
            mujoco.mj_kinematics(m, d)
            r.update_scene(d, camera=cam, scene_option=opt)
            ctr = pc.mean(0)
            _add_box(r.scene, [ctr[0], ctr[1], 0.0006], [A4[0] / 2, A4[1] / 2, 0.0005], [0.97, 0.97, 0.95, 1])
            img = r.render().astype(float)
            out.append(np.clip(img * vig[..., None] + rng.normal(0, 2.5, img.shape), 0, 255).astype(np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(path, out, fps=fps, codec="libx264", macro_block_size=8, quality=6,
                 ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    return final


def synth_test(out_dir: Path = ROOT / "photos" / "synth" / "reality", attempt: int = 2):
    """Clean + perturbed clip of out/desk_01/attempt_N; the twin is perception's result on desk_01__rear50.jpg."""
    from .perception import estimate_scene, perceive_raw
    from .scene import object_height
    from .synth import POSES, _match
    cands = [ROOT / "out" / "desk_01", ROOT.parent.parent / "rehearsal-mvp" / "out" / "desk_01"]
    run = next((c for c in cands if (c / "report.json").exists()), cands[0])
    rep = json.loads((run / "report.json").read_text())
    traj = json.loads((run / f"attempt_{attempt}.traj.json").read_text())
    spec = rep["scene_spec"]
    gt = json.loads((ROOT / "photos/synth/desk_01__rear50.gt.json").read_text())
    photo = ROOT / "photos/synth/desk_01__rear50.jpg"
    raw = perceive_raw(photo, data="public")
    twin = estimate_scene(raw, np.array(gt["paper_corners_px"]), tuple(gt["paper_xy"]), privacy="public",
                          photo_file=str(photo.relative_to(ROOT)))
    pairs, _, _ = _match(twin["objects"], gt)
    to_gt = {e["id"]: g["id"] for e, g in pairs}
    pose = next(p for p in POSES if p[0] == "rear50")
    results = {}
    for tag, perturb in (("clean", None), ("nudged", {"bottle_1": (0.0, 0.05)})):
        clip = out_dir / f"desk_01_attempt{attempt}_{tag}.mp4"
        seen_p = clip.with_suffix(".truth.json")
        if clip.exists() and seen_p.exists():  # reuse the clip so the SAM video call hits the cache
            seen = json.loads(seen_p.read_text())
        else:
            seen = render_clip(spec, traj, pose, tuple(gt["paper_xy"]), clip, perturb=perturb)
            seen_p.write_text(json.dumps(seen))
        # the simulation's final state (never perturbed), in twin ids, with the support height it predicts
        sim_final = {}
        truth = {}
        for e in twin["objects"]:
            gid = to_gt.get(e["id"])
            if gid is None:
                continue
            gobj = next(o for o in spec["objects"] if o["id"] == gid)
            zc = _final_center(spec, traj, gid)
            sim_final[e["id"]] = {"xy": zc[:2], "z_base": max(0.0, zc[2] - object_height(gobj) / 2)}
            truth[e["id"]] = seen[gid]
        res = reality_check(clip, twin, sim_final, data="public", out_dir=out_dir)
        for r in res["per_object"]:
            r["gt_id"] = to_gt.get(r["id"])
            if r["observed_xy"] is not None:
                r["err_vs_truth_cm"] = round(float(np.hypot(*(np.asarray(r["observed_xy"]) -
                                                              np.asarray(truth[r["id"]][:2])))) * 100, 2)
        results[tag] = res
        print(f"\n[{tag}] {res['verdict']}  (camera: {res['camera']['source']}, frames {res['frames']}, "
              f"returned {res['frames_returned']})")
        for r in res["per_object"]:
            print(f"  {r['id']:16s} gt={r['gt_id']:12s} status={r['status']:4s} delta={r['delta_cm']} cm  "
                  f"err_vs_truth={r.get('err_vs_truth_cm')} cm  iou={r.get('mask_iou')} tracked={r['frames_tracked']}")
    errs = [r["err_vs_truth_cm"] for r in results["clean"]["per_object"] if r.get("err_vs_truth_cm") is not None]
    if errs:
        print(f"\nclean clip: tracking->table error vs truth  median {np.median(errs):.2f} cm, "
              f"P90 {np.percentile(errs, 90):.2f} cm, max {max(errs):.2f} cm (n={len(errs)})")
    (out_dir / "reality_synth.json").write_text(json.dumps(results, indent=1, default=float))
    return results


def _final_center(spec, traj, oid):
    from .scene import build_model
    import mujoco
    m = build_model(spec)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, oid)
    o = 1 + 7 * (b - 1)
    return traj[-1][o:o + 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["synth"])
    ap.add_argument("--attempt", type=int, default=2)
    a = ap.parse_args()
    synth_test(attempt=a.attempt)


if __name__ == "__main__":
    main()
