"""Package DROID v2 episodes (datasets/droid/episodes_v2.json) from raw downloads.  $0, local only.

    <python with av+h5py+scipy> -I datasets/droid/_tools/package_v2.py <annot dir> <raw dir> <privacy sheet dir> [slug ...]
    <python> -I datasets/droid/_tools/package_v2.py --pool      # then: shared table height per scene (amendment 1)

Per episode, from the raw files fetched by fetch_v2.py (untrusted input; run with -I):
  pre.png / post.png (first / last frame of the chosen exterior camera, 1280x720), clip.mp4 (960x540, 15 fps H.264),
  thumb.jpg, camera.json (KarlP/droid intrinsics + improved extrinsics), trajectory.json (every control step),
  label.json (DROID success flag + language annotations; evaluation only).
Table height (eval/droid_v2_protocol.md): z_table = min EE z over the episode - 0.185 m (fingertip bound midpoint).
Writes a 24-frame contact sheet of the packaged view to <privacy sheet dir> for the no-people check.
Never copies raw MP4s, stereo, the other cameras or metadata.json (which names the operator).
"""
import glob
import json
import os
import sys
from pathlib import Path

import av
import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

TIP_BELOW_FLANGE = 0.185
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "datasets" / "droid"


def T_of(v):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", v[3:6]).as_matrix()
    T[:3, 3] = v[:3]
    return T


def table_z(ee_z) -> float:
    return float(np.min(ee_z)) - TIP_BELOW_FLANGE


def package(ep, annot, raw, sheets):
    eid, key, slug = ep["eid"], ep["camera"], ep["slug"]
    sup = annot["sup"][eid]
    d = Path(raw) / eid
    m = json.loads((d / "metadata.json").read_text())
    s = m[f"{key}_cam_serial"]
    out = DATA / slug
    out.mkdir(parents=True, exist_ok=True)
    fr = [f.to_image() for f in av.open(glob.glob(str(d / f"{key}_{s}.mp4"))[0]).decode(video=0)]
    h = h5py.File(d / "trajectory.h5", "r")
    rs = h["observation/robot_state"]
    q, g, P = rs["joint_positions"][:], rs["gripper_position"][:], rs["cartesian_position"][:]
    act = h["action"]
    cmd_src = "action/target_gripper_position" if "target_gripper_position" in act else "action/gripper_position"
    cmd = h[cmd_src][:]
    n = min(len(fr), len(q))
    fr, q, g, P, cmd = fr[:n], q[:n], g[:n], P[:n], cmd[:n]
    fr[0].save(out / "pre.png")
    fr[-1].save(out / "post.png")
    oc = av.open(str(out / "clip.mp4"), "w")
    st = oc.add_stream("libx264", rate=15)
    st.width, st.height, st.pix_fmt = 960, 540, "yuv420p"
    st.options = {"crf": "23"}
    for im in fr:
        for p in st.encode(av.VideoFrame.from_image(im.resize((960, 540)))):
            oc.mux(p)
    for p in st.encode():
        oc.mux(p)
    oc.close()
    fx, cx, fy, cy = annot["intr"][eid][s]["cameraMatrix"]
    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    Tc2b = T_of(sup[s])
    Tb2c = np.linalg.inv(Tc2b)
    zt = table_z(P[:, 2])
    Rw = Tb2c[:3, :3]
    tw = Tb2c[:3, 3] + Rw @ np.array([0, 0, zt])
    cam = {"width": fr[0].width, "height": fr[0].height, "K": K, "dist": [0] * 5,
           "model": "pinhole; ZED 2 left, rectified (DROID intrinsics.json, per episode)", "serial": s, "which": key,
           "T_cam_to_base": Tc2b.tolist(), "T_base_to_cam": Tb2c.tolist(),
           "extrinsics_source": "KarlP/droid cam2base_extrinsic_superset.json (post-hoc recalibration)",
           "extrinsics_quality": {k: sup[k] for k in sup if k.startswith(s + "_")},
           "table_plane_base": {"z_table_base_m": round(zt, 4), "assumed_horizontal": True,
                                "source": f"fingertip rule: min EE z ({float(P[:, 2].min()):.4f}) - {TIP_BELOW_FLANGE} m"},
           "robot_base_height_above_table_m": round(-zt, 4),
           "table_plane_status": "v2 fixed rule (eval/droid_v2_protocol.md); no stereo measurement",
           "rehearsal_camera": {"K": K, "R": Rw.tolist(), "t": tw.tolist(),
                                "convention": "rehearsal.calib.Camera: X_cam = R @ X_world + t; world = robot base frame "
                                              "translated so the table top is z=0"},
           "clip_note": "clip.mp4 is resized to 960x540: scale K by 960/width for the clip"}
    (out / "camera.json").write_text(json.dumps(cam, indent=1))
    closed = g > 0.3
    ev = np.flatnonzero(np.diff(closed.astype(int)))
    tr = {"n_steps": int(n), "hz": 15, "duration_s": round(n / 15, 2),
          "ee_frame": "DROID cartesian_position = Franka EE in robot base frame; xyz m, rpy rad",
          "gripper": "Robotiq 2F-85, 0=open 1=closed; closed on nothing reaches ~0.92-0.97, holding stays lower",
          "gripper_events": [{"step": int(i + 1), "event": "close" if closed[i + 1] else "open",
                              "ee_xyz_base": np.round(P[i + 1, :3], 4).tolist()} for i in ev],
          "gripper_max": round(float(g.max()), 3),
          "state_source": "DROID trajectory.h5 observation/robot_state, every control step (15 Hz); step i <-> video frame i",
          "gripper_command_source": cmd_src,
          "joint_positions": np.round(q, 5).tolist(), "gripper_position": np.round(g, 4).tolist(),
          "cartesian_position": np.round(P, 5).tolist(), "gripper_command": np.round(cmd, 4).tolist()}
    (out / "trajectory.json").write_text(json.dumps(tr, indent=0))
    lab = {"episode_id": eid, "dataset": "DROID 1.0.1 raw (gs://gresearch/robotics/droid_raw/1.0.1)",
           "relative_path": sup["relative_path"], "lab": m["lab"], "success": m["success"],
           "current_task_operator": m["current_task"], "language_annotations": annot["lang"].get(eid),
           "scene_id": m["scene_id"], "frames": n, "camera_used": key, "instruction_used": ep["instruction"],
           "license": "CC BY 4.0 (DROID; Khazatsky et al. 2024)"}
    (out / "label.json").write_text(json.dumps(lab, indent=1))
    th = Image.new("RGB", (960, 270))
    th.paste(fr[0].resize((480, 270)), (0, 0))
    th.paste(fr[-1].resize((480, 270)), (480, 0))
    th.save(out / "thumb.jpg", quality=85)
    # privacy sheet: 24 evenly spaced frames of the packaged view
    idx = [round(i * (n - 1) / 23) for i in range(24)]
    W = Image.new("RGB", (6 * 320, 4 * 180 + 14))
    for k, i in enumerate(idx):
        W.paste(fr[i].resize((320, 180)), ((k % 6) * 320, 14 + (k // 6) * 180))
    ImageDraw.Draw(W).text((2, 1), f"{slug} {key} n={n}", fill=(255, 255, 0))
    os.makedirs(sheets, exist_ok=True)
    W.save(Path(sheets) / f"{slug}.jpg", quality=75)
    print(slug, n, "frames", "success" if m["success"] else "FAIL", "z_table", round(zt, 4), "cmd", cmd_src, flush=True)


def scene_key(eid: str) -> str:
    lab, collector, date = eid.split("+")
    return f"{lab}+{collector}+{date[:10]}"


def pool_tables(eps, max_cam_shift=0.05):
    """Protocol amendment 1: episodes of one scene (same lab, operator id and day, camera position within 5 cm in
    the base frame) share one table height = min EE z over all of them - TIP_BELOW_FLANGE.  An episode whose gripper
    never went down to the table (e.g. a marker grasped upright) otherwise gets a table that is too high."""
    info = []
    for ep in eps:
        d = DATA / ep["slug"]
        cam = json.loads((d / "camera.json").read_text())
        P = np.asarray(json.loads((d / "trajectory.json").read_text())["cartesian_position"])
        info.append((ep, cam, float(P[:, 2].min()), np.asarray(cam["T_cam_to_base"])[:3, 3]))
    out = {}
    for ep, cam, zmin, c in info:
        group = [(e2, z2) for e2, _, z2, c2 in info
                 if scene_key(e2["eid"]) == scene_key(ep["eid"]) and np.linalg.norm(c2 - c) <= max_cam_shift]
        zmin_g, src = min((z2, e2["slug"]) for e2, z2 in group)
        zt = zmin_g - TIP_BELOW_FLANGE
        Tb2c = np.asarray(cam["T_base_to_cam"])
        cam["table_plane_base"] = {"z_table_base_m": round(zt, 4), "assumed_horizontal": True,
                                   "source": f"fingertip rule over the scene: min EE z {zmin_g:.4f} ({src}) - "
                                             f"{TIP_BELOW_FLANGE} m; scene = {[e2['slug'] for e2, _ in group]}"}
        cam["robot_base_height_above_table_m"] = round(-zt, 4)
        cam["rehearsal_camera"]["t"] = (Tb2c[:3, 3] + Tb2c[:3, :3] @ np.array([0, 0, zt])).tolist()
        (DATA / ep["slug"] / "camera.json").write_text(json.dumps(cam, indent=1))
        out[ep["slug"]] = round(zt, 4)
    return out


def main(argv):
    if argv and argv[0] == "--pool":
        print(pool_tables(json.loads((DATA / "episodes_v2.json").read_text())["episodes"]))
        return
    annot_dir, raw, sheets, *slugs = argv
    A = Path(annot_dir)
    annot = {"sup": json.loads((A / "cam2base_extrinsic_superset.json").read_text()),
             "intr": json.loads((A / "intrinsics.json").read_text()),
             "lang": json.loads((A / "droid_language_annotations.json").read_text())}
    eps = json.loads((DATA / "episodes_v2.json").read_text())["episodes"]
    for ep in eps:
        if not slugs or ep["slug"] in slugs:
            package(ep, annot, raw, sheets)


if __name__ == "__main__":
    main(sys.argv[1:])
