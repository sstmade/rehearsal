"""Choose the table height per DROID scene and write it into every episode's camera.json.  $0, local.

    python -I datasets/droid/_tools/table_height.py datasets/droid

Inputs (made by table_sweep.py): _tools/sweep_stereo.json (one ZED's own stereo pair) and _tools/sweep_two_ext.json
(ext1+ext2 wide baseline).  Both are camera measurements; neither alone is trustworthy here:
  * the ZED stereo of the SAME unchanged 07-14 scene (camera, robot base and table did not move: phase correlation
    of the table / robot-base image regions between episodes < 1 px) gives -8.9 / -3.3 / -4.9 / -1.3 cm, i.e. its
    absolute disparity offset changes between recordings (ZED self-calibration at start-up);
  * the two-exterior sweep uses ext1's per-episode *predicted* extrinsics (not the calibrated "GT" ones).
The robot is the referee: the fingertips can never be below the table.  With the Robotiq 2F-85 fingertip end
D_TIP = 0.17-0.19 m below the DROID EE point (= Panda flange; checked by projecting candidate tip points into the
grasp frames), the lowest EE height of every episode of a scene bounds the table from above.  Per scene we keep the
measurements that (a) have >= 20 RANSAC inliers and (b) satisfy that bound with the shortest plausible tip (0.17 m),
and take their median.  The spread of the kept values and the margin to the robot bound are reported as residuals.
"""
import json
import sys
from pathlib import Path

import numpy as np

D_TIP = (0.17, 0.19)
SCENES = {"2023-07-14": ["duck_success", "duck_missed_grasp", "duck_fail_nudge", "bowl_fail"],
          "2023-07-27": ["cup_success", "cup_fail"]}


def main(root):
    root = Path(root)
    tools = root / "_tools"
    meas = []
    for f, kind in (("sweep_stereo.json", "zed_stereo"), ("sweep_two_ext.json", "two_exterior")):
        for k, r in json.loads((tools / f).read_text()).items():
            meas.append({"key": k, "kind": kind, "eid": r["eid"], "z": r["z_table_at_0.6_0"],
                         "inliers": r["n_inliers"], "rms_m": r["rms_residual_m"], "tilt_deg": r["tilt_deg"]})
    out = {"method": __doc__.strip().splitlines()[0], "d_tip_m": D_TIP, "scenes": {}}
    for date, slugs in SCENES.items():
        eids, min_ee = set(), []
        for s in slugs:
            lab = json.loads((root / s / "label.json").read_text())
            tr = json.loads((root / s / "trajectory.json").read_text())
            eids.add(lab["episode_id"])
            P = np.asarray(tr["cartesian_position"])
            min_ee.append((float(P[:, 2].min()), s, int(P[:, 2].argmin())))
        z_ee, s_ee, i_ee = min(min_ee)
        bound = z_ee - D_TIP[0]                      # table z <= this
        rows = [dict(m) for m in meas if m["eid"] in eids]
        for m in rows:
            m["kept"] = bool(m["inliers"] >= 20 and m["z"] <= bound)
            m["why_dropped"] = None if m["kept"] else ("< 20 inliers" if m["inliers"] < 20 else
                                                        f"fingertips would be {100 * (m['z'] - bound):.1f} cm "
                                                        f"below the table at step {i_ee} of {s_ee}")
        kept = [m["z"] for m in rows if m["kept"]]
        z = float(np.median(kept))
        out["scenes"][date] = {
            "episodes": slugs, "z_table_base_m": round(z, 4), "robot_base_height_above_table_m": round(-z, 4),
            "kept_spread_m": round(float(np.ptp(kept)), 4), "n_kept": len(kept),
            "robot_bound": {"min_ee_z_base_m": round(z_ee, 4), "episode": s_ee, "step": i_ee,
                            "table_z_max_m": round(bound, 4),
                            "fingertip_end_above_table_m": [round(z_ee - d - z, 4) for d in D_TIP]},
            "measurements": rows}
        for s in slugs:
            p = root / s / "camera.json"
            cam = json.loads(p.read_text())
            Tb2c = np.asarray(cam["T_base_to_cam"], float)
            Rw = Tb2c[:3, :3]
            cam["table_plane_base"] = {"z_table_base_m": round(z, 4), "assumed_horizontal": True,
                                       "source": "datasets/droid/table_height.json"}
            cam["robot_base_height_above_table_m"] = round(-z, 4)
            cam["rehearsal_camera"]["t"] = (Tb2c[:3, 3] + Rw @ np.array([0, 0, z])).tolist()
            cam["table_plane_status"] = (
                f"scene {date}: median of {len(kept)} camera plane sweeps that agree with the robot's fingertip bound "
                f"(spread {100 * np.ptp(kept):.1f} cm); see datasets/droid/table_height.json")
            p.write_text(json.dumps(cam, indent=1))
        print(date, out["scenes"][date]["z_table_base_m"], "kept", kept, "bound", round(bound, 4))
    (root / "table_height.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
