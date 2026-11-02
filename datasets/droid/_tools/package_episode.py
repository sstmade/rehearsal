"""Vendor one packaged DROID episode into datasets/droid/<slug>/ and add the full-rate (15 Hz) robot state.

    <realrobot venv>/bin/python -I datasets/droid/_tools/package_episode.py <realrobot dir> <slug> <out dir>

Copies only the no-face packaged view (pre.png, post.png, clip.mp4, thumb.jpg, camera.json, label.json) made by the
scout (realrobot/scripts/extract.py) and rewrites trajectory.json with every control step of
observation/robot_state/{joint_positions, gripper_position, cartesian_position} from the raw DROID trajectory.h5.
Never copies raw MP4s, stereo videos, contact sheets or metadata.json (which names the operator).
Needs h5py (not a Rehearsal runtime dependency); run it with -I: the inputs are downloaded files.
"""
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

KEEP = ["pre.png", "post.png", "clip.mp4", "thumb.jpg", "camera.json", "label.json"]


def main(rr: str, slug: str, out: str):
    rr, out = Path(rr), Path(out) / slug
    src = rr / "episodes" / slug
    out.mkdir(parents=True, exist_ok=True)
    for f in KEEP:
        shutil.copyfile(src / f, out / f)
    lab = json.loads((src / "label.json").read_text())
    tr = json.loads((src / "trajectory.json").read_text())
    h = h5py.File(rr / "downloads" / "droid_raw" / lab["episode_id"] / "trajectory.h5", "r")
    rs = h["observation/robot_state"]
    q, g, P = rs["joint_positions"][:], rs["gripper_position"][:], rs["cartesian_position"][:]
    assert q.shape[1] == 7 and len(q) == len(g) == len(P) == tr["n_steps"]
    for k in ("ee_xyz_base_every5", "gripper_every5", "joint_positions_every5"):
        tr.pop(k, None)
    tr["state_source"] = "DROID trajectory.h5 observation/robot_state, every control step (15 Hz); step i <-> video frame i"
    tr["joint_positions"] = np.round(q, 5).tolist()
    tr["gripper_position"] = np.round(g, 4).tolist()
    tr["cartesian_position"] = np.round(P, 5).tolist()
    (out / "trajectory.json").write_text(json.dumps(tr, indent=0))
    print(slug, len(q), "steps")


if __name__ == "__main__":
    main(*sys.argv[1:4])
