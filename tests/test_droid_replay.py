"""Real-robot replay (DROID, rehearsal/replay.py): kinematics, robot base pose, camera, grasp model.  No API calls."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rehearsal.replay import EpisodeCamera, command_width, load_episode, load_scene, replay, replay_scene
from rehearsal.scene import PANDA_XML, build_model, support_boxes

ROOT = Path(__file__).resolve().parent.parent
DROID = ROOT / "datasets" / "droid"


def _eul(a):
    x, y, z = a
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


@pytest.mark.parametrize("slug", ["duck_success", "cup_fail"])
def test_droid_ee_is_the_panda_flange(slug):
    """DROID's recorded cartesian_position is the Panda hand origin (flange), turned +45 deg about z."""
    tr = json.loads((DROID / slug / "trajectory.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(PANDA_XML))
    d = mujoco.MjData(m)
    hand = m.body("hand").id
    for i in range(0, len(tr["joint_positions"]), 13):
        d.qpos[:7] = tr["joint_positions"][i]
        mujoco.mj_kinematics(m, d)
        P = np.asarray(tr["cartesian_position"][i])
        assert np.linalg.norm(d.xpos[hand] - P[:3]) < 1e-3
        Rrel = d.xmat[hand].reshape(3, 3).T @ _eul(P[3:])
        assert abs(np.arctan2(Rrel[1, 0], Rrel[0, 0]) - np.pi / 4) < 1e-2


def test_robot_base_pose_moves_link0_and_stand():
    sc = load_scene("duck_success")
    m = build_model(sc)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    assert np.allclose(d.xpos[m.body("link0").id], [0, 0, sc["robot"]["base_z"]])
    stand = [b for b in support_boxes(sc["table"], (0.0, 0.0, sc["robot"]["base_z"], 0.0)) if b["name"] == "robot_stand"]
    assert abs(stand[0]["center"][2] + stand[0]["size"][2] / 2 - sc["robot"]["base_z"]) < 1e-9
    sc2 = dict(sc, robot={"base_x": 0.1, "base_y": -0.2, "base_z": 0.05, "base_yaw": 0.3})
    m2 = build_model(sc2)
    d2 = mujoco.MjData(m2)
    mujoco.mj_kinematics(m2, d2)
    assert np.allclose(d2.xpos[m2.body("link0").id], [0.1, -0.2, 0.05])
    assert np.isclose(d2.xquat[m2.body("link0").id][3], np.sin(0.15))


def test_command_width():
    assert command_width(0.0) == pytest.approx(0.085)
    assert command_width(0.88) == pytest.approx(0.0)
    assert command_width(0.44) == pytest.approx(0.0425)


def test_episode_camera_renders_through_the_real_calibration():
    """The pixel where calib.Camera projects the pot's centre shows the pot in the MuJoCo render."""
    sc = replay_scene(load_scene("duck_success"))
    ep = load_episode("duck_success")
    m = build_model(sc)
    d = mujoco.MjData(m)
    d.qpos[:7] = ep["traj"]["joint_positions"][0]
    mujoco.mj_forward(m, d)
    c = ep["camera"]["rehearsal_camera"]
    s = 0.5
    cam = EpisodeCamera(m, np.diag([s, s, 1.0]) @ np.asarray(c["K"]), c["R"], c["t"], 640, 360)
    cam.renderer.enable_segmentation_rendering()
    seg = cam.render(d)[..., 0]
    from rehearsal.calib import Camera
    K = np.diag([s, s, 1.0]) @ np.asarray(c["K"])
    for oid in ("pot_1", "duck_1"):
        b = m.body(oid).id
        u, v = Camera(K, c["R"], c["t"]).project(d.xpos[b])[0].round().astype(int)
        geoms = {int(g) for g in seg[v - 2:v + 3, u - 2:u + 3].ravel() if g >= 0}
        assert any(m.geom_bodyid[g] == b for g in geoms), oid


def test_replay_grasps_the_duck_and_closes_on_air_when_the_real_gripper_did():
    ok = replay(load_scene("duck_success"), load_episode("duck_success"))
    kinds = [(e["kind"], e.get("obj")) for e in ok["events"]]
    assert ("grasp", "duck_1") in kinds and ok["success"]
    miss = replay(load_scene("duck_missed_grasp"), load_episode("duck_missed_grasp"))
    assert not any(e["kind"] == "grasp" for e in miss["events"])
    assert any(e["kind"] == "closed_on_air" for e in miss["events"]) and not miss["success"]


def test_replay_never_reads_the_outcome_label(monkeypatch):
    real_read = Path.read_text

    def guarded(self, *a, **k):
        assert self.name != "label.json", "the predictor must not read the success label"
        return real_read(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", guarded)
    replay(load_scene("bowl_fail"), load_episode("bowl_fail"))
