"""DROID v2 evaluation: baselines, statistics and packaged data (eval/droid_v2.py, eval/droid_v2_protocol.md)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("droid_v2", ROOT / "eval" / "droid_v2.py")
dv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dv)
DATA = ROOT / "datasets" / "droid"


def _traj(cmd, g, xyz):
    return {"gripper_command": cmd, "gripper_position": g,
            "cartesian_position": [[x, y, z, 3.1, 0, 0] for x, y, z in xyz]}


def test_close_segments():
    assert dv.close_segments([0, 0.9, 0.9, 0, 1, 1]) == [(1, 3), (4, 6)]
    assert dv.close_segments([0, 0, 0]) == []


def test_release_point_moves_nearest_held_object_to_release_xy():
    n = 30
    cmd = [0] * 5 + [1] * 20 + [0] * 5
    g = [0] * 5 + [0.6] * 20 + [0] * 5                     # plateau below 0.9 -> holding
    xyz = [(0.5, 0.0, 0.2)] * 5 + [(0.5 + 0.01 * i, 0.0, 0.25) for i in range(20)] + [(0.7, 0.0, 0.3)] * 5
    final, grasped = dv.release_point(_traj(cmd, g, xyz), {"a": [0.52, 0.01], "b": [0.3, 0.3]})
    assert grasped == ["a"]
    assert final["a"] == pytest.approx([0.69, 0.0])
    assert final["b"] == [0.3, 0.3]
    assert len(xyz) == n


def test_release_point_ignores_empty_close_and_far_objects():
    cmd = [0] * 5 + [1] * 20 + [0] * 5
    empty = [0] * 5 + [0.95] * 20 + [0] * 5               # closed on nothing
    xyz = [(0.5, 0.0, 0.2)] * 30
    assert dv.release_point(_traj(cmd, empty, xyz), {"a": [0.5, 0.0]})[1] == []
    held = [0] * 5 + [0.6] * 20 + [0] * 5
    assert dv.release_point(_traj(cmd, held, xyz), {"a": [0.65, 0.0]})[1] == []     # 15 cm away


def test_gripper_rule_needs_hold_and_lift():
    cmd = [0] * 5 + [1] * 20 + [0] * 5
    hold = [0] * 5 + [0.6] * 20 + [0] * 5
    lift = [(0.5, 0, 0.2)] * 5 + [(0.5, 0, 0.2 + 0.01 * i) for i in range(20)] + [(0.5, 0, 0.4)] * 5
    flat = [(0.5, 0, 0.2)] * 30
    assert dv.gripper_rule(_traj(cmd, hold, lift))
    assert not dv.gripper_rule(_traj(cmd, hold, flat))
    assert not dv.gripper_rule(_traj(cmd, [0] * 5 + [0.95] * 20 + [0] * 5, lift))


def test_wilcoxon_exact_values():
    assert dv.wilcoxon_p([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == pytest.approx(2 / 1024)
    assert dv.wilcoxon_p([1, -2, 3, -4, 5, -6]) == pytest.approx(1.0, abs=0.2)
    assert dv.wilcoxon_p([0, 0, 0]) == 1.0
    # zeros are dropped (zero_method="wilcox")
    assert dv.wilcoxon_p([0, 1, 2, 3, 4, 5, 6]) == pytest.approx(dv.wilcoxon_p([1, 2, 3, 4, 5, 6]))


def _rec(slug, objs, grasped=(), sim=False, real=False, rule=False):
    return {"slug": slug, "objects": objs, "release_point_grasped": list(grasped), "sim_success": sim,
            "real_success": real, "gripper_rule_success": rule}


def _obj(oid, scene, real0, real1, replay, release, replay_disp):
    return {"id": oid, "status": "ok", "scene_xy": scene, "real_start_xy": real0, "real_final_xy": real1,
            "real_final_xy_at_replay_z": real1,
            "real_moved_cm": float(np.linalg.norm(np.subtract(real1, real0))) * 100,
            "pred": {"replay": replay, "stay_put": scene, "release_point": release},
            "replay_displacement_cm": replay_disp, "replay_up_cos": 1.0}


def test_score_subsets_and_agreement():
    recs = []
    for k in range(6):
        recs.append(_rec(f"e{k}", [
            # grasped object: release point is right, replay leaves it (as in the v2 data)
            _obj("cube", [0.5, 0.0], [0.5, 0.0], [0.7, 0.0], [0.5, 0.0], [0.69, 0.0], 0.0),
            # bystander knocked 10 cm: only the replay predicts it
            _obj("cup", [0.4, 0.2], [0.4, 0.2], [0.4, 0.3], [0.4, 0.29], [0.4, 0.2], 9.0)],
            grasped=["cube"], real=True, rule=True))
    s = dv.score(recs, n_boot=200)
    sub = s["subsets"]
    assert sub["all"]["n_objects"] == 12 and sub["moved"]["n_objects"] == 12
    assert sub["collateral"]["n_objects"] == 6                      # the knocked cups
    assert sub["collateral"]["replay_vs_stay_put"]["verdict"] == "replay better"
    assert sub["collateral"]["replay_vs_release_point"]["verdict"] == "replay better"
    assert s["moved_agreement"]["stay_put"]["agree"] == 0
    assert s["moved_agreement"]["release_point"]["agree"] == 6
    assert s["success_agreement"]["gripper_rule"] == 6 and s["success_agreement"]["replay"] == 0


def test_cluster_bootstrap_is_deterministic_and_episode_level():
    rows = [{"episode": "a", "x": 1.0}, {"episode": "a", "x": 1.0}, {"episode": "b", "x": 3.0}]
    ci1 = dv.cluster_bootstrap(rows, lambda s: np.mean([r["x"] for r in s]), n=500)
    ci2 = dv.cluster_bootstrap(rows, lambda s: np.mean([r["x"] for r in s]), n=500)
    assert ci1 == ci2 and ci1[0] == 1.0 and ci1[1] == 3.0


V2 = json.loads((DATA / "episodes_v2.json").read_text())["episodes"]


@pytest.mark.parametrize("ep", V2, ids=[e["slug"] for e in V2])
def test_v2_episode_packaged_without_private_files(ep):
    d = DATA / ep["slug"]
    names = {p.name for p in d.iterdir()}
    assert {"pre.png", "post.png", "clip.mp4", "camera.json", "trajectory.json", "label.json"} <= names
    assert "metadata.json" not in names and not any(n.endswith("-stereo.mp4") for n in names)
    lab = json.loads((d / "label.json").read_text())
    assert lab["episode_id"] == ep["eid"]
    assert not any("operator" in k and k != "current_task_operator" for k in lab)   # no operator name field
    tr = json.loads((d / "trajectory.json").read_text())
    assert len(tr["joint_positions"]) == len(tr["gripper_command"]) == tr["n_steps"]


def test_v2_size_and_labs():
    total = sum(p.stat().st_size for e in V2 for p in (DATA / e["slug"]).iterdir())
    assert total < 150e6
    assert len({e["eid"].split("+")[0] for e in V2}) >= 3
