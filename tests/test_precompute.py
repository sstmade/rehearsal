import json

from rehearsal.rehearse import precompute

from .conftest import REPO


def test_precompute_runs_only_demo_scenes(tmp_path):
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    desk = json.loads((REPO / "scenes" / "desk_01.json").read_text())
    (scenes / "plain.json").write_text(json.dumps(dict(desk, name="plain")))
    (scenes / "demo.json").write_text(json.dumps(dict(
        desk, name="demo", privacy="public", demo_instruction="把杯子放进碗里",
        demo_stress=2, demo_look=0)))
    lines = []
    res = precompute(str(scenes), str(tmp_path / "out"), log=lines.append)
    assert [r["scene"] for r in res] == ["demo"] and res[0]["success"] and res[0]["stress"] == "2/2"
    assert any("data=public" in line for line in lines)
    assert (tmp_path / "out" / "demo" / "report.json").exists()
    assert not (tmp_path / "out" / "plain").exists()
