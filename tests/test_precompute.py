import json

from rehearsal.rehearse import precompute

from .conftest import DESK_FIXTURE, REPO


def test_precompute_runs_only_demo_scenes(tmp_path):
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    desk = json.loads(DESK_FIXTURE.read_text())
    desk = {k: v for k, v in desk.items() if not k.startswith("demo_") and k != "reference"}
    (scenes / "plain.json").write_text(json.dumps(dict(desk, name="plain")))
    (scenes / "demo.json").write_text(json.dumps(dict(
        desk, name="demo", privacy="public", demo_instruction="Put the mug in the bowl",
        demo_instruction_zh="把杯子放进碗里",
        demo_stress=2, demo_look=0)))
    lines = []
    res = precompute(str(scenes), str(tmp_path / "out"), log=lines.append)
    assert [r["scene"] for r in res] == ["demo"] and res[0]["success"] and res[0]["stress"].endswith("/2")
    assert any("data=public" in line for line in lines)
    rep = json.loads((tmp_path / "out" / "demo" / "report.json").read_text())
    assert rep["instruction"] == "把杯子放进碗里"   # offline, no reference -> Chinese fallback
    assert not (tmp_path / "out" / "plain").exists()


def test_demo_instruction_choice(monkeypatch):
    from rehearsal.rehearse import demo_instruction
    monkeypatch.setenv("REHEARSAL_OFFLINE", "1")
    s = {"demo_instruction": "Put the mug on the coaster", "demo_instruction_zh": "把杯子放到杯垫上"}
    assert demo_instruction(s) == "把杯子放到杯垫上"
    ref = dict(s, reference={"goals": [], "naive_plan": {}, "fixed_plan": {}})
    assert demo_instruction(ref) == "Put the mug on the coaster"   # OfflineAgent replays the reference
    assert demo_instruction({"demo_instruction": "x"}) == "x"
