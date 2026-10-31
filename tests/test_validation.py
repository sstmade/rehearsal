import copy
import json

from rehearsal.pipeline import validate_scene, valid_name

from .conftest import REPO

DESK = json.loads((REPO / "scenes" / "desk_01.json").read_text())


def scene(**patch):
    s = copy.deepcopy(DESK)
    s.update(patch)
    return s


def test_shipped_scenes_are_valid():
    for p in (REPO / "scenes").glob("*.json"):
        assert validate_scene(json.loads(p.read_text())) == [], p.name


def test_names():
    assert valid_name("desk_01") and valid_name("a-B_9")
    for bad in ["", "../x", "a/b", "a b", "x" * 65, None, 3, "café"]:
        assert not valid_name(bad)


def test_missing_keys():
    assert validate_scene([]) == ["scene must be a JSON object"]
    s = scene()
    del s["table"]
    assert any("table" in e for e in validate_scene(s))
    assert any("objects" in e for e in validate_scene(scene(objects=[])))


def test_duplicate_and_reserved_ids():
    s = scene()
    s["objects"][1]["id"] = s["objects"][0]["id"]
    assert any("duplicate" in e for e in validate_scene(s))
    s = scene()
    s["objects"][0]["id"] = "hand"
    assert any("reserved" in e for e in validate_scene(s))


def test_off_table_and_sizes():
    s = scene()
    s["objects"][0]["pos"] = [5.0, 0.0]
    assert any("not on the table" in e for e in validate_scene(s))
    s = scene()
    s["objects"][0]["shape"] = {"type": "cylinder", "radius": 2.0, "height": 0.1}
    assert any("radius" in e for e in validate_scene(s))
    s = scene()
    s["objects"][3]["shape"] = {"type": "box", "size": [0.1, 0.1]}
    assert any("size" in e for e in validate_scene(s))
    s = scene()
    s["objects"][0]["shape"] = {"type": "torus"}
    assert any("shape.type" in e for e in validate_scene(s))
    s = scene(table={"center": [0.45, 0], "size": [100, 1]})
    assert any("table size" in e for e in validate_scene(s))


def test_numbers_and_privacy():
    s = scene()
    s["objects"][0]["mass_kg"] = {"nominal": float("nan")}
    assert any("mass" in e for e in validate_scene(s))
    s = scene()
    s["objects"][0]["pos"] = [True, 0.1]
    assert any("pos" in e for e in validate_scene(s))
    assert any("privacy" in e for e in validate_scene(scene(privacy="secret")))
    assert validate_scene(scene(privacy="public")) == []
