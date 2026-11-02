import json
from pathlib import Path
from collections import Counter
from rehearsal.predicates import PREDICATES

ROOT = Path(__file__).resolve().parent.parent
CASES = json.loads((ROOT / "bench/holdout/cases.json").read_text())


def _ids(scene):
    return {o["id"] for o in json.loads((ROOT / "scenes" / f"{scene}.json").read_text())["objects"]}


def _atoms(c):
    out = list(c.get("required", [])) + list(c.get("forbidden", []))
    for alt in c.get("acceptable_alternatives", []) + c.get("plausible_readings", []):
        out += alt
    return out


def test_shape():
    assert len(CASES) == 30
    assert len({c["id"] for c in CASES}) == 30
    assert len({c["scene"] for c in CASES}) >= 7
    assert Counter(c["lang"] for c in CASES)["zh"] == 6
    for c in CASES:
        assert c["expected_status"] in ("goals", "clarify", "refuse")
        assert c["rationale"]


def test_ids_and_preds():
    for c in CASES:
        ids = _ids(c["scene"])
        for a in _atoms(c):
            assert a["pred"] in PREDICATES, (c["id"], a)
            assert a["args"] and set(a["args"]) <= ids, (c["id"], a)
        if c["expected_status"] == "goals":
            assert c["required"]
