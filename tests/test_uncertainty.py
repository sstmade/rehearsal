"""Per-object perception sigma (1) and held-out evaluation of active perception (2)."""
import json

import rehearsal.rehearse as R

from .conftest import REPO

DESK = json.loads((REPO / "scenes" / "desk_01.json").read_text())
EMPTY = {"diagnosis": "", "actions": []}


def with_sigma(scene, **sig):
    s = json.loads(json.dumps(scene))
    for o in s["objects"]:
        if o["id"] in sig:
            o["perception"] = {"sigma_m": sig[o["id"]]}
    return s


def test_object_sigmas_fallback():
    s = with_sigma(DESK, mug_1=0.02, bowl_1=0.001)
    sig = R.object_sigmas(s)
    assert sig[0] == 0.02 and sig[1] == 0.001 and sig[2:] == [R.SIGMA] * (len(DESK["objects"]) - 2)
    assert R.object_sigmas(DESK) == [R.SIGMA] * len(DESK["objects"])


def test_stress_test_uses_scene_sigma_and_seed_block():
    tiny = with_sigma(DESK, **{o["id"]: 1e-6 for o in DESK["objects"]})
    st = R.stress_test(tiny, EMPTY, [], 3, seed0=1000)
    assert [r["seed"] for r in st["runs"]] == [1000, 1001, 1002] and st["seed0"] == 1000
    assert all(r["max_perception_error_cm"] < 0.01 for r in st["runs"])
    assert abs(st["sigma_cm"]["mug_1"] - 1e-4) < 1e-12
    big = with_sigma(DESK, mug_1=0.05)
    st = R.stress_test(big, EMPTY, [], 3, seed0=1000)
    # same seeds -> same unit draws, scaled by sigma: mug error is 0.05/0.012 times the default
    base = R.stress_test(DESK, EMPTY, [], 3, seed0=1000)
    for a, b in zip(st["runs"], base["runs"]):
        assert abs(a["perception_error_cm"]["mug_1"] / b["perception_error_cm"]["mug_1"] - 0.05 / 0.012) < 1e-6
        assert abs(a["perception_error_cm"]["bowl_1"] - b["perception_error_cm"]["bowl_1"]) < 1e-9


def fake_stress(calls):
    """mug_1 at full sigma fails 2 selection runs but only 1 held-out run; nothing else matters."""
    def stress_test(scene, plan, goals, n, sigmas=None, pool=None, trails=False, on_run=None, seed0=0):
        sigmas = R.object_sigmas(scene) if sigmas is None else list(sigmas)
        calls.append((seed0, tuple(sigmas)))
        fails = 0
        if sigmas[0] > R.SIGMA_REFINED:
            fails = 2 if seed0 == 0 else 1
        runs = [{"seed": seed0 + i, "success": i >= fails, "max_perception_error_cm": 1.0, "friction_scale": 1.0,
                 "failed": ["x"], **({"trails": {}} if trails else {})} for i in range(n)]
        return {"n": n, "successes": n - fails, "runs": runs, "seed0": seed0, "sigma_cm": {}}
    return stress_test


def test_active_perception_reports_selection_and_heldout(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(R, "stress_test", fake_stress(calls))
    scene = with_sigma(DESK, bowl_1=0.02)
    p = tmp_path / "s.json"
    p.write_text(json.dumps(scene))
    rep = R.rehearse(str(p), "把杯子放进碗里", 5, str(tmp_path / "out"), log=lambda *_: None, look_rounds=2)
    st, ap = rep["stress_test"], rep["active_perception"]
    assert st["seed0"] == R.HELDOUT_SEED0 and st["successes"] == 4
    assert ap["steps"][0]["look_at"] == "mug_1" and ap["steps"][0]["before"] == 3 and ap["steps"][0]["after"] == 5
    assert ap["selection"] == {"seed0": 0, "n": 5, "before": 3, "after": 5}
    assert ap["heldout"] == {"seed0": R.HELDOUT_SEED0, "n": 5, "before": 4, "after": 5}
    assert ap["final"]["successes"] == 5 and ap["final"]["seed0"] == R.HELDOUT_SEED0
    assert ap["selection_final"]["seed0"] == 0
    assert ap["sigma_cm"]["bowl_1"] == 2.0 and ap["sigma_cm"]["mug_1"] == 1.2
    assert ap["initial_sigma_cm"] == ap["sigma_cm"] and isinstance(ap["sigma_cm_mean"], float)
    # every candidate evaluation happened on the selection block, using the scene's sigmas
    candidate_calls = [c for c in calls if c[0] == 0]
    assert len(candidate_calls) >= len(DESK["objects"]) + 1
    assert all(c[1][1] in (0.02, R.SIGMA_REFINED) for c in calls)
    out = tmp_path / "out" / "desk_01"
    assert (out / "stress_trails.json").exists() and (out / "stress_trails_after.json").exists()
    assert "Held-out check" in (out / "report.md").read_text()
