"""Rehearsal Bench: exporter (consent rules, case format), scorer and the server endpoints. Offline."""
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

from tests.conftest import DESK_FIXTURE, REPO

sys.path.insert(0, str(REPO / "scripts"))
import bench_export as E  # noqa: E402

from bench import caselib as C  # noqa: E402
from bench.score import render_card, score_plan  # noqa: E402

GOALS = [{"pred": "inside", "args": ["mug_1", "bowl_1"]},
         {"pred": "stacked", "args": ["book_large", "book_medium", "book_small"]}]
LOW = {"actions": [{"obj": "mug_1", "dest": "bowl_1", "relation": "in", "transit_height": 0.15, "via": []},
                   {"obj": "book_medium", "dest": "book_large", "relation": "on", "transit_height": 0.15, "via": []},
                   {"obj": "book_small", "dest": "book_medium", "relation": "on", "transit_height": 0.15, "via": []}]}
HIGH = {"actions": [dict(a, transit_height=0.42) for a in LOW["actions"]]}


def _attempt(scene, plan, goals, k):
    from rehearsal.executor import run_plan
    r = run_plan(scene, plan, goals)
    r.pop("frames"), r.pop("trajectory")
    return {"attempt": k, "plan": plan, **r}


@pytest.fixture(scope="module")
def desk_attempts():
    scene = json.loads(DESK_FIXTURE.read_text())
    goals = C.with_implicit_goals([dict(g) for g in GOALS], [o["id"] for o in scene["objects"]])
    return scene, goals, [_attempt(scene, LOW, goals, 1), _attempt(scene, HIGH, goals, 2)]


def _report(tmp_path, desk_attempts, name, privacy):
    scene, goals, atts = desk_attempts
    spec = dict(scene, name=name, privacy=privacy)
    spec.pop("reference", None)
    d = tmp_path / "out" / name
    d.mkdir(parents=True)
    from PIL import Image
    Image.new("RGB", (640, 480), (90, 70, 50)).save(d / "thumb.png")
    rep = {"scene": name, "scene_spec": spec, "instruction": "Put the mug in the bowl, then stack the books.",
           "agent": "test-agent", "reachability": {}, "goals": goals, "attempts": atts,
           "stress_test": None, "active_perception": None, "wall_time_s": 1.0}
    (d / "report.json").write_text(json.dumps(rep))
    return d / "report.json"


def _export(rp, out, **kw):
    return E.export_case(str(rp), out, eval_files=[], log=lambda *_: None, **kw)


def test_upload_needs_owner_consent(tmp_path, desk_attempts):
    rp = _report(tmp_path, desk_attempts, "photo_abc123", "public")
    with pytest.raises(E.ExportRefused, match="owner-consent"):
        _export(rp, tmp_path / "cases")
    assert not (tmp_path / "cases" / "photo_abc123").exists()
    d = _export(rp, tmp_path / "cases", owner_consent=True)
    meta = json.loads((d / "case.json").read_text())
    assert meta["provenance"]["kind"] == "user_photo" and meta["privacy"]["owner_consent"] is True
    assert meta["provenance"]["photo_original"] is None       # never leak the uploads/ path


def test_private_scene_refused_by_default(tmp_path, desk_attempts):
    rp = _report(tmp_path, desk_attempts, "photo_priv01", "private")
    for kw in ({}, {"owner_consent": True}, {"allow_private": True}):
        with pytest.raises(E.ExportRefused):
            _export(rp, tmp_path / "cases", **kw)
    assert _export(rp, tmp_path / "cases", owner_consent=True, allow_private=True).is_dir()


def test_privacy_absent_counts_as_private(tmp_path, desk_attempts):
    rp = _report(tmp_path, desk_attempts, "desk_noprivacy", "public")
    rep = json.loads(rp.read_text())
    rep["scene_spec"].pop("privacy")
    rp.write_text(json.dumps(rep))
    with pytest.raises(E.ExportRefused):
        _export(rp, tmp_path / "cases")


def test_cli_refusal_exit_code(tmp_path, desk_attempts):
    rp = _report(tmp_path, desk_attempts, "photo_cli01", "private")
    assert E.main([str(rp), "--out", str(tmp_path / "cases")]) == 2


def test_exported_case_matches_schema(tmp_path, desk_attempts):
    rp = _report(tmp_path, desk_attempts, "desk_test", "public")
    d = _export(rp, tmp_path / "cases")
    meta = json.loads((d / "case.json").read_text())
    assert C.validate(meta) == []
    assert meta["schema"] == "rehearsal-bench-case" and meta["schema_version"] == C.SCHEMA_VERSION
    assert meta["provenance"]["kind"] == "hand_designed" and meta["provenance"]["photo_is_render"]
    assert (d / "photo.jpg").stat().st_size <= 300_000
    assert set(meta["perception"]["sigma_m"]) == {o["id"] for o in json.loads((d / "scene.json").read_text())["objects"]}
    assert meta["goals_source"] == "planner_compiled"
    fails = [json.loads(line) for line in (d / "failures.jsonl").read_text().splitlines()]
    n_failed = sum(not a["success"] for a in desk_attempts[2])
    assert len(fails) == meta["failures"]["count"] == n_failed
    for f in fails:
        assert C.validate(f, ref="#/$defs/failure") == []
        assert f["contacts"] == C.contact_log([e for e in f["contacts"]])
    for k in ("muse_first", "muse_after_log", "scripted_baseline"):
        assert (d / meta["files"]["plans"][k]).is_file()


def test_schema_rejects_bad_case():
    errs = C.validate({"schema": "rehearsal-bench-case", "schema_version": "9", "id": "../x"})
    assert any("schema_version" in e for e in errs) and any("id" in e for e in errs)
    assert any("missing 'goals'" in e for e in errs)


def test_scripted_baseline_order():
    goals = [{"pred": "stacked", "args": ["a", "b", "c"]}, {"pred": "on", "args": ["cup", "coaster"]},
             {"pred": "upright", "args": ["cup"]}, {"pred": "undisturbed", "args": ["x"], "implicit": True}]
    acts = C.scripted_baseline(goals)["actions"]
    assert [(a["obj"], a["dest"]) for a in acts] == [("b", "a"), ("c", "b"), ("cup", "coaster")]
    assert {a["transit_height"] for a in acts} == {C.BASELINE_TRANSIT}


def test_score_card_deterministic_and_robust_to_bad_plans(tmp_path, desk_attempts):
    rp = _report(tmp_path, desk_attempts, "desk_score", "public")
    d = _export(rp, tmp_path / "cases")
    case = C.load_case(d)
    with C.pool(2) as ex:
        a = score_plan(case, HIGH, n=3, ex=ex)
        b = score_plan(case, HIGH, n=3, ex=ex)
        bad = score_plan(case, {"actions": [{"obj": "unicorn", "dest": "bowl_1", "relation": "on",
                                             "transit_height": 0.3}]}, n=2, ex=ex)
        junk = score_plan(case, {"nope": 1}, n=0, ex=ex)
    assert a["robustness"] == b["robustness"] and a["nominal"] == b["nominal"]
    assert a["robustness"]["n"] == 3 and a["robustness"]["seed0"] == C.SEED0
    assert a["score"] == pytest.approx(a["robustness"]["k"] / 3 if a["nominal"]["pass"] else 0, abs=1e-3)
    assert not bad["nominal"]["pass"] and "invalid plan" in bad["nominal"]["execution_error"] and bad["score"] == 0
    assert bad["robustness"]["k"] == 0
    assert not junk["nominal"]["pass"]
    assert "robustness" in render_card(a)


def test_noise_matches_product_stress_test():
    """The bench's noise draw must equal rehearse._variant's, so bench scores reproduce report numbers."""
    import numpy as np
    scene = json.loads(DESK_FIXTURE.read_text())
    sig = C.object_sigmas(scene)
    rng = np.random.default_rng(1007)
    f, m = rng.uniform(0.6, 1.4), rng.uniform(0.7, 1.3)
    pe = rng.normal(0, 1, (len(sig), 2)) * np.asarray(sig)[:, None]
    v = C.noise(1007, sig)
    assert v["friction_scale"] == f and v["mass_scale"] == m and np.array_equal(v["perception_error"], pe)


def test_server_bench_endpoints(server, root, tmp_path, desk_attempts):
    client, _ = server
    s, body = client.get("/api/bench")
    assert s == 200 and body["n_cases"] == 0
    rp = _report(tmp_path, desk_attempts, "desk_srv", "public")
    _export(rp, root / "bench" / "cases")
    shutil.copy(C.SCHEMA_PATH, root / "bench" / "schema.json")
    s, body = client.get("/api/bench")
    assert s == 200 and body["n_cases"] == 1 and body["cases"][0]["download"] == "/api/bench/case?id=desk_srv"
    s, ct, data = client.raw("GET", "/api/bench/case?id=desk_srv")
    assert s == 200 and ct == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert {"desk_srv/case.json", "desk_srv/scene.json", "desk_srv/photo.jpg", "desk_srv/schema.json"} <= set(names)
    assert client.get("/api/bench/case?id=..%2Fsecret")[0] == 400
    assert client.get("/api/bench/case?id=nope")[0] == 404


def test_leaderboard_on_one_case(tmp_path, desk_attempts):
    from bench.leaderboard import build, render_md
    rp = _report(tmp_path, desk_attempts, "desk_lb", "public")
    _export(rp, tmp_path / "cases")
    lb = build(2, tmp_path / "cases", log=lambda *_: None)
    assert lb["n_cases"] == 1 and set(lb["totals"]) == {"hybrid", "muse_first", "muse_after_log", "scripted_baseline"}
    assert lb["cases"][0]["cells"]["hybrid"] is None and lb["totals"]["hybrid"]["cases"] == 0   # no plans/hybrid.json
    cell = lb["cases"][0]["cells"]["muse_after_log"]
    assert cell["n"] == 2 and 0 <= cell["k"] <= 2
    md = render_md(lb)
    assert "`desk_lb`" in md and "**total**" in md
