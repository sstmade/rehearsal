"""Intent step: the shipped compile prompt (GoalBench arm C), the refusal/clarify path, implicit protection and the
deterministic intent gate. Offline; no Meta API calls."""
import json

import pytest

from tests.conftest import DESK_FIXTURE, REPO

from rehearsal import agents as A
from rehearsal import intent_gate
from rehearsal.predicates import moved_objects, with_implicit_goals

DESK = json.loads(DESK_FIXTURE.read_text())
DESK_IDS = [o["id"] for o in DESK["objects"]]


def g(pred, *args):
    return {"pred": pred, "args": list(args)}


# ------------------------------------------------------------------ 1. compile step = arm C
def _stub_muse(monkeypatch, reply):
    agent = A.MuseAgent.__new__(A.MuseAgent)
    seen = {}
    monkeypatch.setattr(agent, "_json", lambda user, schema, name: seen.update(user=user, schema=schema, name=name)
                        or reply, raising=False)
    return agent, seen


def test_compile_intent_is_byte_identical_to_goalbench_arm_c(monkeypatch):
    from bench.goals import run as R
    monkeypatch.delenv("REHEARSAL_COMPILE", raising=False)
    agent, seen = _stub_muse(monkeypatch, {"status": "ok", "assumptions": ["a"], "question": "", "goals": []})
    summary = [{"id": "mug_1", "label": "mug"}]
    agent.compile_intent("put it away", summary)
    assert seen["user"] == R.user_prompt("C", {"instruction": "put it away"}, summary, 0)
    assert seen["schema"] is R.ARMS["C"][1] is A.INTENT_SCHEMA and seen["name"] == R.ARMS["C"][2]


def test_legacy_compile_flag_runs_arm_a(monkeypatch):
    from bench.goals import run as R
    monkeypatch.setenv("REHEARSAL_COMPILE", "legacy")
    agent, seen = _stub_muse(monkeypatch, {"goals": [g("on", "mug_1", "bowl_1")]})
    out = agent.compile_intent("put it away", [{"id": "mug_1"}])
    assert seen["user"] == R.user_prompt("A", {"instruction": "put it away"}, [{"id": "mug_1"}], 0)
    assert seen["schema"] is A.GOAL_SCHEMA
    assert out["status"] == "goals" and out["compile"] == "legacy" and out["goals"] == [g("on", "mug_1", "bowl_1")]


@pytest.mark.parametrize("raw,status,question,reason", [
    ({"status": "ok", "assumptions": ["x"], "question": "", "goals": [g("on", "a", "b")]}, "goals", "", ""),
    ({"status": "clarify", "assumptions": [], "question": "Which box?", "goals": [g("on", "a", "b")]},
     "clarify", "Which box?", ""),
    ({"status": "refuse", "assumptions": [], "question": "There is no fridge.", "goals": []},
     "refuse", "", "There is no fridge."),
])
def test_normalize_intent(raw, status, question, reason):
    it = A.normalize_intent(raw)
    assert (it["status"], it["question"], it["reason"]) == (status, question, reason)
    assert it["goals"] == (raw["goals"] if status == "goals" else [])
    if status == "clarify":
        assert it["proposed_goals"] == raw["goals"]      # kept for display, never executed


# ------------------------------------------------------------------ 2. refusal path
class StubAgent:
    name = "stub"

    def __init__(self, intent):
        self.intent = intent
        self.planned = False

    def compile_intent(self, instruction, summary):
        return A.normalize_intent(self.intent)

    def plan(self, *a):
        self.planned = True
        raise AssertionError("must not plan")

    revise = plan


@pytest.mark.parametrize("raw,status,by", [
    ({"status": "refuse", "assumptions": [], "question": "There is no fridge in the scene.", "goals": []},
     "refused", "model"),
    ({"status": "clarify", "assumptions": ["several books"], "question": "Which book?", "goals": []},
     "needs_clarification", "model"),
    ({"status": "ok", "assumptions": [], "question": "", "goals": []}, "refused", "gate"),            # empty
    ({"status": "ok", "assumptions": [], "question": "", "goals": [g("inside", "mug_1", "fridge_1")]},
     "refused", "gate"),                                                                               # unknown id
    ({"status": "ok", "assumptions": [], "question": "", "goals": [g("upright", "mug_1"), g("stable", "mug_1")]},
     "needs_clarification", "gate"),                                                                   # vacuous
])
def test_rehearse_stops_without_planning(tmp_path, monkeypatch, raw, status, by):
    from rehearsal import rehearse as R
    stub = StubAgent(raw)
    monkeypatch.setattr(R, "make_agent", lambda **kw: stub)
    rep = R.rehearse(str(DESK_FIXTURE), "put the cup in the fridge", 4, str(tmp_path), log=lambda *_: None,
                     look_rounds=0)
    assert rep["status"] == status and rep["intent"]["decided_by"] == by
    assert rep["attempts"] == [] and rep["stress_test"] is None and rep["goals"] == [] and not stub.planned
    assert rep["message"]
    saved = json.loads((tmp_path / "desk_01" / "report.json").read_text())
    assert saved["status"] == status and saved["message"] == rep["message"]
    md = (tmp_path / "desk_01" / "report.md").read_text()
    assert "Nothing was planned or simulated" in md


def test_refused_job_is_not_a_success(tmp_path, monkeypatch):
    import shutil
    from rehearsal import pipeline, rehearse as R
    (tmp_path / "scenes").mkdir()
    (tmp_path / "out").mkdir()
    shutil.copy(DESK_FIXTURE, tmp_path / "scenes" / "desk_01.json")
    monkeypatch.setattr(R, "make_agent", lambda **kw: StubAgent(
        {"status": "refuse", "assumptions": [], "question": "no fridge", "goals": []}))
    mgr = pipeline.JobManager()
    jid = mgr.submit("rehearse", pipeline.rehearse_job(tmp_path, "desk_01", "fridge", 4, 0))
    v = mgr.wait(jid, 60)
    assert v["state"] == "done", v["log"][-10:]
    res = v["result"]
    assert res["success"] is False and res["status"] == "refused" and res["reason"] == "no fridge"
    assert res["attempts"] == 0 and res["planner"] == "hybrid"


def test_confirmed_overrides_gate_clarify_only():
    from rehearsal.rehearse import compile_step
    off = A.OfflineAgent(DESK)
    vac = StubAgent({"status": "ok", "assumptions": [], "question": "", "goals": [g("upright", "mug_1")]})
    assert compile_step(vac, DESK, "x", [], log=lambda *_: None)["outcome"] == "needs_clarification"
    assert compile_step(vac, DESK, "x", [], confirmed=True, log=lambda *_: None)["outcome"] == "rehearsed"
    ref = StubAgent({"status": "ok", "assumptions": [], "question": "", "goals": [g("on", "mug_1", "shelf_9")]})
    assert compile_step(ref, DESK, "x", [], confirmed=True, log=lambda *_: None)["outcome"] == "refused"
    assert compile_step(off, DESK, "把杯子放进碗里", [{"id": i, "shape": {"size": [0.1, 0.1, 0.1]}} for i in DESK_IDS],
                        log=lambda *_: None)["outcome"] == "rehearsed"


# ------------------------------------------------------------------ 3. implicit protection
def test_mentioned_only_in_upright_or_stable_stays_protected():
    goals = [g("inside", "mug_1", "bowl_1"), g("upright", "bottle_1"), g("stable", "plant_pot")]
    implicit = with_implicit_goals(goals, DESK_IDS)[-1]
    assert implicit["implicit"] and implicit["pred"] == "undisturbed"
    assert {"bottle_1", "plant_pot", "bowl_1"} <= set(implicit["args"])      # destination stays put too
    assert "mug_1" not in implicit["args"]
    legacy = with_implicit_goals(goals, DESK_IDS, protect_mentioned=False)[-1]
    assert "bottle_1" not in legacy["args"] and "bowl_1" not in legacy["args"]


def test_moved_objects_and_stacks():
    goals = [g("stacked", "book_large", "book_medium", "book_small"), g("near", "mug_1", "bottle_1")]
    assert moved_objects(goals) == {"book_medium", "book_small", "mug_1"}
    imp = with_implicit_goals(goals, DESK_IDS)[-1]["args"]
    assert "book_large" in imp and "bottle_1" in imp and "book_small" not in imp


def test_explicit_undisturbed_not_repeated():
    goals = [g("on", "mug_1", "book_large"), g("undisturbed", "bottle_1")]
    imp = with_implicit_goals(goals, DESK_IDS)[-1]["args"]
    assert "bottle_1" not in imp and "book_large" in imp


def test_goalbench_scorer_still_uses_legacy_protection():
    from bench.goals import score as S
    assert S.PROTECT_MENTIONED is False      # published GoalBench numbers stay reproducible


# ------------------------------------------------------------------ 4. intent gate
def test_gate_flags():
    def verdict(goals):
        r = intent_gate.check(goals, DESK)
        return r["verdict"], [f["code"] for f in r["flags"]]
    assert verdict([]) == ("refuse", ["empty"])
    assert verdict([g("undisturbed", "mug_1")]) == ("refuse", ["empty"])
    assert verdict([g("inside", "mug_1", "fridge_1")])[1] == ["unknown_object"]
    assert verdict([g("on", "mug_1")]) == ("refuse", ["malformed"])
    assert verdict([g("upright", "mug_1"), g("stable", "mug_1")]) == ("clarify", ["vacuous"])
    assert verdict([g("on", "mug_1", "book_large"), g("on", "mug_1", "book_small")]) == ("clarify", ["contradiction"])
    assert verdict([g("on", "mug_1", "book_large"), g("clear", "book_large")]) == ("clarify", ["contradiction"])
    assert verdict([g("inside", "mug_1", "bowl_1"), g("upright", "mug_1")]) == ("pass", [])


def test_gate_vacuous_uses_physics_not_syntax():
    # the stack does not hold now, so this is a real request even though it only uses stacked()
    assert intent_gate.check([g("stacked", "book_large", "book_medium")], DESK)["verdict"] == "pass"


def test_effect_preview():
    pv = intent_gate.effect_preview([g("stacked", "book_large", "book_medium", "book_small"), g("upright", "mug_1"),
                                     g("clear", "bowl_1")], DESK)
    assert [(m["obj"], m["dest"]) for m in pv["moves"]] == [("book_medium", "book_large"), ("book_small", "book_medium")]
    assert set(pv["stays"]) == set(DESK_IDS) - {"book_medium", "book_small"}
    assert len(pv["conditions"]) == 2 and "must not move" in pv["stays_text"]


# ------------------------------------------------------------------ 7. hybrid planner is the default
def test_hybrid_is_the_default_planner():
    import inspect
    from rehearsal import pipeline, rehearse as R
    assert inspect.signature(R.rehearse).parameters["planner"].default == "hybrid" == R.DEFAULT_PLANNER
    assert inspect.signature(pipeline.rehearse_job).parameters["planner"].default == "hybrid"
    assert R.ROBUST_N == 20


def test_holdout_scorer_rules():
    from bench.holdout import run as H
    case = {"id": "x", "required": [g("on", "mug_1", "book_large"), g("undisturbed", "bottle_1")],
            "acceptable_alternatives": [], "forbidden": [g("on", "mug_1", "bowl_1")], "expected_status": "goals"}
    assert H.score_goals(case, [g("on", "mug_1", "book_large")], DESK_IDS) == (True, [])
    assert H.score_goals(case, [g("on", "mug_1", "book_large"), g("upright", "bottle_1")], DESK_IDS,
                         protect_mentioned=False)[1] == ["dropped_protection"]
    assert "extra_action" in H.score_goals(case, [g("on", "mug_1", "book_large"), g("on", "bottle_1", "bowl_1")],
                                           DESK_IDS)[1]
    assert H.system_status({"goals": []}) == "refuse" and H.system_status({"status": "clarify"}) == "clarify"
    order = dict(case, order_matters=True, required=[g("on", "mug_1", "book_large"), g("on", "book_small", "bowl_1")],
                 forbidden=[])
    assert H.score_goals(order, [g("on", "book_small", "bowl_1"), g("on", "mug_1", "book_large")],
                         DESK_IDS)[1] == ["wrong_order"]


def test_api_intent_endpoint(server):
    c, srv = server
    assert c.post("/api/intent", {"scene": "nope", "instruction": "x"})[0] == 404
    assert c.post("/api/intent", {"scene": "desk_01", "instruction": ""})[0] == 400
    # Offline server, empty budget dir, frozen test fixture (not the shipped desk_01): no recorded Muse Spark answer
    # exists for this request, so the endpoint says so instead of inventing goals.
    s, r = c.post("/api/intent", {"scene": "desk_01", "instruction": "把杯子放进碗里。"})
    assert s == 200
    v = srv.mgr.wait(r["job"], 60)
    assert v["state"] == "done", v["log"][-10:]
    res = v["result"]
    assert res["status"] == "no_recording" and res["intent"]["goals"] == [] and res["agent"] is None
    assert c.post("/api/rehearse", {"scene": "desk_01", "instruction": "x", "confirmed": "yes"})[0] == 400


def _intent(c, srv, scene, instruction):
    s, r = c.post("/api/intent", {"scene": scene, "instruction": instruction})
    assert s == 200
    v = srv.mgr.wait(r["job"], 60)
    assert v["state"] == "done", v["log"][-10:]
    return v["result"]


@pytest.fixture
def live_scene_server(root, budget_dir, server):
    """The API server over the shipped desk_01 / kitchen_01 scenes (the replay file is keyed by their summaries)."""
    import shutil
    import threading
    from pathlib import Path
    from rehearsal.pipeline import JobManager
    from rehearsal.server import make_server
    for s in ("desk_01", "kitchen_01"):
        shutil.copy(Path(__file__).resolve().parents[1] / "scenes" / f"{s}.json", root / "scenes" / f"{s}.json")
    srv = make_server("127.0.0.1", 0, root=root, mgr=JobManager())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield type(server[0])(srv.server_address[1]), srv
    srv.shutdown()
    srv.server_close()


def test_api_intent_offline_replays_recorded_muse_ask_and_refuse(live_scene_server):
    """R6: the ask / refuse step on the bundled tables comes from Muse Spark's real recorded responses (same request,
    same cache key as a live call), with no API call; an unrecorded instruction is reported, never invented."""
    import json
    from pathlib import Path
    c, srv = live_scene_server
    raw = {(r["case"], r["rep"]): r["output"] for r in map(json.loads, (Path(__file__).resolve().parents[1]
           / "bench/goals/results/raw_C.jsonl").read_text().splitlines())}
    ask = _intent(c, srv, "desk_01", "Put the book in the bowl.")
    assert ask["status"] == "needs_clarification" and ask["intent"]["status"] == "clarify"
    assert ask["intent"]["question"] == raw[("a05", 0)]["question"]
    assert ask["intent"]["assumptions"] == raw[("a05", 0)]["assumptions"]
    assert ask["intent"]["recording"]["cases"] == ["GoalBench a05"]
    no = _intent(c, srv, "kitchen_01", "Put the cup in the fridge.")
    assert no["status"] == "refused" and no["intent"]["reason"] == raw[("f02", 0)]["question"]
    miss = _intent(c, srv, "kitchen_01", "Put the banana in the sink.")
    assert miss["status"] == "no_recording" and miss["intent"]["goals"] == [] and miss["agent"] is None
    assert "Put the cup in the fridge." in miss["intent"]["recorded_instructions"]
    ok = _intent(c, srv, "desk_01", "Put the mug in the bowl.")
    assert ok["status"] == "ready" and ok["intent"]["gate"]["verdict"] == "pass"
    assert ok["intent"]["gate"]["preview"]["moves"][0]["obj"] == "mug_1"


def test_cache_only_never_calls_the_api(monkeypatch, budget_dir):
    from rehearsal import meta_client as M
    monkeypatch.setattr(M, "client", lambda: (_ for _ in ()).throw(AssertionError("API contacted")))
    with pytest.raises(M.CacheMiss):
        M.chat_json([{"role": "user", "content": "not recorded"}], {"type": "object"}, "X", caller="t",
                    cache_only=True)


def test_replay_file_matches_product_request_keys():
    """Every shipped replay entry is keyed by the request the product would send for its scene + instruction."""
    import json
    from pathlib import Path
    from rehearsal import agents as A
    from rehearsal import meta_client as M
    from rehearsal.executor import reachability_report
    root = Path(__file__).resolve().parents[1]
    entries = M.replay_entries()
    assert len(entries) >= 60
    for scene in ("desk_01", "kitchen_01"):
        spec = json.loads((root / "scenes" / f"{scene}.json").read_text())
        summary = A.scene_summary(spec, reachability_report(spec))
        mine = {k: e for k, e in entries.items() if e["scene"] == scene}
        for k, e in mine.items():
            assert A.intent_request_key(summary, e["instruction"], "public") == k
