"""GoalBench: case set integrity, the goal scorer, its statistics, and the near/clear predicates. Offline."""
import json

import pytest

from tests.conftest import DESK_FIXTURE, REPO

from bench.goals import score as S
from bench.goals.cases import CASES

DESK_IDS = ["mug_1", "bowl_1", "bottle_1", "book_large", "book_medium", "book_small", "plant_pot"]
BY_ID = {c["id"]: c for c in CASES}


def g(pred, *args):
    return {"pred": pred, "args": list(args)}


def trial(cid, goals, vocab=S.EXTENDED_VOCAB, **extra):
    c = BY_ID[cid]
    ids = [o["id"] for o in json.loads((REPO / "scenes" / f"{c['scene']}.json").read_text())["objects"]]
    return S.score_trial(c, {"goals": goals, **extra}, ids, vocab)


# ------------------------------------------------------------------ case set
def test_cases_json_in_sync_and_ids_exist():
    assert json.loads((REPO / "bench/goals/cases.json").read_text()) == CASES
    assert 40 <= len(CASES) <= 60 and len({c["id"] for c in CASES}) == len(CASES)
    for c in CASES:
        ids = {o["id"] for o in json.loads((REPO / "scenes" / f"{c['scene']}.json").read_text())["objects"]}
        assert c["expect"] in ("goals", "flag", "either")
        assert c["refs"] or c["expect"] == "flag", c["id"]
        for ref in c["refs"]:
            for key in ("required", "implicit", "optional", "forbidden"):
                for s in ref[key]:
                    for p, args in S.parse_alts(s):
                        assert p in S.EXTENDED_VOCAB and set(args) <= ids, (c["id"], s)


def test_product_prompt_is_byte_identical(monkeypatch):
    from bench.goals import run as R
    from rehearsal.agents import GOAL_SCHEMA, MuseAgent
    seen = {}
    agent = MuseAgent.__new__(MuseAgent)
    monkeypatch.setattr(agent, "_json", lambda user, schema, name: seen.update(user=user, schema=schema) or {"goals": []},
                        raising=False)
    summary = [{"id": "mug_1", "label": "mug"}]
    MuseAgent.compile_goals(agent, "put it away", summary)
    assert R.user_prompt("A", {"instruction": "put it away"}, summary, 0) == seen["user"]
    assert R.user_prompt("A", {"instruction": "put it away"}, summary, 2) == seen["user"] + "\n\n[eval trial 2]"
    assert R.ARMS["A"][1] is GOAL_SCHEMA


# ------------------------------------------------------------------ normalisation
def test_atoms_expand_stacked_and_add_implicit_undisturbed():
    got = S.atoms_of([g("stacked", "book_large", "book_medium", "book_small")], DESK_IDS)
    assert ("on", ("book_medium", "book_large")) in got and ("on", ("book_small", "book_medium")) in got
    assert {("undisturbed", (x,)) for x in ("mug_1", "bowl_1", "bottle_1", "plant_pot")} <= got
    assert ("undisturbed", ("book_small",)) not in got


# ------------------------------------------------------------------ scorer
def test_correct_compilation_passes_order_insensitive():
    s = trial("o01", [g("stacked", "book_large", "book_medium", "book_small")])
    assert s["pass"] and s["exact"] and s["errors"] == []
    s = trial("o01", [g("stable", "book_small"), g("stacked", "book_large", "book_medium", "book_small")])
    assert s["pass"] and not s["exact"]          # harmless extra: passes, but is not an exact match
    s = trial("o01", [g("on", "book_small", "book_medium"), g("on", "book_medium", "book_large")])
    assert s["pass"]


def test_benign_upright_of_moved_object_is_not_an_error_but_not_exact():
    s = trial("e01", [g("inside", "mug_1", "bowl_1"), g("upright", "mug_1")])
    assert s["pass"] and not s["exact"]


def test_wrong_order():
    s = trial("o01", [g("stacked", "book_small", "book_medium", "book_large")])
    assert not s["pass"] and s["primary"] == "wrong_order"


def test_wrong_relation_and_binding():
    assert trial("e01", [g("on", "mug_1", "bowl_1")])["primary"] == "wrong_relation"
    s = trial("r03", [g("on", "book_medium", "book_large")])
    assert s["primary"] == "wrong_binding" and "extra_action" not in s["errors"]


def test_missed_implicit_upright():
    s = trial("i01", [g("inside", "mug_1", "bowl_1")])
    assert not s["pass"] and s["primary"] == "missed_implicit"
    assert trial("i01", [g("inside", "mug_1", "bowl_1"), g("upright", "mug_1")])["pass"]


def test_mentioning_a_keep_object_strips_its_protection():
    # "leave the cactus where it is": upright(cactus) mentions it, so the product stops adding undisturbed(cactus)
    s = trial("i05", [g("on", "mug_1", "coaster_1"), g("upright", "potted_plant_1")])
    assert not s["pass"] and "dropped_protection" in s["errors"]
    assert trial("i05", [g("on", "mug_1", "coaster_1")])["pass"]                       # implicit keeps it
    assert trial("i05", [g("on", "mug_1", "coaster_1"), g("undisturbed", "potted_plant_1")])["pass"]


def test_extra_action_and_hallucination():
    s = trial("e01", [g("inside", "mug_1", "bowl_1"), g("on", "bottle_1", "book_large")])
    assert "extra_action" in s["errors"]
    s = trial("f02", [g("inside", "cup_1", "fridge_1")])
    assert s["primary"] == "hallucinated_object" and "failed_to_flag" in s["errors"]


def test_flag_cases():
    assert trial("f01", [])["pass"]                                            # empty goals = declined
    assert trial("f01", [g("undisturbed", "mug_1", "bowl_1")])["pass"]         # only "change nothing" = declined
    assert trial("f01", [g("inside", "bowl_1", "mug_1")])["primary"] == "failed_to_flag"
    assert trial("f01", [g("inside", "bowl_1", "mug_1")], status="refuse")["pass"]
    assert trial("e01", [], status="clarify")["primary"] == "unnecessary_flag"


def test_either_accepts_flag_or_a_listed_reading():
    assert trial("a01", [], status="clarify")["pass"]
    assert trial("a01", [g("stacked", "book_large", "book_medium", "book_small")])["pass"]
    s = trial("a01", [g("inside", "mug_1", "bowl_1")])
    assert not s["pass"] and "unflagged_ambiguity" in s["errors"]


def test_vocab_gap_and_forbidden():
    s = trial("r05", [g("on", "soup_can_1", "mouse_1")], vocab=S.PRODUCT_VOCAB)
    assert not s["pass"] and "vocab_gap" in s["errors"] and "forbidden_goal" in s["errors"]
    assert trial("r05", [g("near", "soup_can_1", "mouse_1")])["pass"]
    s = trial("i07", [g("on", "stapler_1", "case_1")], vocab=S.PRODUCT_VOCAB)
    assert s["pass_expressible"] and not s["pass"] and s["errors"] == ["vocab_gap"]
    assert trial("i07", [g("on", "stapler_1", "case_1"), g("clear", "laptop_1")])["pass"]


def test_call_failed():
    assert S.score_trial(BY_ID["e01"], {"error": "boom"}, DESK_IDS)["primary"] == "call_failed"


# ------------------------------------------------------------------ statistics
def test_wilson_and_sign_test():
    lo, hi = S.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert S.wilson(0, 0) == (0.0, 1.0)
    assert S.sign_test(0, 0) == 1.0
    assert S.sign_test(0, 6) == pytest.approx(2 / 64)
    rows_x = [{"case": "a", "rep": 0, "score": {"pass": True}}, {"case": "b", "rep": 0, "score": {"pass": False}}]
    rows_y = [{"case": "a", "rep": 0, "score": {"pass": False}}, {"case": "b", "rep": 0, "score": {"pass": False}}]
    p = S.paired(rows_x, rows_y)
    assert (p["x_only"], p["y_only"], p["neither"], p["n"]) == (1, 0, 1, 2)


# ------------------------------------------------------------------ near / clear in MuJoCo
@pytest.fixture(scope="module")
def desk():
    return json.loads(DESK_FIXTURE.read_text())


def test_near_predicate(desk):
    from rehearsal.predicates import near
    from rehearsal.scene import build_model
    from rehearsal.sim import Sim
    sim = Sim(build_model(desk), desk)
    assert not near(sim, "mug_1", "bottle_1")[0]           # 22 cm apart centre to centre: gap 14.5 cm
    close = json.loads(json.dumps(desk))
    next(o for o in close["objects"] if o["id"] == "mug_1")["pos"] = [0.42, 0.11]
    sim = Sim(build_model(close), close)
    assert near(sim, "mug_1", "bottle_1")[0]


def test_clear_predicate_after_stacking(desk):
    from rehearsal.executor import run_plan
    plan = {"actions": [{"obj": "book_small", "dest": "book_large", "relation": "on", "transit_height": 0.42, "via": []}]}
    goals = [g("clear", "book_large"), g("clear", "book_medium"), g("on", "book_small", "book_large"),
             g("near", "book_small", "book_large")]
    r = run_plan(desk, plan, goals)
    ok = {f"{x['pred']}({','.join(x['args'])})": x["ok"] for x in r["goals"]}
    assert ok == {"clear(book_large)": False, "clear(book_medium)": True, "on(book_small,book_large)": True,
                  "near(book_small,book_large)": False}


def test_stable_of_the_destination_is_harmless():
    s = trial("o01", [g("stacked", "book_large", "book_medium", "book_small"), g("stable", "book_large")])
    assert s["pass"]


# ------------------------------------------------------------------ consequence sim
def test_plan_for_places_lower_objects_first_and_empty_goals_certify_nothing(desk):
    from bench.goals.consequence import plan_for
    from rehearsal.executor import run_plan
    from rehearsal.predicates import with_implicit_goals
    p = plan_for([g("on", "book_small", "book_medium"), g("on", "book_medium", "book_large")], desk, 0.4)
    assert [a["obj"] for a in p["actions"]] == ["book_medium", "book_small"]
    # the product has no refusal path: an empty compiled goal set becomes "nothing moves", which an empty plan passes
    r = run_plan(desk, plan_for([], desk, 0.4), with_implicit_goals([], DESK_IDS))
    assert r["success"]
