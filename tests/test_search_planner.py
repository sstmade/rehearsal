"""search_planner: assignments, goal atoms, orders, and end-to-end searches on bundled scenes."""
import json
from pathlib import Path

import pytest

from rehearsal import rehearse as R
from rehearsal.agents import OfflineAgent
from rehearsal.executor import run_plan
from rehearsal.predicates import with_implicit_goals
from rehearsal.search_planner import (assignments_from_goals, describe_failure, merge_assignments, orders,
                                      prefix_goals, search)

REPO = Path(__file__).resolve().parent.parent


def scene(name):
    return json.loads((REPO / "scenes" / f"{name}.json").read_text())


def ref(name):
    s = scene(name)
    return s, with_implicit_goals(s["reference"]["goals"], [o["id"] for o in s["objects"]])


def test_assignments_from_goals_covers_on_inside_stacked():
    goals = [{"pred": "inside", "args": ["mug", "bowl"]}, {"pred": "upright", "args": ["mug"]},
             {"pred": "stacked", "args": ["L", "M", "S"]}, {"pred": "on", "args": ["cup", "coaster"]}]
    assert assignments_from_goals(goals) == [
        {"obj": "mug", "dest": "bowl", "relation": "in"},
        {"obj": "M", "dest": "L", "relation": "on"}, {"obj": "S", "dest": "M", "relation": "on"},
        {"obj": "cup", "dest": "coaster", "relation": "on"}]


def test_merge_prefers_plan_only_when_it_covers_the_goals():
    goals = [{"pred": "on", "args": ["a", "b"]}, {"pred": "on", "args": ["c", "d"]}]
    plan = {"actions": [{"obj": "c", "dest": "d", "relation": "on"}, {"obj": "a", "dest": "b", "relation": "on"}]}
    assert [a["obj"] for a in merge_assignments(goals, plan)] == ["c", "a"]
    partial = {"actions": [{"obj": "c", "dest": "x", "relation": "on"}]}      # wrong destination
    assert merge_assignments(goals, partial) == assignments_from_goals(goals)


def test_prefix_goals_only_judge_what_has_moved():
    goals = with_implicit_goals([{"pred": "stacked", "args": ["L", "M", "S"]}, {"pred": "on", "args": ["c", "d"]}],
                                ["L", "M", "S", "c", "d", "x"])
    pg = prefix_goals(goals, moved={"M"}, to_move={"M", "S", "c"})
    assert {"pred": "on", "args": ["M", "L"]} in pg
    assert all(g["args"][0] not in ("S", "c") for g in pg)
    assert {"pred": "undisturbed", "args": ["x"]} in pg


def test_orders_never_place_on_something_moved_later():
    a = [{"obj": "S", "dest": "M", "relation": "on"}, {"obj": "M", "dest": "L", "relation": "on"},
         {"obj": "c", "dest": "d", "relation": "on"}]
    for o in orders(a):
        objs = [x["obj"] for x in o]
        assert objs.index("M") < objs.index("S")
    assert [x["obj"] for x in orders(a)[0]] == ["M", "S", "c"]   # closest to the given order first


@pytest.mark.parametrize("name", ["windowsill_01", "lamp_01"])
def test_search_solves_hard_scene_and_plan_verifies(name):
    s, goals = ref(name)
    res = search(s, goals, assignments_from_goals(goals), sim_budget=60)
    assert res["success"], res
    assert res["stats"]["sims"] <= 60
    assert run_plan(s, res["plan"], goals)["success"]          # independent re-simulation
    # deterministic
    res2 = search(s, goals, assignments_from_goals(goals), sim_budget=60)
    assert res2["plan"]["actions"] == res["plan"]["actions"]


def test_search_tries_hint_geometry_first():
    s, goals = ref("windowsill_01")
    fixed = s["reference"]["fixed_plan"]
    res = search(s, goals, assignments_from_goals(goals), hint_plan=fixed, sim_budget=10)
    assert res["success"] and res["stats"]["sims"] == 1
    assert res["plan"]["actions"][0]["via"] == fixed["actions"][0]["via"]


def test_search_reports_uncovered_goals_without_simulating():
    s, goals = ref("lamp_01")
    res = search(s, goals, [], sim_budget=5)
    assert not res["success"] and "no assignment" in res["reason"] and res["stats"]["sims"] == 0
    assert "found no plan" in describe_failure(res, [])


def test_search_gives_up_within_budget_on_unreachable_goal():
    s = scene("lamp_01")
    s["objects"][0]["pos"] = [0.95, 0.0]           # far outside the arm's reach
    s["table"] = {"center": [0.6, 0.0], "size": [0.9, 1.0]}
    goals = with_implicit_goals([{"pred": "on", "args": [s["objects"][0]["id"], "mat_1"]}],
                                [o["id"] for o in s["objects"]])
    res = search(s, goals, assignments_from_goals(goals), sim_budget=3, time_budget_s=60)
    assert not res["success"] and res["stats"]["sims"] <= 3


def test_rehearse_hybrid_mode_offline(tmp_path):
    """Offline stand-in compiles the reference goals and proposes the naive plan; search fixes the geometry."""
    rep = R.rehearse(str(REPO / "scenes" / "windowsill_01.json"), "Put the mug on the tray.", 0,
                     str(tmp_path / "out"), log=lambda *_: None, look_rounds=0, planner="hybrid")
    assert rep["planner"] == "hybrid" and rep["search"][0]["success"]
    assert len(rep["attempts"]) == 1 and rep["attempts"][0]["success"]
    assert "search" in rep["agent"]


def test_rehearse_rejects_unknown_planner(tmp_path):
    with pytest.raises(ValueError):
        R.rehearse(str(REPO / "scenes" / "windowsill_01.json"), "x", 0, str(tmp_path), log=lambda *_: None,
                   look_rounds=0, planner="magic")


def test_plan_with_search_reasks_model_when_search_fails():
    s = scene("lamp_01")
    goals = with_implicit_goals([{"pred": "on", "args": ["tin_1", "mat_1"]}], [o["id"] for o in s["objects"]])

    class Agent(OfflineAgent):
        calls = []

        def plan(self, *a):
            return {"diagnosis": "", "actions": [dict(obj="tin_1", dest="phone_1", relation="on",
                                                      transit_height=0.3, via=[])]}

        def revise(self, instruction, summary, goals, history):
            self.calls.append(history[-1]["evidence"][0])
            return {"diagnosis": "", "actions": [dict(obj="tin_1", dest="mat_1", relation="on",
                                                      transit_height=0.35, via=[])]}

    ag = Agent()
    # the plan's assignment (tin on phone) does not cover the goal -> goal-derived assignment is used,
    # so search succeeds without re-asking
    plan, recs = R.plan_with_search(ag, s, "x", [], goals, "hybrid", log=lambda *_: None)
    assert recs[-1]["success"] and not ag.calls
    # force a failure: a goal no assignment can satisfy within a 0-sim budget -> the model is re-asked once
    plan, recs = R.plan_with_search(ag, s, "x", [], goals, "hybrid", log=lambda *_: None, sim_budget=0)
    assert len(recs) == 2 and ag.calls and "geometric search" in ag.calls[0]
