"""The model-side roles: compile instruction -> goals, plan, and diagnose+revise.

`MuseAgent` calls Muse Spark through the Meta Model API (needs MODEL_API_KEY).
`OfflineAgent` is a deterministic stand-in with the same interface, so the simulator loop can
be developed and demoed without API access. It is intentionally naive on the first plan.
"""
from __future__ import annotations

import json
import os

from .scene import object_height

ACTION_ITEM = {
    "type": "object",
    "properties": {
        "obj": {"type": "string"},
        "dest": {"type": "string"},
        "relation": {"type": "string", "enum": ["in", "on"]},
        "transit_height": {"type": "number"},
        # Horizontal waypoints [x, y] carried at transit_height; [] = straight carry (strict schemas
        # need every key required, so "optional" is expressed as an empty list).
        "via": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    },
    "required": ["obj", "dest", "relation", "transit_height", "via"],
    "additionalProperties": False,
}

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "actions": {"type": "array", "items": ACTION_ITEM},
    },
    "required": ["diagnosis", "actions"],
    "additionalProperties": False,
}

GOAL_SCHEMA = {
    "type": "object",
    "properties": {"goals": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "pred": {"type": "string", "enum": ["inside", "on", "upright", "stacked", "stable"]},
            "args": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["pred", "args"], "additionalProperties": False}}},
    "required": ["goals"], "additionalProperties": False,
}

SYSTEM = """You control a Franka Panda arm on a tabletop through one action:
transport(obj, dest, relation in {"in","on"}, transit_height, via) — suction-pick obj from its top
face, lift until the tool tip is at transit_height (meters above the table), carry it horizontally
(through the optional [x, y] waypoints in via, at transit_height; via=[] is a straight line),
and lower it in/on dest. The gripper body and arm sit above the tool tip. The held object hangs BELOW the tool tip by its own height.
Frame: table top z=0, robot base at the origin facing +x. All positions in meters.
Success is checked by a physics simulator, not by you. Objects not mentioned in the task
must not be disturbed."""


def scene_summary(scene: dict, reach: dict) -> list[dict]:
    return [{"id": o["id"], "label": o["label"], "xy": o["pos"], "height": object_height(o),
             "shape": o["shape"], "reachable": reach.get(o["id"])} for o in scene["objects"]]


_SIM_KEYS = ("goals", "execution_error", "events", "evidence")


def compact_attempt(h: dict) -> dict:
    """One attempt as the model sees it in revise(): plan, outcome, failed predicates with numbers,
    the execution error and the simulator's plain-sentence evidence. Raw events (hundreds of
    contact records) are only sent when the executor provides no `evidence`."""
    if not any(k in h for k in _SIM_KEYS):      # already filtered (e.g. eval's outcome_only / none)
        return h
    d = {"attempt": h.get("attempt"), "plan": h.get("plan"), "success": h.get("success")}
    if "goals" in h:
        d["failed_goals"] = [f"{g['pred']}({', '.join(g['args'])}): {g['why']}" for g in h["goals"] if not g["ok"]]
    err = h.get("execution_error")
    if err:
        d["execution_error"] = {k: v for k, v in err.items() if k != "action"}
    if h.get("evidence") is not None:
        d["evidence"] = h["evidence"]
    else:
        d["contacts"] = [e for e in h.get("events", []) if e.get("kind") != "contact"
                         or not {"table", "floor"} & set(e.get("between", []))]
    return d


class MuseAgent:
    """data="public" routes to Contributor mode (only for synthetic / public-photo scenes).
    compact_history=True sends compact_attempt() records to revise() instead of the raw history."""

    def __init__(self, model: str = "muse-spark-1.3", data: str = "private", effort: str | None = None,
                 compact_history: bool = True):
        from .meta_client import muse_model
        self.model, self.data, self.effort = model, data, effort
        self.compact_history = compact_history
        self.name = muse_model(data, model)

    def _json(self, user: str, schema: dict, name: str) -> dict:
        from .meta_client import chat_json
        return chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                         schema, name, caller=f"agent.{name.lower()}", data=self.data,
                         model=self.model, effort=self.effort)

    def compile_goals(self, instruction, summary):
        return self._json(
            f"Scene objects:\n{json.dumps(summary)}\n\nInstruction: {instruction}\n\n"
            "Translate the instruction into checkable predicates. inside(a,b), on(a,b), upright(a), "
            "stacked(bottom,...,top), stable(a). Use only ids from the scene.",
            GOAL_SCHEMA, "Goals")["goals"]

    def plan(self, instruction, summary, goals):
        return self._json(
            f"Scene objects:\n{json.dumps(summary)}\n\nInstruction: {instruction}\n"
            f"Goals to satisfy: {json.dumps(goals)}\n\nWrite the action sequence. "
            "Put an empty string in diagnosis.", ACTION_SCHEMA, "Plan")

    def revise(self, instruction, summary, goals, history):
        if self.compact_history:
            history = [compact_attempt(h) for h in history]
        return self._json(
            f"Scene objects:\n{json.dumps(summary)}\n\nInstruction: {instruction}\n"
            f"Goals: {json.dumps(goals)}\n\nPrevious attempts with simulator evidence "
            f"(failed predicates, contact events with timestamps, IK errors):\n"
            f"{json.dumps(history, indent=1)}\n\n"
            "Explain the physical cause of the latest failure from the evidence, then output a "
            "corrected full action sequence.", ACTION_SCHEMA, "Revision")


class OfflineAgent:
    """Rule-based stand-in. Only knows the demo instruction's vocabulary. When the scene carries a
    verified `reference` block ({goals, naive_plan, fixed_plan}, see docs/scenarios.md), it replays
    that instead: naive_plan as attempt 1, fixed_plan as the revision."""
    name = "offline-stand-in"
    DEFAULT_TRANSIT = 0.15

    def __init__(self, scene: dict | None = None):
        self.reference = (scene or {}).get("reference")

    def compile_goals(self, instruction, summary):
        if self.reference:
            return [dict(g) for g in self.reference["goals"]]
        ids = {o["id"] for o in summary}
        goals = []
        if "杯" in instruction and "碗" in instruction:
            goals += [{"pred": "inside", "args": ["mug_1", "bowl_1"]},
                      {"pred": "upright", "args": ["mug_1"]}]
        if "书" in instruction:
            books = sorted((o for o in summary if o["id"].startswith("book")),
                           key=lambda o: -o["shape"]["size"][0] * o["shape"]["size"][1])
            order = [b["id"] for b in books]
            goals += [{"pred": "stacked", "args": order}, {"pred": "stable", "args": [order[-1]]}]
        return [g for g in goals if set(g["args"]) <= ids]

    def plan(self, instruction, summary, goals):
        if self.reference:
            return json.loads(json.dumps(self.reference["naive_plan"]))
        acts = []
        for g in goals:
            if g["pred"] == "inside":
                acts.append(dict(obj=g["args"][0], dest=g["args"][1], relation="in",
                                 transit_height=self.DEFAULT_TRANSIT, via=[]))
            elif g["pred"] == "stacked":
                for lower, upper in zip(g["args"], g["args"][1:]):
                    acts.append(dict(obj=upper, dest=lower, relation="on",
                                     transit_height=self.DEFAULT_TRANSIT, via=[]))
        return {"diagnosis": "", "actions": acts}

    def revise(self, instruction, summary, goals, history):
        if self.reference:
            return json.loads(json.dumps(self.reference["fixed_plan"]))
        last = history[-1]
        heights = {o["id"]: o["height"] for o in summary}
        bumps = [e for e in last["events"] if e["kind"] == "contact" and e["while_holding"]
                 and not {"table", "floor"} & set(e["between"])]
        acts = [dict(a) for a in last["plan"]["actions"]]
        notes = []
        for e in bumps:
            held = e["while_holding"]
            hit = next((x for x in e["between"] if x not in (held, "robot_arm")), None)
            act = next((a for a in acts if a["obj"] == held), None)
            if not hit or not act or hit == act["dest"] or hit not in heights:
                continue
            need = round(heights[hit] + heights[held] + 0.06, 2)
            if need > act["transit_height"]:
                notes.append(f"t={e['t']}s 时，吸盘在 {e['tcp_z']} m 高度搬运 {held}，"
                             f"撞到了 {hit}（高 {heights[hit]} m）。{held} 挂在吸盘下方 "
                             f"{heights[held]} m，所以搬运高度必须超过 {heights[hit]} + "
                             f"{heights[held]} m，改为 {need} m。")
                act["transit_height"] = need
        diag = " ".join(notes) or "日志里没有能解释失败的碰撞，按原计划重试。"
        return {"diagnosis": diag, "actions": acts}


def make_agent(data: str = "private", scene: dict | None = None):
    if os.environ.get("REHEARSAL_OFFLINE") or not os.environ.get("MODEL_API_KEY"):
        return OfflineAgent(scene)
    return MuseAgent(data=data)
