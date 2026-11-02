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

# --- Intent compile step (GoalBench arm C, docs/goalbench.md, docs/intent.md) -----------------------------------
# The shipped compile prompt and schema are byte-identical to GoalBench arm C (bench/goals/run.py imports them from
# here; tests check it), so the bench numbers describe exactly this request. The pre-GoalBench prompt (arm A, five
# predicates, no way to say no) stays reachable as compile mode "legacy" for eval reproducibility.
GOAL_ITEM_EXT = json.loads(json.dumps(GOAL_SCHEMA["properties"]["goals"]["items"]))
GOAL_ITEM_EXT["properties"]["pred"]["enum"] = ["inside", "on", "upright", "stacked", "stable",
                                               "undisturbed", "near", "clear"]
GOAL_SCHEMA_EXT = {"type": "object", "properties": {"goals": {"type": "array", "items": GOAL_ITEM_EXT}},
                   "required": ["goals"], "additionalProperties": False}
INTENT_SCHEMA = {"type": "object", "properties": {
    "assumptions": {"type": "array", "items": {"type": "string"}},
    "status": {"type": "string", "enum": ["ok", "clarify", "refuse"]},
    "question": {"type": "string"},
    "goals": {"type": "array", "items": GOAL_ITEM_EXT}},
    "required": ["assumptions", "status", "question", "goals"], "additionalProperties": False}

COMPILE_PROMPT_LEGACY = ("Scene objects:\n{summary}\n\nInstruction: {instruction}\n\n"
                         "Translate the instruction into checkable predicates. inside(a,b), on(a,b), upright(a), "
                         "stacked(bottom,...,top), stable(a). Use only ids from the scene.")
COMPILE_PROMPT_VOCAB = ("Scene objects:\n{summary}\n\nInstruction: {instruction}\n\n"
                        "Translate the instruction into checkable predicates. inside(a,b), on(a,b), upright(a), "
                        "stacked(bottom,...,top), stable(a), undisturbed(a,...) (must not be moved or tipped), "
                        "near(a,b) (a ends beside b on the same surface), clear(a) (nothing may end up on top of a). "
                        "Use only ids from the scene.")
COMPILE_PROMPT = COMPILE_PROMPT_VOCAB + (
    "\n\nBefore anything runs, the user sees your assumptions and confirms them. List every assumption you "
    "made in `assumptions`, including constraints the user implies without stating them. Set status to "
    "\"clarify\" and ask one short question in `question` if the instruction has several reasonable readings "
    "that lead to different goals. Set status to \"refuse\" and say why in `question` if it cannot be done with "
    "this robot (one transport action; object sizes; reachability), is unsafe, or names something that is not "
    "in the scene. Otherwise status is \"ok\", `question` is empty, and `goals` holds the predicates.")
COMPILE_MODES = ("confirm", "legacy")


def compile_mode(default: str = "confirm") -> str:
    """REHEARSAL_COMPILE=legacy selects the pre-GoalBench compile prompt (arm A); anything else, the shipped one."""
    v = os.environ.get("REHEARSAL_COMPILE", "").strip().lower()
    return v if v in COMPILE_MODES else default


def normalize_intent(raw: dict, mode: str = "confirm") -> dict:
    """Model output -> the product's intent record (the `intent` key of report.json, see docs/api.md):
    {"status": "goals"|"clarify"|"refuse", "goals", "assumptions", "question", "reason", "compile"}.
    The model's single `question` field carries the clarifying question (clarify) or the refusal reason (refuse)."""
    st = {"ok": "goals", "goals": "goals", "clarify": "clarify", "refuse": "refuse"}.get(raw.get("status", "ok"), "clarify")
    text = (raw.get("question") or "").strip()
    return {"status": st, "goals": [dict(g) for g in raw.get("goals") or []] if st == "goals" else [],
            "proposed_goals": [dict(g) for g in raw.get("goals") or []] if st != "goals" else [],
            "assumptions": [str(a) for a in raw.get("assumptions") or []],
            "question": text if st == "clarify" else "", "reason": text if st == "refuse" else "",
            "compile": mode}


SYSTEM = """You control a Franka Panda arm on a tabletop through one action:
transport(obj, dest, relation in {"in","on"}, transit_height, via) — suction-pick obj from its top
face, lift until the tool tip is at transit_height (meters above the table), carry it horizontally
(through the optional [x, y] waypoints in via, at transit_height; via=[] is a straight line),
and lower it in/on dest. The gripper body and arm sit above the tool tip. The held object hangs BELOW the tool tip by its own height.
Frame: table top z=0, robot base at the origin facing +x. All positions in meters.
Success is checked by a physics simulator, not by you. Objects not mentioned in the task
must not be disturbed. Write every diagnosis, rationale and explanation in English, whatever the
language of the instruction."""


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
        """Legacy compile step (GoalBench arm A, the product before 2026-11-02). Kept for eval reproducibility."""
        return self._json(
            f"Scene objects:\n{json.dumps(summary)}\n\nInstruction: {instruction}\n\n"
            "Translate the instruction into checkable predicates. inside(a,b), on(a,b), upright(a), "
            "stacked(bottom,...,top), stable(a). Use only ids from the scene.",
            GOAL_SCHEMA, "Goals")["goals"]

    def compile_intent(self, instruction, summary, mode: str | None = None) -> dict:
        """The shipped compile step (GoalBench arm C): goals + assumptions, or a clarifying question, or a refusal.
        mode "legacy" (or REHEARSAL_COMPILE=legacy) runs compile_goals and wraps it as status "goals"."""
        mode = mode or compile_mode()
        if mode == "legacy":
            return normalize_intent({"status": "ok", "goals": self.compile_goals(instruction, summary),
                                     "assumptions": [], "question": ""}, "legacy")
        raw = self._json(COMPILE_PROMPT.format(summary=json.dumps(summary), instruction=instruction),
                         INTENT_SCHEMA, "GoalsConfirm")
        return normalize_intent(raw, "confirm")

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


OFFLINE_UNSUPPORTED = ("offline stand-in only understands the bundled demo scenes; "
                       "set MODEL_API_KEY for live Muse Spark")


class OfflineAgent:
    """Rule-based stand-in. Only knows desk_01's vocabulary (Chinese or English: mug/cup + bowl, books). When the scene carries a
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
        low = instruction.lower()
        if ("杯" in instruction and "碗" in instruction) or (
                any(w in low for w in ("mug", "cup")) and "bowl" in low):
            goals += [{"pred": "inside", "args": ["mug_1", "bowl_1"]},
                      {"pred": "upright", "args": ["mug_1"]}]
        if "书" in instruction or "book" in low:
            books = sorted((o for o in summary if o["id"].startswith("book")),
                           key=lambda o: -o["shape"]["size"][0] * o["shape"]["size"][1])
            order = [b["id"] for b in books]
            goals += [{"pred": "stacked", "args": order}, {"pred": "stable", "args": [order[-1]]}]
        goals = [g for g in goals if set(g["args"]) <= ids]
        if not goals:
            raise ValueError(OFFLINE_UNSUPPORTED)
        return goals

    def compile_intent(self, instruction, summary, mode: str | None = None) -> dict:
        """Same record as MuseAgent.compile_intent. The stand-in never asks or refuses on its own; the
        deterministic intent gate (rehearsal/intent_gate.py) still applies to its goals."""
        goals = self.compile_goals(instruction, summary)
        assumptions = (["offline stand-in: replaying the scene's hand-verified reference goals"] if self.reference
                       else ["offline stand-in: goals parsed from keywords in the instruction"])
        return normalize_intent({"status": "ok", "goals": goals, "assumptions": assumptions, "question": ""},
                                mode or "offline")

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
                notes.append(f"At t={e['t']}s, while carrying {held} with the suction tool at "
                             f"{e['tcp_z']} m, it hit {hit} ({heights[hit]} m tall). {held} hangs "
                             f"{heights[held]} m below the suction cup, so the carry height must exceed "
                             f"{heights[hit]} + {heights[held]} m; raising it to {need} m.")
                act["transit_height"] = need
        diag = " ".join(notes) or "No collision in the log explains the failure; retrying the same plan."
        return {"diagnosis": diag, "actions": acts}


def offline_forced() -> bool:
    """REHEARSAL_OFFLINE=1/true/yes forces the stand-in; unset, "", 0, false, no, off do not."""
    return os.environ.get("REHEARSAL_OFFLINE", "").strip().lower() not in ("", "0", "false", "no", "off")


def make_agent(data: str = "private", scene: dict | None = None):
    if offline_forced() or not os.environ.get("MODEL_API_KEY"):
        return OfflineAgent(scene)
    return MuseAgent(data=data)
