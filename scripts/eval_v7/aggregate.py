"""eval_v7: v6 vs v7 outcome table + rehearsal-effort stats (effort.json, OUTCOMES.md) from eval_v7/runs + trace.

Run (rehearsal-mvp/):  .venv/bin/python scripts/eval_v7/aggregate.py
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ext  # noqa: E402
from rehearsal.scene import object_height  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT.parent / "footage" / "tables_sim" / "eval_v7"
V6LOG = ROOT.parent / "footage" / "tables_sim" / "hero" / "exec" / "exec_log.json"
REASONS = ("collision", "tip", "drop", "reach", "disturbance", "placement", "grasp_miss", "unstable", "other_error",
           "other")

V6_DONE = {"06", "29", "31", "35", "41", "43", "44", "45", "50"}
V6_ALREADY = {"02", "22", "25", "26"}
V6_MISSED = {"05", "14", "32", "30"}
# the scene format stores only a yaw, so a toppled object is rebuilt upright: the twin itself says "already done"
TWIN_CANNOT_ENCODE = {"30": "the bottle has fallen over in the photo; the twin format stores only a yaw, so it is "
                            "rebuilt upright and nothing seems to need doing (and top-down suction cannot right it)"}
EXT_LABEL = {"offset": "placement offset", "yaw": "yaw turn", "drop": "release above rim",
             "grip_offset": "grasp-point offset", "extra_height": "carry at 0.50-0.55 m",
             "near": "next-to placement"}


def scene_of(key):
    f = EV / "scenes" / f"t6_{key}.json"
    return json.loads((f if f.exists() else ROOT / "scenes" / f"t6_{key}.json").read_text())


def twin_fixes(key):
    f = EV / "scenes" / f"t6_{key}.json"
    if not f.exists():
        return []
    return [{"object": x["object"], "why": x["why"]} for x in json.loads(f.read_text()).get("eval_v7_fixes", [])]


def plan_extensions(r, sc):
    """Which planner extensions the verified plan relies on, per action, plus every rim release's drop height."""
    used, drops = collections.Counter(), []
    objs = {o["id"]: o for o in sc["objects"]}
    for a in (r.get("plan_exec") or {}).get("actions", []):
        rel = a["relation"]
        if rel.startswith("near@"):
            used["near"] += 1
            continue
        b, dx, dy, dyaw, drop = ext.parse_rel(rel)
        if dx or dy:
            used["offset"] += 1
        if dyaw:
            used["yaw"] += 1
        if ext.parse_grip(rel).any():
            used["grip_offset"] += 1
        if a["transit_height"] in ext.EXTRA_HEIGHTS:
            used["extra_height"] += 1
        if drop:
            used["drop"] += 1
            hd = object_height(objs[a["dest"]])
            drops.append({"obj": a["obj"], "into": a["dest"], "fragile": ext.fragile(objs[a["obj"]]),
                          "container_height_cm": round(100 * hd, 1),
                          "bottom_above_rim_at_release_cm": round(100 * ext.DROP_CLEAR, 1),
                          "max_fall_to_container_floor_cm": round(100 * (hd + ext.DROP_CLEAR), 1)})
    return {EXT_LABEL[k]: v for k, v in used.items()}, drops


def v6_outcome(nn, comp):
    st = (comp.get("intent") or {}).get("status")
    if nn == "34":
        return "empty"
    if st == "clarify":
        return "asked"
    if st == "refuse":
        return "refused"
    if nn in V6_DONE:
        return "done"
    if nn in V6_ALREADY:
        return "already_done"
    if nn in V6_MISSED:
        return "missed"
    return "no_safe_plan"


def main():
    v6log = json.loads(V6LOG.read_text())
    rows, eff = [], {}
    tot = collections.Counter()
    hist_all = collections.Counter()
    slots = [f"{i:02d}" for i in range(1, 51)]
    for nn in slots:
        comp6 = json.loads((ROOT / "out" / "tables_v6" / f"{nn}.json").read_text())
        o6 = v6_outcome(nn, comp6)
        key = "51" if nn == "34" else nn
        rf = EV / "runs" / f"{key}.json"
        r = json.loads(rf.read_text()) if rf.exists() else None
        if o6 in ("asked", "refused"):
            o7 = o6                      # same compiled intent: nothing to rehearse
        elif r is None:
            o7 = "not_run"
        elif r.get("error"):
            o7 = "error"
        else:
            o7 = r.get("outcome") or ("no_safe_plan" if not r.get("planned") else "?")
        if nn in TWIN_CANNOT_ENCODE and o7 == "already_done":
            o7 = "missed"
        s6 = (v6log.get(nn) or {}).get("search") or {}
        row = {"slot": nn, "table": key, "instruction": (r or {}).get("instruction") or comp6.get("instruction"),
               "v6": o6, "v7": o7, "v6_sims": (s6.get("stats") or {}).get("sims"), "v6_wall_s": s6.get("wall_s"),
               "v6_reason": s6.get("reason")}
        sc = scene_of(key) if r else None
        row["twin_fixes"] = twin_fixes(key)
        row["twin_note"] = TWIN_CANNOT_ENCODE.get(nn)
        if r and r.get("planned") and o7 == "done":
            row["extensions"], row["rim_releases"] = plan_extensions(r, sc)
            row["label"] = ", ".join(row["extensions"]) or "stock actions only"
            if row["twin_fixes"]:
                row["label"] += " (twin corrected from photo)"
        if r and r.get("search"):
            s = r["search"]
            st = s.get("stats") or {}
            tf = EV / "trace" / f"{key}.json"
            tr = json.loads(tf.read_text()) if tf.exists() else []
            hist = collections.Counter(t["why"] for t in tr if not t["ok"])
            ok_prefix = sum(1 for t in tr if t["ok"])
            e = {"instruction": row["instruction"], "outcome_v6": o6, "outcome_v7": o7,
                 "planner": r.get("planner"),
                 "candidates_screened": st.get("kin_checks", st.get("spots_tried")),
                 "rejected_in_dry_run_reach": st.get("kin_ik_pruned", 0),
                 "physics_rollouts": st.get("sims", len(tr)),
                 "rollouts_accepted_as_prefix": ok_prefix,
                 "rollouts_rejected": sum(hist.values()),
                 "rejected_by_reason": {k: hist[k] for k in REASONS if hist[k]},
                 "safe_plan_found_at_rollout": st.get("found_at_sim"),
                 "simulated_seconds": st.get("sim_seconds"),
                 "wall_s": s.get("wall_s"), "budget_s": s.get("budget_s"),
                 "search_end": "plan found" if s.get("success") else s.get("reason"),
                 "families_in_plan": st.get("families"),
                 "plan_actions": len(((r.get("plan") or {}).get("actions")) or []),
                 "extensions_in_plan": row.get("extensions"), "rim_releases": row.get("rim_releases"),
                 "twin_corrected_from_photo": [x["object"] for x in row["twin_fixes"]]}
            eff[key] = e
            row.update(v7_sims=e["physics_rollouts"], v7_wall_s=e["wall_s"], found_at=e["safe_plan_found_at_rollout"],
                       families=e["families_in_plan"])
            tot["tasks"] += 1
            tot["candidates_screened"] += e["candidates_screened"] or 0
            tot["physics_rollouts"] += e["physics_rollouts"] or 0
            tot["rollouts_rejected"] += e["rollouts_rejected"]
            tot["rejected_in_dry_run_reach"] += e["rejected_in_dry_run_reach"] or 0
            tot["wall_s"] += e["wall_s"] or 0
            tot["simulated_seconds"] += e["simulated_seconds"] or 0
            hist_all.update(hist)
        rows.append(row)
    c6 = collections.Counter(r["v6"] for r in rows)
    c7 = collections.Counter(r["v7"] for r in rows)
    order = ["done", "already_done", "no_safe_plan", "missed", "asked", "refused", "empty", "not_run", "error"]
    side = [{"outcome": k, "v6": c6.get(k, 0), "v7": c7.get(k, 0)} for k in order if c6.get(k) or c7.get(k)]
    new_done = [r for r in rows if r["v7"] == "done" and r["v6"] != "done"]
    ext_use = collections.Counter(k for r in new_done for k in (r.get("extensions") or {}))
    summary = {"tasks_rehearsed": tot["tasks"], "futures_simulated": tot["physics_rollouts"],
               "candidates_screened_kinematically": tot["candidates_screened"],
               "rejected_in_dry_run_out_of_reach": tot["rejected_in_dry_run_reach"],
               "rollouts_rejected": tot["rollouts_rejected"],
               "rejected_by_reason": {k: hist_all[k] for k in REASONS if hist_all[k]},
               "search_wall_hours": round(tot["wall_s"] / 3600, 2),
               "simulated_robot_minutes": round(tot["simulated_seconds"] / 60, 1),
               "outcomes_v6": dict(c6), "outcomes_v7": dict(c7), "outcomes_side_by_side": side,
               "newly_done": [r["table"] for r in new_done],
               "newly_done_stock_actions_only": [r["table"] for r in new_done if not r.get("extensions")],
               "newly_done_using_extension": dict(ext_use),
               "newly_done_with_twin_correction": [r["table"] for r in new_done if r["twin_fixes"]],
               "rim_releases": [dict(d, table=r["table"]) for r in rows for d in (r.get("rim_releases") or [])],
               "notes": {"checker": "unchanged rehearsal.executor.run_plan + predicates, plus a stricter rim audit "
                                    "(inside() rejected when the object's bottom ends within 1 cm of the rim)",
                         "extensions": "labelled planner extensions in scripts/eval_v7/ext.py (offset, yaw, rim "
                                       "release for non-fragile objects only, grasp-point offset, 0.50/0.55 m carry) "
                                       "and the near-placement demo extension (scripts/user_desk/run_user_desk.py)",
                         "slot_34": "the empty table 34 is replaced by AI-generated table 51 (desk_oak_laptop.png)"}}
    (EV / "effort.json").write_text(json.dumps({"summary": summary, "per_task": eff}, indent=1))
    (EV / "outcomes.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
