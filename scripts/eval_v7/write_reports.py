"""eval_v7: write DIAGNOSIS.md (table section) and OUTCOMES.md from outcomes.json / effort.json.

Run after aggregate.py (rehearsal-mvp/):  .venv/bin/python scripts/eval_v7/write_reports.py
The one-line diagnoses below are written by hand from the traces (scripts/eval_v7/diag.py NN) and the photo overlays.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT.parent / "footage" / "tables_sim" / "eval_v7"

# table: (what blocked v6, class)
DIAG = {
    "03": ("Twin: the laundry-basket box was centred on the basket's front edge, turned 25 deg and 49 cm deep, so its "
           "walls stood over the socks; 640 of 683 v7 rollouts were gripper collisions with that phantom wall. With the basket rebuilt from "
           "the photo (58x34x35 cm, at the far edge, 0.81 m out) every way of lowering or releasing into it fails the "
           "IK dry run (296/296): out of this arm's reach.", "TWIN, then TRUE (reach)"),
    "04": ("Stacking order: once book_2 sits on the narrow book_3 it overhangs book_1, and the suction tip hits book_2 "
           "when reaching for book_1 (180/180 rollouts). Solved by picking book_1 off-centre, away from the overhang "
           "(grasp-point offset) with a yaw turn (found at rollout #179).", "FAMILY"),
    "05": ("Twin put the table lamp on top of the phone (8 cm overlap; the lamp base is at the right edge in the photo), "
           "and v6 had no 'next to' action at all.", "TWIN + FAMILY"),
    "07": ("Every straight carry to the box centre clipped the potted plant; needs a placement offset and a low carry "
           "(v7: 68 plant collisions before the plan at rollout #85).", "FAMILY"),
    "08": ("Twin: the toolbox was rotated about 83 deg and its box included the open lid, so 528 picks hit it; with "
           "the body-only toolbox the search also needs off-centre places.", "TWIN"),
    "09": ("The gripper body hits the mixing-bowl rim when descending to egg_2, which sits against the bowl (647 of 695 "
           "rollouts on the v6 twin). Re-run on the refined-scene agent's corrected twin (flour bag standing, copied "
           "unchanged): the same contact blocks it (946 rollouts). The eggs are too small for a grasp-point offset "
           "(radius under 1.5 cm), so the hand cannot clear the bowl.",
           "TRUE (for this gripper)"),
    "10": ("Toys aimed at the basket centre stack on each other above the rim; needs offsets / yaw. The first v7 pass "
           "left the elephant 5 mm under the rim on the bunny, which the rim audit rejects.", "FAMILY"),
    "13": ("Spools all aimed at the box centre pile up (v6 spent its 126 sims on that); offsets put them side by side.",
           "FAMILY / SEARCH"),
    "14": ("v6 had no 'next to' action: the compiler emitted near() and the shipped planner could not place beside.",
           "FAMILY"),
    "16": ("Each second plate's landing nudges the box, which pushes the marker lying against it 2.2-2.3 cm (the "
           "undisturbed limit is 2 cm). Borderline contact in the twin.", "TWIN (borderline)"),
    "18": ("Twin: the soil bag interpenetrated pot_2 and toppled off the table at settle with no robot action, so all "
           "211 rollouts failed 'drop'. With the bag fixed, two of three pots go in, but the third never fits the tray "
           "beside them (inside() offset or collisions) within 3000 CPU-s.", "TWIN, then GOAL (tray capacity; borderline)"),
    "20": ("One order only (7 items) and every centre placement piled bolts; one fell off. Offsets spread them.",
           "FAMILY"),
    "21": ("Capacity: six ~8 cm apples into a 19 cm bowl 7.4 cm deep; only three fit below the rim, and inside() "
           "rejects piled fruit.", "TRUE (as compiled)"),
    "24": ("Long crayons do not fit the box at their photo yaw; they must be turned (v7: yaw turn, found at #156 after "
           "135 suction misses on thin crayons).", "FAMILY"),
    "27": ("The tall bucket: lowering the hand into it collides with the rim; releasing just above the rim works "
           "(found at #3).", "FAMILY"),
    "28": ("The 10 g glove touching the big bowl is dragged 3-4 cm whenever a bowl nests, and the third bowl tips.",
           "TRUE (as compiled)"),
    "30": ("The bottle has fallen over in the photo; the twin format stores only a yaw, so it is rebuilt upright and "
           "nothing appears to need doing, and top-down suction cannot right a bottle anyway.", "TWIN FORMAT + TRUE"),
    "32": ("Twin: the monitor was one solid 63x23x45 cm block over the keyboard and up to mug_1, so every approach "
           "pressed on it (21/21). In the photo it is a screen on a small stand behind the keyboard. After the fix "
           "the near extension's few candidate spots beside mug_2 are blocked by the keyboard, mouse and mug_3 "
           "(24 rollouts).", "TWIN, then FAMILY (near spots too few)"),
    "36": ("Hand cannot enter the narrow container; release above the rim is needed. First v7 pass left the filter at "
           "rim height (rim audit).", "FAMILY"),
    "37": ("24 cm clay tools into an 11.7 cm mug: lowered, they lie across the rim (the first v7 'pass' did exactly "
           "that and the rim audit rejects it; the re-run under the audit finds nothing in 238 rollouts). They would have "
           "to be inserted upright, which suction-from-above cannot do.", "TRUE (for this gripper)"),
    "38": ("Six tokens = 720 orders; v6 spent its whole budget on orders at the strictest clearance level and never "
           "tried the looser levels. v7 caps orders per level and gives it 3000 CPU-s (found after 1500 failed).",
           "SEARCH"),
    "39": ("The cloth lies against the 18 cm caddy, so the hand hits the caddy when descending on the cloth centre "
           "(789 collisions). The grasp-point-offset re-run gets two of three items in, but the third always "
           "bumps the caddy past the limit or hits it (1,085 rollouts).", "TRUE (as compiled; borderline)"),
    "40": ("Test tubes aimed at the rack centre collide; needs offsets and yaw (found at #116 after 102 suction misses "
           "on the thin tubes).", "FAMILY"),
    "46": ("Three 12 cm dishes on a 36x23 cm tray: inside() wants each centre within the tray's inscribed circle "
           "(about 8 cm), so they overlap and ride up on each other.", "GOAL (predicate approximation)"),
    "47": ("Two 6 cm cubes in an 11 cm wide case: the same inscribed-circle test forces them to overlap and the second "
           "cube tips (360/360).", "GOAL (predicate approximation)"),
    "48": ("The keys lie on envelope_3 in the photo, so moving it moves the keys (321 rollouts); a person would lift "
           "the keys first. A correct refusal for the compiled goal.", "TRUE (as compiled)"),
    "49": ("Stock search stopped at heights up to 0.45 m; the tall object needed a 0.47 m carry plus an offset.",
           "SEARCH / FAMILY"),
}
NAMES = {"done": "done", "already_done": "already done", "no_safe_plan": "not sent (no safe plan)",
         "missed": "missed", "asked": "asked", "refused": "refused", "empty": "empty", "not_run": "not run"}


def main():
    rows = {r["slot"]: r for r in json.loads((EV / "outcomes.json").read_text())}
    eff = json.loads((EV / "effort.json").read_text())
    s = eff["summary"]
    pt = eff["per_task"]

    def v7cell(r):
        o = NAMES.get(r["v7"], r["v7"])
        if r["v7"] == "done":
            o += f" — {r.get('label')}"
            e = pt.get(r["table"], {})
            if e.get("safe_plan_found_at_rollout"):
                o += f"; found at rollout #{e['safe_plan_found_at_rollout']}"
        elif r.get("twin_fixes"):
            o += " (twin corrected from photo)"
        return o

    lines = []
    for nn, (why, cls) in DIAG.items():
        r = rows[nn]
        lines.append(f"| {nn} | {r['instruction']} | {NAMES[r['v6']]} | {why} | {cls} | {v7cell(r)} |")
    fixes = []
    for r in rows.values():
        for f in r.get("twin_fixes") or []:
            fixes.append(f"- **{r['table']} {f['object']}** (twin corrected from photo): {f['why']}")
    d = (EV / "DIAGNOSIS.md").read_text()
    if "FILLED_BY_SCRIPT" in d:
        (EV / "DIAGNOSIS.template.md").write_text(d)
    d = (EV / "DIAGNOSIS.template.md").read_text()
    d = d.replace("FILLED_BY_SCRIPT", "\n".join(lines)).replace("FIXES_BY_SCRIPT", "\n".join(fixes))
    (EV / "DIAGNOSIS.md").write_text(d)

    # OUTCOMES.md
    o = ["# eval_v7 vs v6 — per-table outcomes", "",
         "Physics decides: a table counts as **done** only if the unchanged checker (`rehearsal.executor.run_plan` + "
         "predicates + implicit undisturbed goals) verified the plan **and** the stricter rim audit passed. Every pass "
         "that needs an action family the shipped planner lacks is labelled with it; every twin edited to match the "
         "photo is marked *(twin corrected from photo)*.", "",
         "## Headline tallies (50 slots; slot 34 = new table 51)", "", "| outcome | v6 | v7 |", "|---|---|---|"]
    o += [f"| {NAMES.get(x['outcome'], x['outcome'])} | {x['v6']} | {x['v7']} |" for x in s["outcomes_side_by_side"]]
    nd = s["newly_done"]
    o += ["", f"Newly done in v7: {len(nd)} ({', '.join(nd)}). Stock actions only: "
          f"{', '.join(s['newly_done_stock_actions_only']) or 'none'}. Extension use among them (actions): "
          + ", ".join(f"{k} {v}" for k, v in s["newly_done_using_extension"].items())
          + f". With a twin corrected from the photo: {', '.join(s['newly_done_with_twin_correction']) or 'none'}.", ""]
    if s.get("rim_releases"):
        o += ["### Release above rim (a short drop; non-fragile objects only)", "",
              "| table | object | into | container height | released | max fall to floor |", "|---|---|---|---|---|---|"]
        o += [f"| {x['table']} | {x['obj']} | {x['into']} | {x['container_height_cm']} cm | bottom "
              f"{x['bottom_above_rim_at_release_cm']} cm above rim | {x['max_fall_to_container_floor_cm']} cm |"
              for x in s["rim_releases"]]
        o.append("")
    o += ["## Rehearsal effort (all rehearsed tables)", "",
          f"- {s['tasks_rehearsed']} tasks rehearsed; {s['futures_simulated']:,} futures simulated in physics; "
          f"{s['rollouts_rejected']:,} rejected; {s['candidates_screened_kinematically']:,} candidates screened in the "
          f"kinematic dry run ({s['rejected_in_dry_run_out_of_reach']:,} out of reach); "
          f"{s['simulated_robot_minutes']} simulated robot-minutes; {s['search_wall_hours']} h search wall time.", "",
          "| rejection reason | rollouts |", "|---|---|"]
    o += [f"| {k.replace('_', ' ')} | {v:,} |" for k, v in sorted(s["rejected_by_reason"].items(), key=lambda kv: -kv[1])]
    o += ["", "## Per table", "", "| slot | instruction | v6 | v7 | v6 sims | v7 rollouts | why it changed |",
          "|---|---|---|---|---|---|---|"]
    for nn, r in rows.items():
        if r["v6"] in ("asked", "refused"):
            continue
        why = DIAG.get(nn, ("", ""))[0] if r["v6"] != r["v7"] or nn in DIAG else ""
        if r["v6"] == r["v7"] == "done":
            why = "reproduces v6"
        elif r["v6"] == r["v7"] == "already_done":
            why = "reproduces v6"
        elif nn == "34":
            why = "empty table replaced by AI-generated table 51 (photos/ai_gen/desk_oak_laptop.png, unused before)"
        e = pt.get(r["table"], {})
        o.append(f"| {nn}{' → 51' if nn == '34' else ''} | {r['instruction']} | {NAMES.get(r['v6'])} | {v7cell(r)} | "
                 f"{r.get('v6_sims') or ''} | {e.get('physics_rollouts', '')} | {why} |")
    o += ["", "Asked (5) and refused (4) tables have the same compiled intent in v6 and v7; nothing to rehearse.", ""]
    (EV / "OUTCOMES.md").write_text("\n".join(o))
    print("wrote DIAGNOSIS.md, OUTCOMES.md")


if __name__ == "__main__":
    main()
