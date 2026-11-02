"""DROID episode (real Franka footage, CC BY 4.0) -> Rehearsal scene, with the episode's known camera.

    .venv/bin/python scripts/droid_scene.py duck_success [cup_success ...]      # or: all

For datasets/droid/<slug>/:
1. pre.png (first frame of the exterior camera) -> Muse Spark object listing + SAM 3.1 masks
   (perception.perceive_raw, data="public": DROID is a public CC BY 4.0 dataset -> Contributor mode)
2. every object placed with perception.estimate_scene(camera=<DROID K + calibrated extrinsics>), method "guarded";
   world frame = DROID robot base frame shifted down so the table top is z = 0 (datasets/droid/table_height.json)
3. scene.robot.base_z = height of the Franka's link0 above the table (the DROID robot stands on its own cart)
4. pots, bowls and cups become open containers (shape "bowl": base + wall ring), so `inside` is physical
5. the instruction (DROID language annotation; the operator's task text for failures, which carry none) is compiled
   into predicates by Muse Spark (agents.MuseAgent.compile_goals) - the LLM handles intent, physics checks it
Writes scenes/droid/<slug>.json and docs/media/droid/<slug>.overlay.jpg.  No position is hand-edited; de-overlap is
the exact-footprint SAT of scripts/real_photo_scene.py.  The DROID success label is never read here.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rehearsal.calib import Camera  # noqa: E402
from rehearsal.perception import draw_overlay, estimate_scene, perceive_raw  # noqa: E402
from rehearsal.scene import shape_bbox  # noqa: E402
from real_photo_scene import separate  # noqa: E402

DATA = ROOT / "datasets" / "droid"
EPISODES = ["duck_success", "duck_missed_grasp", "duck_fail_nudge", "bowl_fail", "cup_success", "cup_fail"]
# The task each episode attempted.  Successful episodes carry DROID language annotations; failed ones only the
# operator's short task text, so they get the annotation of the successful episode of the same task (or a plain
# rephrasing of the operator text for the bowl task, which has no successful sibling here).
INSTRUCTION = {
    "duck_success": "Put the orange rubber duck in the pot.",
    "duck_missed_grasp": "Put the orange rubber duck in the pot.",
    "duck_fail_nudge": "Put the orange rubber duck in the pot.",
    "bowl_fail": "Stack the two bowls: put one bowl inside the other.",
    "cup_success": "Put the orange cup in the black bowl.",
    "cup_fail": "Put the orange cup in the black bowl.",
}
TABLE = {"center": [0.65, 0.05], "size": [0.9, 1.1]}     # the lab bench (larger than the camera's view of it)
CONTAINER_WORDS = ("pot", "bowl", "cup", "mug", "pan", "bucket")


def episode_camera(slug: str) -> tuple[Camera, dict]:
    cam = json.loads((DATA / slug / "camera.json").read_text())
    rc = cam["rehearsal_camera"]
    return Camera(rc["K"], rc["R"], rc["t"]), cam


def as_container(o: dict) -> bool:
    """Open containers -> shape "bowl" (thin base + ring of walls) with the fitted radius / height."""
    words = f"{o.get('category', '')} {o.get('label', '')}".lower()
    if not any(w in words for w in CONTAINER_WORDS):
        return False
    s = o["shape"]
    if s["type"] == "box":
        r, h = 0.25 * (s["size"][0] + s["size"][1]), s["size"][2]
    elif s["type"] in ("cylinder", "bowl") and s.get("axis", "z") == "z":
        r, h = s["radius"], s["height"]
    elif s["type"] == "compound":          # e.g. perception's mug template (body + handle) for a plain cup
        bb = shape_bbox(s)
        r, h = 0.25 * (bb[0] + bb[1]), bb[2]
    else:
        return False
    o["shape"] = {"type": "bowl", "radius": round(float(r), 4), "height": round(float(h), 4)}
    o["perception"].setdefault("adjusted", []).append("open_container")
    return True


def build(slug: str, goals_from_llm: bool = True) -> dict:
    cam, camj = episode_camera(slug)
    photo = DATA / slug / "pre.png"
    rel = str(photo.relative_to(ROOT))
    raw = perceive_raw(photo, data="public")
    scene = estimate_scene(raw, None, method="guarded", camera=cam, table=TABLE, privacy="public",
                           name=f"droid_{slug}", photo_file=rel, clamp=False)
    separate(scene["objects"])
    containers = [o["id"] for o in scene["objects"] if as_container(o)]
    lab = json.loads((DATA / slug / "label.json").read_text())
    scene["robot"] = {"base_z": float(camj["robot_base_height_above_table_m"]),
                      "note": "Franka link0 above the table top (DROID robot on its own cart)"}
    scene["title"] = f"DROID {slug.replace('_', ' ')}"
    scene["demo_instruction"] = INSTRUCTION[slug]
    scene["source"] = (f"DROID 1.0.1 episode {lab['episode_id']} (CC BY 4.0, Khazatsky et al. RSS 2024), first frame "
                       f"of {camj['which']} -> Muse Spark listing + SAM 3.1 masks -> guarded fit with the episode's "
                       f"calibrated camera (KarlP/droid extrinsics, table height from datasets/droid/table_height.json). "
                       f"Open containers: {', '.join(containers) or 'none'}. De-overlap: exact-footprint SAT. "
                       f"No hand-edited positions.")
    scene["photo"]["camera_info"]["dataset"] = "DROID 1.0.1 (CC BY 4.0)"
    scene["droid"] = {"slug": slug, "episode_id": lab["episode_id"], "camera": camj["which"],
                      "instruction": INSTRUCTION[slug], "containers": containers}
    if goals_from_llm:
        from rehearsal.agents import MuseAgent, scene_summary
        agent = MuseAgent(data="public")
        goals = agent.compile_goals(INSTRUCTION[slug], scene_summary(scene, {}))
        ids = {o["id"] for o in scene["objects"]}
        scene["droid"]["goals"] = [g for g in goals if set(g["args"]) <= ids]
        scene["droid"]["goals_source"] = f"{agent.name} compile_goals"
    out = ROOT / "scenes" / "droid"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.json").write_text(json.dumps(scene, ensure_ascii=False, indent=1))
    media = ROOT / "docs" / "media" / "droid"
    media.mkdir(parents=True, exist_ok=True)
    draw_overlay(photo, scene).convert("RGB").save(media / f"{slug}.overlay.jpg", quality=85)
    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="+")
    ap.add_argument("--no-llm-goals", action="store_true")
    a = ap.parse_args()
    for slug in (EPISODES if a.slugs == ["all"] else a.slugs):
        sc = build(slug, goals_from_llm=not a.no_llm_goals)
        print(slug, [(o["id"], o["shape"]["type"], [round(v, 3) for v in o["pos"]]) for o in sc["objects"]],
              "goals:", sc["droid"].get("goals"), "not segmented:", sc["perception_notes"]["not_segmented"])


if __name__ == "__main__":
    main()
