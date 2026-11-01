"""Export a rehearsed table (out/<scene>/report.json + scene + photo) as a Rehearsal Bench case.

    .venv/bin/python scripts/bench_export.py --all                     # the 10 bundled public scenes
    .venv/bin/python scripts/bench_export.py kitchen_01                # one scene (by name or report path)
    .venv/bin/python scripts/bench_export.py photo_ab12cd --owner-consent   # a web upload, after the owner agreed

Writes bench/cases/<id>/: case.json (schema bench/schema.json), scene.json, photo.jpg (<= 300 KB),
failures.jsonl (every recorded failed plan with its contact log) and plans/ (cached planner plans).
Zero Meta API calls: everything comes from out/, scenes/ and eval/results/. Failed plans from the eval
files are re-simulated locally (MuJoCo) to recover their contact logs.

Consent rules (opt-in only):
  * A user upload (scene name photo_*, or a photo under uploads/) is refused unless --owner-consent says
    the person who took the photo agreed to share it as a public test case.
  * A scene whose privacy is not "public" (Standard mode, the default for uploads) is refused unless
    BOTH --owner-consent and --allow-private are given.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import caselib as C  # noqa: E402

REPO = C.REPO
BUNDLED = ["desk_01", "desk_02", "kitchen_01", "lamp_01", "storage_01", "windowsill_01",
           "real_ycbv59", "aigen_desk_lamp_mug", "aigen_desk_walnut", "aigen_kitchen"]
EVAL_MAIN = "eval/results/eval_20261031_173045_round3_main.json"
EVAL_FILES = [EVAL_MAIN, "eval/results/eval_20261031_170047_round3_kitchen_zh.json",
              "eval/results/round3_kitchen_refgoals.json"]
PHOTO_MAX_BYTES = 300_000
PHOTO_MAX_SIDE = 1024
CJK = re.compile(r"[㐀-鿿]")


class ExportRefused(Exception):
    pass


# ------------------------------------------------------------------ policy
def is_upload(name: str, scene: dict) -> bool:
    photo = str((scene.get("photo") or {}).get("file") or "")
    return name.startswith("photo_") or photo.startswith("uploads/")


def check_policy(name: str, scene: dict, owner_consent: bool, allow_private: bool) -> None:
    upload = is_upload(name, scene)
    privacy = scene.get("privacy", "private")
    if upload and not owner_consent:
        raise ExportRefused(f"{name} is a user upload; pass --owner-consent only if the person who took the photo "
                            "agreed to publish it as a public test case")
    if privacy != "public" and not (owner_consent and allow_private):
        raise ExportRefused(f"{name} has privacy={privacy!r} (Standard mode / private); refused by default. "
                            "Only with the owner's explicit agreement: --owner-consent --allow-private")


def provenance(name: str, scene: dict) -> dict:
    photo = (scene.get("photo") or {})
    pfile = str(photo.get("file") or "")
    cam = photo.get("camera_info") or {}
    if is_upload(name, scene):
        kind, dataset = "user_photo", None
    elif scene.get("ai_generated") or cam.get("ai_generated"):
        kind, dataset = "ai_generated_photo", None
    elif cam.get("dataset") or pfile.startswith(("photos/real/", "datasets/")):
        kind, dataset = "real_dataset_photo", cam.get("dataset")
    elif pfile:
        kind, dataset = "real_photo", None
    else:
        kind, dataset = "hand_designed", None
    return {"kind": kind, "source": scene.get("source", ""), "dataset": dataset,
            "photo_is_render": kind == "hand_designed",
            "photo_original": None if kind == "user_photo" else (pfile or None)}


def licence_for(prov: dict, owner_consent: bool, licence: str | None) -> dict:
    k = prov["kind"]
    if k == "hand_designed":
        return {"licence": "Rehearsal project (code licence not chosen yet, see THIRD_PARTY_NOTICES.md)",
                "attribution": "Hand-designed scene by the Rehearsal team. photo.jpg is a MuJoCo render "
                               "(Franka Panda model from MuJoCo Menagerie, Apache-2.0).",
                "people_or_private_places": False}
    if k == "real_dataset_photo":
        return {"licence": "MIT (YCB-Video, (c) 2017 University of Washington RSE Lab)" if "YCB" in (prov["dataset"] or "")
                else f"see datasets/SOURCES.md ({prov['dataset']})",
                "attribution": "Xiang et al., PoseCNN, RSS 2018; Hodan et al., BOP benchmark. Photo resized for the case; "
                               "scene geometry derived by the Rehearsal team.",
                "people_or_private_places": False}
    if k == "ai_generated_photo":
        return {"licence": "AI-generated test image made by the project owner with a text-to-image model; same terms as "
                           "the Rehearsal project (not chosen yet)",
                "attribution": "AI-generated, NOT a real capture: it shows no real desk, place or person "
                               "(photos/ai_gen/SOURCES.md).",
                "people_or_private_places": False}
    return {"licence": licence or ("CC BY 4.0 (as agreed by the contributor)" if owner_consent else "none"),
            "attribution": "Contributed by the photo's owner (opt-in).", "people_or_private_places": None}


# ------------------------------------------------------------------ photo
def find_photo(name: str, scene: dict, report_dir: Path) -> Path | None:
    pfile = (scene.get("photo") or {}).get("file")
    if pfile and (REPO / pfile).is_file():
        return REPO / pfile
    for cand in (report_dir / "thumb.png", report_dir / "attempt_1.gif"):
        if cand.is_file():
            return cand
    return None


def write_photo(src: Path, dst: Path) -> dict:
    """JPEG, longest side <= 1024 px, <= 300 KB, no EXIF. Returns {width, height, scale, bytes}."""
    from PIL import Image
    with Image.open(src) as im:
        im.seek(0)
        im = im.convert("RGB")
        w0 = im.width
        side = PHOTO_MAX_SIDE
        while True:
            img = im.copy()
            img.thumbnail((side, side))
            for q in (88, 82, 76, 70, 62, 55):
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=q, optimize=True)
                if buf.tell() <= PHOTO_MAX_BYTES:
                    dst.write_bytes(buf.getvalue())
                    return {"width": img.width, "height": img.height, "scale": round(img.width / w0, 5),
                            "bytes": buf.tell(), "jpeg_quality": q}
            side = int(side * 0.8)


# ------------------------------------------------------------------ failures
def attempt_record(a: dict, case_goals: list[dict], scene: dict, stage: str, src: str, agent: str,
                   instruction: str) -> dict:
    """A failed attempt from report.json. Re-scored against the case goals (which equal the report's
    goals unless the scene carries reference goals); the contact log is the recorded one."""
    violated = [C.goal_str(g) for g in a.get("goals", []) if not g.get("ok")]
    err = a.get("execution_error")
    rec_sim = {"execution_error": err["error"] if err else None, "events": a.get("events", []), "violated": violated}
    return {"stage": stage, "source": src, "planner": agent, "instruction": instruction,
            "plan": a["plan"], "plan_key": C.plan_key(a["plan"]), "repeats": 1,
            "failure_types": C.failure_types(rec_sim, a["plan"]),
            "violated": violated,
            "goals": [{"pred": g["pred"], "args": g["args"], "ok": g["ok"], "why": g["why"]} for g in a.get("goals", [])],
            "execution_error": rec_sim["execution_error"],
            "evidence": a.get("evidence", []),
            "contacts": C.contact_log(a.get("events", [])),
            "disturbed": a.get("disturbed", []),
            "log_origin": "recorded (report.json)"}


def eval_first_plans(name: str, eval_files=EVAL_FILES) -> list[dict]:
    """First plans per repeat from cached eval runs (full_log cells: one per repeat)."""
    out = []
    for rel in eval_files:
        p = REPO / rel
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        for c in d.get("cells", []):
            if c.get("scene") != name or c.get("condition") != "full_log" or c.get("status", "ok") != "ok":
                continue
            if not c.get("attempts"):
                continue
            first, last = c["attempts"][0], c["attempts"][-1]
            out.append({"source": rel, "repeat": c.get("repeat"), "instruction": c["instruction"],
                        "lang": "zh" if CJK.search(c["instruction"]) else "en",
                        "agent": c.get("agent_name", c.get("agent")),
                        "first": first["plan"], "first_ok": first.get("ref_success", first.get("success")),
                        "first_types": first.get("failure_types", []),
                        "after_log": last["plan"], "after_log_ok": last.get("ref_success", last.get("success")),
                        "attempts_used": c.get("attempts_used", len(c["attempts"]))})
    return out


def eval_failure_records(scene: dict, goals: list[dict], rows: list[dict], ex) -> list[dict]:
    """Distinct failed first plans from the eval runs, re-simulated to recover their contact logs."""
    groups: dict[tuple, dict] = {}
    for r in rows:
        if r["first_ok"]:
            continue
        key = (r["instruction"], C.plan_key(r["first"]))
        g = groups.setdefault(key, {"row": r, "repeats": []})
        g["repeats"].append(f"{r['source']}#repeat={r['repeat']}")
    keys = list(groups)
    sims = list(ex.map(_resim, [(scene, groups[k]["row"]["first"], goals) for k in keys])) if keys else []
    recs = []
    for k, sim in zip(keys, sims):
        r = groups[k]["row"]
        recs.append({"stage": "first_plan", "source": groups[k]["repeats"][0], "planner": r["agent"],
                     "instruction": r["instruction"], "lang": r["lang"],
                     "plan": r["first"], "plan_key": k[1], "repeats": len(groups[k]["repeats"]),
                     "repeat_sources": groups[k]["repeats"],
                     "failure_types_recorded": sorted(r["first_types"]),
                     "failure_types": C.failure_types(sim, r["first"]),
                     "violated": sim["violated"], "goals": sim["goals"],
                     "execution_error": sim["execution_error"], "evidence": sim["evidence"],
                     "contacts": C.contact_log(sim["events"]), "disturbed": sim["disturbed"],
                     "log_origin": "re-simulated from the cached plan on the case's code version",
                     "reproduces": not sim["success"]})
    return recs


def _resim(args):
    scene, plan, goals = args
    return C.simulate(scene, plan, goals)


def stress_counts(st: dict) -> dict:
    """How often each predicate / error kind failed across the recorded stress runs."""
    out: dict[str, int] = {}
    for r in st.get("runs", []):
        for f in r.get("failed", []):
            k = f.split(":")[0] if "(" in f.split(":")[0] else ("execution error: " + (
                "protective stop" if "protective" in f else "out of reach (IK)" if "reach" in f.lower() or "ik" in f.lower()
                else f[:60]))
            out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ------------------------------------------------------------------ export
def resolve(target: str) -> tuple[str, Path]:
    """Scene name or path to a report.json -> (name, report path)."""
    p = Path(target)
    if p.suffix == ".json" and p.name == "report.json":
        return p.parent.name, p if p.is_absolute() else REPO / p
    return target, REPO / "out" / target / "report.json"


def export_case(target: str, out_root: Path = C.CASES, owner_consent: bool = False, allow_private: bool = False,
                licence: str | None = None, eval_files=EVAL_FILES, ex=None, log=print) -> Path:
    name, rpath = resolve(target)
    if not rpath.is_file():
        raise FileNotFoundError(f"no report at {rpath} (rehearse the scene first)")
    report = json.loads(rpath.read_text())
    scene = json.loads(json.dumps(report["scene_spec"]))
    check_policy(name, scene, owner_consent, allow_private)

    case_id = name
    d = Path(out_root) / case_id
    if d.exists():
        shutil.rmtree(d)
    (d / "plans").mkdir(parents=True)

    goals, goals_src = C.case_goals(scene, report.get("goals"))
    sig, sig_src = C.sigma_map(scene)
    prov = provenance(name, scene)
    lic = licence_for(prov, owner_consent, licence)

    photo_src = find_photo(name, scene, rpath.parent)
    photo_info = write_photo(photo_src, d / "photo.jpg") if photo_src else None
    if scene.get("photo") and photo_info:
        scene["photo"] = dict(scene["photo"], file="photo.jpg", case_photo_scale=photo_info["scale"])
        if prov["kind"] == "user_photo":
            scene["photo"].pop("paper_corners_px", None)
    (d / "scene.json").write_text(json.dumps(scene, ensure_ascii=False, indent=1))

    instr = report["instruction"]
    en = scene.get("demo_instruction") if CJK.search(instr) else instr
    zh = instr if CJK.search(instr) else scene.get("demo_instruction_zh")

    # cached plans
    atts = report["attempts"]
    agent = report.get("agent", "unknown")
    plans = {"muse_first": atts[0]["plan"], "muse_after_log": atts[-1]["plan"],
             "scripted_baseline": C.scripted_baseline(goals)}
    files_plans = {}
    for k, p in plans.items():
        (d / "plans" / f"{k}.json").write_text(json.dumps(p, ensure_ascii=False, indent=1))
        files_plans[k] = f"plans/{k}.json"
    rows = eval_first_plans(name, eval_files)
    main_rows = [r for r in rows if r["source"] == EVAL_MAIN]
    if main_rows:
        for k in ("first", "after_log"):
            lines = [json.dumps({"source": r["source"], "repeat": r["repeat"], "instruction": r["instruction"],
                                 "plan": r[k], "recorded_success": r[f"{k}_ok"],
                                 **({"attempts_used": r["attempts_used"]} if k == "after_log" else {})},
                                ensure_ascii=False) for r in main_rows]
            (d / "plans" / f"eval_round3_{k}.jsonl").write_text("\n".join(lines) + "\n")
            files_plans[f"eval_round3_{k}"] = f"plans/eval_round3_{k}.jsonl"

    # failures: recorded attempts of the shipped run, then distinct failed first plans from the eval runs
    fails = []
    for i, a in enumerate(atts):
        if a.get("success"):
            continue
        fails.append(attempt_record(a, goals, scene, "first_plan" if i == 0 else "revision",
                                    f"out/{name}/report.json#attempt={a['attempt']}", agent, instr))
    own_ex = ex is None and rows
    if own_ex:
        ex = C.pool()
    try:
        fails += eval_failure_records(scene, goals, rows, ex) if rows else []
    finally:
        if own_ex:
            ex.shutdown()
    for i, f in enumerate(fails, 1):
        f["id"] = f"{case_id}/f{i:03d}"
        probs = C.validate(f, ref="#/$defs/failure", path=f["id"])
        if probs:
            raise ValueError(f"failure record does not match bench/schema.json: {probs[:5]}")
    (d / "failures.jsonl").write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in fails))

    st = report.get("stress_test")
    ap = report.get("active_perception") or {}
    meta = {
        "schema": C.SCHEMA_ID, "schema_version": C.SCHEMA_VERSION,
        "id": case_id, "title": scene.get("title") or name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provenance": prov,
        "privacy": {"scene_privacy": scene.get("privacy", "private"), "user_upload": prov["kind"] == "user_photo",
                    "owner_consent": bool(owner_consent), **lic},
        "instruction": {"en": en, "zh": zh, "rehearsed": instr},
        "goals": goals, "goals_source": goals_src,
        "perception": {"sigma_m": sig, "default_sigma_m": C.SIGMA, "source": sig_src},
        "stress_protocol": C.PROTOCOL,
        "files": {"photo": "photo.jpg" if photo_info else None, "scene": "scene.json",
                  "failures": "failures.jsonl", "plans": files_plans},
        "photo": photo_info,
        "objects": [{"id": o["id"], "label": o.get("label", o["id"]), "shape": o["shape"]["type"]}
                    for o in scene["objects"]],
        "recorded": {
            "product_run": {"report": f"out/{name}/report.json", "agent": agent,
                            "attempts": [{"attempt": a["attempt"], "success": bool(a["success"]),
                                          "violated": [C.goal_str(g) for g in a.get("goals", []) if not g.get("ok")],
                                          "execution_error": (a.get("execution_error") or {}).get("error")}
                                         for a in atts],
                            "stress": {"successes": st["successes"], "n": st["n"], "seed0": st.get("seed0", 0),
                                       "failed_seeds": [r["seed"] for r in st["runs"] if not r["success"]],
                                       "violated_counts": stress_counts(st)} if st else None,
                            "look_again_heldout": ap.get("heldout")},
            "eval_runs": sorted({r["source"] for r in rows}),
            "eval_first_plans": {"n": len(rows), "failed": sum(1 for r in rows if not r["first_ok"])},
        },
        "failures": {"count": len(fails), "first_plan": sum(f["stage"] == "first_plan" for f in fails),
                     "revision": sum(f["stage"] == "revision" for f in fails),
                     "failed_runs_covered": sum(f["repeats"] for f in fails)},
        "code": C.code_version(),
    }
    probs = C.validate(meta)
    if probs:
        raise ValueError(f"case.json for {case_id} does not match bench/schema.json: {probs[:5]}")
    (d / "case.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    log(f"exported {case_id}: {prov['kind']}, {len(fails)} recorded failures, photo "
        f"{photo_info['bytes'] // 1000 if photo_info else 0} KB -> {d.relative_to(REPO) if d.is_relative_to(REPO) else d}")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="scene names or paths to out/<scene>/report.json")
    ap.add_argument("--all", action="store_true", help="export the bundled public scenes: " + ", ".join(BUNDLED))
    ap.add_argument("--out", default=str(C.CASES), help="cases directory (default bench/cases)")
    ap.add_argument("--owner-consent", action="store_true",
                    help="the photo's owner agreed to publish this table as a public test case (required for uploads)")
    ap.add_argument("--allow-private", action="store_true",
                    help="with --owner-consent: also allow a scene saved as private / Standard mode")
    ap.add_argument("--licence", default=None, help="licence for a contributed photo (default CC BY 4.0)")
    args = ap.parse_args(argv)
    targets = list(args.targets) + (BUNDLED if args.all else [])
    if not targets:
        ap.error("give scene names or --all")
    rc = 0
    ex = C.pool()
    try:
        for t in targets:
            try:
                export_case(t, Path(args.out), args.owner_consent, args.allow_private, args.licence, ex=ex)
            except ExportRefused as e:
                print(f"REFUSED {t}: {e}", file=sys.stderr)
                rc = 2
            except (FileNotFoundError, ValueError) as e:
                print(f"ERROR {t}: {e}", file=sys.stderr)
                rc = rc or 1
    finally:
        ex.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
