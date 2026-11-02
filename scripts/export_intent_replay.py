"""Export recorded Muse Spark intent responses for the bundled tables into a shipped replay file ($0, no API call).

For every candidate (scene, instruction) - GoalBench cases, hold-out cases and each scene's demo_instruction - this
builds the request exactly as the product's compile step does (MuseAgent.compile_intent, reachability-based scene
summary, Contributor/Standard routing by the scene's privacy) and computes the same disk-cache key as
rehearsal.meta_client.chat_json. Only requests that already have a real cached response in budget/cache/ are
exported; nothing is generated. The public demo has no API key and no budget/ folder, so /api/intent reads this
file to answer those exact requests offline (see rehearsal/meta_client.py, REPLAY_FILE).

    python scripts/export_intent_replay.py          # writes bench/goals/results/muse_intent_replay.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rehearsal import agents as A  # noqa: E402
from rehearsal import meta_client as M  # noqa: E402
from rehearsal.executor import reachability_report  # noqa: E402


def candidates() -> list[dict]:
    out = []
    for c in json.loads((ROOT / "bench/goals/cases.json").read_text()):
        out.append({"scene": c["scene"], "instruction": c["instruction"], "case": c["id"], "set": "GoalBench"})
    hp = ROOT / "bench/holdout/cases.json"
    if hp.exists():
        for c in json.loads(hp.read_text()):
            out.append({"scene": c["scene"], "instruction": c["instruction"], "case": c["id"], "set": "hold-out"})
    for p in sorted((ROOT / "scenes").glob("*.json")):
        spec = json.loads(p.read_text())
        if spec.get("demo_instruction"):
            out.append({"scene": p.stem, "instruction": spec["demo_instruction"], "case": None, "set": "demo"})
    return out


def main() -> int:
    reach, specs, entries, miss = {}, {}, {}, 0
    for c in candidates():
        s = c["scene"]
        if s not in specs:
            p = ROOT / "scenes" / f"{s}.json"
            if not p.exists():
                continue
            specs[s] = json.loads(p.read_text())
            reach[s] = reachability_report(specs[s])
        spec = specs[s]
        data = "public" if spec.get("privacy") == "public" else "private"
        key = A.intent_request_key(A.scene_summary(spec, reach[s]), c["instruction"], data)
        hit = M._cache_get(key)
        if hit is None:
            miss += 1
            continue
        e = entries.setdefault(key, {"content": hit["content"], "model": hit.get("model"), "scene": s,
                                     "instruction": c["instruction"], "cases": []})
        if c["case"]:
            e["cases"].append(f"{c['set']} {c['case']}")
    path = M.REPLAY_FILE
    path.write_text(json.dumps(dict(sorted(entries.items(), key=lambda kv: (kv[1]["scene"], kv[1]["instruction"]))),
                               ensure_ascii=False, indent=1))
    print(f"{len(entries)} recorded responses -> {path.relative_to(ROOT)} ({miss} candidates had no recording)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
