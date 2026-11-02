"""Run Muse Spark goal compilation on GoalBench and score it.

Arms (same system prompt, scene summary and repeat tags; paired by (case, repeat)):
  A  product   MuseAgent.compile_goals exactly: its user prompt and GOAL_SCHEMA.
  B  vocab     A's prompt with the vocabulary extended by undisturbed / near / clear (the predicates the
               bench found missing). Isolates the vocabulary change.
  C  confirm   B plus one mitigation: the model must list its assumptions and may return status
               "clarify" (one question) or "refuse" instead of goals. In the product the assumptions would be
               shown for confirmation before anything runs.

Repeats k > 0 append an inert "[eval trial k]" line to the user prompt (as EVAL.md does), so each repeat is an
independent sample instead of a cache hit; repeat 0 of arm A is the product's exact request.
Every call goes through rehearsal.meta_client.chat_json (cache + ledger + hard cap). Callers are tagged
goalbench.<arm>.<case>.r<k>; --budget stops the run once the ledger shows that much goalbench spend.

    .venv/bin/python -m bench.goals.run --arms A B C --reps 3 --budget 0.55
    .venv/bin/python -m bench.goals.run --score-only
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.goals import score as S  # noqa: E402
from bench.goals.cases import CASES  # noqa: E402
from rehearsal import agents as A  # noqa: E402
from rehearsal import meta_client as M  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CALLER = "goalbench."

GOAL_ITEM_EXT = json.loads(json.dumps(A.GOAL_SCHEMA["properties"]["goals"]["items"]))
GOAL_ITEM_EXT["properties"]["pred"]["enum"] = ["inside", "on", "upright", "stacked", "stable",
                                               "undisturbed", "near", "clear"]
SCHEMA_B = {"type": "object", "properties": {"goals": {"type": "array", "items": GOAL_ITEM_EXT}},
            "required": ["goals"], "additionalProperties": False}
SCHEMA_C = {"type": "object", "properties": {
    "assumptions": {"type": "array", "items": {"type": "string"}},
    "status": {"type": "string", "enum": ["ok", "clarify", "refuse"]},
    "question": {"type": "string"},
    "goals": {"type": "array", "items": GOAL_ITEM_EXT}},
    "required": ["assumptions", "status", "question", "goals"], "additionalProperties": False}

PROMPT_A = ("Scene objects:\n{summary}\n\nInstruction: {instruction}\n\n"
            "Translate the instruction into checkable predicates. inside(a,b), on(a,b), upright(a), "
            "stacked(bottom,...,top), stable(a). Use only ids from the scene.")
PROMPT_B = ("Scene objects:\n{summary}\n\nInstruction: {instruction}\n\n"
            "Translate the instruction into checkable predicates. inside(a,b), on(a,b), upright(a), "
            "stacked(bottom,...,top), stable(a), undisturbed(a,...) (must not be moved or tipped), "
            "near(a,b) (a ends beside b on the same surface), clear(a) (nothing may end up on top of a). "
            "Use only ids from the scene.")
PROMPT_C = PROMPT_B + (
    "\n\nBefore anything runs, the user sees your assumptions and confirms them. List every assumption you "
    "made in `assumptions`, including constraints the user implies without stating them. Set status to "
    "\"clarify\" and ask one short question in `question` if the instruction has several reasonable readings "
    "that lead to different goals. Set status to \"refuse\" and say why in `question` if it cannot be done with "
    "this robot (one transport action; object sizes; reachability), is unsafe, or names something that is not "
    "in the scene. Otherwise status is \"ok\", `question` is empty, and `goals` holds the predicates.")
ARMS = {"A": (PROMPT_A, A.GOAL_SCHEMA, "Goals", S.PRODUCT_VOCAB),
        "B": (PROMPT_B, SCHEMA_B, "Goals", S.EXTENDED_VOCAB),
        "C": (PROMPT_C, SCHEMA_C, "GoalsConfirm", S.EXTENDED_VOCAB)}

# The product prompt must stay byte-identical to MuseAgent.compile_goals (checked by tests).
assert PROMPT_A.format(summary="S", instruction="I").endswith(
    "stacked(bottom,...,top), stable(a). Use only ids from the scene.")


def load_scenes() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in (ROOT / "scenes").glob("*.json")}


def summaries(scenes: dict[str, dict]) -> dict[str, list[dict]]:
    """Scene summaries exactly as the product builds them (reachability cached on disk)."""
    cache = RESULTS / "reach.json"
    reach = json.loads(cache.read_text()) if cache.exists() else {}
    need = {c["scene"] for c in CASES} - set(reach)
    if need:
        from rehearsal.executor import reachability_report
        for s in sorted(need):
            reach[s] = reachability_report(scenes[s])
        RESULTS.mkdir(exist_ok=True)
        cache.write_text(json.dumps(reach, indent=1))
    return {s: A.scene_summary(scenes[s], reach[s]) for s in {c["scene"] for c in CASES}}


def user_prompt(arm: str, case: dict, summary: list[dict], rep: int) -> str:
    u = ARMS[arm][0].format(summary=json.dumps(summary), instruction=case["instruction"])
    return u + (f"\n\n[eval trial {rep}]" if rep else "")


def goalbench_spend() -> float:
    p = M.budget_dir() / "ledger.jsonl"
    if not p.exists():
        return 0.0
    tot = 0.0
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(e.get("caller", "")).startswith(CALLER):
            tot += e.get("cost", 0.0)
    return tot


_lock = threading.Lock()


def call(arm: str, case: dict, summary: list[dict], rep: int, budget: float) -> dict:
    if goalbench_spend() >= budget:
        raise RuntimeError("budget")
    prompt, schema, name, _ = ARMS[arm]
    msgs = [{"role": "system", "content": A.SYSTEM}, {"role": "user", "content": user_prompt(arm, case, summary, rep)}]
    try:
        return M.chat_json(msgs, schema, name, caller=f"{CALLER}{arm}.{case['id']}.r{rep}", data="public")
    except M.BudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001 - recorded as call_failed
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def raw_path(arm: str) -> Path:
    return RESULTS / f"raw_{arm}.jsonl"


def load_raw(arm: str) -> dict:
    p = raw_path(arm)
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            out[(r["case"], r["rep"])] = r
    return out


def run(arms, reps, budget, only=None, workers=3):
    scenes = load_scenes()
    summ = summaries(scenes)
    todo = []
    for arm in arms:
        done = {k for k, v in load_raw(arm).items() if "error" not in v["output"]}
        for rep in range(reps):
            for c in CASES:
                if (only is None or c["id"] in only) and (c["id"], rep) not in done:
                    todo.append((arm, c, rep))
    print(f"{len(todo)} calls to make; goalbench spend so far ${goalbench_spend():.4f}", flush=True)

    def work(item):
        arm, c, rep = item
        try:
            out = call(arm, c, summ[c["scene"]], rep, budget)
        except (RuntimeError, M.BudgetExceeded) as e:
            return f"stop: {e}"
        with _lock, open(raw_path(arm), "a") as f:
            f.write(json.dumps({"arm": arm, "case": c["id"], "rep": rep, "output": out}, ensure_ascii=False) + "\n")
        return "ok"

    RESULTS.mkdir(exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        from concurrent.futures import as_completed
        futs = [ex.submit(work, t) for t in todo]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r != "ok":
                print(r, flush=True)
            if i % 30 == 0:
                print(f"{i}/{len(todo)} ${goalbench_spend():.4f}", flush=True)
    print(f"done; goalbench spend ${goalbench_spend():.4f}")


def scored_rows(arm: str) -> list[dict]:
    scenes = load_scenes()
    byid = {c["id"]: c for c in CASES}
    rows = []
    for (cid, rep), r in sorted(load_raw(arm).items()):
        c = byid.get(cid)
        if c is None:
            continue
        ids = [o["id"] for o in scenes[c["scene"]]["objects"]]
        rows.append({"arm": arm, "case": cid, "rep": rep, "category": c["category"], "lang": c["lang"],
                     "expect": c["expect"], "output": r["output"],
                     "score": S.score_trial(c, r["output"], ids, ARMS[arm][3])})
    return rows


def report(arms) -> dict:
    rows = {a: scored_rows(a) for a in arms}
    out = {"arms": {}, "paired": {}}
    for a, rs in rows.items():
        out["arms"][a] = {
            "overall": S.aggregate(rs)["all"] if rs else (0, 0),
            "overall_expressible": S.aggregate(rs, field="pass_expressible").get("all", (0, 0)),
            "exact": S.aggregate(rs, field="exact").get("all", (0, 0)),
            "by_category": S.aggregate(rs, key=lambda r: r["category"]),
            "by_lang": S.aggregate(rs, key=lambda r: r["lang"]),
            "primary_errors": dict(S.error_counts(rs)),
            "all_errors": dict(S.error_counts(rs, primary_only=False)),
            "call_failed": sum(r["score"]["primary"] == "call_failed" for r in rs),
        }
        # zh/en pairs: only cases that have both languages
        pairs = {c["pair"] for c in CASES if c["lang"] == "zh"}
        sub = [r for r in rs if next(c for c in CASES if c["id"] == r["case"])["pair"] in pairs]
        out["arms"][a]["paired_lang"] = S.aggregate(sub, key=lambda r: r["lang"])
    for x, y in (("B", "A"), ("C", "B"), ("C", "A")):
        if x in rows and y in rows:
            out["paired"][f"{x}_vs_{y}"] = S.paired(rows[x], rows[y])
            out["paired"][f"{x}_vs_{y}_by_category"] = {
                cat: S.paired([r for r in rows[x] if r["category"] == cat], [r for r in rows[y] if r["category"] == cat])
                for cat in sorted({c["category"] for c in CASES})}
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1, default=list))
    with open(RESULTS / "scored.jsonl", "w") as f:
        for a in arms:
            for r in rows[a]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def print_report(out):
    for a, d in out["arms"].items():
        print(f"\n== arm {a}: pass {S.fmt_rate(*d['overall'])} | expressible-only {S.fmt_rate(*d['overall_expressible'])}"
              f" | exact {S.fmt_rate(*d['exact'])} | call_failed {d['call_failed']}")
        for k, v in sorted(d["by_category"].items()):
            print(f"   {k:12s} {S.fmt_rate(*v)}")
        for k, v in sorted(d["by_lang"].items()):
            print(f"   lang {k:7s} {S.fmt_rate(*v)}")
        for k, v in sorted(d["paired_lang"].items()):
            print(f"   paired-lang {k:3s} {S.fmt_rate(*v)}")
        print("   primary errors:", d["primary_errors"])
    for k, v in out["paired"].items():
        if not k.endswith("category"):
            print(k, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--budget", type=float, default=0.55)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--workers", type=int, default=3, help="concurrent API requests (threads in one process)")
    args = ap.parse_args()
    if not args.score_only:
        from rehearsal import load_env
        load_env()
        run(args.arms, args.reps, args.budget, set(args.only) if args.only else None, args.workers)
    print_report(report(args.arms))
