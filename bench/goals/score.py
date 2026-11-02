"""Score compiled goals against GoalBench references (see cases.py for the case format).

Scoring is done on the goal set the product would actually verify: the compiled goals plus the implicit
undisturbed() goal that `with_implicit_goals` adds for every object they do not mention. stacked() is expanded to
its chain of on() atoms and undisturbed() to one atom per object, so matching is order-insensitive.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from rehearsal.predicates import with_implicit_goals

PRODUCT_VOCAB = frozenset({"inside", "on", "upright", "stacked", "stable", "undisturbed"})  # undisturbed: implicit only
EXTENDED_VOCAB = PRODUCT_VOCAB | {"near", "clear"}
PLACE = ("on", "inside", "near")
ERROR_PRIORITY = ["call_failed", "hallucinated_object", "failed_to_flag", "unnecessary_flag", "vocab_gap",
                  "forbidden_goal", "wrong_binding", "wrong_order", "wrong_relation", "missed_goal",
                  "missed_implicit", "dropped_protection", "extra_action", "over_constraint", "unflagged_ambiguity"]

_ATOM = re.compile(r"^\s*(\w+)\s*\((.*)\)\s*$")


def parse_atom(s: str) -> tuple:
    m = _ATOM.match(s)
    if not m:
        raise ValueError(f"bad atom {s!r}")
    return (m.group(1), tuple(a.strip() for a in m.group(2).split(",") if a.strip()))


def parse_alts(s: str) -> list[tuple]:
    return [parse_atom(a) for a in s.split("|")]


def fmt(atom: tuple) -> str:
    return f"{atom[0]}({','.join(atom[1])})"


def atoms_of(goals: list[dict], ids: list[str]) -> set[tuple]:
    """The verified goal set as atoms: compiled goals + the product's implicit undisturbed()."""
    out = set()
    for g in with_implicit_goals([dict(g) for g in goals], ids):
        p, args = g["pred"], tuple(g["args"])
        if p == "stacked" and len(args) >= 2:
            out |= {("on", (u, l)) for l, u in zip(args, args[1:])}
        elif p == "undisturbed":
            out |= {("undisturbed", (a,)) for a in args}
        else:
            out.add((p, args))
    return out


def is_flagged(output: dict) -> bool:
    """The model declined: status clarify/refuse, or (arms without a status field) no goal that asks for any
    change, i.e. an empty list or only undisturbed(). NB the product has no refusal path: it would plan for such a
    goal set and report success, since nothing moving satisfies undisturbed()."""
    if output.get("status", "ok") != "ok":
        return True
    return not [g for g in output.get("goals") or [] if g.get("pred") != "undisturbed"]


def _ref_atoms(ref: dict, key: str) -> list[list[tuple]]:
    return [parse_alts(s) for s in ref.get(key, [])]


def _mentioned(ref: dict) -> set[str]:
    ids = set()
    for key in ("required", "implicit", "optional"):
        for alts in _ref_atoms(ref, key):
            for _, args in alts:
                ids |= set(args)
    return ids


def _moved(alt_lists) -> set[str]:
    return {args[0] for alts in alt_lists for p, args in alts if p in PLACE and args}


def score_ref(ref: dict, got: set[tuple], ids: list[str], vocab=EXTENDED_VOCAB) -> dict:
    req = [(a, "required") for a in _ref_atoms(ref, "required")] + [(a, "implicit") for a in _ref_atoms(ref, "implicit")]
    keep = sorted((set(ids) - _mentioned(ref)) | set(ref.get("keep", [])))
    req += [([("undisturbed", (k,))], "keep") for k in keep
            if not any(("undisturbed", (k,)) in alts for alts, _ in req)]
    optional = {a for alts in _ref_atoms(ref, "optional") for a in alts}
    forbidden = {a for alts in _ref_atoms(ref, "forbidden") for a in alts}
    moved = _moved([a for a, _ in req])
    got_moved = {args[0] for p, args in got if p in PLACE and args}
    got_ids = {x for p, args in got if p != "undisturbed" for x in args}

    errors, missing, consumed, unexpressible = [], [], set(), []
    satisfied = set()
    for alts, kind in req:
        hit = [a for a in alts if a in got]
        if hit:
            satisfied.add(hit[0])
            continue
        if not any(p in vocab for p, _ in alts):
            unexpressible.append(fmt(alts[0]))
            continue
        p, args = alts[0]
        missing.append(fmt(alts[0]))
        if p == "undisturbed":
            x = args[0]
            if any(c[0] in PLACE and c[1][:1] == (x,) for c in consumed):
                continue        # already counted as the wrong binding that moved x
            errors.append("extra_action" if x in got_moved else "dropped_protection" if x in got_ids else "missed_implicit")
            continue
        if kind == "implicit":
            errors.append("missed_implicit")
            continue
        if p == "on" and ("on", args[::-1]) in got:
            errors.append("wrong_order")
            consumed.add(("on", args[::-1]))
            continue
        rel = [g for g in got if g[1] == args and g[0] != p and g[0] in PLACE]
        if rel:
            errors.append("wrong_relation")
            consumed |= set(rel)
            continue
        bind = [g for g in got if g[0] == p and len(g[1]) == len(args) and g not in satisfied
                and sum(x == y for x, y in zip(g[1], args)) == len(args) - 1]
        if p in PLACE:      # same relation, one argument swapped for another object (or the right pair, other relation)
            bind += [g for g in got if g[0] in PLACE and g[0] != p and len(g[1]) == 2 and len(args) == 2
                     and sum(x == y for x, y in zip(g[1], args)) == 1]
        if bind:
            errors.append("wrong_binding")
            consumed |= set(bind)
            continue
        errors.append("missed_goal")

    all_req_atoms = {a for alts, _ in req for a in alts}
    involved = {x for alts, _ in req for p, args in alts if p != "undisturbed" for x in args}
    benign = optional | {(p, (x,)) for x in involved for p in ("upright", "stable")}
    extra = []
    for g in sorted(got - all_req_atoms - consumed - forbidden):
        # upright/stable of an uninvolved object asks for nothing extra; its one harm (mentioning the object
        # strips the implicit undisturbed) is scored as dropped_protection on the keep atom instead.
        if g[0] in ("undisturbed", "upright", "stable") or g in benign:
            continue
        extra.append(fmt(g))
        errors.append("extra_action" if g[0] in PLACE and g[1] and g[1][0] not in moved else "over_constraint")
    hit_forbidden = sorted(fmt(g) for g in got & forbidden)
    errors += ["forbidden_goal"] * len(hit_forbidden)
    unlisted_extras = [g for g in got - all_req_atoms if g[0] != "undisturbed" and g not in optional]
    return {"errors": errors, "missing": missing, "extra": extra, "forbidden": hit_forbidden,
            "unexpressible": unexpressible, "exact": not errors and not unexpressible and not unlisted_extras}


def score_trial(case: dict, output: dict | None, ids: list[str], vocab=EXTENDED_VOCAB) -> dict:
    """output: {"goals": [...], "status"?: "ok"|"clarify"|"refuse", ...} or None if the call failed."""
    if output is None or "error" in (output or {}):
        return {"pass": False, "pass_expressible": False, "exact": False, "flagged": False,
                "errors": ["call_failed"], "primary": "call_failed"}
    goals = output.get("goals") or []
    flagged = is_flagged(output)
    halluc = sorted({a for g in goals for a in g.get("args", []) if a not in ids})
    base = {"flagged": flagged, "hallucinated": halluc}
    errors = ["hallucinated_object"] if halluc else []
    if case["expect"] == "flag":
        ok = flagged
        errors += [] if ok else ["failed_to_flag"]
        return _done(base, ok and not halluc, ok and not halluc, errors, ok and not halluc)
    if flagged:
        ok = case["expect"] == "either"
        return _done(base, ok, ok, errors + ([] if ok else ["unnecessary_flag"]), ok)
    got = atoms_of(goals, ids)
    best = None
    for i, ref in enumerate(case["refs"]):
        r = score_ref(ref, got, ids, vocab)
        key = (len(r["errors"]) + len(r["unexpressible"]), len(r["errors"]))
        if best is None or key < best[0]:
            best = (key, i, r)
    _, i, r = best
    errs = errors + r["errors"] + (["vocab_gap"] if r["unexpressible"] else [])
    if case["category"] == "ambiguous" and errs:
        errs.append("unflagged_ambiguity")
    clean = not (errors or r["errors"])
    out = _done(base, clean and not r["unexpressible"], clean, errs, r["exact"] and not errors)
    out.update(ref=i, missing=r["missing"], extra=r["extra"], forbidden=r["forbidden"], unexpressible=r["unexpressible"])
    return out


def _done(base, ok, ok_expr, errors, exact):
    errs = sorted(set(errors), key=ERROR_PRIORITY.index)
    return {**base, "pass": bool(ok), "pass_expressible": bool(ok_expr), "exact": bool(exact), "errors": errs,
            "primary": errs[0] if errs else None}


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt_rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*k/n:.0f}% [{100*lo:.0f}, {100*hi:.0f}]" if n else "0/0"


def sign_test(b: int, c: int) -> float:
    """Exact two-sided binomial (McNemar) p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def aggregate(rows: list[dict], key=lambda r: "all", field: str = "pass") -> dict:
    """rows: [{..., "score": score_trial(...)}] -> {group: (k, n)}"""
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        g = key(r)
        acc[g][0] += bool(r["score"][field])
        acc[g][1] += 1
    return {g: tuple(v) for g, v in acc.items()}


def paired(rows_x: list[dict], rows_y: list[dict], field: str = "pass") -> dict:
    """Pairs by (case id, repeat). Returns counts and the sign-test p-value."""
    y = {(r["case"], r["rep"]): r for r in rows_y}
    both = x_only = y_only = neither = 0
    for r in rows_x:
        s = y.get((r["case"], r["rep"]))
        if s is None:
            continue
        a, b = bool(r["score"][field]), bool(s["score"][field])
        both += a and b
        x_only += a and not b
        y_only += b and not a
        neither += not a and not b
    return {"both": both, "x_only": x_only, "y_only": y_only, "neither": neither,
            "n": both + x_only + y_only + neither, "p": sign_test(x_only, y_only)}


def error_counts(rows: list[dict], primary_only: bool = True) -> Counter:
    c = Counter()
    for r in rows:
        s = r["score"]
        if primary_only:
            if s["primary"]:
                c[s["primary"]] += 1
        else:
            c.update(s["errors"])
    return c
