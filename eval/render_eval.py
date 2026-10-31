"""Render EVAL.md + charts from eval/results/*.json.

    .venv/bin/python eval/render_eval.py

Uses the newest result file per tag (eval_<stamp>_<tag>.json), the newest ap_*.json and pc_*.json.
Tags listed in SKIP_TAGS (smoke tests, superseded probes) are ignored.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVAL = Path(__file__).resolve().parent
REPO = EVAL.parent
RES = EVAL / "results"
FIG = EVAL / "figures"
SKIP_TAGS = {"smoke", "muse_probe"}
# reference palette (dataviz skill, light mode), fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PERCEPTION_MEDIAN_CM, PERCEPTION_P90_CM = 0.54, 1.30   # main:docs/perception.md, box_fit, exact corners


def newest_by_tag():
    files = {}
    for p in sorted(RES.glob("eval_*.json")):
        m = re.match(r"eval_\d{8}_\d{6}(?:_(.*))?\.json", p.name)
        tag = (m.group(1) if m else None) or "untagged"
        if tag in SKIP_TAGS:
            continue
        files[tag] = p
    return files


def newest(prefix):
    fs = sorted(RES.glob(f"{prefix}_*.json"))
    return json.loads(fs[-1].read_text()) if fs else None, (fs[-1].name if fs else None)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def frac(k, n):
    return f"{k}/{n}" if n else "–"


def perception_docs() -> str | None:
    try:
        txt = subprocess.run(["git", "-C", str(REPO), "show", "main:docs/perception.md"],
                             capture_output=True, text=True, check=True).stdout
    except Exception:  # noqa: BLE001
        p = REPO / "docs/perception.md"
        txt = p.read_text() if p.exists() else ""
    return txt or None


# ------------------------------------------------------------------ section A
def section_a(L):
    files = newest_by_tag()
    L += ["## A. Closed loop: plan → simulate → read the log → revise", ""]
    if not files:
        L += ["_No run_eval results yet._", ""]
        return {}
    L += ["Each row aggregates n independent (task, repeat) runs. *First try* and *≤2 revisions* are judged "
          "by the agent's own compiled goals (what the product stops on); *ref* re-judges the same simulation "
          "against hand-written reference goals the agent never sees. *Robust* = mean stress-test success of the "
          "final plan (σ = 1.2 cm per axis, friction/mass randomised, seeds 0–39). Conditions only differ in "
          "what the revise step sees: **full_log** (product history: failed predicates with numbers, contact "
          "events, IK/reach diagnostics, protective stops), **outcome_only** (names of failed predicates + an "
          "aborted flag), **none** (\"attempt failed\", plan only). Attempt 1 is shared across conditions (paired).",
          ""]
    summary = {}
    for tag, path in sorted(files.items()):
        r = json.loads(path.read_text())
        L += [f"### `{tag}` — {r['code']}", "",
              f"{path.name} · {r['created']} · wall {r['wall_s']} s · Meta spend (eval.* calls this run) "
              f"${r.get('eval_spend_this_run', 0):.4f}", "",
              "| scene | instruction | agent | feedback | n | first try | ≤2 rev | ≤2 rev (ref) | mean attempts | robust | failure types seen | $ |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in r["aggregate"]:
            ft = ", ".join(f"{k}×{v}" for k, v in sorted(a["failure_types"].items())) or "–"
            rob = f"{a['robustness_mean']*100:.0f}%" if a["robustness_mean"] is not None else "–"
            ins = a["instruction"] if len(a["instruction"]) < 28 else a["instruction"][:26] + "…"
            L.append(f"| {a['scene']} | {ins} | {a['agent']} | {a['condition']} | {a['n']} | "
                     f"{frac(a['first_try'], a['n'])} | {frac(a['within_2_rev'], a['n'])} | "
                     f"{frac(a['within_2_rev_ref'], a['n'])} | {a['mean_attempts']} | {rob} | {ft} | "
                     f"{a['cost_usd']:.4f} |")
            summary[(tag, a["scene"], a["instruction"], a["agent"], a["condition"])] = a
        unsup = sorted({(c["scene"], c["instruction"], c["agent"]) for c in r["cells"] if c.get("status") == "unsupported"})
        if unsup:
            L.append("")
            L.append("Not applicable: " + "; ".join(f"{s} / {i} / {a}" for s, i, a in unsup)
                     + " (OfflineAgent only understands the desk_01 Chinese vocabulary).")
        errs = [c for c in r["cells"] if str(c.get("status", "ok")) not in ("ok", "unsupported")]
        if errs:
            L.append(f"\nErrors: {len(errs)} cells — " + "; ".join(sorted({str(c['status'])[:120] for c in errs})))
        L.append("")
    return summary


def chart_a(summary):
    rows = [(k, v) for k, v in summary.items() if v["needed_revision"] > 0]
    if not rows:
        return None
    groups = defaultdict(dict)
    for (tag, scene, ins, agent, cond), a in rows:
        groups[f"{scene}\n{agent} ({tag})"][cond] = a
    conds = ["full_log", "outcome_only", "none"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=150)
    labels = list(groups)
    w = 0.26
    for j, c in enumerate(conds):
        xs, ys, ns = [], [], []
        for i, g in enumerate(labels):
            a = groups[g].get(c)
            if not a:
                continue
            # among runs whose first try failed: success after revisions (reference goals)
            k = a["within_2_rev_ref"] - (a["first_try_ref"])
            n = a["needed_revision"]
            xs.append(i + (j - 1) * w)
            ys.append(100 * k / n if n else 0)
            ns.append(f"{k}/{n}")
        bars = ax.bar(xs, ys, w - 0.03, color=SERIES[j], label=c)
        for b, t in zip(bars, ns):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, t, ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(labels)), labels, fontsize=7)
    ax.set_ylim(0, 112)
    ax.set_ylabel("recovered within 2 revisions (%)", fontsize=8, color=INK2)
    ax.set_title("A. Recovery after a failed first attempt, by feedback given to revise", fontsize=9, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    out = FIG / "a_feedback_ablation.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ section B
def section_b(L):
    pc, name = newest("pc")
    L += ["## B. Why 2 cm matters: success vs perception error", ""]
    if not pc:
        L += ["_No perception_curve results yet._", ""]
        return None
    L += [f"{name} · {pc['code']} · held-out seeds starting at 1000 · common random numbers across σ "
          "(each seed's error direction is fixed and scaled). Cells: successes/n.", ""]
    sig = [c["sigma_cm"] for c in pc["plans"][0]["curve"]]
    L += ["| scene | plan | " + " | ".join(f"σ={s:g} cm" for s in sig) + " |",
          "|---|---|" + "---|" * len(sig)]
    for p in pc["plans"]:
        L.append(f"| {p['scene']} | {p['agent']} `{p['plan_key'][:6]}` | "
                 + " | ".join(frac(c["successes"], c["n"]) for c in p["curve"]) + " |")
    L += ["", "Plan labels: `ref_naive`/`ref_fixed` are the robotics engineer's hand-verified failing/fixed plans "
          "from the scene's `reference` block; `offline`/`muse` are the agents' final plans and `*_initial` their "
          "first attempts.", ""]
    scenes = sorted({p["scene"] for p in pc["plans"]})
    fig, axes = plt.subplots(1, len(scenes), figsize=(3.0 * len(scenes), 2.9), dpi=150, sharey=True, squeeze=False)
    for ax, sc in zip(axes[0], scenes):
        ax.axvspan(PERCEPTION_MEDIAN_CM, PERCEPTION_P90_CM, color="#d9d8d3", alpha=0.6, lw=0)
        ax.text((PERCEPTION_MEDIAN_CM + PERCEPTION_P90_CM) / 2, 4, "measured\nmed–P90", ha="center",
                fontsize=6, color=INK2)
        ax.axvline(1.2, color=INK2, lw=0.8, ls=":")
        ps = [p for p in pc["plans"] if p["scene"] == sc]
        for j, p in enumerate(ps):
            xs = [c["sigma_cm"] for c in p["curve"]]
            ys = [100 * c["successes"] / c["n"] for c in p["curve"]]
            lo = [100 * c["ci95"][0] for c in p["curve"]]
            hi = [100 * c["ci95"][1] for c in p["curve"]]
            col = SERIES[j % len(SERIES)]
            ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=3, label=p["agent"])
        ax.set_title(sc, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("perception error σ per axis (cm)", fontsize=7, color=INK2)
        ax.set_ylim(-3, 103)
        style(ax)
        ax.legend(fontsize=6, frameon=False, loc="lower left")
    axes[0][0].set_ylabel("simulated success (%)", fontsize=8, color=INK2)
    fig.suptitle("B. Plan success vs injected position error (dotted: σ=1.2 cm used by the stress test)",
                 fontsize=8.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    out = FIG / "b_perception_curve.png"
    fig.savefig(out)
    plt.close(fig)
    L += [f"![perception curve](eval/figures/{out.name})", "",
          f"Grey band: perception placement error measured on synthetic renders (box_fit, exact A4 corners: "
          f"median {PERCEPTION_MEDIAN_CM} cm, P90 {PERCEPTION_P90_CM} cm, radial). Note the band is a *radial* "
          "error drawn on a *per-axis σ* axis; for a 2-D Gaussian the median radial error is 1.18σ, so the "
          "band sits to the right of the matching σ (conservative).", ""]
    return out


# ------------------------------------------------------------------ section C
def section_c(L):
    ap, name = newest("ap")
    L += ["## C. Active perception: which object to look at again", ""]
    if not ap:
        L += ["_No active_perception results yet._", ""]
        return None
    L += [f"{name} · {ap['code']} · σ {ap['sigma_cm']:.1f} → {ap['refined_sigma_cm']:.1f} cm for the re-observed object · "
          f"select on seeds {ap['selection_seeds'][0]}–{ap['selection_seeds'][1]}, **score on held-out seeds "
          f"{ap['held_out_seeds'][0]}–{ap['held_out_seeds'][1]}**. +fixed/−broken = paired discordant seeds vs. no "
          "second look; p = exact two-sided sign test.", "",
          "| scene | plan | no look | VoI pick | random (expected) | heuristic: most in failed predicates | heuristic: nearest carry path | oracle (best on held-out) | look at all |",
          "|---|---|---|---|---|---|---|---|---|"]

    def cell(v, n):
        s = f"{v['successes']:.1f}/{n}" if isinstance(v["successes"], float) else f"{v['successes']}/{n}"
        who = f" **{v['look_at']}**" if v.get("look_at") and v["look_at"] not in ("uniform",) and not str(v["look_at"]).startswith("all") else ""
        st = f" (+{v['fixed']}/−{v['broken']}, p={v['sign_test_p']:.2g})" if "fixed" in v and (v["fixed"] or v["broken"]) else ""
        return s + who + st
    for p in ap["plans"]:
        po = p["policies"]
        n = po["none"]["n"]
        L.append(f"| {p['scene']} | {p['agent']} | {po['none']['successes']}/{n} | {cell(po['voi'], n)} | "
                 f"{cell(po['random'], n)} | {cell(po['fail_pred'], n)} | {cell(po['near_path'], n)} | "
                 f"{cell(po['oracle'], n)} | {cell(po['all'], n)} |")
    L += ["", "Selection-set view (what the product shows at run time, seeds 0–39; optimistic because the pick is "
          "scored on the same draws it was chosen on):", ""]
    for p in ap["plans"]:
        s = p["selection"]
        top = sorted(s["by_object"].items(), key=lambda kv: -kv[1])[:3]
        L.append(f"- {p['scene']}: base {s['base']}/{s['n']}; best " + ", ".join(f"{k} {v}/{s['n']}" for k, v in top))
    L.append("")
    pols = [("none", "no look"), ("voi", "VoI"), ("random", "random"), ("fail_pred", "fail-pred"),
            ("near_path", "near-path"), ("all", "all")]
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=150)
    w = 0.13
    for j, (k, lab) in enumerate(pols):
        xs = [i + (j - 2.5) * w for i in range(len(ap["plans"]))]
        ys = [100 * p["policies"][k]["successes"] / p["policies"][k]["n"] for p in ap["plans"]]
        ax.bar(xs, ys, w - 0.02, color=SERIES[j], label=lab)
    ax.set_xticks(range(len(ap["plans"])), [f"{p['scene']}\n({p['agent']})" for p in ap["plans"]], fontsize=7)
    lo = min(100 * p["policies"]["none"]["successes"] / p["policies"]["none"]["n"] for p in ap["plans"])
    ax.set_ylim(max(0, lo - 15), 101)
    ax.set_ylabel("held-out success (%)", fontsize=8, color=INK2)
    ax.set_title("C. One extra look: VoI vs baselines (held-out seeds)", fontsize=9, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=6.5, frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    out = FIG / "c_active_perception.png"
    FIG.mkdir(exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    L += [f"![active perception](eval/figures/{out.name})", ""]
    return out


def main():
    L = ["# Rehearsal — evaluation", "",
         "_Generated by `eval/render_eval.py` from `eval/results/`. Re-run everything: `eval/run_all.sh` "
         "(offline parts $0; Muse Spark calls are served from the shared response cache when unchanged)._", "",
         "<!-- HEADLINE -->", ""]
    summary = section_a(L)
    fa = chart_a(summary)
    if fa:
        L += [f"![feedback ablation](eval/figures/{fa.name})", ""]
    section_b(L)
    section_c(L)
    L += ["## D. Perception accuracy (from the perception branch, now in main)", ""]
    doc = perception_docs()
    if doc and "box_fit" in doc:
        L += ["From `docs/perception.md` (synthetic renders with known ground truth: 12 images × 4 camera poses, "
              "76 objects; real photos are expected to be worse):", "",
              "| placement method | exact A4 corners (median / P90 cm) | auto-detected corners |", "|---|---|---|",
              "| `bottom_center` (original MVP) | 4.95 / 11.81 | 5.21 / 11.88 |",
              "| **`box_fit`** (default) | **0.54 / 1.30** | **0.79 / 2.80** |", "",
              "These are the numbers drawn as the grey band in B. They are *not* re-measured by eval.", ""]
    else:
        L += ["**TODO:** perception accuracy numbers not found in docs/perception.md.", ""]
    L += [(EVAL / "caveats.md").read_text() if (EVAL / "caveats.md").exists() else ""]
    md = "\n".join(L)
    head = (EVAL / "headline.md")
    md = md.replace("<!-- HEADLINE -->", head.read_text().strip() if head.exists() else "")
    (REPO / "EVAL.md").write_text(md, encoding="utf-8")
    print(f"wrote {REPO / 'EVAL.md'}")


if __name__ == "__main__":
    main()
