"""Sequence 2 "Many futures": composite of the fut_render.py frames + HUD.

.venv/bin/python scripts/vfx/fut_compose.py --nn 50 [--stills 40,150,...]
All numbers come from footage/vfx/_data/NN_futures_summary.json (the logged re-run of the shipped search) and
out/tables_v6/NN.json (instruction, goals): futures = physics rehearsals the planner ran (stats.sims), failures and
their reasons = recorded outcomes of those simulations, SAFE = the plan the search returned (identical to the
shipped plan; checked in the README).
"""
import argparse, json, os, sys, re, collections
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(__file__))
import hud as Hd
from hud import W, H, ACC, WHITE, GREY, RED, GREEN, smooth, ease_out

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT.parent
FF = str(ROOT / ".venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1")
ap = argparse.ArgumentParser()
ap.add_argument("--nn", default="50")
ap.add_argument("--stills", default=None)
A = ap.parse_args()
NN = A.nn
OUTD = PROJ / "footage/vfx"
BL = Path(os.environ.get("FUT_BL", OUTD / f"_work/fut{NN}_blender"))
SUM = json.loads((OUTD / f"_data/{NN}_futures_summary.json").read_text())
META = json.loads((BL / "chips.json").read_text())
COMP = json.loads((ROOT / f"out/tables_v6/{NN}.json").read_text())
INSTR = COMP["instruction"]
NGOALS = len(COMP["intent"]["goals"])
LAB = {o["id"]: o["label"] for o in SUM["objects"]}
NF = 352
F0, FE = META["F_START"], META["F_END_SOL"]
N_FUT = SUM["stats"]["sims"]                        # physics rehearsals counted by the search
sims = [s for s in SUM["sims"] if s["n_act"] > 0]
assert len(sims) == N_FUT, (len(sims), N_FUT)
SOLS = [s for s in sims if s["success"] and s["actions"] == SUM["plan"]["actions"]]
N_SAFE = len(SOLS)
N_FAIL = sum(1 for s in sims if not s["success"])
N_PART = sum(1 for s in sims if s["success"] and s["actions"] != SUM["plan"]["actions"])


def short(oid):
    base = re.sub(r"_\d+$", "", oid)
    return base.split("_")[-1]


def reason(c):
    dist = c.get("disturbed") or []
    if dist:
        d = max(dist, key=lambda x: x.get("moved_cm", 0))
        if d.get("tilt_deg", 0) > 30:
            return f"{short(d['obj'])} tipped {d['tilt_deg']:.0f}°", "tip"
        return f"{short(d['obj'])} pushed {d.get('moved_cm', 0):.1f} cm", "push"
    if c.get("error"):
        e = c["error"]["error"] if isinstance(c["error"], dict) else str(c["error"])
        if "protective" in e:
            return "collision · protective stop", "stop"
        if "IK" in e:
            return "out of reach", "ik"
        return e[:28], "err"
    if c.get("failed"):
        g = c["failed"][0]
        m = re.match(r"(\w+)\((\w+)", g)
        return (f"{short(m.group(2))} not {m.group(1)}" if m else g[:28]), "goal"
    return "failed", "goal"


CH = []
for k, c in META["chips"].items():
    txt, cat = reason(c)
    CH.append({"i": int(k), "f": c["f_fail"], "xy": c["xy"], "txt": txt, "cat": cat})
CH.sort(key=lambda c: c["f"])
# show at most one new chip every 5 frames (the rest are counted in the tally)
shown, last = [], -99
for c in CH:
    if c["f"] - last >= 5:
        shown.append(c); last = c["f"]
for j, c in enumerate(shown):
    c["slot"] = j % 4


def frame(f):
    p = BL / f"b_{f:04d}.png"
    if not p.exists():
        cand = sorted(BL.glob("b_*.png"))
        p = min(cand, key=lambda q: abs(int(q.stem[2:]) - f))
    return Hd.load(p)


TY0, TY1 = 16, 70


def overlay(f, L):
    a_in = smooth(f / 12)
    # ---------------------------------------------------------------- instruction box (bottom centre)
    a_box = a_in * (1 - smooth((f - (F0 + 30)) / 14))
    if a_box > 0.01:
        typed = np.clip((f - TY0) / (TY1 - TY0), 0, 1)
        n = int(round(len(INSTR) * typed))
        s = "›  " + INSTR[:n] + ("▍" if (f // 8) % 2 == 0 or typed < 1 else " ")
        x0 = W / 2 - 330
        y = H - 120
        L.d.rounded_rectangle([x0 * 2, (y - 26) * 2, (W - x0) * 2, (y + 26) * 2], radius=6 * 2,
                              fill=(9, 11, 15, int(185 * a_box)), outline=(255, 255, 255, int(60 * a_box)), width=2)
        L.d.rectangle([x0 * 2, (y - 18) * 2, (x0 + 2) * 2, (y + 18) * 2], fill=Hd.rgba(ACC, a_box))
        L.text((x0 + 22, y), s, "mid", WHITE, a_box)
        L.text((x0, y - 44), "INSTRUCTION", "huds", GREY, a_box)
        if f >= TY1 + 4:
            L.chip(W - x0, y - 44, [("intent", "small", GREY), (f"{NGOALS} goals", "lab", WHITE), ("· Muse Spark", "small", GREY)],
                   a=a_box * smooth((f - TY1 - 4) / 6), anchor="r", h=22)
    # ---------------------------------------------------------------- status line
    if f < F0 - 6:
        L.status(f, "REHEARSAL  ·  INSTRUCTION", f"table {NN}  ·  digital twin ready", a=a_in)
    elif f < FE - 6:
        nf = sum(1 for c in CH if c["f"] <= f)
        L.status(f, f"REHEARSING  ·  {N_FUT} FUTURES IN PHYSICS", f"FAILED {nf:3d}   ·   time x{META['speed']:.1f}", dot=(255, 120, 90) if nf else ACC)
    # ---------------------------------------------------------------- failure chips near the failure point
    for c in shown:
        t = f - c["f"]
        if t < 0 or t > 30:
            continue
        a = min(1, t / 2) * (1 - smooth((t - 18) / 12))
        x, y = c["xy"][0], c["xy"][1]
        dx = 34 + 0 * c["slot"]; dy = -60 - c["slot"] * 30 - t * 0.6
        L.d.line([(x * 2, y * 2), ((x + dx - 6) * 2, (y + dy) * 2)], fill=Hd.rgba(RED, a * 0.8), width=2)
        L.d.ellipse([(x - 3) * 2, (y - 3) * 2, (x + 3) * 2, (y + 3) * 2], fill=Hd.rgba(RED, a))
        L.chip(x + dx, y + dy, [("✕", "lab", RED), (c["txt"], "lab", WHITE), (f"#{c['i']:03d}", "small", GREY)], a=a, acc=RED, h=22)
    # ---------------------------------------------------------------- tally (right)
    if F0 + 10 <= f < FE + 40:
        a = smooth((f - F0 - 10) / 10) * (1 - smooth((f - FE - 20) / 14))
        cats = collections.Counter(re.sub(r"\s*[\d.]+\s*(cm|°)?", "", c["txt"]).strip() for c in CH if c["f"] <= f)
        x, y = W - 60, 150
        L.text((x, y - 30), "FAILURE MODES", "huds", GREY, a * 0.9, anchor="rm")
        for j, (k, v) in enumerate(cats.most_common(5)):
            L.chip(x, y + j * 30, [(k, "lab", WHITE), (f"×{v}", "lab", (255, 140, 120))], a=a, acc=RED, anchor="r", h=22)
        npart = sum(1 for s in sims if s["success"] and s["actions"] != SUM["plan"]["actions"] and F0 + s["sim_time"] / META["speed"] * 30 <= f)
        L.chip(x, y + min(5, len(cats)) * 30 + 12, [("partial plans kept", "lab", WHITE), (f"×{npart}", "lab", (190, 210, 255))], a=a, acc=ACC, anchor="r", h=22)
    # ---------------------------------------------------------------- verdict
    if f >= FE - 6:
        a = smooth((f - FE + 6) / 10)
        xy = META["sol_end_xy"]
        L.brackets((xy[0] - 46, xy[1] - 46, xy[0] + 46, xy[1] + 46), GREEN, a * (1 - smooth((f - FE - 30) / 10)), w=2,
                   grow=10 * (1 - ease_out((f - FE + 6) / 14)))
        y = H - 110
        parts = [("REHEARSED", "bigb", WHITE), ("·", "big", GREY), (f"{N_FUT} FUTURES", "big", WHITE), ("·", "big", GREY),
                 (f"{N_SAFE} SAFE", "bigb", GREEN), ("·", "big", GREY), ("SENT", "bigb", GREEN)]
        L.chip(W / 2, y, parts, a=a, acc=GREEN, anchor="c", h=48, pad=18, typed=min(1, (f - FE + 6) / 16))
        L.text((W / 2, y + 42), f"{N_FAIL} failed in simulation  ·  {N_PART} partial plans extended  ·  1 complete plan passed every goal",
               "huds", (170, 180, 195), a * smooth((f - FE - 8) / 10), anchor="mm")
        L.status(f, "PLAN SENT TO ROBOT", f"{len(SUM['plan']['actions'])} actions  ·  goals {NGOALS}/{NGOALS} in rehearsal", dot=GREEN, a=a)


def render(f):
    img = frame(f)
    fade = smooth(f / 10) * (1 - smooth((f - (NF - 10)) / 10))
    clean = Hd.finish(img * fade, f, strength=0.4, thresh=0.82)
    L = Hd.Layer()
    overlay(f, L)
    return clean, np.clip(L.over(clean), 0, 1)


if A.stills:
    sd = OUTD / "keyframes"; sd.mkdir(exist_ok=True, parents=True)
    for f in [int(x) for x in A.stills.split(",")]:
        c, h = render(f)
        cv2.imwrite(str(sd / f"fut_{f:04d}.jpg"), (h[..., ::-1] * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("stills done", N_FUT, N_SAFE, N_FAIL, N_PART)
    sys.exit(0)
e1 = Hd.Encoder(str(OUTD / "many_futures.mp4"), FF)
e2 = Hd.Encoder(str(OUTD / "many_futures_clean.mp4"), FF)
for f in range(NF):
    c, h = render(f)
    e1.write(h); e2.write(c)
e1.close(); e2.close()
print("done", N_FUT, N_SAFE, N_FAIL, N_PART)
