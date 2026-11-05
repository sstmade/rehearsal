"""Sequence 1 "Photo -> Digital": 2D composite of the Blender frames, the photo, SAM 3.1 masks and the HUD.

.venv/bin/python scripts/vfx/p2d_compose.py [--stills 40,100,...] [--no-encode]
Inputs (all real): the 06 photo, SAM 3.1 masks (out/tables_v6/masks_hi/t6_06.json, sam_conf_crop per object),
the scene labels (scenes/t6_06.json), Blender frames from p2d_render.py, and frame 98 of the shipped
exec_primitive/06.mp4, exec_dark/06.mp4 and exec_blender/frames/f_0098.png (all through the photo camera).
HUD numbers: point count from p2d_points.npz; SAM/Muse cost+latency are TYPICAL values from budget/ledger.jsonl
(no per-call record exists for table 06), labelled "typical".
Outputs footage/vfx/photo_to_digital.mp4 and photo_to_digital_clean.mp4 (no HUD / chips / labels).
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(__file__))
import hud as Hd
from hud import W, H, ACC, WHITE, GREY, smooth, ease_out

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT.parent
FF = str(ROOT / ".venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1")
OUTD = PROJ / "footage/vfx"
WORK = OUTD / "_work"
BL = WORK / "p2d_blender"
ap = argparse.ArgumentParser()
ap.add_argument("--stills", default=None)
ap.add_argument("--no-encode", action="store_true")
A = ap.parse_args()

NF = 528
# ------------------------------------------------------------------ data
sc = json.loads((ROOT / "scenes/t6_06.json").read_text())
MK = json.loads((ROOT / "out/tables_v6/masks_hi/t6_06.json").read_text())
labels = {o["id"]: o["label"] for o in sc["objects"]}
pw, ph = MK["width"], MK["height"]
k = W / pw
y0 = (ph * k - H) / 2
OBJ = []
for oid, poly in MK["masks"].items():
    p = np.array(poly, np.float32) * k
    p[:, 1] -= y0
    OBJ.append({"id": oid, "label": labels.get(oid, oid), "poly": p, "conf": MK["metrics"][oid]["sam_conf_crop"],
                "c": p.mean(0), "bb": (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())})
OBJ.sort(key=lambda o: o["c"][0])
for n, o in enumerate(OBJ, 1):
    o["n"] = n
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(o["poly"]).astype(np.int32)], 255)
    o["mask"] = cv2.GaussianBlur(m, (0, 0), 1.0).astype(np.float32)[..., None] / 255.0
NPTS = int(np.load(PROJ / "footage/vfx/_data/p2d_points.npz")["depth"].size)
NOBJ = len(OBJ)

photo = cv2.imread(str(ROOT / MK["photo"]))[..., ::-1]
photo = cv2.resize(photo, (W, int(round(ph * k))), interpolation=cv2.INTER_LANCZOS4)
photo = photo[int(round(y0)):int(round(y0)) + H].astype(np.float32) / 255.0


def grab98(mp4, out):
    if not out.exists():
        subprocess.run([FF, "-loglevel", "error", "-y", "-i", str(mp4), "-vf", "select=eq(n\\,98)", "-frames:v", "1", str(out)], check=True)
    return Hd.load(out)


HERO = PROJ / "footage/tables_sim/hero"
PRIM = grab98(HERO / "exec_primitive/06.mp4", WORK / "f98_primitive.png")
DARK = grab98(HERO / "exec_dark/06.mp4", WORK / "f98_dark.png")
CYC = Hd.load(HERO / "exec_blender/frames/f_0098.png")
WIRE = Hd.load(BL / "wire_still.png") if (BL / "wire_still.png").exists() else np.zeros((H, W, 3), np.float32)

# placement of the object name list (right column) and per-object chips (above the mask)
LIST_X, LIST_Y = W - 60, 140


def blframe(f):
    p = BL / f"b_{f:04d}.png"
    if not p.exists():
        cand = sorted(BL.glob("b_*.png"))
        p = min(cand, key=lambda q: abs(int(q.stem[2:]) - f))
    return Hd.load(p)


# ------------------------------------------------------------------ timeline
SCAN0, SCAN1 = 80, 136          # scan sweep (frame 80..136), left -> right
IGN0, IGN_DT = 142, 8           # mask ignition: object n starts at IGN0 + n*IGN_DT
EXT0 = 238                      # Blender points take over
WIPE = [(446, 470, PRIM), (472, 494, DARK), (496, 516, CYC)]


def scan_x(f):
    return -40 + (W + 80) * smooth((f - SCAN0) / (SCAN1 - SCAN0))


def sweep_line(img, x, a, col=(0.75, 0.87, 1.0)):
    """thin vertical light plane at x: white core + accent glow"""
    if a <= 0.01:
        return img
    xs = np.arange(W, dtype=np.float32)
    core = np.exp(-((xs - x) / 1.4) ** 2)
    halo = np.exp(-((xs - x) / 22.0) ** 2) * 0.35 + np.exp(-np.clip(x - xs, 0, None) / 160.0) * (xs < x) * 0.06
    prof = (core[None, :, None] * np.array([1, 1, 1]) + halo[None, :, None] * np.array(col)) * a
    return img + prof.astype(np.float32)


def base_frame(f):
    """clean picture (no HUD)"""
    if f <= 70:
        return blframe(f)
    if f < EXT0:
        img = photo.copy()
        # scan: behind the line the photo is "read": slightly cooler/darker, faint raster
        if SCAN0 <= f <= SCAN1 + 20:
            x = scan_x(min(f, SCAN1))
            behind = (np.arange(W)[None, :, None] < x).astype(np.float32)
            fade = 1 - smooth((f - SCAN1) / 20)
            cool = img * np.array([0.86, 0.92, 1.0], np.float32)
            img = img * (1 - behind * 0.5 * fade) + cool * behind * 0.5 * fade
            img = sweep_line(img, x, 1.0 - smooth((f - SCAN1 + 4) / 10))
        # masks: dim the photo, ignite each object
        dim = smooth((f - IGN0 + 6) / 30) * 0.66
        img = img * (1 - dim)
        lines = np.zeros((H, W, 3), np.float32)
        for o in OBJ:
            t = f - (IGN0 + (o["n"] - 1) * IGN_DT)
            if t < 0:
                continue
            flash = np.exp(-t / 6.0) * 0.6 + 0.07
            lift = dim / (1 - dim + 1e-6) * min(1, t / 6) * 0.85     # restore the object's own brightness
            img = img + o["mask"] * (img * lift + np.array([0.25, 0.55, 1.0], np.float32) * flash * 0.45)
            P = o["poly"]; n = len(P)
            kk = max(2, int(ease_out(t / 10) * n))
            pts = np.round(np.array([P[j % n] for j in range(kk + 1)]) * 4).astype(np.int32)
            cv2.polylines(lines, [pts], kk >= n, (0.55, 0.82, 1.0), 2, cv2.LINE_AA, shift=2)
        if f >= IGN0:
            img = img + lines * 0.9 + cv2.GaussianBlur(lines, (0, 0), 5) * 1.6
        # handover to Blender: masks/dim fade so the frame matches b_0238 (the photo)
        if f >= EXT0 - 10:
            u = smooth((f - (EXT0 - 10)) / 10)
            img = img * (1 - u) + photo * u
        return img
    if f <= 446:
        return blframe(f)
    # wipes at the photo camera: wire -> primitive -> dark -> cycles
    img = blframe(446)
    for a0, a1, nxt in WIPE:
        if f >= a0:
            x = -60 + (W + 120) * smooth((f - a0) / (a1 - a0))
            m = np.clip((x - np.arange(W, dtype=np.float32)) / 50.0, 0, 1)[None, :, None]
            img = img * (1 - m) + nxt * m
            if f <= a1:
                img = sweep_line(img, x, 0.9)
    # wire lines linger over the primitive stage, then go
    wl = 1 - smooth((f - 466) / 26)
    img = img + WIRE * 0.35 * wl * (f >= 446)
    return img


def overlay(f, L):
    """HUD/chips/labels for frame f (draws into Layer L)"""
    a_in = smooth(f / 14)
    a_out = 1 - smooth((f - (NF - 14)) / 12)
    if f < SCAN0 - 6:
        L.status(f, "REHEARSAL  ·  INPUT", f"1 PHOTO  ·  {pw}×{ph}  ·  table 06", a=a_in)
        return
    if f < IGN0 - 4:
        L.status(f, "MUSE SPARK  ·  reading scene", "muse-spark-1.3  ·  typical $0.0006 / call  ·  2.2k tok in", a=1.0)
    elif f < EXT0 + 6:
        L.status(f, f"SAM 3.1  ·  {NOBJ} OBJECTS  ·  ~550 ms / mask", "typical latency + $0.0025 / mask (ledger)",
                 a=1 - smooth((f - EXT0) / 6))
    # object list, typed as the scan line passes each object
    a_list = 1 - smooth((f - (EXT0 - 4)) / 10)
    if f >= SCAN0 and a_list > 0.01:
        L.text((LIST_X, LIST_Y - 30), "OBJECTS", "huds", GREY, a_list * 0.9, anchor="rm")
        for o in OBJ:
            tx = SCAN0 + (SCAN1 - SCAN0) * np.clip((o["c"][0] + 40) / (W + 80), 0, 1)
            # invert smooth() roughly: the line passes the object when scan_x == c.x
            ts = next((ff for ff in range(SCAN0, SCAN1 + 1) if scan_x(ff) >= o["c"][0]), SCAN1)
            t = f - ts
            if t < 0:
                continue
            y = LIST_Y + (o["n"] - 1) * 30
            ig = f >= IGN0 + (o["n"] - 1) * IGN_DT
            conf = f"{o['conf']:.2f}" if ig else ""
            L.chip(LIST_X, y, [(f"{o['n']:02d}", "small", GREY), (o["label"], "lab", WHITE)] + ([(conf, "small", (150, 200, 255))] if conf else []),
                   a=a_list * min(1, t / 4), typed=min(1, t / 10), anchor="r", acc=ACC if ig else (90, 100, 115))
    # per-object outlines + brackets during ignition
    for o in OBJ:
        t = f - (IGN0 + (o["n"] - 1) * IGN_DT)
        if t < 0:
            continue
        a = min(1, t / 3) * (1 - smooth((f - (EXT0 - 8)) / 12))
        if a <= 0.01:
            continue
        P = o["poly"]
        n = len(P)
        frac = ease_out(t / 10)
        kk = max(2, int(frac * n))
        pts = [tuple(P[j % n]) for j in range(kk + 1)]
        L.brackets(o["bb"], WHITE, a * 0.85, w=1, grow=6 + 10 * np.exp(-t / 5))
    # depth / assembly HUD
    if EXT0 <= f <= 446:
        a = smooth((f - EXT0 - 4) / 10) * (1 - smooth((f - 440) / 8))
        if f < 372:
            L.status(f, "DEPTH  ·  pixels -> 3D", f"{NPTS:,} POINTS  ·  depth from the twin geometry".replace(",", " "), a=a)
        else:
            L.status(f, "ASSEMBLING TWIN", f"{NOBJ} OBJECTS  ·  SAM 3D meshes + fitted primitives", a=a)
    if f > 446:
        stage = "PRIMITIVES  ·  physics bodies" if f < 472 else ("SIMULATION STUDIO" if f < 496 else "DIGITAL TWIN  ·  READY")
        a = smooth((f - 448) / 8) * a_out
        L.status(f, stage, f"{NOBJ} OBJECTS  ·  MuJoCo  ·  table 06", dot=(120, 230, 170) if f >= 496 else ACC, a=a)


def render(f, with_hud=True):
    img = base_frame(f)
    clean = Hd.finish(img.copy(), f, bloom=not (70 < f < EXT0), strength=(0.55 if f < EXT0 else 0.32) if f <= 446 else 0.18, thresh=(0.75 if f < EXT0 else 0.85) if f <= 446 else 0.92)
    w = smooth((f - 500) / 20)            # the last frames are the untouched Cycles frame (seamless cut to 06_cycles)
    if w > 0:
        clean = clean * (1 - w) + np.clip(img, 0, 1) * w
    if not with_hud:
        return clean, None
    L = Hd.Layer()
    overlay(f, L)
    return clean, np.clip(L.over(clean), 0, 1)


if A.stills:
    sd = OUTD / "keyframes"
    sd.mkdir(parents=True, exist_ok=True)
    for f in [int(x) for x in A.stills.split(",")]:
        c, h = render(f)
        cv2.imwrite(str(sd / f"p2d_{f:04d}.jpg"), (h[..., ::-1] * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("still", f)
    sys.exit(0)

e1 = Hd.Encoder(str(OUTD / "photo_to_digital.mp4"), FF)
e2 = Hd.Encoder(str(OUTD / "photo_to_digital_clean.mp4"), FF)
for f in range(NF):
    c, h = render(f)
    e1.write(h); e2.write(c)
    if f % 50 == 0:
        print("frame", f, flush=True)
e1.close(); e2.close()
print("done")
