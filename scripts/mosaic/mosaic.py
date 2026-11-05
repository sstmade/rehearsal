"""Video photomosaic of robot rehearsals: every cell is an independently playing rehearsal tile stream; together
they form a target animation. Continuous exponential zoom between one full-screen tile and the full mosaic.

    mosaic.py --target _work/target_arm.npy --shot zoomout --name mosaic_zoomout_arm [--hud] [--half]
    shots: zoomout (lead, pull back, hold), zoomin (hold, push in, end on the HQ tile), hold (full mosaic)

Target = any .npy (N,h,w,3) / video / image (16:9). Inputs from pack.py (pool*, sig.npy, hq.npy) and hi-res
re-renders of the cells that get big on screen (tiles.render_scene, automatically).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiles  # noqa: E402

WORK, TILES = tiles.WORK, tiles.TILES
OUTDIR = WORK.parent
FF = str(tiles.ROOT / ".venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1")
GW, GH = 120, 120                           # 16:9 cells (16 x 9 px at 1080p): tiles are never distorted
RV = GH / GW                                # rows visible per column visible (the grid fills the 16:9 frame)
FPS, NF = 30, tiles.NF
OFFS = (0, 30, 60, 90)
PIN = (60, 60)                              # (col, row): the cell that holds the HQ clip
GAP = np.array([6, 8, 12], np.float32)
HI = (640, 1280)


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


# ------------------------------------------------------------------------------------------------ shot curves
def curve(shot, lead=0.6, tz=7.0, hold=4.0):
    pc = np.array([PIN[0] + 0.5, PIN[1] + 0.5])
    fc = np.array([GW / 2, GH / 2])
    out = []
    if shot == "zoomout":
        F = int(round((lead + tz + hold) * FPS))
        us = [smooth((f / FPS - lead) / tz) for f in range(F)]
    elif shot == "zoomin":
        F = int(round((hold + tz + lead) * FPS))
        us = [1 - smooth((f / FPS - hold) / tz) for f in range(F)]
    elif shot == "converge":
        # tiles shrink and multiply from ~50 on screen (V0 across) until the grid is exactly the frame; then hold
        F = int(round((tz + hold) * FPS))
        V0 = CONV_V0
        for f in range(F):
            u = smooth(f / FPS / tz)
            V = math.exp(math.log(V0) + u * (math.log(GW) - math.log(V0)))
            out.append((V, fc[0], fc[1], u))
        return out
    else:
        F = int(round(hold * FPS))
        us = [1.0] * F
    for u in us:
        V = math.exp(u * math.log(GW))
        c = pc * (1 - u) + fc * u
        out.append((V, c[0], c[1], u))
    return out


CONV_V0 = 7.1                               # ~50 tiles on screen at the start


def tindex(args, f, n):
    if args.head is not None:
        return int(min(max(f - args.head, 0) + args.vstart, n - 1))
    return int(args.t0 + f * args.speed) % n


def resolve_field(args, f):
    """(GH, GW) per-cell resolve progress for the converge shot (0 = mosaic tile, 1 = the true target pixels):
    a left-to-right diagonal sweep with a little per-cell jitter."""
    if args.resolve_at is None:
        return None
    t = (f - args.resolve_at * FPS) / FPS
    if t <= 0:
        return None
    yy, xx = np.mgrid[:GH, :GW].astype(np.float32)
    jit = np.random.default_rng(5).random((GH, GW)).astype(np.float32)
    delay = (0.8 * xx / GW + 0.2 * yy / GH) * args.resolve_span + 0.18 * jit * args.resolve_span
    r = np.clip((t - delay) / args.resolve_cell, 0, 1)
    return r * r * (3 - 2 * r)


def grade_amt(V):
    k = min(max(math.log(max(V, 1) / 6) / math.log(GW / 6), 0), 1)
    return 0.10 + 0.50 * k ** 1.3


# ------------------------------------------------------------------------------------------------ target
def load_target(path):
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p, mmap_mode="r")
    if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        im = np.asarray(Image.open(p).convert("RGB").resize((512, 288), Image.LANCZOS))
        return im[None]
    raw = subprocess.run([FF, "-v", "error", "-i", str(p), "-vf", "scale=512:288", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 288, 512, 3)


def target_q(tgt, f, t0=0, speed=1.0, view=None):
    """target frame -> (GH*2, GW*2, 3) float32 quadrant colours.
    view=(V, x0g, y0g): screen-space target (the big picture always fills the frame, whatever the zoom): each cell
    takes the target colours at the place it currently occupies on screen."""
    k = tindex(G["args"], f, len(tgt)) if "args" in G else int(t0 + f * speed) % len(tgt)
    fr = np.asarray(tgt[k])
    if view is None or abs(view[0] - GW) < 1e-6 and abs(view[1]) < 1e-6:
        return cv2.resize(fr, (GW * 2, GH * 2), interpolation=cv2.INTER_AREA).astype(np.float32)
    V, x0g, y0g = view
    vh = V * RV
    w2, h2 = max(2, int(round(2 * V))), max(2, int(round(2 * vh)))
    small = cv2.resize(fr, (w2, h2), interpolation=cv2.INTER_AREA).astype(np.float32)
    ax, ay = w2 / (2 * V), h2 / (2 * vh)
    M = np.array([[ax, 0, (0.25 - x0g) * w2 / V - 0.5], [0, ay, (0.25 - y0g) * h2 / vh - 0.5]], np.float32)
    return cv2.warpAffine(small, M, (GW * 2, GH * 2), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def view_of(c):
    V, cx, cy = c[0], c[1], c[2]
    return (V, cx - V / 2, cy - V * RV / 2)


def q2sig(q):
    """(GH*2, GW*2, 3) -> (GH*GW, 12) in the pool's (2,2,3) order"""
    return q.reshape(GH, 2, GW, 2, 3).transpose(0, 2, 1, 3, 4).reshape(GH * GW, 12)


# ------------------------------------------------------------------------------------------------ assignment
def assign(cv, tgt, args):
    sig = np.load(WORK / "sig.npy")                              # (N, NF, 12)
    N = sig.shape[0]
    K = len(OFFS)
    Ns = N * K
    tile_of = np.arange(Ns) // K
    off_of = np.asarray(OFFS)[np.arange(Ns) % K]
    rng = np.random.default_rng(11)
    ncell = GW * GH
    lo = np.percentile(sig, args.pct_lo, axis=(0, 1))
    hi = np.percentile(sig, args.pct_hi, axis=(0, 1))
    pen = rng.random((ncell, Ns), dtype=np.float32) < args.exclude      # random per-cell exclusions -> variety
    pin = PIN[1] * GW + PIN[0] if args.pin else None
    specs = json.loads((WORK / "pool.json").read_text())
    opening = {}
    if True:   # every shot: a diverse opening wall (no repeats while cells are big)
        # opening wall: every cell that is on screen while <= ~300 tiles are visible gets a DIFFERENT tile; the
        # innermost ~50 get one table each (shipped run, hero camera); then colourful dark-studio views; no two
        # neighbours from the same table. (Cells stay sticky while > sticky_px, so this holds until ~1,300 tiles.)
        mean = sig.mean(1).reshape(N, 4, 3).mean(1)                        # (N, 3) mean colour of each tile
        colour = mean.max(1) - mean.min(1)
        score = 2.0 * colour + 0.5 * mean.mean(1)
        tight = {i for i, c in enumerate(tiles.CAMSET) if c[3] or c[2] < 0.9}
        hero = [i for i, sp in enumerate(specs) if sp["rollout"] == 0 and sp["cam"] == 0 and sp["style"] == "dark"]
        rng.shuffle(hero)
        rest = [i for i in range(N) if i not in set(hero)]
        pri = lambda i: (score[i] + (25 if specs[i]["style"] == "dark" else 0)
                         + (15 if specs[i]["cam"] in tight else 0) + (10 if specs[i]["rollout"] == 0 else 0))
        rest.sort(key=pri, reverse=True)
        order = hero + rest
        cells = [(j, i) for j in range(GH) for i in range(GW)
                 if abs(i + 0.5 - GW / 2) < 9.5 and abs(j + 0.5 - GH / 2) < 9.5]        # ~360 cells
        cells.sort(key=lambda c: (c[1] + 0.5 - GW / 2) ** 2 + ((c[0] + 0.5 - GH / 2) * 9 / 16) ** 2)
        used, scene_at = set(), {}
        for (j, i) in cells:
            nb = {scene_at.get((j + dj, i + di)) for dj in (-1, 0, 1) for di in (-1, 0, 1)}
            t = next((t for t in order if t not in used and specs[t]["scene"] not in nb),
                     next(t for t in order if t not in used))
            used.add(t)
            scene_at[(j, i)] = specs[t]["scene"]
            opening[j * GW + i] = t * K + int(rng.integers(K))
        # the next ring (on screen until cells get small enough to swap, ~1,300 cells): still all different tiles,
        # each the best unused match for the target at the moment swapping starts
        fs = next((f for f, c in enumerate(cv) if 1920 / c[0] < args.sticky_px), len(cv) - 1)
        Vs = cv[fs][0]
        Ts = q2sig(target_q(tgt, fs, args.t0, args.speed, view_of(cv[fs]))) / 255.0
        Ts = lo + Ts * (hi - lo)
        msig = sig.mean(1)
        ring = [(j, i) for j in range(GH) for i in range(GW) if (j * GW + i) not in opening
                and abs(i + 0.5 - GW / 2) < Vs / 2 + 1 and abs(j + 0.5 - GH / 2) < Vs * RV / 2 + 1]
        free = np.ones(N, bool)
        free[[o // K for o in opening.values()]] = False
        for n in rng.permutation(len(ring)):
            j, i = ring[n]
            if not free.any():
                break
            c = ((msig - Ts[j * GW + i]) ** 2).sum(1)
            c[~free] = np.inf
            t = int(c.argmin())
            free[t] = False
            opening[j * GW + i] = t * K + int(rng.integers(K))
    cur = np.zeros(ncell, np.int64)
    dwell = np.zeros(ncell, np.int64)
    A = np.zeros((len(cv), ncell), np.int32)
    for f, (V, *_rest) in enumerate(cv):
        cellpx = 1920 / V
        S = sig[tile_of, (off_of + f) % NF]                      # (Ns, 12)
        T = q2sig(target_q(tgt, f, args.t0, args.speed, view_of(cv[f]) if args.screen else None)) / 255.0
        T = lo + T * (hi - lo)                                   # match in the tiles' own colour range
        S2 = (S * S).sum(1)
        # usage penalty: a tile already used by many cells (previous frame) costs more -> no wall of one dark tile
        cnt = np.bincount(cur // K, minlength=N) if f > 0 else np.zeros(N)
        usage = (args.usage * cnt[tile_of]).astype(np.float32)
        best = np.empty(ncell, np.int64)
        bc = np.empty(ncell, np.float32)
        cc = np.empty(ncell, np.float32)
        for c0 in range(0, ncell, 2304):
            Tc = T[c0:c0 + 2304]
            cost = (Tc * Tc).sum(1)[:, None] + S2[None] - 2 * Tc @ S.T + usage[None]
            cost[pen[c0:c0 + 2304]] = 1e12
            b = cost.argmin(1)
            best[c0:c0 + 2304] = b
            bc[c0:c0 + 2304] = cost[np.arange(len(b)), b]
            cc[c0:c0 + 2304] = cost[np.arange(len(b)), cur[c0:c0 + 2304]]
        if f == 0:
            cur = best.copy()
            for c, sid in opening.items():
                cur[c] = sid
        else:
            dwell += 1
            if cellpx < args.sticky_px:
                margin = args.margin * (1 + 3 * max(0.0, cellpx - 15) / max(args.sticky_px - 15, 1))
                gain = np.sqrt(np.maximum(cc, 0)) - np.sqrt(np.maximum(bc, 0))
                sw = (gain > margin) & (dwell >= args.dwell)
                cur[sw] = best[sw]
                dwell[sw] = 0
        A[f] = cur
        if pin is not None:
            A[f, pin] = -1
        if f % 60 == 0:
            print(f"assign {f}/{len(cv)}  swaps so far ok", flush=True)
    return A


# ------------------------------------------------------------------------------------------------ hi-res tiers
def ensure_hi(cv, A, specs):
    """Re-render (at 640 / 1280 px) every tile stream that becomes bigger than 128 / 640 px on screen."""
    need = {640: set(), 1280: set()}
    for f, (V, cx, cy, _) in enumerate(cv):
        cellpx = 1920 / V
        if cellpx <= 128:
            continue
        lvl = 1280 if cellpx > 640 else 640
        i0, i1 = max(0, int(cx - V / 2) - 1), min(GW, int(math.ceil(cx + V / 2)) + 1)
        h = V * RV
        j0, j1 = max(0, int(cy - h / 2) - 1), min(GH, int(math.ceil(cy + h / 2)) + 1)
        for s in A[f, j0:j1, i0:i1].ravel():
            if s >= 0:
                need[lvl].add(int(s) // len(OFFS))
    need[640] |= need[1280]
    index = {}
    for lvl, ts in need.items():
        groups = {}
        for t in sorted(ts):
            sp = specs[t]
            groups.setdefault((sp["scene"], sp["style"]), []).append(t)
        jobs = []
        for (nn, st), tl in groups.items():
            tag = f"{nn}_{st}_{lvl}"
            meta = TILES / f"{tag}.json"
            have = json.loads(meta.read_text()) if meta.exists() else []
            locs = [specs[t]["local"] for t in tl]
            have_l = [(h["rollout"], h["cam"]) for h in have]
            want_l = [(specs[t]["rollout"], specs[t]["cam"]) for t in tl]
            if not set(want_l) <= set(have_l):
                prev = json.loads((TILES / f"{nn}_{st}.json").read_text())
                allloc = sorted({i for i, p in enumerate(prev) if (p["rollout"], p["cam"]) in set(have_l)} | set(locs))
                jobs.append((nn, st, lvl, lvl * 9 // 16, allloc))
        print(f"hi {lvl}: {len(ts)} streams, {len(jobs)} scene renders", flush=True)
        if jobs:
            with Pool(4, maxtasksperchild=1) as p:
                for tag, msg in p.imap_unordered(tiles.render_scene, jobs):
                    print(" ", tag, msg[:300], flush=True)
        for t in ts:
            sp = specs[t]
            tag = f"{sp['scene']}_{sp['style']}_{lvl}"
            have = json.loads((TILES / f"{tag}.json").read_text())
            row = [(h["rollout"], h["cam"]) for h in have].index((sp["rollout"], sp["cam"]))
            index[(lvl, t)] = (str(TILES / f"{tag}.npy"), row)
    return index


# ------------------------------------------------------------------------------------------------ compositing
G = {}


def init(args, hi_index, A_path, cv, hud):
    G.update(args=args, hi=hi_index, A=np.load(A_path, mmap_mode="r"), cv=cv, hud=hud,
             tgt=load_target(args.target), sig=np.load(WORK / "sig.npy", mmap_mode="r"),
             hq=np.load(WORK / "hq.npy", mmap_mode="r"), mm={},
             full=np.load(args.full, mmap_mode="r") if args.full else None)
    for k in (16, 32, 64, 128):
        G[k] = np.load(WORK / f"pool{k}.npy", mmap_mode="r")
    G["pool"] = json.loads((WORK / "pool.json").read_text())


def hi_frame(lvl, t, fr):
    p, row = G["hi"][(lvl, t)]
    if p not in G["mm"]:
        G["mm"][p] = np.load(p, mmap_mode="r")
    return G["mm"][p][row, fr]


def render(f):
    args = G["args"]
    S = 0.5 if args.half else 1.0
    OW, OH = int(1920 * S), int(1080 * S)
    V, cx, cy, u = G["cv"][f]
    cellpx = OW / V                         # cell width on screen; height = cellpx * 9 / 16
    vh = V * RV
    chpx = OH / vh
    x0g, y0g = cx - V / 2, cy - vh / 2
    i0, i1 = max(0, math.floor(x0g)), min(GW, math.ceil(x0g + V))
    j0, j1 = max(0, math.floor(y0g)), min(GH, math.ceil(y0g + vh))
    nx, ny = i1 - i0, j1 - j0
    lw = next((k for k in (16, 32, 64, 128, 640, 1280) if k >= cellpx * 0.92), 1280)
    lh = lw * 9 // 16
    A = np.asarray(G["A"][f, j0:j1, i0:i1])
    K = len(OFFS)
    t = np.where(A >= 0, A // K, 0)
    fr = (np.asarray(OFFS)[np.where(A >= 0, A % K, 0)] + f) % NF
    hqk = min(f if args.shot != "zoomin" else f, len(G["hq"]) - 1)
    if lw <= 128:
        atlas = np.asarray(G[lw][t.ravel(), fr.ravel()]).reshape(ny, nx, lh, lw, 3)
    else:
        atlas = np.zeros((ny, nx, lh, lw, 3), np.uint8)
        for jj in range(ny):
            for ii in range(nx):
                if A[jj, ii] < 0:
                    continue
                tt, ff = int(t[jj, ii]), int(fr[jj, ii])
                key = (lw, tt) if (lw, tt) in G["hi"] else ((640, tt) if (640, tt) in G["hi"] else None)
                src = hi_frame(*key, ff) if key else G[128][tt, ff]
                atlas[jj, ii] = src if src.shape[1] == lw else cv2.resize(np.asarray(src), (lw, lh),
                                                                           interpolation=cv2.INTER_LINEAR)
    pj, pi = PIN[1] - j0, PIN[0] - i0
    hq_small = None
    R = resolve_field(args, f)
    if R is not None and R.min() >= 1 and G["full"] is not None:      # fully resolved: the true target frame
        return np.asarray(G["full"][tindex(args, f, len(G["full"]))]).copy()
    if args.pin and 0 <= pj < ny and 0 <= pi < nx:
        hqf = np.asarray(G["hq"][hqk])
        atlas[pj, pi] = cv2.resize(hqf, (lw, lh), interpolation=cv2.INTER_AREA)
        hq_small = cv2.resize(hqf, (2, 2), interpolation=cv2.INTER_AREA).astype(np.float32)
    atlas = atlas.transpose(0, 2, 1, 3, 4).reshape(ny * lh, nx * lw, 3)
    # colour grade toward the target (2x2 per cell, bilinear)
    a = grade_amt(V) * args.grade
    gT = 0.0
    if args.grade_time:     # converge: tint toward the target from the first frame, rising smoothly to the resolve
        x = smooth(f / max(1.0, args.resolve_at * FPS)) if args.resolve_at else 1.0
        a = args.grade0 + (args.grade1 - args.grade0) * x
        gT = args.palette0 + (1 - args.palette0) * x
    if a > 0.005:
        tq = target_q(G["tgt"], f, args.t0, args.speed,
                      (V, x0g, y0g) if args.screen else None)[j0 * 2:j1 * 2, i0 * 2:i1 * 2]
        tq = np.clip((tq - args.black) / (255 - args.black), 0, 1) ** args.gamma * 255
        sq = np.asarray(G["sig"][t.ravel(), fr.ravel()]).reshape(ny, nx, 2, 2, 3)
        if hq_small is not None:
            sq[pj, pi] = hq_small
        sq = sq.transpose(0, 2, 1, 3, 4).reshape(ny * 2, nx * 2, 3)
        if R is not None:   # resolving cells: tint -> 100 %
            ac = a + (1 - a) * R[j0:j1, i0:i1]
            D = (np.repeat(np.repeat(ac, 2, 0), 2, 1)[..., None] * (tq - sq)).astype(np.float32)
        else:
            D = (a * (tq - sq)).astype(np.float32)
    else:
        D = None
    # pre-shrink so the final warp only resamples by <= 1.3x
    kx = lw
    if lw > cellpx * 1.3:
        c2 = max(2, int(math.ceil(cellpx * 1.15)))
        atlas = cv2.resize(atlas, (nx * c2, ny * c2 * 9 // 16 if False else int(round(ny * c2 * 9 / 16))),
                           interpolation=cv2.INTER_AREA)
        kx = c2
    atlas = atlas.astype(np.float32)
    if D is not None:
        atlas += cv2.resize(D, (atlas.shape[1], atlas.shape[0]), interpolation=cv2.INTER_LINEAR)
    ky = atlas.shape[0] / ny
    sx, sy = cellpx / kx, chpx / ky
    M = np.array([[sx, 0, (i0 - x0g) * cellpx + 0.5 * sx - 0.5], [0, sy, (j0 - y0g) * chpx + 0.5 * sy - 0.5]],
                 np.float32)
    img = cv2.warpAffine(atlas, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=tuple(float(v) for v in GAP))
    if gT > 0:   # global palette transfer toward the target frame (mean / std per channel)
        tf = cv2.resize(np.asarray(G["tgt"][tindex(args, f, len(G["tgt"]))]), (64, 36),
                        interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1, 3)
        sm = cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA).reshape(-1, 3)
        mt, st = tf.mean(0), tf.std(0) + 1
        mm, sd = sm.mean(0), sm.std(0) + 1
        img = img + gT * ((img - mm) * np.clip(st / sd, 0.6, 1.6) + mt - img)
    # grid gaps (antialiased, separable)
    g = float(np.clip(args.gap * cellpx, 1.0 * S, 7.0 * S))

    def keep(n, o0, lim, cp):
        gpos = o0 + (np.arange(n) + 0.5) / cp
        fr_ = gpos - np.floor(gpos)
        dpx = np.minimum(fr_, 1 - fr_) * cp
        k = np.clip(dpx - g / 2 + 0.5, 0, 1)
        k[(gpos < 0) | (gpos >= lim)] = 0
        return k.astype(np.float32)
    m = keep(OH, y0g, GH, chpx)[:, None] * keep(OW, x0g, GW, cellpx)[None, :]
    if R is not None:   # (the converge shot resolves at V == GW: cells are exactly aligned with the frame)
        Rp = cv2.resize(R, (OW, OH), interpolation=cv2.INTER_NEAREST)
        m = 1 - (1 - m) * (1 - Rp)
    img = img * m[..., None] + GAP * (1 - m[..., None])
    if R is not None and G["full"] is not None:   # tile content fades to the true target pixels, per cell
        R2 = np.clip((Rp - 0.35) / 0.65, 0, 1)
        R2 = (R2 * R2 * (3 - 2 * R2))[..., None]
        fullf = np.asarray(G["full"][tindex(args, f, len(G["full"]))])
        if fullf.shape[1] != OW:
            fullf = cv2.resize(fullf, (OW, OH), interpolation=cv2.INTER_AREA)
        img = img * (1 - R2) + fullf.astype(np.float32) * R2
    img = np.clip(img, 0, 255).astype(np.uint8)
    if G["hud"]:
        img = hud(img, f, V, x0g, y0g, cellpx, S)
    return img


# ------------------------------------------------------------------------------------------------ HUD glint
HUD_CELLS = [(23, 20), (97, 15), (40, 57), (109, 52)]


def _mono(sz):
    from PIL import ImageFont
    for p in ("/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def hud(img, f, V, x0g, y0g, cellpx, S):
    args = G["args"]
    t = f / FPS
    t_on = args.hud_at
    if t < t_on or V < GW * 0.9:
        return img
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    font = _mono(int(15 * S))
    K = len(OFFS)
    for n, (ci, cj) in enumerate(HUD_CELLS):
        a = smooth((t - t_on - 0.35 * n) / 0.35)
        if a <= 0:
            continue
        s = int(G["A"][f, cj, ci])
        tt = s // K if s >= 0 else 0
        sp = G["pool"][tt]
        num = s + 1 if s >= 0 else 1
        if sp["rollout"] == 0 and sp.get("success"):
            status = "passed · sent"
        else:
            status = "not sent"
        label = f"REHEARSAL #{num:,} · {status.upper()}"
        x = (ci - x0g) * cellpx
        y = (cj - y0g) * cellpx
        col = (150, 215, 255, int(255 * a))
        dr.rectangle([x - 1, y - 1, x + cellpx, y + cellpx], outline=col, width=max(1, int(2 * S)))
        up = cj > GH / 2
        lx = x + cellpx + 26 * S if ci < GW * 0.75 else x - 26 * S
        ly = y - 30 * S if not up else y + cellpx + 30 * S
        dr.line([(x + cellpx if lx > x else x, y if not up else y + cellpx), (lx, ly)], fill=col, width=1)
        bb = dr.textbbox((0, 0), label, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        bx = lx if lx > x else lx - tw - 16 * S
        dr.rectangle([bx, ly - th / 2 - 7 * S, bx + tw + 16 * S, ly + th / 2 + 7 * S], fill=(8, 12, 18, int(200 * a)),
                     outline=(150, 215, 255, int(140 * a)))
        dr.text((bx + 8 * S - bb[0], ly - th / 2 - bb[1]), label, font=font, fill=(225, 238, 250, int(255 * a)))
    im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    return np.asarray(im)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(WORK / "target_arm.npy"))
    ap.add_argument("--shot", default="zoomout", choices=("zoomout", "zoomin", "hold", "converge"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--hud", action="store_true", help="(off by default: no on-screen text)")
    ap.add_argument("--grade-time", type=int, default=None, help="time-based tint (default: on for converge)")
    ap.add_argument("--grade0", type=float, default=0.35)
    ap.add_argument("--grade1", type=float, default=0.85)
    ap.add_argument("--palette0", type=float, default=0.55)
    ap.add_argument("--hud-at", type=float, default=9.4)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--t0", type=float, default=0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--lead", type=float, default=0.6)
    ap.add_argument("--tz", type=float, default=7.0)
    ap.add_argument("--hold", type=float, default=4.0)
    ap.add_argument("--grade", type=float, default=1.0)
    ap.add_argument("--black", type=float, default=8.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--gap", type=float, default=0.07)
    ap.add_argument("--exclude", type=float, default=0.75)
    ap.add_argument("--usage", type=float, default=600.0, help="cost per existing use of a tile (diversity)")
    ap.add_argument("--pct-lo", type=float, default=3.0, help="target range mapped onto these tile percentiles")
    ap.add_argument("--pct-hi", type=float, default=93.0)
    ap.add_argument("--margin", type=float, default=6.0)
    ap.add_argument("--dwell", type=int, default=12)
    ap.add_argument("--sticky-px", type=float, default=48)
    ap.add_argument("--stills", default="")
    ap.add_argument("--pin", type=int, default=1, help="centre cell plays the HQ clip (zoom shots)")
    ap.add_argument("--full", default="", help="full-res target frames .npy (N,1080,1920,3) for the resolve")
    ap.add_argument("--head", type=int, default=None, help="target: hold frame --vstart until this output frame")
    ap.add_argument("--vstart", type=int, default=0)
    ap.add_argument("--resolve-at", type=float, default=None, help="s: start of the resolve sweep (converge)")
    ap.add_argument("--resolve-span", type=float, default=1.0)
    ap.add_argument("--resolve-cell", type=float, default=0.7)
    ap.add_argument("--tail", type=float, default=0.0, help="s of resolved target video appended (converge)")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--screen", type=int, default=None, help="screen-space target (default: on for converge)")
    ap.add_argument("--frames", default="", help="only render these frames (comma list) as PNGs")
    a = ap.parse_args()
    if a.grade_time is None:
        a.grade_time = int(a.shot == "converge")
    if a.screen is None:
        a.screen = int(a.shot == "converge")
    cv = curve(a.shot, a.lead, a.tz, a.hold)
    cv += [cv[-1]] * int(round(a.tail * FPS))
    od = Path(a.outdir) if a.outdir else OUTDIR
    od.mkdir(parents=True, exist_ok=True)
    G["args"] = a
    tgt = load_target(a.target)
    specs = json.loads((WORK / "pool.json").read_text())
    Apath = WORK / f"assign_{a.name}.npy"
    A = assign(cv, tgt, a).reshape(len(cv), GH, GW)
    np.save(Apath, A)
    hi = ensure_hi(cv, A, specs) if not a.half or True else {}
    sd = od / "stills" / a.name
    sd.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    if a.frames:
        fl = [int(x) for x in a.frames.split(",")]
        with Pool(a.jobs, initializer=init, initargs=(a, hi, Apath, cv, a.hud)) as p:
            for k, img in zip(fl, p.imap(render, fl)):
                Image.fromarray(img).save(sd / f"f{k:04d}.png")
        return
    S = 0.5 if a.half else 1.0
    OW, OH = int(1920 * S), int(1080 * S)
    out = od / (a.name + ("_half" if a.half else "") + ".mp4")
    enc = subprocess.Popen([FF, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    keys = {int(round(x * FPS)) for x in np.arange(0, len(cv) / FPS, 1.0)} | {len(cv) - 1}
    with Pool(a.jobs, initializer=init, initargs=(a, hi, Apath, cv, a.hud)) as p:
        for k, img in enumerate(p.imap(render, range(len(cv)), chunksize=2)):
            enc.stdin.write(np.ascontiguousarray(img).tobytes())
            if k in keys:
                Image.fromarray(img).save(sd / f"f{k:04d}.png")
    enc.stdin.close()
    enc.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
