"""Pack the rendered tile streams into one pool + a pyramid + colour signatures; decode the HQ centre clip.

    pack.py            -> _work/pool128.npy (N,NF,72,128,3), pool64/32/16.npy, sig.npy (N,NF,12), pool.json
    pack.py --hq       -> _work/hq.npy  (frames of hero/exec_dark/06.mp4 from the task start, 1920x1080)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiles  # noqa: E402

WORK, TILES = tiles.WORK, tiles.TILES
FF = str(tiles.ROOT / ".venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1")
HQ_SRC = tiles.DARK / "06.mp4"
HQ_T0, HQ_N = 7.3, 480


def down(a, k):
    """area downscale by k of (..., h, w, 3) uint8"""
    *lead, h, w, c = a.shape
    b = a.reshape(*lead, h // k, k, w // k, k, c).astype(np.float32).mean((-4, -2))
    return np.clip(b + 0.5, 0, 255).astype(np.uint8)


def pack():
    files = sorted(p for p in TILES.glob("*.json") if p.stem.count("_") == 1 and p.with_suffix(".npy").exists())
    specs, srcs = [], []
    for p in files:
        sp = json.loads(p.read_text())
        for i, s in enumerate(sp):
            s["local"] = i
            specs.append(s)
            srcs.append((p.with_suffix(".npy"), i))
    N, NF = len(specs), tiles.NF
    print(N, "tile streams from", len(files), "scene/style files")
    P = {k: np.lib.format.open_memmap(WORK / f"pool{k}.npy", "w+", np.uint8, (N, NF, k * 9 // 16, k, 3))
         for k in (128, 64, 32, 16)}
    sig = np.zeros((N, NF, 12), np.float32)
    cache = {}
    for n, (f, i) in enumerate(srcs):
        if f not in cache:
            cache = {f: np.load(f, mmap_mode="r")}
        a = np.asarray(cache[f][i])
        P[128][n] = a
        P[64][n] = down(a, 2)
        P[32][n] = down(a, 4)
        P[16][n] = down(a, 8)
        sig[n] = a.reshape(NF, 2, 36, 2, 64, 3).astype(np.float32).mean((2, 4)).reshape(NF, 12)
    for v in P.values():
        v.flush()
    np.save(WORK / "sig.npy", sig)
    (WORK / "pool.json").write_text(json.dumps(specs))


def hq():
    raw = subprocess.run([FF, "-v", "error", "-ss", str(HQ_T0), "-i", str(HQ_SRC), "-frames:v", str(HQ_N),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, 1080, 1920, 3)
    out = np.lib.format.open_memmap(WORK / "hq.npy", "w+", np.uint8, a.shape)
    out[:] = a
    out.flush()
    print("hq frames", a.shape)


def decode(src, dst):
    raw = subprocess.run([FF, "-v", "error", "-i", str(src), "-vf", "scale=1920:1080", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, 1080, 1920, 3)
    out = np.lib.format.open_memmap(dst, "w+", np.uint8, a.shape)
    out[:] = a
    out.flush()
    print("decoded", a.shape, dst)


if __name__ == "__main__":
    if "--decode" in sys.argv:          # pack.py --decode src.mp4 dst.npy  (full-res target frames for the resolve)
        i = sys.argv.index("--decode")
        decode(sys.argv[i + 1], sys.argv[i + 2])
    else:
        hq() if "--hq" in sys.argv else pack()
