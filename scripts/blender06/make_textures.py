"""Helper textures for the Blender t6_06 re-render (run with the project venv, no Blender needed).

  weave_height.png  tileable 1024^2 height map: plain cotton weave (over/under threads, 16 x 16 threads per tile)
                    with fibre noise and slub (thread-thickness variation), for the towel bump
  towel_decal_4x.png  the towel's photo decal (MuJoCo cloth texture) upscaled 4x (Lanczos + light unsharp mask)
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

EXP = Path(__file__).resolve().parents[3] / "footage/tables_sim/hero/exec_blender/_export"
TEX = EXP / "06_traj_tex"


def weave(n=1024, threads=16, seed=7):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n * threads              # thread coordinates
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - ix, y - iy
    # thread cross-section profiles (rounded), per-thread thickness variation (slub)
    slub_w = 0.80 + 0.2 * rng.random(threads)
    slub_h = 0.80 + 0.2 * rng.random(threads)
    warp = np.sqrt(np.clip(1 - ((fx - 0.5) / (0.5 * slub_w[ix % threads])) ** 2, 0, 1))   # vertical threads
    weft = np.sqrt(np.clip(1 - ((fy - 0.5) / (0.5 * slub_h[iy % threads])) ** 2, 0, 1))   # horizontal threads
    over = ((ix + iy) % 2 == 0)                          # plain weave: warp on top where (i+j) even
    # undulation along the thread: rises in the middle of its 'over' span
    bump_warp = 0.55 + 0.45 * np.sin(np.pi * fy)
    bump_weft = 0.55 + 0.45 * np.sin(np.pi * fx)
    h = np.where(over, np.maximum(warp * bump_warp, 0.6 * weft), np.maximum(weft * bump_weft, 0.6 * warp))
    # fibre noise (fine twist streaks along each thread)
    noise = rng.random((n // 4, n // 4))
    noise = np.asarray(Image.fromarray((noise * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC)) / 255.0
    twist = 0.5 + 0.5 * np.sin(2 * np.pi * (np.where(over, fy, fx) * 6 + np.where(over, fx, fy) * 2))
    h = h * (0.85 + 0.1 * twist + 0.05 * noise)
    h = (h - h.min()) / (h.max() - h.min())
    return (h * 255).astype(np.uint8)


def main():
    Image.fromarray(weave()).save(EXP / "weave_height.png")
    src = next(TEX.glob("*cloth0_t.png"))
    im = Image.open(src).convert("RGB")
    up = im.resize((im.width * 4, im.height * 4), Image.LANCZOS).filter(ImageFilter.UnsharpMask(2, 60, 2))
    up.save(EXP / "towel_decal_4x.png")
    print("ok", EXP)


if __name__ == "__main__":
    main()
