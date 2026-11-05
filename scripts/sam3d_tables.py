"""SAM 3D Objects look-alike meshes for the 50-table batch's hero tables (scenes/t6_NN.json), visual only.

Thin driver around scripts/sam3d_assets.py: one scene at a time, one object at a time (each reconstruction peaks at
~16 GB on the 1448 x 1086 photos), only the objects that matter visually (containers and task objects; flat things
and cloth are skipped: cloth is drawn as simulated cloth by photo_skin.py).

The slow part (the Gaussian splat, ~4-6 min/object) depends only on the photo and the SAM 3.1 mask
(perception.mask_px); it is cached in .scratch/sam3d/t6_NN/outputs/<obj>/gaussians.ply.  The fast part (mesh,
decimation, choosing the orientation against the photo through the scene camera, ~10-30 s/object) depends on the
calibration and the twin poses, so after the scenes are rebuilt re-run it with --realign (no reconstruction):

    SAM3D_WEIGHTS=weights/sam-3d-objects-mlx-8bit sam3d_env/bin/python scripts/sam3d_tables.py --realign

Output: assets/sam3d/t6_NN/<obj>.glb + <obj>.json (timing, peak memory, photo IoU of the chosen orientation).
usage: sam3d_env/bin/python scripts/sam3d_tables.py [NN ...] [--realign] [--only id,id]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# priority order; per scene the objects worth a mesh (containers + task objects)
HERO = {
    "43": ["basket_1", "pet_brush_1", "pet_brush_2"],
    "03": ["laundry_basket_1"],
    "01": ["mug_1", "mug_2", "pen_holder_1"],
    "06": ["mixing_bowl_1", "onion_1", "onion_2", "onion_3", "spice_box_1"],
    "09": ["bowl_1", "mug_1", "flour_bag_1", "rolling_pin_1", "egg_1", "egg_2", "egg_3"],
    "46": ["tray_1", "bowl_1", "bowl_2", "bowl_3", "bowl_4", "plate_1", "pouch_1"],
    "28": ["bowl_1", "bowl_2", "bowl_3", "bowl_4"],
    "27": ["bucket_1", "spray_bottle_1", "sponge_1"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*", help="NN ids (default: all hero tables, priority order)")
    ap.add_argument("--realign", action="store_true", help="redo mesh + orientation from the cached splats only")
    ap.add_argument("--only", default="", help="comma-separated object ids (overrides the hero list)")
    ap.add_argument("--faces", default="18000")
    a = ap.parse_args()
    env = dict(os.environ)
    env.setdefault("SAM3D_WEIGHTS", str(ROOT / "weights" / "sam-3d-objects-mlx-8bit"))
    for nn in a.scenes or list(HERO):
        scene = ROOT / "scenes" / f"t6_{nn}.json"
        ids = [o["id"] for o in json.loads(scene.read_text())["objects"]]
        want = [i for i in (a.only.split(",") if a.only else HERO.get(nn, [])) if i in ids]
        missing = set(a.only.split(",") if a.only else HERO.get(nn, [])) - set(ids)
        if missing:
            print(f"[warn] t6_{nn}: no such objects {sorted(missing)}", flush=True)
        if a.realign:   # never reconstruct here: only objects whose splat exists
            want = [i for i in want if (ROOT / ".scratch" / "sam3d" / f"t6_{nn}" / "outputs" / i / "gaussians.ply").exists()]
        if not want:
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / "sam3d_assets.py"), str(scene), "--only", ",".join(want),
               "--faces", a.faces] + (["--realign"] if a.realign else [])
        print(f"== t6_{nn}: {','.join(want)}", flush=True)
        subprocess.run(cmd, cwd=ROOT, env=env, check=False)


if __name__ == "__main__":
    main()
