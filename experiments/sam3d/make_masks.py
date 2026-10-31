"""Rasterize the SAM 3.1 mask polygons stored in a scene JSON into per-object mask PNGs for SAM 3D.
usage: python make_masks.py scenes/real_ycbv59.json OUT_DIR   (run from the repo root; no API calls)"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw

scene, out = json.load(open(sys.argv[1])), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
img = Image.open(scene["photo"]["file"]).convert("RGB")
img.save(out / "image.png")
for o in scene["objects"]:
    poly = (o.get("perception") or {}).get("mask_px")
    if poly:
        m = Image.new("L", img.size, 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in poly], fill=255)
        m.save(out / f"mask_{o['id']}.png")
