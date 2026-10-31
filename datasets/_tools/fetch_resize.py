"""Download originals (curl) into a raw dir, then resize to long side <= 2000 px and save as JPEG q90
without EXIF/GPS metadata. Usage: python -I fetch_resize.py SEL.json RAW_DIR OUT_DIR
SEL.json: {key: {"url": ..., "stem": ...}}"""
import json, os, subprocess, sys
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 60_000_000
sel, raw, out = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
os.makedirs(raw, exist_ok=True); os.makedirs(out, exist_ok=True)
for k, r in sel.items():
    src = os.path.join(raw, r["stem"] + os.path.splitext(r["url"].split("?")[0])[1].lower())
    if not os.path.exists(src):
        subprocess.run(["curl", "-sfL", "--max-time", "180", "--retry", "5", "--retry-delay", "15", "--retry-all-errors", "-A", "RehearsalDataScout/0.1 (hackathon dataset curation; non-commercial)", "-o", src, r["url"]], check=True)
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    im.thumbnail((2000, 2000), Image.LANCZOS)
    dst = os.path.join(out, r["stem"] + ".jpg")
    im.save(dst, "JPEG", quality=90, optimize=True)
    print(k, dst, im.size, os.path.getsize(dst) // 1024, "KB")
