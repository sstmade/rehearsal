"""JPEG siblings of the photos and table textures the web app shows, so the repo / Space ship ~25 MB of JPEGs
instead of ~170 MB of PNGs. Same pixel size, so SAM masks (photo pixels) and camera calibration still line up.

    .venv/bin/python scripts/build_web_photos.py        # writes <file>.web.jpg next to each source; $0

rehearsal/app_api.py serves <file>.web.jpg when it exists, else the source file. The user's own desk is never
touched (it is not in scenes/).
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def main():
    n = tot = 0
    for p in sorted((ROOT / "scenes").glob("*.json")):
        if p.stem.startswith("photo_") or p.name.endswith(".models.json"):
            continue
        spec = json.loads(p.read_text())
        srcs = [(spec.get("photo") or {}).get("file"), (spec.get("visual") or {}).get("table_texture")]
        for f in srcs:
            if not (isinstance(f, str) and f.startswith("photos/")):
                continue
            src = ROOT / f
            dst = src.with_suffix(".web.jpg")
            if not src.is_file() or src.suffix.lower() in (".jpg", ".jpeg"):
                continue
            if not dst.is_file() or dst.stat().st_mtime < src.stat().st_mtime:
                im = Image.open(src)
                if im.mode in ("RGBA", "LA", "P"):
                    im = im.convert("RGBA")
                    bg = Image.new("RGB", im.size, (0, 0, 0)); bg.paste(im, mask=im.split()[-1]); im = bg
                im.convert("RGB").save(dst, quality=86, optimize=True, progressive=True)
                n += 1
            tot += dst.stat().st_size
    print(f"{n} written; web JPEGs total {tot / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
