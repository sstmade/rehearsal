"""Contact sheet + index.json for the HQ wall.  sheet_index.py [--exclude id:reason ...]"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent / "footage/tables_sim/wall_hq"
TW = ROOT.parent / "footage/tables_sim/wall_primitive/_twin"
ex = dict(a.split(":", 1) for a in sys.argv[1:])
ids = sorted(p.stem[3:] for p in TW.glob("t6_*.json"))
cols, tw, th = 7, 320, 180
S = Image.new("RGB", (cols * tw, ((len(ids) + cols - 1) // cols) * th), (20, 20, 20))
d = ImageDraw.Draw(S)
idx = {"_about": "HQ physics twins of the AI eval tables (t6_*, AI-generated test photos only; never the user's desk). "
       "Collision shapes coloured from the photo (dominant colour of each object's projected shape, slightly muted), "
       "table top/frame colour from the photo, onion-scene studio (dark void, glowing grid, warm key + cool rims), Cycles + OIDN. "
       "<id>.png = 1280x720 still from the photo camera with the photo's VERTICAL FOV (scale the photo to 720 px high and centre it to overlay). "
       "<id>.mp4 = 2.5 s (75 f @30) 20 deg smoothstep orbit starting at the photo camera. Arm hidden (stands between camera and table).",
       "tables": {}, "excluded": ex}
for i, n in enumerate(ids):
    p = OUT / f"{n}.png"
    if p.exists():
        S.paste(Image.open(p).convert("RGB").resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
    d.text(((i % cols) * tw + 6, (i // cols) * th + 4), n + (" X" if n in ex else ""), fill=(255, 220, 0))
    if n in ex or not p.exists():
        continue
    J = json.loads((TW / f"t6_{n}.json").read_text())
    mp4 = OUT / f"{n}.mp4"
    idx["tables"][n] = {"scene": f"t6_{n}", "photo": str(ROOT / J["photo"]), "still": str(p),
                        "orbit": str(mp4) if mp4.exists() else None}
S.save(OUT / "_sheet.jpg", quality=85)
(OUT / "index.json").write_text(json.dumps(idx, indent=1, ensure_ascii=False))
print(len(idx["tables"]), "tables;", sum(1 for v in idx["tables"].values() if v["orbit"]), "orbits")
