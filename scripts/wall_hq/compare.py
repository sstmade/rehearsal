"""photo | render pairs grid.  compare.py OUT.jpg RENDER_DIR id ..."""
import json, sys
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
out, rd, ids = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
TW = ROOT.parent / "footage/tables_sim/wall_primitive/_twin"
G = Image.new("RGB", (1280, 360 * len(ids)))
for i, n in enumerate(ids):
    J = json.loads((TW / f"t6_{n}.json").read_text())
    ph = Image.open(ROOT / J["photo"]).convert("RGB"); ph = ph.resize((int(ph.width * 720 / ph.height), 720))
    fr = Image.new("RGB", (1280, 720)); fr.paste(ph, ((1280 - ph.width) // 2, 0))
    G.paste(fr.resize((640, 360)), (0, 360 * i)); G.paste(Image.open(rd / f"{n}.png").convert("RGB").resize((640, 360)), (640, 360 * i))
G.save(out, quality=88)
