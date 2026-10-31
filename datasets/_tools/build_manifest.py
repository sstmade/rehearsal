"""Build datasets/MANIFEST.json, datasets/SOURCES.md and datasets/contact_sheet.jpg.
Run from the worktree root:  python -I datasets/_tools/build_manifest.py <selB.json>
(selB.json = scratch file with the Commons metadata of the Tier B picks: title, page, license, artist, url.)
Tier B visual descriptions below were written by hand by the scout after looking at each photo."""
import glob, json, os, re, sys
from PIL import Image, ImageDraw, ImageFont

ROOT = "datasets"
# stem -> (scene_type, elevation_deg (90 = straight down), reference object or None, objects, notes)
B = {
 "b_dimsum_teapot_bowl_table": ("restaurant table (grey stone)", 45, None,
   ["brass teapot", "stainless steel bowl", "small white tea cup", "white plate with porcelain spoon", "chopsticks", "black card holder with receipt", "paper napkin"],
   "busy background (chairs); QR receipt blurred"),
 "b_enamel_cups_table": ("rustic wooden table", 35, None,
   ["4 white enamel mugs", "3 small orange enamel cups", "small glass jar", "folded cloth"], "warm light, shallow depth of field"),
 "b_preschool_drink_bottles_table": ("worn wooden table (preschool)", 50, None,
   ["~8 plastic drink bottles with straps", "red bottle cap", "pink/red/green plastic baskets", "cloth bags"], "dense clutter, saturated colours, low light"),
 "b_messy_office_desk": ("cluttered office desk", 25, "yellow legal pads (8.5 x 11.75 in), partly occluded",
   ["lotion pump bottle", "small cream bottle", "radio", "headphones", "tape dispenser", "stapler", "pen cup", "paper cup", "pill bottle", "monitor", "legal pads", "papers"],
   "very cluttered, low camera angle, flash"),
 "b_desk_ruler_mug_notebook": ("white desk", 40, "ruler with mm scale (~30 cm) lying flat",
   ["white mug", "spiral notebook", "pencil", "paper clip", "laptop (partial)", "ruler"], "bright, low contrast white-on-white; mug out of focus"),
 "b_desk_laptop_coffee_topdown": ("light wooden desk", 90, "square sticky note (likely 76 x 76 mm, unverified); 13-inch laptop",
   ["laptop", "espresso cup with coffee", "sticky-note pad", "2 pencils", "crumpled paper balls"], "flat lay"),
 "b_desk_camera_notebook_coffee_topdown": ("white desk", 90, "Apple wireless keyboard (~279 x 115 mm); smartphone",
   ["keyboard", "mouse (partial)", "film camera", "eyeglasses", "smartphone", "patterned notebook", "mug of coffee", "car keys"], "flat lay, clean background"),
 "b_notebook_coffee_topdown": ("wooden table", 90, None,
   ["open notebook", "pencil", "espresso cup with coffee"], "flat lay, few objects"),
 "b_brunch_bowl_fruit_topdown": ("weathered blue wooden boards", 90, None,
   ["large bowl (fruit/chia)", "glass of orange juice", "2 avocado halves", "orange", "2 bread slices", "2 grape bunches", "spoon"], "flat lay, textured background"),
 "b_waffle_bowls_topdown": ("table runner on dark table", 90, None,
   ["3 small ramekins (banana, strawberry, blueberry)", "4-compartment dish (chocolate chips)", "plate of waffles", "stacked plates with lid", "pour-over coffee maker", "wood slice"],
   "flat lay"),
 "b_honey_cake_bowl_jug_topdown": ("dark slate surface", 90, None,
   ["white jug (gravy boat) with sauce", "bowl of cereal", "spoon", "slice of honey cake with raspberry"], "flat lay, dark, high contrast"),
 "b_cutting_board_ingredients_topdown": ("kitchen counter (grey tiles)", 90, None,
   ["wooden cutting board", "avocado half", "2 eggs", "cherry tomatoes", "mushrooms", "knife", "spinach", "spring onions", "frying pan (partial)"], "flat lay; many small items"),
 "b_fruit_bowl_table": ("pine kitchen table", 25, None,
   ["ceramic fruit bowl", "bananas", "nectarines/peaches", "apples", "plum"], "single container with fruit; low angle"),
 "b_coffee_pastries_table": ("grey concrete-look table", 70, None,
   ["mug of coffee", "ramekin with chocolate dip and spoon", "plate with croissants", "~7 small cookies"], "portrait orientation"),
 "b_polka_mug_glasses_paper": ("wooden table", 55, "printed paper sheet (A4/Letter), partly out of frame",
   ["polka-dot espresso cup", "saucer/plate", "2 heart cookies", "eyeglasses", "paper sheet"], "dim, vignetting"),
 "b_christmas_table_plates_bottle": ("round dining table, white cloth", 35, "750 ml wine bottle (~30 cm tall)",
   ["wine bottle", "2 wine glasses", "candle in holder", "plate of rolls", "platter with roast", "bowl of salad", "2 bowls (dip, snacks)"], "clean, even light"),
 "b_cluttered_writer_table": ("writer's desk (wood)", 35, None,
   ["2 mugs", "pen holders", "stacks of books and papers"], "extreme clutter, portrait; foreground mugs are the manipulable part"),
}


def lic_url(r):
    return r.get("license_url") or ""


def main(sel_path):
    sel = {v["stem"]: v for v in json.load(open(sel_path)).values()}
    entries, src_lines = [], []
    # Tier A
    for d, label in [("tierA_hope", "HOPE"), ("tierA_ycbv", "YCB-Video")]:
        for jf in sorted(glob.glob(f"{ROOT}/{d}/*.json")):
            g = json.load(open(jf)); tp = g["table_plane"]
            entries.append({
                "path": f"{d}/{g['image']}", "gt_json": f"{d}/{os.path.basename(jf)}", "depth": f"{d}/{g['depth']}",
                "tier": "A", "license": g["license"].split(" (")[0], "license_detail": g["license"],
                "author": "NVIDIA (Tyree et al. 2022)" if label == "HOPE" else "UW RSE Lab (Xiang et al. 2018)",
                "source_url": g["source_url"], "source_member": g["source_member"],
                "has_reference_object": "checkerboard" if "000050" in jf else False,
                "has_ground_truth": "6-DoF object poses + depth + K (camera frame); table plane fitted by scout",
                "scene_type": {"tierA_hope": "black office console table, window backlight"}.get(d, "table/desk (YCB-Video lab scenes)"),
                "approx_camera_elevation_deg": round(90 - tp["camera_tilt_from_vertical_deg"]),
                "camera_height_above_plane_m": tp["camera_height_above_plane_m"],
                "width": g["width"], "height": g["height"],
                "objects": [o["name"] for o in g["objects"]],
                "n_objects": len(g["objects"]),
            })
    src_lines += [
        "## Tier A (ground truth)", "",
        "- **HOPE-Image** (val scenes 6, 7, 8): © 2020 NVIDIA Corporation, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). "
        "Tyree et al., *6-DoF Pose Estimation of Household Objects for Robotic Manipulation: An Accessible Dataset and Benchmark*, IROS 2022. "
        "Obtained via the BOP benchmark ([huggingface.co/datasets/bop-benchmark/hope](https://huggingface.co/datasets/bop-benchmark/hope)).",
        "- **YCB-Video** (BOP'19 test frames from scenes 48, 50, 52, 55, 57, 59): © 2017 Robotics and State Estimation Lab, University of Washington, MIT License. "
        "Xiang et al., *PoseCNN*, RSS 2018; Hodaň et al., BOP benchmark "
        "([huggingface.co/datasets/bop-benchmark/ycbv](https://huggingface.co/datasets/bop-benchmark/ycbv)).",
        "- Table-plane fits, table-frame coordinates and homographies in the JSON files are derived by the Rehearsal team.", "",
        "## Tier B (real photos, no ground truth) — all from Wikimedia Commons, resized to ≤2000 px (long side), EXIF stripped", ""]
    for stem, (scene, elev, ref, objs, notes) in B.items():
        r = sel[stem]
        artist = re.sub(r"\s+", " ", r["artist"]).strip()
        if artist.startswith("No machine-readable author"):
            artist = "Oliverkeenan (Wikimedia Commons user; author per Commons copyright claim)"
        lic = r["license"]
        unsplash = "(Unsplash)" in r["title"]
        entries.append({
            "path": f"tierB_photos/{stem}.jpg", "tier": "B", "license": lic, "license_url": lic_url(r),
            "author": artist, "source_url": r["page"], "file_url": r.get("orig_url", r["url"].split("?")[0]),
            "original_title": r["title"], "has_reference_object": ref or False, "has_ground_truth": False,
            "scene_type": scene, "approx_camera_elevation_deg": elev,
            "objects": objs, "n_objects": len(objs), "notes": notes + ("; originally from Unsplash, CC0 on Commons" if unsplash else ""),
        })
        src_lines.append(f"- `{stem}.jpg` — \"{r['title'][5:]}\" by {artist}, {lic}"
                         f"{' (' + lic_url(r) + ')' if lic_url(r) else ''}, via Wikimedia Commons: {r['page']}. Resized.")
    for e in entries:
        im = Image.open(os.path.join(ROOT, e["path"])); e["width"], e["height"] = im.size
    json.dump({"generated_by": "datasets/_tools/build_manifest.py",
               "elevation_convention": "approx_camera_elevation_deg: angle between the line of sight and the table plane; 90 = straight top-down. Tier A values come from the fitted plane, Tier B are eyeballed (+-10 deg).",
               "images": entries}, open(f"{ROOT}/MANIFEST.json", "w"), indent=1, ensure_ascii=False)
    with open(f"{ROOT}/SOURCES.md", "w") as f:
        f.write("# Test-photo sources and attribution\n\nPaste-ready attribution for the Rehearsal README. "
                "Licenses: CC0, CC BY 2.0, CC BY-SA 2.0/2.5/4.0, MIT, CC BY-NC-SA 4.0 (HOPE: non-commercial only).\n\n")
        f.write("\n".join(src_lines) + "\n")
    # contact sheet
    cols, tw, th = 5, 400, 300
    rows = (len(entries) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 22)), "white"); d = ImageDraw.Draw(sheet)
    for k, e in enumerate(entries):
        im = Image.open(os.path.join(ROOT, e["path"])).convert("RGB"); im.thumbnail((tw - 6, th - 6))
        x, y = (k % cols) * tw, (k // cols) * (th + 22)
        sheet.paste(im, (x + 3 + (tw - 6 - im.width) // 2, y + 3))
        tag = f"[{e['tier']}{'*' if e['has_reference_object'] else ''}] {os.path.basename(e['path'])[:44]}"
        d.text((x + 4, y + th + 4), tag, fill="black")
    sheet.save(f"{ROOT}/contact_sheet.jpg", quality=85)
    print(len(entries), "entries;", sum(e["tier"] == "A" for e in entries), "tier A")


if __name__ == "__main__":
    main(sys.argv[1])
