#!/bin/sh
# Reproduce datasets/tierA_* from public sources (everything is already committed; datasets/ is ~21 MB).
# Tier A members are pulled out of the BOP zips on Hugging Face with HTTP range requests (~55 MB transferred).
# Needs curl + a Python with numpy and pillow (PY=...). Downloads are data only; Python runs with -I.
# Usage (from the repo root): sh datasets/fetch.sh
set -e
PY=${PY:-python3}
T=${T:-$(mktemp -d)}
HF=https://huggingface.co/datasets/bop-benchmark
RZ="$PY -I datasets/_tools/remotezip.py get"
$RZ $HF/hope/resolve/main/hope_models.zip $T models/models_info.json=hope_models_info.json
$RZ $HF/ycbv/resolve/main/ycbv_models.zip $T models/models_info.json=ycbv_models_info.json
for s in 000006 000007 000008; do
  mkdir -p $T/hope/$s
  $RZ $HF/hope/resolve/main/hope_val_realsense.zip $T/hope/$s val/$s/scene_camera.json val/$s/scene_gt.json \
    val/$s/scene_gt_info.json val/$s/rgb/000000.png=rgb_000000.png val/$s/depth/000000.png=depth_000000.png
done
for pair in 000048:001024 000050:000640 000052:000001 000055:000128 000057:000001 000059:001152; do
  s=${pair%:*}; im=${pair#*:}; mkdir -p $T/ycbv/$s
  $RZ $HF/ycbv/resolve/main/ycbv_test_bop19.zip $T/ycbv/$s test/$s/scene_camera.json test/$s/scene_gt.json \
    test/$s/scene_gt_info.json test/$s/rgb/$im.png=rgb_$im.png test/$s/depth/$im.png=depth_$im.png
done
echo "Raw files in $T. Build the JSONs with: $PY -I datasets/_tools/bop_to_json.py <config.json>"
echo "(config format in bop_to_json.py docstring; object-name tables are in tierA_*/README.md / the GT JSONs)."
# Tier B: originals listed in MANIFEST.json ("file_url"); resized with datasets/_tools/fetch_resize.py.
