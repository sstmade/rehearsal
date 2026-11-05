#!/bin/bash
# HQ wall: one worker over a list of scene names.  MODE = still | orbit.  Wrap in tools/heavy.sh.
#   run_hq.sh MODE t6_01 t6_02 ...
cd "$(dirname "$0")/../.."
OUT="$PWD/../footage/tables_sim/wall_hq"; J="$OUT/../wall_primitive/_twin"; BL=/Applications/Blender.app/Contents/MacOS/Blender
FF=$(.venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
MODE=$1; shift
for n in "$@"; do
  id=${n#t6_}
  # RAM guard: keep >= 25% free
  while [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 25 ]; do sleep 10; done
  A=(--json "$J/$n.json" --colors "$OUT/_colors/$n.colors.json")
  if [ "$MODE" = still ]; then
    [ -f "$OUT/$id.png" ] || $BL -b --factory-startup -P scripts/wall_hq/render_hq.py -- "${A[@]}" --still "$OUT/$id.png" --spp 32 > "$OUT/_logs/$id.still.log" 2>&1
  else
    [ -f "$OUT/$id.mp4" ] && continue
    $BL -b --factory-startup -P scripts/wall_hq/render_hq.py -- "${A[@]}" --orbit "$OUT/_frames/$id" --frames 38 --deg 20 --spp 10 > "$OUT/_logs/$id.orbit.log" 2>&1
    "$FF" -loglevel error -y -framerate 15 -i "$OUT/_frames/$id/f_%03d.png" -vf "minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:vsbmc=1,trim=end_frame=75" -c:v libx264 -pix_fmt yuv420p -crf 16 -preset fast "$OUT/$id.mp4.tmp.mp4" && mv "$OUT/$id.mp4.tmp.mp4" "$OUT/$id.mp4"
  fi
  echo "$(date +%T) $MODE $id done"
  [ "$MODE" = orbit ] && .venv/bin/python scripts/wall_hq/sheet_index.py "05:the lamp's collision cylinder stands in front of the photo camera and fills half the frame" > /dev/null 2>&1
done
