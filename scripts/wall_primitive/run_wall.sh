#!/bin/bash
# Primitive wall: one worker over a list of scene names.  MODE = still | orbit.  Wrap in tools/heavy.sh.
#   run_wall.sh MODE t6_01 t6_02 ...
cd "$(dirname "$0")/../.."
OUT="$PWD/../footage/tables_sim/wall_primitive"; J="$OUT/_twin"; BL=/Applications/Blender.app/Contents/MacOS/Blender
FF=$(.venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
MODE=$1; shift
for n in "$@"; do
  id=${n#t6_}
  if [ "$MODE" = still ]; then
    [ -f "$OUT/$id.png" ] || $BL -b --factory-startup -P scripts/wall_primitive/render_wall.py -- --json "$J/$n.json" --still "$OUT/$id.png" --taa 32 > "$OUT/_logs/$id.still.log" 2>&1
  else
    [ -f "$OUT/$id.mp4" ] && continue
    $BL -b --factory-startup -P scripts/wall_primitive/render_wall.py -- --json "$J/$n.json" --orbit "$OUT/_frames/$id" --frames 75 --deg 20 --taa 16 > "$OUT/_logs/$id.orbit.log" 2>&1
    "$FF" -loglevel error -y -framerate 30 -i "$OUT/_frames/$id/f_%03d.png" -c:v libx264 -pix_fmt yuv420p -crf 16 -preset fast "$OUT/$id.mp4.tmp.mp4" && mv "$OUT/$id.mp4.tmp.mp4" "$OUT/$id.mp4"
  fi
  echo "$(date +%T) $MODE $id done"
done
