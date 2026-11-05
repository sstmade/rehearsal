#!/bin/bash
# Re-encode HQ orbits from their 15 fps frames: blend-interpolated to 30 fps, exactly 75 frames (2.5 s). Then rebuild index.
#   reencode_orbits.sh [--wait]   (--wait: loop until all 48 orbit jobs are done, re-encoding each as it appears)
cd "."
OUT="../footage/tables_sim/wall_hq"
FF=$(.venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
EX="05:the lamp's collision cylinder stands in front of the photo camera and fills half the frame"
while true; do
  for m in "$OUT"/[0-9][0-9].mp4; do
    [ -f "$m" ] || continue
    id=$(basename "$m" .mp4); [ -f "$OUT/_frames/$id/.re" ] && continue
    "$FF" -loglevel error -y -framerate 15 -i "$OUT/_frames/$id/f_%03d.png" \
      -vf "minterpolate=fps=30:mi_mode=blend,tpad=stop_mode=clone:stop=10" -frames:v 75 \
      -c:v libx264 -pix_fmt yuv420p -crf 16 -preset fast "$OUT/_re_$id.mp4" && mv "$OUT/_re_$id.mp4" "$m" && touch "$OUT/_frames/$id/.re"
  done
  .venv/bin/python scripts/wall_hq/sheet_index.py "$EX" > /dev/null 2>&1
  n=$(cat "$OUT/_logs/oA.log" "$OUT/_logs/oB.log" | grep -c "orbit .* done")
  [ "$1" = --wait ] && [ "$n" -lt 48 ] || break
  sleep 30
done
echo "$(date +%T) reencode pass done ($n orbit jobs)"
