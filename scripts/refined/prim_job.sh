#!/bin/bash
# Primitive (untextured) version of a refined scene: same export, shots, cameras and timing; the physics collision
# shapes in matte colours.  Wrap in tools/heavy.sh (via queue_runner).
#   prim_job.sh KEY DIR INTENT_KEY pilot|full [RES] [SAMPLES]
set -e
cd "$(dirname "$0")/../.."
KEY=$1; D=$2; IK=$3; MODE=$4; RES=${5:-1600x900}; SPP=${6:-20}
R="$PWD/../footage/tables_sim/hero/refined"
EXP="$R/$D/_export"; BL=/Applications/Blender.app/Contents/MacOS/Blender
rm -f "${EXP:?}/${KEY:?}_prim.blend"; [ "$MODE" = pilot ] && rm -rf "${R:?}/${D:?}/_pilot_prim"
[ -f "$EXP/${KEY}_shots.json" ] || .venv/bin/python scripts/refined/shots.py "$EXP" "$KEY" "$R/_runs/cache"
[ -f "$EXP/${KEY}_prim.blend" ] || $BL -b --factory-startup -P scripts/refined/build_refined.py -- --exp "$EXP" --key "$KEY" \
     --scene "$R/_runs/cache/${KEY}_scene.json" --primitive --save "$EXP/${KEY}_prim.blend" > "$EXP/build_prim.log" 2>&1
if [ "$MODE" = pilot ]; then
  N=$(.venv/bin/python -c "import json;print(len(json.load(open('$EXP/${KEY}_shots.json'))['frames']))")
  FR="0,$((N/3)),$((N*2/3)),$((N-1))"
  $BL -b "$EXP/${KEY}_prim.blend" -P scripts/refined/render_refined.py -- --shots "$EXP/${KEY}_shots.json" \
      --out "$R/$D/_pilot_prim" --frames "$FR" --samples 24 --res 960x540 > "$EXP/pilot_prim.log" 2>&1
else
  $BL -b "$EXP/${KEY}_prim.blend" -P scripts/refined/render_refined.py -- --shots "$EXP/${KEY}_shots.json" \
      --out "$R/$D/_frames_prim" --samples $SPP --res $RES > "$EXP/render_prim.log" 2>&1
  .venv/bin/python scripts/refined/assemble_refined.py "$R/$D" "$KEY" "$EXP" "$R/$D/_frames_prim" "$R/_runs/intent/${IK}.json" --name "${D}_primitive"
fi
