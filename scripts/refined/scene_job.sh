#!/bin/bash
# One refined scene: shots -> .blend -> Cycles frames -> clean + labelled mp4.  Wrap in tools/heavy.sh.
#   scene_job.sh KEY DIR SCENE_JSON INTENT_KEY pilot|full [RES] [SAMPLES]
set -e
cd "$(dirname "$0")/../.."
KEY=$1; D=$2; SCN=$3; IK=$4; MODE=$5; RES=${6:-1280x720}; SPP=${7:-64}
R="$PWD/../footage/tables_sim/hero/refined"
EXP="$R/$D/_export"; BL=/Applications/Blender.app/Contents/MacOS/Blender
if [ "$MODE" = pilot ]; then rm -f "${EXP:?}/${KEY:?}.blend"; rm -rf "${R:?}/${D:?}/_pilot"; fi
[ -f "$EXP/${KEY}_shots.json" ] || .venv/bin/python scripts/refined/shots.py "$EXP" "$KEY" "$R/_runs/cache"
[ -f "$EXP/${KEY}.blend" ] || $BL -b --factory-startup -P scripts/refined/build_refined.py -- --exp "$EXP" --key "$KEY" \
     --scene "$R/_runs/cache/${KEY}_scene.json" --save "$EXP/${KEY}.blend" > "$EXP/build.log" 2>&1
if [ "$MODE" = pilot ]; then
  FR=$(.venv/bin/python -c "
import json;S=json.load(open('$EXP/${KEY}_shots.json'))['frames']
seen={};[seen.setdefault(s['shot'],i+len(S[i:])//3*0) for i,s in enumerate(S)]
print(','.join(str(v) for v in sorted(set([0]+list(seen.values())+[len(S)-1]))))")
  echo "pilot frames $FR"
  $BL -b "$EXP/${KEY}.blend" -P scripts/refined/render_refined.py -- --shots "$EXP/${KEY}_shots.json" \
      --out "$R/$D/_pilot" --frames "$FR" --samples 48 --res 960x540 > "$EXP/pilot.log" 2>&1
else
  $BL -b "$EXP/${KEY}.blend" -P scripts/refined/render_refined.py -- --shots "$EXP/${KEY}_shots.json" \
      --out "$R/$D/_frames" --samples $SPP --res $RES > "$EXP/render.log" 2>&1
  .venv/bin/python scripts/refined/assemble_refined.py "$R/$D" "$KEY" "$EXP" "$R/$D/_frames" "$R/_runs/intent/${IK}.json" --name "${D}"
fi
