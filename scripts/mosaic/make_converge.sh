#!/bin/bash
# converge shots. Run through tools/heavy.sh.
#   T2=<new user-desk twin mp4> scripts/mosaic/make_converge.sh     -> private (+ public unless SKIP_PUBLIC=1)
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
M=../footage/mosaic
[ -n "$SKIP_PACK" ] || $PY scripts/mosaic/pack.py
if [ -n "$T2" ]; then
  # private: target = the user's desk twin, frame 86 held until the resolve completes, then plays on
  $PY scripts/mosaic/pack.py --decode "$T2" $M/private/_work/t2_full.npy
  N=$($PY -c "import numpy as np;print(len(np.load('$M/private/_work/t2_full.npy',mmap_mode='r')))")
  TAIL=$($PY -c "print(round(($N-86)/30,3))")
  $PY scripts/mosaic/mosaic.py --shot converge --name mosaic_converge_userdesk \
    --target $M/private/_work/t2_full.npy --full $M/private/_work/t2_full.npy --pin 0 \
    --tz 5 --hold 2.5 --resolve-at 5.5 --head 225 --vstart 86 --tail $TAIL --outdir $M/private
fi
if [ -z "$SKIP_PUBLIC" ]; then
  # public: target = the dark kitchen hero run (exec_dark/06.mp4 from the task start)
  $PY scripts/mosaic/mosaic.py --shot converge --name mosaic_converge_kitchen \
    --target $M/_work/hq.npy --full $M/_work/hq.npy --pin 0 \
    --tz 5 --hold 2.5 --resolve-at 5.5 --head 225 --vstart 0 --tail 4.0 --outdir $M/public
fi
