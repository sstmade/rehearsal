#!/bin/bash
# zoom shots from the original brief (all public: AI tables only). Run through tools/heavy.sh.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
T=../footage/mosaic/_work/target_arm.npy
$PY scripts/mosaic/mosaic.py --shot zoomout --name mosaic_zoomout_arm --target $T
$PY scripts/mosaic/mosaic.py --shot hold --hold 6 --name mosaic_hold_arm --target $T --t0 200
$PY scripts/mosaic/mosaic.py --shot zoomin --name mosaic_zoomin_reverse --target $T --hold 1.5
