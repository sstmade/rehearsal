#!/bin/bash
# (re)render any missing tile sets, then pack + converge shots. Run through tools/heavy.sh.
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/mosaic/tiles.py --jobs 4
scripts/mosaic/make_converge.sh
