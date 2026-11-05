"""Pixel (u, v) in a scene's photo -> table-frame (x, y) on the plane z (photo camera of the scene file)."""
import json
import sys

import numpy as np


def px2table(sc, u, v, z=0.0):
    c = sc["photo"]["camera"]
    K, R, t = (np.asarray(c[k], float) for k in ("K", "R", "t"))
    C = -R.T @ t
    d = R.T @ np.linalg.solve(K, [u, v, 1.0])
    s = (z - C[2]) / d[2]
    return (C + s * d)[:2]


if __name__ == "__main__":
    sc = json.load(open(sys.argv[1]))
    u, v = float(sys.argv[2]), float(sys.argv[3])
    z = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    print(np.round(px2table(sc, u, v, z), 4).tolist())
