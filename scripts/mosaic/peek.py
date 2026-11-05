"""Contact sheet of tile streams: peek.py file.npy out.png [frame] [max]"""
import sys

import numpy as np
from PIL import Image

a = np.load(sys.argv[1], mmap_mode="r")
f = int(sys.argv[3]) if len(sys.argv) > 3 else a.shape[1] // 2
n = min(a.shape[0], int(sys.argv[4]) if len(sys.argv) > 4 else 64)
cols = 8
rows = (n + cols - 1) // cols
h, w = a.shape[2:4]
img = np.zeros((rows * (h + 2), cols * (w + 2), 3), np.uint8)
for i in range(n):
    r, c = divmod(i, cols)
    img[r * (h + 2):r * (h + 2) + h, c * (w + 2):c * (w + 2) + w] = a[i, f]
Image.fromarray(img).save(sys.argv[2])
