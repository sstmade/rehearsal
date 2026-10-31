# Test photos (Wikimedia Commons)

| File | Source | License |
|---|---|---|
| messy_desk.jpg | https://commons.wikimedia.org/wiki/File:Messy_desk.jpg | CC BY-SA 2.0 |
| desk_8082053723.jpg | https://commons.wikimedia.org/wiki/File:Desk_(8082053723).jpg | CC BY-SA 2.0 |
| gfp_office_desk.jpg | https://commons.wikimedia.org/wiki/File:Gfp-office-desk-area.jpg | Public domain |

Downscaled to 1280 px wide by Wikimedia's thumbnailer. Credit the authors as listed on each file page if shown in the demo.

# Real photos with ground truth (`photos/real/`)

| File | Source | License |
|---|---|---|
| real/real_ycbv59.png | YCB-Video, BOP'19 test split, scene 000059 frame 001152 (`test/000059/rgb/001152.png` in `https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_test_bop19.zip`), unmodified | MIT, Copyright (c) 2017 Robotics and State Estimation Lab, University of Washington. Cite Xiang et al., "PoseCNN", RSS 2018; Hodan et al., BOP. |

`real_ycbv59.overlay.png` (SAM 3.1 masks + estimated footprints) and `real_ycbv59.twin_compare.png` (overlay next to
the MuJoCo twin rendered from the dataset camera) are derived from it under the same license.
