"""Download raw DROID 1.0.1 files for a list of episode ids into an untrusted scratch directory.  $0, no login.

    <python> -I datasets/droid/_tools/fetch_v2.py <annot dir> <out dir> [--meta-only] <eid> [...]

Fetches metadata_<eid>.json (names the operator: NEVER vendor it), trajectory.h5 and the two exterior left-eye MP4s.
Only episodes in KarlP/droid cam2base_extrinsic_superset.json (improved extrinsics) can be fetched.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
import ssl

import certifi

B = "https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1/"


def main(argv):
    annot, out_root, *rest = argv
    meta_only = "--meta-only" in rest
    eids = [e for e in rest if not e.startswith("--")]
    ctx = ssl.create_default_context(cafile=certifi.where())
    sup = json.load(open(os.path.join(annot, "cam2base_extrinsic_superset.json")))
    for eid in eids:
        rp = sup[eid]["relative_path"]
        out = os.path.join(out_root, eid)
        os.makedirs(out, exist_ok=True)

        def get(rel, name):
            p = os.path.join(out, name)
            if not os.path.exists(p):
                with urllib.request.urlopen(B + urllib.parse.quote(rel), context=ctx, timeout=120) as r:
                    data = r.read()
                with open(p, "wb") as f:
                    f.write(data)
        try:
            get(f"{rp}/metadata_{eid}.json", "metadata.json")
            m = json.load(open(os.path.join(out, "metadata.json")))
            get(f"{rp}/trajectory.h5", "trajectory.h5")
            if not meta_only:
                for key in ("ext1", "ext2"):
                    s = m[f"{key}_cam_serial"]
                    get(f"{rp}/recordings/MP4/{s}.mp4", f"{key}_{s}.mp4")
            print(eid, "ok", flush=True)
        except Exception as e:  # noqa: BLE001 - report and continue with the next episode
            print(eid, "FAIL", type(e).__name__, e, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
