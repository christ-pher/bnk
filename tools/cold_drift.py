#!/usr/bin/env python3
"""Has the routing drifted away from a cold-experts file? (docs/EXPERT-MIX.md)

    .venv/bin/python tools/cold_drift.py COLD.gguf COUNTS.bnkc [--threshold 0.10]

The file fixes which experts of each layer are cold and in which band; this compares that with the bands the
current routing counts would choose (the same rule tools/requant_experts uses: per layer, coldest first by
count, the band percentages from the file's own metadata) and prints the share of cold slots whose expert
changed. Exit 0: within the threshold; 3: drifted (regenerate with build/requant_experts); 2: cannot tell.
"""
import argparse
import struct
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])
from gguf import GGUFReader  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cold")
ap.add_argument("counts")
ap.add_argument("--threshold", type=float, default=0.10, help="share of cold slots whose expert changed")
a = ap.parse_args()

d = open(a.counts, "rb").read()
if d[:4] != b"BNKC":
    sys.exit(2)
_, nl, ne = struct.unpack("<III", d[4:16])
c = np.frombuffer(d[16:16 + 4 * nl * ne], dtype=np.uint32).reshape(nl, ne)

r = GGUFReader(a.cold)
bands_s = ""
for f in r.fields.values():
    if f.name == "bnk.cold.bands":
        bands_s = bytes(f.parts[-1]).decode()
if not bands_s:
    sys.exit(2)
pcts = [int(x.split(":")[1]) for x in bands_s.split(",")]
names = [f"b{i}" for i in range(len(pcts))]   # bands are named by index in the file
have = {}   # (layer, band) -> set of experts in the file
for t in r.tensors:
    if t.name.startswith("blk.") and ".cold_" in t.name and t.name.endswith(".experts"):
        layer = int(t.name.split(".")[1])
        band = t.name.split(".cold_")[1][: -len(".experts")]
        have[(layer, band)] = set(int(x) for x in np.asarray(t.data).ravel())
if not have:
    sys.exit(2)
changed = total = 0
for layer in range(nl):
    order = np.argsort(c[layer], kind="stable")   # coldest first
    start = 0
    for name, pct in zip(names, pcts):
        n = ne * pct // 100
        want = set(int(x) for x in order[start:start + n])
        start += n
        got = have.get((layer, name))
        if got is None:
            continue
        changed += len(want - got)
        total += len(want)
share = changed / total if total else 0.0
print(f"cold experts: {changed} of {total} slots would hold another expert today ({100 * share:.1f}%; bands {bands_s})")
sys.exit(3 if share > a.threshold else 0)
