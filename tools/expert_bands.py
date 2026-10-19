#!/usr/bin/env python3
"""The routing-count file's view of a model's experts: how much of the traffic the coldest experts carry.

    .venv/bin/python tools/expert_bands.py ~/.cache/bnk/counts-<model>.bnkc [--layers 48 --experts 512]

Bands are by rank within each layer (the engine ranks experts per layer); the share is of all routings.
"""
import argparse
import struct
import sys

ap = argparse.ArgumentParser()
ap.add_argument("counts")
ap.add_argument("--bands", default="10,20,30,40,50,60,70,80")
a = ap.parse_args()
d = open(a.counts, "rb").read()
magic, ver, nl, ne = d[:4], *struct.unpack("<III", d[4:16])
if magic != b"BNKC":
    sys.exit("not a BNKC counts file")
c = struct.unpack(f"<{nl * ne}I", d[16:16 + 4 * nl * ne])
total = sum(c)
print(f"{nl} layers x {ne} experts, {total:,} routings recorded")
rows = [sorted(c[l * ne:(l + 1) * ne]) for l in range(nl)]   # per layer, coldest first
never = sum(1 for l in range(nl) for v in rows[l] if v == 0)
print(f"experts never routed to: {never} of {nl * ne} ({100 * never / (nl * ne):.1f}%)")
print(f"{'coldest band':>13} {'routing share':>14} {'per-layer min..max share':>26}")
for pct in [int(x) for x in a.bands.split(",")]:
    k = ne * pct // 100
    shares = [sum(r[:k]) / max(1, sum(r)) for r in rows]
    share = sum(sum(r[:k]) for r in rows) / max(1, total)
    print(f"{pct:>12}% {100 * share:13.2f}% {100 * min(shares):12.2f}% .. {100 * max(shares):6.2f}%")
