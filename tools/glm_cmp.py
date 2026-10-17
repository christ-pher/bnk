#!/usr/bin/env python3
"""Compares a glm_ref.py dump directory (.npy per tensor) with bnk's BNK_DUMP_DIR (raw float32 per tensor, every
window's rows appended): correlation and relative error per tensor, plus the row of the largest error.

    python tools/glm_cmp.py REF_DIR BNK_DIR [--rows N]"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("bnk")
ap.add_argument("--rows", type=int, default=0, help="compare the first N rows only")
a = ap.parse_args()
for f in sorted(os.listdir(a.ref)):
    if not f.endswith(".npy"):
        continue
    k = f[:-4]
    p = os.path.join(a.bnk, k)
    if not os.path.exists(p):
        continue
    v = np.load(os.path.join(a.ref, f))
    b = np.fromfile(p, dtype=np.float32)
    per_row = v.size // v.shape[0]
    if b.size > v.size and b.size % per_row == 0:
        b = b[:v.size]
    if b.size != v.size:
        print(f"{k:16s} size mismatch: bnk {b.size} ref {v.size}")
        continue
    b = b.reshape(v.shape)
    if a.rows:
        v, b = v[:a.rows], b[:a.rows]
    x = v.astype(np.float64)
    y = b.astype(np.float64)
    corr = np.corrcoef(x.ravel(), y.ravel())[0, 1] if x.std() > 0 and y.std() > 0 else float("nan")
    rel = np.linalg.norm(x - y) / (np.linalg.norm(x) + 1e-30)
    rowerr = np.linalg.norm(x - y, axis=1) / (np.linalg.norm(x, axis=1) + 1e-30)
    worst = int(np.argmax(rowerr))
    print(f"{k:16s} corr {corr:.6f}  rel err {rel:.3e}  worst row {worst} ({rowerr[worst]:.2e})  "
          f"rows>1e-2: {(rowerr > 1e-2).sum()}/{len(rowerr)}")
