#!/usr/bin/env python3
"""Writes a small GGUF of random weight matrices in the given ggml formats, for tests/test_dequant.

    .venv/bin/python tests/make_quant_gguf.py OUT.gguf [--rows 512] [--cols 2560] [--types nvfp4,iq4_nl,q8_0]

NVFP4 blocks are generated directly (random e2m1 codes, UE4M3 scales over the normal range), so every code and scale
value occurs; the other formats are quantized from random normal floats with gguf-py. test_dequant then checks
bnk's GPU unpack (raw blocks, and R layout with BNK_TEST_R=1) against ggml's CPU to_float."""
import argparse
import numpy as np
import gguf
from gguf.constants import GGMLQuantizationType as T

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--rows", type=int, default=512)
ap.add_argument("--cols", type=int, default=2560)
ap.add_argument("--types", default="nvfp4,q8_0")
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

w = gguf.GGUFWriter(a.out, "bnk-test")
w.add_uint32("bnk.test.rows", a.rows)
for ty in a.types.split(","):
    ty = ty.strip().lower()
    if ty == "nvfp4":
        assert a.cols % 64 == 0
        nb = a.cols // 64
        d = rng.integers(0x08, 0x7E, size=(a.rows, nb, 4), dtype=np.uint8)  # UE4M3 normals, no 0x7F
        d[:, 0, 0] = 0  # a zero scale and a subnormal one somewhere
        d[:, 0, 1] = 3
        qs = rng.integers(0, 256, size=(a.rows, nb, 32), dtype=np.uint8)
        raw = np.concatenate([d, qs], axis=-1).reshape(a.rows, nb * 36)
        w.add_tensor("w_nvfp4.weight", raw, raw_dtype=T.NVFP4)  # the byte shape implies the logical one
        continue
    qt = {"q8_0": T.Q8_0, "q4_0": T.Q4_0}[ty]  # the formats gguf-py can quantize
    x = rng.standard_normal((a.rows, a.cols)).astype(np.float32) * 0.05
    q = gguf.quants.quantize(x, qt)
    w.add_tensor(f"w_{ty}.weight", q, raw_dtype=qt)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", a.out)
