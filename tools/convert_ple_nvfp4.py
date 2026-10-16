#!/usr/bin/env python3
"""Packs the n-gram PLE table of nvidia/Qwen3.8-Flash-Next-FP4-Local (128 NVFP4 shards of 2,500,012 x 160) into a GGUF
bnk reads with --ple-gguf (or BNK_PLE_GGUF): `per_layer_token_embd.weight` as bnk's 16-wide NVFP4 type
(GGML_TYPE_NVFP4_16 = 43: one UE4M3 scale byte then 8 code bytes per 16 weights, 90 bytes per row, 28.8 GB) plus
`per_layer_token_embd.scale` (F32 [1], the checkpoint's weight_scale_2). Bit for bit the checkpoint's data: the
e2m1 codes are re-ordered from ModelOpt's pairs (byte i: elements 2i, 2i+1) into ggml's split layout (byte i:
elements i, i+8); the UE4M3 scale bytes are copied.

    .venv/bin/python tools/convert_ple_nvfp4.py SNAPSHOT_DIR OUT.gguf [--check IQ3_S-00002-of-00002.gguf]

Streams one shard at a time (numpy only; no torch). --check compares the first rows with the IQ3_S release's
IQ4_NL table (the base model's: the correlation should be high, not 1)."""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

GGML_TYPE_NVFP4_16 = 43
GGML_TYPE_F32 = 0
KV_U32, KV_F32, KV_STR = 4, 6, 8
ALIGN = 32

ap = argparse.ArgumentParser()
ap.add_argument("snapshot")
ap.add_argument("out")
ap.add_argument("--check", default=None, help="the IQ3_S release's shard holding per_layer_token_embd (IQ4_NL)")
ap.add_argument("--rows", type=int, default=0, help="stop after this many rows (testing)")
a = ap.parse_args()
snap = Path(a.snapshot)
index = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
prefix = [n for n in index if n.endswith("ngram_embedding.shard_0.weight")]
assert len(prefix) == 1, prefix
prefix = prefix[0][: -len("shard_0.weight")]
n_shards = sum(1 for n in index if n.startswith(prefix) and n.endswith(".weight"))
scale2_name = prefix[: -len("ngram_embedding.")] + "ngram_embedding.weight_scale_2"
print(f"{n_shards} shards, scale2 tensor {scale2_name}", flush=True)


class Shard:
    """Lazily opened safetensors file: header + mmap."""
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.hdr = json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def tensor(self, name):
        t = self.hdr[name]
        o0, o1 = t["data_offsets"]
        return self.mm[self.base + o0: self.base + o1], t["dtype"], t["shape"]


shards: dict[str, Shard] = {}


def get(name):
    f = index[name]
    if f not in shards:
        shards[f] = Shard(snap / f)
    return shards[f].tensor(name)


s2, dt, sh = get(scale2_name)
assert dt == "F32" and int(np.prod(sh)) == 1
scale2 = float(np.frombuffer(s2.tobytes(), dtype=np.float32)[0])
print(f"scale2 = {scale2:.6g}", flush=True)

# shapes first (the row count goes in the header)
rows_of = []
dim = None
for i in range(n_shards):
    _, dt, sh = get(f"{prefix}shard_{i}.weight")
    assert dt == "U8" and len(sh) == 2, (dt, sh)
    _, dts, shs = get(f"{prefix}shard_{i}.weight_scale")
    assert dts == "F8_E4M3" and shs[0] == sh[0] and shs[1] * 8 == sh[1], (dts, shs, sh)
    if dim is None:
        dim = sh[1] * 2
    assert sh[1] * 2 == dim
    rows_of.append(sh[0])
n_rows = sum(rows_of)
if a.rows:
    n_rows = min(n_rows, a.rows)
nblk = dim // 16
row_bytes = nblk * 9
print(f"table {n_rows} x {dim}, {row_bytes} bytes per row, {n_rows * row_bytes / 2**30:.2f} GiB", flush=True)


def pack(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """w u8 [N, dim/2] (ModelOpt pairs), s u8 [N, dim/16] e4m3 -> [N, row_bytes] nvfp4_16 blocks."""
    N = w.shape[0]
    w = w.reshape(N, nblk, 8)
    e = np.empty((N, nblk, 16), np.uint8)
    e[:, :, 0::2] = w & 0x0F
    e[:, :, 1::2] = w >> 4
    out = np.empty((N, nblk, 9), np.uint8)
    out[:, :, 0] = s & 0x7F
    out[:, :, 1:] = e[:, :, :8] | (e[:, :, 8:] << 4)
    return out.reshape(N, row_bytes)


# --- GGUF writer -------------------------------------------------------------------------------------------------
def kv_str(k, v):
    k, v = k.encode(), v.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", KV_STR) + struct.pack("<Q", len(v)) + v


def kv_u32(k, v):
    k = k.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", KV_U32) + struct.pack("<I", v)


def tinfo(name, ne, ttype, off):
    n = name.encode()
    return (struct.pack("<Q", len(n)) + n + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", x) for x in ne)
            + struct.pack("<I", ttype) + struct.pack("<Q", off))


tab_bytes = n_rows * row_bytes
tab_off = 0
scale_off = (tab_bytes + ALIGN - 1) // ALIGN * ALIGN
hdr = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 2) + struct.pack("<Q", 3)
hdr += kv_str("general.architecture", "qwen4exp")
hdr += kv_str("general.name", "Qwen3.8-Flash-Next-FP4-Local PLE table")
hdr += kv_u32("general.alignment", ALIGN)
hdr += tinfo("per_layer_token_embd.weight", [dim, n_rows], GGML_TYPE_NVFP4_16, tab_off)
hdr += tinfo("per_layer_token_embd.scale", [1], GGML_TYPE_F32, scale_off)
pad = (-len(hdr)) % ALIGN
with open(a.out, "wb") as f:
    f.write(hdr + b"\0" * pad)
    done = 0
    first = None
    for i in range(n_shards):
        if done >= n_rows:
            break
        w, _, _ = get(f"{prefix}shard_{i}.weight")
        s, _, _ = get(f"{prefix}shard_{i}.weight_scale")
        N = rows_of[i]
        w = np.asarray(w).reshape(N, dim // 2)
        s = np.asarray(s).reshape(N, nblk)
        take = min(N, n_rows - done)
        packed = pack(w[:take], s[:take])
        if first is None:
            first = (w[:16].copy(), s[:16].copy(), packed[:16].copy())
        f.write(packed.tobytes())
        done += take
        fname = index[f"{prefix}shard_{i}.weight"]
        shards.pop(fname, None)  # drop the mapping (page cache)
        print(f"  shard {i}: {take} rows ({done / n_rows * 100:.1f}%)", flush=True)
    f.write(b"\0" * ((scale_off - tab_bytes) % ALIGN if scale_off > tab_bytes else 0))
    f.seek(len(hdr) + pad + scale_off)
    f.write(struct.pack("<f", scale2))
print(f"wrote {a.out}", flush=True)

# --- self-check: the packed rows decode (ggml convention) to the ModelOpt values ----------------------------------
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
KV = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)


def e4m3(x):
    x = x.astype(np.int32) & 0x7F
    e, m = x >> 3, (x & 7).astype(np.float32)
    v = np.where(e == 0, m * 2.0 ** -9, (1 + m / 8) * 2.0 ** (e - 7))
    return np.where(x == 0x7F, np.nan, v).astype(np.float32)


w, s, packed = first
ref = np.empty((16, dim), np.float32)   # ModelOpt: element 2i low, 2i+1 high
lo, hi = w & 0x0F, w >> 4
ref[:, 0::2] = E2M1[lo]
ref[:, 1::2] = E2M1[hi]
ref *= np.repeat(e4m3(s), 16, axis=1) * scale2
blk = packed.reshape(16, nblk, 9)
got = np.empty((16, nblk, 16), np.float32)
got[:, :, :8] = KV[blk[:, :, 1:] & 0x0F]
got[:, :, 8:] = KV[blk[:, :, 1:] >> 4]
got *= (e4m3(blk[:, :, 0]) * 0.5)[:, :, None] * scale2
got = got.reshape(16, dim)
print("self-check max |diff| =", float(np.nanmax(np.abs(got - ref))), "(must be 0); rms", float(np.sqrt((ref ** 2).mean())))
if a.check:
    from gguf import GGUFReader, quants
    r = GGUFReader(a.check)
    t = [t for t in r.tensors if t.name == "per_layer_token_embd.weight"][0]
    base = quants.dequantize(t.data[:16], t.tensor_type).astype(np.float32)
    print(f"vs the IQ3_S release's table (16 rows): corr {np.corrcoef(base.ravel(), ref.ravel())[0, 1]:.4f}, rms {np.sqrt((base ** 2).mean()):.4g} / {np.sqrt((ref ** 2).mean()):.4g}")
