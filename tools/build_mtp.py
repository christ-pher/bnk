"""Build bnk's MTP draft-layer GGUF from the official Qwen checkpoint.

    .venv/bin/python tools/build_mtp.py --out ~/.cache/bnk/mtp/mtp-q2_0.gguf

Only the `mtp.*` tensors are fetched: the safetensors index and shard headers are read, then each tensor's byte
range is requested (about 5 GB in all), cached under --cache, and checked against a second read of its header
offsets. Dense weights stay BF16 (the engine converts them to Q8_0 at load), norms become F32, and the 512 routed
experts are quantized to ggml's Q2_0 (64 weights per block, levels {-1, 0, 1, 2} x d) with a per-block search for
the scale of least squared error (both signs of d, so the +2 level lands on whichever side needs it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = "Qwen/Qwen3.8-Flash-Next"
QK = 64  # ggml QK2_0


def fetch(url: str, rng: tuple[int, int] | None = None, tries: int = 5) -> bytes:
    hdr = {"User-Agent": "bnk-build-mtp"}
    if rng:
        hdr["Range"] = f"bytes={rng[0]}-{rng[1]}"
    tok = os.environ.get("HF_TOKEN")
    if tok:
        hdr["Authorization"] = f"Bearer {tok}"
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r:
                data = r.read()
            if rng and len(data) != rng[1] - rng[0] + 1:
                raise IOError(f"short read {len(data)}")
            return data
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            print(f"  retry {k + 1}: {e}", file=sys.stderr)
            time.sleep(2 + 3 * k)
    raise AssertionError


def mtp_tensors(repo: str) -> dict[str, dict]:
    """name -> {shard, dtype, shape, start, end} (absolute byte range in the shard) for every mtp.* tensor."""
    base = f"https://huggingface.co/{repo}/resolve/main/"
    index = json.loads(fetch(base + "model.safetensors.index.json"))
    names = sorted(n for n in index["weight_map"] if n.startswith("mtp."))
    out = {}
    for shard in sorted({index["weight_map"][n] for n in names}):
        n_hdr = struct.unpack("<Q", fetch(base + shard, (0, 7)))[0]
        header = json.loads(fetch(base + shard, (8, 8 + n_hdr - 1)))
        for n, meta in header.items():
            if n in names:
                a, b = meta["data_offsets"]
                out[n] = {"shard": shard, "dtype": meta["dtype"], "shape": meta["shape"], "start": 8 + n_hdr + a,
                          "end": 8 + n_hdr + b - 1}
    missing = set(names) - set(out)
    if missing:
        raise SystemExit(f"tensors missing from shard headers: {sorted(missing)}")
    return out


def load(repo: str, name: str, meta: dict, cache: Path, raw: bool = False) -> np.ndarray:
    if meta["dtype"] != "BF16":
        raise SystemExit(f"{name}: expected BF16, found {meta['dtype']}")
    path = cache / (name + ".bin")
    want = meta["end"] - meta["start"] + 1
    if not path.exists() or path.stat().st_size != want:
        print(f"  fetching {name} ({want / 2**20:.1f} MiB)", flush=True)
        url = f"https://huggingface.co/{repo}/resolve/main/{meta['shard']}"
        part = path.with_suffix(".part")
        with open(part, "wb") as f:
            step = 256 << 20
            for a in range(meta["start"], meta["end"] + 1, step):
                f.write(fetch(url, (a, min(meta["end"], a + step - 1))))
        part.rename(path)
    bits = np.memmap(path, dtype=np.uint16, mode="r").reshape(meta["shape"])
    if raw:
        return bits  # BF16 bit patterns, memory-mapped
    return bf16_to_f32(np.asarray(bits))


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def q2_0(w: np.ndarray) -> np.ndarray:
    """Rows of w (last dim a multiple of 64) -> ggml block_q2_0 bytes (fp16 d, 16 bytes of 2-bit codes)."""
    rows = w.reshape(-1, QK).astype(np.float32)
    out = np.empty((rows.shape[0], 2 + QK // 4), dtype=np.uint8)
    # candidate scales: fractions of the block's largest magnitude, either sign
    fr = np.linspace(0.40, 1.10, 22)
    chunk = 1 << 16
    for c0 in range(0, rows.shape[0], chunk):
        x = rows[c0:c0 + chunk]
        amax = np.abs(x).max(axis=1, keepdims=True)
        best_err = np.full((x.shape[0], 1), np.inf, dtype=np.float32)
        best_d = np.zeros((x.shape[0], 1), dtype=np.float32)
        for sign in (1.0, -1.0):
            for f in fr:
                # the largest code (+2) reaches 2d, so d spans amax/2 .. amax
                d = (sign * f * amax / 2.0).astype(np.float16).astype(np.float32)
                safe = np.where(d == 0, 1.0, d)
                q = np.clip(np.rint(x / safe), -1, 2)
                err = ((q * d - x) ** 2).sum(axis=1, keepdims=True)
                better = err < best_err
                best_err = np.where(better, err, best_err)
                best_d = np.where(better, d, best_d)
        safe = np.where(best_d == 0, 1.0, best_d)
        codes = (np.clip(np.rint(x / safe), -1, 2) + 1).astype(np.uint8)  # 00=-1 01=0 10=+1 11=+2
        codes = codes.reshape(-1, QK // 4, 4)
        packed = codes[:, :, 0] | (codes[:, :, 1] << 2) | (codes[:, :, 2] << 4) | (codes[:, :, 3] << 6)
        out[c0:c0 + chunk, :2] = best_d.astype(np.float16).view(np.uint8).reshape(-1, 2)
        out[c0:c0 + chunk, 2:] = packed
    return out


_SRC: np.ndarray | None = None


def _init(path: str, shape: list[int]):
    global _SRC
    _SRC = np.memmap(path, dtype=np.uint16, mode="r").reshape(shape)


def _quant_expert(e: int) -> np.ndarray:
    return q2_0(bf16_to_f32(np.asarray(_SRC[e])))


def quantize_experts(path: Path, shape: list[int], workers: int) -> np.ndarray:
    import multiprocessing as mp
    ne, rows, cols = shape
    out = np.empty((ne, rows, cols // QK * (2 + QK // 4)), dtype=np.uint8)
    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(str(path), shape)) as pool:
        for e, q in enumerate(pool.imap(_quant_expert, range(ne), chunksize=4)):
            out[e] = q.reshape(rows, -1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "bnk" / "mtp" / "official"), help="downloaded BF16 tensors")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    import gguf  # upstream gguf-py (pip install from llama.cpp's gguf-py)

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    print(f"reading the tensor list of {args.repo} ...", flush=True)
    metas = mtp_tensors(args.repo)
    (cache / "manifest.json").write_text(json.dumps(metas, indent=1))
    print(f"{len(metas)} mtp tensors", flush=True)

    w = gguf.GGUFWriter(args.out, "qwen4exp-mtp")
    w.add_string("bnk.mtp.source", f"{args.repo} BF16 checkpoint, mtp.* tensors")
    w.add_string("bnk.mtp.expert_format", "q2_0")
    w.add_string("bnk.mtp.expert_quantizer", "per-block least-squares scale search (tools/build_mtp.py)")
    digest = hashlib.sha256()
    for name, meta in sorted(metas.items()):
        bits = load(args.repo, name, meta, cache, raw=True)
        digest.update(name.encode())
        for k in range(0, bits.shape[0]):
            digest.update(np.asarray(bits[k]).tobytes())
        if name.endswith("experts.down_proj") or name.endswith("experts.gate_up_proj"):
            t0 = time.time()
            # ggml shape [in, out, n_expert]: the numpy array is [n_expert][out][in // 64 blocks of 18 bytes]
            q = quantize_experts(cache / (name + ".bin"), meta["shape"], args.workers)
            w.add_tensor(name, q, raw_dtype=gguf.GGMLQuantizationType.Q2_0)  # q holds the byte shape
            print(f"  {name}: Q2_0 {meta['shape']} in {time.time() - t0:.0f} s", flush=True)
            continue
        t = bf16_to_f32(np.asarray(bits))
        if t.ndim == 1:
            w.add_tensor(name, t.astype(np.float32))
        else:
            bf = (t.view(np.uint32) >> 16).astype(np.uint16)  # exact: the source is BF16
            w.add_tensor(name, bf, raw_shape=list(t.shape), raw_dtype=gguf.GGMLQuantizationType.BF16)
    w.add_string("bnk.mtp.source_sha256", digest.hexdigest())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
