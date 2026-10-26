"""The DSpark draft head of DeepSeek-V4.1-Flash-1017 as a GGUF, through upstream llama.cpp's converter.

    ~/.cache/bnk/convert-venv/bin/python tools/dspark_gguf.py --llama-cpp DIR [--tensors ~/.cache/bnk/dspark]
                                                              [--out ~/.cache/bnk/dspark/DeepSeek-V4.1-Flash-1017-DSpark.gguf]

tools/dspark_fetch.py leaves the checkpoint's mtp.* tensors as raw files plus a manifest. This script writes them
into one safetensors shard, fetches the repository's config.json and tokenizer files, writes an index that names
only that shard, and runs `convert_hf_to_gguf.py --dspark` (conversion/deepseek41.py: DeepseekV41DSparkModel) on
the directory. The result is llama.cpp's DFLASH layout: stages blk.0-2, experts MXFP4 bit-exact, dense BF16, fc =
main_proj, enc.output_norm = main_norm, output_norm = mtp.2.norm, markov_w1/w2, conf_proj (docs/DSPARK.md)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np
from safetensors.numpy import save_file

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
RAW = f"https://huggingface.co/{REPO}/raw/main/"
DTYPES = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "F8_E8M0": np.uint8, "I8": np.int8, "F16": np.float16}


def fetch_text(name: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    with urllib.request.urlopen(RAW + name, timeout=120) as r, open(dst, "wb") as f:
        f.write(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", required=True, help="an upstream llama.cpp checkout (convert_hf_to_gguf.py, gguf-py)")
    ap.add_argument("--tensors", default=os.path.expanduser("~/.cache/bnk/dspark"))
    ap.add_argument("--out", default="")
    ap.add_argument("--outtype", default="bf16")
    a = ap.parse_args()
    tdir = a.tensors
    out = a.out or os.path.join(tdir, "DeepSeek-V4.1-Flash-1017-DSpark.gguf")
    manifest = json.load(open(os.path.join(tdir, "manifest.json")))
    hf = os.path.join(tdir, "hf")   # the synthetic checkpoint directory the converter reads
    os.makedirs(hf, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        fetch_text(name, os.path.join(hf, name))
    shard = os.path.join(hf, "model-dspark.safetensors")
    if not os.path.exists(shard):
        tensors = {}
        for name, meta in sorted(manifest.items()):
            raw = np.fromfile(os.path.join(tdir, name + ".bin"), dtype=np.uint8)
            dt = DTYPES[meta["dtype"]]
            arr = raw.view(dt).reshape(meta["shape"])
            if meta["dtype"] == "BF16":
                arr = arr.view(np.uint16)
            tensors[name] = arr
        # safetensors' NumPy writer has no bf16 / fp8: write those as the same bytes with the right dtype strings
        md = {}
        save_file({k: (v if v.dtype != np.uint16 else v) for k, v in tensors.items()}, shard + ".tmp", metadata={"format": "pt"})
        # patch the header's dtype strings (uint16 -> BF16, uint8 -> the fp8 kind, int8 stays)
        import struct
        with open(shard + ".tmp", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            body = f.read()
        for name, meta in manifest.items():
            hdr[name]["dtype"] = meta["dtype"]
        hb = json.dumps(hdr, separators=(",", ":")).encode()
        hb += b" " * ((8 - len(hb) % 8) % 8)
        with open(shard, "wb") as f:
            f.write(struct.pack("<Q", len(hb)))
            f.write(hb)
            f.write(body)
        os.remove(shard + ".tmp")
    # an index naming only our shard for the mtp.* tensors (the converter with --dspark reads nothing else)
    idx = json.load(open(os.path.join(hf, "model.safetensors.index.json")))
    idx["weight_map"] = {k: "model-dspark.safetensors" for k in idx["weight_map"] if k.startswith("mtp.")}
    json.dump(idx, open(os.path.join(hf, "model.safetensors.index.json"), "w"))
    env = dict(os.environ, PYTHONPATH=os.path.join(a.llama_cpp, "gguf-py"))
    cmd = [sys.executable, os.path.join(a.llama_cpp, "convert_hf_to_gguf.py"), hf, "--dspark", "--outtype", a.outtype,
           "--outfile", out]
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd, env=env, cwd=a.llama_cpp)


if __name__ == "__main__":
    sys.exit(main())
