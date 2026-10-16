#!/usr/bin/env python3
"""Converts NVIDIA's ModelOpt NVFP4 checkpoint of Qwen3.8-Flash-Next (nvidia/Qwen3.8-Flash-Next-FP4-Local) to a GGUF
bnk can serve on V100s, through llama.cpp's converter.

    /root/.cache/bnk/convert-venv/bin/python tools/convert_nvfp4.py SNAPSHOT_DIR OUT.gguf [--no-ple] [--fp8 q8_0|bf16]

llama.cpp's convert_hf_to_gguf.py (pinned checkout, see LLAMA_CPP below) already knows the architecture (qwen4exp) and
repacks ModelOpt NVFP4 bit for bit into ggml's NVFP4 blocks, writing each tensor's fp32 global scale (weight_scale_2)
as a side tensor `<name>.scale` ([n_expert] for the routed experts). Two things in this checkpoint it cannot read:

  * the routed experts' block scales are stored "csf"-compressed (weight_scale_encoding: csf): per 16-row slab,
    16 row bases then 4-bit codes (scale = base + code), plus a list of exceptions (row * C + col | byte << 24).
    This script decodes them into the dense e4m3 `weight_scale` tensors the converter expects;
  * the dense projections are MXFP8 (e4m3 weights, one UE8M0 exponent per 32 weights, as uint8), which the
    converter's FP8 path would read as integer scales. This script dequantizes them to float and has the converter
    store them as Q8_0 (--fp8 q8_0, default) or BF16.

The n-gram PLE table (51B parameters, NVFP4 with 160-wide rows: not a multiple of ggml's 64-wide NVFP4 block) is
left out with --no-ple; bnk then takes it from another GGUF (--ple-gguf, e.g. the IQ3_S release's IQ4_NL table).
A bnk-native 16-wide NVFP4 table is planned (docs/NVFP4.md).

The MTP head is left out (bnk drafts with its own MTP file). The output's extra `.scale` / `.input_scale` tensors
are read (scale) or ignored (input_scale) by bnk's loader."""
import argparse
import json
import os
import runpy
import sys
from pathlib import Path

LLAMA_CPP = Path(os.environ.get("BNK_LLAMA_CPP", str(Path.home() / ".cache/bnk/llama.cpp")))
LLAMA_CPP_COMMIT = "f3363593f457169cdfbdfe22ea399b1cc139ba18"

ap = argparse.ArgumentParser()
ap.add_argument("snapshot", help="the checkpoint directory (config.json, tokenizer, tensors/*.safetensors)")
ap.add_argument("out", help="output GGUF")
ap.add_argument("--no-ple", action="store_true", help="leave the n-gram PLE table out (use bnk --ple-gguf)")
ap.add_argument("--fp8", choices=["q8_0", "bf16"], default="q8_0", help="how to store the MXFP8 dense weights")
ap.add_argument("--staging", default=None, help="staging dir with the trimmed index (default: next to OUT)")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

sys.path.insert(0, str(LLAMA_CPP))
sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gguf  # noqa: E402
from conversion import base as cbase  # noqa: E402
from conversion.base import ModelBase, LazyTorchTensor  # noqa: E402

snap = Path(args.snapshot).resolve()
index = json.load(open(snap / "model.safetensors.index.json"))
weight_map = index["weight_map"]


def is_ple_table(name: str) -> bool:
    return ".ple_embedding.ngram_embedding." in name


# --- staging dir: the checkpoint's files, minus the shards we leave out, with an index to match -----------------
staging = Path(args.staging) if args.staging else Path(args.out).resolve().parent / (Path(args.out).stem + ".staging")
staging.mkdir(parents=True, exist_ok=True)
(staging / "tensors").mkdir(exist_ok=True)
# the index must list exactly the tensors of the shards present (the converter checks), so the PLE table's tensors
# are dropped in the hook below, not here; only shards made of nothing but the table may be absent
keep = {n: f for n, f in weight_map.items() if (snap / f).exists()}
absent = {f for n, f in weight_map.items() if not (snap / f).exists()}
bad = sorted({f for n, f in weight_map.items() if f in absent and not (args.no_ple and is_ple_table(n))})
if bad:
    sys.exit(f"shards missing from {snap}: {bad}")
shards = sorted(set(keep.values()))
for f in snap.iterdir():
    if f.name == "tensors" or f.name == "model.safetensors.index.json":
        continue
    dst = staging / f.name
    if not dst.exists():
        dst.symlink_to(f)
for s in shards:
    dst = staging / s
    if not dst.exists():
        dst.symlink_to(snap / s)
json.dump({"metadata": index.get("metadata", {}), "weight_map": keep}, open(staging / "model.safetensors.index.json", "w"))
print(f"staging: {staging} ({len(shards)} shards, {len(keep)} tensors)", flush=True)


# --- the csf decode ----------------------------------------------------------------------------------------------
def decode_csf(fixed: np.ndarray, exceptions: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """-> uint8 [rows, cols/16] e4m3 scale bytes."""
    sc = cols // 16
    fixed = fixed.reshape(rows // 16, 16 + 16 * (sc // 2))
    bases = fixed[:, :16]                               # [slabs, 16]
    codes = fixed[:, 16:].reshape(rows // 16, 16, sc // 2)
    scale = np.empty((rows // 16, 16, sc), np.uint8)
    scale[:, :, 0::2] = codes & 0xF
    scale[:, :, 1::2] = codes >> 4
    scale += bases[:, :, None]
    scale = scale.reshape(rows, sc)
    if exceptions.size:
        idx = exceptions & 0xFFFFFF
        scale[idx // sc, idx % sc] = (exceptions >> 24).astype(np.uint8)
    return scale


def to_np(t) -> np.ndarray:
    t = LazyTorchTensor.to_eager(t)
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8).numpy()
    return t.numpy()


forced_q8: set[str] = set()
ple_row_dim: int | None = None
csf_scales: set[str] = set()
# the csf exception lists are U32, which the converter's safetensors dtype map leaves out
LazyTorchTensor._dtype_str_map["U32"] = torch.uint32
LazyTorchTensor._dtype_map[torch.uint32] = np.uint32
LazyTorchTensor._dtype_byteswap_map[torch.uint32] = np.uint32
orig_index = ModelBase.index_tensors


def index_tensors(self, remote_hf_model_id=None):
    t = orig_index(self, remote_hf_model_id)
    if args.no_ple:  # the converter asserts that every quantized module has tensors
        ql = (self.hparams.get("quantization_config") or {}).get("quantized_layers") or {}
        for k in [k for k in ql if is_ple_table(k + ".")]:
            del ql[k]
    n_csf = n_fp8 = n_ple = 0
    for name in list(t.keys()):
        if name.endswith(".weight_scale.nvfp4_csf_fixed"):
            base = name[: -len(".nvfp4_csf_fixed")]          # X.weight_scale
            wname = base[: -len("_scale")]                   # X.weight
            exc = base + ".nvfp4_csf_exceptions"
            w = t[wname]()
            rows, cols = int(w.shape[0]), int(w.shape[1]) * 2

            def gen(fx=t[name], ex=t[exc], rows=rows, cols=cols):
                s = decode_csf(to_np(fx()).ravel(), to_np(ex()).ravel().astype(np.uint32), rows, cols)
                return torch.from_numpy(s).view(torch.float8_e4m3fn)
            t[base] = gen
            csf_scales.add(base)
            del t[name], t[exc]
            n_csf += 1
        elif args.no_ple and is_ple_table(name):
            if name.endswith("shard_0.weight"):
                global ple_row_dim
                ple_row_dim = int(t[name]().shape[-1]) * 2
            del t[name]
            n_ple += 1
    for name in list(t.keys()):
        if not name.endswith(".weight_scale") or name not in t or name in csf_scales:
            continue
        wname = name[: -len("_scale")]
        if wname not in t:
            continue
        w, s = t[wname](), t[name]()
        if w.dtype != torch.float8_e4m3fn or s.dtype != torch.uint8 or s.ndim != 2:
            continue

        def gen(w=t[wname], s=t[name]):
            wf = LazyTorchTensor.to_eager(w()).float()
            e = LazyTorchTensor.to_eager(s()).to(torch.int32) - 127
            scale = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e)
            return wf * scale.repeat_interleave(32, dim=1)[:, : wf.shape[1]]
        t[wname] = gen
        del t[name]
        forced_q8.add(wname)
        n_fp8 += 1
    print(f"bnk: decoded {n_csf} csf scale tensors, {n_fp8} MXFP8 matrices to float"
          f"{' (-> Q8_0)' if args.fp8 == 'q8_0' else ''}, {n_ple} PLE table tensors left out", flush=True)
    return t


ModelBase.index_tensors = index_tensors

# Qwen4ExpTextModel.__init__ resets _ple_row_dim after indexing; restore it before the metadata is written, so bnk
# still gets embedding_length_per_layer_input with the table left out
from conversion.qwen4exp import Qwen4ExpTextModel  # noqa: E402
orig_params = Qwen4ExpTextModel.set_gguf_parameters


def set_gguf_parameters(self):
    if args.no_ple and self._ple_row_dim is None:
        hp = self.hparams
        self._ple_row_dim = ple_row_dim or hp["ple_embed_dim"] // ((hp["ngram_size"] - 1) * hp["heads_per_ngram"])
        print(f"bnk: PLE row width {self._ple_row_dim} (table left out)", flush=True)
    orig_params(self)


Qwen4ExpTextModel.set_gguf_parameters = set_gguf_parameters

orig_force = ModelBase.tensor_force_quant


def tensor_force_quant(self, name, new_name, bid, n_dims):
    if args.fp8 == "q8_0" and name in forced_q8 and n_dims >= 2:
        return gguf.GGMLQuantizationType.Q8_0
    return orig_force(self, name, new_name, bid, n_dims)


ModelBase.tensor_force_quant = tensor_force_quant

sys.argv = ["convert_hf_to_gguf.py", str(staging), "--outfile", args.out, "--outtype", "bf16", "--no-mtp",
            "--model-name", "Qwen3.8-Flash-Next-NVFP4"] + (["--dry-run"] if args.dry_run else [])
print("bnk: running", " ".join(sys.argv), flush=True)
runpy.run_path(str(LLAMA_CPP / "convert_hf_to_gguf.py"), run_name="__main__")
