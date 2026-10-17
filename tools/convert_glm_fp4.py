#!/usr/bin/env python3
"""Converts NVIDIA's ModelOpt NVFP4 checkpoint of GLM-5.3-Flash (nvidia/GLM-5.3-Flash-FP4-Local, 320B-A18B) to a GGUF
bnk can serve (architecture `glm5next`), one shard at a time, with no llama.cpp in the loop.

    .venv/bin/python tools/convert_glm_fp4.py OUT.gguf [--expert-layers 3-44] [--dl-dir DIR] [--keep-shards]
                                                       [--resume] [--dry-run] [--threads 8]

The checkpoint is 177 GB in 40 shards and the GGUF ~182 GB, so the source never sits on disk whole: the output is
planned first (every tensor's offset, a sparse file of the final size, `OUT.plan.json` beside it), then each shard is
downloaded, its tensors converted and written in place (pwrite), and the shard deleted. A run can stop and be
resumed (`--resume` skips the shards the plan marks done); `--expert-layers` limits which MoE layers' experts are
filled in now (their shards), the rest can be added by a later run. The dense weights come from shards 3-5.

What is written (docs/GLM.md has the architecture):

  * routed experts (42 layers x 288, NVFP4 with "csf_32x2_v1"-compressed block scales): decoded and repacked bit for
    bit into ggml's NVFP4 blocks (type 40: 4 UE4M3 scales + 32 code bytes per 64 weights), one tensor per layer and
    matrix `blk.N.ffn_{gate,up,down}_exps.weight` [288, rows, cols] with the fp32 global scales (`weight_scale_2`)
    as `blk.N.ffn_*_exps.scale` [288], llama.cpp's convention (ggml halves the scale and doubles the e2m1 table);
  * MXFP8 (e4m3 + one UE8M0 exponent per 32) and FP8_PB_WO (e4m3 + one fp32 scale per 128 x 128 block) dense
    projections: dequantized and stored as Q8_0 (the one re-quantization, as for Qwen3.8-Flash-Next-FP4-Local);
  * BF16 matrices (hyper-connections, routers, the indexers' projections, kv_b, the token table) as BF16; 1-D
    tensors (norms, biases, gates' A_log / dt_bias) as F32; the conv taps as F32 [channels, 4];
  * the tokenizer (byte-level BPE, the GLM-4 pre-tokenizer), the chat template, the sampling defaults.
  * The MTP layer (45) and the vision tower are left out (bnk drafts with --mtp later; vision is not used).

csf_32x2_v1 (verified on real experts: decoded scales reach exactly 6 x 448 at the block holding the matrix's amax, and
every exception sits where the 4-bit code saturated): a matrix's [rows, cols/16] e4m3 scale bytes are split in 32
pages of rows/32 rows; per row a u8 base (`csf32_bases`), per scale a 4-bit code (`csf32_codes`, low nibble first):
byte = base + code. Scales the code cannot reach are exceptions: `csf32_page_bases` [32] u32 is each page's byte offset
into `csf32_exceptions` (u16 each), `csf32_offsets` [32 x 257] u16 the exception index (relative to the page) at the
start of each of the page's 256 segments of 64 scales (and its end); an exception is (byte << 8) | position in its
segment."""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

LLAMA_CPP = Path(os.environ.get("BNK_LLAMA_CPP", str(Path.home() / ".cache/bnk/llama.cpp")))
sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
import gguf  # noqa: E402
from gguf.constants import GGMLQuantizationType as T  # noqa: E402

REPO = "nvidia/GLM-5.3-Flash-FP4-Local"
ARCH = "glm5next"
PFX = "model.language_model."

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--repo", default=REPO)
ap.add_argument("--revision", default=None)
ap.add_argument("--dl-dir", default=str(Path.home() / ".cache/bnk/glm-dl"), help="where shards are downloaded to")
ap.add_argument("--expert-layers", default="3-44", help="MoE layers whose experts are converted in this run (A-B)")
ap.add_argument("--no-dense", action="store_true", help="skip the dense shards (3-5) this run")
ap.add_argument("--keep-shards", action="store_true", help="do not delete a shard after converting it")
ap.add_argument("--resume", action="store_true", help="continue a plan left by an earlier run")
ap.add_argument("--dry-run", action="store_true", help="plan and print sizes only")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--prefetch", type=int, default=1, help="shards downloaded ahead")
args = ap.parse_args()

dl = Path(args.dl_dir)
dl.mkdir(parents=True, exist_ok=True)
out = Path(args.out).resolve()
plan_path = out.with_suffix(out.suffix + ".plan.json")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ the hub
from huggingface_hub import hf_hub_download, hf_hub_url  # noqa: E402
import urllib.request  # noqa: E402


def small(name: str) -> Path:
    return Path(hf_hub_download(args.repo, name, revision=args.revision, local_dir=str(dl)))


def shard_header(fname: str) -> dict:
    """The safetensors header of a shard on the hub (one range request), cached in the download dir."""
    cache = dl / "headers" / (fname + ".json")
    if cache.exists():
        return json.load(open(cache))
    url = hf_hub_url(args.repo, fname, revision=args.revision)
    tok = os.environ.get("HF_TOKEN")
    hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
    req = urllib.request.Request(url, headers={**hdr, "Range": "bytes=0-7"})
    with urllib.request.urlopen(req) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    req = urllib.request.Request(url, headers={**hdr, "Range": f"bytes=8-{7 + n}"})
    with urllib.request.urlopen(req) as r:
        h = json.loads(r.read(n))
    cache.parent.mkdir(exist_ok=True)
    json.dump(h, open(cache, "w"))
    return h


class Shard:
    """A downloaded safetensors shard, read through a memory map."""

    def __init__(self, path: Path, name: str):
        self.path = path
        self.name = name
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U8": np.uint8, "U16": np.uint16, "U32": np.uint32,
          "F8_E4M3": np.uint8, "I32": np.int32, "I64": np.int64}

    def get(self, name: str) -> np.ndarray:
        h = self.header[name]
        a, b = h["data_offsets"]
        return self.mm[self.base + a:self.base + b].view(self.DT[h["dtype"]]).reshape(h["shape"])

    def has(self, name: str) -> bool:
        return name in self.header


# ------------------------------------------------------------------------------------------------ formats
_E4M3 = np.array([
    (lambda b: (-1.0 if b & 0x80 else 1.0) * (float("nan") if (b & 0x7F) == 0x7F else
                ((b & 7) / 8.0 * 2.0 ** -6 if ((b >> 3) & 0xF) == 0 else (1 + (b & 7) / 8.0) * 2.0 ** (((b >> 3) & 0xF) - 7))))(b)
    for b in range(256)], dtype=np.float32)


def bf16_to_f32(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.uint32) << 16).view(np.float32)


def mxfp8_to_f32(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """e4m3 weights [R, C], UE8M0 exponents [R, C/32] -> fp32."""
    R, C = w.shape
    f = _E4M3[w].reshape(R, C // 32, 32)
    e = np.ldexp(np.float32(1.0), s.astype(np.int32) - 127).astype(np.float32)
    return (f * e[:, :, None]).reshape(R, C)


def pbwo_to_f32(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """e4m3 weights [R, C], fp32 scales [R/128, 1, C/128, 1] (per 128 x 128 block) -> fp32."""
    R, C = w.shape
    s = s.reshape(s.shape[0], s.shape[2])
    br, bc = R // s.shape[0], C // s.shape[1]
    f = _E4M3[w].reshape(s.shape[0], br, s.shape[1], bc)
    return (f * s[:, None, :, None]).reshape(R, C)


def q8_0(x: np.ndarray) -> np.ndarray:
    return gguf.quants.quantize(np.ascontiguousarray(x, dtype=np.float32), T.Q8_0)


def csf32_decode(bases, codes, exc, off, pb, rows: int, cols: int) -> np.ndarray:
    """-> u8 [rows, cols/16] e4m3 scale bytes (sign bit clear)."""
    sc = cols // 16
    c = np.empty(codes.size * 2, np.uint8)
    c[0::2] = codes & 0xF
    c[1::2] = codes >> 4
    s = (c.reshape(rows, sc).astype(np.int16) + bases.astype(np.int16)[:, None]).astype(np.uint8)
    if exc.size:
        off = off.reshape(32, 257).astype(np.int64)
        pb = pb.astype(np.int64)
        if not (pb % 2 == 0).all() or not (pb // 2 == np.concatenate([[0], np.cumsum(off[:, 256])[:-1]])).all():
            raise ValueError("csf32: page bases do not match the segment offsets")
        counts = (off[:, 1:] - off[:, :-1]).ravel()            # exceptions per (page, segment)
        if counts.sum() != exc.size or (counts < 0).any():
            raise ValueError("csf32: exception counts do not add up")
        seg = np.repeat(np.arange(32 * 256), counts)            # each exception's page * 256 + segment
        page, sg = seg // 256, seg % 256
        flat = s.reshape(32, -1)
        idx = sg * 64 + (exc & 0xFF).astype(np.int64)
        flat[page, idx] = (exc >> 8).astype(np.uint8)
    return s & 0x7F


def nvfp4_pack(w: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """ModelOpt packed codes [R, C/2] (low nibble first) and scale bytes [R, C/16] -> ggml NVFP4 rows [R, C/64*36]."""
    R = w.shape[0]
    nb = scales.shape[1]
    v = w.reshape(R, nb, 8)
    vals = np.empty((R, nb, 16), np.uint8)
    vals[:, :, 0::2] = v & 0x0F
    vals[:, :, 1::2] = v >> 4
    qs = vals[:, :, :8] | (vals[:, :, 8:] << 4)
    ns = nb // 4
    d = (scales & 0x7F).reshape(R, ns, 4)
    return np.concatenate([d, qs.reshape(R, ns, 32)], axis=-1).reshape(R, ns * 36)


# ------------------------------------------------------------------------------------------------ the plan
cfg_path, idx_path = small("config.json"), small("model.safetensors.index.json")
tok_path, tmpl_path, gen_path = small("tokenizer.json"), small("chat_template.jinja"), small("generation_config.json")
cfg = json.load(open(cfg_path))
tc = cfg["text_config"]
wm = json.load(open(idx_path))["weight_map"]
n_layer = tc["num_hidden_layers"]
n_exp = tc["n_routed_experts"]
E, F_EXP, F_DENSE = tc["hidden_size"], tc["moe_intermediate_size"], tc["intermediate_size"]
layer_types = tc["layer_types"]
mlp_types = tc["mlp_layer_types"]
la = tc["linear_attn_config"]
lo_, hi_ = (int(x) for x in args.expert_layers.split("-"))
expert_layers = [il for il in range(n_layer) if mlp_types[il] == "sparse" and lo_ <= il <= hi_]

shards = sorted({f for f in wm.values()})
log(f"{args.repo}: {n_layer} layers, {len(shards)} shards; this run: dense {'no' if args.no_dense else 'yes'}, experts of layers {expert_layers[0] if expert_layers else '-'}..{expert_layers[-1] if expert_layers else '-'}")

# tensor plan: gguf name -> (byte shape, raw type, nbytes, source)
Entry = dict
plan: dict[str, Entry] = {}


def add(name: str, byte_shape: list[int], ty: T, src):
    nbytes = int(np.prod(byte_shape))
    plan[name] = {"shape": byte_shape, "type": int(ty), "nbytes": nbytes, "src": src}


def bytes_of(ty: T, shape: list[int]) -> list[int]:
    """byte shape of a logical [..., cols] shape in format ty"""
    bs, tsz = gguf.GGML_QUANT_SIZES[ty]
    return list(shape[:-1]) + [shape[-1] // bs * tsz]


# headers of the shards: shapes and dtypes of every tensor
headers: dict[str, dict] = {}
for f in shards:
    headers[f] = shard_header(f)
H = {}
for f, h in headers.items():
    for k, v in h.items():
        if k != "__metadata__":
            H[k] = (v["dtype"], v["shape"], f)


def hf(name: str):
    return H.get(name)


def dense_matrix(hf_name: str, gname: str):
    """A projection: MXFP8 (u8 scale [R, C/32]) or FP8_PB_WO (f32 scale 4-D) -> Q8_0; BF16 -> BF16."""
    dt, shape, f = hf(hf_name)
    R, C = shape
    sc = hf(hf_name.replace(".weight", ".weight_scale"))
    if dt == "F8_E4M3":
        kind = "mxfp8" if sc[0] == "U8" else "pbwo"
        add(gname, bytes_of(T.Q8_0, [R, C]), T.Q8_0, {"hf": hf_name, "kind": kind, "shard": f})
    elif dt == "BF16":
        add(gname, [R, C * 2], T.BF16, {"hf": hf_name, "kind": "raw", "shard": f})
    else:
        raise ValueError(f"{hf_name}: unexpected dtype {dt}")


def part_kind(hf_name: str):
    dt, shape, f = hf(hf_name)
    if dt == "F8_E4M3":
        return ("mxfp8" if hf(hf_name.replace(".weight", ".weight_scale"))[0] == "U8" else "pbwo"), f
    if dt == "BF16":
        return "raw", f
    raise ValueError(f"{hf_name}: unexpected dtype {dt}")


def stacked_matrix(gname: str, parts: list[str], q8: bool):
    """Matrices sharing their input, stacked along the rows (one GEMV in the engine): q8 stores every part as Q8_0
    (BF16 parts quantized), else every part must be BF16 and the stack is BF16."""
    rows, cols, srcs, r0 = 0, None, [], 0
    for hfn in parts:
        dt, shape, f = hf(hfn)
        R, C = shape
        if cols is None:
            cols = C
        if C != cols:
            raise ValueError(f"{gname}: {hfn} has {C} columns, not {cols}")
        kind, f = part_kind(hfn)
        srcs.append({"hf": hfn, "kind": kind, "shard": f, "row0": r0, "rows": R})
        rows += R
        r0 += R
    ty = T.Q8_0 if q8 else T.BF16
    if not q8 and any(p["kind"] != "raw" for p in srcs):
        raise ValueError(f"{gname}: a BF16 stack needs BF16 parts")
    byte_shape = bytes_of(ty, [rows, cols])
    plan[gname] = {"shape": byte_shape, "type": int(ty), "nbytes": int(np.prod(byte_shape)),
                   "src": {"kind": "stack", "parts": srcs, "q8": q8, "row_bytes": byte_shape[1]}}


def vec_f32(hf_name: str, gname: str, shape=None):
    dt, sh, f = hf(hf_name)
    sh = shape or sh
    add(gname, [int(np.prod(sh)) * 4] if len(sh) == 1 else [sh[0], sh[1] * 4], T.F32, {"hf": hf_name, "kind": "f32", "shard": f})


dense_matrix(PFX + "embed_tokens.weight", "token_embd.weight")
dense_matrix("lm_head.weight", "output.weight")
vec_f32(PFX + "norm.weight", "output_norm.weight")
for il in range(n_layer):
    p = f"{PFX}layers.{il}."
    b = f"blk.{il}."
    for hc in ("hc_attn", "hc_ffn"):
        dense_matrix(p + hc + "_fn", b + hc + "_fn.weight")
        vec_f32(p + hc + "_base", b + hc + "_base")
        vec_f32(p + hc + "_scale", b + hc + "_scale")
    vec_f32(p + "input_layernorm.weight", b + "attn_norm.weight")
    vec_f32(p + "post_attention_layernorm.weight", b + "ffn_norm.weight")
    if layer_types[il] == "linear_attention":
        # q | k | v stacked (one GEMV; the conv channels in the same order), their conv taps F32 [3 C, K]
        stacked_matrix(b + "kda_qkv.weight", [p + f"self_attn.{m}_proj.weight" for m in ("q", "k", "v")], q8=True)
        convs = [p + f"self_attn.{m}_conv1d.weight" for m in ("q", "k", "v")]
        dt, sh, f = hf(convs[0])   # BF16 [C, 1, K] each (a layer's three may sit in different shards)
        add(b + "kda_conv.weight", [3 * sh[0], sh[2] * 4], T.F32,
            {"kind": "f32_stack", "parts": [{"hf": h, "shard": hf(h)[2], "row0": i * sh[0]} for i, h in enumerate(convs)],
             "reshape": [sh[0], sh[2]]})
        dense_matrix(p + "self_attn.o_proj.weight", b + "kda_out.weight")
        # the forget gate's, the output gate's and beta's first projections share x: f_a | g_a | b (BF16)
        stacked_matrix(b + "kda_fgb_a.weight", [p + f"self_attn.{m}_proj.weight" for m in ("f_a", "g_a", "b")], q8=False)
        dense_matrix(p + "self_attn.f_b_proj.weight", b + "kda_f_b.weight")
        dense_matrix(p + "self_attn.g_b_proj.weight", b + "kda_g_b.weight")
        vec_f32(p + "self_attn.dt_bias", b + "kda_dt_bias")
        vec_f32(p + "self_attn.A_log", b + "kda_a_log")
        vec_f32(p + "self_attn.o_norm.weight", b + "kda_o_norm.weight")
    else:
        # q_a | kv_a share x; q_b | the indexer's wq_b share the normalized q_a output; the indexer's wk |
        # weights_proj | k-pool gate share x (all Q8_0)
        stacked_matrix(b + "attn_qkv_a.weight", [p + "self_attn.q_a_proj.weight", p + "self_attn.kv_a_proj_with_mqa.weight"], q8=True)
        vec_f32(p + "self_attn.q_a_layernorm.weight", b + "attn_q_a_norm.weight")
        vec_f32(p + "self_attn.kv_a_layernorm.weight", b + "attn_kv_a_norm.weight")
        stacked_matrix(b + "attn_q_b_idx.weight", [p + "self_attn.q_b_proj.weight", p + "self_attn.indexer.wq_b.weight"], q8=True)
        dense_matrix(p + "self_attn.kv_b_proj.weight", b + "attn_kv_b.weight")
        dense_matrix(p + "self_attn.o_proj.weight", b + "attn_output.weight")
        stacked_matrix(b + "idx_x.weight", [p + "self_attn.indexer.wk.weight", p + "self_attn.indexer.weights_proj.weight",
                                             p + "self_attn.indexer.index_kpool_compress_gate"], q8=True)
        vec_f32(p + "self_attn.indexer.k_norm.weight", b + "idx_k_norm.weight")
        vec_f32(p + "self_attn.indexer.k_norm.bias", b + "idx_k_norm.bias")
        vec_f32(p + "self_attn.indexer.index_kpool_compress_ape", b + "idx_ape")
    if mlp_types[il] == "dense":
        stacked_matrix(b + "ffn_gate_up.weight", [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"], q8=True)
        dense_matrix(p + "mlp.down_proj.weight", b + "ffn_down.weight")
    else:
        dense_matrix(p + "mlp.gate.weight", b + "ffn_gate_inp.weight")
        vec_f32(p + "mlp.gate.e_score_correction_bias", b + "ffn_gate_inp.bias")
        stacked_matrix(b + "ffn_gate_up_shexp.weight", [p + "mlp.shared_experts.gate_proj.weight", p + "mlp.shared_experts.up_proj.weight"], q8=True)
        dense_matrix(p + "mlp.shared_experts.down_proj.weight", b + "ffn_down_shexp.weight")
        for m in ("gate", "up", "down"):
            dt, sh, f = hf(p + f"mlp.experts.0.{m}_proj.weight")
            R, C = sh[0], sh[1] * 2
            add(b + f"ffn_{m}_exps.weight", [n_exp, R, C // 64 * 36], T.NVFP4,
                {"kind": "experts", "layer": il, "mat": m, "rows": R, "cols": C})
            add(b + f"ffn_{m}_exps.scale", [n_exp * 4], T.F32, {"kind": "expert_scales", "layer": il, "mat": m})

total = sum(e["nbytes"] for e in plan.values())
exp_bytes = sum(e["nbytes"] for e in plan.values() if e["src"]["kind"] == "experts")
log(f"plan: {len(plan)} tensors, {total / 1e9:.1f} GB (experts {exp_bytes / 1e9:.1f} GB, dense {(total - exp_bytes) / 1e9:.1f} GB)")

# which shards this run needs
need_shards = set()
if not args.no_dense:
    for e in plan.values():
        if "shard" in e["src"]:
            need_shards.add(e["src"]["shard"])
        for part in e["src"].get("parts", []):
            need_shards.add(part["shard"])
for name, f in wm.items():
    m = re.match(re.escape(PFX) + r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$", name)
    if m and int(m.group(1)) in expert_layers:
        need_shards.add(f)
need_shards = sorted(need_shards)
log(f"shards needed this run: {len(need_shards)}: {', '.join(s[6:11] for s in need_shards)}")
if args.dry_run:
    sys.exit(0)

# ------------------------------------------------------------------------------------------------ tokenizer / metadata
def tokenizer_meta():
    t = json.load(open(tok_path))
    vocab = t["model"]["vocab"]
    n_vocab = tc["vocab_size"]
    tokens = [f"[PAD{i}]" for i in range(n_vocab)]
    types = [int(gguf.TokenType.UNUSED)] * n_vocab
    for s, i in vocab.items():
        tokens[i] = s
        types[i] = int(gguf.TokenType.NORMAL)
    for a in t["added_tokens"]:
        tokens[a["id"]] = a["content"]
        types[a["id"]] = int(gguf.TokenType.CONTROL if a["special"] else gguf.TokenType.USER_DEFINED)
    merges = [" ".join(m) if isinstance(m, list) else m for m in t["model"]["merges"]]
    return tokens, types, merges


def write_metadata(w: gguf.GGUFWriter):
    w.add_name("GLM-5.3-Flash-FP4-Local")
    w.add_string("general.basename", "GLM-5.3-Flash")
    w.add_string("general.source", args.repo)
    w.add_uint32("general.file_type", int(gguf.LlamaFileType.MOSTLY_Q8_0))
    k = lambda s: f"{ARCH}.{s}"  # noqa: E731
    w.add_uint32(k("block_count"), n_layer)
    w.add_uint32(k("nextn_predict_layers"), 0)
    w.add_uint32(k("context_length"), tc["max_position_embeddings"])
    w.add_uint32(k("embedding_length"), E)
    w.add_uint32(k("vocab_size"), tc["vocab_size"])
    w.add_uint32(k("feed_forward_length"), F_DENSE)
    w.add_uint32(k("leading_dense_block_count"), tc["first_k_dense_replace"])
    w.add_uint32(k("expert_count"), n_exp)
    w.add_uint32(k("expert_used_count"), tc["num_experts_per_tok"])
    w.add_uint32(k("expert_shared_count"), tc["n_shared_experts"])
    w.add_uint32(k("expert_feed_forward_length"), F_EXP)
    w.add_uint32(k("expert_shared_feed_forward_length"), F_EXP * tc["n_shared_experts"])
    w.add_float32(k("expert_weights_scale"), tc["routed_scaling_factor"])
    w.add_bool(k("expert_weights_norm"), tc["norm_topk_prob"])
    w.add_string(k("expert_gating_func"), tc["scoring_func"])
    w.add_uint32(k("expert_group_count"), tc["n_group"])
    w.add_uint32(k("expert_group_used_count"), tc["topk_group"])
    w.add_float32(k("swiglu_limit"), tc["swiglu_limit"])
    w.add_float32(k("attention.layer_norm_rms_epsilon"), tc["rms_norm_eps"])
    w.add_array(k("layer_types"), [0 if t == "linear_attention" else 1 for t in layer_types])
    w.add_array(k("mlp_layer_types"), [0 if t == "dense" else 1 for t in mlp_types])
    # hyper-connections (mHC)
    w.add_uint32(k("hc.mult"), tc["hc_mult"])
    w.add_float32(k("hc.eps"), tc["hc_eps"])
    w.add_uint32(k("hc.sinkhorn_iters"), tc["hc_sinkhorn_iters"])
    # KDA linear attention
    w.add_uint32(k("kda.head_count"), la["num_heads"])
    w.add_uint32(k("kda.head_dim"), la["head_dim"])
    w.add_uint32(k("kda.conv_kernel"), la["short_conv_kernel_size"])
    w.add_float32(k("kda.gate_lower_bound"), la["gate_lower_bound"])
    # MLA + DSA
    w.add_uint32(k("attention.head_count"), tc["num_attention_heads"])
    w.add_uint32(k("attention.head_count_kv"), tc["num_key_value_heads"])
    w.add_uint32(k("attention.q_lora_rank"), tc["q_lora_rank"])
    w.add_uint32(k("attention.kv_lora_rank"), tc["kv_lora_rank"])
    w.add_uint32(k("attention.key_length"), tc["qk_nope_head_dim"])
    w.add_uint32(k("attention.rope_length"), tc["qk_rope_head_dim"])
    w.add_uint32(k("attention.value_length"), tc["v_head_dim"])
    w.add_uint32(k("attention.indexer.head_count"), tc["index_n_heads"])
    w.add_uint32(k("attention.indexer.key_length"), tc["index_head_dim"])
    w.add_uint32(k("attention.indexer.top_k"), tc["index_topk"])
    w.add_uint32(k("attention.indexer.kpool"), tc["index_kpool"])
    w.add_bool(k("attention.indexer.select_tail"), tc["index_kpool_always_select_tail"])
    # tokenizer
    tokens, types, merges = tokenizer_meta()
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("glm4")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    gen = json.load(open(gen_path))
    eos = gen["eos_token_id"] if isinstance(gen["eos_token_id"], list) else [gen["eos_token_id"]]
    w.add_eos_token_id(eos[0])
    w.add_pad_token_id(gen.get("pad_token_id", eos[0]))
    w.add_array(k("eos_token_ids"), eos)
    w.add_bool("tokenizer.ggml.add_bos_token", False)
    w.add_chat_template(open(tmpl_path).read())
    w.add_float32("general.sampling.temp", float(gen.get("temperature", 1.0)))
    w.add_float32("general.sampling.top_p", float(gen.get("top_p", 0.95)))


# ------------------------------------------------------------------------------------------------ the output file
state = None
if args.resume and plan_path.exists() and out.exists():
    state = json.load(open(plan_path))
    if state["plan"] != {n: [e["shape"], e["type"], e["nbytes"]] for n, e in plan.items()}:
        sys.exit("the existing plan does not match this run's (different layers or checkpoint?)")
    log(f"resuming: {len(state['done'])} shards done")
else:
    if out.exists() and not args.resume:
        sys.exit(f"{out} exists (use --resume to continue it, or remove it)")
    w = gguf.GGUFWriter(str(out), ARCH)
    write_metadata(w)
    for n, e in plan.items():
        w.add_tensor_info(n, e["shape"], np.uint8, e["nbytes"], raw_dtype=T(e["type"]))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    fout = w.fout[0]
    fout.flush()
    pos = fout.tell()
    data_start = gguf.GGUFWriter.ggml_pad(pos, w.data_alignment)
    fout.write(bytes(data_start - pos))
    fout.flush()
    fout.close()
    offsets = {}
    off = 0
    for n, e in plan.items():
        offsets[n] = off
        off += gguf.GGUFWriter.ggml_pad(e["nbytes"], w.data_alignment)
    os.truncate(out, data_start + off)   # a sparse file of the final size
    state = {"data_start": data_start, "offsets": offsets, "done": [],
             "plan": {n: [e["shape"], e["type"], e["nbytes"]] for n, e in plan.items()}}
    json.dump(state, open(plan_path, "w"))
    log(f"{out}: header + metadata {data_start / 1e6:.1f} MB, data {off / 1e9:.1f} GB (sparse)")

fd = os.open(out, os.O_RDWR)
data_start = state["data_start"]
offsets = state["offsets"]
done = set(state["done"])
plan_lock = threading.Lock()


def pwrite(name: str, arr: np.ndarray, byte_off: int = 0):
    buf = np.ascontiguousarray(arr).view(np.uint8).ravel()
    n = os.pwrite(fd, buf, data_start + offsets[name] + byte_off)
    if n != buf.size:
        raise OSError(f"short write for {name}")


def mark_done(f: str):
    with plan_lock:
        done.add(f)
        state["done"] = sorted(done)
        json.dump(state, open(plan_path, "w"))


# ------------------------------------------------------------------------------------------------ conversion
def convert_matrix(hfn: str, kind: str, sh: Shard) -> np.ndarray:
    if kind == "raw":
        return sh.get(hfn)
    w8 = sh.get(hfn)
    s = sh.get(hfn.replace(".weight", ".weight_scale"))
    x = mxfp8_to_f32(w8, s) if kind == "mxfp8" else pbwo_to_f32(w8, s)
    return q8_0(x)


def convert_dense(name: str, e: Entry, sh: Shard):
    src = e["src"]
    kind = src["kind"]
    if kind == "stack":   # the parts this shard holds
        for part in src["parts"]:
            if part["shard"] != sh.name:
                continue
            hfn = part["hf"]
            if src["q8"] and part["kind"] == "raw":
                x = q8_0(bf16_to_f32(sh.get(hfn)))
            else:
                x = convert_matrix(hfn, part["kind"], sh)
            pwrite(name, x, part["row0"] * src["row_bytes"])
        return
    if kind == "f32_stack":
        for part in src["parts"]:
            if part["shard"] != sh.name:
                continue
            x = bf16_to_f32(sh.get(part["hf"])).reshape(src["reshape"]).astype(np.float32)
            pwrite(name, x, part["row0"] * src["reshape"][1] * 4)
        return
    hfn = src["hf"]
    if kind == "raw":
        pwrite(name, sh.get(hfn))
    elif kind == "f32":
        a = sh.get(hfn)
        dt = sh.header[hfn]["dtype"]
        x = bf16_to_f32(a) if dt == "BF16" else a.astype(np.float32)
        if "reshape" in src:
            x = x.reshape(src["reshape"])
        pwrite(name, x.astype(np.float32))
    elif kind in ("mxfp8", "pbwo"):
        w8 = sh.get(hfn)
        s = sh.get(hfn.replace(".weight", ".weight_scale"))
        x = mxfp8_to_f32(w8, s) if kind == "mxfp8" else pbwo_to_f32(w8, s)
        pwrite(name, q8_0(x))
    else:
        raise ValueError(kind)


def convert_expert(il: int, ex: int, m: str, sh: Shard):
    hfn = f"{PFX}layers.{il}.mlp.experts.{ex}.{m}_proj."
    w = sh.get(hfn + "weight")
    rows, cols = w.shape[0], w.shape[1] * 2
    sb = hfn + "weight_scale."
    if sh.has(sb + "csf32_codes"):
        scales = csf32_decode(sh.get(sb + "csf32_bases"), sh.get(sb + "csf32_codes"), sh.get(sb + "csf32_exceptions"),
                              sh.get(sb + "csf32_offsets"), sh.get(sb + "csf32_page_bases"), rows, cols)
    else:
        scales = sh.get(hfn + "weight_scale").reshape(rows, cols // 16)
    packed = nvfp4_pack(w, scales)
    gname = f"blk.{il}.ffn_{m}_exps.weight"
    pwrite(gname, packed, ex * packed.nbytes)
    s2 = sh.get(hfn + "weight_scale_2").astype(np.float32).reshape(1)
    pwrite(f"blk.{il}.ffn_{m}_exps.scale", s2, ex * 4)


EXP_RE = re.compile(re.escape(PFX) + r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")


def convert_shard(f: str, sh: Shard, pool: ThreadPoolExecutor):
    jobs = []
    n_dense = n_exp_ = 0
    if not args.no_dense:
        for name, e in plan.items():
            src = e["src"]
            if src.get("shard") == f or any(p["shard"] == f for p in src.get("parts", [])):
                jobs.append(pool.submit(convert_dense, name, e, sh))
                n_dense += 1
    for hfn in sh.header:
        m = EXP_RE.match(hfn)
        if m and int(m.group(1)) in expert_layers:
            jobs.append(pool.submit(convert_expert, int(m.group(1)), int(m.group(2)), m.group(3), sh))
            n_exp_ += 1
    for j in jobs:
        j.result()
    return n_dense, n_exp_


def download(f: str) -> Path:
    return Path(hf_hub_download(args.repo, f, revision=args.revision, local_dir=str(dl)))


todo = [f for f in need_shards if f not in done]
log(f"{len(todo)} shards to convert")
t_all = time.time()
with ThreadPoolExecutor(max_workers=args.threads) as pool, ThreadPoolExecutor(max_workers=max(1, args.prefetch)) as dlp:
    futures = {}
    for i, f in enumerate(todo[:args.prefetch]):
        futures[f] = dlp.submit(download, f)
    for i, f in enumerate(todo):
        t0 = time.time()
        path = futures.pop(f).result() if f in futures else download(f)
        t1 = time.time()
        for g in todo[i + 1:i + 1 + args.prefetch]:
            if g not in futures:
                futures[g] = dlp.submit(download, g)
        sh = Shard(path, f)
        nd, ne = convert_shard(f, sh, pool)
        del sh
        t2 = time.time()
        if not args.keep_shards:
            os.remove(path)
        mark_done(f)
        log(f"{f[6:11]}: {nd} dense + {ne} expert matrices, download {t1 - t0:.0f} s, convert {t2 - t1:.0f} s "
            f"({i + 1}/{len(todo)}, {(time.time() - t_all) / 60:.1f} min)")
os.fsync(fd)
os.close(fd)
log(f"done: {out} ({out.stat().st_size / 1e9:.1f} GB apparent, {os.stat(out).st_blocks * 512 / 1e9:.1f} GB on disk)")
