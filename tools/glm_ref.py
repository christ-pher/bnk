#!/usr/bin/env python3
"""Reference forward of a glm5next GGUF (tools/convert_glm_fp4.py) in fp32 on the CPU, following the Hugging Face
modeling code (Glm5NextForConditionalGeneration, transformers 5.16) step by step, for validating bnk's kernels.

    /root/.cache/bnk/convert-venv/bin/python tools/glm_ref.py MODEL.gguf --tokens "1 2 3" [--layers 4] [--dump DIR]
                                                                        [--dump-layer K] [--compare DIR]

Reads the same weights bnk reads (dequantized here: Q8_0, NVFP4 with the per-expert global scales, BF16 / F16 /
F32), so the only differences left are bnk's fp16 / int8 activation paths and accumulation order. The routed
experts are loaded on demand (only those the tokens route to). Dense attention only (prompts shorter than the
indexer's budget: every pool is selected), so --tokens must be at most index_topk + kpool - 1 tokens.

--dump DIR writes layer K's intermediates with the names bnk's BNK_DUMP_DIR uses (hc_attn_w, attn_in, attn_out,
hc_combine, ffn_in, ffn_out, l_last, and the attention-specific ones), each as float32 [T][...]; --compare DIR reads
bnk's dumps from DIR and prints the correlation and relative error per tensor."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(os.environ.get("BNK_LLAMA_CPP", str(Path.home() / ".cache/bnk/llama.cpp"))) / "gguf-py"))
import gguf  # noqa: E402
from gguf.constants import GGMLQuantizationType as T  # noqa: E402

torch.set_num_threads(24)

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tokens", required=True, help="token ids (space separated) or @FILE")
ap.add_argument("--layers", type=int, default=-1, help="run this many layers (default all)")
ap.add_argument("--dump", default="")
ap.add_argument("--dump-layer", type=int, default=0)
ap.add_argument("--compare", default="")
ap.add_argument("--max-new", type=int, default=0, help="greedy tokens to generate after the prompt (slow)")
args = ap.parse_args()

if args.tokens.startswith("@"):
    toks = [int(x) for x in open(args.tokens[1:]).read().replace(",", " ").split()]
else:
    toks = [int(x) for x in args.tokens.replace(",", " ").split()]

# ------------------------------------------------------------------------------------------------ the GGUF
r = gguf.GGUFReader(args.model)
tensors = {t.name: t for t in r.tensors}
fields = r.fields


def kv(name, default=None):
    f = fields.get(name)
    if f is None:
        return default
    if f.types[0] == gguf.GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode()
    if f.types[0] == gguf.GGUFValueType.ARRAY:
        return [f.parts[i][0] for i in f.data]
    return f.parts[f.data[0]][0]


A = "glm5next."
n_layer = int(kv(A + "block_count"))
E = int(kv(A + "embedding_length"))
n_exp = int(kv(A + "expert_count"))
k_used = int(kv(A + "expert_used_count"))
F_exp = int(kv(A + "expert_feed_forward_length"))
F_dense = int(kv(A + "feed_forward_length"))
hc = int(kv(A + "hc.mult"))
hc_eps = float(kv(A + "hc.eps"))
hc_iters = int(kv(A + "hc.sinkhorn_iters"))
eps = float(kv(A + "attention.layer_norm_rms_epsilon"))
H_kda = int(kv(A + "kda.head_count"))
S_kda = int(kv(A + "kda.head_dim"))
K_conv = int(kv(A + "kda.conv_kernel"))
lower = float(kv(A + "kda.gate_lower_bound"))
H = int(kv(A + "attention.head_count"))
q_lora = int(kv(A + "attention.q_lora_rank"))
kv_lora = int(kv(A + "attention.kv_lora_rank"))
Dk = int(kv(A + "attention.key_length"))
Dv = int(kv(A + "attention.value_length"))
ih = int(kv(A + "attention.indexer.head_count"))
idim = int(kv(A + "attention.indexer.key_length"))
topk = int(kv(A + "attention.indexer.top_k"))
kpool = int(kv(A + "attention.indexer.kpool"))
layer_types = kv(A + "layer_types")
mlp_types = kv(A + "mlp_layer_types")
route_scale = float(kv(A + "expert_weights_scale"))
swiglu_limit = float(kv(A + "swiglu_limit"))
select_tail = bool(kv(A + "attention.indexer.select_tail", True))

_E4M3 = np.array([(lambda b: (-1.0 if b & 0x80 else 1.0) * (float("nan") if (b & 0x7F) == 0x7F else
                   ((b & 7) / 8.0 * 2.0 ** -6 if ((b >> 3) & 0xF) == 0 else (1 + (b & 7) / 8.0) * 2.0 ** (((b >> 3) & 0xF) - 7))))(b)
                  for b in range(256)], dtype=np.float32)
_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def dequant(t, rows=None) -> np.ndarray:
    """A tensor (or a slice of leading-dim rows) as fp32 with its logical shape."""
    shape = [int(x) for x in reversed(t.shape)]   # numpy order
    data = np.asarray(t.data)
    if t.tensor_type == T.F32:
        return data.reshape(shape).astype(np.float32)
    if t.tensor_type == T.F16:
        return data.reshape(shape).astype(np.float32)
    if t.tensor_type == T.BF16:
        u = data.view(np.uint16).reshape(shape)
        return (u.astype(np.uint32) << 16).view(np.float32)
    if t.tensor_type == T.Q8_0:
        cols = shape[-1]
        raw = data.view(np.uint8).reshape(-1, cols // 32, 34)
        d = raw[:, :, :2].copy().view(np.float16).astype(np.float32)
        q = raw[:, :, 2:].view(np.int8).astype(np.float32)
        return (q * d).reshape(shape)
    if t.tensor_type == T.NVFP4:
        cols = shape[-1]
        raw = data.view(np.uint8).reshape(-1, cols // 64, 36)
        sc = _E4M3[raw[:, :, :4] & 0x7F]                       # [R, nb, 4]
        qs = raw[:, :, 4:].reshape(-1, cols // 64, 4, 8)         # 16-blocks: byte i = elem i | elem i+8 << 4
        vals = np.empty(qs.shape[:3] + (16,), np.float32)
        vals[..., :8] = _E2M1[qs & 0xF]
        vals[..., 8:] = _E2M1[qs >> 4]
        return (vals * sc[..., None]).reshape(shape)
    raise ValueError(f"{t.name}: type {t.tensor_type}")


def W(name) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(dequant(tensors[name])))


def expert(il, m, e) -> torch.Tensor:
    """one expert's matrix [rows, cols] with its global scale"""
    t = tensors[f"blk.{il}.ffn_{m}_exps.weight"]
    shape = [int(x) for x in reversed(t.shape)]   # [n_exp, rows, cols]
    rows, cols = shape[1], shape[2]
    per = rows * cols // 64 * 36
    raw = np.asarray(t.data).view(np.uint8).reshape(-1)[e * per:(e + 1) * per].reshape(rows, cols // 64, 36)
    sc = _E4M3[raw[:, :, :4] & 0x7F]
    qs = raw[:, :, 4:].reshape(rows, cols // 64, 4, 8)
    vals = np.empty((rows, cols // 64, 4, 16), np.float32)
    vals[..., :8] = _E2M1[qs & 0xF]
    vals[..., 8:] = _E2M1[qs >> 4]
    w = (vals * sc[..., None]).reshape(rows, cols)
    g = dequant(tensors[f"blk.{il}.ffn_{m}_exps.scale"])[e]
    return torch.from_numpy(w * np.float32(g))


# ------------------------------------------------------------------------------------------------ the maths
def rmsnorm(x, w, e=eps):
    v = x.float()
    return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + e) * w


def sigmoid(x):
    return torch.sigmoid(x)


def swiglu(g, u):
    if swiglu_limit > 0:
        g = g.clamp(max=swiglu_limit)
        u = u.clamp(-swiglu_limit, swiglu_limit)
    return torch.nn.functional.silu(g) * u


dumps = {}


def dump(il, name, x):
    if args.dump and il == args.dump_layer:
        dumps[name] = x.detach().float().cpu().numpy().reshape(x.shape[0], -1)


def mhc(streams, il, which):
    """streams [T, hc, E] -> (w [T, (2+hc) hc], collapsed [T, E])"""
    b = f"blk.{il}.{which}"
    fn, base, scale = W(b + "_fn.weight"), W(b + "_base"), W(b + "_scale")
    Tn = streams.shape[0]
    flat = streams.reshape(Tn, -1)
    flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + eps)
    f = flat @ fn.T
    pre_w, post_w, comb_w = f.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = base.split([hc, hc, hc * hc])
    pre = sigmoid(pre_w * scale[0] + pre_b) + hc_eps
    post = 2 * sigmoid(post_w * scale[1] + post_b)
    comb = torch.softmax((comb_w * scale[2] + comb_b).view(Tn, hc, hc), dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(hc_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    collapsed = (pre.unsqueeze(-1) * streams).sum(1)
    w = torch.cat([pre, post, comb.reshape(Tn, hc * hc)], dim=-1)
    return w, post, comb, collapsed


def kda(x, il):
    """x [T, E] (normed) -> [T, E]; the conv state and recurrent state start at zero (one prompt)"""
    b = f"blk.{il}."
    Tn = x.shape[0]
    qkv = x @ W(b + "kda_qkv.weight").T                       # [T, 3 H S]
    conv = W(b + "kda_conv.weight")                            # [3 H S, K]
    dump(il, "kda_qkv", qkv)
    xpad = torch.cat([torch.zeros(K_conv - 1, qkv.shape[1]), qkv], 0)   # [T+K-1, C]
    co = torch.stack([(xpad[t:t + K_conv] * conv.T).sum(0) for t in range(Tn)])   # causal depthwise conv
    co = torch.nn.functional.silu(co)
    dump(il, "kda_conv_raw", co)
    fgb = x @ W(b + "kda_fgb_a.weight").T                    # [T, 2 S + H]
    f = fgb[:, :S_kda] @ W(b + "kda_f_b.weight").T           # [T, H S]
    z = fgb[:, S_kda:2 * S_kda] @ W(b + "kda_g_b.weight").T
    beta = sigmoid(fgb[:, 2 * S_kda:])                       # [T, H]
    g = (f + W(b + "kda_dt_bias")).view(Tn, H_kda, S_kda)
    g = lower * sigmoid(torch.exp(W(b + "kda_a_log")).view(1, H_kda, 1) * g)
    dump(il, "kda_g", g.reshape(Tn, -1))
    dump(il, "kda_beta", beta)
    q, k, v = co.split([H_kda * S_kda] * 3, dim=-1)
    q = q.view(Tn, H_kda, S_kda)
    k = k.view(Tn, H_kda, S_kda)
    v = v.view(Tn, H_kda, S_kda)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    dump(il, "kda_conv", torch.cat([q.reshape(Tn, -1), k.reshape(Tn, -1), v.reshape(Tn, -1)], -1))
    q = q * S_kda ** -0.5
    state = torch.zeros(H_kda, S_kda, S_kda)
    out = torch.zeros(Tn, H_kda, S_kda)
    for t in range(Tn):
        state = state * torch.exp(g[t])[..., None]
        kv_mem = (state * k[t][..., None]).sum(-2)
        delta = (v[t] - kv_mem) * beta[t][..., None]
        state = state + k[t].unsqueeze(-1) * delta.unsqueeze(-2)
        out[t] = (state * q[t].unsqueeze(-1)).sum(-2)
    dump(il, "kda_o", out.reshape(Tn, -1))
    on = rmsnorm(out, W(b + "kda_o_norm.weight")) * sigmoid(z.view(Tn, H_kda, S_kda))
    dump(il, "kda_normed", on.reshape(Tn, -1))
    return on.reshape(Tn, -1) @ W(b + "kda_out.weight").T


def mla(x, il):
    b = f"blk.{il}."
    Tn = x.shape[0]
    qkv_a = x @ W(b + "attn_qkv_a.weight").T
    qa, c = qkv_a[:, :q_lora], qkv_a[:, q_lora:]
    qr = rmsnorm(qa, W(b + "attn_q_a_norm.weight"))
    c = rmsnorm(c, W(b + "attn_kv_a_norm.weight"))
    dump(il, "mla_qr", torch.cat([qr, c], -1))
    qb = qr @ W(b + "attn_q_b_idx.weight").T                 # [T, H Dk + ih idim]
    dump(il, "mla_qb", qb)
    q = qb[:, :H * Dk].view(Tn, H, Dk)
    kvb = W(b + "attn_kv_b.weight")                           # [H (Dk + Dv), kv_lora]
    kvx = (c @ kvb.T).view(Tn, H, Dk + Dv)
    kk, vv = kvx[..., :Dk], kvx[..., Dk:]
    scores = torch.einsum("thd,shd->hts", q, kk) * Dk ** -0.5
    mask = torch.triu(torch.ones(Tn, Tn, dtype=torch.bool), 1)   # True = hidden
    # the DSA indexer: rows beyond the budget see only their top pools (of kpool cells) and their tail
    ix = x @ W(b + "idx_x.weight").T                              # [T, idim + ih + idim]
    kidx = torch.nn.functional.layer_norm(ix[:, :idim], (idim,), W(b + "idx_k_norm.weight"), W(b + "idx_k_norm.bias"), 1e-6)
    wh = ix[:, idim:idim + ih] * ih ** -0.5
    gate = ix[:, idim + ih:]
    qidx = qb[:, H * Dk:].view(Tn, ih, idim)
    n_pools = Tn // kpool
    dump(il, "idx_k", kidx)
    if n_pools > 0:
        ape = W(b + "idx_ape")                                    # [kpool, idim]
        gk = kidx[:n_pools * kpool].view(n_pools, kpool, idim)
        gg = gate[:n_pools * kpool].view(n_pools, kpool, idim) + ape[None]
        pooled = (torch.softmax(gg, dim=1) * gk).sum(1)           # [n_pools, idim]
        dump(il, "idx_pooled", pooled)
        sc = torch.relu(torch.einsum("thd,pd->thp", qidx, pooled) * idim ** -0.5)
        isc = torch.einsum("th,thp->tp", wh, sc)                  # [T, n_pools]
        dense_cells = topk + kpool - 1
        for t in range(Tn):
            if t + 1 <= dense_cells:
                continue
            nb = (t + 1) // kpool                                 # complete pools before the tail
            k = min(topk // kpool, nb)
            sel = torch.topk(isc[t, :nb], k).indices
            allowed = torch.zeros(Tn, dtype=torch.bool)
            for p_ in sel.tolist():
                allowed[p_ * kpool:(p_ + 1) * kpool] = True
            allowed[nb * kpool:t + 1] = True                      # the tail (always)
            mask[t] |= ~allowed
    scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    o = torch.einsum("hts,shd->thd", p, vv)                   # [T, H, Dv]
    dump(il, "mla_o", o.reshape(Tn, -1))
    # the absorbed latent form bnk computes, for comparison: olat = sum_s p c_s
    dump(il, "mla_olat", torch.einsum("hts,sl->thl", p, c).reshape(Tn, -1))
    return o.reshape(Tn, -1) @ W(b + "attn_output.weight").T


def moe(x, il):
    b = f"blk.{il}."
    Tn = x.shape[0]
    logits = x @ W(b + "ffn_gate_inp.weight").T
    s = sigmoid(logits)
    choice = s + W(b + "ffn_gate_inp.bias")
    idx = torch.topk(choice, k_used, dim=-1).indices
    w = s.gather(1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * route_scale
    dump(il, "route_ids", idx.float())
    dump(il, "route_w", w)
    gu = x @ W(b + "ffn_gate_up_shexp.weight").T
    sh = swiglu(gu[:, :F_exp], gu[:, F_exp:]) @ W(b + "ffn_down_shexp.weight").T
    out = sh.clone()
    cache = {}
    for t in range(Tn):
        for j in range(k_used):
            e = int(idx[t, j])
            if e not in cache:
                cache[e] = (expert(il, "gate", e), expert(il, "up", e), expert(il, "down", e))
            g_, u_, d_ = cache[e]
            h = swiglu(x[t] @ g_.T, x[t] @ u_.T)
            out[t] += w[t, j] * (h @ d_.T)
    return out


def ffn(x, il):
    b = f"blk.{il}."
    gu = x @ W(b + "ffn_gate_up.weight").T
    return swiglu(gu[:, :F_dense], gu[:, F_dense:]) @ W(b + "ffn_down.weight").T


def forward(ids, n_layers):
    emb = W("token_embd.weight")
    x = emb[ids]                                              # [T, E]
    streams = x.unsqueeze(1).expand(-1, hc, -1).contiguous()  # [T, hc, E]
    for il in range(n_layers):
        b = f"blk.{il}."
        w, post, comb, h = mhc(streams, il, "hc_attn")
        dump(il, "hc_attn_w", w)
        dump(il, "hc_attn_in", h)
        h = rmsnorm(h, W(b + "attn_norm.weight"))
        dump(il, "attn_in", h)
        a = mla(h, il) if layer_types[il] == 1 else kda(h, il)
        dump(il, "attn_out", a)
        streams = post.unsqueeze(-1) * a.unsqueeze(1) + torch.matmul(comb.transpose(-1, -2), streams)
        dump(il, "hc_combine", streams.reshape(streams.shape[0], -1))
        w, post, comb, h = mhc(streams, il, "hc_ffn")
        h = rmsnorm(h, W(b + "ffn_norm.weight"))
        dump(il, "ffn_in", h)
        m = moe(h, il) if mlp_types[il] == 1 else ffn(h, il)
        dump(il, "ffn_out", m)
        streams = post.unsqueeze(-1) * m.unsqueeze(1) + torch.matmul(comb.transpose(-1, -2), streams)
        dump(il, "l_last", streams.reshape(streams.shape[0], -1))
        print(f"layer {il} done ({'MLA' if layer_types[il] else 'KDA'}, {'MoE' if mlp_types[il] else 'dense'}); "
              f"residual rms {streams.pow(2).mean().sqrt():.4f}", flush=True)
    x = streams.mean(1)
    x = rmsnorm(x, W("output_norm.weight"))
    return x @ W("output.weight").T


n_layers = n_layer if args.layers < 0 else args.layers
logits = forward(torch.tensor(toks), n_layers)
if n_layers == n_layer:
    top = torch.topk(logits[-1], 10)
    print("next-token top-10:", [(int(i), round(float(v), 3)) for v, i in zip(top.values, top.indices)])
    if args.max_new:
        ids = list(toks)
        for _ in range(args.max_new):
            ids.append(int(torch.argmax(forward(torch.tensor(ids), n_layers)[-1])))
            print("generated:", ids[len(toks):], flush=True)

if args.dump:
    os.makedirs(args.dump, exist_ok=True)
    for k, v in dumps.items():
        np.save(os.path.join(args.dump, k + ".npy"), v)
    print(f"dumped {len(dumps)} tensors of layer {args.dump_layer} to {args.dump}")

if args.compare:
    for k, v in dumps.items():
        p = os.path.join(args.compare, k)
        if not os.path.exists(p):
            continue
        b = np.fromfile(p, dtype=np.float32)
        if b.size > v.size and b.size % (v.size // v.shape[0]) == 0:
            b = b[:v.size]   # bnk appends every window's rows: the prompt's come first
        if b.size != v.size:
            print(f"{k:16s} size mismatch: bnk {b.size} ref {v.size}")
            continue
        b = b.reshape(v.shape)
        a = v.astype(np.float64).ravel()
        bb = b.astype(np.float64).ravel()
        corr = np.corrcoef(a, bb)[0, 1] if a.std() > 0 and bb.std() > 0 else float("nan")
        rel = np.linalg.norm(a - bb) / (np.linalg.norm(a) + 1e-30)
        print(f"{k:16s} corr {corr:.6f}  rel err {rel:.3e}  ref rms {np.sqrt((a * a).mean()):.4g}  bnk rms {np.sqrt((bb * bb).mean()):.4g}")
