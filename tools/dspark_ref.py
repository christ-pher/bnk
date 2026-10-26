"""A NumPy reference of the DSpark draft block, checked against the engine's dump.

    BNK_DS_SPARK_DUMP=/tmp/d BNK_DS_SPARK_DUMP_N=1 build/bnk run --mtp ... (docs/DSPARK.md)
    .venv/bin/python tools/dspark_ref.py /tmp/d.0 [--tensors ~/.cache/bnk/dspark]

Recomputes, from the checkpoint's raw mtp.* tensors (tools/dspark_fetch.py: fp8 with 32x32 e8m0 block scales, fp4
experts with one e8m0 scale per 32, bf16 heads) and the dumped inputs (the target rows fed into the rings, the block's
embeddings), what inference/model.py computes for the block: the ring latents, each stage's attention output,
residual after the attention, FFN input, routed and shared expert outputs, residual after the stage, and the collapsed
state before the head. Prints the relative RMS error of each dumped tensor against the reference, stage by stage.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

E, H, D, QL, R, HC, G, OL, F, NE, K = 5120, 64, 512, 1280, 64, 4, 8, 1024, 2304, 128, 3
EPS, HC_EPS, SINK_ITERS, WIN, ROUTE_SCALE, CLAMP = 1e-20, 1e-6, 20, 128, 1.5, 10.0


class Weights:
    def __init__(self, tdir: str):
        self.tdir = tdir
        self.man = json.load(open(os.path.join(tdir, "manifest.json")))
        self._fp8 = None

    def raw(self, name: str) -> np.ndarray:
        m = self.man[name]
        a = np.fromfile(os.path.join(self.tdir, name + ".bin"), dtype=np.uint8)
        return a.reshape(m["shape"]) if m["dtype"] in ("F8_E4M3", "F8_E8M0", "I8") else a

    def fp8_table(self) -> np.ndarray:
        if self._fp8 is None:
            t = np.zeros(256, np.float32)
            for b in range(256):
                s, e, mnt = (b >> 7) & 1, (b >> 3) & 0xF, b & 7
                if e == 0:
                    v = mnt / 8.0 * 2.0 ** -6
                elif e == 15 and mnt == 7:
                    v = np.nan
                else:
                    v = (1 + mnt / 8.0) * 2.0 ** (e - 7)
                t[b] = -v if s else v
            self._fp8 = t
        return self._fp8

    @staticmethod
    def e8m0(a: np.ndarray) -> np.ndarray:
        return np.exp2(a.astype(np.float32) - 127.0)

    def get(self, name: str) -> np.ndarray:
        """a tensor as float32, dequantized"""
        m = self.man[name]
        dt = m["dtype"]
        if dt == "F32":
            return np.fromfile(os.path.join(self.tdir, name + ".bin"), dtype=np.float32).reshape(m["shape"])
        if dt == "BF16":
            u = np.fromfile(os.path.join(self.tdir, name + ".bin"), dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(m["shape"])
        if dt == "F8_E4M3":
            w = self.fp8_table()[self.raw(name)]
            s = self.e8m0(self.raw(name.replace(".weight", ".scale")))
            rows, cols = w.shape
            bo, bi = (rows + s.shape[0] - 1) // s.shape[0], (cols + s.shape[1] - 1) // s.shape[1]
            sf = np.repeat(np.repeat(s, bo, 0)[:rows], bi, 1)[:, :cols]
            return w * sf
        if dt == "I8":   # fp4 e2m1 pairs, low nibble first, one e8m0 scale per 32 along the input dim
            p = self.raw(name)
            lo, hi = p & 0x0F, (p >> 4) & 0x0F
            codes = np.stack((lo, hi), -1).reshape(p.shape[0], p.shape[1] * 2)
            vals = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)[codes]
            s = self.e8m0(self.raw(name.replace(".weight", ".scale")))
            return vals * np.repeat(s, 32, 1)
        raise ValueError(dt)


def rmsnorm(x, w, eps=EPS):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, pos, inverse=False):
    """x [..., D]: the last R dims rotated at position pos, adjacent pairs as complex numbers (base 10000)"""
    freqs = 1.0 / (10000.0 ** (np.arange(0, R, 2, dtype=np.float64) / R))
    ang = pos * freqs
    if inverse:
        ang = -ang
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    y = x.copy()
    a, b = x[..., -R::2], x[..., -R + 1::2]
    y[..., -R::2] = a * c - b * s
    y[..., -R + 1::2] = a * s + b * c
    return y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def hc_mixes(res, fn, scale, base):
    """res [T][HC][E] -> pre [T][HC], post [T][HC], comb [T][HC][HC] (kernel.py hc_split_sinkhorn)"""
    T = res.shape[0]
    xf = res.reshape(T, HC * E)
    rs = 1.0 / np.sqrt((xf * xf).mean(-1, keepdims=True) + EPS)
    m = (xf @ fn.T) * rs
    pre = sigmoid(m[:, :HC] * scale[0] + base[:HC]) + HC_EPS
    post = 2 * sigmoid(m[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])
    c = (m[:, 2 * HC:] * scale[2] + base[2 * HC:]).reshape(T, HC, HC)
    c = np.exp(c - c.max(-1, keepdims=True))
    c = c / c.sum(-1, keepdims=True) + HC_EPS
    c = c / (c.sum(-2, keepdims=True) + HC_EPS)
    for _ in range(SINK_ITERS - 1):
        c = c / (c.sum(-1, keepdims=True) + HC_EPS)
        c = c / (c.sum(-2, keepdims=True) + HC_EPS)
    return pre, post, c


def hc_pre(res, pre):
    return np.einsum("ts,tse->te", pre, res)


def hc_post(x, res, post, comb):
    return post[:, :, None] * x[:, None, :] + np.einsum("tjk,tje->tke", comb, res)


def rel(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.sqrt(((a - b) ** 2).mean()) / (np.sqrt((b * b).mean()) + 1e-30))


def load(prefix, what, dtype=np.float32):
    fn = f"{prefix}.{what}"
    return np.fromfile(fn, dtype=dtype) if os.path.exists(fn) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="the dump's prefix with its number, e.g. /tmp/d.0")
    ap.add_argument("--tensors", default=os.path.expanduser("~/.cache/bnk/dspark"))
    ap.add_argument("--no-ffn", action="store_true", help="skip the routed experts (slow): compare up to the FFN input")
    a = ap.parse_args()
    W = Weights(a.tensors)
    pre = a.prefix

    anchor, pos = load(pre, "block.hdr", np.int32)
    x = load(pre, "x").reshape(-1, E)
    T = x.shape[0]
    print(f"block: anchor {anchor} at position {pos}, {T} rows")

    # the rings: every fed row -> main_x -> per stage kv latent at its position
    fc, fnorm = W.get("mtp.0.main_proj.weight"), W.get("mtp.0.main_norm.weight")
    rings = [{} for _ in range(3)]
    # every feed of this dump and the earlier ones (the ring accumulates across draft calls): prefix.K.feedN for K <= ours
    base, num = pre.rsplit(".", 1)
    feeds = []
    for k in range(int(num) + 1):
        feeds += glob.glob(f"{base}.{k}.feed*.hdr")
    feeds.sort(key=lambda s: int(s.split(".feed")[1].split(".")[0]))
    nfed = 0
    for hdr in feeds:
        p0, n = np.fromfile(hdr, dtype=np.int32)
        rows = np.fromfile(hdr.replace(".hdr", ".rows"), dtype=np.float32).reshape(n, 3 * E)
        mx = rmsnorm(rows @ fc.T, fnorm)
        for s in range(3):
            wkv, kvn = W.get(f"mtp.{s}.attn.wkv.weight"), W.get(f"mtp.{s}.attn.kv_norm.weight")
            kv = rmsnorm(mx @ wkv.T, kvn)
            for i in range(n):
                rings[s][p0 + i] = rope(kv[i], p0 + i)
        nfed += n
    print(f"fed {nfed} rows into the rings; ring positions {min(rings[0]) if rings[0] else None}..{max(rings[0]) if rings[0] else None}")

    res = np.repeat(x[:, None, :], HC, 1).astype(np.float32)   # [T][HC][E]
    pre_mix = np.zeros((T, HC), np.float32)
    pre_mix[:, 0] = 1.0
    for s in range(3):
        print(f"--- stage {s}")
        pfx = f"mtp.{s}."
        attn_pre, post, comb = hc_mixes(res, W.get(pfx + "hc_attn_fn"), W.get(pfx + "hc_attn_scale"), W.get(pfx + "hc_attn_base"))
        cur = hc_pre(res, pre_mix)
        xn = rmsnorm(cur, W.get(pfx + "attn_norm.weight"))
        qr = rmsnorm(xn @ W.get(pfx + "attn.wq_a.weight").T, W.get(pfx + "attn.q_norm.weight"))
        q = (qr @ W.get(pfx + "attn.wq_b.weight").T).reshape(T, H, D)
        kv = rmsnorm(xn @ W.get(pfx + "attn.wkv.weight").T, W.get(pfx + "attn.kv_norm.weight"))
        for t in range(T):
            q[t] = rope(q[t], pos + t)
            kv[t] = rope(kv[t], pos + t)
        keys = [rings[s][p] for p in range(max(0, pos - WIN), pos) if p in rings[s]]
        keys = np.stack(keys + [kv[t] for t in range(T)])   # [N][D]
        sinks = W.get(pfx + "attn.attn_sink")
        scores = np.einsum("thd,nd->thn", q, keys) / np.sqrt(D)
        mx = scores.max(-1, keepdims=True)
        p = np.exp(scores - mx)
        denom = p.sum(-1, keepdims=True) + np.exp(sinks[None, :, None] - mx)
        o = np.einsum("thn,nd->thd", p / denom, keys)
        for t in range(T):
            o[t] = rope(o[t], pos + t, inverse=True)
        wo_a = W.get(pfx + "attn.wo_a.weight").reshape(G, OL, H * D // G)
        og = o.reshape(T, G, H * D // G)
        oa = np.einsum("tgd,grd->tgr", og, wo_a).reshape(T, G * OL)
        attn_out = oa @ W.get(pfx + "attn.wo_b.weight").T
        d = load(pre, f"attn_out{s}")
        if d is not None:
            print(f"attn_out: rel err {rel(d, attn_out):.4f}  (ref rms {np.sqrt((attn_out**2).mean()):.4f}, ours {np.sqrt((d**2).mean()):.4f})")
        res = hc_post(attn_out, res, post, comb)
        d = load(pre, f"res_attn{s}")
        if d is not None:
            print(f"res after attention: rel err {rel(d, res):.4f}")
        ffn_pre, post, comb = hc_mixes(res, W.get(pfx + "hc_ffn_fn"), W.get(pfx + "hc_ffn_scale"), W.get(pfx + "hc_ffn_base"))
        cur = hc_pre(res, attn_pre)
        xn = rmsnorm(cur, W.get(pfx + "ffn_norm.weight"))
        d = load(pre, f"ffn_in{s}")
        if d is not None:
            print(f"ffn input: rel err {rel(d, xn):.4f}")
        if a.no_ffn:
            return
        # routing
        logits = xn @ W.get(pfx + "ffn.gate.weight").T
        sc = np.sqrt(np.log1p(np.exp(logits)))
        bias = W.get(pfx + "ffn.gate.bias")
        idx = np.argsort(-(sc + bias), -1)[:, :K]
        w = np.take_along_axis(sc, idx, -1)
        w = w / (w.sum(-1, keepdims=True) + 1e-20) * ROUTE_SCALE
        ti = load(pre, f"tasks{s}.ids", np.int32)
        if ti is not None:
            ours = {(int(ti[2 * i]), int(ti[2 * i + 1])) for i in range(len(ti) // 2)}
            ref = {(t, int(idx[t, j])) for t in range(T) for j in range(K)}
            print(f"routing: {len(ours & ref)} of {len(ref)} (token, expert) picks agree")

        def expert(xrow, e, prefix):
            g = xrow @ W.get(f"{prefix}.w1.weight").T
            u = xrow @ W.get(f"{prefix}.w3.weight").T
            u = np.clip(u, -CLAMP, CLAMP)
            g = np.minimum(g, CLAMP)
            h = g / (1 + np.exp(-g)) * u
            return h @ W.get(f"{prefix}.w2.weight").T

        routed = np.zeros((T, E), np.float32)
        for t in range(T):
            for j in range(K):
                routed[t] += w[t, j] * expert(xn[t], idx[t, j], f"{pfx}ffn.experts.{idx[t, j]}")
        d = load(pre, f"routed{s}")
        if d is not None:
            print(f"routed experts: rel err {rel(d, routed):.4f}  (ref rms {np.sqrt((routed**2).mean()):.4f}, ours {np.sqrt((d**2).mean()):.4f})")
        shared = np.stack([expert(xn[t], 0, f"{pfx}ffn.shared_experts") for t in range(T)])
        d = load(pre, f"shared{s}")
        if d is not None:
            print(f"shared expert: rel err {rel(d, shared):.4f}")
        res = hc_post(routed + shared, res, post, comb)
        d = load(pre, f"res{s}")
        if d is not None:
            print(f"res after stage: rel err {rel(d, res):.4f}")
        pre_mix = ffn_pre
    mixed = hc_pre(res, pre_mix)
    d = load(pre, "mixed")
    if d is not None:
        print(f"--- collapsed state before the head: rel err {rel(d, mixed):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
