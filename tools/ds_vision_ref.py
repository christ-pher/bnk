#!/usr/bin/env python3
"""DeepSeek-V4.1's vision encoder in fp32 numpy (deepseek-ai/DeepSeek-V4.1-Flash-1017 inference/vision.py: the ViT, the
aligner, the image span with its start / newline / end embeddings), from the mmproj file (its tensors are the
checkpoint's, bit for bit), against build/test_vision (src/vision, fp16 weights on the GPU). The image is prepared as
the server prepares it (serve/vision.py deepseek_grid, grey letterboxing).

    tools/ds_vision_ref.py MMPROJ IMAGE [IMAGE...] [--test-vision build/test_vision] [--cache DIR]

--cache keeps each image's reference rows (a large image takes minutes in numpy).

Prints per image: the grid, the encode time, the cosine between our rows and the reference's (mean, worst) and the
worst row's index.
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import gguf
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from serve.vision import deepseek_grid  # noqa: E402


def load(path):
    r = gguf.GGUFReader(path)
    t = {}
    for x in r.tensors:
        raw = np.asarray(x.data)
        f = (raw.view('<u2').astype(np.uint32) << 16).view('<f4') if x.tensor_type == gguf.GGMLQuantizationType.BF16 else raw.astype('<f4')
        t[x.name] = f.reshape([int(d) for d in reversed(x.shape)])
    kv = {f.name: f.contents() for f in r.fields.values() if f.name.startswith("clip.")}
    return t, kv


def rms(x, w, eps):
    return w * (x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)))


def rope_tables(nh, nw, dim, theta):
    inv = 1.0 / theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
    hp = np.repeat(np.arange(nh), nw).astype(np.float32)
    wp = np.tile(np.arange(nw), nh).astype(np.float32)
    f = np.concatenate([hp[:, None] * inv, wp[:, None] * inv], axis=1)   # [N, dim]
    return np.cos(f)[:, None, :], np.sin(f)[:, None, :]


def rotary(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def gelu(x):
    from math import erf
    return 0.5 * x * (1.0 + np.vectorize(erf, otypes=[np.float32])(x / math.sqrt(2.0)))


def encode(t, kv, rgb, H, W):
    P, m = int(kv["clip.vision.patch_size"]), int(kv["clip.vision.projector.scale_factor"])
    E, Hn = int(kv["clip.vision.embedding_length"]), int(kv["clip.vision.attention.head_count"])
    eps = float(kv.get("clip.vision.attention.layer_norm_epsilon", 1e-6))
    nh, nw = H // P, W // P
    x = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5                      # [H, W, 3]
    patches = x.reshape(nh, P, nw, P, 3).transpose(0, 2, 4, 1, 3).reshape(nh * nw, 3 * P * P)
    x = patches @ t["v.patch_embd.weight"].reshape(E, -1).T + t["v.patch_embd.bias"]
    D = E // Hn
    cos, sin = rope_tables(nh, nw, D // 2, 10000.0)
    for il in range(int(kv["clip.vision.block_count"])):
        b = f"v.blk.{il}."
        h = rms(x, t[b + "ln1.weight"], eps)
        qkv = h @ t[b + "attn_qkv.weight"].T + t[b + "attn_qkv.bias"]
        q, k, v = (a.reshape(-1, Hn, D) for a in np.split(qkv, 3, axis=-1))
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        o = np.empty_like(q)
        for hd in range(Hn):
            s = (q[:, hd] @ k[:, hd].T) / math.sqrt(D)
            s = np.exp(s - s.max(axis=-1, keepdims=True))
            o[:, hd] = (s / s.sum(axis=-1, keepdims=True)) @ v[:, hd]
        x = x + o.reshape(-1, E) @ t[b + "attn_out.weight"].T + t[b + "attn_out.bias"]
        h = rms(x, t[b + "ln2.weight"], eps)
        g, u = h @ t[b + "ffn_gate.weight"].T, h @ t[b + "ffn_up.weight"].T
        x = x + ((g / (1.0 + np.exp(-g))) * u) @ t[b + "ffn_down.weight"].T
    x = rms(x, t["v.post_ln.weight"], eps)
    # the aligner: the grid padded to multiples of m with zeros, m x m blocks unfolded as torch's (channel, ky, kx)
    gh, gw = -(-nh // m), -(-nw // m)
    g = np.zeros((gh * m, gw * m, E), np.float32)
    g[:nh, :nw] = x.reshape(nh, nw, E)
    u = g.reshape(gh, m, gw, m, E).transpose(0, 2, 4, 1, 3).reshape(gh * gw, E * m * m)
    y = gelu(u @ t["mm.1.weight"].T + t["mm.1.bias"]) @ t["mm.2.weight"].T + t["mm.2.bias"]
    rows = [t["v.token_embd.img_start"][None]]
    for r in range(gh):
        rows += [y[r * gw:(r + 1) * gw], t["v.image_newline"][None]]
    rows.append(t["v.token_embd.img_end"][None])
    return np.concatenate(rows).astype(np.float32), (gh, gw)


def prepare(path, kv):
    im = Image.open(path)
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
    im = im.convert("RGB")
    _, _, h, w = deepseek_grid(im.width, im.height, int(kv["clip.vision.patch_size"]),
                               int(kv["clip.vision.projector.scale_factor"]), 1024,
                               int(kv.get("clip.vision.image_min_pixels", 544 * 544)))
    if (h, w) != (im.height, im.width):
        im = ImageOps.pad(im, (w, h), color=(127, 127, 127))
    return np.asarray(im, dtype=np.uint8), h, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mmproj")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--test-vision", default=str(ROOT / "build" / "test_vision"))
    ap.add_argument("--cache", help="a folder for the reference rows")
    a = ap.parse_args()
    t, kv = load(a.mmproj)
    for img in a.images:
        rgb, H, W = prepare(img, kv)
        cached = os.path.join(a.cache, f"{Path(img).name}-{W}x{H}.npy") if a.cache else None
        if cached and os.path.exists(cached):
            ref = np.load(cached)
            gh, gw = -(-(H // int(kv["clip.vision.patch_size"])) // 3), -(-(W // int(kv["clip.vision.patch_size"])) // 3)
        else:
            ref, (gh, gw) = encode(t, kv, rgb, H, W)
            if cached:
                os.makedirs(a.cache, exist_ok=True)
                np.save(cached, ref)
        with tempfile.TemporaryDirectory() as d:
            raw, emb = os.path.join(d, "im.rgb"), os.path.join(d, "im.emb")
            rgb.tofile(raw)
            out = subprocess.run([a.test_vision, a.mmproj, raw, str(W), str(H), emb, "2"], capture_output=True, text=True)
            if out.returncode:
                print(out.stdout, out.stderr)
                sys.exit(1)
            ours = np.fromfile(emb, dtype=np.float32).reshape(ref.shape)
        cos = (ours * ref).sum(1) / (np.linalg.norm(ours, axis=1) * np.linalg.norm(ref, axis=1) + 1e-30)
        ms = [l for l in out.stdout.splitlines() if l.startswith("encode")][-1].split(",")[-1].strip()
        print(f"{Path(img).name}: {W}x{H}, grid {gh}x{gw}, {len(ref)} cells, encode {ms}; cosine mean {cos.mean():.5f}, "
              f"worst {cos.min():.5f} (cell {int(cos.argmin())}); rel. error {np.linalg.norm(ours - ref) / np.linalg.norm(ref):.4f}")


if __name__ == "__main__":
    main()
