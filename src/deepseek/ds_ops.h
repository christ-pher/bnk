// DeepSeek-V4.1 kernels: hyper-connection mixes (Sinkhorn), pair RoPE with YaRN, the window + compressed-KV
// attention with sinks, the KV compressor, the indexer (scores, candidate blocks, top-k by radix select), the
// Engram gate, sqrt-softplus routing. Activations fp32, token-major; positions come from device memory
// (`pos0`: the window's first position) so that forwards can be captured as graphs.
#pragma once

#include <cstdint>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/gemv.h"

namespace bnk {

constexpr int kDsRing = 128 + kMaxWindow;   // slots of the sliding-window ring: the window plus a decode window

// rotary inverse frequencies for `n_rot` dims (n_rot/2 pairs); yarn_orig > 0 applies YaRN
std::vector<float> ds_rope_inv_freq(int n_rot, float base, float factor, int yarn_orig, float beta_fast, float beta_slow);

// ---- hyper-connections
// per token: m = fn . rmsnorm(res) (res [T][hc*E] as one vector), then pre = sigmoid(m[0:hc]*s0 + b) + eps,
// post = 2 sigmoid(m[hc:2hc]*s1 + b), comb [hc][hc] = sinkhorn(softmax(m[2hc:]*s2 + b) + eps)
void ds_hc_mixes(const float * res, const float * fn, const float * scale, const float * base, float * pre, float * post,
                 float * comb, int T, int hc, int E, float eps, float hc_eps, int iters, cudaStream_t s);
// out[t][e] = sum_s pre[t][s] * res[t][s*E + e]
void ds_hc_pre(const float * res, const float * pre, float * out, int T, int hc, int E, cudaStream_t s);
// res[t][k][e] = post[t][k] * x[t][e] + sum_j comb[t][j*hc + k] * res[t][j][e]   (in place)
void ds_hc_post(const float * x, float * res, const float * post, const float * comb, int T, int hc, int E, cudaStream_t s);
// pre[t][s] = (s == 0)
void ds_onehot(float * pre, int T, int hc, cudaStream_t s);

// ---- attention
// q [T][H][D]: the last n_rot dims of every head rotated at position pos0 + t (adjacent pairs)
void ds_rope_heads(float * q, int T, int H, int D, int n_rot, const int * pos0, const float * inv_freq, cudaStream_t s);
// kv [T][D] -> rmsnorm * w, rotated at pos0 + t, stored fp16 into ring slot (pos0 + t) % kDsRing; with `origin` >= 0
// the destination is a linear buffer instead: row (pos0 + t - origin) (a prompt chunk's window buffer)
void ds_kv_prep(const float * kv, const float * w, half * ring, int T, int D, int n_rot, const int * pos0,
                const float * inv_freq, float eps, cudaStream_t s, int origin = -1);
// out[t][h][:] over the window (ring slots of positions max(0, p-W+1)..p; with origin >= 0 the rows p - origin of a
// linear buffer) and the selected compressed rows (idx[t][k] into comp [.][D], -1 = none; idx may be null: window
// only), with the per-head sink in the softmax denominator; the output's last n_rot dims get the query's rotation
// removed. H % 8 == 0, D == 512.
void ds_attention(const float * q, const half * ring, const half * comp, const int32_t * idx, int k_sel,
                  const float * sinks, float * out, int T, int H, int D, int W, int n_rot, const int * pos0,
                  const float * inv_freq, float scale, cudaStream_t s, int origin = -1);
// after a prompt chunk: positions [p_from, p_to) of the linear window buffer (row p - origin) into the ring
void ds_ring_fill(const half * win, int origin, half * ring, int p_from, int p_to, int D, cudaStream_t s);
// h[n][f] = silu(min(g, limit)) * clamp(u, -limit, limit) from gu [n][2F] (gate | up); limit <= 0: no clamp
void ds_silu_mul_clamp(const float * gu, float * h, int N, int F, float limit, cudaStream_t s);

// ---- KV compression (kv-source layers)
// From the window's raw projections kv [T][D] / score [T][D] (score unused for ratio 1) and the state (the raw
// row of position pos0 - 1 when pos0 % ratio != 0): every group completing inside the window gets its latent
// (softmax-pooled, rmsnorm * w, rotated at position group * ratio) in ckc[group][D] (fp16) and its index key
// (rmsnorm(idx_k . latent) * idx_k_w, rotated likewise) in ikc[group][ID]. idx_k: fp16 [ID][D].
void ds_compress(const float * kv, const float * score, const float * st_kv, const float * st_sc, const float * w,
                 const half * idx_k, const float * idx_k_w, half * ckc, half * ikc, int T, int ratio, int D, int ID,
                 int n_rot, const int * pos0, const float * inv_freq, float eps, cudaStream_t s);
// after a commit of c rows: the state takes row c - 1 when (pos0 + c) % ratio != 0
void ds_compress_commit(const float * kv, const float * score, float * st_kv, float * st_sc, int c, int ratio, int D,
                        const int * pos0, cudaStream_t s);

// ---- indexer (index-source layers)
// iq [T][IH][ID]: rotate the last n_rot dims at pos0 + t
// scores[t][c] = sum_h relu(iq[t][h] . ik[c]) * iw[t][h] for c < n_vis(t) = (pos0 + t + 1) / ratio, -inf beyond
// (up to n_max rows of the key cache; grid sized for n_max)
void ds_index_scores(const float * iq, const half * ikc, const float * iw, float * scores, int T, int IH, int ID,
                     int ratio, int n_max, int64_t lds, const int * pos0, cudaStream_t s);
// the top `k` of scores[t][0..n_vis) by radix select: idx[t][k] (-1 past n_vis). cand (optional, [T][n_blocks]
// bytes): positions in blocks marked 0 are excluded.
void ds_topk_select(const float * scores, int64_t lds, int T, int ratio, int k, const int * pos0, const uint8_t * cand,
                    int bs, int32_t * idx, cudaStream_t s);
// candidate blocks: block scores (max over bs positions; the block of position n_vis - 1 pinned), the top
// `k_blocks` marked 1 in cand[t][..] (bytes, row stride ldb = (lds + bs - 1) / bs), the rest 0 up to n_blocks
// (the grid, from the context bucket). scratch: [T][ldb] floats.
void ds_cand_blocks(const float * scores, int64_t lds, int T, int ratio, int bs, int k_blocks, int n_blocks,
                    const int * pos0, float * scratch, uint8_t * cand, cudaStream_t s);

// ---- Engram
// key [T][hc][E], value [T][E] (the wkv projection's rows), w [hc][E] (q_weight * k_weight):
// gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)), dot = rmsnorm-weighted key/stream product / sqrt(E);
// res[t][s] += gate * value[t]
void ds_engram_gate(float * res, const float * key, int64_t ldk, const float * value, int64_t ldv, const float * w, int T,
                    int hc, int E, float eps, cudaStream_t s);

// ---- routing: s = sqrt(softplus(logits)); top-k of s + bias; w = s / (sum + 1e-20) * scale (norm: renormalize)
void ds_route(const float * logits, const float * bias, int T, int n_exp, int k, float scale, bool norm, int32_t * ids,
              float * w, cudaStream_t s);

// ---- misc
// y[t][:] = x[t][:] (n floats per row) for rows [0, T): a strided copy helper for the pipeline transfers
void ds_copy_rows(const float * x, int64_t ldx, float * y, int64_t ldy, int T, int n, cudaStream_t s);

}  // namespace bnk
