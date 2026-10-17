// glm5next kernels: mHC hyper-connections, KDA linear attention, MLA attention over a latent cache with the DSA
// indexer's k-pool selection, the sigmoid router. Activations are fp32, token-major [T][...].
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/gemv.h"
#include "kernels/qsa.h"

namespace bnk {

// ---- mHC (manifold-constrained hyper-connections), hc streams of E
// xn[t][:] = res[t][:] / rms(res[t][0 .. hc E)) (one unweighted norm over the flattened streams)
void mhc_norm(const float * res, float * xn, int T, int hc, int E, float eps, cudaStream_t s);
// from the fn projection's outputs f [T][(2 + hc) hc] (pre | post | comb), their bases and the 3 scales:
// w[t] = pre[hc] = sigmoid(f s0 + b) + eps, post[hc] = 2 sigmoid(f s1 + b), comb[hc][hc] = sinkhorn(softmax(f s2 + b)
// + eps) (iters rounds of column then row normalization, the first a column one)
void mhc_weights(const float * f, const float * base, const float * scale, float * w, int T, int hc, float eps, int iters,
                 cudaStream_t s);
// mixed[t][e] = sum_s pre[s] res[t][s][e]
void mhc_collapse(const float * res, const float * w, float * mixed, int T, int hc, int E, cudaStream_t s);
// res[t][j][e] = post[j] out[t][e] + sum_i comb[i][j] res[t][i][e]   (in place)
void mhc_combine(float * res, const float * out, const float * w, int T, int hc, int E, cudaStream_t s);
// x[t][e] = mean_s res[t][s][e]   (the head's collapse)
void hc_mean(const float * res, float * x, int T, int hc, int E, cudaStream_t s);

// ---- KDA (Kimi delta attention): H heads of S, conv channels q | k | v (C = 3 H S)
// l2-normalizes the q and k heads of the conv outputs in place (eps inside the sqrt)
void kda_l2norm(float * co, int T, int C, int H, int S, float eps, cudaStream_t s);
// g[t][h][i] = lower * sigmoid(exp(a_log[h]) (f[t][h][i] + dt[h][i]))   (f: the forget projection's output, in place)
void kda_gate(float * f, const float * dt, const float * a_log, float lower, int T, int H, int S, cudaStream_t s);
// the delta rule with a decay per (head, key channel): state [H][S][S] (state[h][i*S+j] = S[i][j], decayed along i);
// out written when non-null; the state after `commit` tokens written back (0 = keep)
void kda_recurrence(const float * conv_out, int C, const float * g, const float * beta, float * state, float * out,
                    int T, int H, int S, int commit, cudaStream_t s);

// ---- MLA (NoPE), latent cache rows of Dl halves (kv_q8 cells as the KV cache's)
// writes the normalized latent rows c [T][Dl] into the cache at positions pos0 + t
void mla_store_latent(const float * c, int64_t ldc, half * cache, int T, int Dl, const int * pos0, cudaStream_t s);
// absorbed queries: qabs[t][h][:] = scale * sum_d kv_b[h Dk+v + d][:] q[t][h][d]   (kv_b F16 [H (Dk + Dv)][Dl],
// its K rows first per head); q [T][H][Dk] (row stride ldq)
void mla_absorb_q(const float * q, int64_t ldq, const half * kv_b, float * qabs, int T, int H, int Dk, int Dv, int Dl,
                  float scale, cudaStream_t s);
// o[t][h][:] = sum_j kv_b[h (Dk + Dv) + Dk + i][j] olat[t][h][j]
void mla_out(const float * olat, const half * kv_b, float * o, int T, int H, int Dk, int Dv, int Dl, cudaStream_t s);
// attention of T decode rows over the latent cache: dense rows (n_sel[t] == 0) attend to cells [0, pos0 + t], sparse
// ones to the selected pools' cells and the tail (as qsa_attention); olat [T][H][Dl]; scratch as attention()'s with
// D = Dl
void mla_attention(const float * qabs, const half * cache, const int32_t * sel, const int32_t * n_sel, float * olat,
                   int T, int H, int Dl, int r, int K, const int * pos0, float * scratch, cudaStream_t s);
// prompt rows [t0, t1) (one warp per (row, head)), pos0 a host value
void mla_attention_prefill(const float * qabs, const half * cache, const int32_t * sel, const int32_t * n_sel,
                           float * olat, int t0, int t1, int H, int Dl, int r, int K, int pos0, cudaStream_t s);

// ---- DSA indexer with k-pool compression (pools of r cells)
// from the indexer projection ix [T][id + ih + id] (key | head weights | pool gate): LayerNorm(key) (weight, bias)
// and the gate logits as fp16 rows kraw[pos0 + t][2 id]
void dsa_store(const float * ix, int ldx, const float * ln_w, const float * ln_b, half * kraw, int T, int id, int ih,
               const int * pos0, float eps, cudaStream_t s);
// pooled keys of the pools completed within [pos0, pos0 + T): pooled[b][d] = sum_j p[j][d] k[r b + j][d], p the
// softmax over j of gate[r b + j][d] + ape[j][d]
void dsa_pool(const half * kraw, const float * ape, float * pooled, int T, int id, int r, const int * pos0, cudaStream_t s);
// scores[t][b] = sum_h w[t][h] relu(q[t][h] . pooled[b] * id^-0.5) * ih^-0.5 for the complete pools before row t's
// tail (rows that are dense are skipped); q [T][ih][id] (row stride ldq), w [T][ih] (stride ldw)
void dsa_score(const float * q, int64_t ldq, const float * w, int64_t ldw, const float * pooled, float * scores,
               int max_blocks, int T, int ih, int id, int r, int dense_cells, const int * pos0, int t_off, cudaStream_t s);
// the top-K pools of every row from its scores (qsa's radix select): sel[t][K], n_sel[t] (0 = dense row)
void dsa_topk(const float * scores, int max_blocks, int32_t * sel, int32_t * n_sel, int T, int K, int r,
              int dense_cells, const int * pos0, int t_off, cudaStream_t s);

// ---- routing: sigmoid scores, the k best by score + bias, weights = scores normalized (norm) and scaled
void route_sigmoid_topk(const float * logits, const float * bias, int T, int n_exp, int k, int32_t * ids, float * w,
                        float w_scale, bool norm, cudaStream_t s);

// ---- swiglu with the clamp: h = silu(min(g, limit)) * clamp(u, -limit, limit)   (limit <= 0: no clamp)
void swiglu_clamp(const float * g, const float * u, float * h, int64_t n, float limit, cudaStream_t s);
// the same, quantizing h into `a` (gu [T][2 F] rows: g then u)
void quantize_act_swiglu(const float * gu, int64_t ldx, int T, int64_t F, float limit, ActQ8 & a, cudaStream_t s);
// y[t][:] = rmsnorm(x[t][:]) * w   (rows of n, with a bias term optional) -- LayerNorm (mean removed) when `mean`
void layernorm_rows(const float * x, const float * w, const float * b, float * y, int rows, int n, int64_t ldx, int64_t ldy,
                    float eps, cudaStream_t s);

}  // namespace bnk
