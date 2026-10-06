// Elementwise / normalization / attention / routing kernels. Activations are fp32, token-major [T][...].
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace bnk {

// ---- hyper-connections
void hc_init(const float * x, float * res, int T, int hc, int E, cudaStream_t s);
// xn[t][s*E+e] = rmsnorm(res[t][s]) * w[s*E+e]
void hc_norm(const float * res, const float * w, float * xn, int T, int hc, int E, float eps, cudaStream_t s);
// lo = silu(lo * scale)
void silu_scale(float * x, int64_t n, float scale, cudaStream_t s);
// mixed[t][e] = mean_s xn[t][s*E+e] * sigmoid(g[t][s*E+e])
void hc_mix(const float * xn, const float * g, float * mixed, int T, int hc, int E, cudaStream_t s);
// res[t][s] += out[t] * 2*sigmoid(inj[t][s]/hc)
void hc_combine(float * res, const float * out, const float * inj, int T, int hc, int E, cudaStream_t s);

// ---- generic
void rmsnorm_rows(const float * x, const float * w, float * y, int rows, int n, int64_t ldx, int64_t ldy, float eps,
                  cudaStream_t s);
void silu_mul(const float * g, const float * u, float * h, int64_t n, cudaStream_t s);
void add_scaled(float * y, const float * x, const float * scale_per_token, int T, int n, cudaStream_t s);
void copy_f32(float * dst, const float * src, int64_t n, cudaStream_t s);
// rows [0, n) of buf (row_elems floats each) = rows [from, from + n): an in-order shift, overlap allowed
void shift_rows(float * buf, int64_t row_elems, int from, int n, cudaStream_t s);
// shift_rows over several buffers of the same row size at once (bufs: a device array of nbuf pointers)
void shift_rows_multi(float * const * bufs, int nbuf, int64_t row_elems, int from, int n, cudaStream_t s);

// ---- gated delta net
// conv_in [T+K-1][C]: the K-1 history rows then the new inputs; out = silu(conv) [T][C]
void gdn_conv(const float * conv_in, const float * w, float * out, int T, int C, int K, cudaStream_t s);
// q/k l2norm, beta = sigmoid(beta), g = softplus(alpha + dt) * a  (in place)
void gdn_prep(float * conv_out, int T, int C, int nk, int nv, int S, float * alpha_g, float * beta, const float * dt,
              const float * a, float eps, cudaStream_t s);
// Runs the delta rule over T tokens from `state` [nv][S][S] (column-major per head: state[h][j*S+i] = S[i][j]).
// Writes outputs when out != null; writes the state reached after `commit` tokens back (0 = keep).
void gdn_recurrence(const float * conv_out, int C, const float * g, const float * beta, float * state, float * out,
                    int T, int nk, int nv, int S, int commit, cudaStream_t s);
// y[t][h][:] = rmsnorm(o[t][h][:]) * w * sigmoid(z[t][h][:])
void gated_rmsnorm(const float * o, const float * z, const float * w, float * y, int T, int H, int S, float eps,
                   cudaStream_t s);

// ---- full attention
// From qfull [T][H][2*D] (q|gate per head) and k/v [T][Hkv][D]: normalizes, rotates (NeoX, n_rot dims) and
// writes q [T][H][D] and the cache rows at positions pos0+t.
void attn_prep(const float * qfull, const float * k, const float * v, const float * qn, const float * kn,
               float * q, half * kcache, half * vcache, int T, int H, int Hkv, int D, int n_rot, float base,
               const int * pos0, float eps, cudaStream_t s);
// out[t][h][:] = softmax(q k^T * scale) v over cache rows [0, pos0+t]   (causal), then *= sigmoid(gate)
// pos0 lives in device memory (graph-friendly); the split count is fixed.
void attention(const float * q, const half * kcache, const half * vcache, const float * qfull_gate, float * out,
               int T, int H, int Hkv, int D, const int * pos0, float scale, float * scratch, cudaStream_t s);
size_t attention_scratch_floats(int T, int H, int D, int max_ctx);
// Prompt attention: rows t < T at positions pos0+t over cache rows [0, pos0+t]; one warp per (t, head).
void attention_prefill(const float * q, const half * kcache, const half * vcache, const float * qfull_gate,
                       float * out, int T, int H, int Hkv, int D, int pos0, float scale, cudaStream_t s);

// ---- MoE routing: softmax over n_exp, top-k, renormalized weights
void route_topk(const float * logits, int T, int n_exp, int k, int32_t * ids, float * w, float w_scale,
                cudaStream_t s);
void sigmoid_inplace(float * x, int64_t n, cudaStream_t s);

// ---- PLE
// gate[t][s] = sigmoid(sgn*sqrt(|sum(knorm*qnorm)/sqrt(E)|)) ; gated[t][s][e] = value[t][e]*gate ;
// normalized = rmsnorm(gated[t][s]) * w_conv;  writes gated and normalized
void ple_gate(const float * key, const float * res, const float * value, const float * w_key, const float * w_query,
              const float * w_conv, float * gated, float * normalized, int T, int hc, int E, float eps,
              cudaStream_t s);
// res += gated + silu(dilated causal conv over hist_and_new [(K-1)*dil + T][C]) with fp16 weights w[c*K+k]
void ple_conv_add(float * res, const float * gated, const float * hist_new, const half * w, int T, int C, int K,
                  int dil, cudaStream_t s);

// ---- head
void argmax_rows(const float * logits, int T, int n, int32_t * out, cudaStream_t s);
// The K largest of each of T rows of n floats (K <= 1024): ids[t][K] and their values vals[t][K], unordered; ties
// at the K-th value go to the lowest indices, so the set is deterministic.
void topk_rows(const float * x, int T, int n, int K, int32_t * ids, float * vals, cudaStream_t s);
// one row: argmax and its softmax probability
void argmax_prob(const float * logits, int n, int32_t * id, float * prob, cudaStream_t s);
// R[t][s][:] += e[t][:]
void add_bcast_streams(float * R, const float * e, int T, int hc, int E, cudaStream_t s);

// dst row i = src row ids[i] (raw bytes, row_bytes a multiple of 16)
void gather_bytes_rows(const uint8_t * src, size_t row_bytes, const int32_t * ids, int n, uint8_t * dst, cudaStream_t s);
// id[0] = table[id[0]]
void map_id(int32_t * id, const int32_t * table, cudaStream_t s);

}  // namespace bnk
