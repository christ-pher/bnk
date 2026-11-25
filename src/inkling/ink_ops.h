// Inkling 2's kernels besides the matrix products (those are bnk's gemv / Gemm). Activations are fp32 rows; the
// K/V caches fp16, indexed by position modulo their ring (the window layers' rings hold window + a prompt chunk,
// the full layers' the whole context).
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace bnk {

constexpr int kInkMaxStates = 10;   // a depth mix's states (the blocks so far and cur)

struct InkStates {
    const float * p[kInkMaxStates];   // each [T][E]
    int n = 0;
};

// y[r] = x[r] / rms(x[r]) * w (w may be null), rows of n
void ink_rms(const float * x, int64_t ldx, const float * w, float * y, int64_t ldy, int rows, int n, float eps,
             cudaStream_t s);
// The depth mix: per token, softmax over the states s of q_j . rms(x_s) / sqrt(E), out_j = sum_s p_js rms(x_s);
// out [nq][T][E]. The last state is cur.
void ink_mix(const InkStates & st, const float * q, int nq, int T, int E, float * out, cudaStream_t s);
// Causal short convolution (4 taps, weight w [C][4], tap 3 the current token) with a residual:
// y[t] = (x[t] + sum_k w[k] x[t-3+k] + sum_lag dyn[t][h][lag] x[t-lag]) * scale, h = c / (C / H) (dyn may be null);
// x rows of stride ldx, y [T][C]; state: 3 rows (stride ld_state) holding the inputs before x[0], oldest first,
// advanced past the first n_keep rows of x (0: left as it is).
void ink_conv(const float * x, int64_t ldx, float * y, int T, int C, float * state, int64_t ld_state, const float * w,
              const float * dyn, int ld_dyn, int H, float scale, int n_keep, cudaStream_t s);
// advances the state (3 rows of C) past the first n rows of x (stride C): a verify window's commit
void ink_conv_advance(const float * x, float * state, int n, int C, cudaStream_t s);
// x[t][h] (D values) = clip(x * scale[h]) * tau[t]: rows scaled to an L2 norm of at most thr (scale, tau optional)
void ink_clip_heads(float * x, int T, int H, int D, const float * scale, float thr, const float * tau, cudaStream_t s);
// writes k and v rows [T][KVD] into the caches' cells (pos0 + t) % ring
void ink_kv_store(const float * k, const float * v, half * K, half * V, int ring, int pos0, int T, int KVD,
                  cudaStream_t s);
// Attention with a relative-position bias and a sink: query t (position pos0 + t) over keys at positions
// [lo, pos0 + t] (lo = pos0 + t - window + 1 for window layers, 0 for window 0), logits
// q.k scale + R_t . rel_bias[min(d, extent)] (0 at d >= extent) - sink_t, plus a zero logit in the softmax.
void ink_attention(const float * q, const float * R, const float * sink, const half * K, const half * V, int ring,
                   const float * rel_bias, int extent, int n_rel, int window, int pos0, int T, int H, int KVH, int D,
                   float scale, float * out, float * scratch, int scratch_cap, cudaStream_t s);
size_t ink_attention_scratch_floats(int T, int H, int D, int max_ctx);
void ink_mul_sigmoid(float * x, const float * g, int64_t n, cudaStream_t s);
void ink_add(float * y, const float * a, const float * b, int64_t n, cudaStream_t s);
// Routing: p = sigmoid(logit), the top k of p + bias, weights p / (sum + 1e-20) * w_scale
void ink_route(const float * logits, const float * bias, int T, int n_exp, int k, float w_scale, int32_t * ids,
               float * w, cudaStream_t s);
// h[i] = situ(gs * g, gs * u) * roww[i] * wmul, g | u the halves of gu rows (2F)
void ink_situ(const float * gu, int64_t ldgu, float * h, int64_t ldh, int n, int F, float beta, float lb, float gs,
              const float * roww, float wmul, cudaStream_t s);
void ink_gather(const float * x, int64_t ldx, const int32_t * rows, int n, int E, float * y, cudaStream_t s);
// out[t] = sum_j src[inv[t * k + j]] (the experts' rows of token t, grouped by expert)
void ink_moe_sum(float * out, const float * src, const int32_t * inv, int T, int k, int E, cudaStream_t s);
// y[t] = [rms(a[t]); rms(b[t])] (2E)
void ink_concat_rms2(const float * a, const float * b, float * y, int T, int E, float eps, cudaStream_t s);
// out = cur - tanh(f) cur + tanh(g) msg
void ink_ctrl_combine(float * out, const float * cur, const float * msg, const float * g, const float * f, int64_t n,
                      cudaStream_t s);
void ink_softcap(float * x, int64_t n, float cap, cudaStream_t s);

}  // namespace bnk
