#include "inkling/ink_ops.h"

#include <algorithm>
#include <cfloat>
#include <cuda_bf16.h>
#include <cmath>
#include <stdexcept>

#include "kernels/gemv.h"

namespace bnk {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// sum over a block of 256 threads (red: 8 floats of shared memory); every thread gets the result
__device__ __forceinline__ float block_sum256(float v, float * red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = l < 8 ? red[l] : 0.f;
    return warp_sum(t);
}

__global__ void k_rms(const float * x, int64_t ldx, const float * w, float * y, int64_t ldy, int n, float eps) {
    __shared__ float red[8];
    const float * xr = x + blockIdx.x * ldx;
    float * yr = y + blockIdx.x * ldy;
    float ss = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum256(ss, red);
    const float r = rsqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * (w ? w[i] : 1.f);
}

// the depth mix in two passes: per (token, state) its 1/rms and the queries' scores; then the rows, parallel over E
__global__ void k_mix_scores(InkStates st, const float * q, int nq, int E, float * sc) {
    __shared__ float red[8][5];
    const int t = blockIdx.x, s = blockIdx.y;
    const float * x = st.p[s] + (int64_t) t * E;
    float v5[5] = {0, 0, 0, 0, 0};   // the sum of squares, then the queries' dots
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        const float v = x[i];
        v5[0] += v * v;
        for (int j = 0; j < nq; ++j) v5[1 + j] += q[(int64_t) j * E + i] * v;
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        v5[k] = warp_sum(v5[k]);
        if (l == 0) red[w][k] = v5[k];
    }
    __syncthreads();
    if (threadIdx.x < 5) {
        float a = 0;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) a += red[i][threadIdx.x];
        red[0][threadIdx.x] = a;   // each thread sums and writes its own column only
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float * o = sc + ((int64_t) t * kInkMaxStates + s) * 5;
        const float r = rsqrtf(red[0][0] / E + 1e-6f);
        o[0] = r;
        for (int j = 0; j < nq; ++j) o[1 + j] = red[0][1 + j] * r * rsqrtf((float) E);
    }
}

__global__ void k_mix_out(InkStates st, int nq, int T, int E, const float * sc, float * out) {
    __shared__ float w[4][kInkMaxStates];
    const int t = blockIdx.y, S = st.n;
    if (threadIdx.x < nq) {
        const int j = threadIdx.x;
        const float * o = sc + (int64_t) t * kInkMaxStates * 5;
        float m = -FLT_MAX;
        for (int s = 0; s < S; ++s) m = fmaxf(m, o[s * 5 + 1 + j]);
        float z = 0;
        for (int s = 0; s < S; ++s) z += __expf(o[s * 5 + 1 + j] - m);
        for (int s = 0; s < S; ++s) w[j][s] = __expf(o[s * 5 + 1 + j] - m) / z * o[s * 5];
    }
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    float acc[4] = {0, 0, 0, 0};
    for (int s = 0; s < S; ++s) {
        const float v = st.p[s][(int64_t) t * E + i];
        for (int j = 0; j < nq; ++j) acc[j] += w[j][s] * v;
    }
    for (int j = 0; j < nq; ++j) out[((int64_t) j * T + t) * E + i] = acc[j];
}

// one block (1024 threads) per token: every state's sum of squares and query dots in one pass, the softmax, the
// mixed rows, their norms
struct InkNorms { const float * w[4]; };
constexpr int kMixT = 512;
template <int NQ>
__global__ void __launch_bounds__(kMixT) k_mix_norm(InkStates st, const float * q, int T, int E, InkNorms nm,
                                                   float eps, float * out) {
    constexpr int NV = 1 + NQ;   // per state: the sum of squares, then the dots
    constexpr int kPer = 5;      // E <= 2560: a thread's elements, every state's kept in registers
    __shared__ float red[kMixT / 32][kInkMaxStates * NV];
    __shared__ float tot[kInkMaxStates * NV];
    __shared__ float wts[NQ][kInkMaxStates];
    __shared__ float rr[NQ];
    const int t = blockIdx.x, S = st.n, w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const int per = (E + kMixT - 1) / kMixT;
    float xv[kInkMaxStates][kPer];
    // every load first (states x elements in flight), then the arithmetic
#pragma unroll
    for (int s2 = 0; s2 < kInkMaxStates; ++s2)
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int i = threadIdx.x + k * kMixT;
            xv[s2][k] = (s2 < S && k < per && i < E) ? st.p[s2][(int64_t) t * E + i] : 0.f;
        }
    float acc[kInkMaxStates * NV];
#pragma unroll
    for (int i = 0; i < kInkMaxStates * NV; ++i) acc[i] = 0.f;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int i = threadIdx.x + k * kMixT;
        if (k >= per || i >= E) break;
        float qv[NQ];
#pragma unroll
        for (int j = 0; j < NQ; ++j) qv[j] = q[(int64_t) j * E + i];
#pragma unroll
        for (int s2 = 0; s2 < kInkMaxStates; ++s2) {
            const float v = xv[s2][k];
            acc[s2 * NV] += v * v;
#pragma unroll
            for (int j = 0; j < NQ; ++j) acc[s2 * NV + 1 + j] += qv[j] * v;
        }
    }
#pragma unroll
    for (int i = 0; i < kInkMaxStates * NV; ++i) {
        if (i / NV < S) {
            const float v = warp_sum(acc[i]);
            if (l == 0) red[w][i] = v;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < S * NV; i += kMixT) {
        float a = 0;
        for (int k = 0; k < kMixT / 32; ++k) a += red[k][i];
        tot[i] = a;
    }
    __syncthreads();
    if (threadIdx.x < NQ) {
        const int j = threadIdx.x;
        float sc[kInkMaxStates], m = -FLT_MAX;
        for (int s2 = 0; s2 < S; ++s2) {
            const float r = rsqrtf(tot[s2 * NV] / E + 1e-6f);
            sc[s2] = tot[s2 * NV + 1 + j] * r * rsqrtf((float) E);
            m = fmaxf(m, sc[s2]);
        }
        float z = 0;
        for (int s2 = 0; s2 < S; ++s2) z += __expf(sc[s2] - m);
        for (int s2 = 0; s2 < kInkMaxStates; ++s2)
            wts[j][s2] = s2 < S ? __expf(sc[s2] - m) / z * rsqrtf(tot[s2 * NV] / E + 1e-6f) : 0.f;
    }
    __syncthreads();
    float o[NQ][kPer];
    float ss[NQ];
#pragma unroll
    for (int j = 0; j < NQ; ++j) {
        ss[j] = 0.f;
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            float a = 0.f;
#pragma unroll
            for (int s2 = 0; s2 < kInkMaxStates; ++s2) a += wts[j][s2] * xv[s2][k];
            o[j][k] = a;
            ss[j] += a * a;   // elements past E are 0
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NQ; ++j) {
        const float v = warp_sum(ss[j]);
        if (l == 0) red[w][j] = v;
    }
    __syncthreads();
    if (threadIdx.x < NQ) {
        float a = 0;
        for (int i = 0; i < kMixT / 32; ++i) a += red[i][threadIdx.x];
        rr[threadIdx.x] = rsqrtf(a / E + eps);
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int i = threadIdx.x + k * kMixT;
        if (k >= per || i >= E) break;
#pragma unroll
        for (int j = 0; j < NQ; ++j)
            out[((int64_t) j * T + t) * E + i] = nm.w[j] ? o[j][k] * rr[j] * nm.w[j][i] : o[j][k];
    }
}

// one thread per channel, the tokens in order
__global__ void k_conv(const float * x, int64_t ldx, float * y, int T, int C, float * state, int64_t lds,
                       const float * w, const float * dyn, int ld_dyn, int H, float scale, int n_keep) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h0 = state[c], h1 = state[lds + c], h2 = state[2 * lds + c];   // x[-3], x[-2], x[-1]
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const int hd = dyn ? c / (C / H) : 0;
    float k0 = h0, k1 = h1, k2 = h2;   // the state after n_keep rows
    for (int t = 0; t < T; ++t) {
        const float xt = x[(int64_t) t * ldx + c];
        float o = w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt;
        if (dyn) {
            const float * d = dyn + (int64_t) t * ld_dyn + hd * 4;   // lag 0..3: x[t], x[t-1], x[t-2], x[t-3]
            o += d[0] * xt + d[1] * h2 + d[2] * h1 + d[3] * h0;
        }
        y[(int64_t) t * C + c] = (xt + o) * scale;
        h0 = h1; h1 = h2; h2 = xt;
        if (t + 1 == n_keep) { k0 = h0; k1 = h1; k2 = h2; }
    }
    if (n_keep > 0) {
        state[c] = k0; state[lds + c] = k1; state[2 * lds + c] = k2;
    }
}

// prompt chunks: a thread per (channel, token), the state read for the first three rows, advanced afterwards
__global__ void k_conv_par(const float * x, int64_t ldx, float * y, int T, int C, const float * state, int64_t lds,
                           const float * w, const float * dyn, int ld_dyn, int H, float scale) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= C) return;
    float X[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int tt = t - 3 + k;
        X[k] = tt >= 0 ? x[(int64_t) tt * ldx + c] : state[(int64_t) (3 + tt) * lds + c];
    }
    float o = w[c * 4] * X[0] + w[c * 4 + 1] * X[1] + w[c * 4 + 2] * X[2] + w[c * 4 + 3] * X[3];
    if (dyn) {
        const float * d = dyn + (int64_t) t * ld_dyn + (c / (C / H)) * 4;
        o += d[0] * X[3] + d[1] * X[2] + d[2] * X[1] + d[3] * X[0];
    }
    y[(int64_t) t * C + c] = (X[3] + o) * scale;
}

__global__ void k_conv_keep(const float * x, int64_t ldx, int n, int C, float * state, int64_t lds) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h[3] = {state[c], state[lds + c], state[2 * lds + c]};
    for (int t = max(0, n - 3); t < n; ++t) { h[0] = h[1]; h[1] = h[2]; h[2] = x[(int64_t) t * ldx + c]; }
    if (n < 3) {   // fewer than three new rows: the shift above started from the old state's last rows
        float g[3] = {state[c], state[lds + c], state[2 * lds + c]};
        for (int t = 0; t < n; ++t) { g[0] = g[1]; g[1] = g[2]; g[2] = x[(int64_t) t * ldx + c]; }
        h[0] = g[0]; h[1] = g[1]; h[2] = g[2];
    }
    state[c] = h[0]; state[lds + c] = h[1]; state[2 * lds + c] = h[2];
}

__global__ void k_conv_advance(const float * x, float * state, int n, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h[3] = {state[c], state[C + c], state[2 * C + c]};
    for (int t = 0; t < n; ++t) { h[0] = h[1]; h[1] = h[2]; h[2] = x[(int64_t) t * C + c]; }
    state[c] = h[0]; state[C + c] = h[1]; state[2 * C + c] = h[2];
}

// one warp per (t, h) row of D values
__global__ void k_clip_heads(float * x, int T, int H, int D, const float * scale, float thr, const float * tau,
                             int64_t ld) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= T * H) return;
    const int t = row / H, h = row % H;
    float * xr = x + (int64_t) t * ld + (int64_t) h * D;
    float ss = 0;
    for (int i = lane; i < D; i += 32) {
        const float v = xr[i] * (scale ? scale[h * D + i] : 1.f);
        ss += v * v;
    }
    ss = warp_sum(ss);
    const float f = thr / fmaxf(sqrtf(ss), thr) * (tau ? tau[t] : 1.f);
    for (int i = lane; i < D; i += 32) xr[i] = xr[i] * (scale ? scale[h * D + i] : 1.f) * f;
}

__global__ void k_kv_store(const float * k, const float * v, half * K, half * V, int ring, const int * pos0p, int KVD) {
    const int t = blockIdx.y;
    const int64_t cell = (int64_t) ((*pos0p + t) % ring);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < KVD; i += gridDim.x * blockDim.x) {
        K[cell * KVD + i] = __float2half(k[(int64_t) t * KVD + i]);
        V[cell * KVD + i] = __float2half(v[(int64_t) t * KVD + i]);
    }
}

// One block (8 warps) per (head, query). Lane l holds dims 4l..4l+3 (D = 128); warp w takes keys lo + w + 8i.
constexpr int kAttnWarps = 8;
__global__ void __launch_bounds__(256) k_attention(const float * q, const float * R, const float * sink,
        const half * K, const half * V, int ring, const float * rel_bias, int extent, int n_rel, int window,
        const int * pos0p, int H, int KVH, float scale, float * out) {
    const int pos0 = *pos0p;
    extern __shared__ float table[];   // [extent + 1]
    __shared__ float sm[kAttnWarps], sl[kAttnWarps];
    __shared__ float sacc[kAttnWarps][128];
    const int h = blockIdx.x, t = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int D = 128, KVD = KVH * D, kh = h / (H / KVH);
    const int qp = pos0 + t;
    const int lo = window > 0 ? max(0, qp - window + 1) : 0;
    const float sk = sink[(int64_t) t * H + h];
    const float * Rt = R + ((int64_t) t * H + h) * n_rel;
    for (int d = threadIdx.x; d <= extent; d += blockDim.x) {
        float b = 0;
        if (d < extent)
            for (int r = 0; r < n_rel; ++r) b += Rt[r] * rel_bias[(int64_t) d * n_rel + r];
        table[d] = b - sk;
    }
    const float4 q4 = *(const float4 *) (q + ((int64_t) t * H + h) * D + lane * 4);
    __syncthreads();
    float m = -FLT_MAX, l = 0, a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int kp = lo + w; kp <= qp; kp += kAttnWarps) {
        const int64_t cell = kp % ring;
        const half2 * kr = (const half2 *) (K + cell * KVD + kh * D + lane * 4);
        const float2 k01 = __half22float2(kr[0]), k23 = __half22float2(kr[1]);
        float dot = q4.x * k01.x + q4.y * k01.y + q4.z * k23.x + q4.w * k23.y;
        dot = warp_sum(dot);
        const int d = qp - kp;
        const float s = dot * scale + table[d < extent ? d : extent];
        const float mn = fmaxf(m, s);
        const float c = __expf(m - mn), p = __expf(s - mn);
        const half2 * vr = (const half2 *) (V + cell * KVD + kh * D + lane * 4);
        const float2 v01 = __half22float2(vr[0]), v23 = __half22float2(vr[1]);
        l = l * c + p;
        a0 = a0 * c + p * v01.x; a1 = a1 * c + p * v01.y; a2 = a2 * c + p * v23.x; a3 = a3 * c + p * v23.y;
        m = mn;
    }
    if (lane == 0) { sm[w] = m; sl[w] = l; }
    sacc[w][lane * 4] = a0; sacc[w][lane * 4 + 1] = a1; sacc[w][lane * 4 + 2] = a2; sacc[w][lane * 4 + 3] = a3;
    __syncthreads();
    if (threadIdx.x < D) {
        float M = 0.f;   // the sink's zero logit
        for (int i = 0; i < kAttnWarps; ++i) M = fmaxf(M, sm[i]);
        float L = __expf(0.f - M), acc = 0;
        for (int i = 0; i < kAttnWarps; ++i) {
            if (sl[i] == 0) continue;
            const float f = __expf(sm[i] - M);
            L += sl[i] * f;
            acc += sacc[i][threadIdx.x] * f;
        }
        out[((int64_t) t * H + h) * D + threadIdx.x] = acc / L;
    }
}

__global__ void k_mul_sigmoid(float * x, const float * g, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        x[i] *= 1.f / (1.f + __expf(-g[i]));
}
__global__ void k_add(float * y, const float * a, const float * b, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = a[i] + b[i];
}

// one warp per token, n_exp <= 256
__global__ void k_route(const float * logits, const float * bias, int T, int n_exp, int k, float w_scale,
                        int32_t * ids, float * w) {
    const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (t >= T) return;
    float p[8], sel[8];
    const int per = (n_exp + 31) / 32;
    for (int i = 0; i < per; ++i) {
        const int e = lane + 32 * i;
        if (e < n_exp) {
            p[i] = 1.f / (1.f + expf(-logits[(int64_t) t * n_exp + e]));
            sel[i] = p[i] + bias[e];
        } else {
            p[i] = 0; sel[i] = -FLT_MAX;
        }
    }
    float sum = 0, mine_w[8];
    int mine_e[8];
    for (int j = 0; j < k; ++j) {
        float bv = -FLT_MAX;
        int be = 0x7fffffff;
        for (int i = 0; i < per; ++i) {
            const int e = lane + 32 * i;
            if (sel[i] > bv || (sel[i] == bv && e < be)) { bv = sel[i]; be = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (ov > bv || (ov == bv && oe < be)) { bv = ov; be = oe; }
        }
        float pv = 0;
        if ((be & 31) == lane) {
            pv = p[be / 32];
            sel[be / 32] = -FLT_MAX;
        }
        pv = __shfl_sync(0xffffffffu, pv, be & 31);
        mine_e[j] = be;
        mine_w[j] = pv;
        sum += pv;
    }
    if (lane == 0)
        for (int j = 0; j < k; ++j) {
            ids[t * k + j] = mine_e[j];
            w[t * k + j] = mine_w[j] / (sum + 1e-20f) * w_scale;
        }
}

__global__ void k_situ(const float * gu, int64_t ldgu, float * h, int64_t ldh, int F, float beta, float lb, float gs,
                       const float * roww, float wmul) {
    const int i = blockIdx.y;
    const float rw = (roww ? roww[i] : 1.f) * wmul;
    for (int f = blockIdx.x * blockDim.x + threadIdx.x; f < F; f += gridDim.x * blockDim.x) {
        const float g = gu[(int64_t) i * ldgu + f] * gs, u = gu[(int64_t) i * ldgu + F + f] * gs;
        const float a = beta * tanhf(g / beta) * (1.f / (1.f + expf(-g)));
        const float b = lb > 0 ? lb * tanhf(u / lb) : u;
        h[(int64_t) i * ldh + f] = a * b * rw;
    }
}

__global__ void k_gather(const float * x, int64_t ldx, const int32_t * rows, int E, float * y) {
    const int i = blockIdx.y;
    const float * xr = x + (int64_t) rows[i] * ldx;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < E; e += gridDim.x * blockDim.x) y[(int64_t) i * E + e] = xr[e];
}
__global__ void k_moe_sum(float * out, const float * src, const int32_t * inv, int k, int E) {
    const int t = blockIdx.y;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < E; e += gridDim.x * blockDim.x) {
        float acc = 0;
        for (int j = 0; j < k; ++j) acc += src[(int64_t) inv[t * k + j] * E + e];
        out[(int64_t) t * E + e] = acc;
    }
}

__global__ void k_concat_rms2(const float * a, const float * b, float * y, int E, float eps) {
    __shared__ float red[8];
    const int t = blockIdx.x;
    for (int part = 0; part < 2; ++part) {
        const float * x = (part ? b : a) + (int64_t) t * E;
        float ss = 0;
        for (int i = threadIdx.x; i < E; i += blockDim.x) ss += x[i] * x[i];
        ss = block_sum256(ss, red);
        const float r = rsqrtf(ss / E + eps);
        for (int i = threadIdx.x; i < E; i += blockDim.x) y[(int64_t) t * 2 * E + part * E + i] = x[i] * r;
    }
}

__global__ void k_ctrl_combine(float * out, const float * cur, const float * msg, const float * gf, int E) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E; i += gridDim.x * blockDim.x) {
        const int64_t o = (int64_t) t * E + i;
        const float g = gf[(int64_t) t * 2 * E + i], f = gf[(int64_t) t * 2 * E + E + i];
        out[o] = cur[o] - tanhf(f) * cur[o] + tanhf(g) * msg[o];
    }
}

__global__ void k_mul_sigmoid_rows(float * x, const float * g, int64_t ldg, int n) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[(int64_t) t * n + i] *= 1.f / (1.f + __expf(-g[(int64_t) t * ldg + i]));
}

__global__ void k_softcap(float * x, int64_t n, float cap) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        x[i] = cap * tanhf(x[i] / cap);
}

__global__ void k_round_bf16(const float * x, int64_t ldx, int cols, float * y) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cols; i += gridDim.x * blockDim.x)
        y[(int64_t) t * cols + i] = __bfloat162float(__float2bfloat16(x[(int64_t) t * ldx + i]));
}

struct ConvJobs { InkConvJob j[kInkConvJobs]; };

// decode windows: a thread per (channel, job), the tokens in order
__global__ void k_conv_jobs(ConvJobs J, int T, int64_t lds, int64_t ldw, int n_keep) {
    const InkConvJob & jb = J.j[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x, C = jb.C;
    if (c >= C) return;
    float * st = jb.state;
    float h0 = st[c], h1 = st[lds + c], h2 = st[2 * lds + c];
    const float w0 = jb.w[c * 4], w1 = jb.w[c * 4 + 1], w2 = jb.w[c * 4 + 2], w3 = jb.w[c * 4 + 3];
    const int hd = jb.dyn ? c / (C / jb.H) : 0;
    float k0 = h0, k1 = h1, k2 = h2;
    for (int t = 0; t < T; ++t) {
        float xt = jb.x[(int64_t) t * jb.ldx + c];
        if (jb.x2) xt += jb.x2[(int64_t) t * C + c];
        if (jb.win) jb.win[(int64_t) t * ldw + c] = xt;
        float o = w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt;
        if (jb.dyn) {
            const float * d = jb.dyn + (int64_t) t * jb.ld_dyn + hd * 4;
            o += d[0] * xt + d[1] * h2 + d[2] * h1 + d[3] * h0;
        }
        float y = (xt + o) * jb.scale;
        if (jb.res) y += jb.res[(int64_t) t * C + c];
        jb.y[(int64_t) t * C + c] = y;
        if (jb.y2) jb.y2[(int64_t) t * C + c] = y;
        h0 = h1; h1 = h2; h2 = xt;
        if (t + 1 == n_keep) { k0 = h0; k1 = h1; k2 = h2; }
    }
    if (n_keep > 0) { st[c] = k0; st[lds + c] = k1; st[2 * lds + c] = k2; }
}

// prompt chunks: a thread per (channel, token, job); the input sum first (X), the state advanced afterwards
__global__ void k_conv_jobs_sum(ConvJobs J, int T, float * xs, int64_t ldxs) {
    const InkConvJob & jb = J.j[blockIdx.z];
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= jb.C) return;
    float xt = jb.x[(int64_t) t * jb.ldx + c];
    if (jb.x2) xt += jb.x2[(int64_t) t * jb.C + c];
    xs[(int64_t) blockIdx.z * T * ldxs + (int64_t) t * ldxs + c] = xt;
}
__global__ void k_conv_jobs_par(ConvJobs J, int T, const float * xs, int64_t ldxs, int64_t lds) {
    const InkConvJob & jb = J.j[blockIdx.z];
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y, C = jb.C;
    if (c >= C) return;
    const float * x = xs + (int64_t) blockIdx.z * T * ldxs;
    float X[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int tt = t - 3 + k;
        X[k] = tt >= 0 ? x[(int64_t) tt * ldxs + c] : jb.state[(int64_t) (3 + tt) * lds + c];
    }
    float o = jb.w[c * 4] * X[0] + jb.w[c * 4 + 1] * X[1] + jb.w[c * 4 + 2] * X[2] + jb.w[c * 4 + 3] * X[3];
    if (jb.dyn) {
        const float * d = jb.dyn + (int64_t) t * jb.ld_dyn + (c / (C / jb.H)) * 4;
        o += d[0] * X[3] + d[1] * X[2] + d[2] * X[1] + d[3] * X[0];
    }
    float y = (X[3] + o) * jb.scale;
    if (jb.res) y += jb.res[(int64_t) t * C + c];
    jb.y[(int64_t) t * C + c] = y;
    if (jb.y2) jb.y2[(int64_t) t * C + c] = y;
}
__global__ void k_conv_jobs_keep(ConvJobs J, int T, const float * xs, int64_t ldxs, int64_t lds, int n) {
    const InkConvJob & jb = J.j[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= jb.C) return;
    const float * x = xs + (int64_t) blockIdx.y * T * ldxs;
    float g[3] = {jb.state[c], jb.state[lds + c], jb.state[2 * lds + c]};
    for (int t = max(0, n - 3); t < n; ++t) { g[0] = g[1]; g[1] = g[2]; g[2] = x[(int64_t) t * ldxs + c]; }
    if (n < 3) {
        float h[3] = {jb.state[c], jb.state[lds + c], jb.state[2 * lds + c]};
        for (int t = 0; t < n; ++t) { h[0] = h[1]; h[1] = h[2]; h[2] = x[(int64_t) t * ldxs + c]; }
        g[0] = h[0]; g[1] = h[1]; g[2] = h[2];
    }
    jb.state[c] = g[0]; jb.state[lds + c] = g[1]; jb.state[2 * lds + c] = g[2];
}

// a warp per row: q (H rows of D, clipped with q_clip and tau), R (H rows of n_rel, rel_clip and tau), k (KVH rows,
// clipped), v (KVH rows, RMS-normed) of each token
__global__ void k_post_proj(float * q, float * R, int64_t ld, const float * q_clip, const float * rel_clip, float qk_clip,
                            float rel_thr, const float * tau, float * k, float * v, int T, int H, int KVH, int D,
                            int n_rel, float eps) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    const int per = 2 * H + 2 * KVH;
    if (row >= T * per) return;
    const int t = row / per, r = row % per;
    float * x;
    const float * sc = nullptr;
    int n;
    float thr, tf = 1.f;
    bool rms = false;
    if (r < H) { x = q + (int64_t) t * ld + (int64_t) r * D; sc = q_clip + r * D; n = D; thr = qk_clip; if (tau) tf = tau[t]; }
    else if (r < 2 * H) { const int h = r - H; x = R + (int64_t) t * ld + (int64_t) h * n_rel; sc = rel_clip + h * n_rel; n = n_rel; thr = rel_thr; if (tau) tf = tau[t]; }
    else if (r < 2 * H + KVH) { x = k + ((int64_t) t * KVH + (r - 2 * H)) * D; n = D; thr = qk_clip; }
    else { x = v + ((int64_t) t * KVH + (r - 2 * H - KVH)) * D; n = D; thr = 0; rms = true; }
    float ss = 0;
    for (int i = lane; i < n; i += 32) {
        const float y = x[i] * (sc ? sc[i] : 1.f);
        ss += y * y;
    }
    ss = warp_sum(ss);
    const float f = rms ? rsqrtf(ss / n + eps) : thr / fmaxf(sqrtf(ss), thr) * tf;
    for (int i = lane; i < n; i += 32) x[i] = x[i] * (sc ? sc[i] : 1.f) * f;
}

struct CtrlJobs { InkCtrlJob j[kInkCtrlJobs]; };

__global__ void k_ctrl_concat(CtrlJobs J, int E) {
    __shared__ float red[8];
    const InkCtrlJob & jb = J.j[blockIdx.y];
    const int t = blockIdx.x;
    for (int part = 0; part < 2; ++part) {
        const float * x = (part ? jb.msg : jb.cur) + (int64_t) t * E;
        float ss = 0;
        for (int i = threadIdx.x; i < E; i += blockDim.x) ss += x[i] * x[i];
        ss = block_sum256(ss, red);
        const float r = rsqrtf(ss / E + 1e-6f);
        for (int i = threadIdx.x; i < E; i += blockDim.x) jb.in2[(int64_t) t * 2 * E + part * E + i] = x[i] * r;
    }
}

__device__ __forceinline__ void bf16x8(const uint4 q, float * w) {
    const uint32_t u[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) { w[2 * k] = __uint_as_float(u[k] << 16); w[2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u); }
}

// g[t][r] = down[r] . in2[t] (rank rows of 2E): a block (256 threads) per (row, job), the threads over the row's
// 8-value chunks, the tokens' sums reduced in shared memory (a warp a row left 24 blocks for a whole GPU)
template <int NT>
__global__ void k_ctrl_down(CtrlJobs J, int T, int E, int rank) {
    __shared__ float red[8][NT];
    const InkCtrlJob & jb = J.j[blockIdx.y];
    const int r = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int K = 2 * E;
    const uint4 * row = (const uint4 *) ((const uint16_t *) jb.down + (int64_t) r * K);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int c = threadIdx.x; c < K / 8; c += blockDim.x) {
        float wv[8];
        bf16x8(row[c], wv);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t >= T) break;
            const float4 a = *(const float4 *) (jb.in2 + (int64_t) t * K + c * 8);
            const float4 b2 = *(const float4 *) (jb.in2 + (int64_t) t * K + c * 8 + 4);
            acc[t] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w + wv[4] * b2.x + wv[5] * b2.y + wv[6] * b2.z + wv[7] * b2.w;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        if (t >= T) break;
        const float v = warp_sum(acc[t]);
        if (lane == 0) red[w][t] = v;
    }
    __syncthreads();
    if (threadIdx.x < T) {
        float a = 0;
        for (int i = 0; i < 8; ++i) a += red[i][threadIdx.x];
        jb.g[(int64_t) threadIdx.x * rank + r] = a;
    }
}

// gf[t][r] = gfw[r] . g[t] (2E rows of rank): a thread per (row, job), g in shared memory
template <int NT>
__global__ void k_ctrl_up(CtrlJobs J, int T, int E, int rank) {
    extern __shared__ float gs[];   // [T][rank]
    const InkCtrlJob & jb = J.j[blockIdx.y];
    for (int i = threadIdx.x; i < T * rank; i += blockDim.x) gs[i] = jb.g[i];
    __syncthreads();
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= 2 * E) return;
    const uint4 * row = (const uint4 *) ((const uint16_t *) jb.gfw + (int64_t) r * rank);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int c = 0; c < rank / 8; ++c) {
        float w[8];
        bf16x8(row[c], w);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t >= T) break;
            const float * g = gs + t * rank + c * 8;
            acc[t] += w[0] * g[0] + w[1] * g[1] + w[2] * g[2] + w[3] * g[3] + w[4] * g[4] + w[5] * g[5] + w[6] * g[6] + w[7] * g[7];
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t)
        if (t < T) jb.gf[(int64_t) t * 2 * E + r] = acc[t];
}

// out = cur - tanh(f) cur + tanh(g) msg, then (norm) its RMS norm: a block per (token, job)
__global__ void k_ctrl_combine_norm(CtrlJobs J, int E, float eps) {
    __shared__ float red[8];
    const InkCtrlJob & jb = J.j[blockIdx.y];
    const int t = blockIdx.x;
    constexpr int kPer = 16;   // E <= 4096 at 256 threads
    float o[kPer];
    float ss = 0;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        o[k] = 0.f;
        if (i < E) {
            const int64_t x = (int64_t) t * E + i;
            const float g = jb.gf[(int64_t) t * 2 * E + i], f = jb.gf[(int64_t) t * 2 * E + E + i];
            o[k] = jb.cur[x] - tanhf(f) * jb.cur[x] + tanhf(g) * jb.msg[x];
            ss += o[k] * o[k];
        }
    }
    const float r = jb.norm ? rsqrtf(block_sum256(ss, red) / E + eps) : 1.f;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < E) jb.out[(int64_t) t * E + i] = jb.norm ? o[k] * r * jb.norm[i] : o[k];
    }
}

int grid_for(int64_t n, int threads = 256) { return (int) std::min<int64_t>((n + threads - 1) / threads, 4096); }

}  // namespace

void ink_rms(const float * x, int64_t ldx, const float * w, float * y, int64_t ldy, int rows, int n, float eps,
             cudaStream_t s) {
    if (rows <= 0) return;
    k_rms<<<rows, 256, 0, s>>>(x, ldx, w, y, ldy, n, eps);
    check_launch("ink_rms");
}

void ink_mix(const InkStates & st, const float * q, int nq, int T, int E, float * out, float * scratch, cudaStream_t s) {
    if (st.n < 1 || st.n > kInkMaxStates || nq < 1 || nq > 4) throw std::runtime_error("ink_mix: states or queries out of range");
    k_mix_scores<<<dim3(T, st.n), 256, 0, s>>>(st, q, nq, E, scratch);
    k_mix_out<<<dim3((E + 255) / 256, T), 256, 0, s>>>(st, nq, T, E, scratch, out);
    check_launch("ink_mix");
}

void ink_mix_norm(const InkStates & st, const float * q, int nq, int T, int E, const float * const * norm, float eps,
                  float * out, cudaStream_t s) {
    if (st.n < 1 || st.n > kInkMaxStates || nq < 1 || nq > 4 || E > 5 * kMixT) throw std::runtime_error("ink_mix_norm: out of range");
    InkNorms nm{};
    for (int j = 0; j < nq; ++j) nm.w[j] = norm ? norm[j] : nullptr;
    switch (nq) {
        case 1: k_mix_norm<1><<<T, kMixT, 0, s>>>(st, q, T, E, nm, eps, out); break;
        case 2: k_mix_norm<2><<<T, kMixT, 0, s>>>(st, q, T, E, nm, eps, out); break;
        case 3: k_mix_norm<3><<<T, kMixT, 0, s>>>(st, q, T, E, nm, eps, out); break;
        default: k_mix_norm<4><<<T, kMixT, 0, s>>>(st, q, T, E, nm, eps, out); break;
    }
    check_launch("ink_mix_norm");
}

void ink_conv(const float * x, int64_t ldx, float * y, int T, int C, float * state, int64_t ld_state, const float * w,
              const float * dyn, int ld_dyn, int H, float scale, int n_keep, cudaStream_t s) {
    if (T > 16) {
        k_conv_par<<<dim3((C + 127) / 128, T), 128, 0, s>>>(x, ldx, y, T, C, state, ld_state, w, dyn, ld_dyn, H, scale);
        if (n_keep > 0) k_conv_keep<<<(C + 127) / 128, 128, 0, s>>>(x, ldx, n_keep, C, state, ld_state);
    } else {
        k_conv<<<(C + 127) / 128, 128, 0, s>>>(x, ldx, y, T, C, state, ld_state, w, dyn, ld_dyn, H, scale, n_keep);
    }
    check_launch("ink_conv");
}

void ink_conv_advance(const float * x, float * state, int n, int C, cudaStream_t s) {
    if (n <= 0) return;
    k_conv_advance<<<(C + 127) / 128, 128, 0, s>>>(x, state, n, C);
    check_launch("ink_conv_advance");
}

void ink_clip_heads(float * x, int T, int H, int D, const float * scale, float thr, const float * tau, cudaStream_t s,
                    int64_t ld) {
    const int rows = T * H;
    k_clip_heads<<<(rows + 7) / 8, 256, 0, s>>>(x, T, H, D, scale, thr, tau, ld ? ld : (int64_t) H * D);
    check_launch("ink_clip_heads");
}

void ink_kv_store(const float * k, const float * v, half * K, half * V, int ring, const int * pos0, int T, int KVD,
                  cudaStream_t s) {
    k_kv_store<<<dim3((KVD + 255) / 256, T), 256, 0, s>>>(k, v, K, V, ring, pos0, KVD);
    check_launch("ink_kv_store");
}

size_t ink_attention_scratch_floats(int, int, int, int) { return 0; }

void ink_attention(const float * q, const float * R, const float * sink, const half * K, const half * V, int ring,
                   const float * rel_bias, int extent, int n_rel, int window, const int * pos0, int T, int H, int KVH, int D,
                   float scale, float * out, float *, int, cudaStream_t s) {
    if (D != 128) throw std::runtime_error("ink_attention: head dim 128 only");
    const size_t sh = (size_t) (extent + 1) * sizeof(float);
    k_attention<<<dim3(H, T), 32 * kAttnWarps, sh, s>>>(q, R, sink, K, V, ring, rel_bias, extent, n_rel, window,
                                                         pos0, H, KVH, scale, out);
    check_launch("ink_attention");
}

void ink_mul_sigmoid(float * x, const float * g, int64_t n, cudaStream_t s) {
    k_mul_sigmoid<<<grid_for(n), 256, 0, s>>>(x, g, n);
    check_launch("ink_mul_sigmoid");
}
void ink_add(float * y, const float * a, const float * b, int64_t n, cudaStream_t s) {
    k_add<<<grid_for(n), 256, 0, s>>>(y, a, b, n);
    check_launch("ink_add");
}

void ink_route(const float * logits, const float * bias, int T, int n_exp, int k, float w_scale, int32_t * ids,
               float * w, cudaStream_t s) {
    if (n_exp > 256 || k > 8) throw std::runtime_error("ink_route: too many experts");
    k_route<<<(T + 7) / 8, 256, 0, s>>>(logits, bias, T, n_exp, k, w_scale, ids, w);
    check_launch("ink_route");
}

void ink_situ(const float * gu, int64_t ldgu, float * h, int64_t ldh, int n, int F, float beta, float lb, float gs,
              const float * roww, float wmul, cudaStream_t s) {
    if (n <= 0) return;
    k_situ<<<dim3((F + 255) / 256, n), 256, 0, s>>>(gu, ldgu, h, ldh, F, beta, lb, gs, roww, wmul);
    check_launch("ink_situ");
}

void ink_gather(const float * x, int64_t ldx, const int32_t * rows, int n, int E, float * y, cudaStream_t s) {
    if (n <= 0) return;
    k_gather<<<dim3((E + 255) / 256, n), 256, 0, s>>>(x, ldx, rows, E, y);
    check_launch("ink_gather");
}
void ink_moe_sum(float * out, const float * src, const int32_t * inv, int T, int k, int E, cudaStream_t s) {
    if (T <= 0) return;
    k_moe_sum<<<dim3((E + 255) / 256, T), 256, 0, s>>>(out, src, inv, k, E);
    check_launch("ink_moe_sum");
}

void ink_concat_rms2(const float * a, const float * b, float * y, int T, int E, float eps, cudaStream_t s) {
    k_concat_rms2<<<T, 256, 0, s>>>(a, b, y, E, eps);
    check_launch("ink_concat_rms2");
}

void ink_ctrl_combine(float * out, const float * cur, const float * msg, const float * gf, int T, int E, cudaStream_t s) {
    k_ctrl_combine<<<dim3((E + 255) / 256, T), 256, 0, s>>>(out, cur, msg, gf, E);
    check_launch("ink_ctrl_combine");
}

void ink_mul_sigmoid_rows(float * x, const float * g, int64_t ldg, int T, int n, cudaStream_t s) {
    k_mul_sigmoid_rows<<<dim3((n + 255) / 256, T), 256, 0, s>>>(x, g, ldg, n);
    check_launch("ink_mul_sigmoid_rows");
}

void ink_round_bf16(const float * x, int64_t ldx, int T, int cols, float * y, cudaStream_t s) {
    k_round_bf16<<<dim3((cols + 255) / 256, T), 256, 0, s>>>(x, ldx, cols, y);
    check_launch("ink_round_bf16");
}

void ink_conv_jobs(const InkConvJob * jobs, int n, int T, int64_t ld_state, int64_t ld_win, int n_keep, float * xs,
                   cudaStream_t s) {
    if (n < 1 || n > kInkConvJobs) throw std::runtime_error("ink_conv_jobs: jobs");
    ConvJobs J{};
    int C = 0;
    for (int i = 0; i < n; ++i) { J.j[i] = jobs[i]; C = std::max(C, jobs[i].C); }
    if (T <= 16) {
        k_conv_jobs<<<dim3((C + 127) / 128, n), 128, 0, s>>>(J, T, ld_state, ld_win, n_keep);
    } else {
        // the inputs' sums first (the state rows read below are the old ones), then the outputs, then the state
        k_conv_jobs_sum<<<dim3((C + 127) / 128, T, n), 128, 0, s>>>(J, T, xs, C);
        k_conv_jobs_par<<<dim3((C + 127) / 128, T, n), 128, 0, s>>>(J, T, xs, C, ld_state);
        if (n_keep > 0) k_conv_jobs_keep<<<dim3((C + 127) / 128, n), 128, 0, s>>>(J, T, xs, C, ld_state, n_keep);
    }
    check_launch("ink_conv_jobs");
}

void ink_post_proj(float * q, float * R, int64_t ld, const float * q_clip, const float * rel_clip, float qk_clip,
                   float rel_thr, const float * tau, float * k, float * v, int T, int H, int KVH, int D, int n_rel,
                   float eps, cudaStream_t s) {
    const int rows = T * (2 * H + 2 * KVH);
    k_post_proj<<<(rows + 7) / 8, 256, 0, s>>>(q, R, ld, q_clip, rel_clip, qk_clip, rel_thr, tau, k, v, T, H, KVH, D,
                                                n_rel, eps);
    check_launch("ink_post_proj");
}

static CtrlJobs ctrl_jobs(const InkCtrlJob * jobs, int n) {
    if (n < 1 || n > kInkCtrlJobs) throw std::runtime_error("ink_ctrl: jobs");
    CtrlJobs J{};
    for (int i = 0; i < n; ++i) J.j[i] = jobs[i];
    return J;
}
void ink_ctrl_concat(const InkCtrlJob * jobs, int n, int T, int E, cudaStream_t s) {
    k_ctrl_concat<<<dim3(T, n), 256, 0, s>>>(ctrl_jobs(jobs, n), E);
    check_launch("ink_ctrl_concat");
}
void ink_ctrl_down(const InkCtrlJob * jobs, int n, int T, int E, int rank, cudaStream_t s) {
    if (T > 16) throw std::runtime_error("ink_ctrl_down: window");
    const dim3 g(rank, n);
    if (T == 1) k_ctrl_down<1><<<g, 256, 0, s>>>(ctrl_jobs(jobs, n), T, E, rank);
    else if (T <= 4) k_ctrl_down<4><<<g, 256, 0, s>>>(ctrl_jobs(jobs, n), T, E, rank);
    else k_ctrl_down<16><<<g, 256, 0, s>>>(ctrl_jobs(jobs, n), T, E, rank);
    check_launch("ink_ctrl_down");
}
void ink_ctrl_up(const InkCtrlJob * jobs, int n, int T, int E, int rank, cudaStream_t s) {
    if (T > 16 || rank % 8) throw std::runtime_error("ink_ctrl_up: window");
    const dim3 g((2 * E + 127) / 128, n);
    const size_t sh = (size_t) T * rank * 4;
    if (T == 1) k_ctrl_up<1><<<g, 128, sh, s>>>(ctrl_jobs(jobs, n), T, E, rank);
    else if (T <= 4) k_ctrl_up<4><<<g, 128, sh, s>>>(ctrl_jobs(jobs, n), T, E, rank);
    else k_ctrl_up<16><<<g, 128, sh, s>>>(ctrl_jobs(jobs, n), T, E, rank);
    check_launch("ink_ctrl_up");
}
void ink_ctrl_combine_norm(const InkCtrlJob * jobs, int n, int T, int E, float eps, cudaStream_t s) {
    if (E > 16 * 256) throw std::runtime_error("ink_ctrl_combine_norm: width");
    k_ctrl_combine_norm<<<dim3(T, n), 256, 0, s>>>(ctrl_jobs(jobs, n), E, eps);
    check_launch("ink_ctrl_combine_norm");
}

void ink_softcap(float * x, int64_t n, float cap, cudaStream_t s) {
    k_softcap<<<grid_for(n), 256, 0, s>>>(x, n, cap);
    check_launch("ink_softcap");
}

}  // namespace bnk
