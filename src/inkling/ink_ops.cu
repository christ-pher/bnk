#include "inkling/ink_ops.h"

#include <cfloat>
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

// one block per token; states S <= kInkMaxStates, queries nq <= 4
__global__ void k_mix(InkStates st, const float * q, int nq, int T, int E, float * out) {
    __shared__ float red[8];
    __shared__ float rms[kInkMaxStates], sc[4][kInkMaxStates];
    const int t = blockIdx.x, S = st.n;
    for (int s = 0; s < S; ++s) {
        const float * x = st.p[s] + (int64_t) t * E;
        float ss = 0, d[4] = {0, 0, 0, 0};
        for (int i = threadIdx.x; i < E; i += blockDim.x) {
            const float v = x[i];
            ss += v * v;
            for (int j = 0; j < nq; ++j) d[j] += q[(int64_t) j * E + i] * v;
        }
        ss = block_sum256(ss, red);
        float dj[4];
        for (int j = 0; j < nq; ++j) dj[j] = block_sum256(d[j], red);
        if (threadIdx.x == 0) {
            const float r = rsqrtf(ss / E + 1e-6f);
            rms[s] = r;
            for (int j = 0; j < nq; ++j) sc[j][s] = dj[j] * r * rsqrtf((float) E);
        }
    }
    __syncthreads();
    if (threadIdx.x < nq) {
        const int j = threadIdx.x;
        float m = -FLT_MAX;
        for (int s = 0; s < S; ++s) m = fmaxf(m, sc[j][s]);
        float z = 0;
        for (int s = 0; s < S; ++s) z += __expf(sc[j][s] - m);
        for (int s = 0; s < S; ++s) sc[j][s] = __expf(sc[j][s] - m) / z * rms[s];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        float acc[4] = {0, 0, 0, 0};
        for (int s = 0; s < S; ++s) {
            const float v = st.p[s][(int64_t) t * E + i];
            for (int j = 0; j < nq; ++j) acc[j] += sc[j][s] * v;
        }
        for (int j = 0; j < nq; ++j) out[((int64_t) j * T + t) * E + i] = acc[j];
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

__global__ void k_conv_advance(const float * x, float * state, int n, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h[3] = {state[c], state[C + c], state[2 * C + c]};
    for (int t = 0; t < n; ++t) { h[0] = h[1]; h[1] = h[2]; h[2] = x[(int64_t) t * C + c]; }
    state[c] = h[0]; state[C + c] = h[1]; state[2 * C + c] = h[2];
}

// one warp per (t, h) row of D values
__global__ void k_clip_heads(float * x, int T, int H, int D, const float * scale, float thr, const float * tau) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= T * H) return;
    const int t = row / H, h = row % H;
    float * xr = x + (int64_t) row * D;
    float ss = 0;
    for (int i = lane; i < D; i += 32) {
        const float v = xr[i] * (scale ? scale[h * D + i] : 1.f);
        ss += v * v;
    }
    ss = warp_sum(ss);
    const float f = thr / fmaxf(sqrtf(ss), thr) * (tau ? tau[t] : 1.f);
    for (int i = lane; i < D; i += 32) xr[i] = xr[i] * (scale ? scale[h * D + i] : 1.f) * f;
}

__global__ void k_kv_store(const float * k, const float * v, half * K, half * V, int ring, int pos0, int KVD) {
    const int t = blockIdx.y;
    const int64_t cell = (int64_t) ((pos0 + t) % ring);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < KVD; i += gridDim.x * blockDim.x) {
        K[cell * KVD + i] = __float2half(k[(int64_t) t * KVD + i]);
        V[cell * KVD + i] = __float2half(v[(int64_t) t * KVD + i]);
    }
}

// One block (8 warps) per (head, query). Lane l holds dims 4l..4l+3 (D = 128); warp w takes keys lo + w + 8i.
constexpr int kAttnWarps = 8;
__global__ void __launch_bounds__(256) k_attention(const float * q, const float * R, const float * sink,
        const half * K, const half * V, int ring, const float * rel_bias, int extent, int n_rel, int window,
        int pos0, int H, int KVH, float scale, float * out) {
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

__global__ void k_ctrl_combine(float * out, const float * cur, const float * msg, const float * g, const float * f,
                               int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        out[i] = cur[i] - tanhf(f[i]) * cur[i] + tanhf(g[i]) * msg[i];
}

__global__ void k_softcap(float * x, int64_t n, float cap) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        x[i] = cap * tanhf(x[i] / cap);
}

int grid_for(int64_t n, int threads = 256) { return (int) std::min<int64_t>((n + threads - 1) / threads, 4096); }

}  // namespace

void ink_rms(const float * x, int64_t ldx, const float * w, float * y, int64_t ldy, int rows, int n, float eps,
             cudaStream_t s) {
    if (rows <= 0) return;
    k_rms<<<rows, 256, 0, s>>>(x, ldx, w, y, ldy, n, eps);
    check_launch("ink_rms");
}

void ink_mix(const InkStates & st, const float * q, int nq, int T, int E, float * out, cudaStream_t s) {
    if (st.n < 1 || st.n > kInkMaxStates || nq < 1 || nq > 4) throw std::runtime_error("ink_mix: states or queries out of range");
    k_mix<<<T, 256, 0, s>>>(st, q, nq, T, E, out);
    check_launch("ink_mix");
}

void ink_conv(const float * x, int64_t ldx, float * y, int T, int C, float * state, int64_t ld_state, const float * w,
              const float * dyn, int ld_dyn, int H, float scale, int n_keep, cudaStream_t s) {
    k_conv<<<(C + 127) / 128, 128, 0, s>>>(x, ldx, y, T, C, state, ld_state, w, dyn, ld_dyn, H, scale, n_keep);
    check_launch("ink_conv");
}

void ink_conv_advance(const float * x, float * state, int n, int C, cudaStream_t s) {
    if (n <= 0) return;
    k_conv_advance<<<(C + 127) / 128, 128, 0, s>>>(x, state, n, C);
    check_launch("ink_conv_advance");
}

void ink_clip_heads(float * x, int T, int H, int D, const float * scale, float thr, const float * tau, cudaStream_t s) {
    const int rows = T * H;
    k_clip_heads<<<(rows + 7) / 8, 256, 0, s>>>(x, T, H, D, scale, thr, tau);
    check_launch("ink_clip_heads");
}

void ink_kv_store(const float * k, const float * v, half * K, half * V, int ring, int pos0, int T, int KVD,
                  cudaStream_t s) {
    k_kv_store<<<dim3((KVD + 255) / 256, T), 256, 0, s>>>(k, v, K, V, ring, pos0, KVD);
    check_launch("ink_kv_store");
}

size_t ink_attention_scratch_floats(int, int, int, int) { return 0; }

void ink_attention(const float * q, const float * R, const float * sink, const half * K, const half * V, int ring,
                   const float * rel_bias, int extent, int n_rel, int window, int pos0, int T, int H, int KVH, int D,
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

void ink_ctrl_combine(float * out, const float * cur, const float * msg, const float * g, const float * f, int64_t n,
                      cudaStream_t s) {
    k_ctrl_combine<<<grid_for(n), 256, 0, s>>>(out, cur, msg, g, f, n);
    check_launch("ink_ctrl_combine");
}

void ink_softcap(float * x, int64_t n, float cap, cudaStream_t s) {
    k_softcap<<<grid_for(n), 256, 0, s>>>(x, n, cap);
    check_launch("ink_softcap");
}

}  // namespace bnk
