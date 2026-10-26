#include "deepseek/ds_ops.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <mma.h>

#include "core/util.h"
#include "kernels/actq.cuh"
#include "kernels/route.cuh"
#include "kernels/ops.h"

namespace bnk {

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// every thread gets the block's total; sh: 32 floats; blockDim.x a multiple of 32
__device__ __forceinline__ float block_sum(float v, float * sh) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    float r = l < nw ? sh[l] : 0.f;
    return warp_sum(r);
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }

// the rotation angle of pair i at `pos`: reduced in double (positions reach a million, fp32 would lose the angle)
__device__ __forceinline__ void rot_sc(int pos, const float * inv_freq, int i, float & sn, float & cs) {
    const double th = fmod((double) pos * (double) inv_freq[i], 6.283185307179586);
    sincosf((float) th, &sn, &cs);
}

}  // namespace

std::vector<float> ds_rope_inv_freq(int n_rot, float base, float factor, int yarn_orig, float beta_fast, float beta_slow) {
    const int half = n_rot / 2;
    std::vector<double> f(half);
    for (int i = 0; i < half; ++i) f[i] = pow((double) base, -2.0 * i / n_rot);
    if (yarn_orig > 0 && factor != 1.f) {
        auto cd = [&](double rot) { return n_rot * log(yarn_orig / (rot * 2 * M_PI)) / (2 * log((double) base)); };
        const int low = std::max((int) floor(cd(beta_fast)), 0), high = std::min((int) ceil(cd(beta_slow)), n_rot - 1);
        for (int i = 0; i < half; ++i) {
            double ramp = ((double) i - low) / std::max((double) (high - low), 1e-3);
            ramp = std::min(1.0, std::max(0.0, ramp));
            const double smooth = 1.0 - ramp;
            f[i] = f[i] / factor * (1.0 - smooth) + f[i] * smooth;
        }
    }
    std::vector<float> out(half);
    for (int i = 0; i < half; ++i) out[i] = (float) f[i];
    return out;
}

// ------------------------------------------------------------------------------------------ hyper-connections
// block per token, 1024 threads: warp j < MIX takes mixer row j, the warps after it the sum of squares
template <int HC>
__global__ void __launch_bounds__(1024) hc_mixes_k(const float * __restrict__ res, const float * __restrict__ fn,
                                                   const float * __restrict__ scale, const float * __restrict__ base,
                                                   float * pre, float * post, float * comb, int E, float eps, float hc_eps,
                                                   int iters) {
    constexpr int MIX = (2 + HC) * HC;
    __shared__ float m[MIX];
    __shared__ float ssh[32];
    const int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const int64_t D = (int64_t) HC * E;
    const float * x = res + (int64_t) t * D;
    if (w < MIX) {
        const float * row = fn + (int64_t) w * D;
        float acc = 0.f;
        for (int64_t i = lane; i < D; i += 32) acc += row[i] * x[i];
        acc = warp_sum(acc);
        if (lane == 0) m[w] = acc;
    } else {
        float ss = 0.f;
        for (int64_t i = (int64_t) (w - MIX) * 32 + lane; i < D; i += (int64_t) (nw - MIX) * 32) ss += x[i] * x[i];
        ss = warp_sum(ss);
        if (lane == 0) ssh[w - MIX] = ss;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ss = 0.f;
        for (int i = 0; i < nw - MIX; ++i) ss += ssh[i];
        const float r = rsqrtf(ss / (float) D + eps);
        float c[HC][HC];
        for (int s = 0; s < HC; ++s) {
            pre[t * HC + s] = sigmoidf_(m[s] * r * scale[0] + base[s]) + hc_eps;
            post[t * HC + s] = 2.f * sigmoidf_(m[HC + s] * r * scale[1] + base[HC + s]);
        }
        for (int j = 0; j < HC; ++j) {
            float mx = -FLT_MAX;
            for (int k = 0; k < HC; ++k) {
                c[j][k] = m[2 * HC + j * HC + k] * r * scale[2] + base[2 * HC + j * HC + k];
                mx = fmaxf(mx, c[j][k]);
            }
            float sum = 0.f;
            for (int k = 0; k < HC; ++k) { c[j][k] = __expf(c[j][k] - mx); sum += c[j][k]; }
            for (int k = 0; k < HC; ++k) c[j][k] = c[j][k] / sum + hc_eps;
        }
        for (int it = 0; it < iters; ++it) {
            if (it > 0)
                for (int j = 0; j < HC; ++j) {
                    float s = 0.f;
                    for (int k = 0; k < HC; ++k) s += c[j][k];
                    for (int k = 0; k < HC; ++k) c[j][k] /= (s + hc_eps);
                }
            for (int k = 0; k < HC; ++k) {
                float s = 0.f;
                for (int j = 0; j < HC; ++j) s += c[j][k];
                for (int j = 0; j < HC; ++j) c[j][k] /= (s + hc_eps);
            }
        }
        for (int j = 0; j < HC; ++j)
            for (int k = 0; k < HC; ++k) comb[(t * HC + j) * HC + k] = c[j][k];
    }
}

// Small windows: one block per token reads the 2 MB mixer matrix alone (85 us on one SM). Instead block (t, j) takes
// mixer row j (j = MIX: the sum of squares) into a scratch of MIX + 1 floats per token, and a second kernel does the
// sigmoids and the Sinkhorn iterations from it.
// the finish from the scratch's MIX + 1 dots of token t (one thread)
template <int HC>
__device__ __forceinline__ void hc_mix_finish(const float * __restrict__ m, const float * __restrict__ scale,
                                              const float * __restrict__ base, float * pre, float * post, float * comb, int t,
                                              int E, float eps, float hc_eps, int iters) {
    constexpr int MIX = (2 + HC) * HC;
    const float r = rsqrtf(__ldcg(m + MIX) / (float) ((int64_t) HC * E) + eps);
    float c[HC][HC];
    for (int s = 0; s < HC; ++s) {
        pre[t * HC + s] = sigmoidf_(__ldcg(m + s) * r * scale[0] + base[s]) + hc_eps;
        post[t * HC + s] = 2.f * sigmoidf_(__ldcg(m + HC + s) * r * scale[1] + base[HC + s]);
    }
    for (int j = 0; j < HC; ++j) {
        float mx = -FLT_MAX;
        for (int k = 0; k < HC; ++k) {
            c[j][k] = __ldcg(m + 2 * HC + j * HC + k) * r * scale[2] + base[2 * HC + j * HC + k];
            mx = fmaxf(mx, c[j][k]);
        }
        float sum = 0.f;
        for (int k = 0; k < HC; ++k) { c[j][k] = __expf(c[j][k] - mx); sum += c[j][k]; }
        for (int k = 0; k < HC; ++k) c[j][k] = c[j][k] / sum + hc_eps;
    }
    for (int it = 0; it < iters; ++it) {
        if (it > 0)
            for (int j = 0; j < HC; ++j) {
                float s = 0.f;
                for (int k = 0; k < HC; ++k) s += c[j][k];
                for (int k = 0; k < HC; ++k) c[j][k] /= (s + hc_eps);
            }
        for (int k = 0; k < HC; ++k) {
            float s = 0.f;
            for (int j = 0; j < HC; ++j) s += c[j][k];
            for (int j = 0; j < HC; ++j) c[j][k] /= (s + hc_eps);
        }
    }
    for (int j = 0; j < HC; ++j)
        for (int k = 0; k < HC; ++k) comb[(t * HC + j) * HC + k] = c[j][k];
}
// cnt: a counter per token (zero between calls): the last of a token's MIX + 1 blocks runs the finish, so the two
// launches became one (the counter's reset is its own)
template <int HC>
__global__ void __launch_bounds__(256) hc_mix_dots_k(const float * __restrict__ res, const float * __restrict__ fn, float * scratch,
                                                     unsigned * cnt, const float * __restrict__ scale, const float * __restrict__ base,
                                                     float * pre, float * post, float * comb, int E, float eps, float hc_eps, int iters) {
    constexpr int MIX = (2 + HC) * HC;
    __shared__ float part[8];
    __shared__ int s_last;
    const int t = blockIdx.x, j = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t D = (int64_t) HC * E;
    const float4 * x = (const float4 *) (res + (int64_t) t * D);
    float acc = 0.f;
    if (j < MIX) {
        const float4 * row = (const float4 *) (fn + (int64_t) j * D);
        for (int64_t i = threadIdx.x; i < D / 4; i += 256) {
            const float4 a = row[i], b = x[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int64_t i = threadIdx.x; i < D / 4; i += 256) {
            const float4 b = x[i];
            acc += b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) part[w] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int i = 0; i < 8; ++i) s += part[i];
        scratch[t * (MIX + 1) + j] = s;
        __threadfence();
        s_last = atomicAdd(&cnt[t], 1u) == (unsigned) MIX;
    }
    __syncthreads();
    if (s_last && threadIdx.x == 0) {
        __threadfence();
        hc_mix_finish<HC>(scratch + t * (MIX + 1), scale, base, pre, post, comb, t, E, eps, hc_eps, iters);
        cnt[t] = 0;
    }
}

void ds_hc_mixes(const float * res, const float * fn, const float * scale, const float * base, float * pre, float * post,
                 float * comb, int T, int hc, int E, float eps, float hc_eps, int iters, cudaStream_t s, float * scratch) {
    if (hc != 4) throw std::runtime_error("ds_hc_mixes: hc must be 4");
    if (scratch && T <= kDsMixSlots && (hc * E) % 4 == 0) {
        unsigned * cnt = (unsigned *) (scratch + kDsMixSlots * 25);
        hc_mix_dots_k<4><<<dim3(T, 25), 256, 0, s>>>(res, fn, scratch, cnt, scale, base, pre, post, comb, E, eps, hc_eps, iters);
        return;
    }
    hc_mixes_k<4><<<T, 1024, 0, s>>>(res, fn, scale, base, pre, post, comb, E, eps, hc_eps, iters);
}

__global__ void hc_pre_k(const float * res, const float * pre, float * out, int hc, int E) {
    const int t = blockIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    float acc = 0.f;
    for (int s = 0; s < hc; ++s) acc += pre[t * hc + s] * res[((int64_t) t * hc + s) * E + e];
    out[(int64_t) t * E + e] = acc;
}
void ds_hc_pre(const float * res, const float * pre, float * out, int T, int hc, int E, cudaStream_t s) {
    hc_pre_k<<<dim3((E + 255) / 256, T), 256, 0, s>>>(res, pre, out, hc, E);
}

// block per token: mixed = sum_s pre[s] res[s] (P float4 a thread, in registers), xn = rmsnorm(mixed) * w, and xn
// quantized (quant4_store: what quantize_act writes, bit for bit). Three launches were one each.
template <int HC, int P>
__global__ void hc_pre_norm_q_k(const float * __restrict__ res, const float * __restrict__ pre, const float * __restrict__ w,
                                float * mixed, float * xn, int E, float eps, ActQ8 q) {
    __shared__ float sh[32];
    const int t = blockIdx.x, n4 = E / 4;
    float pw[HC];
#pragma unroll
    for (int s = 0; s < HC; ++s) pw[s] = pre[t * HC + s];
    float4 v[P];
    float ss = 0.f;
#pragma unroll
    for (int p = 0; p < P; ++p) {
        const int i = threadIdx.x + p * blockDim.x;
        v[p] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (i < n4) {
#pragma unroll
            for (int s = 0; s < HC; ++s) {
                const float4 r = ((const float4 *) (res + ((int64_t) t * HC + s) * E))[i];
                v[p].x += pw[s] * r.x; v[p].y += pw[s] * r.y; v[p].z += pw[s] * r.z; v[p].w += pw[s] * r.w;
            }
            if (mixed) ((float4 *) (mixed + (int64_t) t * E))[i] = v[p];
        }
        ss += v[p].x * v[p].x + v[p].y * v[p].y + v[p].z * v[p].z + v[p].w * v[p].w;
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / E + eps);
#pragma unroll
    for (int p = 0; p < P; ++p) {
        const int i = threadIdx.x + p * blockDim.x;
        if (i < n4) {
            const float4 g = ((const float4 *) w)[i];
            const float4 o = make_float4(v[p].x * inv * g.x, v[p].y * inv * g.y, v[p].z * inv * g.z, v[p].w * inv * g.w);
            ((float4 *) (xn + (int64_t) t * E))[i] = o;
            quant4_store(o, (int64_t) t * E + 4 * i, q);
        }
    }
}
void ds_hc_pre_norm_q(const float * res, const float * pre, const float * w, float * mixed, float * xn, int T, int hc, int E,
                      float eps, ActQ8 & q, cudaStream_t s) {
    const int n4 = E / 4;
    int P = 1;
    while (P < 8 && (n4 + P - 1) / P > 512) P *= 2;
    const int threads = (n4 + P - 1) / P;
    // whole warps over whole quant blocks (every 8-lane group in or out of the row together), else the three launches
    if (hc != 4 || E % 128 || threads % 32 || threads * P != n4) {
        ds_hc_pre(res, pre, mixed, T, hc, E, s);
        rmsnorm_rows(mixed, w, xn, T, E, E, E, eps, s);
        quantize_act(xn, E, T, E, q, s);
        return;
    }
    q.T = T;
    q.cols = q.cols_pad = E;
    switch (P) {
        case 1: hc_pre_norm_q_k<4, 1><<<T, threads, 0, s>>>(res, pre, w, mixed, xn, E, eps, q); break;
        case 2: hc_pre_norm_q_k<4, 2><<<T, threads, 0, s>>>(res, pre, w, mixed, xn, E, eps, q); break;
        case 4: hc_pre_norm_q_k<4, 4><<<T, threads, 0, s>>>(res, pre, w, mixed, xn, E, eps, q); break;
        default: hc_pre_norm_q_k<4, 8><<<T, threads, 0, s>>>(res, pre, w, mixed, xn, E, eps, q);
    }
}

template <int HC>
__global__ void hc_post_k(const float * x, float * res, const float * post, const float * comb, int E) {
    const int t = blockIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    float old[HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) old[j] = res[((int64_t) t * HC + j) * E + e];
    const float xv = x[(int64_t) t * E + e];
#pragma unroll
    for (int k = 0; k < HC; ++k) {
        float v = post[t * HC + k] * xv;
#pragma unroll
        for (int j = 0; j < HC; ++j) v += comb[(t * HC + j) * HC + k] * old[j];
        res[((int64_t) t * HC + k) * E + e] = v;
    }
}
void ds_hc_post(const float * x, float * res, const float * post, const float * comb, int T, int hc, int E, cudaStream_t s) {
    if (hc != 4) throw std::runtime_error("ds_hc_post: hc must be 4");
    hc_post_k<4><<<dim3((E + 255) / 256, T), 256, 0, s>>>(x, res, post, comb, E);
}

__global__ void onehot_k(float * pre, int hc) { pre[blockIdx.x * hc + threadIdx.x] = threadIdx.x == 0 ? 1.f : 0.f; }
void ds_onehot(float * pre, int T, int hc, cudaStream_t s) { onehot_k<<<T, hc, 0, s>>>(pre, hc); }

// ------------------------------------------------------------------------------------------ attention
// grid (T, H), block n_rot/2: pair i of the head's last n_rot dims
__global__ void rope_heads_k(float * q, int H, int D, int n_rot, const int * pos0p, const float * inv_freq) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    float * v = q + ((int64_t) t * H + h) * D + (D - n_rot) + 2 * i;
    float sn, cs;
    rot_sc(*pos0p + t, inv_freq, i, sn, cs);
    const float a = v[0], b = v[1];
    v[0] = a * cs - b * sn;
    v[1] = a * sn + b * cs;
}
void ds_rope_heads(float * q, int T, int H, int D, int n_rot, const int * pos0, const float * inv_freq, cudaStream_t s) {
    rope_heads_k<<<dim3(T, H), n_rot / 2, 0, s>>>(q, H, D, n_rot, pos0, inv_freq);
}

// grid T, block D
__global__ void kv_prep_k(const float * kv, const float * w, half * ring, int D, int n_rot, const int * pos0p,
                          const float * inv_freq, float eps, int origin) {
    __shared__ float sh[32];
    __shared__ float buf[512];
    const int t = blockIdx.x, d = threadIdx.x, pos = *pos0p + t;
    const float x = kv[(int64_t) t * D + d];
    const float ss = block_sum(x * x, sh);
    buf[d] = x * rsqrtf(ss / D + eps) * w[d];
    __syncthreads();
    float out = buf[d];
    const int r0 = D - n_rot;
    if (d >= r0) {
        const int j = d - r0, i = j >> 1;
        float sn, cs;
        rot_sc(pos, inv_freq, i, sn, cs);
        const float a = buf[r0 + 2 * i], b = buf[r0 + 2 * i + 1];
        out = (j & 1) ? a * sn + b * cs : a * cs - b * sn;
    }
    ring[(int64_t) (origin >= 0 ? pos - origin : pos % kDsRing) * D + d] = __float2half(out);
}
void ds_kv_prep(const float * kv, const float * w, half * ring, int T, int D, int n_rot, const int * pos0,
                const float * inv_freq, float eps, cudaStream_t s, int origin) {
    if (D != 512) throw std::runtime_error("ds_kv_prep: D must be 512");
    kv_prep_k<<<T, D, 0, s>>>(kv, w, ring, D, n_rot, pos0, inv_freq, eps, origin);
}

// grid T, block 512: the token's query heads rotated (H * n_rot / 2 pairs over the block), then kv_prep_k's work
__global__ void __launch_bounds__(512) rope_kv_prep_k(float * q, int H, int D, int n_rot, const float * kv, const float * w,
                                                      half * ring, const int * pos0p, const float * inv_freq, float eps, int origin) {
    __shared__ float sh[32];
    __shared__ float buf[512];
    const int t = blockIdx.x, d = threadIdx.x, pos = *pos0p + t;
    const int np = n_rot / 2;
    for (int i = d; i < H * np; i += 512) {
        const int h = i / np, pr = i - h * np;
        float * v = q + ((int64_t) t * H + h) * D + (D - n_rot) + 2 * pr;
        float sn, cs;
        rot_sc(pos, inv_freq, pr, sn, cs);
        const float a = v[0], b = v[1];
        v[0] = a * cs - b * sn;
        v[1] = a * sn + b * cs;
    }
    const float x = kv[(int64_t) t * D + d];
    const float ss = block_sum(x * x, sh);
    buf[d] = x * rsqrtf(ss / D + eps) * w[d];
    __syncthreads();
    float out = buf[d];
    const int r0 = D - n_rot;
    if (d >= r0) {
        const int j = d - r0, i = j >> 1;
        float sn, cs;
        rot_sc(pos, inv_freq, i, sn, cs);
        const float a = buf[r0 + 2 * i], b = buf[r0 + 2 * i + 1];
        out = (j & 1) ? a * sn + b * cs : a * cs - b * sn;
    }
    ring[(int64_t) (origin >= 0 ? pos - origin : pos % kDsRing) * D + d] = __float2half(out);
}
void ds_rope_kv_prep(float * q, int H, const float * kv, const float * w, half * ring, int T, int D, int n_rot, const int * pos0,
                     const float * inv_freq, float eps, cudaStream_t s, int origin) {
    if (D != 512) throw std::runtime_error("ds_rope_kv_prep: D must be 512");
    rope_kv_prep_k<<<T, D, 0, s>>>(q, H, D, n_rot, kv, w, ring, pos0, inv_freq, eps, origin);
}

__global__ void ring_fill_k(const half * win, int origin, half * ring, int p_from, int D) {
    const int p = p_from + blockIdx.x;
    ring[(int64_t) (p % kDsRing) * D + threadIdx.x] = win[(int64_t) (p - origin) * D + threadIdx.x];
}
void ds_ring_fill(const half * win, int origin, half * ring, int p_from, int p_to, int D, cudaStream_t s) {
    if (p_to <= p_from) return;
    ring_fill_k<<<p_to - p_from, D, 0, s>>>(win, origin, ring, p_from, D);
}

__global__ void silu_mul_clamp_k(const float * gu, float * h, int F, float limit) {
    const int n = blockIdx.y;
    for (int f = blockIdx.x * blockDim.x + threadIdx.x; f < F; f += gridDim.x * blockDim.x) {
        float g = gu[(int64_t) n * 2 * F + f], u = gu[(int64_t) n * 2 * F + F + f];
        if (limit > 0.f) { g = fminf(g, limit); u = fminf(fmaxf(u, -limit), limit); }
        h[(int64_t) n * F + f] = g / (1.f + __expf(-g)) * u;
    }
}
void ds_silu_mul_clamp(const float * gu, float * h, int N, int F, float limit, cudaStream_t s) {
    silu_mul_clamp_k<<<dim3((F + 255) / 256, N), 256, 0, s>>>(gu, h, F, limit);
}

// grid (T, H/8), 256 threads: warp = head; lane holds dims [lane*16, lane*16+16) of D = 512
__global__ void __launch_bounds__(256) attention_k(const float * __restrict__ q, const half * __restrict__ ring,
                                                   const half * __restrict__ comp, const int32_t * __restrict__ idx,
                                                   int k_sel, const float * __restrict__ sinks, float * out, int H, int W,
                                                   int n_rot, const int * pos0p, const float * inv_freq, float scale,
                                                   int origin) {
    constexpr int D = 512;
    const int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.y * 8 + w;
    const int pos = *pos0p + t;
    float qv[16], acc[16];
    const float * qp = q + ((int64_t) t * H + h) * D + lane * 16;
#pragma unroll
    for (int d = 0; d < 16; ++d) { qv[d] = qp[d] * scale; acc[d] = 0.f; }
    float m = -1e30f, l = 0.f;
    const int n_win = min(W, pos + 1);
    const int n_keys = n_win + (idx ? k_sel : 0);
    // keys in batches of KB: their rows loaded together (memory parallelism), their dots reduced together (the warp
    // shuffles of independent chains overlap), one online-softmax step per batch. One key at a time ran at 30 ms per
    // 2,048-token layer: the per-key reduction and exponentials serialized.
    constexpr int KB = 8;
    for (int key0 = 0; key0 < n_keys; key0 += KB) {
        uint4 r0[KB], r1[KB];
        bool ok[KB];
        float sd[KB];
#pragma unroll
        for (int j = 0; j < KB; ++j) {
            const int key = key0 + j;
            ok[j] = key < n_keys;
            const half * row = nullptr;
            if (ok[j]) {
                if (key < n_win) {
                    const int p = pos - n_win + 1 + key;
                    row = ring + (int64_t) (origin >= 0 ? p - origin : p % kDsRing) * D;
                } else {
                    const int c = idx[(int64_t) t * k_sel + (key - n_win)];
                    if (c < 0) ok[j] = false;
                    else row = comp + (int64_t) c * D;
                }
            }
            if (ok[j]) {
                const uint4 * rp = (const uint4 *) (row + lane * 16);
                r0[j] = rp[0];
                r1[j] = rp[1];
            } else {
                r0[j] = make_uint4(0, 0, 0, 0);
                r1[j] = r0[j];
            }
        }
#pragma unroll
        for (int j = 0; j < KB; ++j) {
            const half2 * h0 = (const half2 *) &r0[j], * h1 = (const half2 *) &r1[j];
            float d = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 a = __half22float2(h0[i]), b = __half22float2(h1[i]);
                d += qv[2 * i] * a.x + qv[2 * i + 1] * a.y + qv[8 + 2 * i] * b.x + qv[8 + 2 * i + 1] * b.y;
            }
            sd[j] = d;
        }
#pragma unroll
        for (int j = 0; j < KB; ++j) sd[j] = warp_sum(sd[j]);
        float mn = m;
#pragma unroll
        for (int j = 0; j < KB; ++j) if (ok[j]) mn = fmaxf(mn, sd[j]);
        const float corr = __expf(m - mn);
        l *= corr;
#pragma unroll
        for (int d = 0; d < 16; ++d) acc[d] *= corr;
#pragma unroll
        for (int j = 0; j < KB; ++j) {
            if (!ok[j]) continue;
            const float p = __expf(sd[j] - mn);
            l += p;
            const half2 * h0 = (const half2 *) &r0[j], * h1 = (const half2 *) &r1[j];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 a = __half22float2(h0[i]), b = __half22float2(h1[i]);
                acc[2 * i] += p * a.x; acc[2 * i + 1] += p * a.y; acc[8 + 2 * i] += p * b.x; acc[8 + 2 * i + 1] += p * b.y;
            }
        }
        m = mn;
    }
    l += __expf(sinks[h] - m);
    const float inv = 1.f / l;
#pragma unroll
    for (int d = 0; d < 16; ++d) acc[d] *= inv;
    // the query's rotation removed from the last n_rot dims (pairs stay inside a lane: 16 is even)
    const int r0 = D - n_rot;
    if (lane * 16 + 15 >= r0) {
#pragma unroll
        for (int d = 0; d < 16; d += 2) {
            const int dim = lane * 16 + d;
            if (dim >= r0) {
                float sn, cs;
                rot_sc(pos, inv_freq, (dim - r0) >> 1, sn, cs);
                const float a = acc[d], b = acc[d + 1];
                acc[d] = a * cs + b * sn;
                acc[d + 1] = -a * sn + b * cs;
            }
        }
    }
    float * op = out + ((int64_t) t * H + h) * D + lane * 16;
#pragma unroll
    for (int d = 0; d < 16; ++d) op[d] = acc[d];
}
// one key at a time (the default: see ds_attention)
__global__ void __launch_bounds__(256) attention_k1(const float * __restrict__ q, const half * __restrict__ ring,
                                                   const half * __restrict__ comp, const int32_t * __restrict__ idx,
                                                   int k_sel, const float * __restrict__ sinks, float * out, int H, int W,
                                                   int n_rot, const int * pos0p, const float * inv_freq, float scale,
                                                   int origin) {
    constexpr int D = 512;
    const int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.y * 8 + w;
    const int pos = *pos0p + t;
    float qv[16], acc[16];
    const float * qp = q + ((int64_t) t * H + h) * D + lane * 16;
#pragma unroll
    for (int d = 0; d < 16; ++d) { qv[d] = qp[d] * scale; acc[d] = 0.f; }
    float m = -1e30f, l = 0.f;
    const int n_win = min(W, pos + 1);
    const int n_keys = n_win + (idx ? k_sel : 0);
    for (int key = 0; key < n_keys; ++key) {
        const half * row;
        if (key < n_win) {
            const int p = pos - n_win + 1 + key;
            row = ring + (int64_t) (origin >= 0 ? p - origin : p % kDsRing) * D;
        } else {
            const int c = idx[(int64_t) t * k_sel + (key - n_win)];
            if (c < 0) continue;
            row = comp + (int64_t) c * D;
        }
        const uint4 * rp = (const uint4 *) (row + lane * 16);
        const uint4 r0 = rp[0], r1 = rp[1];
        const half2 * h0 = (const half2 *) &r0, * h1 = (const half2 *) &r1;
        float kf[16];
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const float2 a = __half22float2(h0[d]), b = __half22float2(h1[d]);
            kf[2 * d] = a.x; kf[2 * d + 1] = a.y; kf[8 + 2 * d] = b.x; kf[8 + 2 * d + 1] = b.y;
        }
        float sdot = 0.f;
#pragma unroll
        for (int d = 0; d < 16; ++d) sdot += qv[d] * kf[d];
        sdot = warp_sum(sdot);
        const float mn = fmaxf(m, sdot);
        const float corr = __expf(m - mn), p = __expf(sdot - mn);
        l = l * corr + p;
#pragma unroll
        for (int d = 0; d < 16; ++d) acc[d] = acc[d] * corr + p * kf[d];
        m = mn;
    }
    l += __expf(sinks[h] - m);
    const float inv = 1.f / l;
#pragma unroll
    for (int d = 0; d < 16; ++d) acc[d] *= inv;
    // the query's rotation removed from the last n_rot dims (pairs stay inside a lane: 16 is even)
    const int r0 = D - n_rot;
    if (lane * 16 + 15 >= r0) {
#pragma unroll
        for (int d = 0; d < 16; d += 2) {
            const int dim = lane * 16 + d;
            if (dim >= r0) {
                float sn, cs;
                rot_sc(pos, inv_freq, (dim - r0) >> 1, sn, cs);
                const float a = acc[d], b = acc[d + 1];
                acc[d] = a * cs + b * sn;
                acc[d + 1] = -a * sn + b * cs;
            }
        }
    }
    float * op = out + ((int64_t) t * H + h) * D + lane * 16;
#pragma unroll
    for (int d = 0; d < 16; ++d) op[d] = acc[d];
}
// Small windows: a block per (token, head), its 8 warps each taking every 8th key (one key at a time as above), the
// eight partial softmax states merged through shared memory. One key at a time with 8 blocks for a token kept 8 SMs
// busy for 0.5 ms per layer; this keeps 512.
__global__ void __launch_bounds__(256) attention_split_k(const float * __restrict__ q, const half * __restrict__ ring,
                                                         const half * __restrict__ comp, const int32_t * __restrict__ idx,
                                                         int k_sel, const float * __restrict__ sinks, float * out, int H, int W,
                                                         int n_rot, const int * pos0p, const float * inv_freq, float scale,
                                                         int origin, int blk = 0) {
    constexpr int D = 512, NW = 8;
    __shared__ float s_acc[NW][D];
    __shared__ float s_m[NW], s_l[NW];
    const int t = blockIdx.x, h = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pos = *pos0p + t;
    float qv[16], acc[16];
    const float * qp = q + ((int64_t) t * H + h) * D + lane * 16;
#pragma unroll
    for (int d = 0; d < 16; ++d) { qv[d] = qp[d] * scale; acc[d] = 0.f; }
    float m = -1e30f, l = 0.f;
    // blk > 0 (the DSpark drafter's block): every row sees the W positions before the block and all blk block rows
    const int p_first = blk ? max(0, *pos0p - W) : pos - min(W, pos + 1) + 1;
    const int n_win = blk ? *pos0p + blk - p_first : min(W, pos + 1);
    const int n_keys = n_win + (idx ? k_sel : 0);
    for (int key = w; key < n_keys; key += NW) {
        const half * row;
        if (key < n_win) {
            const int p = p_first + key;
            row = ring + (int64_t) (origin >= 0 ? p - origin : p % kDsRing) * D;
        } else {
            const int c = idx[(int64_t) t * k_sel + (key - n_win)];
            if (c < 0) continue;
            row = comp + (int64_t) c * D;
        }
        const uint4 * rp = (const uint4 *) (row + lane * 16);
        const uint4 r0 = rp[0], r1 = rp[1];
        const half2 * h0 = (const half2 *) &r0, * h1 = (const half2 *) &r1;
        float kf[16];
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const float2 a = __half22float2(h0[d]), b = __half22float2(h1[d]);
            kf[2 * d] = a.x; kf[2 * d + 1] = a.y; kf[8 + 2 * d] = b.x; kf[8 + 2 * d + 1] = b.y;
        }
        float sdot = 0.f;
#pragma unroll
        for (int d = 0; d < 16; ++d) sdot += qv[d] * kf[d];
        sdot = warp_sum(sdot);
        const float mn = fmaxf(m, sdot);
        const float corr = __expf(m - mn), p = __expf(sdot - mn);
        l = l * corr + p;
#pragma unroll
        for (int d = 0; d < 16; ++d) acc[d] = acc[d] * corr + p * kf[d];
        m = mn;
    }
#pragma unroll
    for (int d = 0; d < 16; ++d) s_acc[w][lane * 16 + d] = acc[d];
    if (lane == 0) { s_m[w] = m; s_l[w] = l; }
    __syncthreads();
    if (w == 0) {
        float M = -1e30f;
        for (int i = 0; i < NW; ++i) M = fmaxf(M, s_m[i]);
        float L = 0.f, f[NW];
        for (int i = 0; i < NW; ++i) { f[i] = s_l[i] > 0.f ? __expf(s_m[i] - M) : 0.f; L += s_l[i] * f[i]; }
#pragma unroll
        for (int d = 0; d < 16; ++d) {
            float a = 0.f;
            for (int i = 0; i < NW; ++i) a += s_acc[i][lane * 16 + d] * f[i];
            acc[d] = a;
        }
        L += __expf(sinks[h] - M);
        const float inv = 1.f / L;
#pragma unroll
        for (int d = 0; d < 16; ++d) acc[d] *= inv;
        const int r0 = D - n_rot;
        if (lane * 16 + 15 >= r0) {
#pragma unroll
            for (int d = 0; d < 16; d += 2) {
                const int dim = lane * 16 + d;
                if (dim >= r0) {
                    float sn, cs;
                    rot_sc(pos, inv_freq, (dim - r0) >> 1, sn, cs);
                    const float a = acc[d], b = acc[d + 1];
                    acc[d] = a * cs + b * sn;
                    acc[d + 1] = -a * sn + b * cs;
                }
            }
        }
        float * op = out + ((int64_t) t * H + h) * D + lane * 16;
#pragma unroll
        for (int d = 0; d < 16; ++d) op[d] = acc[d];
    }
}

// ------------------------------------------------------------------------------------------ attention on the tensor cores
// A prompt chunk: a block takes 32 heads of one token (the 64 query heads share the key rows: one KV head), 512 threads.
// The rows (window + selected, at most kTcMax) go by in tiles of 32, staged once in shared memory and used for both
// products: S = Q K^T (wmma 16x16x16; a warp a (head tile, row tile, k-quarter), the four partials summed in the
// softmax), the online softmax per head in fp32 (the running max's correction applied to the O accumulators through
// the fragment row map read at the start), O += P V (a warp a head tile x 64 columns, 4 accumulators). fp16's range: a
// head's query is scaled by a power of two to a largest value of ~2^12 (exact), the probabilities by 2^14; the key rows
// are fp16 already; fp32 accumulators. Logit error ~1e-4 absolute from the fp16 rounding of q (the three-phase
// high + low design of Project Maya's sparse_attn_v7_kernel, which this follows, was shared-memory bound: 136 block
// syncs per block; this has ~80 with twice the heads a block). One block per SM (85 KB of shared memory).
constexpr int kTcThreads = 512, kTcMax = 640, kTcHeads = 32, kTcRows = 32, kTcLd = 520, kTcLdP = kTcRows + 8;
constexpr float kTcPScale = 16384.f;
constexpr size_t kTcQ = (size_t) kTcHeads * kTcLd * 2;            // Q (fp16)
constexpr size_t kTcK = (size_t) kTcRows * kTcLd * 2;             // the tile's key rows (fp16)
constexpr size_t kTcS = (size_t) 16 * 16 * 16 * 4;                 // 16 partial S tiles (fp32); the row map at the start
constexpr size_t kTcP = (size_t) kTcHeads * kTcLdP * 2;           // P (fp16)
constexpr size_t kTcSmem = kTcQ + kTcK + kTcS + kTcP;
static_assert(kTcQ + kTcK >= (size_t) kTcHeads * 512 * 4, "O at the end fits the Q + K regions");
static_assert(kTcSmem <= 90 * 1024, "a V100's shared memory (96 KB) with room for the static arrays");

__global__ void __launch_bounds__(kTcThreads, 1) attention_tc_k(const float * __restrict__ q, const half * __restrict__ ring,
                                                                const half * __restrict__ comp, const int32_t * __restrict__ idx,
                                                                int k_sel, const float * __restrict__ sinks, float * out, int H,
                                                                int W, int n_rot, const int * pos0p, const float * inv_freq,
                                                                float scale, int origin) {
    namespace wm = nvcuda::wmma;
    constexpr int D = 512;
    extern __shared__ __align__(32) unsigned char tc_smem[];
    half * Qs = (half *) tc_smem;
    half * Ks = (half *) (tc_smem + kTcQ);
    float * Ss = (float *) (tc_smem + kTcQ + kTcK);
    half * Ps = (half *) (tc_smem + kTcQ + kTcK + kTcS);
    float * Os = (float *) tc_smem;   // the end
    __shared__ float s_qf[kTcHeads], s_corr[kTcHeads], s_den[kTcHeads];
    __shared__ int32_t s_sel[kTcMax];   // the token's selected rows (a fetch resolves its row without a global lookup)
    const int t = blockIdx.y, hb = blockIdx.x * kTcHeads;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int pos = *pos0p + t;
    const int n_win = min(W, pos + 1);
    const int total = n_win + (idx ? k_sel : 0);
    if (idx) for (int j = tid; j < k_sel; j += kTcThreads) s_sel[j] = idx[(int64_t) t * k_sel + j];
    auto row_of = [&](int j) -> const half * {   // the row of key j (nullptr: none)
        if (j >= total) return nullptr;
        if (j < n_win) {
            const int p = pos - n_win + 1 + j;
            return ring + (int64_t) (origin >= 0 ? p - origin : p % kDsRing) * D;
        }
        const int c = s_sel[j - n_win];
        return c < 0 ? nullptr : comp + (int64_t) c * D;
    };
    // the queries of heads `warp` and `warp + 16`: each to a largest value of ~2^12 (exact), as fp16
    for (int hh = warp; hh < kTcHeads; hh += 16) {
        const float * qr = q + ((int64_t) t * H + hb + hh) * D;
        float a = 0.f;
        for (int c = lane; c < D; c += 32) a = fmaxf(a, fabsf(qr[c]));
        a = warp_max(a);
        const int e = a > 0.f ? (int) ((__float_as_uint(a) >> 23) & 0xffu) - 127 : 0;
        const float up = ldexpf(1.f, 12 - e);
        for (int c = lane; c < D; c += 32) Qs[hh * kTcLd + c] = __float2half_rn(qr[c] * up);
        if (lane == 0) s_qf[hh] = scale / up;
    }
    // the accumulator fragment's row map: element i of every accumulator fragment lies in row rowmap[i]
    for (int i = tid; i < 256; i += kTcThreads) Ss[i] = (float) (i >> 4);
    __syncthreads();
    int rowmap[8];
    {
        wm::fragment<wm::accumulator, 16, 16, 16, float> f;
        wm::load_matrix_sync(f, Ss, 16, wm::mem_row_major);
#pragma unroll
        for (int i = 0; i < 8; ++i) rowmap[i] = (int) f.x[i];
    }
    // this warp's parts: S partial (head tile sh, row tile sr, dims [sk, sk + 128)); O (head tile oh, columns oc..oc+64)
    const int sh = warp & 1, sr = (warp >> 1) & 1, sk = (warp >> 2) * 128;
    const int oh = warp & 1, oc = (warp >> 1) * 64;
    wm::fragment<wm::accumulator, 16, 16, 16, float> o[4];
#pragma unroll
    for (int f = 0; f < 4; ++f) wm::fill_fragment(o[f], 0.f);
    // the softmax state of head `hh` (lanes 0-15: head 2 warp, lanes 16-31: head 2 warp + 1), rows lane & 15 and + 16
    const int hh = 2 * warp + (lane >> 4), lr = lane & 15;
    float m = -INFINITY, l = 0.f;
    // the tile's rows: a thread row tid / 16, dims (tid & 15) * 32; the next tile's fetched during this tile's products
    const int fr = tid >> 4, fv = (tid & 15) * 32;
    uint4 nxt[4];
    auto fetch = [&](int j) {
        const half * p = row_of(j);
        if (p) {
            const uint4 * s4 = (const uint4 *) (p + fv);
#pragma unroll
            for (int i = 0; i < 4; ++i) nxt[i] = s4[i];
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) nxt[i] = make_uint4(0u, 0u, 0u, 0u);
        }
    };
    fetch(fr);
    for (int r0 = 0; r0 < total; r0 += kTcRows) {
        __syncthreads();   // the tile before's products done with Ks and Ps
        {
            uint4 * d = (uint4 *) (Ks + fr * kTcLd + fv);
#pragma unroll
            for (int i = 0; i < 4; ++i) d[i] = nxt[i];
        }
        __syncthreads();
        if (r0 + kTcRows < total) fetch(r0 + kTcRows + fr);
        {   // S partial: 8 k-steps of 16 over this warp's quarter of the dims
            wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
            wm::fill_fragment(acc, 0.f);
#pragma unroll
            for (int kk = 0; kk < 128; kk += 16) {
                wm::fragment<wm::matrix_a, 16, 16, 16, half, wm::row_major> a;
                wm::fragment<wm::matrix_b, 16, 16, 16, half, wm::col_major> b;
                wm::load_matrix_sync(a, Qs + sh * 16 * kTcLd + sk + kk, kTcLd);
                wm::load_matrix_sync(b, Ks + sr * 16 * kTcLd + sk + kk, kTcLd);
                wm::mma_sync(acc, a, b, acc);
            }
            wm::store_matrix_sync(Ss + warp * 256, acc, 16, wm::mem_row_major);
        }
        __syncthreads();
        {   // the softmax of head hh over the tile's 32 rows: two rows a lane, the 16 lanes of the head reduce
            const float qf = s_qf[hh];
            float sv[2];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int rr = lr + 16 * k, j = r0 + rr;   // row rr of the tile: row tile rr / 16
                const bool ok = j < total && (j < n_win || s_sel[j - n_win] >= 0);
                const float * ps = Ss + ((hh >> 4) | ((rr >> 4) << 1)) * 256 + (hh & 15) * 16 + (rr & 15);   // (warp (sh, sr, 0))
                sv[k] = ok ? (ps[0] + ps[4 * 256] + ps[8 * 256] + ps[12 * 256]) * qf : -INFINITY;
            }
            float mt = fmaxf(sv[0], sv[1]);
#pragma unroll
            for (int off = 8; off; off >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, off));
            const float mn = fmaxf(m, mt);   // finite from the first tile on (the window has a row)
            const float corr = __expf(m - mn);
            float ps = 0.f;
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const float pv = __expf(sv[k] - mn);
                ps += pv;
                Ps[hh * kTcLdP + lr + 16 * k] = __float2half_rn(pv * kTcPScale);
            }
#pragma unroll
            for (int off = 8; off; off >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off);
            l = l * corr + ps;
            m = mn;
            if (lr == 0) s_corr[hh] = corr;
        }
        __syncthreads();
        {   // O = O corr + P V over the tile's two row tiles
#pragma unroll
            for (int f = 0; f < 4; ++f)
#pragma unroll
                for (int i = 0; i < 8; ++i) o[f].x[i] *= s_corr[oh * 16 + rowmap[i]];
#pragma unroll
            for (int rt = 0; rt < kTcRows; rt += 16) {
                wm::fragment<wm::matrix_a, 16, 16, 16, half, wm::row_major> a;
                wm::load_matrix_sync(a, Ps + oh * 16 * kTcLdP + rt, kTcLdP);
#pragma unroll
                for (int f = 0; f < 4; ++f) {
                    wm::fragment<wm::matrix_b, 16, 16, 16, half, wm::row_major> b;
                    wm::load_matrix_sync(b, Ks + rt * kTcLd + oc + f * 16, kTcLd);
                    wm::mma_sync(o[f], a, b, o[f]);
                }
            }
        }
    }
    if (lr == 0) s_den[hh] = l + __expf(sinks[hb + hh] - m);
    __syncthreads();   // the last products done with Qs / Ks: O takes the region
#pragma unroll
    for (int f = 0; f < 4; ++f) wm::store_matrix_sync(Os + oh * 16 * D + oc + f * 16, o[f], D, wm::mem_row_major);
    __syncthreads();
    // out = O / (2^14 den), the query's rotation removed from the last n_rot dims (adjacent pairs)
    const int r0d = D - n_rot;
    for (int i = tid; i < kTcHeads * (D / 2); i += kTcThreads) {
        const int h = i / (D / 2), c = (i % (D / 2)) * 2;
        const float inv = 1.f / (kTcPScale * s_den[h]);
        float a = Os[h * D + c] * inv, b = Os[h * D + c + 1] * inv;
        if (c >= r0d) {
            float sn, cs;
            rot_sc(pos, inv_freq, (c - r0d) >> 1, sn, cs);
            const float a0 = a;
            a = a0 * cs + b * sn;
            b = -a0 * sn + b * cs;
        }
        *(float2 *) (out + ((int64_t) t * H + hb + h) * D + c) = make_float2(a, b);
    }
}
// the kernel's dynamic shared memory above 48 KB: allowed once per device
static void attention_tc_ready() {
    static bool done[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16 || done[dev]) return;
    CUDA_CHECK(cudaFuncSetAttribute(attention_tc_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kTcSmem));
    done[dev] = true;
}

void ds_attention_block(const float * q, const half * ring, const float * sinks, float * out, int T, int H, int D, int W, int n_rot,
                        const int * pos0, const float * inv_freq, float scale, cudaStream_t s) {
    if (D != 512 || T > 16) throw std::runtime_error("ds_attention_block: D must be 512 and T at most 16");
    attention_split_k<<<dim3(T, H), 256, 0, s>>>(q, ring, nullptr, nullptr, 0, sinks, out, H, W, n_rot, pos0, inv_freq, scale, -1, T);
}

void ds_attention(const float * q, const half * ring, const half * comp, const int32_t * idx, int k_sel,
                  const float * sinks, float * out, int T, int H, int D, int W, int n_rot, const int * pos0,
                  const float * inv_freq, float scale, cudaStream_t s, int origin) {
    if (D != 512 || H % 8) throw std::runtime_error("ds_attention: D must be 512 and H a multiple of 8");
    static const bool split_off = getenv("BNK_DS_ATTN_SPLIT") && atoi(getenv("BNK_DS_ATTN_SPLIT")) == 0;
    if (T <= 16 && !split_off) {
        attention_split_k<<<dim3(T, H), 256, 0, s>>>(q, ring, comp, idx, k_sel, sinks, out, H, W, n_rot, pos0, inv_freq, scale, origin);
        return;
    }
    // prompt chunks: the tensor-core kernel (BNK_DS_ATTN_TC=0: the fp32 one-key-at-a-time kernel below)
    static const int tc = getenv("BNK_DS_ATTN_TC") ? atoi(getenv("BNK_DS_ATTN_TC")) : 1;
    if (tc && H % kTcHeads == 0 && W + (idx ? k_sel : 0) <= kTcMax) {
        attention_tc_ready();
        attention_tc_k<<<dim3(H / kTcHeads, T), kTcThreads, kTcSmem, s>>>(q, ring, comp, idx, k_sel, sinks, out, H, W, n_rot, pos0, inv_freq, scale, origin);
        return;
    }
    // BNK_DS_ATTN_BATCH=1: the batched-keys variant (slower on a V100: 52 vs 40 ms on a 2,048-token chunk with the
    // selection, 12.5 vs 7.9 window-only; the extra registers cost more occupancy than the overlap gains)
    static const bool batched = getenv("BNK_DS_ATTN_BATCH") && atoi(getenv("BNK_DS_ATTN_BATCH")) != 0;
    if (batched) attention_k<<<dim3(T, H / 8), 256, 0, s>>>(q, ring, comp, idx, k_sel, sinks, out, H, W, n_rot, pos0, inv_freq, scale, origin);
    else attention_k1<<<dim3(T, H / 8), 256, 0, s>>>(q, ring, comp, idx, k_sel, sinks, out, H, W, n_rot, pos0, inv_freq, scale, origin);
}

// ------------------------------------------------------------------------------------------ compressor
// grid ceil(T / ratio) (one block per group that may complete in the window), block D = 512
__global__ void __launch_bounds__(512) compress_k(const float * kv, const float * score, const float * st_kv,
                                                  const float * st_sc, const float * w, const half * idx_k,
                                                  const float * idx_k_w, half * ckc, half * ikc, int T, int ratio, int D,
                                                  int ID, int n_rot, const int * pos0p, const float * inv_freq, float eps) {
    __shared__ float sh[32];
    __shared__ float L[512];
    __shared__ float ik[128];
    const int pos0 = *pos0p, d = threadIdx.x, g = blockIdx.x;
    int j;   // the group
    float v;
    if (ratio == 1) {
        if (g >= T) return;
        j = pos0 + g;
        v = kv[(int64_t) g * D + d];
    } else {
        // the g-th position of the window that completes a group (p % ratio == ratio - 1)
        const int first = pos0 + ((ratio - 1 - pos0 % ratio) + ratio) % ratio;
        const int p = first + g * ratio;
        if (p > pos0 + T - 1) return;
        j = p / ratio;
        // softmax-pooled over the group's positions (per dimension); earlier positions may be in the state
        float mx = -FLT_MAX;
        for (int q = p - ratio + 1; q <= p; ++q) {
            const float sc = q < pos0 ? st_sc[(q - (pos0 - ratio + 1)) * D + d] : score[(int64_t) (q - pos0) * D + d];
            mx = fmaxf(mx, sc);
        }
        float num = 0.f, den = 0.f;
        for (int q = p - ratio + 1; q <= p; ++q) {
            const bool st = q < pos0;
            const int sidx = q - (pos0 - ratio + 1);
            const float sc = st ? st_sc[sidx * D + d] : score[(int64_t) (q - pos0) * D + d];
            const float kvv = st ? st_kv[sidx * D + d] : kv[(int64_t) (q - pos0) * D + d];
            const float e = __expf(sc - mx);
            num += e * kvv;
            den += e;
        }
        v = num / den;
    }
    const float ss = block_sum(v * v, sh);
    L[d] = v * rsqrtf(ss / D + eps) * w[d];
    __syncthreads();
    // the index key from the RoPE-free latent
    float ikv = 0.f;
    if (d < ID) {
        const half * row = idx_k + (int64_t) d * D;
        float acc = 0.f;
        for (int i = 0; i < D; i += 2) {
            const float2 f = __half22float2(*(const half2 *) (row + i));
            acc += f.x * L[i] + f.y * L[i + 1];
        }
        ikv = acc;
    }
    const float iss = block_sum(d < ID ? ikv * ikv : 0.f, sh);
    if (d < ID) ik[d] = ikv * rsqrtf(iss / ID + eps) * idx_k_w[d];
    __syncthreads();
    const int gpos = j * ratio;
    {   // the latent, rotated
        float out = L[d];
        const int r0 = D - n_rot;
        if (d >= r0) {
            const int jj = d - r0, i = jj >> 1;
            float sn, cs;
            rot_sc(gpos, inv_freq, i, sn, cs);
            const float a = L[r0 + 2 * i], b = L[r0 + 2 * i + 1];
            out = (jj & 1) ? a * sn + b * cs : a * cs - b * sn;
        }
        ckc[(int64_t) j * D + d] = __float2half(out);
    }
    if (d < ID) {   // the index key, rotated
        float out = ik[d];
        const int r0 = ID - n_rot;
        if (d >= r0) {
            const int jj = d - r0, i = jj >> 1;
            float sn, cs;
            rot_sc(gpos, inv_freq, i, sn, cs);
            const float a = ik[r0 + 2 * i], b = ik[r0 + 2 * i + 1];
            out = (jj & 1) ? a * sn + b * cs : a * cs - b * sn;
        }
        ikc[(int64_t) j * ID + d] = __float2half(out);
    }
}
void ds_compress(const float * kv, const float * score, const float * st_kv, const float * st_sc, const float * w,
                 const half * idx_k, const float * idx_k_w, half * ckc, half * ikc, int T, int ratio, int D, int ID,
                 int n_rot, const int * pos0, const float * inv_freq, float eps, cudaStream_t s) {
    if (D != 512 || ID > 128 || ratio < 1 || ratio > 2) throw std::runtime_error("ds_compress: unsupported shape");
    const int groups = ratio == 1 ? T : (T + ratio - 1) / ratio;
    compress_k<<<groups, D, 0, s>>>(kv, score, st_kv, st_sc, w, idx_k, idx_k_w, ckc, ikc, T, ratio, D, ID, n_rot, pos0,
                                    inv_freq, eps);
}

__global__ void compress_commit_k(const float * kv, const float * score, float * st_kv, float * st_sc, int c, int ratio,
                                  int D, const int * pos0p) {
    const int rem = (*pos0p + c) % ratio;   // positions of the open group after the commit
    if (rem == 0) return;
    const int d = threadIdx.x;
    // the open group's positions pos0 + c - rem .. pos0 + c - 1: from the window rows (or the old state when a
    // window row precedes the window start: only the first `rem` state rows then)
    for (int i = 0; i < rem; ++i) {
        const int r = c - rem + i;   // window row (negative: it was in the state at index ratio - 1 + r ... only for ratio 2: index 0)
        float kvv, sc;
        if (r >= 0) {
            kvv = kv[(int64_t) r * D + d];
            sc = score ? score[(int64_t) r * D + d] : 0.f;
        } else {
            kvv = st_kv[(r + ratio - 1) * D + d];
            sc = st_sc ? st_sc[(r + ratio - 1) * D + d] : 0.f;
        }
        __syncthreads();
        st_kv[i * D + d] = kvv;
        if (st_sc) st_sc[i * D + d] = sc;
    }
}
void ds_compress_commit(const float * kv, const float * score, float * st_kv, float * st_sc, int c, int ratio, int D,
                        const int * pos0, cudaStream_t s) {
    if (ratio < 2) return;
    compress_commit_k<<<1, D, 0, s>>>(kv, score, st_kv, st_sc, c, ratio, D, pos0);
}

// ------------------------------------------------------------------------------------------ indexer
// grid (ceil(n_max / 4), T), 128 threads: warp = one compressed row; lane holds 4 dims of ID = 128
__global__ void __launch_bounds__(128) index_scores_k(const float * __restrict__ iq, const half * __restrict__ ikc,
                                                      const float * __restrict__ iw, float * scores, int IH, int ID,
                                                      int ratio, int n_max, int64_t lds, const int * pos0p) {
    const int t = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c = blockIdx.x * 4 + w;
    if (c >= n_max) return;
    const int n_vis = (*pos0p + t + 1) / ratio;
    if (c >= n_vis) {
        if (lane == 0) scores[(int64_t) t * lds + c] = -INFINITY;
        return;
    }
    const uint2 kr = *(const uint2 *) (ikc + (int64_t) c * ID + lane * 4);
    const half2 * k2 = (const half2 *) &kr;
    const float2 ka = __half22float2(k2[0]), kb = __half22float2(k2[1]);
    float acc = 0.f;
    const float * qb = iq + (int64_t) t * IH * ID + lane * 4;
    for (int h = 0; h < IH; ++h) {
        const float4 qv = *(const float4 *) (qb + (int64_t) h * ID);
        float d = qv.x * ka.x + qv.y * ka.y + qv.z * kb.x + qv.w * kb.y;
        d = warp_sum(d);
        acc += fmaxf(d, 0.f) * iw[t * IH + h];
    }
    if (lane == 0) scores[(int64_t) t * lds + c] = acc;
}
void ds_index_scores(const float * iq, const half * ikc, const float * iw, float * scores, int T, int IH, int ID,
                     int ratio, int n_max, int64_t lds, const int * pos0, cudaStream_t s) {
    if (ID != 128) throw std::runtime_error("ds_index_scores: ID must be 128");
    if (n_max <= 0) return;
    index_scores_k<<<dim3((n_max + 3) / 4, T), 128, 0, s>>>(iq, ikc, iw, scores, IH, ID, ratio, n_max, lds, pos0);
}

// Radix select of the k largest of n floats (one block of 1024 threads), reading value i through `val(i)` and
// a validity `ok(i)` (invalid values count as the smallest key). MODE 0 writes their indices to idx[0..k)
// (-1 past n), MODE 1 marks them in mask[i] (1 / 0).
__device__ __forceinline__ uint32_t f2key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
template <int MODE, typename Val, typename Ok>
__device__ void radix_select(int n, int k, Val val, Ok ok, int32_t * idx, uint8_t * mask) {
    __shared__ int hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ int s_krem, s_cnt_gt, s_cnt_eq;
    if (MODE == 1)
        for (int i = threadIdx.x; i < n; i += blockDim.x) mask[i] = 0;
    if (k <= 0) {
        if (MODE == 0) for (int i = threadIdx.x; i < k; i += blockDim.x) idx[i] = -1;
        return;
    }
    const int kk = min(k, n);
    if (threadIdx.x == 0) { s_prefix = 0; s_krem = kk; }
    __syncthreads();
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        const uint32_t pmask = pass == 0 ? 0u : (0xffffffffu << (shift + 8));
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const uint32_t prefix = s_prefix;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t key = ok(i) ? f2key(val(i)) : 0u;
            if ((key & pmask) == prefix) atomicAdd(&hist[(key >> shift) & 255], 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int cum = 0, krem = s_krem, b = 255;
            for (; b >= 0; --b) {
                cum += hist[b];
                if (cum >= krem) break;
            }
            if (b < 0) b = 0;   // (cannot happen: kk <= n)
            s_krem = krem - (cum - hist[b]);
            s_prefix = prefix | ((uint32_t) b << shift);
        }
        __syncthreads();
    }
    const uint32_t thr = s_prefix;
    const int krem = s_krem, n_gt = kk - krem;
    if (threadIdx.x == 0) { s_cnt_gt = 0; s_cnt_eq = 0; }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const uint32_t key = ok(i) ? f2key(val(i)) : 0u;
        if (key > thr) {
            const int p = atomicAdd(&s_cnt_gt, 1);
            if (MODE == 0) idx[p] = i; else mask[i] = 1;
        } else if (key == thr) {
            const int p = atomicAdd(&s_cnt_eq, 1);
            if (p < krem) {
                if (MODE == 0) idx[n_gt + p] = i; else mask[i] = 1;
            }
        }
    }
    if (MODE == 0)
        for (int i = kk + threadIdx.x; i < k; i += blockDim.x) idx[i] = -1;
}

__global__ void __launch_bounds__(1024) topk_select_k(const float * scores, int64_t lds, int ratio, int k, const int * pos0p,
                                                      const uint8_t * cand, int bs, int64_t ldc, int32_t * idx) {
    const int t = blockIdx.x;
    const int n_vis = (*pos0p + t + 1) / ratio;
    const float * row = scores + (int64_t) t * lds;
    const uint8_t * cm = cand ? cand + (int64_t) t * ldc : nullptr;
    radix_select<0>(n_vis, k, [&](int i) { return row[i]; }, [&](int i) { return cm ? cm[i / bs] != 0 : true; },
                    idx + (int64_t) t * k, nullptr);
}
void ds_topk_select(const float * scores, int64_t lds, int T, int ratio, int k, const int * pos0, const uint8_t * cand,
                    int bs, int32_t * idx, cudaStream_t s) {
    const int64_t ldc = cand ? (lds + bs - 1) / bs : 0;
    topk_select_k<<<T, 1024, 0, s>>>(scores, lds, ratio, k, pos0, cand, bs > 0 ? bs : 1, ldc, idx);
}

__global__ void cand_scores_k(const float * scores, int64_t lds, int ratio, int bs, int n_blocks, int64_t ldb,
                              const int * pos0p, float * bscore) {
    const int t = blockIdx.y, b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const int n_vis = (*pos0p + t + 1) / ratio;
    const int nb = (n_vis + bs - 1) / bs;
    float v = -INFINITY;
    if (b < nb) {
        const float * row = scores + (int64_t) t * lds;
        for (int i = b * bs; i < min(n_vis, (b + 1) * bs); ++i) v = fmaxf(v, row[i]);
        if (b == (n_vis - 1) / bs) v = FLT_MAX;   // the block holding the newest position always survives
    }
    bscore[(int64_t) t * ldb + b] = v;
}
__global__ void __launch_bounds__(1024) cand_select_k(const float * bscore, int64_t ldb, int ratio, int bs, int k_blocks,
                                                      const int * pos0p, uint8_t * cand) {
    const int t = blockIdx.x;
    const int n_vis = (*pos0p + t + 1) / ratio;
    const int nb = (n_vis + bs - 1) / bs;
    const float * row = bscore + (int64_t) t * ldb;
    radix_select<1>(nb, k_blocks, [&](int i) { return row[i]; }, [&](int) { return true; }, nullptr,
                    cand + (int64_t) t * ldb);
}
void ds_cand_blocks(const float * scores, int64_t lds, int T, int ratio, int bs, int k_blocks, int n_blocks,
                    const int * pos0, float * scratch, uint8_t * cand, cudaStream_t s) {
    if (n_blocks <= 0) return;
    const int64_t ldb = (lds + bs - 1) / bs;
    cand_scores_k<<<dim3((n_blocks + 255) / 256, T), 256, 0, s>>>(scores, lds, ratio, bs, n_blocks, ldb, pos0, scratch);
    cand_select_k<<<T, 1024, 0, s>>>(scratch, ldb, ratio, bs, k_blocks, pos0, cand);
}

// ------------------------------------------------------------------------------------------ Engram
// grid (T, hc), 1024 threads
__global__ void __launch_bounds__(1024) engram_gate_k(float * res, const float * key, int64_t ldk, const float * value,
                                                      int64_t ldv, const float * w, int hc, int E, float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.x, s = blockIdx.y;
    float * h = res + ((int64_t) t * hc + s) * E;
    const float * kk = key + (int64_t) t * ldk + (int64_t) s * E, * ww = w + (int64_t) s * E, * vv = value + (int64_t) t * ldv;
    float sh2 = 0.f, sk2 = 0.f, dt = 0.f;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const float hv = h[e], kv = kk[e];
        sh2 += hv * hv;
        sk2 += kv * kv;
        dt += hv * ww[e] * kv;
    }
    sh2 = block_sum(sh2, sh);
    sk2 = block_sum(sk2, sh);
    dt = block_sum(dt, sh);
    const float rstd = rsqrtf(sh2 / E + eps) * rsqrtf(sk2 / E + eps);
    const float dot = dt * rstd * rsqrtf((float) E);
    const float gate = sigmoidf_(copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot));
    for (int e = threadIdx.x; e < E; e += blockDim.x) h[e] += gate * vv[e];
}
void ds_engram_gate(float * res, const float * key, int64_t ldk, const float * value, int64_t ldv, const float * w, int T,
                    int hc, int E, float eps, cudaStream_t s) {
    engram_gate_k<<<dim3(T, hc), 1024, 0, s>>>(res, key, ldk, value, ldv, w, hc, E, eps);
}

// ------------------------------------------------------------------------------------------ routing
__global__ void route_k(const float * logits, const float * bias, int n_exp, int k, float scale, int norm, int32_t * ids,
                        float * w) {
    ds_route_warp(logits, bias, n_exp, k, scale, norm, ids, w, blockIdx.x, threadIdx.x);
}
void ds_route(const float * logits, const float * bias, int T, int n_exp, int k, float scale, bool norm, int32_t * ids,
              float * w, cudaStream_t s) {
    if (n_exp > 32 * 16 || k > 32) throw std::runtime_error("ds_route: too many experts");
    route_k<<<T, 32, 0, s>>>(logits, bias, n_exp, k, scale, norm ? 1 : 0, ids, w);
}

// ------------------------------------------------------------------------------------------ the PCIe lane
// one block: the T * k predicted pairs (<= 128) walked by thread 0 (a few dozen comparisons)
__global__ void pull_pick_k(const int32_t * ids, const float * w, int n, const int32_t * slot_of, int32_t * out, int P,
                            int32_t * state) {
    if (threadIdx.x) return;
    for (int p = 0; p < P; ++p) {
        int best = -1;
        float bw = -1.f;
        for (int i = 0; i < n; ++i) {
            const int e = ids[i];
            if (e < 0 || slot_of[e] >= 0 || w[i] <= bw) continue;
            bool taken = false;
            for (int q = 0; q < p; ++q) taken |= out[q] == e;
            if (!taken) { best = e; bw = w[i]; }
        }
        out[p] = best;
    }
    if (state) *state = out[0] >= 0 ? 1 : 0;   // a pull in flight (the commit makes it 2, the restore 0)
}
void ds_pull_pick(const int32_t * ids, const float * w, int T, int k, const int32_t * slot_of, int32_t * out, int P, cudaStream_t s,
                  int32_t * state) {
    pull_pick_k<<<1, 32, 0, s>>>(ids, w, T * k, slot_of, out, P, state);
}
// grid (blocks, P): the blob of pick p, 16 bytes a thread per step, eight loads in flight. Few blocks on purpose: the
// link (~12 GB/s) is full with ~1 MB of requests in flight, and the main stream's kernels must find free SMs beside
// the pull (512 blocks of it stalled decode's own work for the pull's duration: 50 -> 75 ms a token).
constexpr int kPullBlocks = 16, kPullUnroll = 8;
__global__ void __launch_bounds__(256) pull_copy_k(const int32_t * picks, const uint64_t * blob_ptr, size_t n16, uint8_t * raw) {
    const int p = blockIdx.y, e = picks[p];
    if (e < 0) return;
    const uint4 * src = (const uint4 *) blob_ptr[e];
    uint4 * dst = (uint4 *) (raw + (size_t) p * n16 * 16);
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (kPullUnroll - 1) * stride < n16; i += kPullUnroll * stride) {
        uint4 v[kPullUnroll];
#pragma unroll
        for (int u = 0; u < kPullUnroll; ++u) v[u] = src[i + u * stride];
#pragma unroll
        for (int u = 0; u < kPullUnroll; ++u) dst[i + u * stride] = v[u];
    }
    for (; i < n16; i += stride) dst[i] = src[i];
}
void ds_pull_copy(const int32_t * picks, int P, const uint64_t * blob_ptr, size_t bytes, uint8_t * raw, cudaStream_t s) {
    if (bytes % 16) throw std::runtime_error("ds_pull_copy: blob bytes not a multiple of 16");
    static const int blocks = getenv("BNK_DS_PULL_BLOCKS") ? atoi(getenv("BNK_DS_PULL_BLOCKS")) : kPullBlocks;
    pull_copy_k<<<dim3(blocks, P), 256, 0, s>>>(picks, blob_ptr, bytes / 16, raw);
}
// after the copy: the slots patched in, then (fence) the state 2. A plan kernel that runs before this sees the experts
// as misses (the CPU takes them); one that runs after finds them resident.
__global__ void pull_commit_k(const int32_t * picks, int P, int32_t * slot_of, int slot0, int32_t * staged, int32_t * state) {
    const int p = threadIdx.x;
    if (p < P) {
        const int e = picks[p];
        if (e >= 0) slot_of[e] = slot0 + p;
        staged[p] = e;
    }
    __threadfence();
    __syncthreads();
    if (p == 0 && state && *((volatile int32_t *) state) == 1) *((volatile int32_t *) state) = 2;
}
void ds_pull_commit(const int32_t * picks, int P, int32_t * slot_of, int slot0, int32_t * staged, cudaStream_t s, int32_t * state) {
    pull_commit_k<<<1, 32, 0, s>>>(picks, P, slot_of, slot0, staged, state);
}
// after the layer's FFN: a pull still in flight (state 1) is waited for here (it had the FFN's time on top of its
// window; stats[2] counts these), then the slots are unpatched and the state is 0 again
__global__ void pull_restore_k(int32_t * staged, int P, int32_t * slot_of, int slot0, const HitList * hits, uint32_t * stats,
                               int32_t * state) {
    __shared__ int s_had;
    if (threadIdx.x == 0) {
        int st = state ? *((volatile int32_t *) state) : 2;
        if (st == 1) {
            atomicAdd(&stats[2], 1u);
            while ((st = *((volatile int32_t *) state)) == 1) __nanosleep(500);
        }
        __threadfence();
        s_had = st == 2;
    }
    __syncthreads();
    const int p = threadIdx.x;
    if (s_had && p < P) {
        const int e = staged[p];
        if (e >= 0) {
            slot_of[e] = -1;
            staged[p] = -1;
            int used = 0;
            for (int i = 0; i < hits->n; ++i) used |= hits->e[i].slot == slot0 + p;
            atomicAdd(&stats[0], 1u);
            if (used) atomicAdd(&stats[1], 1u);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && state) *((volatile int32_t *) state) = 0;
}
void ds_pull_restore(int32_t * staged, int P, int32_t * slot_of, int slot0, const HitList * hits, uint32_t * stats, cudaStream_t s,
                     int32_t * state) {
    pull_restore_k<<<1, 32, 0, s>>>(staged, P, slot_of, slot0, hits, stats, state);
}

__global__ void copy_rows_k(const float * x, int64_t ldx, float * y, int64_t ldy, int n) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[(int64_t) t * ldy + i] = x[(int64_t) t * ldx + i];
}
__global__ void copy_bytes_k(const uint4 * src, uint4 * dst, size_t n16) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
}
void ds_copy_from_host(const void * src, void * dst, size_t bytes, cudaStream_t s) {
    if (bytes % 16) throw std::runtime_error("ds_copy_from_host: bytes must be a multiple of 16");
    const size_t n16 = bytes / 16, b = (n16 + 255) / 256;
    copy_bytes_k<<<(unsigned) (b < 4096 ? b : 4096), 256, 0, s>>>((const uint4 *) src, (uint4 *) dst, n16);
}
__global__ void rows_from_host_k(const float4 * src, const int32_t * map, float4 * dst, int E4) {
    const int t = blockIdx.y, m = map[t];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E4; i += gridDim.x * blockDim.x)
        dst[(size_t) t * E4 + i] = m >= 0 ? src[(size_t) m * E4 + i] : make_float4(0.f, 0.f, 0.f, 0.f);
}
__global__ void rows_gather_k(const float4 * x, const int32_t * tok, float4 * xc, int E4) {
    const int m = blockIdx.y, t = tok[m];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E4; i += gridDim.x * blockDim.x) xc[(size_t) m * E4 + i] = x[(size_t) t * E4 + i];
}
void ds_rows_gather(const float * x, const int32_t * tok, float * xc, int M, int E, cudaStream_t s) {
    if (E % 4) throw std::runtime_error("ds_rows_gather: E % 4");
    if (M > 0) rows_gather_k<<<dim3(4, M), 256, 0, s>>>((const float4 *) x, tok, (float4 *) xc, E / 4);
}
void ds_rows_from_host(const float * src, const int32_t * map, float * dst, int N, int E, cudaStream_t s) {
    if (E % 4) throw std::runtime_error("ds_rows_from_host: E % 4");
    rows_from_host_k<<<dim3(4, N), 256, 0, s>>>((const float4 *) src, map, (float4 *) dst, E / 4);
}
__global__ void zero_half_k(half * p, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) p[i] = __float2half(0.f);
}
__global__ void add_k(const float * a, const float * b, float * y, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) y[i] = a[i] + b[i];
}
void ds_add(const float * a, const float * b, float * y, size_t n, cudaStream_t s) {
    add_k<<<(unsigned) std::min<size_t>((n + 255) / 256, 4096), 256, 0, s>>>(a, b, y, n);
}
__global__ void hc_mean_k(const float * res, float * out, int64_t ldo, int hc, int E) {
    const int t = blockIdx.y;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < E; e += gridDim.x * blockDim.x) {
        float a = 0.f;
        for (int s = 0; s < hc; ++s) a += res[((int64_t) t * hc + s) * E + e];
        out[(int64_t) t * ldo + e] = a / hc;
    }
}
void ds_hc_mean(const float * res, float * out, int64_t ldo, int T, int hc, int E, cudaStream_t s) {
    hc_mean_k<<<dim3((E + 255) / 256, T), 256, 0, s>>>(res, out, ldo, hc, E);
}

void ds_zero_half(half * p, size_t n, cudaStream_t s) { zero_half_k<<<(unsigned) std::min<size_t>((n + 255) / 256, 4096), 256, 0, s>>>(p, n); }
__global__ void ring_read_k(const half * ring, int origin, half * kwin, int p_from, int D) {
    const int p = p_from + blockIdx.x;
    const half * src = ring + (int64_t) (p % kDsRing) * D;
    half * dst = kwin + (int64_t) (p - origin) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) dst[d] = src[d];
}
void ds_ring_read(const half * ring, int origin, half * kwin, int p_from, int p_to, int D, cudaStream_t s) {
    if (p_to <= p_from) return;
    ring_read_k<<<p_to - p_from, 256, 0, s>>>(ring, origin, kwin, p_from, D);
}

void ds_copy_rows(const float * x, int64_t ldx, float * y, int64_t ldy, int T, int n, cudaStream_t s) {
    copy_rows_k<<<dim3(min(256, (n + 255) / 256), T), 256, 0, s>>>(x, ldx, y, ldy, n);
}

}  // namespace bnk
