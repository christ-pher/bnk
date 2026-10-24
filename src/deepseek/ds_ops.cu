#include "deepseek/ds_ops.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "core/util.h"

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
template <int HC>
__global__ void __launch_bounds__(256) hc_mix_dots_k(const float * __restrict__ res, const float * __restrict__ fn, float * scratch, int E) {
    constexpr int MIX = (2 + HC) * HC;
    __shared__ float part[8];
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
    }
}
template <int HC>
__global__ void hc_mix_finish_k(const float * __restrict__ scratch, const float * __restrict__ scale, const float * __restrict__ base,
                                float * pre, float * post, float * comb, int E, float eps, float hc_eps, int iters) {
    constexpr int MIX = (2 + HC) * HC;
    const int t = blockIdx.x;
    const float * m = scratch + t * (MIX + 1);
    const float r = rsqrtf(m[MIX] / (float) ((int64_t) HC * E) + eps);
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

void ds_hc_mixes(const float * res, const float * fn, const float * scale, const float * base, float * pre, float * post,
                 float * comb, int T, int hc, int E, float eps, float hc_eps, int iters, cudaStream_t s, float * scratch) {
    if (hc != 4) throw std::runtime_error("ds_hc_mixes: hc must be 4");
    if (scratch && T <= 16 && (hc * E) % 4 == 0) {
        hc_mix_dots_k<4><<<dim3(T, 25), 256, 0, s>>>(res, fn, scratch, E);
        hc_mix_finish_k<4><<<T, 1, 0, s>>>(scratch, scale, base, pre, post, comb, E, eps, hc_eps, iters);
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
                                                         int origin) {
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
    const int n_win = min(W, pos + 1);
    const int n_keys = n_win + (idx ? k_sel : 0);
    for (int key = w; key < n_keys; key += NW) {
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

void ds_attention(const float * q, const half * ring, const half * comp, const int32_t * idx, int k_sel,
                  const float * sinks, float * out, int T, int H, int D, int W, int n_rot, const int * pos0,
                  const float * inv_freq, float scale, cudaStream_t s, int origin) {
    if (D != 512 || H % 8) throw std::runtime_error("ds_attention: D must be 512 and H a multiple of 8");
    static const bool split_off = getenv("BNK_DS_ATTN_SPLIT") && atoi(getenv("BNK_DS_ATTN_SPLIT")) == 0;
    if (T <= 16 && !split_off) {
        attention_split_k<<<dim3(T, H), 256, 0, s>>>(q, ring, comp, idx, k_sel, sinks, out, H, W, n_rot, pos0, inv_freq, scale, origin);
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
__device__ __forceinline__ float softplusf_(float x) { return x > 20.f ? x : x < -20.f ? __expf(x) : log1pf(__expf(x)); }

// one warp per token; n_exp <= 32 * 16
__global__ void route_k(const float * logits, const float * bias, int n_exp, int k, float scale, int norm, int32_t * ids,
                        float * w) {
    const int t = blockIdx.x, lane = threadIdx.x;
    constexpr int PER = 16;
    float s[PER], sc[PER];
    bool taken[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int e = lane + 32 * j;
        s[j] = e < n_exp ? sqrtf(softplusf_(logits[(int64_t) t * n_exp + e])) : 0.f;
        sc[j] = e < n_exp ? s[j] + bias[e] : -FLT_MAX;
        taken[j] = e >= n_exp;
    }
    float sum = 0.f;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int bj = -1;
#pragma unroll
        for (int j = 0; j < PER; ++j)
            if (!taken[j] && sc[j] > best) { best = sc[j]; bj = j; }
        // the warp's best: highest score, lowest expert id on ties
        float wb = warp_max(best);
        int cand = (best == wb && bj >= 0) ? lane + 32 * bj : 0x7fffffff;
#pragma unroll
        for (int o = 16; o; o >>= 1) cand = min(cand, __shfl_xor_sync(0xffffffffu, cand, o));
        const int e = cand;
        const int owner = e & 31, j = e >> 5;
        float sv = 0.f;
        if (lane == owner) { taken[j] = true; sv = s[j]; }
        sv = __shfl_sync(0xffffffffu, sv, owner);
        sum += sv;
        if (lane == 0) { ids[t * k + r] = e; w[t * k + r] = sv; }
    }
    if (lane == 0) {
        const float f = (norm && k > 1) ? scale / (sum + 1e-20f) : scale;
        for (int r = 0; r < k; ++r) w[t * k + r] *= f;
    }
}
void ds_route(const float * logits, const float * bias, int T, int n_exp, int k, float scale, bool norm, int32_t * ids,
              float * w, cudaStream_t s) {
    if (n_exp > 32 * 16 || k > 32) throw std::runtime_error("ds_route: too many experts");
    route_k<<<T, 32, 0, s>>>(logits, bias, n_exp, k, scale, norm ? 1 : 0, ids, w);
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
    const int t = blockIdx.y;
    const bool have = map[t] >= 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E4; i += gridDim.x * blockDim.x)
        dst[(size_t) t * E4 + i] = have ? src[(size_t) t * E4 + i] : make_float4(0.f, 0.f, 0.f, 0.f);
}
void ds_rows_from_host(const float * src, const int32_t * map, float * dst, int N, int E, cudaStream_t s) {
    if (E % 4) throw std::runtime_error("ds_rows_from_host: E % 4");
    rows_from_host_k<<<dim3(4, N), 256, 0, s>>>((const float4 *) src, map, (float4 *) dst, E / 4);
}
__global__ void zero_half_k(half * p, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) p[i] = __float2half(0.f);
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
