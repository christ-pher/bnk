#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "kernels/glm_ops.h"
#include "kernels/kvq.cuh"
#include "kernels/ops.h"

namespace bnk {

namespace {

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float bsum(float v, float * sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = wsum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = threadIdx.x < nw ? sh[threadIdx.x] : 0.f;
    if (wid == 0) v = wsum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
inline int nblk(int64_t n, int t) { return (int) ((n + t - 1) / t); }

}  // namespace

// ------------------------------------------------------------------------------------------- mHC
__global__ void mhc_norm_k(const float * res, float * xn, int n, float eps) {
    __shared__ float sh[32];
    const float * x = res + (int64_t) blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = bsum(ss, sh);
    const float inv = rsqrtf(ss / n + eps);
    float * y = xn + (int64_t) blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv;
}
void mhc_norm(const float * res, float * xn, int T, int hc, int E, float eps, cudaStream_t s) {
    mhc_norm_k<<<T, 512, 0, s>>>(res, xn, hc * E, eps);
}

// one thread per token: hc <= 8
__global__ void mhc_weights_k(const float * f, const float * base, const float * scale, float * w, int T, int hc,
                              float eps, int iters) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const int nw = (2 + hc) * hc;
    const float * x = f + (int64_t) t * nw;
    float * o = w + (int64_t) t * nw;
    for (int i = 0; i < hc; ++i) o[i] = sigm(x[i] * scale[0] + base[i]) + eps;
    for (int i = 0; i < hc; ++i) o[hc + i] = 2.f * sigm(x[hc + i] * scale[1] + base[hc + i]);
    float c[64];
    for (int i = 0; i < hc; ++i) {   // softmax over each row of the hc x hc matrix
        float mx = -FLT_MAX;
        for (int j = 0; j < hc; ++j) {
            c[i * hc + j] = x[2 * hc + i * hc + j] * scale[2] + base[2 * hc + i * hc + j];
            mx = fmaxf(mx, c[i * hc + j]);
        }
        float sum = 0.f;
        for (int j = 0; j < hc; ++j) {
            c[i * hc + j] = __expf(c[i * hc + j] - mx);
            sum += c[i * hc + j];
        }
        for (int j = 0; j < hc; ++j) c[i * hc + j] = c[i * hc + j] / sum + eps;
    }
    auto cols = [&]() {
        for (int j = 0; j < hc; ++j) {
            float sum = 0.f;
            for (int i = 0; i < hc; ++i) sum += c[i * hc + j];
            for (int i = 0; i < hc; ++i) c[i * hc + j] /= sum + eps;
        }
    };
    auto rows = [&]() {
        for (int i = 0; i < hc; ++i) {
            float sum = 0.f;
            for (int j = 0; j < hc; ++j) sum += c[i * hc + j];
            for (int j = 0; j < hc; ++j) c[i * hc + j] /= sum + eps;
        }
    };
    cols();
    for (int it = 1; it < iters; ++it) {
        rows();
        cols();
    }
    for (int i = 0; i < hc * hc; ++i) o[2 * hc + i] = c[i];
}
void mhc_weights(const float * f, const float * base, const float * scale, float * w, int T, int hc, float eps, int iters,
                 cudaStream_t s) {
    if (hc > 8) { fprintf(stderr, "mhc: hc %d unsupported\n", hc); abort(); }
    mhc_weights_k<<<nblk(T, 32), 32, 0, s>>>(f, base, scale, w, T, hc, eps, iters);
}

__global__ void mhc_collapse_k(const float * res, const float * w, float * mixed, int hc, int E) {
    const int t = blockIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const float * pre = w + (int64_t) t * (2 + hc) * hc;
    const float * r = res + (int64_t) t * hc * E + e;
    float acc = 0.f;
    for (int s = 0; s < hc; ++s) acc += pre[s] * r[(int64_t) s * E];
    mixed[(int64_t) t * E + e] = acc;
}
void mhc_collapse(const float * res, const float * w, float * mixed, int T, int hc, int E, cudaStream_t s) {
    mhc_collapse_k<<<dim3(nblk(E, 256), T), 256, 0, s>>>(res, w, mixed, hc, E);
}

__global__ void mhc_combine_k(float * res, const float * out, const float * w, int hc, int E) {
    const int t = blockIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const float * ww = w + (int64_t) t * (2 + hc) * hc;
    const float * post = ww + hc, * comb = ww + 2 * hc;
    float * r = res + (int64_t) t * hc * E + e;
    const float o = out[(int64_t) t * E + e];
    float in[8];
    for (int i = 0; i < hc; ++i) in[i] = r[(int64_t) i * E];
    for (int j = 0; j < hc; ++j) {
        float acc = post[j] * o;
        for (int i = 0; i < hc; ++i) acc += comb[i * hc + j] * in[i];
        r[(int64_t) j * E] = acc;
    }
}
void mhc_combine(float * res, const float * out, const float * w, int T, int hc, int E, cudaStream_t s) {
    mhc_combine_k<<<dim3(nblk(E, 256), T), 256, 0, s>>>(res, out, w, hc, E);
}

__global__ void hc_mean_k(const float * res, float * x, int hc, int E) {
    const int t = blockIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const float * r = res + (int64_t) t * hc * E + e;
    float acc = 0.f;
    for (int s = 0; s < hc; ++s) acc += r[(int64_t) s * E];
    x[(int64_t) t * E + e] = acc / hc;
}
void hc_mean(const float * res, float * x, int T, int hc, int E, cudaStream_t s) {
    hc_mean_k<<<dim3(nblk(E, 256), T), 256, 0, s>>>(res, x, hc, E);
}

// ------------------------------------------------------------------------------------------- KDA
// grid (2H, T): q heads then k heads of the conv output row
__global__ void kda_l2norm_k(float * co, int C, int S, float eps) {
    __shared__ float sh[32];
    float * x = co + (int64_t) blockIdx.y * C + blockIdx.x * S;
    float ss = 0.f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) ss += x[i] * x[i];
    ss = bsum(ss, sh);
    const float inv = 1.f / sqrtf(ss + eps);
    for (int i = threadIdx.x; i < S; i += blockDim.x) x[i] *= inv;
}
void kda_l2norm(float * co, int T, int C, int H, int S, float eps, cudaStream_t s) {
    kda_l2norm_k<<<dim3(2 * H, T), 128, 0, s>>>(co, C, S, eps);
}

__global__ void kda_gate_k(float * f, const float * dt, const float * a_log, float lower, int64_t n, int HS, int S) {
    const int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = (int) (i % HS);
    const float rate = __expf(a_log[c / S]);
    f[i] = lower * sigm(rate * (f[i] + dt[c]));
}
void kda_gate(float * f, const float * dt, const float * a_log, float lower, int T, int H, int S, cudaStream_t s) {
    const int64_t n = (int64_t) T * H * S;
    kda_gate_k<<<nblk(n, 256), 256, 0, s>>>(f, dt, a_log, lower, n, H * S, S);
}

// The delta rule (ops.cu gdn_rec4_k) with the decay per key channel: each block owns 32 columns j of one head's
// S x S state, split over NW warps of R rows; S[i][j] *= exp(g[t][h][i]).
template <int S, int NW>
__global__ void __launch_bounds__(32 * NW) kda_rec_k(const float * __restrict__ co, int C, const float * __restrict__ g,
                                                 const float * __restrict__ beta, float * __restrict__ state,
                                                 float * __restrict__ out, int T, int H, int commit) {
    constexpr int R = S / NW;
    const int h = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int j = blockIdx.y * 32 + lane;
    __shared__ float qs[S], ks[S], gs[S], red[NW][32];
    float col[R];
    float * st = state + (int64_t) h * S * S + (int64_t) (w * R) * S + j;
#pragma unroll
    for (int i = 0; i < R; ++i) col[i] = st[(int64_t) i * S];
    const float scale = rsqrtf((float) S);
    const int HS = H * S;
    for (int t = 0; t < T; ++t) {
        const float * row = co + (int64_t) t * C;
        __syncthreads();
        for (int i = threadIdx.x; i < S; i += blockDim.x) {
            qs[i] = row[h * S + i];
            ks[i] = row[HS + h * S + i];
            gs[i] = __expf(g[(int64_t) t * HS + h * S + i]);
        }
        __syncthreads();
        const float b = beta[t * H + h];
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            col[i] *= gs[w * R + i];
            kv += col[i] * ks[w * R + i];
        }
        red[w][lane] = kv;
        __syncthreads();
        kv = 0.f;
#pragma unroll
        for (int q = 0; q < NW; ++q) kv += red[q][lane];
        const float v = row[2 * HS + h * S + j];
        const float delta = (v - kv) * b;
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            col[i] += ks[w * R + i] * delta;
            o += col[i] * qs[w * R + i];
        }
        __syncthreads();
        red[w][lane] = o;
        __syncthreads();
        if (out && w == 0) {
            float os = 0.f;
#pragma unroll
            for (int q = 0; q < NW; ++q) os += red[q][lane];
            out[((int64_t) t * H + h) * S + j] = os * scale;
        }
        if (t + 1 == commit) {
#pragma unroll
            for (int i = 0; i < R; ++i) st[(int64_t) i * S] = col[i];
        }
    }
}
void kda_recurrence(const float * conv_out, int C, const float * g, const float * beta, float * state, float * out,
                    int T, int H, int S, int commit, cudaStream_t s) {
    if (S != 128) { fprintf(stderr, "kda: unsupported head dim %d\n", S); abort(); }
    kda_rec_k<128, 4><<<dim3(H, 128 / 32), 128, 0, s>>>(conv_out, C, g, beta, state, out, T, H, commit);
}

// ------------------------------------------------------------------------------------------- MLA
template <bool Q8>
__global__ void mla_store_k(const float * c, int64_t ldc, half * cache, int Dl, const int * pos0p) {
    const int t = blockIdx.x, i = threadIdx.x;
    kv_store<Q8>(cache, (int64_t) (*pos0p + t), Dl, i, c[(int64_t) t * ldc + i]);
}
void mla_store_latent(const float * c, int64_t ldc, half * cache, int T, int Dl, const int * pos0, cudaStream_t s) {
    if (Dl % 32 != 0 || Dl > 1024) { fprintf(stderr, "mla: latent width %d unsupported\n", Dl); abort(); }
    (kv_q8() ? mla_store_k<true> : mla_store_k<false>)<<<T, Dl, 0, s>>>(c, ldc, cache, Dl, pos0);
}

// grid (T, H), block Dl threads (j)
__global__ void mla_absorb_k(const float * q, int64_t ldq, const half * kv_b, float * qabs, int H, int Dk, int Dv, int Dl,
                             float scale) {
    extern __shared__ float qs[];
    const int t = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
    for (int d = threadIdx.x; d < Dk; d += blockDim.x) qs[d] = q[(int64_t) t * ldq + h * Dk + d];
    __syncthreads();
    const half * w = kv_b + (int64_t) h * (Dk + Dv) * Dl + j;
    float acc = 0.f;
    for (int d = 0; d < Dk; ++d) acc += __half2float(w[(int64_t) d * Dl]) * qs[d];
    qabs[((int64_t) t * H + h) * Dl + j] = acc * scale;
}
void mla_absorb_q(const float * q, int64_t ldq, const half * kv_b, float * qabs, int T, int H, int Dk, int Dv, int Dl,
                  float scale, cudaStream_t s) {
    mla_absorb_k<<<dim3(T, H), Dl, Dk * sizeof(float), s>>>(q, ldq, kv_b, qabs, H, Dk, Dv, Dl, scale);
}

// grid (T, H), block Dv threads (i): each reads its own row of kv_b (Dl halves, 16-byte loads)
__global__ void mla_out_k(const float * olat, const half * kv_b, float * o, int H, int Dk, int Dv, int Dl) {
    extern __shared__ float os[];
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    for (int j = threadIdx.x; j < Dl; j += blockDim.x) os[j] = olat[((int64_t) t * H + h) * Dl + j];
    __syncthreads();
    const uint4 * w = (const uint4 *) (kv_b + ((int64_t) h * (Dk + Dv) + Dk + i) * Dl);
    float acc = 0.f;
    for (int j8 = 0; j8 < Dl / 8; ++j8) {
        const uint4 raw = __ldg(w + j8);
        const half2 * h2 = (const half2 *) &raw;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __half22float2(h2[k]);
            acc += f.x * os[j8 * 8 + 2 * k] + f.y * os[j8 * 8 + 2 * k + 1];
        }
    }
    o[((int64_t) t * H + h) * Dv + i] = acc;
}
void mla_out(const float * olat, const half * kv_b, float * o, int T, int H, int Dk, int Dv, int Dl, cudaStream_t s) {
    mla_out_k<<<dim3(T, H), Dv, Dl * sizeof(float), s>>>(olat, kv_b, o, H, Dk, Dv, Dl);
}

// key i of row t: dense rows walk cells 0..q; sparse rows the selected pools' cells, then the tail
__device__ __forceinline__ int dsa_cell(int i, const int32_t * sel, int nsel, int r, int tail_start) {
    const int nsc = nsel * r;
    return i < nsc ? sel[i / r] * r + i % r : tail_start + (i - nsc);
}

constexpr int kMlaSplits = 64;

// Split-KV decoding over the latent: grid (T H, splits), 8 warps; lane holds 16 of Dl = 512 dims; K = V = the row.
template <bool Q8>
__global__ void __launch_bounds__(256) mla_attn_partial_k(const float * qabs, const half * cache, const int32_t * sel,
                                                          const int32_t * n_sel, int H, int Dl, int r, int K,
                                                          const int * pos0p, float * part_o, float * part_ml, int nsplit) {
    const int th = blockIdx.x, t = th / H, split = blockIdx.y;
    const int qpos = *pos0p + t;
    const int ns = sel ? n_sel[t] : 0;
    const int tail_start = (qpos + 1) / r * r;
    const int n_keys = ns == 0 ? qpos + 1 : ns * r + (qpos + 1 - tail_start);
    const int kps = max(64, (n_keys + nsplit - 1) / nsplit);
    const int k0 = split * kps, k1 = min(n_keys, k0 + kps);
    const int32_t * sl = sel ? sel + (int64_t) t * K : nullptr;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    float qv[16];
    const float * qp = qabs + (int64_t) th * Dl + lane * 16;
#pragma unroll
    for (int d = 0; d < 16; ++d) qv[d] = qp[d];
    float m = -FLT_MAX, l = 0.f, acc[16];
#pragma unroll
    for (int d = 0; d < 16; ++d) acc[d] = 0.f;
    for (int i = k0 + w; i < k1; i += 8) {
        const int key = ns == 0 ? i : dsa_cell(i, sl, ns, r, tail_start);
        const uint4 ra = kv_load8<Q8>(cache, key, Dl, lane * 16), rb = kv_load8<Q8>(cache, key, Dl, lane * 16 + 8);
        const half2 * ka = (const half2 *) &ra, * kb = (const half2 *) &rb;
        float f[16];
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const float2 a = __half22float2(ka[d]), b = __half22float2(kb[d]);
            f[2 * d] = a.x; f[2 * d + 1] = a.y; f[8 + 2 * d] = b.x; f[8 + 2 * d + 1] = b.y;
        }
        float sdot = 0.f;
#pragma unroll
        for (int d = 0; d < 16; ++d) sdot += qv[d] * f[d];
        sdot = wsum(sdot);
        const float mn = fmaxf(m, sdot);
        const float corr = __expf(m - mn), p = __expf(sdot - mn);
        l = l * corr + p;
#pragma unroll
        for (int d = 0; d < 16; ++d) acc[d] = acc[d] * corr + p * f[d];
        m = mn;
    }
    __shared__ float sm[8], slr[8], so[8][512];
    if (lane == 0) { sm[w] = m; slr[w] = l; }
#pragma unroll
    for (int d = 0; d < 16; ++d) so[w][lane * 16 + d] = acc[d];
    __syncthreads();
    if (w == 0) {
        float M = -FLT_MAX;
        for (int i = 0; i < 8; ++i) M = fmaxf(M, sm[i]);
        float L = 0.f, o[16];
#pragma unroll
        for (int d = 0; d < 16; ++d) o[d] = 0.f;
        for (int i = 0; i < 8; ++i) {
            if (slr[i] == 0.f) continue;
            const float c = __expf(sm[i] - M);
            L += slr[i] * c;
#pragma unroll
            for (int d = 0; d < 16; ++d) o[d] += so[i][lane * 16 + d] * c;
        }
        float * po = part_o + ((int64_t) th * nsplit + split) * Dl + lane * 16;
#pragma unroll
        for (int d = 0; d < 16; ++d) po[d] = o[d];
        if (lane == 0) {
            part_ml[((int64_t) th * nsplit + split) * 2 + 0] = M;
            part_ml[((int64_t) th * nsplit + split) * 2 + 1] = L;
        }
    }
}

__global__ void mla_attn_combine_k(const float * part_o, const float * part_ml, float * out, int Dl, int nsplit) {
    const int th = blockIdx.x, i = threadIdx.x;
    float M = -FLT_MAX;
    for (int s = 0; s < nsplit; ++s)
        if (part_ml[((int64_t) th * nsplit + s) * 2 + 1] > 0.f) M = fmaxf(M, part_ml[((int64_t) th * nsplit + s) * 2]);
    float L = 0.f, o = 0.f;
    for (int s = 0; s < nsplit; ++s) {
        const float l = part_ml[((int64_t) th * nsplit + s) * 2 + 1];
        if (l == 0.f) continue;
        const float c = __expf(part_ml[((int64_t) th * nsplit + s) * 2] - M);
        L += l * c;
        o += part_o[((int64_t) th * nsplit + s) * Dl + i] * c;
    }
    out[(int64_t) th * Dl + i] = o / L;
}

void mla_attention(const float * qabs, const half * cache, const int32_t * sel, const int32_t * n_sel, float * olat,
                   int T, int H, int Dl, int r, int K, const int * pos0, float * scratch, cudaStream_t s) {
    if (Dl != 512) { fprintf(stderr, "mla: latent width %d unsupported\n", Dl); abort(); }
    const int ns = kMlaSplits;
    float * part_o = scratch;
    float * part_ml = scratch + (size_t) T * H * ns * Dl;
    (kv_q8() ? mla_attn_partial_k<true> : mla_attn_partial_k<false>)<<<dim3(T * H, ns), 256, 0, s>>>(
        qabs, cache, sel, n_sel, H, Dl, r, K, pos0, part_o, part_ml, ns);
    mla_attn_combine_k<<<T * H, Dl, 0, s>>>(part_o, part_ml, olat, Dl, ns);
}

// prompt rows: one warp per (t, h), lane holds 16 dims
template <bool Q8>
__global__ void mla_attn_prefill_k(const float * qabs, const half * cache, const int32_t * sel, const int32_t * n_sel,
                                   float * olat, int t0, int t1, int H, int Dl, int r, int K, int pos0) {
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int t = t0 + w / H, h = w % H;
    if (t >= t1) return;
    const int qpos = pos0 + t;
    const int ns = sel ? n_sel[t] : 0;
    const int tail_start = (qpos + 1) / r * r;
    const int n_keys = ns == 0 ? qpos + 1 : ns * r + (qpos + 1 - tail_start);
    const int32_t * sl = sel ? sel + (int64_t) t * K : nullptr;
    float qv[16];
    const float * qp = qabs + ((int64_t) t * H + h) * Dl + lane * 16;
#pragma unroll
    for (int d = 0; d < 16; ++d) qv[d] = qp[d];
    float m = -FLT_MAX, l = 0.f, acc[16];
#pragma unroll
    for (int d = 0; d < 16; ++d) acc[d] = 0.f;
    for (int i = 0; i < n_keys; ++i) {
        const int key = ns == 0 ? i : dsa_cell(i, sl, ns, r, tail_start);
        const uint4 ra = kv_load8<Q8>(cache, key, Dl, lane * 16), rb = kv_load8<Q8>(cache, key, Dl, lane * 16 + 8);
        const half2 * ka = (const half2 *) &ra, * kb = (const half2 *) &rb;
        float f[16];
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const float2 a = __half22float2(ka[d]), b = __half22float2(kb[d]);
            f[2 * d] = a.x; f[2 * d + 1] = a.y; f[8 + 2 * d] = b.x; f[8 + 2 * d + 1] = b.y;
        }
        float sdot = 0.f;
#pragma unroll
        for (int d = 0; d < 16; ++d) sdot += qv[d] * f[d];
        sdot = wsum(sdot);
        const float mn = fmaxf(m, sdot);
        const float corr = __expf(m - mn), p = __expf(sdot - mn);
        l = l * corr + p;
#pragma unroll
        for (int d = 0; d < 16; ++d) acc[d] = acc[d] * corr + p * f[d];
        m = mn;
    }
    float * o = olat + ((int64_t) t * H + h) * Dl + lane * 16;
#pragma unroll
    for (int d = 0; d < 16; ++d) o[d] = acc[d] / l;
}
void mla_attention_prefill(const float * qabs, const half * cache, const int32_t * sel, const int32_t * n_sel,
                           float * olat, int t0, int t1, int H, int Dl, int r, int K, int pos0, cudaStream_t s) {
    if (Dl != 512) { fprintf(stderr, "mla: latent width %d unsupported\n", Dl); abort(); }
    const int warps = (t1 - t0) * H;
    if (warps <= 0) return;
    (kv_q8() ? mla_attn_prefill_k<true> : mla_attn_prefill_k<false>)<<<(warps + 7) / 8, 256, 0, s>>>(
        qabs, cache, sel, n_sel, olat, t0, t1, H, Dl, r, K, pos0);
}

// ------------------------------------------------------------------------------------------- DSA indexer
// grid T, block id: LayerNorm of the key, the gate logits beside it (fp16)
__global__ void dsa_store_k(const float * ix, int ldx, const float * ln_w, const float * ln_b, half * kraw, int id, int ih,
                            const int * pos0p, float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.x, i = threadIdx.x;
    const float * x = ix + (int64_t) t * ldx;
    const float v = x[i];
    const float mean = bsum(v, sh) / id;
    const float dv = v - mean;
    const float var = bsum(dv * dv, sh) / id;
    const float y = dv * rsqrtf(var + eps) * ln_w[i] + ln_b[i];
    half * row = kraw + (int64_t) (*pos0p + t) * 2 * id;
    row[i] = __float2half(y);
    row[id + i] = __float2half(x[id + ih + i]);
}
void dsa_store(const float * ix, int ldx, const float * ln_w, const float * ln_b, half * kraw, int T, int id, int ih,
               const int * pos0, float eps, cudaStream_t s) {
    dsa_store_k<<<T, id, 0, s>>>(ix, ldx, ln_w, ln_b, kraw, id, ih, pos0, eps);
}

// grid: the pools to (re)compute, block id threads (one dim each): r <= 8
__global__ void dsa_pool_k(const half * kraw, const float * ape, float * pooled, int id, int r, int b0) {
    const int b = b0 + blockIdx.x, d = threadIdx.x;
    float lg[8], kk[8];
    float mx = -FLT_MAX;
    for (int j = 0; j < r; ++j) {
        const half * row = kraw + (int64_t) (b * r + j) * 2 * id;
        kk[j] = __half2float(row[d]);
        lg[j] = __half2float(row[id + d]) + ape[j * id + d];
        mx = fmaxf(mx, lg[j]);
    }
    float sum = 0.f, acc = 0.f;
    for (int j = 0; j < r; ++j) {
        const float p = __expf(lg[j] - mx);
        sum += p;
        acc += p * kk[j];
    }
    pooled[(int64_t) b * id + d] = acc / sum;
}
// the first pool touching the window to the last one completed inside it: a host-side range needs pos0, so the
// kernel takes it from device memory and bounds its grid by the largest possible count
__global__ void dsa_pool_win_k(const half * kraw, const float * ape, float * pooled, int id, int r, int T, const int * pos0p) {
    const int pos0 = *pos0p;
    const int b0 = pos0 / r, nb = (pos0 + T) / r;
    const int b = b0 + blockIdx.x, d = threadIdx.x;
    if (b >= nb) return;
    float lg[8], kk[8];
    float mx = -FLT_MAX;
    for (int j = 0; j < r; ++j) {
        const half * row = kraw + (int64_t) (b * r + j) * 2 * id;
        kk[j] = __half2float(row[d]);
        lg[j] = __half2float(row[id + d]) + ape[j * id + d];
        mx = fmaxf(mx, lg[j]);
    }
    float sum = 0.f, acc = 0.f;
    for (int j = 0; j < r; ++j) {
        const float p = __expf(lg[j] - mx);
        sum += p;
        acc += p * kk[j];
    }
    pooled[(int64_t) b * id + d] = acc / sum;
}
void dsa_pool(const half * kraw, const float * ape, float * pooled, int T, int id, int r, const int * pos0, cudaStream_t s) {
    if (r > 8) { fprintf(stderr, "dsa: pool size %d unsupported\n", r); abort(); }
    dsa_pool_win_k<<<T / r + 2, id, 0, s>>>(kraw, ape, pooled, id, r, T, pos0);
}

// grid (pools of 256, T): one thread per pool, the row's queries and head weights in shared memory
__global__ void dsa_score_k(const float * q, int64_t ldq, const float * w, int64_t ldw, const float * pooled,
                            float * scores, int max_blocks, int ih, int id, int r, int dense_cells, const int * pos0p,
                            int t_off) {
    extern __shared__ float qs[];   // ih * id, then ih weights
    const int t = blockIdx.y;
    const int qpos = *pos0p + t_off + t;
    if (qpos + 1 <= dense_cells) return;
    const int nb = (qpos + 1) / r;
    float * ws = qs + ih * id;
    const float qscale = rsqrtf((float) id);
    for (int i = threadIdx.x; i < ih * id; i += blockDim.x) qs[i] = q[(int64_t) t * ldq + i] * qscale;
    for (int i = threadIdx.x; i < ih; i += blockDim.x) ws[i] = w[(int64_t) t * ldw + i] * rsqrtf((float) ih);
    __syncthreads();
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float4 * kp = (const float4 *) (pooled + (int64_t) b * id);
    float acc[32];
#pragma unroll
    for (int h = 0; h < 32; ++h) acc[h] = 0.f;
    for (int i = 0; i < id / 4; ++i) {
        const float4 kv = __ldg(kp + i);
#pragma unroll
        for (int h = 0; h < 32; ++h) {
            if (h < ih) {
                const float * qh = qs + h * id + 4 * i;
                acc[h] += qh[0] * kv.x + qh[1] * kv.y + qh[2] * kv.z + qh[3] * kv.w;
            }
        }
    }
    float sc = 0.f;
#pragma unroll
    for (int h = 0; h < 32; ++h)
        if (h < ih) sc += ws[h] * fmaxf(acc[h], 0.f);
    // the head weights have either sign, so can the score: the radix top-k compares the bits as unsigned, so
    // store them in the order-preserving unsigned form (positive: sign bit set; negative: all bits flipped)
    unsigned u = __float_as_uint(sc);
    u = (sc >= 0.f) ? (u | 0x80000000u) : ~u;
    scores[(int64_t) t * max_blocks + b] = __uint_as_float(u);
}
void dsa_score(const float * q, int64_t ldq, const float * w, int64_t ldw, const float * pooled, float * scores,
               int max_blocks, int T, int ih, int id, int r, int dense_cells, const int * pos0, int t_off, cudaStream_t s) {
    if (ih > 32 || id % 4) { fprintf(stderr, "dsa: %d indexer heads of %d unsupported\n", ih, id); abort(); }
    dsa_score_k<<<dim3((max_blocks + 255) / 256, T), 256, (ih * id + ih) * sizeof(float), s>>>(
        q, ldq, w, ldw, pooled, scores, max_blocks, ih, id, r, dense_cells, pos0, t_off);
}

// ------------------------------------------------------------------------------------------- routing
// one warp per token: sigmoid scores; the k best by score + bias; weights = the scores, normalized, scaled
template <int PER>
__global__ void route_sigmoid_k(const float * logits, const float * bias, int n_exp, int k, int32_t * ids, float * w,
                                float w_scale, int norm, int T) {
    const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= T) return;
    const float * lg = logits + (int64_t) t * n_exp;
    float v[PER], ch[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int e = lane + 32 * i;
        v[i] = e < n_exp ? sigm(lg[e]) : 0.f;
        ch[i] = e < n_exp ? v[i] + bias[e] : -FLT_MAX;
    }
    float wsum_ = 0.f, myw = 0.f;
    int myid = 0;
    for (int j = 0; j < k; ++j) {
        float best = -FLT_MAX;
        int bi = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < PER; ++i)
            if (ch[i] > best) { best = ch[i]; bi = lane + 32 * i; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, best, o);
            const int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        float sv = 0.f;
#pragma unroll
        for (int i = 0; i < PER; ++i)
            if (bi == lane + 32 * i) { sv = v[i]; ch[i] = -FLT_MAX; }
        sv = wsum(sv);   // the winner's score (one lane holds it)
        wsum_ += sv;
        if (lane == j) { myw = sv; myid = bi; }
    }
    if (lane < k) {
        const float sc = (w_scale != 0.f ? w_scale : 1.f) / (norm ? wsum_ + 1e-20f : 1.f);
        ids[t * k + lane] = myid;
        w[t * k + lane] = myw * sc;
    }
}
void route_sigmoid_topk(const float * logits, const float * bias, int T, int n_exp, int k, int32_t * ids, float * w,
                        float w_scale, bool norm, cudaStream_t s) {
    if (n_exp > 1024 || k > 32) { fprintf(stderr, "route_sigmoid: unsupported shape\n"); abort(); }
    const int tpb = 8, blocks = (T + tpb - 1) / tpb;
    if (n_exp <= 512) route_sigmoid_k<16><<<blocks, 32 * tpb, 0, s>>>(logits, bias, n_exp, k, ids, w, w_scale, norm, T);
    else route_sigmoid_k<32><<<blocks, 32 * tpb, 0, s>>>(logits, bias, n_exp, k, ids, w, w_scale, norm, T);
}

// ------------------------------------------------------------------------------------------- swiglu / norms
__global__ void swiglu_clamp_k(const float * g, const float * u, float * h, int64_t n, float limit) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        float gv = g[i], uv = u[i];
        if (limit > 0.f) {
            gv = fminf(gv, limit);
            uv = fminf(fmaxf(uv, -limit), limit);
        }
        h[i] = silu(gv) * uv;
    }
}
void swiglu_clamp(const float * g, const float * u, float * h, int64_t n, float limit, cudaStream_t s) {
    swiglu_clamp_k<<<nblk(n, 256), 256, 0, s>>>(g, u, h, n, limit);
}

// rows of 2F (g | u) -> h [T][F] then quantize_act: two launches (fused later)
void quantize_act_swiglu(const float * gu, int64_t ldx, int T, int64_t F, float limit, ActQ8 & a, cudaStream_t s) {
    (void) gu; (void) ldx; (void) T; (void) F; (void) limit; (void) a; (void) s;
    fprintf(stderr, "quantize_act_swiglu: not implemented\n");
    abort();
}

__global__ void layernorm_rows_k(const float * x, const float * w, const float * b, float * y, int n, int64_t ldx,
                                 int64_t ldy, float eps) {
    __shared__ float sh[32];
    const float * xr = x + (int64_t) blockIdx.x * ldx;
    float s1 = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s1 += xr[i];
    const float mean = bsum(s1, sh) / n;
    float s2 = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const float d = xr[i] - mean; s2 += d * d; }
    const float inv = rsqrtf(bsum(s2, sh) / n + eps);
    float * yr = y + (int64_t) blockIdx.x * ldy;
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = (xr[i] - mean) * inv * w[i] + (b ? b[i] : 0.f);
}
void layernorm_rows(const float * x, const float * w, const float * b, float * y, int rows, int n, int64_t ldx, int64_t ldy,
                    float eps, cudaStream_t s) {
    layernorm_rows_k<<<rows, 128, 0, s>>>(x, w, b, y, n, ldx, ldy, eps);
}

}  // namespace bnk
