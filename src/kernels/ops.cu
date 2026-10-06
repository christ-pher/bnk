#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "kernels/ops.h"

namespace bnk {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
// Block-wide sum; `sh` needs 32 floats.
__device__ __forceinline__ float block_sum(float v, float * sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = threadIdx.x < nw ? sh[threadIdx.x] : 0.f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.f + __expf(-x)); }

static inline int nblk(int64_t n, int t) { return (int) ((n + t - 1) / t); }

// ------------------------------------------------------------------------------------------- HC
__global__ void hc_init_k(const float * x, float * res, int hc, int E) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hc * E; i += gridDim.x * blockDim.x)
        res[(int64_t) t * hc * E + i] = x[(int64_t) t * E + i % E];
}
void hc_init(const float * x, float * res, int T, int hc, int E, cudaStream_t s) {
    hc_init_k<<<dim3(nblk(hc * E, 256), T), 256, 0, s>>>(x, res, hc, E);
}

// one block per (t, stream)
__global__ void rmsnorm_rows_k(const float * x, const float * w, float * y, int n, int64_t ldx, int64_t ldy,
                               int64_t wstride_rows, float eps) {
    __shared__ float sh[32];
    const int r = blockIdx.x;
    const float * xr = x + r * ldx;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / n + eps);
    const float * wr = w ? w + (r % wstride_rows) * n : nullptr;
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[r * ldy + i] = xr[i] * inv * (wr ? wr[i] : 1.f);
}

void hc_norm(const float * res, const float * w, float * xn, int T, int hc, int E, float eps, cudaStream_t s) {
    // rows are (t, s); weight row = s
    rmsnorm_rows_k<<<T * hc, 256, 0, s>>>(res, w, xn, E, E, E, hc, eps);
}

void rmsnorm_rows(const float * x, const float * w, float * y, int rows, int n, int64_t ldx, int64_t ldy, float eps,
                  cudaStream_t s) {
    rmsnorm_rows_k<<<rows, n >= 256 ? 256 : 128, 0, s>>>(x, w, y, n, ldx, ldy, 1, eps);
}

__global__ void silu_scale_k(float * x, int64_t n, float scale) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        x[i] = siluf_(x[i] * scale);
}
void silu_scale(float * x, int64_t n, float scale, cudaStream_t s) {
    silu_scale_k<<<nblk(n, 256), 256, 0, s>>>(x, n, scale);
}

__global__ void hc_mix_k(const float * xn, const float * g, float * mixed, int hc, int E) {
    const int t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    float acc = 0.f;
    for (int s = 0; s < hc; ++s) {
        const int64_t i = (int64_t) t * hc * E + s * E + e;
        acc += xn[i] * sigmoidf_(g[i]);
    }
    mixed[(int64_t) t * E + e] = acc / hc;
}
void hc_mix(const float * xn, const float * g, float * mixed, int T, int hc, int E, cudaStream_t s) {
    hc_mix_k<<<dim3(nblk(E, 256), T), 256, 0, s>>>(xn, g, mixed, hc, E);
}

__global__ void hc_combine_k(float * res, const float * out, const float * inj, int hc, int E) {
    const int t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    const float o = out[(int64_t) t * E + e];
    for (int s = 0; s < hc; ++s) {
        const float w = 2.f * sigmoidf_(inj[t * hc + s] / hc);
        res[(int64_t) t * hc * E + s * E + e] += o * w;
    }
}
void hc_combine(float * res, const float * out, const float * inj, int T, int hc, int E, cudaStream_t s) {
    hc_combine_k<<<dim3(nblk(E, 256), T), 256, 0, s>>>(res, out, inj, hc, E);
}

// ------------------------------------------------------------------------------------------- generic
__global__ void silu_mul_k(const float * g, const float * u, float * h, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        h[i] = siluf_(g[i]) * u[i];
}
void silu_mul(const float * g, const float * u, float * h, int64_t n, cudaStream_t s) {
    silu_mul_k<<<nblk(n, 256), 256, 0, s>>>(g, u, h, n);
}

__global__ void add_scaled_k(float * y, const float * x, const float * sc, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[(int64_t) t * n + i] += x[(int64_t) t * n + i] * (sc ? sc[t] : 1.f);
}
void add_scaled(float * y, const float * x, const float * sc, int T, int n, cudaStream_t s) {
    add_scaled_k<<<dim3(nblk(n, 256), T), 256, 0, s>>>(y, x, sc, n);
}

void copy_f32(float * dst, const float * src, int64_t n, cudaStream_t s) {
    cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, s);
}

__global__ void shift_rows_k(float * buf, int64_t row_elems, int from, int n) {
    for (int64_t c = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; c < row_elems; c += (int64_t) gridDim.x * blockDim.x)
        for (int i = 0; i < n; ++i) buf[(int64_t) i * row_elems + c] = buf[(int64_t) (from + i) * row_elems + c];
}
__global__ void shift_rows_multi_k(float * const * bufs, int64_t row_elems, int from, int n) {
    float * buf = bufs[blockIdx.y];
    for (int64_t c = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; c < row_elems; c += (int64_t) gridDim.x * blockDim.x)
        for (int i = 0; i < n; ++i) buf[(int64_t) i * row_elems + c] = buf[(int64_t) (from + i) * row_elems + c];
}
void shift_rows_multi(float * const * bufs, int nbuf, int64_t row_elems, int from, int n, cudaStream_t s) {
    if (from == 0 || n <= 0 || nbuf <= 0) return;
    shift_rows_multi_k<<<dim3(nblk(row_elems, 256), nbuf), 256, 0, s>>>(bufs, row_elems, from, n);
}

void shift_rows(float * buf, int64_t row_elems, int from, int n, cudaStream_t s) {
    if (from == 0 || n <= 0) return;
    shift_rows_k<<<nblk(row_elems, 256), 256, 0, s>>>(buf, row_elems, from, n);
}

__global__ void sigmoid_k(float * x, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        x[i] = sigmoidf_(x[i]);
}
void sigmoid_inplace(float * x, int64_t n, cudaStream_t s) { sigmoid_k<<<nblk(n, 256), 256, 0, s>>>(x, n); }

// ------------------------------------------------------------------------------------------- GDN
__global__ void gdn_conv_k(const float * in, const float * w, float * out, int T, int C, int K) {
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += in[(int64_t) (t + k) * C + c] * w[c * K + k];
    out[(int64_t) t * C + c] = siluf_(acc);
}
void gdn_conv(const float * conv_in, const float * w, float * out, int T, int C, int K, cudaStream_t s) {
    gdn_conv_k<<<dim3(nblk(C, 256), T), 256, 0, s>>>(conv_in, w, out, T, C, K);
}

// grid (T, nk + nk) blocks for q/k heads (l2norm), plus a tail block per token for g/beta
__global__ void gdn_prep_k(float * co, int C, int nk, int nv, int S, float * alpha_g, float * beta, const float * dt,
                           const float * a, float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.y;
    const int h = blockIdx.x;
    if (h < 2 * nk) {
        float * x = co + (int64_t) t * C + h * S;  // q heads then k heads, contiguous
        float ss = 0.f;
        for (int i = threadIdx.x; i < S; i += blockDim.x) ss += x[i] * x[i];
        ss = block_sum(ss, sh);
        const float inv = rsqrtf(ss + eps);
        for (int i = threadIdx.x; i < S; i += blockDim.x) x[i] *= inv;
    } else {
        for (int i = threadIdx.x; i < nv; i += blockDim.x) {
            const float b = beta[t * nv + i];
            beta[t * nv + i] = sigmoidf_(b);
            const float z = alpha_g[t * nv + i] + dt[i];
            const float sp = z > 20.f ? z : log1pf(__expf(z));
            alpha_g[t * nv + i] = sp * a[i];
        }
    }
}
void gdn_prep(float * conv_out, int T, int C, int nk, int nv, int S, float * alpha_g, float * beta, const float * dt,
              const float * a, float eps, cudaStream_t s) {
    gdn_prep_k<<<dim3(2 * nk + 1, T), 128, 0, s>>>(conv_out, C, nk, nv, S, alpha_g, beta, dt, a, eps);
}

// grid (nv, S/32): each block owns 32 columns j of one head's S (S[:, j]) in registers.
// The state is stored row-major per head, state[h][i*S + j] = S[i][j], so the column loads coalesce.
template <int S>
__global__ void __launch_bounds__(32) gdn_rec_k(const float * __restrict__ co, int C, const float * __restrict__ g,
                                                const float * __restrict__ beta, float * __restrict__ state,
                                                float * __restrict__ out, int T, int nk, int nv, int commit) {
    const int h = blockIdx.x;
    const int j = blockIdx.y * 32 + threadIdx.x;
    const int hk = h % nk;
    __shared__ float qs[S], ks[S];
    float col[S];
    float * st = state + (int64_t) h * S * S + j;
#pragma unroll
    for (int i = 0; i < S; ++i) col[i] = st[(int64_t) i * S];
    const float scale = rsqrtf((float) S);
    for (int t = 0; t < T; ++t) {
        const float * row = co + (int64_t) t * C;
        __syncwarp();
        for (int i = threadIdx.x; i < S; i += 32) {
            qs[i] = row[hk * S + i];
            ks[i] = row[nk * S + hk * S + i];
        }
        __syncwarp();
        const float v = row[2 * nk * S + h * S + j];
        const float decay = __expf(g[t * nv + h]);
        const float b = beta[t * nv + h];
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < S; ++i) {
            col[i] *= decay;
            kv += col[i] * ks[i];
        }
        const float delta = (v - kv) * b;
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < S; ++i) {
            col[i] += ks[i] * delta;
            o += col[i] * qs[i];
        }
        if (out) out[((int64_t) t * nv + h) * S + j] = o * scale;
        if (t + 1 == commit) {
#pragma unroll
            for (int i = 0; i < S; ++i) st[(int64_t) i * S] = col[i];
        }
    }
}
void gdn_recurrence(const float * conv_out, int C, const float * g, const float * beta, float * state, float * out,
                    int T, int nk, int nv, int S, int commit, cudaStream_t s) {
    if (S != 128) { fprintf(stderr, "gdn: unsupported state size %d\n", S); abort(); }
    gdn_rec_k<128><<<dim3(nv, 128 / 32), 32, 0, s>>>(conv_out, C, g, beta, state, out, T, nk, nv, commit);
}

__global__ void gated_rmsnorm_k(const float * o, const float * z, const float * w, float * y, int S, float eps) {
    __shared__ float sh[32];
    const int64_t r = blockIdx.x;  // (t, h)
    const float * x = o + r * S;
    float ss = 0.f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / S + eps);
    for (int i = threadIdx.x; i < S; i += blockDim.x) y[r * S + i] = x[i] * inv * w[i] * sigmoidf_(z[r * S + i]);
}
void gated_rmsnorm(const float * o, const float * z, const float * w, float * y, int T, int H, int S, float eps,
                   cudaStream_t s) {
    gated_rmsnorm_k<<<T * H, 128, 0, s>>>(o, z, w, y, S, eps);
}

// ------------------------------------------------------------------------------------------- attention
// grid (T, H + Hkv), block D threads
__global__ void attn_prep_k(const float * qfull, const float * k, const float * v, const float * qn, const float * kn,
                            float * q, half * kc, half * vc, int H, int Hkv, int D, int n_rot, float base,
                            const int * pos0p, float eps) {
    const int pos0 = *pos0p;
    __shared__ float sh[32];
    __shared__ float buf[512];
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int i = threadIdx.x;
    const bool isq = h < H;
    const int hk = h - H;
    const float x = isq ? qfull[((int64_t) t * H + h) * 2 * D + i] : k[((int64_t) t * Hkv + hk) * D + i];
    const float ss = block_sum(x * x, sh);
    const float y = x * rsqrtf(ss / D + eps) * (isq ? qn[i] : kn[i]);
    buf[i] = y;
    __syncthreads();
    float r = y;
    const int half_rot = n_rot / 2;
    if (i < n_rot) {
        const int p = i < half_rot ? i : i - half_rot;
        const float theta = (float) (pos0 + t) * powf(base, -2.f * p / n_rot);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        r = i < half_rot ? buf[i] * cs - buf[i + half_rot] * sn : buf[i] * cs + buf[i - half_rot] * sn;
    }
    if (isq) {
        q[((int64_t) t * H + h) * D + i] = r;
    } else {
        const int64_t row = ((int64_t) (pos0 + t) * Hkv + hk) * D + i;
        kc[row] = __float2half(r);
        vc[row] = __float2half(v[((int64_t) t * Hkv + hk) * D + i]);
    }
}
void attn_prep(const float * qfull, const float * k, const float * v, const float * qn, const float * kn, float * q,
               half * kcache, half * vcache, int T, int H, int Hkv, int D, int n_rot, float base, const int * pos0,
               float eps, cudaStream_t s) {
    attn_prep_k<<<dim3(T, H + Hkv), D, 0, s>>>(qfull, k, v, qn, kn, q, kcache, vcache, H, Hkv, D, n_rot, base, pos0,
                                                eps);
}

// Split-KV flash decoding. grid (T*H, nsplit), 256 threads = 8 warps; each warp takes keys in turn.
// lane holds dims [lane*8, lane*8+8) of D=256.
__global__ void attn_partial_k(const float * q, const half * kc, const half * vc, int H, int Hkv, int D,
                               const int * pos0p, float scale, float * part_o, float * part_ml, int nsplit) {
    const int th = blockIdx.x;
    const int t = th / H, h = th % H;
    const int hk = h / (H / Hkv);
    const int split = blockIdx.y;
    const int n_kv = *pos0p + t + 1;
    // fixed split count; small contexts leave the tail splits empty
    const int keys_per_split = max(64, (n_kv + nsplit - 1) / nsplit);
    const int k0 = split * keys_per_split;
    const int k1 = min(n_kv, k0 + keys_per_split);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    float qv[8];
    const float * qp = q + ((int64_t) t * H + h) * D + lane * 8;
#pragma unroll
    for (int d = 0; d < 8; ++d) qv[d] = qp[d] * scale;
    float m = -FLT_MAX, l = 0.f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int key = k0 + w; key < k1; key += 8) {
        const half * kp = kc + ((int64_t) key * Hkv + hk) * D + lane * 8;
        const uint4 kraw = *(const uint4 *) kp;
        const half2 * k2 = (const half2 *) &kraw;
        float sdot = 0.f;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            float2 f = __half22float2(k2[d]);
            sdot += qv[2 * d] * f.x + qv[2 * d + 1] * f.y;
        }
        sdot = warp_sum(sdot);
        const float mn = fmaxf(m, sdot);
        const float corr = __expf(m - mn), p = __expf(sdot - mn);
        l = l * corr + p;
        const uint4 vraw = *(const uint4 *) (vc + ((int64_t) key * Hkv + hk) * D + lane * 8);
        const half2 * v2 = (const half2 *) &vraw;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            float2 f = __half22float2(v2[d]);
            acc[2 * d] = acc[2 * d] * corr + p * f.x;
            acc[2 * d + 1] = acc[2 * d + 1] * corr + p * f.y;
        }
        m = mn;
    }
    // combine the 8 warps
    __shared__ float sm[8], sl[8], so[8][256];
    if (lane == 0) { sm[w] = m; sl[w] = l; }
#pragma unroll
    for (int d = 0; d < 8; ++d) so[w][lane * 8 + d] = acc[d];
    __syncthreads();
    if (w == 0) {
        float M = -FLT_MAX;
        for (int i = 0; i < 8; ++i) M = fmaxf(M, sm[i]);
        float L = 0.f, o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < 8; ++i) {
            if (sl[i] == 0.f) continue;
            const float c = __expf(sm[i] - M);
            L += sl[i] * c;
#pragma unroll
            for (int d = 0; d < 8; ++d) o[d] += so[i][lane * 8 + d] * c;
        }
        float * po = part_o + ((int64_t) th * nsplit + split) * D + lane * 8;
#pragma unroll
        for (int d = 0; d < 8; ++d) po[d] = o[d];
        if (lane == 0) {
            part_ml[((int64_t) th * nsplit + split) * 2 + 0] = M;
            part_ml[((int64_t) th * nsplit + split) * 2 + 1] = L;
        }
    }
}

__global__ void attn_combine_k(const float * part_o, const float * part_ml, const float * qfull, float * out, int H,
                               int D, int nsplit) {
    const int th = blockIdx.x;
    const int t = th / H, h = th % H;
    const int i = threadIdx.x;
    float M = -FLT_MAX;
    for (int s = 0; s < nsplit; ++s)
        if (part_ml[((int64_t) th * nsplit + s) * 2 + 1] > 0.f) M = fmaxf(M, part_ml[((int64_t) th * nsplit + s) * 2]);
    float L = 0.f, o = 0.f;
    for (int s = 0; s < nsplit; ++s) {
        const float l = part_ml[((int64_t) th * nsplit + s) * 2 + 1];
        if (l == 0.f) continue;
        const float c = __expf(part_ml[((int64_t) th * nsplit + s) * 2] - M);
        L += l * c;
        o += part_o[((int64_t) th * nsplit + s) * D + i] * c;
    }
    const float gate = qfull[((int64_t) t * H + h) * 2 * D + D + i];
    out[((int64_t) t * H + h) * D + i] = (o / L) * sigmoidf_(gate);
}

constexpr int kAttnSplits = 64;

size_t attention_scratch_floats(int T, int H, int D, int) { return (size_t) T * H * kAttnSplits * (D + 2); }

void attention(const float * q, const half * kcache, const half * vcache, const float * qfull_gate, float * out, int T,
               int H, int Hkv, int D, const int * pos0, float scale, float * scratch, cudaStream_t s) {
    const int ns = kAttnSplits;
    float * part_o = scratch;
    float * part_ml = scratch + (size_t) T * H * ns * D;
    attn_partial_k<<<dim3(T * H, ns), 256, 0, s>>>(q, kcache, vcache, H, Hkv, D, pos0, scale, part_o, part_ml, ns);
    attn_combine_k<<<T * H, D, 0, s>>>(part_o, part_ml, qfull_gate, out, H, D, ns);
}

// One warp per (t, h); D = 256 (8 dims per lane). Online softmax over the causal keys, then the output gate.
__global__ void attn_prefill_k(const float * q, const half * kc, const half * vc, const float * qfull, float * out,
                               int T, int H, int Hkv, int D, int pos0, float scale) {
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (w >= T * H) return;
    const int t = w / H, h = w % H, hk = h / (H / Hkv);
    float qv[8];
    const float * qp = q + ((int64_t) t * H + h) * D + lane * 8;
#pragma unroll
    for (int d = 0; d < 8; ++d) qv[d] = qp[d] * scale;
    float m = -FLT_MAX, l = 0.f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int n_kv = pos0 + t + 1;
    for (int key = 0; key < n_kv; ++key) {
        const uint4 kraw = *(const uint4 *) (kc + ((int64_t) key * Hkv + hk) * D + lane * 8);
        const half2 * k2 = (const half2 *) &kraw;
        float s = 0.f;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const float2 f = __half22float2(k2[d]);
            s += qv[2 * d] * f.x + qv[2 * d + 1] * f.y;
        }
        s = warp_sum(s);
        const float mn = fmaxf(m, s);
        const float corr = __expf(m - mn), p = __expf(s - mn);
        l = l * corr + p;
        const uint4 vraw = *(const uint4 *) (vc + ((int64_t) key * Hkv + hk) * D + lane * 8);
        const half2 * v2 = (const half2 *) &vraw;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const float2 f = __half22float2(v2[d]);
            acc[2 * d] = acc[2 * d] * corr + p * f.x;
            acc[2 * d + 1] = acc[2 * d + 1] * corr + p * f.y;
        }
        m = mn;
    }
    const float * gate = qfull + ((int64_t) t * H + h) * 2 * D + D + lane * 8;
    float * o = out + ((int64_t) t * H + h) * D + lane * 8;
#pragma unroll
    for (int d = 0; d < 8; ++d) o[d] = acc[d] / l * sigmoidf_(gate[d]);
}
void attention_prefill(const float * q, const half * kcache, const half * vcache, const float * qfull_gate,
                       float * out, int T, int H, int Hkv, int D, int pos0, float scale, cudaStream_t s) {
    if (D != 256) { fprintf(stderr, "attention_prefill: head dim %d unsupported\n", D); abort(); }
    const int warps = T * H;
    attn_prefill_k<<<(warps + 7) / 8, 256, 0, s>>>(q, kcache, vcache, qfull_gate, out, T, H, Hkv, D, pos0, scale);
}

// ------------------------------------------------------------------------------------------- routing
// one warp per token, n_exp <= 1024: softmax over all experts, top-k by repeated warp argmax
__global__ void route_topk_k(const float * logits, int n_exp, int k, int32_t * ids, float * w, float w_scale) {
    const int t = blockIdx.x;
    const int lane = threadIdx.x;
    const float * lg = logits + (int64_t) t * n_exp;
    constexpr int PER = 32;  // n_exp / 32 values per lane
    float v[PER];
    const int per = (n_exp + 31) / 32;
    float mx = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        v[i] = (i < per && lane + 32 * i < n_exp) ? lg[lane + 32 * i] : -FLT_MAX;
        mx = fmaxf(mx, v[i]);
    }
    mx = warp_max(mx);
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        v[i] = v[i] == -FLT_MAX ? 0.f : __expf(v[i] - mx);
        sum += v[i];
    }
    sum = warp_sum(sum);
    float wsum = 0.f;
    float myw = 0.f;
    int myid = 0;
    for (int j = 0; j < k; ++j) {
        float best = -1.f;
        int bi = 0;
#pragma unroll
        for (int i = 0; i < PER; ++i)
            if (v[i] > best) { best = v[i]; bi = lane + 32 * i; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, best, o);
            const int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if ((bi & 31) == lane) v[bi >> 5] = -2.f;
        const float p = best / sum;
        wsum += p;
        if (lane == j) { myw = p; myid = bi; }
    }
    if (lane < k) {
        const float sc = (w_scale != 0.f ? w_scale : 1.f) / wsum;
        ids[t * k + lane] = myid;
        w[t * k + lane] = myw * sc;
    }
}
void route_topk(const float * logits, int T, int n_exp, int k, int32_t * ids, float * w, float w_scale,
                cudaStream_t s) {
    if (n_exp > 1024 || k > 32) { fprintf(stderr, "route_topk: unsupported shape\n"); abort(); }
    route_topk_k<<<T, 32, 0, s>>>(logits, n_exp, k, ids, w, w_scale);
}

// ------------------------------------------------------------------------------------------- PLE
// one block per (t, stream)
__global__ void ple_gate_k(const float * key, const float * res, const float * value, const float * wk,
                           const float * wq, const float * wc, float * gated, float * normalized, int hc, int E,
                           float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.x / hc, s = blockIdx.x % hc;
    const float * kr = key + ((int64_t) t * hc + s) * E;
    const float * qr = res + ((int64_t) t * hc + s) * E;
    float sk = 0.f, sq = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) { sk += kr[i] * kr[i]; sq += qr[i] * qr[i]; }
    sk = block_sum(sk, sh);
    sq = block_sum(sq, sh);
    const float ik = rsqrtf(sk / E + eps), iq = rsqrtf(sq / E + eps);
    float dot = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) dot += (kr[i] * ik * wk[s * E + i]) * (qr[i] * iq * wq[s * E + i]);
    dot = block_sum(dot, sh) / sqrtf((float) E);
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float sg = dot > 0.f ? 1.f : (dot < 0.f ? -1.f : 0.f);
    const float gate = sigmoidf_(sg * mag);
    float * gr = gated + ((int64_t) t * hc + s) * E;
    float sg2 = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        const float v = value[(int64_t) t * E + i] * gate;
        gr[i] = v;
        sg2 += v * v;
    }
    sg2 = block_sum(sg2, sh);
    const float ig = rsqrtf(sg2 / E + eps);
    float * nr = normalized + ((int64_t) t * hc + s) * E;
    for (int i = threadIdx.x; i < E; i += blockDim.x) nr[i] = gr[i] * ig * wc[s * E + i];
}
void ple_gate(const float * key, const float * res, const float * value, const float * w_key, const float * w_query,
              const float * w_conv, float * gated, float * normalized, int T, int hc, int E, float eps,
              cudaStream_t s) {
    ple_gate_k<<<T * hc, 256, 0, s>>>(key, res, value, w_key, w_query, w_conv, gated, normalized, hc, E, eps);
}

__global__ void ple_conv_add_k(float * res, const float * gated, const float * hn, const half * w, int C, int K,
                               int dil) {
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int hist = (K - 1) * dil;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += __half2float(w[c * K + k]) * hn[(int64_t) (hist + t - (K - 1 - k) * dil) * C + c];
    const int64_t i = (int64_t) t * C + c;
    res[i] += gated[i] + siluf_(acc);
}
void ple_conv_add(float * res, const float * gated, const float * hist_new, const half * w, int T, int C, int K,
                  int dil, cudaStream_t s) {
    ple_conv_add_k<<<dim3(nblk(C, 256), T), 256, 0, s>>>(res, gated, hist_new, w, C, K, dil);
}

// ------------------------------------------------------------------------------------------- head
__global__ void argmax_k(const float * lg, int n, int32_t * out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float * x = lg + (int64_t) blockIdx.x * n;
    float best = -FLT_MAX;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > best) { best = x[i]; idx = i; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, best, o);
        const int oi = __shfl_xor_sync(0xffffffff, idx, o);
        if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { bv[wid] = best; bi[wid] = idx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int q = 1; q < (int) (blockDim.x >> 5); ++q)
            if (bv[q] > bv[0] || (bv[q] == bv[0] && bi[q] < bi[0])) { bv[0] = bv[q]; bi[0] = bi[q]; }
        out[blockIdx.x] = bi[0];
    }
}
// float -> unsigned with the same order (negative values flipped entirely, positive ones get the top bit)
__device__ __forceinline__ unsigned order_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One block (1024 threads) per row: radix select of the K-th largest key (8 bits at a time from the top), then
// everything above it and the lowest-index equal ones (each thread scans a contiguous range; block scans place
// the picks). The same selection as qsa_topk_k, on order-preserving keys.
__global__ void __launch_bounds__(1024) topk_rows_k(const float * x, int n, int K, int32_t * ids, float * vals) {
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix_s, want_s;
    const int t = blockIdx.x;
    const float * row = x + (int64_t) t * n;
    int32_t * oid = ids + (int64_t) t * K;
    float * ov = vals + (int64_t) t * K;
    unsigned prefix = 0, mask = 0, want = K;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const unsigned v = order_key(row[i]);
            if ((v & mask) == prefix) atomicAdd(&hist[(v >> shift) & 255], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {   // the digit whose bin holds the want-th largest (256 bins: serial is fine)
            unsigned acc = 0;
            int d = 255;
            for (; d > 0; --d) {
                if (acc + hist[d] >= want) break;
                acc += hist[d];
            }
            prefix_s = prefix | ((unsigned) d << shift);
            want_s = want - acc;
        }
        __syncthreads();
        prefix = prefix_s;
        want = want_s;
        mask |= 255u << shift;
        __syncthreads();
    }
    __shared__ int sa[1024], se[1024];
    __shared__ int wa[32], we[32];
    const int per = (n + blockDim.x - 1) / blockDim.x;
    const int b0 = threadIdx.x * per, b1 = min(n, b0 + per);
    int na = 0, ne = 0;
    for (int b = b0; b < b1; ++b) {
        const unsigned v = order_key(row[b]);
        na += v > prefix;
        ne += v == prefix;
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int a = na, e = ne;
    for (int o = 1; o < 32; o <<= 1) {
        const int va = __shfl_up_sync(0xffffffff, a, o), ve = __shfl_up_sync(0xffffffff, e, o);
        if (lane >= o) { a += va; e += ve; }
    }
    if (lane == 31) { wa[wid] = a; we[wid] = e; }
    __syncthreads();
    if (wid == 0) {
        const int nw = blockDim.x >> 5;
        int p = lane < nw ? wa[lane] : 0, q = lane < nw ? we[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int vp = __shfl_up_sync(0xffffffff, p, o), vq = __shfl_up_sync(0xffffffff, q, o);
            if (lane >= o) { p += vp; q += vq; }
        }
        if (lane < nw) { wa[lane] = p; we[lane] = q; }
    }
    __syncthreads();
    int ia = (wid ? wa[wid - 1] : 0) + a - na, ie = (wid ? we[wid - 1] : 0) + e - ne;
    for (int b = b0; b < b1; ++b) {
        const float f = row[b];
        const unsigned v = order_key(f);
        if (v > prefix) {
            const int slot = ia + min(ie, (int) want);
            oid[slot] = b;
            ov[slot] = f;
            ++ia;
        } else if (v == prefix) {
            if (ie < (int) want) {
                oid[ia + ie] = b;
                ov[ia + ie] = f;
            }
            ++ie;
        }
    }
}

void topk_rows(const float * x, int T, int n, int K, int32_t * ids, float * vals, cudaStream_t s) {
    if (T <= 0 || K <= 0) return;
    topk_rows_k<<<T, 1024, 0, s>>>(x, n, K, ids, vals);
}

void argmax_rows(const float * logits, int T, int n, int32_t * out, cudaStream_t s) {
    argmax_k<<<T, 1024, 0, s>>>(logits, n, out);
}

__global__ void argmax_prob_k(const float * x, int n, int32_t * out, float * prob) {
    __shared__ float bv[32], sh[32];
    __shared__ int bi[32];
    float best = -FLT_MAX;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > best) { best = x[i]; idx = i; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, best, o);
        const int oi = __shfl_xor_sync(0xffffffff, idx, o);
        if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { bv[wid] = best; bi[wid] = idx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int q = 1; q < (int) (blockDim.x >> 5); ++q)
            if (bv[q] > bv[0] || (bv[q] == bv[0] && bi[q] < bi[0])) { bv[0] = bv[q]; bi[0] = bi[q]; }
    }
    __syncthreads();
    const float mx = bv[0];
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += __expf(x[i] - mx);
    s = block_sum(s, sh);
    if (threadIdx.x == 0) {
        out[0] = bi[0];
        prob[0] = 1.f / s;
    }
}
// Many-block version: each block its local max (lowest index on ties) and sum of exp relative to it; one small
// block combines them (sum_b s_b * exp(m_b - M)). Partials live in fixed device memory: graph-capture safe, and
// only the drafter (one stream) calls this.
constexpr int kAmaxBlocks = 96;
__device__ float g_amax_m[kAmaxBlocks], g_amax_s[kAmaxBlocks];
__device__ int g_amax_i[kAmaxBlocks];
__global__ void argmax_prob_part_k(const float * x, int n) {
    __shared__ float bv[32], sh[32];
    __shared__ int bi[32];
    float best = -FLT_MAX;
    int idx = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (x[i] > best) { best = x[i]; idx = i; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, best, o);
        const int oi = __shfl_xor_sync(0xffffffff, idx, o);
        if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { bv[wid] = best; bi[wid] = idx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int q = 1; q < (int) (blockDim.x >> 5); ++q)
            if (bv[q] > bv[0] || (bv[q] == bv[0] && bi[q] < bi[0])) { bv[0] = bv[q]; bi[0] = bi[q]; }
    }
    __syncthreads();
    const float mx = bv[0];
    float s = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += __expf(x[i] - mx);
    s = block_sum(s, sh);
    if (threadIdx.x == 0) {
        g_amax_m[blockIdx.x] = mx;
        g_amax_i[blockIdx.x] = bi[0];
        g_amax_s[blockIdx.x] = s;
    }
}
__global__ void argmax_prob_fin_k(int nb, int32_t * out, float * prob) {
    if (threadIdx.x != 0) return;
    float M = g_amax_m[0];
    int I = g_amax_i[0];
    for (int b = 1; b < nb; ++b)
        if (g_amax_m[b] > M || (g_amax_m[b] == M && g_amax_i[b] < I)) { M = g_amax_m[b]; I = g_amax_i[b]; }
    float s = 0.f;
    for (int b = 0; b < nb; ++b) s += g_amax_s[b] * __expf(g_amax_m[b] - M);
    out[0] = I;
    prob[0] = 1.f / s;
}
void argmax_prob(const float * logits, int n, int32_t * id, float * prob, cudaStream_t s) {
    static const bool one = getenv("BNK_ARGMAX_ONE") != nullptr;
    if (one) {
        argmax_prob_k<<<1, 1024, 0, s>>>(logits, n, id, prob);
        return;
    }
    const int nb = std::min(kAmaxBlocks, std::max(1, (n + 2047) / 2048));
    argmax_prob_part_k<<<nb, 256, 0, s>>>(logits, n);
    argmax_prob_fin_k<<<1, 32, 0, s>>>(nb, id, prob);
}

__global__ void add_bcast_k(float * R, const float * e, int hc, int E) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hc * E; i += gridDim.x * blockDim.x)
        R[(int64_t) t * hc * E + i] += e[(int64_t) t * E + i % E];
}
void add_bcast_streams(float * R, const float * e, int T, int hc, int E, cudaStream_t s) {
    add_bcast_k<<<dim3(nblk(hc * E, 256), T), 256, 0, s>>>(R, e, hc, E);
}

__global__ void gather_bytes_rows_k(const uint8_t * src, size_t rb, const int32_t * ids, int n, uint8_t * dst) {
    const uint4 * sp = (const uint4 *) (src + (size_t) ids[blockIdx.x] * rb);
    uint4 * dp = (uint4 *) (dst + (size_t) blockIdx.x * rb);
    for (size_t k = threadIdx.x; k < rb / 16; k += blockDim.x) dp[k] = sp[k];
}
void gather_bytes_rows(const uint8_t * src, size_t row_bytes, const int32_t * ids, int n, uint8_t * dst, cudaStream_t s) {
    if (n > 0) gather_bytes_rows_k<<<n, 256, 0, s>>>(src, row_bytes, ids, n, dst);
}

__global__ void map_id_k(int32_t * id, const int32_t * table) { id[0] = table[id[0]]; }
void map_id(int32_t * id, const int32_t * table, cudaStream_t s) { map_id_k<<<1, 1, 0, s>>>(id, table); }

}  // namespace bnk
