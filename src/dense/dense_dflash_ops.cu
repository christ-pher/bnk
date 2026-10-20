// The DFlash2 drafter's own kernels (dense_dflash.h): the grouped two-tap convolution over a block's rows, per-head
// RMSNorm + RoPE, the K/V ring, the block's attention (non-causal, sliding window) and the candidate selector.
#include <cfloat>

#include "core/util.h"
#include "dense/dense_dflash_ops.h"

namespace bnk {

// out[t][c] = sum_tap (base[side][tap][c] + dyn[t][side][tap][c / gsize]) * in[t - tap][c] (rows before the block's
// first are zero); dyn rows hold [2 sides][2 taps][E / gsize]
__global__ void dflash_conv_k(const float * __restrict__ in, const float * __restrict__ dyn, const float * __restrict__ base,
                              float * __restrict__ out, int E, int gsize, int side) {
    const int t = blockIdx.y, G = E / gsize;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < E; c += gridDim.x * blockDim.x) {
        const int g = c / gsize;
        const float * dr = dyn + (size_t) t * 4 * G + side * 2 * G;
        float acc = (base[(side * 2 + 0) * E + c] + dr[g]) * in[(size_t) t * E + c];
        if (t > 0) acc += (base[(side * 2 + 1) * E + c] + dr[G + g]) * in[(size_t) (t - 1) * E + c];
        out[(size_t) t * E + c] = acc;
    }
}

void dflash_conv(const float * in, const float * dyn, const float * base, float * out, int T, int E, int gsize, int side,
                 cudaStream_t s) {
    dflash_conv_k<<<dim3((E + 255) / 256, T), 256, 0, s>>>(in, dyn, base, out, E, gsize, side);
}

// one (row, head) per block, D/2 threads: x = rms(x) * w, then RoPE (neox halves) at position *pos0 + row
__global__ void dflash_headnorm_rope_k(float * __restrict__ x, int nh, int D, const float * __restrict__ w,
                                       const int * __restrict__ pos0, float theta, float eps) {
    const int row = blockIdx.x / nh, hd = blockIdx.x % nh, i = threadIdx.x, half_d = D / 2;
    float * v = x + ((size_t) row * nh + hd) * D;
    const float a = v[i], b = v[i + half_d];
    float ss = a * a + b * b;
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    if ((i & 31) == 0) red[i >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
    for (int k = 0; k < (half_d + 31) / 32; ++k) tot += red[k];
    const float r = rsqrtf(tot / D + eps);
    const float x1 = a * r * w[i], x2 = b * r * w[i + half_d];
    const double ang = (double) (*pos0 + row) * pow((double) theta, -2.0 * i / D);
    const float cs = (float) cos(ang), sn = (float) sin(ang);
    v[i] = x1 * cs - x2 * sn;
    v[i + half_d] = x2 * cs + x1 * sn;
}

void dflash_headnorm_rope(float * x, int rows, int nh, int D, const float * w, const int * pos0_dev, float theta,
                          float eps, cudaStream_t s) {
    dflash_headnorm_rope_k<<<rows * nh, D / 2, 0, s>>>(x, nh, D, w, pos0_dev, theta, eps);
}

// rows' K and V (f32 [rows][n]) into the ring (f16 [ring][n]) at positions *pos0 + row (mod ring)
__global__ void dflash_kv_store_k(const float * __restrict__ k, const float * __restrict__ v, half * __restrict__ kr,
                                  half * __restrict__ vr, int n, int ring, const int * __restrict__ pos0) {
    const int row = blockIdx.y;
    const size_t slot = (size_t) ((*pos0 + row) & (ring - 1));
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        kr[slot * n + i] = __float2half(k[(size_t) row * n + i]);
        vr[slot * n + i] = __float2half(v[(size_t) row * n + i]);
    }
}

void dflash_kv_store(const float * k, const float * v, half * kring, half * vring, int rows, int n, int ring,
                     const int * pos0_dev, cudaStream_t s) {
    dflash_kv_store_k<<<dim3((n + 255) / 256, rows), 256, 0, s>>>(k, v, kring, vring, n, ring, pos0_dev);
}

// ---- the block's attention: R query rows (positions s..s+R-1) x H heads over the keys at positions lo..lo+nkeys-1
// (context, then the block itself), all in the ring; key p visible to query q when |q - p| < win (not causal).
// S[h][r][k] = q . k / sqrt(D): a warp per key, its 4 dims per lane, every (head of the group, row) pair in turn
__global__ void dflash_scores_k(const float * __restrict__ q, const half * __restrict__ kr, float * __restrict__ S,
                                const DFlashParams * __restrict__ par, int R, int H, int HKV, int D, int ring, int win,
                                int maxk, float scale) {
    const int lane = threadIdx.x & 31, kk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), kh = blockIdx.y;
    const int nkeys = par->nkeys, lo = par->lo, s0 = par->s;
    if (kk >= nkeys) return;
    const int kp = lo + kk;
    const half * kv = kr + ((size_t) (kp & (ring - 1)) * HKV + kh) * D;
    const int G = H / HKV;
    for (int d0 = 0; d0 < D; d0 += 128) {   // D = 128: one pass
        const int d = d0 + lane * 4;
        const float2 k01 = __half22float2(*(const __half2 *) (kv + d)), k23 = __half22float2(*(const __half2 *) (kv + d + 2));
        for (int j = 0; j < G; ++j) {
            const int h = kh * G + j;
            for (int r = 0; r < R; ++r) {
                const float4 qv = *(const float4 *) (q + ((size_t) r * H + h) * D + d);
                float dot = qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y;
                for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, o);
                if (lane == 0) {
                    const int qp = s0 + r, dist = qp > kp ? qp - kp : kp - qp;
                    S[((size_t) h * R + r) * maxk + kk] = dist < win ? dot * scale : -FLT_MAX;
                }
            }
        }
    }
}

// each (head, row): softmax over its keys, in place
__global__ void dflash_softmax_k(float * __restrict__ S, const DFlashParams * __restrict__ par, int maxk) {
    float * row = S + (size_t) blockIdx.x * maxk;
    const int n = par->nkeys;
    __shared__ float red[32];
    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, row[i]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = -FLT_MAX;
    for (int k = 0; k < (int) (blockDim.x >> 5); ++k) m = fmaxf(m, red[k]);
    __syncthreads();
    float z = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = row[i] == -FLT_MAX ? 0.f : __expf(row[i] - m);
        row[i] = e;
        z += e;
    }
    for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffff, z, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = z;
    __syncthreads();
    z = 0.f;
    for (int k = 0; k < (int) (blockDim.x >> 5); ++k) z += red[k];
    const float inv = 1.f / z;
    for (int i = threadIdx.x; i < n; i += blockDim.x) row[i] *= inv;
}

// out[r][h][d] = sum_k P[h][r][k] V[key k][kh][d]: a block per (head, row), a thread per dim
__global__ void dflash_pv_k(const float * __restrict__ S, const half * __restrict__ vr, float * __restrict__ out,
                            const DFlashParams * __restrict__ par, int R, int H, int HKV, int D, int ring, int maxk) {
    const int h = blockIdx.x / R, r = blockIdx.x % R, d = threadIdx.x, kh = h / (H / HKV);
    const float * p = S + ((size_t) h * R + r) * maxk;
    const int n = par->nkeys, lo = par->lo;
    float acc = 0.f;
    for (int k = 0; k < n; ++k) {
        const float w = p[k];
        if (w != 0.f) acc += w * __half2float(vr[((size_t) ((lo + k) & (ring - 1)) * HKV + kh) * D + d]);
    }
    out[((size_t) r * H + h) * D + d] = acc;
}

void dflash_attention(const float * q, const half * kring, const half * vring, float * S, float * out,
                      const DFlashParams * par_dev, int R, int H, int HKV, int D, int ring, int win, int maxk,
                      cudaStream_t s) {
    dflash_scores_k<<<dim3((maxk + 7) / 8, HKV), 256, 0, s>>>(q, kring, S, par_dev, R, H, HKV, D, ring, win, maxk,
                                                              rsqrtf((float) D));
    dflash_softmax_k<<<H * R, 256, 0, s>>>(S, par_dev, maxk);
    dflash_pv_k<<<H * R, D, 0, s>>>(S, vring, out, par_dev, R, H, HKV, D, ring, maxk);
}

// ---- the selector: one warp; for each of the n positions, candidate j (lane < K) scores
// unary[j] + sum_r pre[prev][r] * hp[r] * succ[cand_j][r]; the best becomes prev
__global__ void dflash_select_k(const int32_t * __restrict__ cand, const float * __restrict__ unary,
                                const float * __restrict__ hp, const half * __restrict__ pre, const half * __restrict__ succ,
                                const DFlashParams * __restrict__ par, int n, int K, int rank, int32_t * __restrict__ out,
                                float * __restrict__ prob, float * __restrict__ qout) {
    const int lane = threadIdx.x;
    int prev = par->anchor;
    const float temp = par->temp;
    for (int i = 0; i < n; ++i) {
        float sc = -FLT_MAX;
        int c = -1;
        if (lane < K) {
            c = cand[i * K + lane];
            const half * pp = pre + (size_t) prev * rank;
            const half * sp = succ + (size_t) c * rank;
            const float * hh = hp + (size_t) i * rank;
            float acc = 0.f;
            for (int r = 0; r < rank; ++r) acc += __half2float(pp[r]) * hh[r] * __half2float(sp[r]);
            sc = unary[i * K + lane] + acc;
        }
        float best = sc;
        int bl = lane;
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffff, best, o);
            const int ol = __shfl_xor_sync(0xffffffff, bl, o);
            if (ob > best || (ob == best && ol < bl)) best = ob, bl = ol;
        }
        if (temp > 0.f) {   // sampled: q = softmax(scores / temp) over the candidates, drawn with the uniform u[i]
            const float w = lane < K ? __expf((sc - best) / temp) : 0.f;
            float ws = w;
            for (int o = 16; o > 0; o >>= 1) ws += __shfl_xor_sync(0xffffffff, ws, o);
            float cum = w;   // inclusive prefix over the lanes
            for (int o = 1; o < 32; o <<= 1) {
                const float t = __shfl_up_sync(0xffffffff, cum, o);
                if (lane >= o) cum += t;
            }
            const float target = par->u[i] * ws;
            const unsigned hit = __ballot_sync(0xffffffff, lane < K && cum >= target);
            bl = hit ? __ffs(hit) - 1 : 0;
            if (lane < K) qout[i * K + lane] = w / ws;
        }
        // prob: the chosen candidate's share of the draft logits' softmax over the K (the selector's own softmax
        // is near 1 almost always; this one tracks how often the target accepts)
        const float u = lane < K ? unary[i * K + lane] : -FLT_MAX;
        float um = u;
        for (int o = 16; o > 0; o >>= 1) um = fmaxf(um, __shfl_xor_sync(0xffffffff, um, o));
        float e = lane < K ? __expf(u - um) : 0.f;
        for (int o = 16; o > 0; o >>= 1) e += __shfl_xor_sync(0xffffffff, e, o);
        const float uc = __shfl_sync(0xffffffff, u, bl);
        prev = __shfl_sync(0xffffffff, c, bl);
        if (lane == 0) {
            out[i] = prev;
            prob[i] = __expf(uc - um) / e;
        }
    }
}

void dflash_select(const int32_t * cand, const float * unary, const float * hp, const half * pre, const half * succ,
                   const DFlashParams * par_dev, int n, int K, int rank, int32_t * out, float * prob, float * qout,
                   cudaStream_t s) {
    dflash_select_k<<<1, 32, 0, s>>>(cand, unary, hp, pre, succ, par_dev, n, K, rank, out, prob, qout);
}

}  // namespace bnk
