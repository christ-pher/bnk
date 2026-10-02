#include <cfloat>
#include <cstdio>
#include <cstdlib>

#include <mma.h>

#include "kernels/qsa.h"

namespace bnk {

__device__ __forceinline__ float qsa_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float qsa_block_sum(float v, float * sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = qsa_warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = threadIdx.x < nw ? sh[threadIdx.x] : 0.f;
    if (wid == 0) v = qsa_warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}
__device__ __forceinline__ float qsa_sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }

// NeoX rotation of x[0..n_rot) in shared memory at position pos; thread i owns element i
__device__ __forceinline__ float qsa_rope(const float * x, int i, int n_rot, float base, int pos) {
    if (i >= n_rot) return x[i];
    const int half = n_rot / 2;
    const int p = i < half ? i : i - half;
    const float theta = (float) pos * powf(base, -2.f * p / n_rot);
    float sn, cs;
    sincosf(theta, &sn, &cs);
    return i < half ? x[i] * cs - x[i + half] * sn : x[i] * cs + x[i - half] * sn;
}

// ------------------------------------------------------------------------------------------- keys
__global__ void qsa_store_k(const float * k, half * kraw, int id, const int * pos0p) {
    const int t = blockIdx.x;
    const int pos = *pos0p + t;
    for (int i = threadIdx.x; i < id; i += blockDim.x) kraw[(int64_t) pos * id + i] = __float2half(k[(int64_t) t * id + i]);
}
void qsa_store_keys(const float * k, half * kraw, int T, int id, const int * pos0, cudaStream_t s) {
    qsa_store_k<<<T, 128, 0, s>>>(k, kraw, id, pos0);
}

// grid: candidate blocks [floor(pos0/r), floor(pos0/r) + gridDim.x); the ones ending inside the window pool
__global__ void qsa_pool_k(const half * kraw, const float * k_norm, float * pooled, int T, int id, int r, int n_rot,
                           float base, float eps, const int * pos0p) {
    __shared__ float sh[32];
    __shared__ float x[512];
    const int pos0 = *pos0p;
    const int b = pos0 / r + blockIdx.x;
    const int end = b * r + r - 1;
    if (end < pos0 || end > pos0 + T - 1) return;
    const int i = threadIdx.x;
    float m = 0.f;
    if (i < id) {
        for (int j = 0; j < r; ++j) m += __half2float(kraw[(int64_t) (b * r + j) * id + i]);
        m /= r;
    }
    const float ss = qsa_block_sum(i < id ? m * m : 0.f, sh);
    if (i < id) x[i] = m * rsqrtf(ss / id + eps) * k_norm[i];
    __syncthreads();
    if (i < id) pooled[(int64_t) b * id + i] = qsa_rope(x, i, n_rot, base, b * r);
}
void qsa_pool(const half * kraw, const float * k_norm, float * pooled, int T, const QsaShape & sh, const int * pos0,
              cudaStream_t s) {
    qsa_pool_k<<<T / sh.ratio + 2, 128, 0, s>>>(kraw, k_norm, pooled, T, sh.id, sh.ratio, sh.n_rot, sh.rope_base,
                                                sh.eps, pos0);
}

// grid (T, ih), id threads
__global__ void qsa_q_k(float * q, const float * q_norm, int ih, int id, int n_rot, float base, float eps,
                        const int * pos0p) {
    __shared__ float sh[32];
    __shared__ float x[512];
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    float * qp = q + ((int64_t) t * ih + h) * id;
    const float v = i < id ? qp[i] : 0.f;
    const float ss = qsa_block_sum(v * v, sh);
    if (i < id) x[i] = v * rsqrtf(ss / id + eps) * q_norm[i];
    __syncthreads();
    if (i < id) qp[i] = qsa_rope(x, i, n_rot, base, *pos0p + t);
}
void qsa_queries(float * q, const float * q_norm, int T, const QsaShape & sh, const int * pos0, cudaStream_t s) {
    qsa_q_k<<<dim3(T, sh.ih), 128, 0, s>>>(q, q_norm, sh.ih, sh.id, sh.n_rot, sh.rope_base, sh.eps, pos0);
}

// ------------------------------------------------------------------------------------------- selection
// grid (blocks of 256 candidate blocks, T): score[t][b] = sum_h relu(q[t][h] . pooled[b])
__global__ void qsa_score_k(const float * q, const float * pooled, float * scores, int max_blocks, int ih, int id,
                            int r, int dense_cells, const int * pos0p, int t_off) {
    extern __shared__ float qs[];  // ih * id
    const int t = blockIdx.y;
    const int qpos = *pos0p + t_off + t;
    if (qpos + 1 <= dense_cells) return;
    const int nb = (qpos + 1) / r;  // complete blocks before the tail
    for (int i = threadIdx.x; i < ih * id; i += blockDim.x) qs[i] = q[(int64_t) t * ih * id + i];
    __syncthreads();
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float4 * kp = (const float4 *) (pooled + (int64_t) b * id);
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < id / 4; ++i) {
        const float4 kv = __ldg(kp + i);
#pragma unroll
        for (int h = 0; h < 8; ++h) {
            if (h < ih) {
                const float * qh = qs + h * id + 4 * i;
                acc[h] += qh[0] * kv.x + qh[1] * kv.y + qh[2] * kv.z + qh[3] * kv.w;
            }
        }
    }
    float sc = 0.f;
#pragma unroll
    for (int h = 0; h < 8; ++h)
        if (h < ih) sc += fmaxf(acc[h], 0.f);
    scores[(int64_t) t * max_blocks + b] = sc;
}

// one block (1024 threads) per t: radix select of the top K scores (non-negative floats compare as uints)
__global__ void qsa_topk_k(const float * scores, int max_blocks, int32_t * sel, int32_t * n_sel, int K, int r,
                           int dense_cells, const int * pos0p, int t_off) {
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix_s, want_s;
    const int t = blockIdx.x;
    const int qpos = *pos0p + t_off + t;
    int32_t * out = sel + (int64_t) t * K;
    if (qpos + 1 <= dense_cells) {
        if (threadIdx.x == 0) n_sel[t] = 0;
        return;
    }
    const int nb = (qpos + 1) / r;
    const unsigned * sc = (const unsigned *) (scores + (int64_t) t * max_blocks);
    if (nb <= K) {
        for (int b = threadIdx.x; b < nb; b += blockDim.x) out[b] = b;
        if (threadIdx.x == 0) n_sel[t] = nb;
        return;
    }
    // find the K-th largest key: digits from the top
    unsigned prefix = 0, mask = 0;
    unsigned want = K;  // how many still to take from elements matching the prefix
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int b = threadIdx.x; b < nb; b += blockDim.x) {
            const unsigned v = sc[b];
            if ((v & mask) == prefix) atomicAdd(&hist[(v >> shift) & 255], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
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
    // prefix is the K-th largest value: take everything above it and the first `want` equal ones, in block order
    // (each thread scans a contiguous range; a block-wide exclusive scan places its picks)
    __shared__ int sa[1024], se[1024];
    const int per = (nb + blockDim.x - 1) / blockDim.x;
    const int b0 = threadIdx.x * per, b1 = min(nb, b0 + per);
    int na = 0, ne = 0;
    for (int b = b0; b < b1; ++b) {
        const unsigned v = sc[b];
        na += v > prefix;
        ne += v == prefix;
    }
    sa[threadIdx.x] = na;
    se[threadIdx.x] = ne;
    __syncthreads();
    if (threadIdx.x == 0) {
        int ca = 0, ce = 0;
        for (int i = 0; i < (int) blockDim.x; ++i) {
            const int a = sa[i], e = se[i];
            sa[i] = ca;
            se[i] = ce;
            ca += a;
            ce += e;
        }
    }
    __syncthreads();
    // slot of a pick = (#above before it) + min(#equal before it, want): block order, equal ones past `want` dropped
    int ia = sa[threadIdx.x], ie = se[threadIdx.x];
    for (int b = b0; b < b1; ++b) {
        const unsigned v = sc[b];
        if (v > prefix) {
            out[ia + min(ie, (int) want)] = b;
            ++ia;
        } else if (v == prefix) {
            if (ie < (int) want) out[ia + ie] = b;
            ++ie;
        }
    }
    if (threadIdx.x == 0) n_sel[t] = K;
}

// Prompt rows: the same scores as a register-tiled fp32 product over 64 query rows x 64 blocks, the indexer heads'
// relu summed in the epilogue. Each pooled key is read once per 64 queries instead of once per query.
// 256 threads, each owning a 4 x 4 patch of the tile.
__global__ void __launch_bounds__(256) qsa_score_tiled_k(const float * __restrict__ q, const float * __restrict__ pooled,
                                                         float * __restrict__ scores, int T, int max_blocks, int ih,
                                                         int id, int r, int dense_cells, const int * pos0p, int t_off) {
    constexpr int TM = 64, TN = 64, TK = 32;
    __shared__ float Qs[TK][TM + 4];
    __shared__ float Ps[TK][TN + 4];
    const int pos0 = *pos0p + t_off;
    const int m0 = blockIdx.y * TM, b0 = blockIdx.x * TN;
    const int last = min(T, m0 + TM) - 1;
    if (last < 0 || b0 >= (pos0 + last + 1) / r) return;  // no row of this tile reaches these blocks
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    float sum[4][4] = {};
    for (int h = 0; h < ih; ++h) {
        float acc[4][4] = {};
        for (int k0 = 0; k0 < id; k0 += TK) {
            for (int i = threadIdx.x; i < TM * TK; i += 256) {
                const int m = i / TK, kk = i % TK, t = m0 + m;
                Qs[kk][m] = t < T ? q[((int64_t) t * ih + h) * id + k0 + kk] : 0.f;
            }
            for (int i = threadIdx.x; i < TN * TK; i += 256) {
                const int n = i / TK, kk = i % TK, b = b0 + n;
                Ps[kk][n] = b < max_blocks ? pooled[(int64_t) b * id + k0 + kk] : 0.f;
            }
            __syncthreads();
#pragma unroll 8
            for (int kk = 0; kk < TK; ++kk) {
                const float4 a = *(const float4 *) &Qs[kk][ty * 4];
                const float4 b = *(const float4 *) &Ps[kk][tx * 4];
                const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) sum[i][j] += fmaxf(acc[i][j], 0.f);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = m0 + ty * 4 + i;
        if (t >= T) continue;
        const int qpos = pos0 + t;
        if (qpos + 1 <= dense_cells) continue;
        const int nb = (qpos + 1) / r;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int b = b0 + tx * 4 + j;
            if (b < nb) scores[(int64_t) t * max_blocks + b] = sum[i][j];
        }
    }
}

void qsa_select(const float * q, const float * pooled, float * scores, int max_blocks, int32_t * sel, int32_t * n_sel,
                int T, const QsaShape & sh, const int * pos0, int t_off, cudaStream_t s) {
    static const bool old = getenv("BNK_QSA_OLD") != nullptr;
    if (T > 8 && sh.id % 32 == 0 && !old)  // prompt rows: tiled
        qsa_score_tiled_k<<<dim3((max_blocks + 63) / 64, (T + 63) / 64), 256, 0, s>>>(
            q, pooled, scores, T, max_blocks, sh.ih, sh.id, sh.ratio, sh.dense_cells(), pos0, t_off);
    else  // a decode window: one pass per query over the blocks
        qsa_score_k<<<dim3((max_blocks + 255) / 256, T), 256, sh.ih * sh.id * sizeof(float), s>>>(
            q, pooled, scores, max_blocks, sh.ih, sh.id, sh.ratio, sh.dense_cells(), pos0, t_off);
    qsa_topk_k<<<T, 1024, 0, s>>>(scores, max_blocks, sel, n_sel, sh.top_blocks, sh.ratio, sh.dense_cells(), pos0,
                                  t_off);
}

// ------------------------------------------------------------------------------------------- attention
// key i of row t: dense rows walk cells 0..q; sparse rows the selected blocks' cells, then the tail
__device__ __forceinline__ int qsa_cell(int i, const int32_t * sel, int nsel, int r, int tail_start) {
    const int nsc = nsel * r;
    return i < nsc ? sel[i / r] * r + i % r : tail_start + (i - nsc);
}

constexpr int kQsaSplits = 64;

__global__ void qsa_attn_partial_k(const float * q, const half * kc, const half * vc, const int32_t * sel,
                                   const int32_t * n_sel, int H, int Hkv, int D, int r, int K, const int * pos0p,
                                   float scale, float * part_o, float * part_ml, int nsplit) {
    const int th = blockIdx.x;
    const int t = th / H, h = th % H;
    const int hk = h / (H / Hkv);
    const int split = blockIdx.y;
    const int qpos = *pos0p + t;
    const int ns = n_sel[t];
    const int tail_start = (qpos + 1) / r * r;
    const int n_keys = ns == 0 ? qpos + 1 : ns * r + (qpos + 1 - tail_start);
    const int kps = max(64, (n_keys + nsplit - 1) / nsplit);
    const int k0 = split * kps, k1 = min(n_keys, k0 + kps);
    const int32_t * sl = sel + (int64_t) t * K;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    float qv[8];
    const float * qp = q + ((int64_t) t * H + h) * D + lane * 8;
#pragma unroll
    for (int d = 0; d < 8; ++d) qv[d] = qp[d] * scale;
    float m = -FLT_MAX, l = 0.f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = k0 + w; i < k1; i += 8) {
        const int key = ns == 0 ? i : qsa_cell(i, sl, ns, r, tail_start);
        const uint4 kraw = *(const uint4 *) (kc + ((int64_t) key * Hkv + hk) * D + lane * 8);
        const half2 * k2 = (const half2 *) &kraw;
        float sdot = 0.f;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const float2 f = __half22float2(k2[d]);
            sdot += qv[2 * d] * f.x + qv[2 * d + 1] * f.y;
        }
        sdot = qsa_warp_sum(sdot);
        const float mn = fmaxf(m, sdot);
        const float corr = __expf(m - mn), p = __expf(sdot - mn);
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
    __shared__ float sm[8], slr[8], so[8][256];
    if (lane == 0) { sm[w] = m; slr[w] = l; }
#pragma unroll
    for (int d = 0; d < 8; ++d) so[w][lane * 8 + d] = acc[d];
    __syncthreads();
    if (w == 0) {
        float M = -FLT_MAX;
        for (int i = 0; i < 8; ++i) M = fmaxf(M, sm[i]);
        float L = 0.f, o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < 8; ++i) {
            if (slr[i] == 0.f) continue;
            const float c = __expf(sm[i] - M);
            L += slr[i] * c;
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

__global__ void qsa_attn_combine_k(const float * part_o, const float * part_ml, const float * qfull, float * out, int H,
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
    out[((int64_t) t * H + h) * D + i] = (o / L) * qsa_sigmoid(gate);
}

void qsa_attention(const float * q, const half * kc, const half * vc, const float * qfull_gate, const int32_t * sel,
                   const int32_t * n_sel, float * out, int T, const QsaShape & sh, const int * pos0, float scale,
                   float * scratch, cudaStream_t s) {
    if (sh.D != 256) { fprintf(stderr, "qsa: head dim %d unsupported\n", sh.D); abort(); }
    const int ns = kQsaSplits;
    float * part_o = scratch;
    float * part_ml = scratch + (size_t) T * sh.H * ns * sh.D;
    qsa_attn_partial_k<<<dim3(T * sh.H, ns), 256, 0, s>>>(q, kc, vc, sel, n_sel, sh.H, sh.Hkv, sh.D, sh.ratio,
                                                          sh.top_blocks, pos0, scale, part_o, part_ml, ns);
    qsa_attn_combine_k<<<T * sh.H, sh.D, 0, s>>>(part_o, part_ml, qfull_gate, out, sh.H, sh.D, ns);
}

// prompt rows: one warp per (t, h)
__global__ void qsa_attn_prefill_k(const float * q, const half * kc, const half * vc, const float * qfull,
                                   const int32_t * sel, const int32_t * n_sel, float * out, int t0, int t1, int H,
                                   int Hkv, int D, int r, int K, int pos0, float scale) {
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (w >= (t1 - t0) * H) return;
    const int ti = w / H, h = w % H, t = t0 + ti, hk = h / (H / Hkv);
    const int qpos = pos0 + t;
    const int ns = n_sel[ti];
    const int tail_start = (qpos + 1) / r * r;
    const int n_keys = ns == 0 ? qpos + 1 : ns * r + (qpos + 1 - tail_start);
    const int32_t * sl = sel + (int64_t) ti * K;
    float qv[8];
    const float * qp = q + ((int64_t) t * H + h) * D + lane * 8;
#pragma unroll
    for (int d = 0; d < 8; ++d) qv[d] = qp[d] * scale;
    float m = -FLT_MAX, l = 0.f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < n_keys; ++i) {
        const int key = ns == 0 ? i : qsa_cell(i, sl, ns, r, tail_start);
        const uint4 kraw = *(const uint4 *) (kc + ((int64_t) key * Hkv + hk) * D + lane * 8);
        const half2 * k2 = (const half2 *) &kraw;
        float sdot = 0.f;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const float2 f = __half22float2(k2[d]);
            sdot += qv[2 * d] * f.x + qv[2 * d + 1] * f.y;
        }
        sdot = qsa_warp_sum(sdot);
        const float mn = fmaxf(m, sdot);
        const float corr = __expf(m - mn), p = __expf(sdot - mn);
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
    for (int d = 0; d < 8; ++d) o[d] = acc[d] / l * qsa_sigmoid(gate[d]);
}

// Prompt rows on tensor cores: one block (4 warps) per (row, KV head) computes the G = H / Hkv query heads that
// share the KV head together (padded to 16), so each gathered key/value tile is loaded once for all of them.
// Per tile of 32 keys: S = Q K^T (wmma), online softmax, O = O * corr + P V (wmma, O kept in registers).
constexpr int QA_KT = 32;   // keys per tile
constexpr int QA_D = 256;   // head dim this kernel is built for
constexpr int QA_LD = QA_D + 8;
__global__ void __launch_bounds__(128) qsa_attn_prefill_tc_k(const float * q, const half * kc, const half * vc,
                                                              const float * qfull, const int32_t * sel,
                                                              const int32_t * n_sel, float * out, int t0, int H,
                                                              int Hkv, int r, int K, int pos0, float scale) {
    using namespace nvcuda;
    __shared__ __align__(32) half Qs[16 * QA_LD];
    __shared__ __align__(32) half KVs[QA_KT * QA_LD];
    __shared__ __align__(32) float Ss[16 * QA_KT];
    __shared__ __align__(32) half Ps[16 * (QA_KT + 8)];
    __shared__ __align__(32) float Tmp[16 * QA_D];
    __shared__ float corr_s[16], l_s[16], m_s[16];
    __shared__ int cells[QA_KT];
    const int ti = blockIdx.x, hk = blockIdx.y, t = t0 + ti;
    const int G = H / Hkv, tid = threadIdx.x, warp = tid >> 5;
    const int qpos = pos0 + t;
    const int ns = n_sel[ti];
    const int tail_start = (qpos + 1) / r * r;
    const int n_keys = ns == 0 ? qpos + 1 : ns * r + (qpos + 1 - tail_start);
    const int32_t * sl = sel + (int64_t) ti * K;
    // queries of the group (scaled), zero rows past G
    for (int i = tid; i < 16 * QA_D; i += 128) {
        const int g = i / QA_D, d = i % QA_D;
        Qs[g * QA_LD + d] = __float2half(g < G ? q[((int64_t) t * H + hk * G + g) * QA_D + d] * scale : 0.f);
    }
    if (tid < 16) {
        m_s[tid] = -FLT_MAX;
        l_s[tid] = 0.f;
    }
    // this thread's slice of O[16][256]: row orow, dims [ocol, ocol + 32)
    const int orow = tid / 8, ocol = (tid % 8) * 32;
    float o[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i] = 0.f;
    for (int k0 = 0; k0 < n_keys; k0 += QA_KT) {
        const int nk = min(QA_KT, n_keys - k0);
        if (tid < QA_KT) cells[tid] = tid < nk ? (ns == 0 ? k0 + tid : qsa_cell(k0 + tid, sl, ns, r, tail_start)) : -1;
        __syncthreads();
        // K tile: 32 keys x 256 dims, 16-byte loads
        for (int i = tid; i < QA_KT * QA_D / 8; i += 128) {
            const int k = i / (QA_D / 8), d8 = (i % (QA_D / 8)) * 8;
            const int c = cells[k];
            *(uint4 *) &KVs[k * QA_LD + d8] = c >= 0 ? *(const uint4 *) (kc + ((int64_t) c * Hkv + hk) * QA_D + d8)
                                                     : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        // S[16][32]: warps 0 and 1 each own 16 keys
        if (warp < 2) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf;
            wmma::fill_fragment(sf, 0.f);
#pragma unroll 4
            for (int kd = 0; kd < QA_D; kd += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
                wmma::load_matrix_sync(af, Qs + kd, QA_LD);
                wmma::load_matrix_sync(bf, KVs + warp * 16 * QA_LD + kd, QA_LD);
                wmma::mma_sync(sf, af, bf, sf);
            }
            wmma::store_matrix_sync(Ss + warp * 16, sf, QA_KT, wmma::mem_row_major);
        }
        __syncthreads();
        // V tile into the same buffer; meanwhile the softmax update (8 threads per row)
        for (int i = tid; i < QA_KT * QA_D / 8; i += 128) {
            const int k = i / (QA_D / 8), d8 = (i % (QA_D / 8)) * 8;
            const int c = cells[k];
            *(uint4 *) &KVs[k * QA_LD + d8] = c >= 0 ? *(const uint4 *) (vc + ((int64_t) c * Hkv + hk) * QA_D + d8)
                                                     : make_uint4(0, 0, 0, 0);
        }
        {
            const int row = tid / 8, sub = tid % 8;  // each of the 8 threads takes 4 keys of the row
            float mx = -FLT_MAX;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = sub * 4 + j;
                if (k < nk) mx = fmaxf(mx, Ss[row * QA_KT + k]);
            }
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            const float m_old = m_s[row], m_new = fmaxf(m_old, mx);
            float ps = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = sub * 4 + j;
                const float p = k < nk ? __expf(Ss[row * QA_KT + k] - m_new) : 0.f;
                Ps[row * (QA_KT + 8) + k] = __float2half(p);
                ps += p;
            }
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) ps += __shfl_xor_sync(0xffffffff, ps, off);
            __syncwarp();
            if (sub == 0) {
                const float c = __expf(m_old - m_new);
                corr_s[row] = c;
                l_s[row] = l_s[row] * c + ps;
                m_s[row] = m_new;
            }
        }
        __syncthreads();
        // P V: each warp 4 of the 16 dim tiles, into Tmp
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf[QA_KT / 16];
#pragma unroll
            for (int kk = 0; kk < QA_KT / 16; ++kk) wmma::load_matrix_sync(pf[kk], Ps + kk * 16, QA_KT + 8);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int dt = warp * 4 + j;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> of;
                wmma::fill_fragment(of, 0.f);
#pragma unroll
                for (int kk = 0; kk < QA_KT / 16; ++kk) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
                    wmma::load_matrix_sync(vf, KVs + kk * 16 * QA_LD + dt * 16, QA_LD);
                    wmma::mma_sync(of, pf[kk], vf, of);
                }
                wmma::store_matrix_sync(Tmp + dt * 16, of, QA_D, wmma::mem_row_major);
            }
        }
        __syncthreads();
        const float c = corr_s[orow];
#pragma unroll
        for (int i = 0; i < 32; ++i) o[i] = o[i] * c + Tmp[orow * QA_D + ocol + i];
        __syncthreads();
    }
    if (orow < G) {
        const int h = hk * G + orow;
        const float inv = 1.f / l_s[orow];
        const float * gate = qfull + ((int64_t) t * H + h) * 2 * QA_D + QA_D + ocol;
        float * op = out + ((int64_t) t * H + h) * QA_D + ocol;
#pragma unroll
        for (int i = 0; i < 32; ++i) op[i] = o[i] * inv * qsa_sigmoid(gate[i]);
    }
}

void qsa_attention_prefill(const float * q, const half * kc, const half * vc, const float * qfull_gate,
                           const int32_t * sel, const int32_t * n_sel, float * out, int t0, int t1,
                           const QsaShape & sh, int pos0, float scale, cudaStream_t s) {
    const int warps = (t1 - t0) * sh.H;
    if (warps <= 0) return;
    static const bool old = getenv("BNK_QSA_OLD") != nullptr;
    if (!old && sh.D == QA_D && sh.H / sh.Hkv <= 16) {
        qsa_attn_prefill_tc_k<<<dim3(t1 - t0, sh.Hkv), 128, 0, s>>>(q, kc, vc, qfull_gate, sel, n_sel, out, t0, sh.H,
                                                                   sh.Hkv, sh.ratio, sh.top_blocks, pos0, scale);
        return;
    }
    qsa_attn_prefill_k<<<(warps + 7) / 8, 256, 0, s>>>(q, kc, vc, qfull_gate, sel, n_sel, out, t0, t1, sh.H, sh.Hkv,
                                                       sh.D, sh.ratio, sh.top_blocks, pos0, scale);
}

}  // namespace bnk
