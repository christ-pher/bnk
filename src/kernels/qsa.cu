#include <cfloat>
#include <cstdio>
#include <cstdlib>

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

void qsa_select(const float * q, const float * pooled, float * scores, int max_blocks, int32_t * sel, int32_t * n_sel,
                int T, const QsaShape & sh, const int * pos0, int t_off, cudaStream_t s) {
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

void qsa_attention_prefill(const float * q, const half * kc, const half * vc, const float * qfull_gate,
                           const int32_t * sel, const int32_t * n_sel, float * out, int t0, int t1,
                           const QsaShape & sh, int pos0, float scale, cudaStream_t s) {
    const int warps = (t1 - t0) * sh.H;
    if (warps <= 0) return;
    qsa_attn_prefill_k<<<(warps + 7) / 8, 256, 0, s>>>(q, kc, vc, qfull_gate, sel, n_sel, out, t0, t1, sh.H, sh.Hkv,
                                                       sh.D, sh.ratio, sh.top_blocks, pos0, scale);
}

}  // namespace bnk
