// Inkling 2's MoE on the device for decode / verify windows: the routed pairs grouped by expert (a plan), gate | up
// with SITU and the routing weight folded in, down, and the per-token sum. Every expert is resident (one matrix
// per layer for gate | up, one for down); weights are read once per expert whatever its tokens. Formats: NVFP4
// (with the per-expert global scales) and BF16 (the last layer).
#include "inkling/ink_moe.h"

#include <cfloat>
#include <cuda_bf16.h>
#include <stdexcept>

#include "kernels/gemv.h"
#include "kernels/quant.cuh"

namespace bnk {

namespace {

__device__ __forceinline__ float wsum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// one block: counting sort of the T*k pairs by expert (ascending experts, pairs in order within one)
__global__ void k_plan(const int32_t * ids, const float * w, int n, int k, int n_exp, InkMoePlan * P) {
    __shared__ int cnt[kInkMaxExperts + 1];
    for (int e = threadIdx.x; e <= n_exp; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < n; p += blockDim.x) atomicAdd(&cnt[ids[p] + 1], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int nu = 0;
        for (int e = 0; e < n_exp; ++e) {
            if (cnt[e + 1]) {
                P->expert[nu] = e;
                P->start[nu] = cnt[e];
                ++nu;
            }
            cnt[e + 1] += cnt[e];
        }
        P->start[nu] = n;
        P->n_u = nu;
        P->k = k;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < n; p += blockDim.x) {
        const int e = ids[p];
        int rank = 0;
        for (int q = 0; q < p; ++q) rank += ids[q] == e;
        const int slot = cnt[e] + rank;
        P->tok[slot] = p / k;
        P->wt[slot] = w[p];
        P->pair_slot[p] = slot;
        P->slot_e[slot] = e;
    }
}

// y[slot][r] = scale[e] * W_e[r] . x[src(slot)] for the rows r < R of each routed expert e (rows_e rows per expert):
// a warp per row, its lanes over the row in 4-byte words (coalesced), x of up to NT slots in shared memory; src is
// the slot's token (x by token) or the slot itself (x by slot)
// routing (a warp per token: sigmoid, the top k of sigmoid + bias, weights / (sum + 1e-20) * w_scale) and the plan
// in one block
__global__ void k_route_plan(const float * logits, const float * bias, int T, int n_exp, int k, float w_scale,
                             InkMoePlan * P) {
    __shared__ int ids[kInkMaxPairs];
    __shared__ float ws[kInkMaxPairs];
    __shared__ int cnt[kInkMaxExperts + 1];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const int per = (n_exp + 31) / 32;
    for (int t = warp; t < T; t += nw) {
        float p[8], sel[8];
        for (int i = 0; i < per; ++i) {
            const int e = lane + 32 * i;
            if (e < n_exp) {
                p[i] = 1.f / (1.f + expf(-logits[(int64_t) t * n_exp + e]));
                sel[i] = p[i] + bias[e];
            } else {
                p[i] = 0; sel[i] = -FLT_MAX;
            }
        }
        float sum = 0;
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
            if (lane == 0) { ids[t * k + j] = be; ws[t * k + j] = pv; }
            sum += pv;
        }
        __syncwarp();
        if (lane < k) ws[t * k + lane] = ws[t * k + lane] / (sum + 1e-20f) * w_scale;
    }
    const int n = T * k;
    for (int e = threadIdx.x; e <= n_exp; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int q = threadIdx.x; q < n; q += blockDim.x) atomicAdd(&cnt[ids[q] + 1], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int nu = 0;
        for (int e = 0; e < n_exp; ++e) {
            if (cnt[e + 1]) {
                P->expert[nu] = e;
                P->start[nu] = cnt[e];
                ++nu;
            }
            cnt[e + 1] += cnt[e];
        }
        P->start[nu] = n;
        P->n_u = nu;
        P->k = k;
    }
    __syncthreads();
    for (int q = threadIdx.x; q < n; q += blockDim.x) {
        const int e = ids[q];
        int rank = 0;
        for (int r = 0; r < q; ++r) rank += ids[r] == e;
        const int slot = cnt[e] + rank;
        P->tok[slot] = q / k;
        P->wt[slot] = ws[q];
        P->pair_slot[q] = slot;
        P->slot_e[slot] = e;
    }
}

template <int FMT, int NT, int RW>
__global__ void __launch_bounds__(256) k_egemv(const InkMoePlan * P, const float * x, bool by_slot, int cols,
        const uint8_t * W, size_t row_bytes, int rows_e, const float * scale, float * y) {
    extern __shared__ float xs[];   // [NT][cols]
    __shared__ float lut[16];
    const int u = blockIdx.y;
    if (u >= P->n_u) return;
    if (threadIdx.x < 16) lut[threadIdx.x] = (float) kvalues_fp4[threadIdx.x];
    const int lane = threadIdx.x & 31, r0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * RW;
    const int e = P->expert[u], s0 = P->start[u], n = P->start[u + 1] - s0;
    const float sc = scale ? scale[e] : 1.f;
    if (FMT == QT_NVFP4 && row_bytes / 4 > 32 * 11) return;   // wider rows than the preload holds (not Inkling's)
    for (int g = 0; g < n; g += NT) {
        const int m = min(NT, n - g);
        __syncthreads();
        for (int i = 0; i < m; ++i) {
            const int src = by_slot ? s0 + g + i : P->tok[s0 + g + i];
            const float4 * xr = (const float4 *) (x + (size_t) src * cols);
            for (int c = threadIdx.x; c < cols / 4; c += blockDim.x) ((float4 *) (xs + i * cols))[c] = xr[c];
        }
        __syncthreads();
        for (int rr = 0; rr < RW; ++rr) {
            const int r = r0 + rr;
            if (r >= rows_e) break;
            const uint8_t * row = W + ((size_t) e * rows_e + r) * row_bytes;
            float acc[NT];
#pragma unroll
            for (int i = 0; i < NT; ++i) acc[i] = 0.f;
            if (FMT == QT_NVFP4) {
                // every word of the lane's share of the row in flight at once (rows of up to 32 * kMaxW words)
                constexpr int kMaxW = 11;
                const uint32_t * row32 = (const uint32_t *) row;
                const int words = (int) (row_bytes / 4);
                uint32_t q[kMaxW];
#pragma unroll
                for (int t = 0; t < kMaxW; ++t) {
                    const int w = lane + 32 * t;
                    q[t] = w < words ? __ldg(row32 + w) : 0u;
                }
#pragma unroll
                for (int t = 0; t < kMaxW; ++t) {
                    const int w = lane + 32 * t;
                    if (w >= words) break;
                    const int b = w / 9, j = w - b * 9;
                    if (j == 0) continue;   // the block's four scales
                    const int sb = (j - 1) >> 1, half = (j - 1) & 1;
                    const float d = ue4m3h(row[b * 36 + sb]);
                    const int base = b * 64 + sb * 16 + half * 4;
                    float wl[4], wh[4];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        wl[k] = d * lut[(q[t] >> (8 * k)) & 0xF];
                        wh[k] = d * lut[(q[t] >> (8 * k + 4)) & 0xF];
                    }
#pragma unroll
                    for (int i = 0; i < NT; ++i) {
                        if (i >= m) break;
                        const float4 xl = *(const float4 *) (xs + i * cols + base);
                        const float4 xh = *(const float4 *) (xs + i * cols + base + 8);
                        acc[i] += wl[0] * xl.x + wl[1] * xl.y + wl[2] * xl.z + wl[3] * xl.w +
                                  wh[0] * xh.x + wh[1] * xh.y + wh[2] * xh.z + wh[3] * xh.w;
                    }
                }
            } else {   // BF16: lanes over 8-value chunks
                const uint4 * r16 = (const uint4 *) row;
                const int chunks = cols / 8;
                for (int c = lane; c < chunks; c += 32) {
                    const uint4 q = r16[c];
                    const uint32_t w4[4] = {q.x, q.y, q.z, q.w};
                    float wv[8];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        wv[2 * k] = __uint_as_float(w4[k] << 16);
                        wv[2 * k + 1] = __uint_as_float(w4[k] & 0xffff0000u);
                    }
#pragma unroll
                    for (int i = 0; i < NT; ++i) {
                        if (i >= m) break;
                        const float4 a0 = *(const float4 *) (xs + i * cols + c * 8);
                        const float4 a1 = *(const float4 *) (xs + i * cols + c * 8 + 4);
                        acc[i] += wv[0] * a0.x + wv[1] * a0.y + wv[2] * a0.z + wv[3] * a0.w +
                                  wv[4] * a1.x + wv[5] * a1.y + wv[6] * a1.z + wv[7] * a1.w;
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                if (i >= m) break;
                const float v = wsum(acc[i]);
                if (lane == 0) y[(size_t) (s0 + g + i) * rows_e + r] = v * sc;
            }
        }
    }
}

// The NVFP4 product on int8 activations (dp4a): y[slot][r] = scale[e] * sum over 32-sub-blocks of
// (d0 * dp(w[0:16], q[0:16]) + d1 * dp(w[16:32], q[16:32])) * qd; a warp per row, its lanes over sub-blocks, the
// activations of up to NT slots in shared memory.
template <int NT, int RW>
__global__ void __launch_bounds__(256) k_egemv_q(const InkMoePlan * P, const int8_t * xq, const float * xd,
        int64_t cols_pad, bool by_slot, int cols, const uint8_t * W, size_t row_bytes, int rows_e, const float * scale,
        float * y) {
    extern __shared__ int32_t sq[];   // [NT][cols / 4] codes, then [NT][cols / 32] scales
    float * sd = (float *) (sq + NT * cols / 4);
    const int u = blockIdx.y;
    if (u >= P->n_u) return;
    const int lane = threadIdx.x & 31, r0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * RW;
    const int e = P->expert[u], s0 = P->start[u], n = P->start[u + 1] - s0;
    const float sc = scale ? scale[e] : 1.f;
    const int nsb = cols / 32;
    for (int g = 0; g < n; g += NT) {
        const int m = min(NT, n - g);
        __syncthreads();
        for (int i = 0; i < m; ++i) {
            const int src = by_slot ? s0 + g + i : P->tok[s0 + g + i];
            const int32_t * qr = (const int32_t *) (xq + (size_t) src * cols_pad);
            for (int c = threadIdx.x; c < cols / 4; c += blockDim.x) sq[i * (cols / 4) + c] = qr[c];
            for (int c = threadIdx.x; c < nsb; c += blockDim.x) sd[i * nsb + c] = xd[(size_t) src * (cols_pad / 32) + c];
        }
        __syncthreads();
        for (int rr = 0; rr < RW; ++rr) {
            const int r = r0 + rr;
            if (r >= rows_e) break;
            const uint8_t * row = W + ((size_t) e * rows_e + r) * row_bytes;
            float acc[NT];
#pragma unroll
            for (int i = 0; i < NT; ++i) acc[i] = 0.f;
            for (int sb = lane; sb < nsb; sb += 32) {
                Unpacked uw;
                unpack_sub<QT_NVFP4>(row, sb, uw);
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    if (i >= m) break;
                    const int4 * xa = (const int4 *) (sq + i * (cols / 4) + sb * 8);
                    const int4 a = xa[0], b = xa[1];
                    const int s0v = __dp4a(uw.w[0], a.x, __dp4a(uw.w[1], a.y, __dp4a(uw.w[2], a.z, __dp4a(uw.w[3], a.w, 0))));
                    const int s1v = __dp4a(uw.w[4], b.x, __dp4a(uw.w[5], b.y, __dp4a(uw.w[6], b.z, __dp4a(uw.w[7], b.w, 0))));
                    acc[i] += (uw.d0 * (float) s0v + uw.d1 * (float) s1v) * sd[i * nsb + sb];
                }
            }
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                if (i >= m) break;
                const float v = wsum(acc[i]);
                if (lane == 0) y[(size_t) (s0 + g + i) * rows_e + r] = v * sc;
            }
        }
    }
}

template <int NT>
void egemv_q_nt(const InkMoePlan * P, const int8_t * xq, const float * xd, int64_t cols_pad, bool by_slot, int cols,
                const uint8_t * W, size_t rb, int rows_e, const float * scale, float * y, int n_pairs, cudaStream_t s) {
    constexpr int RW = 4;
    const dim3 g((rows_e + 8 * RW - 1) / (8 * RW), n_pairs);
    const size_t sh = (size_t) NT * cols + (size_t) NT * (cols / 32) * 4;
    k_egemv_q<NT, RW><<<g, 256, sh, s>>>(P, xq, xd, cols_pad, by_slot, cols, W, rb, rows_e, scale, y);
}
void egemv_q(const InkMoePlan * P, const int8_t * xq, const float * xd, int64_t cols_pad, bool by_slot, int cols,
             const uint8_t * W, size_t rb, int rows_e, const float * scale, float * y, int n_pairs, int T, cudaStream_t s) {
    if (T == 1) egemv_q_nt<1>(P, xq, xd, cols_pad, by_slot, cols, W, rb, rows_e, scale, y, n_pairs, s);
    else if (T == 2) egemv_q_nt<2>(P, xq, xd, cols_pad, by_slot, cols, W, rb, rows_e, scale, y, n_pairs, s);
    else egemv_q_nt<4>(P, xq, xd, cols_pad, by_slot, cols, W, rb, rows_e, scale, y, n_pairs, s);
}

// h[slot][f] = situ(gu[slot][f], gu[slot][F + f]) * weight[slot]
__global__ void k_situ_slots(const InkMoePlan * P, const float * gu, int F, float beta, float lb, float * h, int n) {
    const int sl = blockIdx.y;
    if (sl >= n) return;
    for (int f = blockIdx.x * blockDim.x + threadIdx.x; f < F; f += gridDim.x * blockDim.x) {
        const float g = gu[(size_t) sl * 2 * F + f], u = gu[(size_t) sl * 2 * F + F + f];
        const float a = beta * tanhf(g / beta) * (1.f / (1.f + expf(-g)));
        const float b = lb > 0 ? lb * tanhf(u / lb) : u;
        h[(size_t) sl * F + f] = a * b * P->wt[sl];
    }
}

template <int FMT, int NT>
void egemv_nt(const InkMoePlan * P, const float * x, bool by_slot, int cols, const uint8_t * W, size_t rb, int rows_e,
              const float * scale, float * y, int n_pairs, cudaStream_t s) {
    constexpr int RW = 4;
    const dim3 g((rows_e + 8 * RW - 1) / (8 * RW), n_pairs);
    k_egemv<FMT, NT, RW><<<g, 256, (size_t) NT * cols * 4, s>>>(P, x, by_slot, cols, W, rb, rows_e, scale, y);
}

// tokens an expert can hold in this window: at most T (one token routes to an expert once)
template <int FMT>
void egemv(const InkMoePlan * P, const float * x, bool by_slot, int cols, const uint8_t * W, size_t rb, int rows_e,
           const float * scale, float * y, int n_pairs, int T, cudaStream_t s) {
    if (T == 1) egemv_nt<FMT, 1>(P, x, by_slot, cols, W, rb, rows_e, scale, y, n_pairs, s);
    else if (T == 2) egemv_nt<FMT, 2>(P, x, by_slot, cols, W, rb, rows_e, scale, y, n_pairs, s);
    else egemv_nt<FMT, 4>(P, x, by_slot, cols, W, rb, rows_e, scale, y, n_pairs, s);
}

void egemv_any(int type, const InkMoePlan * P, const float * x, bool by_slot, int cols, const uint8_t * W, size_t rb,
               int rows_e, const float * scale, float * y, int n_pairs, int T, cudaStream_t s) {
    if (type == QT_NVFP4) egemv<QT_NVFP4>(P, x, by_slot, cols, W, rb, rows_e, scale, y, n_pairs, T, s);
    else if (type == QT_BF16) egemv<QT_BF16>(P, x, by_slot, cols, W, rb, rows_e, scale, y, n_pairs, T, s);
    else throw std::runtime_error("ink_moe_window: expert format");
}

__global__ void k_sum(const InkMoePlan * P, const float * src, int T, int k, int E, float * out) {
    const int t = blockIdx.y;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < E; e += gridDim.x * blockDim.x) {
        float acc = 0;
        for (int j = 0; j < k; ++j) acc += src[(size_t) P->pair_slot[t * k + j] * E + e];
        out[(size_t) t * E + e] = acc;
    }
}

}  // namespace

void ink_moe_window(const InkMoeArgs & a, cudaStream_t s) {
    const int n = a.T * a.k;
    if (a.T < 1 || a.T > kInkWinMax || n > kInkMaxPairs || a.n_exp > kInkMaxExperts)
        throw std::runtime_error("ink_moe_window: window out of range");
    if (a.logits) k_route_plan<<<1, 256, 0, s>>>(a.logits, a.bias, a.T, a.n_exp, a.k, a.w_scale, a.plan);
    else k_plan<<<1, 128, 0, s>>>(a.ids, a.w, n, a.k, a.n_exp, a.plan);
    static const bool fq = getenv("BNK_INK_MOE_F32") != nullptr;   // tests: fp32 activations (shared-memory bound)
    const bool q = a.q && !fq;
    ActQ8 aq;
    aq.q = a.q; aq.d = a.qd;
    if (q && a.gu_type == QT_NVFP4) {
        quantize_act(a.x, a.E, a.T, a.E, aq, s);
        egemv_q(a.plan, aq.q, aq.d, aq.cols_pad, false, a.E, a.gate_up, a.gu_row_bytes, 2 * a.F, a.gu_scale, a.gu, n, a.T, s);
    } else {
        egemv_any(a.gu_type, a.plan, a.x, false, a.E, a.gate_up, a.gu_row_bytes, 2 * a.F, a.gu_scale, a.gu, n, a.T, s);
    }
    k_situ_slots<<<dim3((a.F + 255) / 256, n), 256, 0, s>>>(a.plan, a.gu, a.F, a.beta, a.lb, a.h, n);
    if (q && a.down_type == QT_NVFP4) {
        quantize_act(a.h, a.F, n, a.F, aq, s);
        egemv_q(a.plan, aq.q, aq.d, aq.cols_pad, true, a.F, a.down, a.down_row_bytes, a.E, a.down_scale, a.y, n, a.T, s);
    } else {
        egemv_any(a.down_type, a.plan, a.h, true, a.F, a.down, a.down_row_bytes, a.E, a.down_scale, a.y, n, a.T, s);
    }
    k_sum<<<dim3((a.E + 255) / 256, a.T), 256, 0, s>>>(a.plan, a.y, a.T, a.k, a.E, a.out);
    check_launch("ink_moe_window");
}

}  // namespace bnk
