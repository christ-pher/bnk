#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "kernels/moe.h"
#include "kernels/route.cuh"
#include "kernels/qdot.cuh"
#include "kernels/rfmt.cuh"
#include "kernels/rfmt_impl.h"

namespace bnk {

__global__ void bump_seq_k(uint32_t * seq) { *seq += 1; }

__device__ __forceinline__ uint64_t gtimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}
// debug timestamps per layer: [0] plan signalled, [1] wait start, [2] wait end
__device__ uint64_t g_moe_ts[64][3];
__device__ int g_moe_layer;
void bump_seq(uint32_t * seq, cudaStream_t st) { bump_seq_k<<<1, 1, 0, st>>>(seq); }

// ------------------------------------------------------------------------------------------- plan
// One warp. Lane j < T*k takes routed pair j: hits are deduplicated by expert with ballots.
// One thread per (token, slot) pair, everything in shared memory (n = T * k <= kMaxRouted = 128). The outputs are
// in pair order, as before: a hit entry is created by the first pair of its expert, misses keep their pair order
// (the CPU sums a token's experts in that order, so results do not depend on this kernel's scheduling).
__device__ __forceinline__ void moe_plan_body(const int32_t * ids, const float * w, int T, int k,
                                                         const int32_t * slot_of, HitList * hits, MoeMsg * msg,
                                                         const float * x, int E, const uint32_t * seq,
                                                         uint32_t * counts) {
    __shared__ int s_e[kMaxRouted], s_sl[kMaxRouted];
    __shared__ int s_first[kMaxRouted], s_miss[kMaxRouted];
    __shared__ unsigned s_mask[kMaxRouted];
    __shared__ int s_nh, s_nm;
    const int n = T * k, j = threadIdx.x;
    const bool in = j < kMaxRouted;   // threads past kMaxRouted (a bigger block) only join the barriers
    const bool valid = j < n;
    const int e = valid ? ids[j] : -1;
    const int t = valid ? j / k : 0;
    const float wt = valid ? w[j] : 0.f;
    const int sl = valid ? slot_of[e] : -1;
    if (valid && counts) atomicAdd(&counts[e], 1u);
    if (in) {
        s_e[j] = e;
        s_sl[j] = sl;
        s_mask[j] = 0;
    }
    __syncthreads();
    // is this pair its expert's first hit?
    bool first = valid && sl >= 0;
    for (int q = 0; first && q < j; ++q) first = s_e[q] != e;
    if (in) {
        s_first[j] = first;
        s_miss[j] = valid && sl < 0;
    }
    __syncthreads();
    // exclusive prefix sums (n <= 128: one warp walks it)
    if (j < 32) {
        int fh = 0, fm = 0;
        for (int b = 0; b < kMaxRouted; b += 32) {
            const int q = b + j;
            const int vf = s_first[q], vm = s_miss[q];
            const unsigned bf = __ballot_sync(0xffffffff, vf), bm = __ballot_sync(0xffffffff, vm);
            const unsigned lt = (1u << j) - 1;
            s_first[q] = fh + __popc(bf & lt);   // the entry a first pair creates
            s_miss[q] = fm + __popc(bm & lt);
            fh += __popc(bf);
            fm += __popc(bm);
        }
        if (j == 0) {
            s_nh = fh;
            s_nm = fm;
        }
    }
    __syncthreads();
    // every hit pair finds its expert's entry (the first pair's), and sets its token's bit and weight
    if (valid && sl >= 0) {
        int q = 0;
        while (s_e[q] != e || s_sl[q] < 0) ++q;   // the first pair of this expert
        const int ent = s_first[q];
        hits->pair_ent[j] = ent;
        atomicOr(&s_mask[ent], 1u << t);
        hits->e[ent].w[t] = wt;
        if (q == j) {
            hits->e[ent].expert = e;
            hits->e[ent].slot = sl;
        }
    }
    if (valid && sl < 0) {
        hits->pair_ent[j] = -1;
        const int pos = s_miss[j];
        msg->miss_t[pos] = t;
        msg->miss_e[pos] = e;
        msg->miss_w[pos] = wt;
    }
    __syncthreads();
    if (j < s_nh) hits->e[j].mask = s_mask[j];
    if (j == 0) {
        hits->n = s_nh;
        hits->n_miss = s_nm;
        msg->n_miss = s_nm;
        msg->T = T;
    }
    if (s_nm > 0) {
        float4 * mx = (float4 *) msg->x();
        const float4 * xs = (const float4 *) x;
        for (int i = j; i < T * E / 4; i += blockDim.x) mx[i] = xs[i];
        __threadfence_system();
    }
    __syncthreads();
    if (j == 0) {
        msg->seq_req = *seq;
        g_moe_ts[g_moe_layer & 63][0] = gtimer();
    }
}

__global__ void __launch_bounds__(kMaxRouted) moe_plan_k(const int32_t * ids, const float * w, int T, int k,
                                                         const int32_t * slot_of, HitList * hits, MoeMsg * msg,
                                                         const float * x, int E, const uint32_t * seq,
                                                         uint32_t * counts) {
    moe_plan_body(ids, w, T, k, slot_of, hits, msg, x, E, seq, counts);
}

// route_topk and moe_plan in one launch: warp t picks token t's experts (as route_topk_k), then the block plans
template <int PER>
__global__ void __launch_bounds__(256) moe_route_plan_k(const float * logits, int n_exp, float w_scale, int32_t * ids,
                                                        float * w, int T, int k, const int32_t * slot_of,
                                                        HitList * hits, MoeMsg * msg, const float * x, int E,
                                                        const uint32_t * seq, uint32_t * counts) {
    const int t = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (t < T) route_topk_warp<PER>(logits, n_exp, k, ids, w, w_scale, t, lane);
    __syncthreads();   // the ids / weights (global, this block's writes) for the plan
    moe_plan_body(ids, w, T, k, slot_of, hits, msg, x, E, seq, counts);
}

void moe_route_plan(const float * logits, int n_exp, float w_scale, int32_t * ids, float * w, int T, int k,
                    const MoeLayerDesc & d, MoeScratch & s, MoeMsg * msg, const float * x, int E, const uint32_t * seq,
                    uint32_t * counts_layer, cudaStream_t st) {
    if (T * k > kMaxRouted || T > 8 || k > 32 || n_exp > 1024) {
        fprintf(stderr, "moe_route_plan: unsupported shape\n");
        abort();
    }
    if (n_exp <= 512)
        moe_route_plan_k<16><<<1, 256, 0, st>>>(logits, n_exp, w_scale, ids, w, T, k, d.slot_of, s.hits, msg, x, E, seq,
                                               counts_layer);
    else
        moe_route_plan_k<32><<<1, 256, 0, st>>>(logits, n_exp, w_scale, ids, w, T, k, d.slot_of, s.hits, msg, x, E, seq,
                                               counts_layer);
}

// DeepSeek's routing (route.cuh: ds_route_warp) and the plan in one launch: warp t picks token t's experts
__global__ void __launch_bounds__(256) moe_route_plan_ds_k(const float * logits, const float * bias, int n_exp, int k,
                                                           float scale, int norm, int32_t * ids, float * w, int T,
                                                           const int32_t * slot_of, HitList * hits, MoeMsg * msg,
                                                           const float * x, int E, const uint32_t * seq, uint32_t * counts) {
    const int t = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (t < T) ds_route_warp(logits, bias, n_exp, k, scale, norm, ids, w, t, lane);
    __syncthreads();   // the ids / weights (global, this block's writes) for the plan
    moe_plan_body(ids, w, T, k, slot_of, hits, msg, x, E, seq, counts);
}

void moe_route_plan_ds(const float * logits, const float * bias, int n_exp, int k, float scale, bool norm, int32_t * ids,
                       float * w, int T, const MoeLayerDesc & d, MoeScratch & s, MoeMsg * msg, const float * x, int E,
                       const uint32_t * seq, uint32_t * counts_layer, cudaStream_t st) {
    if (T * k > kMaxRouted || T > 8 || k > 32 || n_exp > 512) {
        fprintf(stderr, "moe_route_plan_ds: unsupported shape\n");
        abort();
    }
    moe_route_plan_ds_k<<<1, 256, 0, st>>>(logits, bias, n_exp, k, scale, norm ? 1 : 0, ids, w, T, d.slot_of, s.hits, msg, x, E,
                                           seq, counts_layer);
}

void moe_plan(const int32_t * ids, const float * w, int T, int k, const MoeLayerDesc & d, MoeScratch & s, MoeMsg * msg,
              const float * x, int E, const uint32_t * seq, uint32_t * counts_layer, cudaStream_t st) {
    if (T * k > kMaxRouted) { fprintf(stderr, "moe_plan: %d pairs exceed %d\n", T * k, kMaxRouted); abort(); }
    moe_plan_k<<<1, kMaxRouted, 0, st>>>(ids, w, T, k, d.slot_of, s.hits, msg, x, E, seq, counts_layer);
}

// ------------------------------------------------------------------------------------------- hits
// grid (kMaxRouted, ceil(2F/8)); warp per gate/up row of one hit expert.
template <int FMT, int NT>
__global__ void __launch_bounds__(256) moe_gu_k(const HitList * __restrict__ hits, const uint8_t * __restrict__ base,
                                                size_t blob, size_t gate_bytes, size_t grow, int F, int E,
                                                const int8_t * __restrict__ aq, const float * __restrict__ ad,
                                                int64_t cols_pad, float * __restrict__ gu, int slot_lo, int slot_hi) {
    const int ent = blockIdx.x;
    if (ent >= hits->n) return;
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (r >= 2 * F) return;
    const HitEntry & h = hits->e[ent];
    if (h.slot < slot_lo || h.slot >= slot_hi) return;   // another region's expert
    const uint32_t mask = h.mask;
    const uint8_t * row = base + (size_t) (h.slot - slot_lo) * blob + (r < F ? (size_t) r * grow : gate_bytes + (size_t) (r - F) * grow);
    const int nsb = E / 32;
    const int64_t nb = cols_pad / 32;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int sb = lane; sb < nsb; sb += 32) {
        Unpacked u;
        unpack_sub<FMT>(row, sb, u);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (mask & (1u << t)) {
                const int4 * ap = (const int4 *) (aq + t * cols_pad + sb * 32);
                acc[t] += qdot_sub<FMT>(u, __ldg(ap), __ldg(ap + 1), __ldg(ad + t * nb + sb));
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        if (mask & (1u << t)) {
            const float v = warp_reduce_sum(acc[t]);
            if (lane == 0) gu[((size_t) ent * kMaxWindow + t) * 2 * F + r] = v;
        }
    }
}

__device__ float g_moe_limit = 0.f;   // SwiGLU clamp (DeepSeek-V4), 0 = none
void moe_set_swiglu_limit(float limit) { cudaMemcpyToSymbol(g_moe_limit, &limit, sizeof(float)); }

// grid (kMaxRouted, W); block F threads (F % 32 == 0): h = silu(g)*u quantized per 32
__global__ void moe_act_k(const HitList * __restrict__ hits, const float * __restrict__ gu, int F,
                          int8_t * __restrict__ hq, float * __restrict__ hd, ExpScales es, int cold_lo) {
    const int ent = blockIdx.x, t = blockIdx.y;
    if (ent >= hits->n || !(hits->e[ent].mask & (1u << t))) return;
    const size_t base = (size_t) ent * kMaxWindow + t;
    const float * g = gu + base * 2 * F;
    const int ex = hits->e[ent].expert;
    const bool cold = hits->e[ent].slot >= cold_lo;   // a cold band's blocks carry the global scale already
    const float sg = es.gate && !cold ? es.gate[ex] : 1.f, sud = es.up && !cold ? es.up[ex] * es.down[ex] : 1.f;
    const float lim = g_moe_limit;
    for (int r = threadIdx.x; r < F; r += blockDim.x) {
        float gv = g[r] * sg, uv = g[F + r];
        if (lim > 0.f) { gv = fminf(gv, lim); uv = fminf(fmaxf(uv, -lim), lim); }
        const float h = gv / (1.f + __expf(-gv)) * uv * sud;
        float d;
        const int8_t q = quant_lane(h, d);
        hq[base * F + r] = q;
        if ((threadIdx.x & 31) == 0) hd[base * (F / 32) + r / 32] = d;
    }
}

// grid (kMaxRouted, ceil(E/8)); warp per down row
template <int FMT, int NT>
__global__ void __launch_bounds__(256) moe_down_k(const HitList * __restrict__ hits, const uint8_t * __restrict__ base,
                                                  size_t blob, size_t down_off, size_t drow, int F, int E,
                                                  const int8_t * __restrict__ hq, const float * __restrict__ hd,
                                                  float * __restrict__ part, int slot_lo, int slot_hi) {
    const int ent = blockIdx.x;
    if (ent >= hits->n) return;
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (r >= E) return;
    const HitEntry & h = hits->e[ent];
    const uint32_t mask = h.mask;
    if (h.slot < slot_lo || h.slot >= slot_hi) return;
    const uint8_t * row = base + (size_t) (h.slot - slot_lo) * blob + down_off + (size_t) r * drow;
    const int nsb = F / 32;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int sb = lane; sb < nsb; sb += 32) {
        Unpacked u;
        unpack_sub<FMT>(row, sb, u);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (mask & (1u << t)) {
                const size_t b = (size_t) ent * kMaxWindow + t;
                const int4 * ap = (const int4 *) (hq + b * F + sb * 32);
                acc[t] += qdot_sub<FMT>(u, ap[0], ap[1], hd[b * (F / 32) + sb]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        if (mask & (1u << t)) {
            const float v = warp_reduce_sum(acc[t]);
            if (lane == 0) part[((size_t) ent * kMaxWindow + t) * E + r] = v;
        }
    }
}

// ---- R-layout hit kernels: LPR lanes per row, the entry's token activations staged in shared memory
// One row's dot products with an expert's N tokens: LPR lanes per row stride over its sub-blocks, then reduce. The
// tokens are packed 4 bits each in `tp` (token i: activations at aq + tok_i * sq, scales at ad + tok_i * sd). N is the
// expert's own token count (a block-uniform switch picks it): a token the expert was not routed is never computed
// (a mask test inside the loop was predicated, so all NT dot products issued anyway).
template <int FMT, int LPR, int N>
__device__ __forceinline__ void row_dots(const uint8_t * row, const ROff & o, int nb, int sub, uint32_t tp,
                                         const int8_t * aq, int64_t sq, const float * ad, int64_t sd, float (&acc)[4]) {
#pragma unroll
    for (int i = 0; i < N; ++i) acc[i] = 0.f;
    for (int sb = sub; sb < nb; sb += LPR) {
        Unpacked u;
        unpack_sm<FMT>(row, o, sb, u);
#pragma unroll
        for (int i = 0; i < N; ++i) {
            const int t = (tp >> (4 * i)) & 15;
            const int4 * ap = (const int4 *) (aq + t * sq + sb * 32);
            acc[i] += qdot_sub<FMT, RT<FMT>::HAS_MIN>(u, ap[0], ap[1], ad[t * sd + sb]);
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int off = LPR / 2; off > 0; off >>= 1) acc[i] += __shfl_xor_sync(0xffffffff, acc[i], off);
    }
}

template <int FMT, int NT, int LPR>
__device__ __forceinline__ void row_dots_n(int n, const uint8_t * row, const ROff & o, int nb, int sub, uint32_t tp,
                                           const int8_t * aq, int64_t sq, const float * ad, int64_t sd, float (&acc)[4]) {
    switch (n) {   // the same for the whole block
        case 1: row_dots<FMT, LPR, 1>(row, o, nb, sub, tp, aq, sq, ad, sd, acc); break;
        case 2: if constexpr (NT >= 2) row_dots<FMT, LPR, 2>(row, o, nb, sub, tp, aq, sq, ad, sd, acc); break;
        case 3: if constexpr (NT >= 4) row_dots<FMT, LPR, 3>(row, o, nb, sub, tp, aq, sq, ad, sd, acc); break;
        default: if constexpr (NT >= 4) row_dots<FMT, LPR, 4>(row, o, nb, sub, tp, aq, sq, ad, sd, acc);
    }
}

// the expert's tokens packed 4 bits each (in order), their count; past the count, the last token repeats
__device__ __forceinline__ uint32_t pack_tokens(uint32_t mask, int & n) {
    uint32_t tp = 0;
    n = 0;
    int last = 0;
#pragma unroll
    for (int t = 0; t < kMaxWindow; ++t)
        if (mask & (1u << t)) {
            tp |= (uint32_t) t << (4 * n);
            ++n;
            last = t;
        }
    for (int i = n; i < 8; ++i) tp |= (uint32_t) last << (4 * i);
    return tp;
}

// gate and up rows of every hit expert (gu_out[ent][t][2F]); activations read through L1
template <int FMT, int NT, int LPR>
__global__ void __launch_bounds__(256) moe_gu_r_k(const HitList * __restrict__ hits, const uint8_t * __restrict__ base,
                                                  size_t blob, size_t up_off, size_t grow, ROff o, int F, int E,
                                                  const int8_t * __restrict__ aq, const float * __restrict__ ad,
                                                  int64_t cols_pad, float * __restrict__ gu_out, int slot_lo, int slot_hi) {
    const int ent = blockIdx.x;
    if (ent >= hits->n) return;
    if (hits->e[ent].slot < slot_lo || hits->e[ent].slot >= slot_hi) return;   // another region's expert
    int n;
    const uint32_t tp = pack_tokens(hits->e[ent].mask, n);
    rt_tables_load<FMT>();
    __syncthreads();
    constexpr int RPW = 32 / LPR;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane % LPR;
    const int r0 = blockIdx.y * (8 * RPW) + warp * RPW + lane / LPR, r = min(r0, 2 * F - 1);   // all lanes shuffle
    const uint8_t * row = base + (size_t) (hits->e[ent].slot - slot_lo) * blob +
                          (r < F ? (size_t) r * grow : up_off + (size_t) (r - F) * grow);
    // up to 4 tokens per pass (an expert with 5-8 tokens of a window, rare, reads its rows twice): the 8-token path
    // needed registers that cost every block occupancy
    for (int p0 = 0; p0 < n; p0 += 4) {
        const uint32_t tq = tp >> (4 * p0);
        float acc[4];
        row_dots_n<FMT, NT, LPR>(min(4, n - p0), row, o, E / 32, sub, tq, aq, cols_pad, ad, cols_pad / 32, acc);
        if (sub == 0 && r0 < 2 * F) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (p0 + i < n) gu_out[((size_t) ent * kMaxWindow + ((tq >> (4 * i)) & 15)) * 2 * F + r] = acc[i];
        }
    }
}

// down rows of every hit expert over its tokens' activated hidden rows (part[ent][t][E])
template <int FMT, int NT, int LPR>
__global__ void __launch_bounds__(256) moe_down_r_k(const HitList * __restrict__ hits, const uint8_t * __restrict__ base,
                                                    size_t blob, size_t down_off, size_t drow, ROff o, int F, int E,
                                                    const int8_t * __restrict__ hq, const float * __restrict__ hd,
                                                    float * __restrict__ part, int slot_lo, int slot_hi) {
    const int ent = blockIdx.x;
    if (ent >= hits->n) return;
    int n;
    const uint32_t tp = pack_tokens(hits->e[ent].mask, n);
    const int nb = F / 32;
    rt_tables_load<FMT>();
    __syncthreads();
    constexpr int RPW = 32 / LPR;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane % LPR;
    const int r0 = blockIdx.y * (8 * RPW) + warp * RPW + lane / LPR, r = min(r0, E - 1);
    if (hits->e[ent].slot < slot_lo || hits->e[ent].slot >= slot_hi) return;
    const uint8_t * row = base + (size_t) (hits->e[ent].slot - slot_lo) * blob + down_off + (size_t) r * drow;
    const size_t b0 = (size_t) ent * kMaxWindow;
    for (int p0 = 0; p0 < n; p0 += 4) {   // up to 4 tokens per pass, as above
        const uint32_t tq = tp >> (4 * p0);
        float acc[4];
        row_dots_n<FMT, NT, LPR>(min(4, n - p0), row, o, nb, sub, tq, hq + b0 * F, F, hd + b0 * nb, nb, acc);
        if (sub == 0 && r0 < E) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (p0 + i < n) part[(b0 + ((tq >> (4 * i)) & 15)) * E + r] = acc[i];
        }
    }
}

// ---- several regions per launch: the body of the R-layout kernels, per format, picked per block
template <int FMT, int NT, int LPR>
__device__ __forceinline__ void gu_r_body(const HitList * __restrict__ hits, int ent, const MoeRegion & rg, int F, int E,
                                          const int8_t * __restrict__ aq, const float * __restrict__ ad, int64_t cols_pad,
                                          float * __restrict__ gu_out) {
    int n;
    const uint32_t tp = pack_tokens(hits->e[ent].mask, n);
    rt_tables_load<FMT>();
    __syncthreads();
    constexpr int RPW = 32 / LPR;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane % LPR;
    const int r0 = blockIdx.y * (8 * RPW) + warp * RPW + lane / LPR, r = min(r0, 2 * F - 1);
    const ROff o{rg.go[0], rg.go[1], rg.go[2], rg.go[3], rg.go[4]};
    const uint8_t * row = rg.base + (size_t) (hits->e[ent].slot - rg.lo) * rg.blob +
                          (r < F ? (size_t) r * rg.grow : rg.up_off + (size_t) (r - F) * rg.grow);
    for (int p0 = 0; p0 < n; p0 += 4) {
        const uint32_t tq = tp >> (4 * p0);
        float acc[4];
        row_dots_n<FMT, NT, LPR>(min(4, n - p0), row, o, E / 32, sub, tq, aq, cols_pad, ad, cols_pad / 32, acc);
        if (sub == 0 && r0 < 2 * F) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (p0 + i < n) gu_out[((size_t) ent * kMaxWindow + ((tq >> (4 * i)) & 15)) * 2 * F + r] = acc[i];
        }
    }
}

template <int FMT, int NT, int LPR>
__device__ __forceinline__ void down_r_body(const HitList * __restrict__ hits, int ent, const MoeRegion & rg, int F, int E,
                                            const int8_t * __restrict__ hq, const float * __restrict__ hd,
                                            float * __restrict__ part) {
    int n;
    const uint32_t tp = pack_tokens(hits->e[ent].mask, n);
    const int nb = F / 32;
    rt_tables_load<FMT>();
    __syncthreads();
    constexpr int RPW = 32 / LPR;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane % LPR;
    const int r0 = blockIdx.y * (8 * RPW) + warp * RPW + lane / LPR, r = min(r0, E - 1);
    const ROff o{rg.dof[0], rg.dof[1], rg.dof[2], rg.dof[3], rg.dof[4]};
    const uint8_t * row = rg.base + (size_t) (hits->e[ent].slot - rg.lo) * rg.blob + rg.down_off + (size_t) r * rg.drow;
    const size_t b0 = (size_t) ent * kMaxWindow;
    for (int p0 = 0; p0 < n; p0 += 4) {
        const uint32_t tq = tp >> (4 * p0);
        float acc[4];
        row_dots_n<FMT, NT, LPR>(min(4, n - p0), row, o, nb, sub, tq, hq + b0 * F, F, hd + b0 * nb, nb, acc);
        if (sub == 0 && r0 < E) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (p0 + i < n) part[(b0 + ((tq >> (4 * i)) & 15)) * E + r] = acc[i];
        }
    }
}

__device__ __forceinline__ int region_of(const MoeRegions & R, int slot) {
    for (int i = 0; i < R.n; ++i)
        if (slot >= R.r[i].lo && slot < R.r[i].hi) return i;
    return -1;
}

#define BNK_REGION_SWITCH(fmt, BODY)                                   \
    switch (fmt) {                                                       \
        case QT_Q4_K: BODY(QT_Q4_K); break;                              \
        case QT_Q5_K: BODY(QT_Q5_K); break;                              \
        case QT_Q6_K: BODY(QT_Q6_K); break;                              \
        case QT_IQ2_S: BODY(QT_IQ2_S); break;                            \
        case QT_IQ2_XXS: BODY(QT_IQ2_XXS); break;                        \
        case QT_IQ2_XS: BODY(QT_IQ2_XS); break;                          \
        case QT_IQ3_XXS: BODY(QT_IQ3_XXS); break;                        \
        case QT_IQ3_S: BODY(QT_IQ3_S); break;                            \
        case QT_IQ4_NL: BODY(QT_IQ4_NL); break;                          \
        case QT_IQ4_XS: BODY(QT_IQ4_XS); break;                          \
        case QT_Q8_0: BODY(QT_Q8_0); break;                              \
        case QT_Q2_0: BODY(QT_Q2_0); break;                              \
        case QT_NVFP4: BODY(QT_NVFP4); break;                            \
        default: break;                                                  \
    }

template <int NT, int LPR>
__global__ void __launch_bounds__(256) moe_gu_rm_k(const HitList * __restrict__ hits, MoeRegions R, int F, int E,
                                                   const int8_t * __restrict__ aq, const float * __restrict__ ad,
                                                   int64_t cols_pad, float * __restrict__ gu_out) {
    const int ent = blockIdx.x;
    if (ent >= hits->n) return;
    const int reg = region_of(R, hits->e[ent].slot);
    if (reg < 0) return;
    const MoeRegion & rg = R.r[reg];
#define BNK_GU_BODY(FMT) gu_r_body<FMT, NT, LPR>(hits, ent, rg, F, E, aq, ad, cols_pad, gu_out)
    BNK_REGION_SWITCH(rg.gate_type, BNK_GU_BODY)
#undef BNK_GU_BODY
}

template <int NT, int LPR>
__global__ void __launch_bounds__(256) moe_down_rm_k(const HitList * __restrict__ hits, MoeRegions R, int F, int E,
                                                     const int8_t * __restrict__ hq, const float * __restrict__ hd,
                                                     float * __restrict__ part) {
    const int ent = blockIdx.x;
    if (ent >= hits->n) return;
    const int reg = region_of(R, hits->e[ent].slot);
    if (reg < 0) return;
    const MoeRegion & rg = R.r[reg];
#define BNK_DN_BODY(FMT) down_r_body<FMT, NT, LPR>(hits, ent, rg, F, E, hq, hd, part)
    BNK_REGION_SWITCH(rg.down_type, BNK_DN_BODY)
#undef BNK_DN_BODY
}

static int lpr_for(int nsb) { return nsb >= 128 ? 32 : nsb >= 48 ? 16 : nsb >= 24 ? 8 : 4; }

MoeRegions moe_regions(const MoeLayerDesc * descs, int n) {
    MoeRegions R;
    R.n = std::min(n, kMaxRegions);
    for (int i = 0; i < R.n; ++i) {
        const MoeLayerDesc & d = descs[i];
        MoeRegion & rg = R.r[i];
        rg.base = d.base;
        rg.blob = d.blob;
        rg.up_off = d.gate_bytes;
        rg.grow = d.grow;
        rg.down_off = d.gate_bytes + d.up_bytes;
        rg.drow = d.drow;
        for (int q = 0; q < 5; ++q) { rg.go[q] = d.go[q]; rg.dof[q] = d.dof[q]; }
        rg.lo = d.slot_lo;
        rg.hi = d.slot_hi;
        rg.gate_type = d.gate_type;
        rg.down_type = d.down_type;
    }
    return R;
}

template <int NT>
static void regions_launch(const MoeRegions & R, const ExpScales & scales, int cold_lo, MoeScratch & s, const ActQ8 & xq,
                           int T, int k, int E, int F, cudaStream_t st) {
    const int maxent = T * k;
    auto gu = [&]<int LPR>() {
        constexpr int rows = 8 * (32 / LPR);
        moe_gu_rm_k<NT, LPR><<<dim3(maxent, (2 * F + rows - 1) / rows), 256, 0, st>>>(s.hits, R, F, E, xq.q, xq.d, xq.cols_pad, s.gu);
    };
    static const int lpr_env = getenv("BNK_MOE_LPR")                         ? atoi(getenv("BNK_MOE_LPR"))
                               : getenv("BNK_EXACT") && atoi(getenv("BNK_EXACT")) ? std::min(8, lpr_for(E / 32))
                                                                                 : 0;
    switch (lpr_env ? lpr_env : NT == 1 ? lpr_for(E / 32) : std::min(8, lpr_for(E / 32))) {
        case 32: gu.template operator()<32>(); break;
        case 16: gu.template operator()<16>(); break;
        case 8: gu.template operator()<8>(); break;
        default: gu.template operator()<4>();
    }
    moe_act_k<<<dim3(maxent, T), F <= 1024 ? F : 1024, 0, st>>>(s.hits, s.gu, F, s.hq, s.hd, scales, cold_lo);
    auto dn = [&]<int LPR>() {
        constexpr int rows = 8 * (32 / LPR);
        moe_down_rm_k<NT, LPR><<<dim3(maxent, (E + rows - 1) / rows), 256, 0, st>>>(s.hits, R, F, E, s.hq, s.hd, s.part);
    };
    switch (lpr_for(F / 32)) {
        case 32: dn.template operator()<32>(); break;
        case 16: dn.template operator()<16>(); break;
        case 8: dn.template operator()<8>(); break;
        default: dn.template operator()<4>();
    }
}

void moe_hits_regions(const MoeRegions & R, const ExpScales & scales, int cold_lo, MoeScratch & s, const ActQ8 & xq, int T,
                      int k, int E, int F, cudaStream_t st) {
    if (T == 1) regions_launch<1>(R, scales, cold_lo, s, xq, T, k, E, F, st);
    else if (T == 2) regions_launch<2>(R, scales, cold_lo, s, xq, T, k, E, F, st);
    else if (T <= 4) regions_launch<4>(R, scales, cold_lo, s, xq, T, k, E, F, st);
    else regions_launch<8>(R, scales, cold_lo, s, xq, T, k, E, F, st);
    check_launch("moe_hits_regions");
}

// phase 0: gate / up for this region's experts; 1: the activation of every entry (once per layer); 2: down
template <int NT>
static void hits_launch_r(const MoeLayerDesc & d, MoeScratch & s, const ActQ8 & xq, int T, int k, int E, int F,
                          cudaStream_t st, int phase, int cold_lo) {
    const int maxent = T * k;
    const ROff go{d.go[0], d.go[1], d.go[2], d.go[3], d.go[4]}, dof{d.dof[0], d.dof[1], d.dof[2], d.dof[3], d.dof[4]};
    auto gu = [&]<int FMT>() {
        auto L = [&]<int LPR>() {
            constexpr int rows = 8 * (32 / LPR);
            moe_gu_r_k<FMT, NT, LPR><<<dim3(maxent, (2 * F + rows - 1) / rows), 256, 0, st>>>(
                s.hits, d.base, d.blob, d.gate_bytes, d.grow, go, F, E, xq.q, xq.d, xq.cols_pad, s.gu, d.slot_lo, d.slot_hi);
        };
        // several tokens: 8 lanes per row (more rows in flight per warp; bench_moe, 5 tokens: 187 -> 179 us on an
        // IQ2_S layer); one token: lpr_for (16 for 2,560 columns), which is faster there
        // (BNK_EXACT=1: 8 for every window, so the sums do not depend on the window's size)
        static const int lpr_env = getenv("BNK_MOE_LPR")                         ? atoi(getenv("BNK_MOE_LPR"))
                                   : getenv("BNK_EXACT") && atoi(getenv("BNK_EXACT")) ? std::min(8, lpr_for(E / 32))
                                                                                     : 0;
        switch (lpr_env ? lpr_env : NT == 1 ? lpr_for(E / 32) : std::min(8, lpr_for(E / 32))) {
            case 32: L.template operator()<32>(); break;
            case 16: L.template operator()<16>(); break;
            case 8: L.template operator()<8>(); break;
            default: L.template operator()<4>();
        }
    };
    if (phase == 0) BNK_DISPATCH_R(d.gate_type, gu);
    if (phase == 1) moe_act_k<<<dim3(maxent, T), F <= 1024 ? F : 1024, 0, st>>>(s.hits, s.gu, F, s.hq, s.hd, d.scales, cold_lo);
    auto dn = [&]<int FMT>() {
        auto L = [&]<int LPR>() {
            constexpr int rows = 8 * (32 / LPR);
            moe_down_r_k<FMT, NT, LPR><<<dim3(maxent, (E + rows - 1) / rows), 256, 0, st>>>(
                s.hits, d.base, d.blob, d.gate_bytes + d.up_bytes, d.drow, dof, F, E, s.hq, s.hd, s.part, d.slot_lo, d.slot_hi);
        };
        switch (lpr_for(F / 32)) {
            case 32: L.template operator()<32>(); break;
            case 16: L.template operator()<16>(); break;
            case 8: L.template operator()<8>(); break;
            default: L.template operator()<4>();
        }
    };
    if (phase == 2) BNK_DISPATCH_R(d.down_type, dn);
}

// ---- cache fill: ggml blob -> R slot, one thread per row
// one thread per (row, sub-block) of an expert: gate/up rows have gl.nsb sub-blocks, down rows dl.nsb
__global__ void moe_repack_k(const uint8_t * src, size_t src_stride, uint8_t * const * dst, int F, int E, RLayout gl,
                             size_t g_srb, RLayout dl, size_t d_srb) {
    const int e = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_gu = (int64_t) 2 * F * gl.nsb;
    if (i >= n_gu + (int64_t) E * dl.nsb) return;
    const uint8_t * s = src + (size_t) e * src_stride;
    uint8_t * d = dst[e];
    if (i < n_gu) {
        const int r = (int) (i / gl.nsb), sb = (int) (i % gl.nsb);
        rfmt_repack_sb(gl, s + (size_t) r * g_srb, d + (size_t) r * gl.row_bytes, sb);
    } else {
        const int64_t j = i - n_gu;
        const int r = (int) (j / dl.nsb), sb = (int) (j % dl.nsb);
        rfmt_repack_sb(dl, s + (size_t) 2 * F * g_srb + (size_t) r * d_srb,
                       d + (size_t) 2 * F * gl.row_bytes + (size_t) r * dl.row_bytes, sb);
    }
}

void moe_repack_blobs(const uint8_t * src, size_t src_stride, uint8_t * const * dst, int n, int F, int E,
                      int gate_type, size_t g_rb, int down_type, size_t d_rb, cudaStream_t st) {
    if (n <= 0) return;
    const RLayout gl = r_layout(gate_type, E), dl = r_layout(down_type, F);
    const int64_t work = (int64_t) 2 * F * gl.nsb + (int64_t) E * dl.nsb;
    moe_repack_k<<<dim3((unsigned) ((work + 255) / 256), n), 256, 0, st>>>(src, src_stride, dst, F, E, gl, g_rb, dl, d_rb);
    check_launch("moe_repack");
}

template <int NT>
static void hits_launch(const MoeLayerDesc & d, MoeScratch & s, const ActQ8 & xq, int T, int k, int E, int F,
                        cudaStream_t st, int phase, int cold_lo) {
    const int maxent = T * k;
    auto gu = [&]<int FMT>() {
        moe_gu_k<FMT, NT><<<dim3(maxent, (2 * F + 7) / 8), 256, 0, st>>>(s.hits, d.base, d.blob, d.gate_bytes, d.grow,
                                                                         F, E, xq.q, xq.d, xq.cols_pad, s.gu, d.slot_lo, d.slot_hi);
    };
    if (phase == 0) BNK_DISPATCH_DP4A(d.gate_type, gu);
    if (phase == 1) moe_act_k<<<dim3(maxent, T), F <= 1024 ? F : 1024, 0, st>>>(s.hits, s.gu, F, s.hq, s.hd, d.scales, cold_lo);
    auto dn = [&]<int FMT>() {
        moe_down_k<FMT, NT><<<dim3(maxent, (E + 7) / 8), 256, 0, st>>>(s.hits, d.base, d.blob,
                                                                      d.gate_bytes + d.up_bytes, d.drow, F, E, s.hq,
                                                                      s.hd, s.part, d.slot_lo, d.slot_hi);
    };
    if (phase == 2) BNK_DISPATCH_DP4A(d.down_type, dn);
}

void moe_hits(const MoeLayerDesc & d, MoeScratch & s, const ActQ8 & xq, int T, int k, int E, int F, cudaStream_t st,
              int phase, int cold_lo) {
    for (int ph = phase < 0 ? 0 : phase; ph <= (phase < 0 ? 2 : phase); ++ph) {
        if (d.rlay) {
            if (T == 1) hits_launch_r<1>(d, s, xq, T, k, E, F, st, ph, cold_lo);
            else if (T == 2) hits_launch_r<2>(d, s, xq, T, k, E, F, st, ph, cold_lo);
            else if (T <= 4) hits_launch_r<4>(d, s, xq, T, k, E, F, st, ph, cold_lo);
            else hits_launch_r<8>(d, s, xq, T, k, E, F, st, ph, cold_lo);
            check_launch("moe_hits_r");
            continue;
        }
        if (T == 1) hits_launch<1>(d, s, xq, T, k, E, F, st, ph, cold_lo);
        else if (T == 2) hits_launch<2>(d, s, xq, T, k, E, F, st, ph, cold_lo);
        else if (T <= 4) hits_launch<4>(d, s, xq, T, k, E, F, st, ph, cold_lo);
        else hits_launch<8>(d, s, xq, T, k, E, F, st, ph, cold_lo);
        check_launch("moe_hits");
    }
}

// ------------------------------------------------------------------------------------------- wait / reduce
__global__ void moe_wait_k(const HitList * hits, MoeMsg * msg, const uint32_t * seq) {
    const int L = g_moe_layer & 63;
    g_moe_ts[L][1] = gtimer();
    if (hits->n_miss != 0) {
        const uint32_t want = *seq;
        while (msg->seq_done != want) __nanosleep(200);
        __threadfence_system();
    }
    g_moe_ts[L][2] = gtimer();
    g_moe_layer = L + 1;
}

void moe_debug_reset(cudaStream_t st) {
    static const int zero = 0;
    cudaMemcpyToSymbolAsync(g_moe_layer, &zero, 4, 0, cudaMemcpyHostToDevice, st);
}

void moe_debug_times(uint64_t (*out)[3], int n) {
    cudaMemcpyFromSymbol(out, g_moe_ts, sizeof(uint64_t) * 3 * n);
    int zero = 0;
    cudaMemcpyToSymbol(g_moe_layer, &zero, 4);
}
void moe_wait(const MoeScratch & s, MoeMsg * msg, const uint32_t * seq, cudaStream_t st) {
    moe_wait_k<<<1, 1, 0, st>>>(s.hits, msg, seq);
}

__global__ void moe_reduce_k(const HitList * __restrict__ hits, const float * __restrict__ part, MoeMsg * msg,
                             const float * __restrict__ shared, const float * __restrict__ sgate,
                             float * __restrict__ out, int E, int k, const uint32_t * seq, int raw_gate) {
    // seq: moe_wait's work first (the CPU's answer when the layer had misses; block 0 keeps the timestamps), which
    // saves its launch
    if (seq) {
        const bool b0 = blockIdx.x == 0 && blockIdx.y == 0;
        if (threadIdx.x == 0) {
            const int L = g_moe_layer & 63;
            if (b0) g_moe_ts[L][1] = gtimer();
            if (hits->n_miss != 0) {
                const uint32_t want = *seq;
                while (msg->seq_done != want) __nanosleep(200);
                __threadfence_system();
            }
            if (b0) {
                g_moe_ts[L][2] = gtimer();
                g_moe_layer = L + 1;   // only block 0 uses it here; the next plan runs after this launch
            }
        }
    }
    // a token's experts added in its own routing order (pair t * k + slot), so the sum does not depend on the other
    // tokens of the window (the hit list's order does); the entries and weights read once per block
    __shared__ int s_ent[32];
    __shared__ float s_w[32];
    const int t = blockIdx.y;
    if (threadIdx.x < k) {
        const int ent = hits->pair_ent[t * k + threadIdx.x];
        s_ent[threadIdx.x] = ent;
        s_w[threadIdx.x] = ent >= 0 ? hits->e[ent].w[t] : 0.f;
    }
    __syncthreads();
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= E) return;
    const float sg = sgate[t];
    float acc = shared[(size_t) t * E + r] * (raw_gate ? 1.f / (1.f + __expf(-sg)) : sg);
    for (int q = 0; q < k; ++q)
        if (s_ent[q] >= 0) acc += s_w[q] * part[((size_t) s_ent[q] * kMaxWindow + t) * E + r];
    if (hits->n_miss > 0) acc += ((volatile float *) msg->out(E))[(size_t) t * E + r];
    out[(size_t) t * E + r] = acc;
}

void moe_reduce(const MoeScratch & s, MoeMsg * msg, const float * shared, const float * sgate, float * out, int T,
                int k, int E, cudaStream_t st, const uint32_t * wait_seq, bool raw_gate) {
    if (k > 32) { fprintf(stderr, "moe_reduce: k %d > 32\n", k); abort(); }
    moe_reduce_k<<<dim3((E + 255) / 256, T), 256, 0, st>>>(s.hits, s.part, msg, shared, sgate, out, E, k, wait_seq, raw_gate ? 1 : 0);
}

}  // namespace bnk

namespace bnk {

template <int FMT, bool RL>
__global__ void __launch_bounds__(256) moe_gu_list_k(const PfItem * __restrict__ items, size_t gate_bytes, size_t grow,
                                                     int F, int E, const int8_t * __restrict__ aq,
                                                     const float * __restrict__ ad, int64_t cols_pad,
                                                     float * __restrict__ gu, ROff ro) {
    const PfItem & it = items[blockIdx.x];
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (r >= 2 * F) return;
    const uint8_t * row = it.blob + (r < F ? (size_t) r * grow : gate_bytes + (size_t) (r - F) * grow);
    const int nsb = E / 32, n = it.n;
    const int64_t nb = cols_pad / 32;
    float acc[kMaxWindow];
#pragma unroll
    for (int t = 0; t < kMaxWindow; ++t) acc[t] = 0.f;
    for (int sb = lane; sb < nsb; sb += 32) {
        Unpacked u;
        if constexpr (RL) RT<FMT>::unpack(row, ro, sb, u);
        else unpack_sub<FMT>(row, sb, u);
#pragma unroll
        for (int t = 0; t < kMaxWindow; ++t) {
            if (t < n) {
                const int64_t tk = it.tok[t];
                const int4 * ap = (const int4 *) (aq + tk * cols_pad + sb * 32);
                acc[t] += RL ? qdot_sub<FMT, RT<FMT>::HAS_MIN>(u, __ldg(ap), __ldg(ap + 1), __ldg(ad + tk * nb + sb))
                             : qdot_sub<FMT>(u, __ldg(ap), __ldg(ap + 1), __ldg(ad + tk * nb + sb));
            }
        }
    }
#pragma unroll
    for (int t = 0; t < kMaxWindow; ++t) {
        if (t < n) {
            const float v = warp_reduce_sum(acc[t]);
            if (lane == 0) gu[(size_t) it.pair[t] * 2 * F + r] = v;
        }
    }
}

// grid (items, 8 tokens); F threads
__global__ void moe_act_list_k(const PfItem * __restrict__ items, const float * __restrict__ gu, int F,
                               int8_t * __restrict__ hq, float * __restrict__ hd, ExpScales es) {
    const PfItem & it = items[blockIdx.x];
    if ((int) blockIdx.y >= it.n) return;
    const size_t pr = it.pair[blockIdx.y];
    const float * g = gu + pr * 2 * F;
    const float sg = es.gate ? es.gate[it.expert] : 1.f, sud = es.up ? es.up[it.expert] * es.down[it.expert] : 1.f;
    for (int r = threadIdx.x; r < F; r += blockDim.x) {
        const float gv = g[r] * sg;
        const float h = gv / (1.f + __expf(-gv)) * g[F + r] * sud;
        float d;
        const int8_t q = quant_lane(h, d);
        hq[pr * F + r] = q;
        if ((threadIdx.x & 31) == 0) hd[pr * (F / 32) + r / 32] = d;
    }
}

template <int FMT, bool RL>
__global__ void __launch_bounds__(256) moe_down_list_k(const PfItem * __restrict__ items, size_t down_off, size_t drow,
                                                       int F, int E, const int8_t * __restrict__ hq,
                                                       const float * __restrict__ hd, half * __restrict__ D, ROff ro) {
    const PfItem & it = items[blockIdx.x];
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (r >= E) return;
    const uint8_t * row = it.blob + down_off + (size_t) r * drow;
    const int nsb = F / 32, n = it.n;
    float acc[kMaxWindow];
#pragma unroll
    for (int t = 0; t < kMaxWindow; ++t) acc[t] = 0.f;
    for (int sb = lane; sb < nsb; sb += 32) {
        Unpacked u;
        if constexpr (RL) RT<FMT>::unpack(row, ro, sb, u);
        else unpack_sub<FMT>(row, sb, u);
#pragma unroll
        for (int t = 0; t < kMaxWindow; ++t) {
            if (t < n) {
                const size_t pr = it.pair[t];
                const int4 * ap = (const int4 *) (hq + pr * F + sb * 32);
                acc[t] += RL ? qdot_sub<FMT, RT<FMT>::HAS_MIN>(u, ap[0], ap[1], hd[pr * (F / 32) + sb])
                             : qdot_sub<FMT>(u, ap[0], ap[1], hd[pr * (F / 32) + sb]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < kMaxWindow; ++t) {
        if (t < n) {
            const float v = warp_reduce_sum(acc[t]);
            if (lane == 0) D[(size_t) it.pair[t] * E + r] = __float2half(v);
        }
    }
}

#ifndef BNK_LIST_PASS
#define BNK_LIST_PASS 8
#endif
constexpr int kListPass = BNK_LIST_PASS;   // tokens per pass over a row

// One row's dot products with N of an item's tokens (activations at a + idx_i * sa, scales at s + idx_i * ss): LPR
// lanes per row stride over its sub-blocks, then reduce.
template <int FMT, int LPR, int N>
__device__ __forceinline__ void list_dots(const uint8_t * row, const ROff & o, int nb, int sub, const int64_t (&idx)[kListPass],
                                          const int8_t * a, int64_t sa, const float * s, int64_t ss,
                                          float (&acc)[kListPass]) {
#pragma unroll
    for (int i = 0; i < N; ++i) acc[i] = 0.f;
    for (int sb = sub; sb < nb; sb += LPR) {
        Unpacked u;
        unpack_sm<FMT>(row, o, sb, u);
#pragma unroll
        for (int i = 0; i < N; ++i) {
            const int4 * ap = (const int4 *) (a + idx[i] * sa + sb * 32);
            acc[i] += qdot_sub<FMT, RT<FMT>::HAS_MIN>(u, __ldg(ap), __ldg(ap + 1), __ldg(s + idx[i] * ss + sb));
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int off = LPR / 2; off > 0; off >>= 1) acc[i] += __shfl_xor_sync(0xffffffff, acc[i], off);
    }
}

// The R-layout (resident) list items, as the decode kernels run their hits: LPR lanes per row, up to 4 tokens per
// pass. GU: gate and up rows (nrows 2F over E columns) into gu; else down rows (E over F) into D.
template <int FMT, int LPR, bool GU>
__global__ void __launch_bounds__(256) moe_list_r_k(const PfItem * __restrict__ items, size_t off, size_t rowb, int F,
                                                    int nrows, int ncols, const int8_t * __restrict__ a, int64_t sa,
                                                    const float * __restrict__ s, int64_t ss, float * __restrict__ gu,
                                                    half * __restrict__ D, ROff ro) {
    const PfItem & it = items[blockIdx.x];
    rt_tables_load<FMT>();
    __syncthreads();
    constexpr int RPW = 32 / LPR;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, sub = lane % LPR;
    const int r0 = blockIdx.y * (8 * RPW) + warp * RPW + lane / LPR, r = min(r0, nrows - 1);   // all lanes shuffle
    const uint8_t * row = GU ? it.blob + (r < F ? (size_t) r * rowb : off + (size_t) (r - F) * rowb)
                             : it.blob + off + (size_t) r * rowb;
    const int n = it.n, nb = ncols / 32;
    for (int p0 = 0; p0 < n; p0 += kListPass) {
        const int m = min(kListPass, n - p0);
        int64_t idx[kListPass];
#pragma unroll
        for (int i = 0; i < kListPass; ++i) idx[i] = GU ? it.tok[p0 + min(i, m - 1)] : it.pair[p0 + min(i, m - 1)];
        float acc[kListPass];
        switch (m) {   // block-uniform
            case 1: list_dots<FMT, LPR, 1>(row, ro, nb, sub, idx, a, sa, s, ss, acc); break;
            case 2: list_dots<FMT, LPR, 2>(row, ro, nb, sub, idx, a, sa, s, ss, acc); break;
            case 3: list_dots<FMT, LPR, 3>(row, ro, nb, sub, idx, a, sa, s, ss, acc); break;
#if BNK_LIST_PASS > 4
            case 4: list_dots<FMT, LPR, 4>(row, ro, nb, sub, idx, a, sa, s, ss, acc); break;
            case 5: list_dots<FMT, LPR, 5>(row, ro, nb, sub, idx, a, sa, s, ss, acc); break;
            case 6: list_dots<FMT, LPR, 6>(row, ro, nb, sub, idx, a, sa, s, ss, acc); break;
            case 7: list_dots<FMT, LPR, 7>(row, ro, nb, sub, idx, a, sa, s, ss, acc); break;
#endif
            default: list_dots<FMT, LPR, kListPass>(row, ro, nb, sub, idx, a, sa, s, ss, acc);
        }
        if (sub == 0 && r0 < nrows) {
#pragma unroll
            for (int i = 0; i < kListPass; ++i)
                if (i < m) {
                    const size_t pr = it.pair[p0 + i];
                    if constexpr (GU) gu[pr * 2 * F + r] = acc[i];
                    else D[pr * nrows + r] = __float2half(acc[i]);
                }
        }
    }
}

// list_dots over activations staged in shared memory (token i's row at xs + i * ncols, scales at ds + i * nb)
template <int FMT, int LPR, int N>
__device__ __forceinline__ void list_dots_s(const uint8_t * row, const ROff & o, int nb, int sub, const int8_t * xs,
                                            const float * ds, float (&acc)[kMaxWindow]) {
#pragma unroll
    for (int i = 0; i < N; ++i) acc[i] = 0.f;
    for (int sb = sub; sb < nb; sb += LPR) {
        Unpacked u;
        unpack_sm<FMT>(row, o, sb, u);
#pragma unroll
        for (int i = 0; i < N; ++i) {
            const int4 * ap = (const int4 *) (xs + (size_t) i * nb * 32 + sb * 32);
            acc[i] += qdot_sub<FMT, RT<FMT>::HAS_MIN>(u, ap[0], ap[1], ds[i * nb + sb]);
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int off = LPR / 2; off > 0; off >>= 1) acc[i] += __shfl_xor_sync(0xffffffff, acc[i], off);
    }
}

// moe_list_r_k with the item's activations (all its tokens, one pass) copied to shared memory once per block, and
// each block walking RIT row groups: the activations were re-read from L1/L2 by every row (32 B per token per
// sub-block against ~14 B of weights), which held the kernel to ~250 GB/s of weights.
template <int FMT, int LPR, bool GU, int RIT>
__global__ void __launch_bounds__(256) moe_list_s_k(const PfItem * __restrict__ items, size_t off, size_t rowb, int F,
                                                    int nrows, int ncols, const int8_t * __restrict__ a, int64_t sa,
                                                    const float * __restrict__ s, int64_t ss, float * __restrict__ gu,
                                                    half * __restrict__ D, ROff ro) {
    extern __shared__ int4 list_smem[];
    const PfItem & it = items[blockIdx.x];
    const int n = it.n, nb = ncols / 32;
    int8_t * xs = (int8_t *) list_smem;
    float * ds = (float *) (xs + (size_t) kMaxWindow * ncols);
    rt_tables_load<FMT>();
    for (int i = 0; i < n; ++i) {
        const int64_t src = GU ? it.tok[i] : it.pair[i];
        const int4 * g = (const int4 *) (a + src * sa);
        int4 * d = (int4 *) (xs + (size_t) i * ncols);
        for (int j = threadIdx.x; j < ncols / 16; j += blockDim.x) d[j] = __ldg(g + j);
        for (int j = threadIdx.x; j < nb; j += blockDim.x) ds[i * nb + j] = __ldg(s + src * ss + j);
    }
    __syncthreads();
    constexpr int RPW = 32 / LPR;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, sub = lane % LPR;
    for (int g = 0; g < RIT; ++g) {
        const int r0 = (blockIdx.y * RIT + g) * (8 * RPW) + warp * RPW + lane / LPR;
        if ((blockIdx.y * RIT + g) * (8 * RPW) >= nrows) break;   // block-uniform
        const int r = min(r0, nrows - 1);   // all lanes shuffle
        const uint8_t * row = GU ? it.blob + (r < F ? (size_t) r * rowb : off + (size_t) (r - F) * rowb)
                                 : it.blob + off + (size_t) r * rowb;
        float acc[kMaxWindow];
        switch (n) {   // block-uniform
            case 1: list_dots_s<FMT, LPR, 1>(row, ro, nb, sub, xs, ds, acc); break;
            case 2: list_dots_s<FMT, LPR, 2>(row, ro, nb, sub, xs, ds, acc); break;
            case 3: list_dots_s<FMT, LPR, 3>(row, ro, nb, sub, xs, ds, acc); break;
            case 4: list_dots_s<FMT, LPR, 4>(row, ro, nb, sub, xs, ds, acc); break;
            case 5: list_dots_s<FMT, LPR, 5>(row, ro, nb, sub, xs, ds, acc); break;
            case 6: list_dots_s<FMT, LPR, 6>(row, ro, nb, sub, xs, ds, acc); break;
            case 7: list_dots_s<FMT, LPR, 7>(row, ro, nb, sub, xs, ds, acc); break;
            default: list_dots_s<FMT, LPR, kMaxWindow>(row, ro, nb, sub, xs, ds, acc);
        }
        if (sub == 0 && r0 < nrows) {
#pragma unroll
            for (int i = 0; i < kMaxWindow; ++i)
                if (i < n) {
                    const size_t pr = it.pair[i];
                    if constexpr (GU) gu[pr * 2 * F + r] = acc[i];
                    else D[pr * nrows + r] = __float2half(acc[i]);
                }
        }
    }
}

void moe_list(const PfItem * items, int n_items, const MoeLayerDesc & d, const ActQ8 & xq, int E, int F, float * gu,
              int8_t * hq, float * hd, half * D, cudaStream_t st) {
    if (n_items <= 0) return;
    const ROff go{d.go[0], d.go[1], d.go[2], d.go[3], d.go[4]}, dof{d.dof[0], d.dof[1], d.dof[2], d.dof[3], d.dof[4]};
    static const bool old_list = getenv("BNK_LIST_OLD") != nullptr;
    // BNK_LIST_SMEM=0: activations read from global memory (moe_list_r_k)
    static const bool list_smem = !getenv("BNK_LIST_SMEM") || atoi(getenv("BNK_LIST_SMEM")) != 0;
    const size_t smem_gu = (size_t) kMaxWindow * (E + E / 32 * 4), smem_dn = (size_t) kMaxWindow * (F + F / 32 * 4);
    if (d.rlay && !old_list && list_smem && smem_gu <= 32768 && E % 16 == 0 && F % 16 == 0) {
        auto gk = [&]<int FMT>() {
            constexpr int LPR = 8, RIT = 4, rows = 8 * (32 / LPR) * RIT;
            moe_list_s_k<FMT, LPR, true, RIT><<<dim3(n_items, (2 * F + rows - 1) / rows), 256, smem_gu, st>>>(
                items, d.gate_bytes, d.grow, F, 2 * F, E, xq.q, xq.cols_pad, xq.d, xq.cols_pad / 32, gu, nullptr, go);
        };
        BNK_DISPATCH_R(d.gate_type, gk);
        moe_act_list_k<<<dim3(n_items, kMaxWindow), F <= 1024 ? F : 1024, 0, st>>>(items, gu, F, hq, hd, d.scales);
        auto dk = [&]<int FMT>() {
            constexpr int LPR = 4, RIT = 4, rows = 8 * (32 / LPR) * RIT;
            moe_list_s_k<FMT, LPR, false, RIT><<<dim3(n_items, (E + rows - 1) / rows), 256, smem_dn, st>>>(
                items, d.gate_bytes + d.up_bytes, d.drow, F, E, F, hq, F, hd, F / 32, nullptr, D, dof);
        };
        BNK_DISPATCH_R(d.down_type, dk);
        check_launch("moe_list_s");
        return;
    }
    if (d.rlay && !old_list) {
        auto gk = [&]<int FMT>() {
            auto L = [&]<int LPR>() {
                constexpr int rows = 8 * (32 / LPR);
                moe_list_r_k<FMT, LPR, true><<<dim3(n_items, (2 * F + rows - 1) / rows), 256, 0, st>>>(
                    items, d.gate_bytes, d.grow, F, 2 * F, E, xq.q, xq.cols_pad, xq.d, xq.cols_pad / 32, gu, nullptr, go);
            };
            L.template operator()<8>();   // 4 and 16 measured the same
        };
        BNK_DISPATCH_R(d.gate_type, gk);
        moe_act_list_k<<<dim3(n_items, kMaxWindow), F <= 1024 ? F : 1024, 0, st>>>(items, gu, F, hq, hd, d.scales);
        auto dk = [&]<int FMT>() {
            auto L = [&]<int LPR>() {
                constexpr int rows = 8 * (32 / LPR);
                moe_list_r_k<FMT, LPR, false><<<dim3(n_items, (E + rows - 1) / rows), 256, 0, st>>>(
                    items, d.gate_bytes + d.up_bytes, d.drow, F, E, F, hq, F, hd, F / 32, nullptr, D, dof);
            };
            L.template operator()<4>();   // 2 and 8 measured the same or slower
        };
        BNK_DISPATCH_R(d.down_type, dk);
        check_launch("moe_list_r");
        return;
    }
    auto g = [&]<int FMT>() {
        if (d.rlay)
            moe_gu_list_k<FMT, true><<<dim3(n_items, (2 * F + 7) / 8), 256, 0, st>>>(items, d.gate_bytes, d.grow, F, E,
                                                                                   xq.q, xq.d, xq.cols_pad, gu, go);
        else
            moe_gu_list_k<FMT, false><<<dim3(n_items, (2 * F + 7) / 8), 256, 0, st>>>(items, d.gate_bytes, d.grow, F, E,
                                                                                    xq.q, xq.d, xq.cols_pad, gu, go);
    };
    BNK_DISPATCH_DP4A(d.gate_type, g);
    moe_act_list_k<<<dim3(n_items, kMaxWindow), F <= 1024 ? F : 1024, 0, st>>>(items, gu, F, hq, hd, d.scales);
    auto dn = [&]<int FMT>() {
        if (d.rlay)
            moe_down_list_k<FMT, true><<<dim3(n_items, (E + 7) / 8), 256, 0, st>>>(items, d.gate_bytes + d.up_bytes,
                                                                                 d.drow, F, E, hq, hd, D, dof);
        else
            moe_down_list_k<FMT, false><<<dim3(n_items, (E + 7) / 8), 256, 0, st>>>(items, d.gate_bytes + d.up_bytes,
                                                                                  d.drow, F, E, hq, hd, D, dof);
    };
    BNK_DISPATCH_DP4A(d.down_type, dn);
    check_launch("moe_list");
}

}  // namespace bnk
