#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "kernels/moe.h"
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
__global__ void __launch_bounds__(kMaxRouted) moe_plan_k(const int32_t * ids, const float * w, int T, int k,
                                                         const int32_t * slot_of, HitList * hits, MoeMsg * msg,
                                                         const float * x, int E, const uint32_t * seq,
                                                         uint32_t * counts) {
    __shared__ int s_e[kMaxRouted], s_sl[kMaxRouted];
    __shared__ int s_first[kMaxRouted], s_miss[kMaxRouted];
    __shared__ unsigned s_mask[kMaxRouted];
    __shared__ int s_nh, s_nm;
    const int n = T * k, j = threadIdx.x;
    const bool valid = j < n;
    const int e = valid ? ids[j] : -1;
    const int t = valid ? j / k : 0;
    const float wt = valid ? w[j] : 0.f;
    const int sl = valid ? slot_of[e] : -1;
    if (valid && counts) atomicAdd(&counts[e], 1u);
    s_e[j] = e;
    s_sl[j] = sl;
    s_mask[j] = 0;
    __syncthreads();
    // is this pair its expert's first hit?
    bool first = valid && sl >= 0;
    for (int q = 0; first && q < j; ++q) first = s_e[q] != e;
    s_first[j] = first;
    s_miss[j] = valid && sl < 0;
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
        atomicOr(&s_mask[ent], 1u << t);
        hits->e[ent].w[t] = wt;
        if (q == j) {
            hits->e[ent].expert = e;
            hits->e[ent].slot = sl;
        }
    }
    if (valid && sl < 0) {
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
                                                int64_t cols_pad, float * __restrict__ gu) {
    const int ent = blockIdx.x;
    if (ent >= hits->n) return;
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (r >= 2 * F) return;
    const HitEntry & h = hits->e[ent];
    const uint32_t mask = h.mask;
    const uint8_t * row = base + (size_t) h.slot * blob + (r < F ? (size_t) r * grow : gate_bytes + (size_t) (r - F) * grow);
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

// grid (kMaxRouted, W); block F threads (F % 32 == 0): h = silu(g)*u quantized per 32
__global__ void moe_act_k(const HitList * __restrict__ hits, const float * __restrict__ gu, int F,
                          int8_t * __restrict__ hq, float * __restrict__ hd) {
    const int ent = blockIdx.x, t = blockIdx.y;
    if (ent >= hits->n || !(hits->e[ent].mask & (1u << t))) return;
    const size_t base = (size_t) ent * kMaxWindow + t;
    const float * g = gu + base * 2 * F;
    for (int r = threadIdx.x; r < F; r += blockDim.x) {
        const float gv = g[r];
        const float h = gv / (1.f + __expf(-gv)) * g[F + r];
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
                                                  float * __restrict__ part) {
    const int ent = blockIdx.x;
    if (ent >= hits->n) return;
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (r >= E) return;
    const HitEntry & h = hits->e[ent];
    const uint32_t mask = h.mask;
    const uint8_t * row = base + (size_t) h.slot * blob + down_off + (size_t) r * drow;
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
                                                  int64_t cols_pad, float * __restrict__ gu_out) {
    const int ent = blockIdx.x;
    if (ent >= hits->n) return;
    int n;
    const uint32_t tp = pack_tokens(hits->e[ent].mask, n);
    rt_tables_load<FMT>();
    __syncthreads();
    constexpr int RPW = 32 / LPR;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane % LPR;
    const int r0 = blockIdx.y * (8 * RPW) + warp * RPW + lane / LPR, r = min(r0, 2 * F - 1);   // all lanes shuffle
    const uint8_t * row = base + (size_t) hits->e[ent].slot * blob +
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
                                                    float * __restrict__ part) {
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
    const uint8_t * row = base + (size_t) hits->e[ent].slot * blob + down_off + (size_t) r * drow;
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

static int lpr_for(int nsb) { return nsb >= 128 ? 32 : nsb >= 48 ? 16 : nsb >= 24 ? 8 : 4; }

template <int NT>
static void hits_launch_r(const MoeLayerDesc & d, MoeScratch & s, const ActQ8 & xq, int T, int k, int E, int F,
                          cudaStream_t st) {
    const int maxent = T * k;
    const ROff go{d.go[0], d.go[1], d.go[2], d.go[3], d.go[4]}, dof{d.dof[0], d.dof[1], d.dof[2], d.dof[3], d.dof[4]};
    auto gu = [&]<int FMT>() {
        auto L = [&]<int LPR>() {
            constexpr int rows = 8 * (32 / LPR);
            moe_gu_r_k<FMT, NT, LPR><<<dim3(maxent, (2 * F + rows - 1) / rows), 256, 0, st>>>(
                s.hits, d.base, d.blob, d.gate_bytes, d.grow, go, F, E, xq.q, xq.d, xq.cols_pad, s.gu);
        };
        // several tokens: 8 lanes per row (more rows in flight per warp; bench_moe, 5 tokens: 187 -> 179 us on an
        // IQ2_S layer); one token: lpr_for (16 for 2,560 columns), which is faster there
        static const int lpr_env = getenv("BNK_MOE_LPR") ? atoi(getenv("BNK_MOE_LPR")) : 0;
        switch (lpr_env ? lpr_env : NT == 1 ? lpr_for(E / 32) : std::min(8, lpr_for(E / 32))) {
            case 32: L.template operator()<32>(); break;
            case 16: L.template operator()<16>(); break;
            case 8: L.template operator()<8>(); break;
            default: L.template operator()<4>();
        }
    };
    BNK_DISPATCH_R(d.gate_type, gu);
    moe_act_k<<<dim3(maxent, T), F <= 1024 ? F : 1024, 0, st>>>(s.hits, s.gu, F, s.hq, s.hd);
    auto dn = [&]<int FMT>() {
        auto L = [&]<int LPR>() {
            constexpr int rows = 8 * (32 / LPR);
            moe_down_r_k<FMT, NT, LPR><<<dim3(maxent, (E + rows - 1) / rows), 256, 0, st>>>(
                s.hits, d.base, d.blob, d.gate_bytes + d.up_bytes, d.drow, dof, F, E, s.hq, s.hd, s.part);
        };
        switch (lpr_for(F / 32)) {
            case 32: L.template operator()<32>(); break;
            case 16: L.template operator()<16>(); break;
            case 8: L.template operator()<8>(); break;
            default: L.template operator()<4>();
        }
    };
    BNK_DISPATCH_R(d.down_type, dn);
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
                        cudaStream_t st) {
    const int maxent = T * k;
    auto gu = [&]<int FMT>() {
        moe_gu_k<FMT, NT><<<dim3(maxent, (2 * F + 7) / 8), 256, 0, st>>>(s.hits, d.base, d.blob, d.gate_bytes, d.grow,
                                                                         F, E, xq.q, xq.d, xq.cols_pad, s.gu);
    };
    BNK_DISPATCH_DP4A(d.gate_type, gu);
    moe_act_k<<<dim3(maxent, T), F <= 1024 ? F : 1024, 0, st>>>(s.hits, s.gu, F, s.hq, s.hd);
    auto dn = [&]<int FMT>() {
        moe_down_k<FMT, NT><<<dim3(maxent, (E + 7) / 8), 256, 0, st>>>(s.hits, d.base, d.blob,
                                                                      d.gate_bytes + d.up_bytes, d.drow, F, E, s.hq,
                                                                      s.hd, s.part);
    };
    BNK_DISPATCH_DP4A(d.down_type, dn);
}

void moe_hits(const MoeLayerDesc & d, MoeScratch & s, const ActQ8 & xq, int T, int k, int E, int F, cudaStream_t st) {
    if (d.rlay) {
        if (T == 1) hits_launch_r<1>(d, s, xq, T, k, E, F, st);
        else if (T == 2) hits_launch_r<2>(d, s, xq, T, k, E, F, st);
        else if (T <= 4) hits_launch_r<4>(d, s, xq, T, k, E, F, st);
        else hits_launch_r<8>(d, s, xq, T, k, E, F, st);
        check_launch("moe_hits_r");
        return;
    }
    if (T == 1) hits_launch<1>(d, s, xq, T, k, E, F, st);
    else if (T == 2) hits_launch<2>(d, s, xq, T, k, E, F, st);
    else if (T <= 4) hits_launch<4>(d, s, xq, T, k, E, F, st);
    else hits_launch<8>(d, s, xq, T, k, E, F, st);
    check_launch("moe_hits");
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
                             float * __restrict__ out, int E) {
    const int t = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= E) return;
    float acc = shared[(size_t) t * E + r] * sgate[t];
    const int n = hits->n;
    for (int i = 0; i < n; ++i) {
        const HitEntry & h = hits->e[i];
        if (h.mask & (1u << t)) acc += h.w[t] * part[((size_t) i * kMaxWindow + t) * E + r];
    }
    if (hits->n_miss > 0) acc += ((volatile float *) msg->out(E))[(size_t) t * E + r];
    out[(size_t) t * E + r] = acc;
}
void moe_reduce(const MoeScratch & s, MoeMsg * msg, const float * shared, const float * sgate, float * out, int T,
                int k, int E, cudaStream_t st) {
    moe_reduce_k<<<dim3((E + 255) / 256, T), 256, 0, st>>>(s.hits, s.part, msg, shared, sgate, out, E);
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
                               int8_t * __restrict__ hq, float * __restrict__ hd) {
    const PfItem & it = items[blockIdx.x];
    if ((int) blockIdx.y >= it.n) return;
    const size_t pr = it.pair[blockIdx.y];
    const float * g = gu + pr * 2 * F;
    for (int r = threadIdx.x; r < F; r += blockDim.x) {
        const float gv = g[r];
        const float h = gv / (1.f + __expf(-gv)) * g[F + r];
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

void moe_list(const PfItem * items, int n_items, const MoeLayerDesc & d, const ActQ8 & xq, int E, int F, float * gu,
              int8_t * hq, float * hd, half * D, cudaStream_t st) {
    if (n_items <= 0) return;
    const ROff go{d.go[0], d.go[1], d.go[2], d.go[3], d.go[4]}, dof{d.dof[0], d.dof[1], d.dof[2], d.dof[3], d.dof[4]};
    auto g = [&]<int FMT>() {
        if (d.rlay)
            moe_gu_list_k<FMT, true><<<dim3(n_items, (2 * F + 7) / 8), 256, 0, st>>>(items, d.gate_bytes, d.grow, F, E,
                                                                                   xq.q, xq.d, xq.cols_pad, gu, go);
        else
            moe_gu_list_k<FMT, false><<<dim3(n_items, (2 * F + 7) / 8), 256, 0, st>>>(items, d.gate_bytes, d.grow, F, E,
                                                                                    xq.q, xq.d, xq.cols_pad, gu, go);
    };
    BNK_DISPATCH_DP4A(d.gate_type, g);
    moe_act_list_k<<<dim3(n_items, kMaxWindow), F <= 1024 ? F : 1024, 0, st>>>(items, gu, F, hq, hd);
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
