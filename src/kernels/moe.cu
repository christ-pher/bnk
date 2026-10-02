#include "kernels/moe.h"
#include "kernels/qdot.cuh"

namespace bnk {

__global__ void bump_seq_k(uint32_t * seq) { *seq += 1; }
void bump_seq(uint32_t * seq, cudaStream_t st) { bump_seq_k<<<1, 1, 0, st>>>(seq); }

// ------------------------------------------------------------------------------------------- plan
// One warp. Lane j < T*k takes routed pair j: hits are deduplicated by expert with ballots.
__global__ void moe_plan_k(const int32_t * ids, const float * w, int T, int k, const int32_t * slot_of, HitList * hits,
                           MoeMsg * msg, const float * x, int E, const uint32_t * seq, uint32_t * counts) {
    const int lane = threadIdx.x & 31;
    const int n = T * k;
    __shared__ int n_hit_s, n_miss_s;
    if (threadIdx.x < 32) {
        int nh = 0, nm = 0;
        for (int base = 0; base < n; base += 32) {
            const int j = base + lane;
            const bool valid = j < n;
            const int e = valid ? ids[j] : -1;
            const int t = valid ? j / k : 0;
            const float wt = valid ? w[j] : 0.f;
            const int sl = valid ? slot_of[e] : -1;
            if (valid && counts) atomicAdd(&counts[e], 1u);
            // misses: compact in lane order
            const unsigned mm = __ballot_sync(0xffffffff, valid && sl < 0);
            if (valid && sl < 0) {
                const int pos = nm + __popc(mm & ((1u << lane) - 1));
                msg->miss_t[pos] = t;
                msg->miss_e[pos] = e;
                msg->miss_w[pos] = wt;
            }
            nm += __popc(mm);
            // hits: the first lane (over the whole list) with a given expert creates the entry
            for (int src = 0; src < 32; ++src) {
                const int se = __shfl_sync(0xffffffff, e, src);
                const int ssl = __shfl_sync(0xffffffff, sl, src);
                const int st = __shfl_sync(0xffffffff, t, src);
                const float sw = __shfl_sync(0xffffffff, wt, src);
                if (ssl < 0) continue;  // also covers invalid lanes (sl = -1)
                // find an existing entry (lanes scan the list in parallel)
                int found = -1;
                for (int b = 0; b < nh; b += 32) {
                    const int q = b + lane;
                    const unsigned m = __ballot_sync(0xffffffff, q < nh && hits->e[q].expert == se);
                    if (m) { found = b + __ffs(m) - 1; break; }
                }
                if (found < 0) {
                    found = nh++;
                    if (lane == 0) {
                        hits->e[found].expert = se;
                        hits->e[found].slot = ssl;
                        hits->e[found].mask = 0;
                    }
                }
                if (lane == 0) {
                    hits->e[found].mask |= 1u << st;
                    hits->e[found].w[st] = sw;
                }
                __syncwarp();
            }
        }
        if (lane == 0) {
            hits->n = nh;
            hits->n_miss = nm;
            msg->n_miss = nm;
            msg->T = T;
            n_hit_s = nh;
            n_miss_s = nm;
        }
    }
    __syncthreads();
    if (n_miss_s > 0) {
        float4 * mx = (float4 *) msg->x();
        const float4 * xs = (const float4 *) x;
        for (int i = threadIdx.x; i < T * E / 4; i += blockDim.x) mx[i] = xs[i];
        __threadfence_system();
    }
    __syncthreads();
    if (threadIdx.x == 0) msg->seq_req = *seq;
}

void moe_plan(const int32_t * ids, const float * w, int T, int k, const MoeLayerDesc & d, MoeScratch & s, MoeMsg * msg,
              const float * x, int E, const uint32_t * seq, uint32_t * counts_layer, cudaStream_t st) {
    moe_plan_k<<<1, 256, 0, st>>>(ids, w, T, k, d.slot_of, s.hits, msg, x, E, seq, counts_layer);
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
    if (T == 1) hits_launch<1>(d, s, xq, T, k, E, F, st);
    else if (T == 2) hits_launch<2>(d, s, xq, T, k, E, F, st);
    else if (T <= 4) hits_launch<4>(d, s, xq, T, k, E, F, st);
    else hits_launch<8>(d, s, xq, T, k, E, F, st);
}

// ------------------------------------------------------------------------------------------- wait / reduce
__global__ void moe_wait_k(const HitList * hits, MoeMsg * msg, const uint32_t * seq) {
    if (hits->n_miss == 0) return;
    const uint32_t want = *seq;
    while (msg->seq_done != want) __nanosleep(200);
    __threadfence_system();
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
