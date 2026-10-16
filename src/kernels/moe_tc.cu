// Prompt-path routed experts on the tensor cores: every token routed to an expert against the expert's quantized
// rows, the weights unpacked once per 64-token tile straight into fp16 A fragments. No fp16 copy of the expert goes
// through HBM and no per-expert cuBLAS kernel runs: one launch covers every (expert, token tile, row tile) of the
// layer. The product is mma.m8n8k4 as in the decode GEMV (mma.cuh): a warp owns 8 rows of each of its two row
// groups, its 4 quad pairs take 4 consecutive sub-blocks of K and add up at the end; the tile's activations are
// staged in shared memory per K step and read as B fragments.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "kernels/mma.cuh"
#include "kernels/moe.h"
#include "kernels/qdot.cuh"
#include "kernels/rfmt.cuh"

namespace bnk {

constexpr int kTcK = 128;         // K per step: 4 sub-blocks, one per quad pair
constexpr int kTcLd = kTcK + 8;   // shared-memory row stride (halfs): 8 tokens' 16-byte reads hit 8 different banks
// 8 warps per block, one block per SM (the gate/up kernel takes 255 registers). 4 warps and two blocks per SM
// measured slower (every block fills its own copy of the activation tile: twice the L2 traffic per row).
constexpr int kTcWarps = 8, kTcThreads = 32 * kTcWarps;
constexpr int kTcRows = 8 * kTcWarps;   // rows per row group of a block
constexpr int kTcFill = kTcTile * 16 / kTcThreads;   // uint4 of the activation tile each thread stages per step

// a row's sub-block as A fragments: R-layout rows (tables in shared memory when the format has them) or ggml rows
template <int FMT, bool RL>
__device__ __forceinline__ void tc_weights(const uint8_t * row, const ROff & o, int sb, uint32_t (&w)[16]) {
    Unpacked u;
    if constexpr (RL) {
        unpack_sm<FMT>(row, o, sb, u);
        mma_weights_u<RT<FMT>::HAS_MIN>(u, w);
    } else {
        unpack_sub<FMT>(row, sb, u);
        mma_weights_u<QTraits<FMT>::HAS_MIN>(u, w);
    }
}

// Bytes per sub-block of an R-layout row's two big fields (the codes, and the high bits / signs / scales that come
// next), to prefetch the next step's sub-block into L1 while the current step's products run. The remaining
// fields are 1-2 bytes per sub-block and sit in lines already touched.
template <int FMT> struct RFields { static constexpr int A = 16, B = 2; };
template <> struct RFields<QT_Q5_K> { static constexpr int A = 16, B = 4; };
template <> struct RFields<QT_Q6_K> { static constexpr int A = 16, B = 8; };
template <> struct RFields<QT_Q8_0> { static constexpr int A = 32, B = 2; };
template <> struct RFields<QT_IQ4_XS> { static constexpr int A = 16, B = 1; };
template <> struct RFields<QT_IQ2_S> { static constexpr int A = 4, B = 4; };
template <> struct RFields<QT_IQ2_XXS> { static constexpr int A = 4, B = 4; };
template <> struct RFields<QT_IQ2_XS> { static constexpr int A = 8, B = 1; };
template <> struct RFields<QT_IQ3_XXS> { static constexpr int A = 8, B = 4; };
template <> struct RFields<QT_IQ3_S> { static constexpr int A = 8, B = 4; };
template <> struct RFields<QT_Q2_0> { static constexpr int A = 8, B = 1; };

template <int FMT, bool RL>
__device__ __forceinline__ void tc_prefetch(const uint8_t * row, const ROff & o, int sb) {
    if constexpr (RL) {
        const uint8_t * a = row + o.a + (size_t) RFields<FMT>::A * sb;
        const uint8_t * b = row + o.b + (size_t) RFields<FMT>::B * sb;
        asm volatile("prefetch.global.L1 [%0];" ::"l"(a));
        asm volatile("prefetch.global.L1 [%0];" ::"l"(b));
    }
}

// GU: the block covers gate rows [R y, R y + R) (R = kTcRows) and the same up rows (K = E), and writes h =
// silu(gate) * up as fp16 to out[pair][ldo]; the activation of pair p is x[perm[p]]. Otherwise the down matrix's
// rows [2 R y, 2 R y + 2 R) (K = F), the activation of pair p is x[p], and out[pair][row] = the product. `off` is
// the up (GU) or down matrix's offset in the blob, rowb its row stride, nrows the matrix's rows (F or E), ncols its K.
//
// Software pipelined: the activation tile of step s+1 is loaded into registers before step s's products and stored
// into the other half of the double-buffered tile after them; the weights of step s+1 are unpacked after the
// products too, from lines prefetched into L1 a step earlier. One barrier per step.
template <int FMT, bool RL, bool GU>
__global__ void __launch_bounds__(kTcThreads, 1) moe_tc_k(const TcItem * __restrict__ items, const TcBlock * __restrict__ blocks,
                                                   size_t off, size_t rowb, int nrows, int ncols,
                                                   const half * __restrict__ x, int64_t ldx,
                                                   const int32_t * __restrict__ perm, half * __restrict__ out,
                                                   int64_t ldo, ROff ro, ExpScales es) {
    __shared__ __align__(16) half xs[2][kTcTile * kTcLd];
    const TcBlock bk = blocks[blockIdx.x];
    const TcItem it = items[bk.item];
    // NVFP4 global scales: gate and up before the silu product, down on its product (all stored as fp16)
    const float sg = (GU && es.gate) ? es.gate[it.expert] : 1.f, su = (GU && es.up) ? es.up[it.expert] : 1.f;
    const float sd = (!GU && es.down) ? es.down[it.expert] : 1.f;
    const int p0 = it.p0 + bk.t0;                   // the tile's first pair
    const int n = min(kTcTile, it.n - bk.t0);       // its tokens
    const int ngrp = (n + 7) >> 3;                  // token groups of 8
    if constexpr (RL) rt_tables_load<FMT>();        // (synchronized below, before the first unpack)
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int qp = (lane >> 2) & 3, j = (lane & 3) | ((lane >> 4) << 2);   // quad pair; row / token within it
    const uint8_t * rowp[2];
    if constexpr (GU) {
        const int r = min(blockIdx.y * kTcRows + warp * 8 + j, nrows - 1);
        rowp[0] = it.blob + (size_t) r * rowb;         // gate row
        rowp[1] = it.blob + off + (size_t) r * rowb;   // the same up row
    } else {
#pragma unroll
        for (int g = 0; g < 2; ++g)
            rowp[g] = it.blob + off + (size_t) min(blockIdx.y * 2 * kTcRows + g * kTcRows + warp * 8 + j, nrows - 1) * rowb;
    }
    const int nsb = ncols / 32, nsteps = (nsb + 3) / 4;
    // the activation tile of step s for k in [128 s, 128 s + 128): 64 tokens x 16 uint4, zeros past the tile's
    // tokens and past K; every load of a thread is issued before its first store
    auto fill_load = [&](int s, uint4 (&fill)[kTcFill]) {
#pragma unroll
        for (int f = 0; f < kTcFill; ++f) {
            const int i = threadIdx.x + f * kTcThreads, t = i >> 4, c16 = i & 15, k0 = s * 128 + c16 * 8;
            fill[f] = make_uint4(0u, 0u, 0u, 0u);
            if (t < n && k0 < ncols) {
                const int pr = p0 + t;
                const int64_t row = perm ? (int64_t) perm[pr] : (int64_t) pr;
                fill[f] = __ldg((const uint4 *) (x + row * ldx + k0));
            }
        }
    };
    auto fill_store = [&](int s, const uint4 (&fill)[kTcFill]) {
#pragma unroll
        for (int f = 0; f < kTcFill; ++f) {
            const int i = threadIdx.x + f * kTcThreads;
            *(uint4 *) (xs[s & 1] + (i >> 4) * kTcLd + (i & 15) * 8) = fill[f];
        }
    };
    // this thread's rows' sub-block of step s as A fragments (zeros past K)
    auto weights = [&](int s, uint32_t (&w)[2][16]) {
        const int sb = s * 4 + qp;
        if (sb < nsb) {
            tc_weights<FMT, RL>(rowp[0], ro, sb, w[0]);
            tc_weights<FMT, RL>(rowp[1], ro, sb, w[1]);
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) w[0][i] = w[1][i] = 0u;
        }
    };
    auto prefetch = [&](int s) {
        const int sb = s * 4 + qp;
        if (s < nsteps && sb < nsb) {
            tc_prefetch<FMT, RL>(rowp[0], ro, sb);
            tc_prefetch<FMT, RL>(rowp[1], ro, sb);
        }
    };
    float c[2][8][8];
#pragma unroll
    for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int i = 0; i < 8; ++i) c[g][t][i] = 0.f;
    uint32_t w[2][16];
    {
        uint4 fill[kTcFill];
        fill_load(0, fill);
        prefetch(0);
        prefetch(1);
        fill_store(0, fill);
        __syncthreads();   // the tile of step 0, and the tables
        weights(0, w);
    }
    for (int s = 0; s < nsteps; ++s) {
        const bool more = s + 1 < nsteps;
        uint4 fill[kTcFill];
        if (more) fill_load(s + 1, fill);
        const half * xt = xs[s & 1];
        // two token groups at a time: 4 independent accumulator chains per mma step
#pragma unroll
        for (int g8 = 0; g8 < 8; g8 += 2) {
            if (g8 < ngrp) {   // block-uniform
                uint32_t xa[16], xb[16];
                const uint4 * pa = (const uint4 *) (xt + (g8 * 8 + j) * kTcLd + qp * 32);
                const uint4 * pb = (const uint4 *) (xt + (g8 * 8 + 8 + j) * kTcLd + qp * 32);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ((uint4 *) xa)[i] = pa[i];
                    ((uint4 *) xb)[i] = pb[i];
                }
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    mma884(c[0][g8], w[0][2 * kk], w[0][2 * kk + 1], xa[2 * kk], xa[2 * kk + 1]);
                    mma884(c[1][g8], w[1][2 * kk], w[1][2 * kk + 1], xa[2 * kk], xa[2 * kk + 1]);
                    mma884(c[0][g8 + 1], w[0][2 * kk], w[0][2 * kk + 1], xb[2 * kk], xb[2 * kk + 1]);
                    mma884(c[1][g8 + 1], w[1][2 * kk], w[1][2 * kk + 1], xb[2 * kk], xb[2 * kk + 1]);
                }
            }
        }
        if (more) {
            fill_store(s + 1, fill);   // the other buffer: nobody reads it until the barrier below
            prefetch(s + 2);
            weights(s + 1, w);
            __syncthreads();
        }
    }
    // the 4 quad pairs hold partial sums of the same tiles in the same registers: add them (fixed order)
#pragma unroll
    for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                c[g][t][i] += __shfl_xor_sync(0xffffffff, c[g][t][i], 4);
                c[g][t][i] += __shfl_xor_sync(0xffffffff, c[g][t][i], 8);
            }
    if (qp != 0) return;
#pragma unroll
    for (int g8 = 0; g8 < 8; ++g8) {
        if (g8 >= ngrp) break;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int rr = (lane & 1) + (i & 2) + ((lane >> 4) << 2), t = (i & 4) + (lane & 2) + (i & 1);
            const int tt = g8 * 8 + t;
            if (tt >= n) continue;
            half * o = out + (int64_t) (p0 + tt) * ldo;
            if constexpr (GU) {
                const int row = blockIdx.y * kTcRows + warp * 8 + rr;
                if (row < nrows) {
                    const float gv = c[0][g8][i] * sg, uv = c[1][g8][i] * su;
                    o[row] = __float2half(gv / (1.f + __expf(-gv)) * uv);
                }
            } else {
#pragma unroll
                for (int g = 0; g < 2; ++g) {
                    const int row = blockIdx.y * 2 * kTcRows + g * kTcRows + warp * 8 + rr;
                    if (row < nrows) o[row] = __float2half(c[g][g8][i] * sd);
                }
            }
        }
    }
}

void moe_tc_gu(const TcItem * items, const TcBlock * blocks, int nblocks, const MoeLayerDesc & d, const half * x,
               int64_t ldx, const int32_t * perm, half * h, int E, int F, cudaStream_t st) {
    if (nblocks <= 0) return;
    if (E % kTcK != 0 || F % kTcRows != 0) throw std::runtime_error("moe_tc: unsupported expert shape");
    const ROff go{d.go[0], d.go[1], d.go[2], d.go[3], d.go[4]};
    const dim3 grid(nblocks, F / kTcRows);
    if (d.rlay) {
        auto f = [&]<int FMT>() {
            moe_tc_k<FMT, true, true><<<grid, kTcThreads, 0, st>>>(items, blocks, d.gate_bytes, d.grow, F, E, x, ldx, perm, h, F, go, d.scales);
        };
        BNK_DISPATCH_R(d.gate_type, f);
    } else {
        auto f = [&]<int FMT>() {
            moe_tc_k<FMT, false, true><<<grid, kTcThreads, 0, st>>>(items, blocks, d.gate_bytes, d.grow, F, E, x, ldx, perm, h, F, go, d.scales);
        };
        BNK_DISPATCH_DP4A(d.gate_type, f);
    }
    check_launch("moe_tc_gu");
}

void moe_tc_down(const TcItem * items, const TcBlock * blocks, int nblocks, const MoeLayerDesc & d, const half * h,
                 half * D, int E, int F, cudaStream_t st) {
    if (nblocks <= 0) return;
    if (F % 32 != 0 || E % (2 * kTcRows) != 0) throw std::runtime_error("moe_tc: unsupported expert shape");
    const ROff dof{d.dof[0], d.dof[1], d.dof[2], d.dof[3], d.dof[4]};
    const size_t off = d.gate_bytes + d.up_bytes;
    const dim3 grid(nblocks, E / (2 * kTcRows));
    if (d.rlay) {
        auto f = [&]<int FMT>() {
            moe_tc_k<FMT, true, false><<<grid, kTcThreads, 0, st>>>(items, blocks, off, d.drow, E, F, h, F, nullptr, D, E, dof, d.scales);
        };
        BNK_DISPATCH_R(d.down_type, f);
    } else {
        auto f = [&]<int FMT>() {
            moe_tc_k<FMT, false, false><<<grid, kTcThreads, 0, st>>>(items, blocks, off, d.drow, E, F, h, F, nullptr, D, E, dof, d.scales);
        };
        BNK_DISPATCH_DP4A(d.down_type, f);
    }
    check_launch("moe_tc_down");
}

}  // namespace bnk
