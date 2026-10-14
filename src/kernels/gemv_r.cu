// GEMV v2: R-layout quant rows (dp4a) and float rows, LPR lanes per row, optional split-K.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mma.h>

#include "core/util.h"
#include "kernels/gemv.h"
#include "kernels/qdot.cuh"
#include "kernels/rfmt.cuh"

namespace bnk {

static int sm_count2() {
    static int n = 0;
    if (!n) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    }
    return n;
}

// ------------------------------------------------------------------------------ quant (R layout)
// Block: 8 warps. Without split-K each warp owns 32/LPR rows (LPR lanes per row) and blocks stride over
// row groups. With split-K (SK = 8) all 8 warps share the block's row group and split its sub-blocks.
template <int FMT, int NT, int LPR, bool SPLITK, bool SMEM>
__global__ void __launch_bounds__(256, 4) gemv_r_k(const uint8_t * __restrict__ W, size_t row_bytes, ROff o, int rows,
                                                int nsb, const int8_t * __restrict__ aq, const float * __restrict__ ad,
                                                const int16_t * __restrict__ asum, int64_t cols_pad, int T,
                                                float * __restrict__ y, int64_t ldy, int accumulate) {
    extern __shared__ __align__(16) uint8_t smem[];
    constexpr int RPW = 32 / LPR;  // rows per warp
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane % LPR, rsel = lane / LPR;
    const int64_t nb = cols_pad / 32;

    const int8_t * aq_s = aq;
    const float * ad_s = ad;
    const int16_t * as_s = asum;
    if constexpr (SMEM) {
        int8_t * q = (int8_t *) smem;
        float * d = (float *) (smem + (size_t) NT * cols_pad);
        int * sm2 = (int *) (d + NT * nb);
        const int nq16 = (int) (T * cols_pad / 16);
        for (int i = threadIdx.x; i < nq16; i += 256) ((int4 *) q)[i] = ((const int4 *) aq)[i];
        for (int i = threadIdx.x; i < T * nb; i += 256) d[i] = ad[i];
        if (RT<FMT>::HAS_MIN && asum)
            for (int i = threadIdx.x; i < T * nb; i += 256) sm2[i] = ((const int *) asum)[i];
        __syncthreads();
        aq_s = q;
        ad_s = d;
        if (asum) as_s = (const int16_t *) sm2;
    }
    __shared__ float red[SPLITK ? 8 : 1][32][NT];

    const int groups_per_block = SPLITK ? 1 : 8;  // row groups (of RPW rows) per block per step
    const int ngroups = (rows + RPW - 1) / RPW;
    for (int g0 = blockIdx.x * groups_per_block; g0 < ngroups; g0 += gridDim.x * groups_per_block) {
        const int grp = SPLITK ? g0 : g0 + warp;
        const int r = grp * RPW + rsel;
        const bool rok = grp < ngroups && r < rows;
        float acc[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] = 0.f;
        if (rok) {
            const uint8_t * row = W + (size_t) r * row_bytes;
            int sb0 = sub, sb1 = nsb, step = LPR;
            if constexpr (SPLITK) {
                const int per = (nsb + 7) / 8;
                sb0 = warp * per + sub;
                sb1 = min(nsb, (warp + 1) * per);
            }
#pragma unroll 2
            for (int sb = sb0; sb < sb1; sb += step) {
                Unpacked u;
                RT<FMT>::unpack(row, o, sb, u);
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if (t < T) {
                        const int4 * ap = (const int4 *) (aq_s + t * cols_pad + sb * 32);
                        const int4 a0 = ap[0], a1 = ap[1];
                        int i0 = __dp4a(u.w[0], a0.x, 0);
                        i0 = __dp4a(u.w[1], a0.y, i0);
                        i0 = __dp4a(u.w[2], a0.z, i0);
                        i0 = __dp4a(u.w[3], a0.w, i0);
                        int i1 = __dp4a(u.w[4], a1.x, 0);
                        i1 = __dp4a(u.w[5], a1.y, i1);
                        i1 = __dp4a(u.w[6], a1.z, i1);
                        i1 = __dp4a(u.w[7], a1.w, i1);
                        float v = u.d0 * (float) i0 + u.d1 * (float) i1;
                        if constexpr (RT<FMT>::HAS_MIN) {
                            int s0, s1;
                            if (as_s) {
                                const int pk = ((const int *) as_s)[t * nb + sb];
                                s0 = (int) (int16_t) (pk & 0xffff);
                                s1 = pk >> 16;
                            } else {
                                s0 = __dp4a(a0.x, 0x01010101, 0);
                                s0 = __dp4a(a0.y, 0x01010101, s0);
                                s0 = __dp4a(a0.z, 0x01010101, s0);
                                s0 = __dp4a(a0.w, 0x01010101, s0);
                                s1 = __dp4a(a1.x, 0x01010101, 0);
                                s1 = __dp4a(a1.y, 0x01010101, s1);
                                s1 = __dp4a(a1.z, 0x01010101, s1);
                                s1 = __dp4a(a1.w, 0x01010101, s1);
                            }
                            v -= u.m0 * (float) s0 + u.m1 * (float) s1;
                        }
                        acc[t] += ad_s[t * nb + sb] * v;
                    }
                }
            }
        }
        // reduce over the LPR lanes of a row
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int off = LPR / 2; off > 0; off >>= 1) acc[t] += __shfl_xor_sync(0xffffffff, acc[t], off);
        }
        if constexpr (SPLITK) {
            if (sub == 0) {
#pragma unroll
                for (int t = 0; t < NT; ++t) red[warp][rsel][t] = acc[t];
            }
            __syncthreads();
            if (warp == 0 && sub == 0 && rok) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if (t < T) {
                        float s = 0.f;
#pragma unroll
                        for (int w = 0; w < 8; ++w) s += red[w][rsel][t];
                        float * yp = y + t * ldy + r;
                        *yp = accumulate ? *yp + s : s;
                    }
                }
            }
            __syncthreads();
        } else if (sub == 0 && rok) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < T) {
                    float * yp = y + t * ldy + r;
                    *yp = accumulate ? *yp + acc[t] : acc[t];
                }
            }
        }
    }
}

template <int FMT, int NT, int LPR, bool SPLITK>
static void launch_r2(const QMat & W, const ActQ8 & a, int T, float * y, int64_t ldy, bool acc, cudaStream_t s) {
    const int nsb = (int) (W.cols / 32);
    constexpr int RPW = 32 / LPR;
    const int ngroups = (int) ((W.rows + RPW - 1) / RPW);
    const size_t smem = (size_t) NT * a.cols_pad + (size_t) NT * (a.cols_pad / 32) * 8;
    // static shared memory (the split-K reduction) plus this must stay under the 48 KiB default
    const size_t static_smem = SPLITK ? (size_t) 8 * 32 * NT * 4 : 0;
    static const bool no_smem = getenv("BNK_GEMV_NOSMEM") != nullptr;
    const bool use_smem = !no_smem && NT <= 4 && smem + static_smem <= 46 * 1024;
    int grid;
    if (SPLITK) grid = ngroups;
    else grid = std::min((ngroups + 7) / 8, sm_count2() * (use_smem ? 4 : 8));
    const ROff o{W.r_off[0], W.r_off[1], W.r_off[2], W.r_off[3], W.r_off[4]};
    const int accf = acc ? 1 : 0;
    if (use_smem)
        gemv_r_k<FMT, NT, LPR, SPLITK, true><<<grid, 256, smem, s>>>((const uint8_t *) W.data, W.row_bytes, o,
                                                                    (int) W.rows, nsb, a.q, a.d, a.s, a.cols_pad, T, y,
                                                                    ldy, accf);
    else
        gemv_r_k<FMT, NT, LPR, SPLITK, false><<<grid, 256, 0, s>>>((const uint8_t *) W.data, W.row_bytes, o,
                                                                  (int) W.rows, nsb, a.q, a.d, a.s, a.cols_pad, T, y, ldy,
                                                                  accf);
    check_launch("gemv_r");
}

template <int NT>
static void launch_r1(const QMat & W, const ActQ8 & a, int T, float * y, int64_t ldy, bool acc, cudaStream_t s) {
    const int nsb = (int) (W.cols / 32);
    // few rows and a long K: split K across the 8 warps of a block
    const bool splitk = W.rows * 8 < (int64_t) sm_count2() * 64 && nsb >= 64;
    auto f = [&]<int FMT>() {
        if (splitk) launch_r2<FMT, NT, 32, true>(W, a, T, y, ldy, acc, s);
        else if (nsb >= 24) launch_r2<FMT, NT, 32, false>(W, a, T, y, ldy, acc, s);
        else if (nsb >= 12) launch_r2<FMT, NT, 16, false>(W, a, T, y, ldy, acc, s);
        else if (nsb >= 6) launch_r2<FMT, NT, 8, false>(W, a, T, y, ldy, acc, s);
        else launch_r2<FMT, NT, 4, false>(W, a, T, y, ldy, acc, s);
    };
    BNK_DISPATCH_R(W.type, f);
}

// ------------------------------------------------------------------------------ quant (R layout), tensor cores
// Several tokens: the dp4a kernel above repeats the dot product (and its scale arithmetic) per token, so at 8 tokens
// it is bound by instruction issue (~25% of HBM bandwidth). Here a thread unpacks a row's sub-block once (the same
// RT<FMT>::unpack), turns it into fp16 weights (d * q - m) in registers and feeds them straight to the tensor cores:
// mma.m8n8k4 (sm_70) takes, per thread of a quad pair, 4 consecutive weights of one of 8 rows (A) and 4 activations
// of one of 8 tokens (B), fp32 accumulators. A warp owns 8 rows; its 4 quad pairs take 4 consecutive sub-blocks
// (the loads of a row stay contiguous) and add up at the end. The warps of a block split K and add up through
// shared memory, in a fixed order (deterministic, and a token's result does not depend on the other tokens).

// 4 int8 (bytes of w) -> 2 half2, exact: bytes ^ 0x80 under the fp16 exponent of 1024 give 1024 + 128 + q
__device__ __forceinline__ void i8x4_to_h2(uint32_t w, half2 & lo, half2 & hi) {
    const uint32_t u = w ^ 0x80808080u;
    const uint32_t a = __byte_perm(u, 0x64646464u, 0x5140), b = __byte_perm(u, 0x64646464u, 0x7362);
    const half2 off = __float2half2_rn(1152.f);
    lo = __hsub2(*(const half2 *) &a, off);
    hi = __hsub2(*(const half2 *) &b, off);
}

__device__ __forceinline__ uint32_t h2u(half2 h) { return *(const uint32_t *) &h; }
// fp16 saturates rather than overflow to inf (an outlier activation stays finite)
__device__ __forceinline__ float sat_h(float v) { return fminf(fmaxf(v, -65504.f), 65504.f); }

// c (8x8, f32) += A (8x4, this thread's row: a0 a1) x B (4x8, this thread's column: b0 b1), per quad pair
__device__ __forceinline__ void mma884(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// FMT: an R-layout quant format, or QT_F16 (plain fp16 rows: no unpacking). XF: the activations as fp32 (x, ldx),
// converted on the way; else fp16 (the fp16 copy quantize_act writes). KS warps split a row group's K (1, 2, 4 or
// 8); a block holds 8 / KS row groups of 8 rows.
// a row's sub-block sb as 16 fp16 pairs (d * q - m), for the A fragments
template <int FMT>
__device__ __forceinline__ void mma_weights(const uint8_t * row, const ROff & o, int sb, uint32_t (&w)[16]) {
    if constexpr (FMT == QT_F16) {
        const uint4 * wp = (const uint4 *) (row + (size_t) sb * 64);
#pragma unroll
        for (int i = 0; i < 4; ++i) ((uint4 *) w)[i] = __ldg(wp + i);
    } else {
        Unpacked u;
        RT<FMT>::unpack(row, o, sb, u);
        const half2 d0 = __float2half2_rn(u.d0), d1 = __float2half2_rn(u.d1);
        half2 m0 = __float2half2_rn(0.f), m1 = m0;
        if constexpr (RT<FMT>::HAS_MIN) {
            m0 = __float2half2_rn(-u.m0);
            m1 = __float2half2_rn(-u.m1);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            half2 lo, hi;
            i8x4_to_h2((uint32_t) u.w[i], lo, hi);
            w[2 * i] = h2u(__hfma2(lo, i < 4 ? d0 : d1, i < 4 ? m0 : m1));
            w[2 * i + 1] = h2u(__hfma2(hi, i < 4 ? d0 : d1, i < 4 ? m0 : m1));
        }
    }
}

// FMT: an R-layout quant format, or QT_F16 (plain fp16 rows: no unpacking). XF: the activations as fp32 (x, ldx),
// converted on the way; else fp16 (the fp16 copy quantize_act writes). RG row groups of 8 rows per warp share each
// activation load (RG = 2: half the activation traffic, which was ~2.5x the weight bytes). KS warps split a row
// block's K (1, 2, 4 or 8); a block holds 8 / KS row blocks of 8 * RG rows.
template <int FMT, bool XF, int RG>
__global__ void __launch_bounds__(256) gemv_mma_k(const uint8_t * __restrict__ W, size_t row_bytes, ROff o, int rows,
                                                  int nsb, const void * __restrict__ xv_, int64_t ldx, int T,
                                                  float * __restrict__ y, int64_t ldy, int accumulate, int ks) {
    __shared__ float part[8][64 * RG];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int grp = blockIdx.x * (8 / ks) + warp / ks, kp = warp % ks;
    const int qp = (lane >> 2) & 3, j = (lane & 3) | ((lane >> 4) << 2);   // quad pair; row / token within it
    const uint8_t * rowp[RG];
#pragma unroll
    for (int g = 0; g < RG; ++g) rowp[g] = W + (size_t) min(grp * 8 * RG + g * 8 + j, rows - 1) * row_bytes;
    const int per = (nsb + ks - 1) / ks, sb0 = kp * per, sb1 = min(nsb, sb0 + per);
    float c[RG][8];
#pragma unroll
    for (int g = 0; g < RG; ++g)
#pragma unroll
        for (int i = 0; i < 8; ++i) c[g][i] = 0.f;
    for (int s = sb0; s < sb1; s += 4) {
        const int sb = s + qp;
        const bool ok = sb < sb1;
        uint32_t xw[16];   // token j's 32 activations of this sub-block, fp16
        if (!ok || j >= T) {   // past K, or a token past the window (its column is not stored): zeros, no loads
#pragma unroll
            for (int i = 0; i < 16; ++i) xw[i] = 0;
        } else if constexpr (XF) {
            const float4 * xp = (const float4 *) ((const float *) xv_ + (int64_t) j * ldx + (int64_t) sb * 32);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float4 f = __ldg(xp + i);
                xw[2 * i] = h2u(__floats2half2_rn(sat_h(f.x), sat_h(f.y)));
                xw[2 * i + 1] = h2u(__floats2half2_rn(sat_h(f.z), sat_h(f.w)));
            }
        } else {
            const uint4 * xp = (const uint4 *) ((const half *) xv_ + (int64_t) j * ldx + (int64_t) sb * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) ((uint4 *) xw)[i] = __ldg(xp + i);
        }
#pragma unroll
        for (int g = 0; g < RG; ++g) {
            uint32_t w[16];
            if (ok) mma_weights<FMT>(rowp[g], o, sb, w);
            else
#pragma unroll
                for (int i = 0; i < 16; ++i) w[i] = 0;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) mma884(c[g], w[2 * kk], w[2 * kk + 1], xw[2 * kk], xw[2 * kk + 1]);
        }
    }
    // the 4 quad pairs hold partial sums of the same tiles in the same registers
#pragma unroll
    for (int g = 0; g < RG; ++g)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c[g][i] += __shfl_xor_sync(0xffffffff, c[g][i], 4);
            c[g][i] += __shfl_xor_sync(0xffffffff, c[g][i], 8);
        }
    if (qp == 0) {
#pragma unroll
        for (int g = 0; g < RG; ++g)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int rr = (lane & 1) + (i & 2) + ((lane >> 4) << 2), t = (i & 4) + (lane & 2) + (i & 1);
                part[warp][t * 8 * RG + g * 8 + rr] = c[g][i];
            }
    }
    __syncthreads();
    // each row block's K parts added in a fixed order (deterministic)
    constexpr int E = 64 * RG;   // (token, row) pairs per row block
    for (int idx = threadIdx.x; idx < (8 / ks) * E; idx += 256) {
        const int gg = idx / E, e = idx % E;
        const int t = e / (8 * RG), rr = (blockIdx.x * (8 / ks) + gg) * 8 * RG + e % (8 * RG);
        if (t < T && rr < rows) {
            float v = 0.f;
            for (int w = 0; w < ks; ++w) v += part[gg * ks + w][e];
            float * yp = y + (int64_t) t * ldy + rr;
            *yp = accumulate ? *yp + v : v;
        }
    }
}

// K-splitting warps per row group: ~8 sub-blocks or more each, up to 8
static int mma_ks(int nsb) {
    int ks = 8;
    while (ks > 1 && nsb < ks * 8) ks >>= 1;
    return ks;
}

template <int FMT, bool XF>
static void launch_mma_k(const QMat & W, const void * x, int64_t ldx, int T, float * y, int64_t ldy, bool acc,
                         cudaStream_t s) {
    const int nsb = (int) (W.cols / 32), ks = mma_ks(nsb);
    // BNK_GEMV_MMA_RG=2: 16 rows per warp (half the activation loads; measured no faster on a V100, so off)
    static const int rg_env = getenv("BNK_GEMV_MMA_RG") ? atoi(getenv("BNK_GEMV_MMA_RG")) : 1;
    const int rg = rg_env == 2 && W.rows >= 2048 ? 2 : 1;
    const int groups = (int) ((W.rows + 8 * rg - 1) / (8 * rg)), per_block = 8 / ks;
    const ROff o{W.r_off[0], W.r_off[1], W.r_off[2], W.r_off[3], W.r_off[4]};
    auto k = rg == 2 ? gemv_mma_k<FMT, XF, 2> : gemv_mma_k<FMT, XF, 1>;
    k<<<(groups + per_block - 1) / per_block, 256, 0, s>>>((const uint8_t *) W.data, W.row_bytes, o, (int) W.rows, nsb,
                                                           x, ldx, T, y, ldy, acc ? 1 : 0, ks);
    check_launch("gemv_mma");
}

static void launch_mma(const QMat & W, const ActQ8 & a, int T, float * y, int64_t ldy, bool acc, cudaStream_t s) {
    auto f = [&]<int FMT>() { launch_mma_k<FMT, false>(W, a.h, a.cols_pad, T, y, ldy, acc, s); };
    BNK_DISPATCH_R(W.type, f);
}

// from this many tokens on, the tensor-core kernel (when the activations have their fp16 copy); BNK_GEMV_MMA_MIN
static int mma_min_tokens() {
    // BNK_EXACT=1: every window on the tensor cores, so a token's result never depends on how many rows share its
    // window (batched / two-lane decoding then equal decoding alone, token for token)
    static const int n = getenv("BNK_GEMV_MMA_MIN") ? atoi(getenv("BNK_GEMV_MMA_MIN"))
                         : getenv("BNK_EXACT") && atoi(getenv("BNK_EXACT")) ? 1 : 4;
    return n;
}

void gemv_r(const QMat & W, const ActQ8 & a, int T, float * y, int64_t ldy, bool accumulate, cudaStream_t s) {
    if (a.h && T >= mma_min_tokens() && mma_min_tokens() > 0) return launch_mma(W, a, T, y, ldy, accumulate, s);
    if (T == 1) launch_r1<1>(W, a, T, y, ldy, accumulate, s);
    else if (T == 2) launch_r1<2>(W, a, T, y, ldy, accumulate, s);
    else if (T <= 4) launch_r1<4>(W, a, T, y, ldy, accumulate, s);
    else launch_r1<8>(W, a, T, y, ldy, accumulate, s);
}

// ------------------------------------------------------------------------------ float rows
template <int FMT>
__device__ __forceinline__ void loadf8(const uint8_t * row, int64_t i, float (&w)[8]) {
    if constexpr (FMT == QT_F32) {
        const float4 * p = (const float4 *) (row + i * 4);
        const float4 a = __ldg(p), b = __ldg(p + 1);
        w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
    } else if constexpr (FMT == QT_F16) {
        const uint4 v = __ldg((const uint4 *) (row + i * 2));
        const __half2 * h = (const __half2 *) &v;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __half22float2(h[k]);
            w[2 * k] = f.x;
            w[2 * k + 1] = f.y;
        }
    } else {
        const uint4 v = __ldg((const uint4 *) (row + i * 2));
        const uint32_t * u = (const uint32_t *) &v;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            w[2 * k] = __uint_as_float(u[k] << 16);
            w[2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u);
        }
    }
}

// chunks of 8 elements; LPR lanes per row; optional split-K over the block's 8 warps
template <int FMT, int NT, int LPR, bool SPLITK>
__global__ void __launch_bounds__(256) gemv_f_k(const uint8_t * __restrict__ W, size_t row_bytes, int rows,
                                                int nch, const float * __restrict__ x, int64_t ldx, int T,
                                                float * __restrict__ y, int64_t ldy, int accumulate) {
    constexpr int RPW = 32 / LPR;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane % LPR, rsel = lane / LPR;
    __shared__ float red[SPLITK ? 8 : 1][32][NT];
    const int groups_per_block = SPLITK ? 1 : 8;
    const int ngroups = (rows + RPW - 1) / RPW;
    for (int g0 = blockIdx.x * groups_per_block; g0 < ngroups; g0 += gridDim.x * groups_per_block) {
        const int grp = SPLITK ? g0 : g0 + warp;
        const int r = grp * RPW + rsel;
        const bool rok = grp < ngroups && r < rows;
        float acc[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] = 0.f;
        if (rok) {
            const uint8_t * row = W + (size_t) r * row_bytes;
            int c0 = sub, c1 = nch;
            if constexpr (SPLITK) {
                const int per = (nch + 7) / 8;
                c0 = warp * per + sub;
                c1 = min(nch, (warp + 1) * per);
            }
#pragma unroll 4
            for (int c = c0; c < c1; c += LPR) {
                float w[8];
                loadf8<FMT>(row, (int64_t) c * 8, w);
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if (t < T) {
                        const float4 * xp = (const float4 *) (x + t * ldx + (int64_t) c * 8);
                        const float4 a = __ldg(xp), b = __ldg(xp + 1);
                        acc[t] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y +
                                  w[6] * b.z + w[7] * b.w;
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int off = LPR / 2; off > 0; off >>= 1) acc[t] += __shfl_xor_sync(0xffffffff, acc[t], off);
        }
        if constexpr (SPLITK) {
            if (sub == 0) {
#pragma unroll
                for (int t = 0; t < NT; ++t) red[warp][rsel][t] = acc[t];
            }
            __syncthreads();
            if (warp == 0 && sub == 0 && rok) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if (t < T) {
                        float s = 0.f;
#pragma unroll
                        for (int w = 0; w < 8; ++w) s += red[w][rsel][t];
                        float * yp = y + t * ldy + r;
                        *yp = accumulate ? *yp + s : s;
                    }
                }
            }
            __syncthreads();
        } else if (sub == 0 && rok) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < T) {
                    float * yp = y + t * ldy + r;
                    *yp = accumulate ? *yp + acc[t] : acc[t];
                }
            }
        }
    }
}

template <int FMT, int NT, int LPR, bool SPLITK>
static void launch_f2(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy, bool acc,
                      cudaStream_t s) {
    const int nch = (int) (W.cols / 8);
    constexpr int RPW = 32 / LPR;
    const int ngroups = (int) ((W.rows + RPW - 1) / RPW);
    const int grid = SPLITK ? ngroups : std::min((ngroups + 7) / 8, sm_count2() * 8);
    gemv_f_k<FMT, NT, LPR, SPLITK><<<grid, 256, 0, s>>>((const uint8_t *) W.data, W.row_bytes, (int) W.rows, nch, x,
                                                        ldx, T, y, ldy, acc ? 1 : 0);
    check_launch("gemv_f");
}

template <int FMT, int NT>
static void launch_f1(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy, bool acc,
                      cudaStream_t s) {
    const int nch = (int) (W.cols / 8);
    const bool splitk = W.rows * 8 < (int64_t) sm_count2() * 64 && nch >= 256;
    if (splitk) launch_f2<FMT, NT, 32, true>(W, x, ldx, T, y, ldy, acc, s);
    else if (nch >= 96 || nch % 32 == 0) launch_f2<FMT, NT, 32, false>(W, x, ldx, T, y, ldy, acc, s);
    else if (nch % 8 == 0 || nch >= 48) launch_f2<FMT, NT, 8, false>(W, x, ldx, T, y, ldy, acc, s);
    else launch_f2<FMT, NT, 4, false>(W, x, ldx, T, y, ldy, acc, s);
}

template <int NT>
static void launch_f0(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy, bool acc,
                      cudaStream_t s) {
    switch (W.type) {
        case QT_F32: launch_f1<QT_F32, NT>(W, x, ldx, T, y, ldy, acc, s); break;
        case QT_F16: launch_f1<QT_F16, NT>(W, x, ldx, T, y, ldy, acc, s); break;
        case QT_BF16: launch_f1<QT_BF16, NT>(W, x, ldx, T, y, ldy, acc, s); break;
        default: bnk_unsupported_format(W.type);
    }
}

void gemv_float2(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy, bool accumulate,
                 cudaStream_t s) {
    // fp16 rows, several tokens: the tensor cores (x read as fp32 and converted)
    // (not tiny matrices: a few row groups cannot fill the GPU)
    if (W.type == QT_F16 && W.rows >= 256 && W.cols % 32 == 0 && ldx % 4 == 0 && T >= mma_min_tokens() &&
        mma_min_tokens() > 0)
        return launch_mma_k<QT_F16, true>(W, x, ldx, T, y, ldy, accumulate, s);
    if (T == 1) launch_f0<1>(W, x, ldx, T, y, ldy, accumulate, s);
    else if (T == 2) launch_f0<2>(W, x, ldx, T, y, ldy, accumulate, s);
    else if (T <= 4) launch_f0<4>(W, x, ldx, T, y, ldy, accumulate, s);
    else launch_f0<8>(W, x, ldx, T, y, ldy, accumulate, s);
}

}  // namespace bnk
