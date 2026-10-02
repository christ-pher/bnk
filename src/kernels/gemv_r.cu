// GEMV v2: R-layout quant rows (dp4a) and float rows, LPR lanes per row, optional split-K.
#include <algorithm>
#include <cstdio>

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
__global__ void __launch_bounds__(256) gemv_r_k(const uint8_t * __restrict__ W, size_t row_bytes, ROff o, int rows,
                                                int nsb, const int8_t * __restrict__ aq, const float * __restrict__ ad,
                                                int64_t cols_pad, int T, float * __restrict__ y, int64_t ldy,
                                                int accumulate) {
    extern __shared__ __align__(16) uint8_t smem[];
    constexpr int RPW = 32 / LPR;  // rows per warp
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = lane % LPR, rsel = lane / LPR;
    const int64_t nb = cols_pad / 32;

    const int8_t * aq_s = aq;
    const float * ad_s = ad;
    if constexpr (SMEM) {
        int8_t * q = (int8_t *) smem;
        float * d = (float *) (smem + (size_t) NT * cols_pad);
        const int nq16 = (int) (T * cols_pad / 16);
        for (int i = threadIdx.x; i < nq16; i += 256) ((int4 *) q)[i] = ((const int4 *) aq)[i];
        for (int i = threadIdx.x; i < T * nb; i += 256) d[i] = ad[i];
        __syncthreads();
        aq_s = q;
        ad_s = d;
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
                            int s0 = __dp4a(a0.x, 0x01010101, 0);
                            s0 = __dp4a(a0.y, 0x01010101, s0);
                            s0 = __dp4a(a0.z, 0x01010101, s0);
                            s0 = __dp4a(a0.w, 0x01010101, s0);
                            int s1 = __dp4a(a1.x, 0x01010101, 0);
                            s1 = __dp4a(a1.y, 0x01010101, s1);
                            s1 = __dp4a(a1.z, 0x01010101, s1);
                            s1 = __dp4a(a1.w, 0x01010101, s1);
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
    const size_t smem = (size_t) NT * a.cols_pad + (size_t) NT * (a.cols_pad / 32) * 4;
    const bool use_smem = smem <= 48 * 1024;
    int grid;
    if (SPLITK) grid = ngroups;
    else grid = std::min((ngroups + 7) / 8, sm_count2() * (use_smem ? 4 : 8));
    const ROff o{W.r_off[0], W.r_off[1], W.r_off[2], W.r_off[3], W.r_off[4]};
    const int accf = acc ? 1 : 0;
    if (use_smem)
        gemv_r_k<FMT, NT, LPR, SPLITK, true><<<grid, 256, smem, s>>>((const uint8_t *) W.data, W.row_bytes, o,
                                                                    (int) W.rows, nsb, a.q, a.d, a.cols_pad, T, y,
                                                                    ldy, accf);
    else
        gemv_r_k<FMT, NT, LPR, SPLITK, false><<<grid, 256, 0, s>>>((const uint8_t *) W.data, W.row_bytes, o,
                                                                  (int) W.rows, nsb, a.q, a.d, a.cols_pad, T, y, ldy,
                                                                  accf);
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

void gemv_r(const QMat & W, const ActQ8 & a, int T, float * y, int64_t ldy, bool accumulate, cudaStream_t s) {
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
    if (T == 1) launch_f0<1>(W, x, ldx, T, y, ldy, accumulate, s);
    else if (T == 2) launch_f0<2>(W, x, ldx, T, y, ldy, accumulate, s);
    else if (T <= 4) launch_f0<4>(W, x, ldx, T, y, ldy, accumulate, s);
    else launch_f0<8>(W, x, ldx, T, y, ldy, accumulate, s);
}

}  // namespace bnk
