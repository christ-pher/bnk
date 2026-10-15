// mma.m8n8k4 (sm_70) helpers shared by the tensor-core GEMV (gemv_r.cu) and the prompt path's expert GEMM
// (moe_tc.cu): int8 -> fp16 conversion, the warp-level product, and a row's sub-block as fp16 A fragments.
#pragma once

#include <cuda_fp16.h>

#include "kernels/quant.cuh"
#include "kernels/rfmt.cuh"

namespace bnk {

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

// the weights of an unpacked sub-block as 16 fp16 pairs (d * q - m), for the A fragments. MIN: the unpacker's
// HAS_MIN (R-layout and ggml-layout unpackers of the same format may differ, e.g. Q6_K)
template <bool MIN>
__device__ __forceinline__ void mma_weights_u(const Unpacked & u, uint32_t (&w)[16]) {
    const half2 d0 = __float2half2_rn(u.d0), d1 = __float2half2_rn(u.d1);
    half2 m0 = __float2half2_rn(0.f), m1 = m0;
    if constexpr (MIN) {
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

// a row's sub-block sb (R layout, or plain fp16 rows for QT_F16) as A fragments
template <int FMT>
__device__ __forceinline__ void mma_weights(const uint8_t * row, const ROff & o, int sb, uint32_t (&w)[16]) {
    if constexpr (FMT == QT_F16) {
        const uint4 * wp = (const uint4 *) (row + (size_t) sb * 64);
#pragma unroll
        for (int i = 0; i < 4; ++i) ((uint4 *) w)[i] = __ldg(wp + i);
    } else {
        Unpacked u;
        RT<FMT>::unpack(row, o, sb, u);
        mma_weights_u<RT<FMT>::HAS_MIN>(u, w);
    }
}


}  // namespace bnk
