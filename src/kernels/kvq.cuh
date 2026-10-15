// 8-bit K/V cache cells (kv_q8()): each (cell, KV head) row holds its D values as int8, then D / 32 fp16 scales
// (q8_0's blocks of 32: value = q * scale, scale = max |x| / 127 over the block). The cache stays a half array
// with kv_row_halves(D) halves per row; the kernels read 8 consecutive values at a time as 8 halves.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace bnk {

// 8 int8 values times a scale as 8 halves: q + 128 is the low mantissa byte of 1024 + (q + 128) (high byte 0x64),
// minus 1152 that is q exactly; one rounding, in the multiply
__device__ __forceinline__ uint4 kv_dequant8(uint2 qv, half2 s) {
    const uint32_t a = qv.x ^ 0x80808080u, b = qv.y ^ 0x80808080u;
    const half2 bias = __half2half2(__ushort_as_half((unsigned short) 0x6480));   // 1152
    uint4 o;
    uint32_t * w = (uint32_t *) &o;
    w[0] = __byte_perm(a, 0x64646464u, 0x4140);
    w[1] = __byte_perm(a, 0x64646464u, 0x4342);
    w[2] = __byte_perm(b, 0x64646464u, 0x4140);
    w[3] = __byte_perm(b, 0x64646464u, 0x4342);
    half2 * h = (half2 *) &o;
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __hmul2(__hsub2(h[i], bias), s);
    return o;
}

template <bool Q8>
__device__ __forceinline__ uint4 kv_load8(const half * c, int64_t row, int D, int d8) {
    if constexpr (!Q8) {
        return *(const uint4 *) (c + row * D + d8);
    } else {
        const int8_t * r = (const int8_t *) (c + row * (D / 2 + D / 32));
        return kv_dequant8(*(const uint2 *) (r + d8), __half2half2(((const half *) (r + D))[d8 >> 5]));
    }
}

// value x of element i of a row (row-major cache index row): written by one full warp per 32 elements (i % 32 ==
// lane), as attn_prep's blocks of D threads do
template <bool Q8>
__device__ __forceinline__ void kv_store(half * c, int64_t row, int D, int i, float x) {
    if constexpr (!Q8) {
        c[row * D + i] = __float2half(x);
    } else {
        float a = fabsf(x);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off));
        const float s = a / 127.f;
        int8_t * r = (int8_t *) (c + row * (D / 2 + D / 32));
        r[i] = (int8_t) __float2int_rn(s > 0.f ? x / s : 0.f);
        if ((i & 31) == 0) ((half *) (r + D))[i >> 5] = __float2half(s);
    }
}

}  // namespace bnk
