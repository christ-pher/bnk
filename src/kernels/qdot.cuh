// The dp4a dot product of one unpacked 32-weight sub-block with one quantized activation sub-block.
#pragma once

#include "kernels/quant.cuh"

namespace bnk {

// a0/a1: the 32 int8 activations of the sub-block, da: their scale.
template <int FMT, bool MIN = QTraits<FMT>::HAS_MIN>
__device__ __forceinline__ float qdot_sub(const Unpacked & u, const int4 a0, const int4 a1, float da) {
    int i0 = __dp4a(u.w[0], a0.x, 0);
    i0 = __dp4a(u.w[1], a0.y, i0);
    i0 = __dp4a(u.w[2], a0.z, i0);
    i0 = __dp4a(u.w[3], a0.w, i0);
    int i1 = __dp4a(u.w[4], a1.x, 0);
    i1 = __dp4a(u.w[5], a1.y, i1);
    i1 = __dp4a(u.w[6], a1.z, i1);
    i1 = __dp4a(u.w[7], a1.w, i1);
    float v = u.d0 * (float) i0 + u.d1 * (float) i1;
    if constexpr (MIN) {
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
    return da * v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// Quantize 32 floats held one per lane into int8 + scale (lane 0 gets the scale back too).
__device__ __forceinline__ int8_t quant_lane(float v, float & d) {
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    d = amax / 127.f;
    const float id = d > 0.f ? 1.f / d : 0.f;
    return (int8_t) __float2int_rn(v * id);
}

}  // namespace bnk
