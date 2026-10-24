// The activation quantization of one block, as a device function shared by the kernels that quantize on the way out
// (ops.cu's rmsnorm, the DeepSeek engine's hyper-connection pre-mix).
#pragma once
#include <cuda_fp16.h>

#include "kernels/gemv.h"

namespace bnk {

// One quantized activation block of 32 values spread over 8 lanes (4 each, consecutive lanes): what quantize_act_k
// writes for it (scale amax / 127, codes, the two 16-value code sums, the fp16 copy), bit for bit. `i` is the index of
// this lane's first value in the activation buffer.
__device__ __forceinline__ void quant4_store(float4 v, int64_t i, ActQ8 q) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float dd = amax / 127.f;
    const float id = dd > 0.f ? 1.f / dd : 0.f;
    const int a = __float2int_rn(v.x * id), b = __float2int_rn(v.y * id), c = __float2int_rn(v.z * id),
              e = __float2int_rn(v.w * id);
    *(char4 *) (q.q + i) = make_char4((signed char) a, (signed char) b, (signed char) c, (signed char) e);
    const int lane = threadIdx.x & 31;
    if ((lane & 7) == 0) q.d[i / 32] = dd;
    if (q.s) {
        int sm = a + b + c + e;
        sm += __shfl_xor_sync(0xffffffff, sm, 2);
        sm += __shfl_xor_sync(0xffffffff, sm, 1);
        if ((lane & 3) == 0) q.s[i / 16] = (int16_t) sm;
    }
    if (q.h) {
        const half2 h0 = __floats2half2_rn(fminf(fmaxf(v.x, -65504.f), 65504.f), fminf(fmaxf(v.y, -65504.f), 65504.f));
        const half2 h1 = __floats2half2_rn(fminf(fmaxf(v.z, -65504.f), 65504.f), fminf(fmaxf(v.w, -65504.f), 65504.f));
        *(half2 *) (q.h + i) = h0;
        *(half2 *) (q.h + i + 2) = h1;
    }
}

}  // namespace bnk
