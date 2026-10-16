// Top-k expert routing of one token by one warp (softmax over the router logits, the k largest, weights normalized
// and scaled), shared by route_topk (ops.cu) and the fused routing + plan launch (moe.cu).
#pragma once
#include <cfloat>
#include <cstdint>

namespace bnk {

__device__ __forceinline__ float rt_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float rt_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

template <int PER>   // n_exp / 32 values per lane (rounded up); one warp, token t
// wmul (optional, [n_exp]) multiplies the weight of expert e by wmul[e] after the renormalization: the per-expert
// global scales of NVFP4 up / down matrices ride on the routing weight (both are linear in the expert's output).
__device__ __forceinline__ void route_topk_warp(const float * logits, int n_exp, int k, int32_t * ids, float * w,
                                                float w_scale, const float * wmul, int t, int lane) {
    const float * lg = logits + (int64_t) t * n_exp;
    float v[PER];
    float mx = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        v[i] = lane + 32 * i < n_exp ? lg[lane + 32 * i] : -FLT_MAX;
        mx = fmaxf(mx, v[i]);
    }
    mx = rt_warp_max(mx);
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        v[i] = v[i] == -FLT_MAX ? 0.f : __expf(v[i] - mx);
        sum += v[i];
    }
    sum = rt_warp_sum(sum);
    float wsum = 0.f;
    float myw = 0.f;
    int myid = 0;
    for (int j = 0; j < k; ++j) {
        float best = -1.f;
        int bi = 0;
#pragma unroll
        for (int i = 0; i < PER; ++i)
            if (v[i] > best) { best = v[i]; bi = lane + 32 * i; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, best, o);
            const int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        // drop the winner (a static index: a runtime one put v[] in local memory)
#pragma unroll
        for (int i = 0; i < PER; ++i)
            if (bi == lane + 32 * i) v[i] = -2.f;
        const float p = best / sum;
        wsum += p;
        if (lane == j) { myw = p; myid = bi; }
    }
    if (lane < k) {
        const float sc = (w_scale != 0.f ? w_scale : 1.f) / wsum;
        ids[t * k + lane] = myid;
        w[t * k + lane] = myw * sc * (wmul ? wmul[myid] : 1.f);
    }
}

}  // namespace bnk
