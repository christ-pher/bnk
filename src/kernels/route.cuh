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
__device__ __forceinline__ void route_topk_warp(const float * logits, int n_exp, int k, int32_t * ids, float * w,
                                                float w_scale, int t, int lane) {
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
        w[t * k + lane] = myw * sc;
    }
}


// DeepSeek-V4's routing of one token by one warp: s = sqrt(softplus(logit)), the k largest of s + bias (lowest id on
// ties), w = s * scale / (sum + 1e-20) (norm) or s * scale. n_exp <= 512.
__device__ __forceinline__ float ds_softplusf(float x) { return x > 20.f ? x : x < -20.f ? __expf(x) : log1pf(__expf(x)); }
__device__ __forceinline__ void ds_route_warp(const float * logits, const float * bias, int n_exp, int k, float scale,
                                              int norm, int32_t * ids, float * w, int t, int lane) {
    constexpr int PER = 16;
    float s[PER], sc[PER];
    bool taken[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int e = lane + 32 * j;
        s[j] = e < n_exp ? sqrtf(ds_softplusf(logits[(int64_t) t * n_exp + e])) : 0.f;
        sc[j] = e < n_exp ? s[j] + bias[e] : -FLT_MAX;
        taken[j] = e >= n_exp;
    }
    float sum = 0.f;
    for (int r = 0; r < k; ++r) {
        float best = -FLT_MAX;
        int bj = -1;
#pragma unroll
        for (int j = 0; j < PER; ++j)
            if (!taken[j] && sc[j] > best) { best = sc[j]; bj = j; }
        const float wb = rt_warp_max(best);
        int cand = (best == wb && bj >= 0) ? lane + 32 * bj : 0x7fffffff;
#pragma unroll
        for (int o = 16; o; o >>= 1) cand = min(cand, __shfl_xor_sync(0xffffffffu, cand, o));
        const int e = cand;
        const int owner = e & 31, j = e >> 5;
        float sv = 0.f;
        if (lane == owner) {
#pragma unroll
            for (int jj = 0; jj < PER; ++jj)
                if (jj == j) { taken[jj] = true; sv = s[jj]; }
        }
        sv = __shfl_sync(0xffffffffu, sv, owner);
        sum += sv;
        if (lane == 0) { ids[t * k + r] = e; w[t * k + r] = sv; }
    }
    if (lane == 0) {
        const float f = (norm && k > 1) ? scale / (sum + 1e-20f) : scale;
        for (int r = 0; r < k; ++r) w[t * k + r] *= f;
    }
}

}  // namespace bnk
