// Kernels of the dense engine's residual stream: the residual add fused into the next RMSNorm, whose output is
// quantized for the dp4a GEMVs in the same pass (what rmsnorm_rows + quantize_act would write, bit for bit).
#include "dense/dense_ops.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_fp16.h>

#include "kernels/ops.h"

namespace bnk {

namespace {

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float bsum(float v, float * sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = wsum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = threadIdx.x < nw ? sh[threadIdx.x] : 0.f;
    if (wid == 0) v = wsum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

// One quantized activation block of 32 values over 8 consecutive lanes (4 each): scale amax / 127, codes, the two
// 16-value code sums, the fp16 copy (as kernels/ops.cu quant4_store). i: this lane's first value's index.
__device__ __forceinline__ void q4_store(float4 v, int64_t i, ActQ8 q) {
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

// One row per block, P float4 per thread held in registers: res (+= pend), xn = rmsnorm(res) * w, quantized.
template <int P, bool ADD>
__global__ void norm_q_k(float * __restrict__ res, const float * __restrict__ pend, const float * __restrict__ w,
                         float * __restrict__ xn, int n, float eps, ActQ8 q) {
    __shared__ float sh[32];
    const int r = blockIdx.x, n4 = n / 4;
    float4 * xr = (float4 *) (res + (int64_t) r * n);
    float4 v[P];
    float ss = 0.f;
#pragma unroll
    for (int p = 0; p < P; ++p) {
        const int i = threadIdx.x + p * blockDim.x;
        v[p] = i < n4 ? xr[i] : make_float4(0.f, 0.f, 0.f, 0.f);
        if constexpr (ADD) {
            if (i < n4) {
                const float4 o = ((const float4 *) (pend + (int64_t) r * n))[i];
                v[p].x += o.x; v[p].y += o.y; v[p].z += o.z; v[p].w += o.w;
                xr[i] = v[p];
            }
        }
        ss += v[p].x * v[p].x + v[p].y * v[p].y + v[p].z * v[p].z + v[p].w * v[p].w;
    }
    ss = bsum(ss, sh);
    const float inv = rsqrtf(ss / n + eps);
    const float4 * wr = (const float4 *) w;
    float4 * yr = (float4 *) (xn + (int64_t) r * n);
#pragma unroll
    for (int p = 0; p < P; ++p) {
        const int i = threadIdx.x + p * blockDim.x;
        if (i >= n4) continue;
        const float4 g = wr[i];
        const float4 o = make_float4(v[p].x * inv * g.x, v[p].y * inv * g.y, v[p].z * inv * g.z, v[p].w * inv * g.w);
        yr[i] = o;
        q4_store(o, (int64_t) r * n + 4 * i, q);
    }
}

__global__ void add_rows_k(float * __restrict__ y, const float * __restrict__ x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
__global__ void add_rows_f16x2_k(float * __restrict__ y, const half * __restrict__ a, const half * __restrict__ b, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += __half2float(a[i]) + __half2float(b[i]);
}

}  // namespace

void dense_norm_q(float * res, const float * pend, const float * w, float * xn, int T, int E, float eps, ActQ8 & q,
                  cudaStream_t s) {
    const int n4 = E / 4;
    int P = 1;
    while (P < 8 && (n4 + P - 1) / P > 512) P *= 2;
    const int threads = (n4 + P - 1) / P;
    if (E % 128 || threads % 32 || getenv("BNK_RMSNORM_SCALAR")) {   // whole warps over whole quant blocks only
        if (pend) add_rows(res, pend, (int64_t) T * E, s);
        rmsnorm_rows(res, w, xn, T, E, E, E, eps, s);
        quantize_act(xn, E, T, E, q, s);
        return;
    }
    q.T = T;
    q.cols = q.cols_pad = E;
    auto L = [&]<int PP>() {
        if (pend) norm_q_k<PP, true><<<T, threads, 0, s>>>(res, pend, w, xn, E, eps, q);
        else norm_q_k<PP, false><<<T, threads, 0, s>>>(res, nullptr, w, xn, E, eps, q);
    };
    switch (P) {
        case 1: L.template operator()<1>(); break;
        case 2: L.template operator()<2>(); break;
        case 4: L.template operator()<4>(); break;
        default: L.template operator()<8>();
    }
}

void add_rows(float * y, const float * x, int64_t n, cudaStream_t s) {
    add_rows_k<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(y, x, n);
}
void add_rows_f16x2(float * y, const half * a, const half * b, int64_t n, cudaStream_t s) {
    add_rows_f16x2_k<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(y, a, b, n);
}

}  // namespace bnk
