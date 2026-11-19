// Kernels of the dense engine's residual stream: the residual add fused into the next RMSNorm, whose output is
// quantized for the dp4a GEMVs in the same pass (what rmsnorm_rows + quantize_act would write, bit for bit).
#include "dense/dense_ops.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>

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


namespace {
struct LoraUpArgs {
    const half * b[3];
    float * y[3];
    int64_t ld[3];
    int rows[3];
    int off[3];
};

// one thread per output row (of the three matrices in turn); the A product sits in shared memory and is read as a
// broadcast. RANK > 0: the row's weights in 16-byte loads issued together (RANK / 8 of them), else a plain loop.
template <int RANK, int NT>
__global__ void lora_up_k(LoraUpArgs p, const float * __restrict__ t, int T, int width, int rank_rt) {
    extern __shared__ float ts[];
    for (int i = threadIdx.x; i < T * width; i += blockDim.x) ts[i] = t[i];
    __syncthreads();
    const int rank = RANK > 0 ? RANK : rank_rt;
    const int total = p.rows[0] + p.rows[1] + p.rows[2];
    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < total; g += gridDim.x * blockDim.x) {
        int i = 0, row = g;
        if (row >= p.rows[0]) {
            row -= p.rows[0];
            i = 1;
            if (row >= p.rows[1]) {
                row -= p.rows[1];
                i = 2;
            }
        }
        const half * br = p.b[i] + (int64_t) row * rank;
        const float * tb = ts + p.off[i];
        float acc[NT];
#pragma unroll
        for (int tt = 0; tt < NT; ++tt) acc[tt] = 0.f;
        if constexpr (RANK > 0) {
            uint4 wv[RANK / 8];
#pragma unroll
            for (int j = 0; j < RANK / 8; ++j) wv[j] = ((const uint4 *) br)[j];
            const float4 * tb4 = (const float4 *) tb;   // width and the offsets are multiples of 8
            const int w4 = width / 4;
#pragma unroll
            for (int j = 0; j < RANK / 8; ++j) {
                const half2 * h = (const half2 *) &wv[j];
                const float2 w0 = __half22float2(h[0]), w1 = __half22float2(h[1]), w2 = __half22float2(h[2]),
                             w3 = __half22float2(h[3]);
#pragma unroll
                for (int tt = 0; tt < NT; ++tt) {
                    if (NT == kMaxGemvRows && tt >= T) break;
                    const float4 a = tb4[tt * w4 + 2 * j], c = tb4[tt * w4 + 2 * j + 1];
                    acc[tt] += w0.x * a.x + w0.y * a.y + w1.x * a.z + w1.y * a.w + w2.x * c.x + w2.y * c.y + w3.x * c.z + w3.y * c.w;
                }
            }
        } else {
            for (int k = 0; k < rank; k += 2) {
                const float2 w = __half22float2(*(const half2 *) (br + k));
#pragma unroll
                for (int tt = 0; tt < NT; ++tt)
                    if (tt < T) acc[tt] += w.x * tb[tt * width + k] + w.y * tb[tt * width + k + 1];
            }
        }
        float * y = p.y[i];
        const int64_t ld = p.ld[i];
#pragma unroll
        for (int tt = 0; tt < NT; ++tt)
            if (tt < T) y[tt * ld + row] += acc[tt];
    }
}
template <int NT>
void launch_lora_up(const LoraUpArgs & p, const float * t, int T, int width, int rank, int grid, int threads, size_t sh,
                    cudaStream_t s) {
    switch (rank) {
        case 8: lora_up_k<8, NT><<<grid, threads, sh, s>>>(p, t, T, width, rank); break;
        case 16: lora_up_k<16, NT><<<grid, threads, sh, s>>>(p, t, T, width, rank); break;
        case 32: lora_up_k<32, NT><<<grid, threads, sh, s>>>(p, t, T, width, rank); break;
        case 64: lora_up_k<64, NT><<<grid, threads, sh, s>>>(p, t, T, width, rank); break;
        default: lora_up_k<0, NT><<<grid, threads, sh, s>>>(p, t, T, width, rank); break;
    }
}
}  // namespace

void lora_up(const QMat * b, float * const * y, const int64_t * ld, const int * off, int n, const float * t, int T,
             int width, cudaStream_t s) {
    LoraUpArgs p{};
    int total = 0, rank = 0, k = 0;
    for (int i = 0; i < n && i < 3; ++i, ++k) {
        if (!b[i].valid() || !y[i]) {
            --k;
            continue;
        }
        p.b[k] = (const half *) b[i].data;
        p.y[k] = y[i];
        p.ld[k] = ld[i];
        p.rows[k] = (int) b[i].rows;
        p.off[k] = off[i];
        if (rank && rank != (int) b[i].cols) throw std::runtime_error("lora_up: B matrices of different ranks");
        rank = (int) b[i].cols;
        total += (int) b[i].rows;
    }
    if (!total) return;
    if (T < 1 || T > kMaxGemvRows) throw std::runtime_error("lora_up: window out of range");
    static int sms = 0;
    if (!sms) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    const int threads = 256, grid = std::min((total + threads - 1) / threads, sms * 4);
    const size_t sh = (size_t) T * width * sizeof(float);
    switch (T) {   // the window's rows exactly up to 8 (the decode and verify windows), 16 as the bound beyond
        case 1: launch_lora_up<1>(p, t, T, width, rank, grid, threads, sh, s); break;
        case 2: launch_lora_up<2>(p, t, T, width, rank, grid, threads, sh, s); break;
        case 3: launch_lora_up<3>(p, t, T, width, rank, grid, threads, sh, s); break;
        case 4: launch_lora_up<4>(p, t, T, width, rank, grid, threads, sh, s); break;
        case 5: launch_lora_up<5>(p, t, T, width, rank, grid, threads, sh, s); break;
        case 6: launch_lora_up<6>(p, t, T, width, rank, grid, threads, sh, s); break;
        case 7: launch_lora_up<7>(p, t, T, width, rank, grid, threads, sh, s); break;
        case 8: launch_lora_up<8>(p, t, T, width, rank, grid, threads, sh, s); break;
        default: launch_lora_up<kMaxGemvRows>(p, t, T, width, rank, grid, threads, sh, s); break;
    }
    check_launch("lora_up");
}

}  // namespace bnk
