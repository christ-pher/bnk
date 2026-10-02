// Decode-window GEMV kernels: dp4a over unpacked quant formats, and a float path.
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "kernels/gemv.h"
#include "kernels/quant.cuh"
#include "kernels/rfmt.cuh"
#include <vector>

namespace bnk {

void check_launch(const char * what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": launch failed: " + cudaGetErrorString(e));
}

void bnk_unsupported_format(int fmt) {
    throw std::runtime_error("unsupported weight format (ggml type " + std::to_string(fmt) + ")");
}

// ------------------------------------------------------------------------------ activation quantize
// One warp per 32-value block: d = amax/127, q = round(x/d).
__global__ void quantize_act_k(const float * __restrict__ x, int64_t ldx, int64_t cols, int8_t * __restrict__ q,
                               float * __restrict__ d, int16_t * __restrict__ sums, int64_t cols_pad, int nblk_total) {
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (gw >= nblk_total) return;
    const int64_t nb = cols_pad / 32;
    const int t = gw / nb;
    const int64_t b = gw % nb;
    const int64_t c = b * 32 + lane;
    const float v = c < cols ? x[t * ldx + c] : 0.f;
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float dd = amax / 127.f;
    const float id = dd > 0.f ? 1.f / dd : 0.f;
    const int qi = __float2int_rn(v * id);
    q[t * cols_pad + c] = (int8_t) qi;
    if (lane == 0) d[t * nb + b] = dd;
    if (sums) {
        int sm = qi;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) sm += __shfl_xor_sync(0xffffffff, sm, o);  // within each half-warp
        if ((lane & 15) == 0) sums[(t * nb + b) * 2 + (lane >> 4)] = (int16_t) sm;
    }
}

void quantize_act(const float * x, int64_t ldx, int T, int64_t cols, ActQ8 & a, cudaStream_t s) {
    a.T = T;
    a.cols = cols;
    a.cols_pad = (cols + 31) / 32 * 32;
    const int nblk = (int) (T * (a.cols_pad / 32));
    const int threads = 256;
    quantize_act_k<<<(nblk * 32 + threads - 1) / threads, threads, 0, s>>>(x, ldx, cols, a.q, a.d, a.s, a.cols_pad, nblk);
}

// ------------------------------------------------------------------------------ dp4a GEMV
template <int FMT, int NT>
__global__ void __launch_bounds__(256) gemv_dp4a_k(const uint8_t * __restrict__ W, size_t row_bytes, int rows,
                                                   int nsb, const int8_t * __restrict__ aq,
                                                   const float * __restrict__ ad, int64_t cols_pad, int T,
                                                   float * __restrict__ y, int64_t ldy, int accumulate) {
    using Q = QTraits<FMT>;
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;
    const int64_t nb = cols_pad / 32;

    for (int r = warp; r < rows; r += nwarps) {
        const uint8_t * row = W + (size_t) r * row_bytes;
        float acc[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] = 0.f;

        for (int sb = lane; sb < nsb; sb += 32) {
            Unpacked u;
            unpack_sub<FMT>(row, sb, u);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < T) {
                    const int4 * ap = (const int4 *) (aq + t * cols_pad + sb * 32);
                    const int4 a0 = __ldg(ap), a1 = __ldg(ap + 1);
                    int i0 = __dp4a(u.w[0], a0.x, 0);
                    i0 = __dp4a(u.w[1], a0.y, i0);
                    i0 = __dp4a(u.w[2], a0.z, i0);
                    i0 = __dp4a(u.w[3], a0.w, i0);
                    int i1 = __dp4a(u.w[4], a1.x, 0);
                    i1 = __dp4a(u.w[5], a1.y, i1);
                    i1 = __dp4a(u.w[6], a1.z, i1);
                    i1 = __dp4a(u.w[7], a1.w, i1);
                    const float da = __ldg(ad + t * nb + sb);
                    float v = u.d0 * (float) i0 + u.d1 * (float) i1;
                    if constexpr (Q::HAS_MIN) {
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
                    acc[t] += da * v;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[t] += __shfl_xor_sync(0xffffffff, acc[t], o);
        }
        if (lane == 0) {
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

// Reference path: dequantized weights times fp32 activations (no activation quantization).
template <int FMT, int NT, bool RLAY>
__global__ void __launch_bounds__(256) gemv_ref_k(const uint8_t * __restrict__ W, size_t row_bytes, int rows, int nsb,
                                                  const float * __restrict__ x, int64_t ldx, int T,
                                                  float * __restrict__ y, int64_t ldy, int accumulate, ROff ro) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;
    for (int r = warp; r < rows; r += nwarps) {
        const uint8_t * row = W + (size_t) r * row_bytes;
        float acc[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] = 0.f;
        for (int sb = lane; sb < nsb; sb += 32) {
            Unpacked u;
            if constexpr (RLAY) RT<FMT>::unpack(row, ro, sb, u);
            else unpack_sub<FMT>(row, sb, u);
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                const float w = (k < 16 ? u.d0 : u.d1) * (float) (int8_t) (u.w[k / 4] >> (8 * (k % 4))) - (k < 16 ? u.m0 : u.m1);
#pragma unroll
                for (int t = 0; t < NT; ++t)
                    if (t < T) acc[t] += w * x[t * ldx + sb * 32 + k];
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[t] += __shfl_xor_sync(0xffffffff, acc[t], o);
        }
        if (lane == 0)
            for (int t = 0; t < NT && t < T; ++t) {
                float * yp = y + t * ldy + r;
                *yp = accumulate ? *yp + acc[t] : acc[t];
            }
    }
}

static int sm_count() {
    static int n = 0;
    if (!n) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    }
    return n;
}

static int grid_for_rows(int64_t rows, int warps_per_block) {
    int64_t blocks = (rows + warps_per_block - 1) / warps_per_block;
    int64_t cap = (int64_t) sm_count() * 8;
    return (int) (blocks < cap ? blocks : cap);
}

template <int NT>
static void launch_dp4a(const QMat & W, const ActQ8 & a, int T, float * y, int64_t ldy, bool acc, cudaStream_t s) {
    const int nsb = (int) (W.cols / 32);
    const int grid = grid_for_rows(W.rows, 8);
    auto f = [&]<int FMT>() {
        gemv_dp4a_k<FMT, NT><<<grid, 256, 0, s>>>((const uint8_t *) W.data, W.row_bytes, (int) W.rows, nsb, a.q, a.d,
                                                  a.cols_pad, T, y, ldy, acc ? 1 : 0);
    };
    BNK_DISPATCH_DP4A(W.type, f);
}

// ------------------------------------------------------------------------------ float GEMV
template <int FMT>
__device__ __forceinline__ void load8(const uint8_t * row, int64_t i, float (&w)[8]) {
    if constexpr (FMT == QT_F32) {
        const float4 * p = (const float4 *) (row + i * 4);
        float4 a = __ldg(p), b = __ldg(p + 1);
        w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
    } else if constexpr (FMT == QT_F16) {
        const uint4 v = __ldg((const uint4 *) (row + i * 2));
        const __half2 * h = (const __half2 *) &v;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 f = __half22float2(h[k]);
            w[2 * k] = f.x;
            w[2 * k + 1] = f.y;
        }
    } else {  // BF16
        const uint4 v = __ldg((const uint4 *) (row + i * 2));
        const uint32_t * u = (const uint32_t *) &v;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            w[2 * k] = __uint_as_float(u[k] << 16);
            w[2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u);
        }
    }
}

template <int FMT, int NT>
__global__ void __launch_bounds__(256) gemv_float_k(const uint8_t * __restrict__ W, size_t row_bytes, int rows,
                                                    int64_t cols, const float * __restrict__ x, int64_t ldx, int T,
                                                    float * __restrict__ y, int64_t ldy, int accumulate) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;
    for (int r = warp; r < rows; r += nwarps) {
        const uint8_t * row = W + (size_t) r * row_bytes;
        float acc[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] = 0.f;
        for (int64_t i = (int64_t) lane * 8; i < cols; i += 256) {
            float w[8];
            load8<FMT>(row, i, w);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < T) {
                    const float4 * xp = (const float4 *) (x + t * ldx + i);
                    const float4 a = __ldg(xp), b = __ldg(xp + 1);
                    acc[t] += w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * b.x + w[5] * b.y +
                              w[6] * b.z + w[7] * b.w;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[t] += __shfl_xor_sync(0xffffffff, acc[t], o);
        }
        if (lane == 0) {
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

template <int NT>
static void launch_float(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy, bool acc,
                         cudaStream_t s) {
    if (W.cols % 8) throw std::runtime_error("float GEMV needs cols % 8 == 0");
    const int grid = grid_for_rows(W.rows, 8);
    const uint8_t * w = (const uint8_t *) W.data;
    const int a = acc ? 1 : 0;
    switch (W.type) {
        case QT_F32: gemv_float_k<QT_F32, NT><<<grid, 256, 0, s>>>(w, W.row_bytes, (int) W.rows, W.cols, x, ldx, T, y, ldy, a); break;
        case QT_F16: gemv_float_k<QT_F16, NT><<<grid, 256, 0, s>>>(w, W.row_bytes, (int) W.rows, W.cols, x, ldx, T, y, ldy, a); break;
        case QT_BF16: gemv_float_k<QT_BF16, NT><<<grid, 256, 0, s>>>(w, W.row_bytes, (int) W.rows, W.cols, x, ldx, T, y, ldy, a); break;
        default: bnk_unsupported_format(W.type);
    }
}

void gemv(const QMat & W, const ActQ8 * a, const float * x, int64_t ldx, int T, float * y, int64_t ldy,
          bool accumulate, cudaStream_t s) {
    if (T < 1 || T > kMaxWindow) throw std::runtime_error("gemv: window out of range");
    if (W.layout == 1) {
        if (!a || a->cols != W.cols) throw std::runtime_error("gemv: activation not quantized for this width");
        gemv_r(W, *a, T, y, ldy, accumulate, s);
        return;
    }
    if (is_float_format(W.type) && !getenv("BNK_OLD_FLOAT")) {
        gemv_float2(W, x, ldx, T, y, ldy, accumulate, s);
        return;
    }
    if (is_float_format(W.type)) {
        if (T == 1) launch_float<1>(W, x, ldx, T, y, ldy, accumulate, s);
        else if (T == 2) launch_float<2>(W, x, ldx, T, y, ldy, accumulate, s);
        else if (T <= 4) launch_float<4>(W, x, ldx, T, y, ldy, accumulate, s);
        else launch_float<8>(W, x, ldx, T, y, ldy, accumulate, s);
        return;
    }
    if (!a || a->cols != W.cols) throw std::runtime_error("gemv: activation not quantized for this width");
    if (T == 1) launch_dp4a<1>(W, *a, T, y, ldy, accumulate, s);
    else if (T == 2) launch_dp4a<2>(W, *a, T, y, ldy, accumulate, s);
    else if (T <= 4) launch_dp4a<4>(W, *a, T, y, ldy, accumulate, s);
    else launch_dp4a<8>(W, *a, T, y, ldy, accumulate, s);
}

static bool use_ref_gemv() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("BNK_GEMV_REF"); v = e && *e == '1'; }
    return v == 1;
}

void gemv_auto(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy, bool accumulate,
               ActQ8 & scratch, cudaStream_t s) {
    if (!is_float_format(W.type) && use_ref_gemv()) {
        const int grid = grid_for_rows(W.rows, 8);
        const ROff ro{W.r_off[0], W.r_off[1], W.r_off[2], W.r_off[3], W.r_off[4]};
        auto f = [&]<int FMT>() {
            if (W.layout == 1)
                gemv_ref_k<FMT, 8, true><<<grid, 256, 0, s>>>((const uint8_t *) W.data, W.row_bytes, (int) W.rows,
                                                              (int) (W.cols / 32), x, ldx, T, y, ldy, accumulate ? 1 : 0, ro);
            else
                gemv_ref_k<FMT, 8, false><<<grid, 256, 0, s>>>((const uint8_t *) W.data, W.row_bytes, (int) W.rows,
                                                               (int) (W.cols / 32), x, ldx, T, y, ldy, accumulate ? 1 : 0, ro);
        };
        BNK_DISPATCH_DP4A(W.type, f);
        return;
    }
    if (is_float_format(W.type)) {
        gemv(W, nullptr, x, ldx, T, y, ldy, accumulate, s);
    } else {
        quantize_act(x, ldx, T, W.cols, scratch, s);
        gemv(W, &scratch, nullptr, 0, T, y, ldy, accumulate, s);
    }
}

// ------------------------------------------------------------------------------ dequantize
template <typename O> __device__ __forceinline__ O cvt_out(float v);
template <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }
template <> __device__ __forceinline__ half cvt_out<half>(float v) { return __float2half(v); }

// the 32 dequantized weights of a sub-block, written with vector stores
template <typename O>
__device__ __forceinline__ void store32(const Unpacked & u, O * o) {
    float v[32];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float dd = k < 4 ? u.d0 : u.d1, mm = k < 4 ? u.m0 : u.m1;
#pragma unroll
        for (int j = 0; j < 4; ++j) v[4 * k + j] = dd * (float) (int8_t) (u.w[k] >> (8 * j)) - mm;
    }
    if constexpr (sizeof(O) == 2) {
        uint4 * o4 = (uint4 *) o;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            half2 h[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) h[j] = __floats2half2_rn(v[8 * q + 2 * j], v[8 * q + 2 * j + 1]);
            o4[q] = *(uint4 *) h;
        }
    } else {
        float4 * o4 = (float4 *) o;
#pragma unroll
        for (int q = 0; q < 8; ++q) o4[q] = make_float4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
    }
}

template <int FMT, typename O = float>
__global__ void dequant_k(const uint8_t * __restrict__ W, size_t row_bytes, const int32_t * __restrict__ ids,
                          int64_t r0, int n, int64_t cols, O * __restrict__ out) {
    const int64_t nsb = cols / 32;
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (int64_t) n * nsb) return;
    const int i = (int) (gid / nsb);
    const int sb = (int) (gid % nsb);
    const int64_t r = ids ? ids[i] : r0 + i;
    const uint8_t * row = W + (size_t) r * row_bytes;
    O * o = out + (int64_t) i * cols + sb * 32;
    if constexpr (is_float_fmt(FMT)) {
        float w[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            load8<FMT>(row, sb * 32 + 8 * k, w);
#pragma unroll
            for (int j = 0; j < 8; ++j) o[8 * k + j] = cvt_out<O>(w[j]);
        }
    } else {
        Unpacked u;
        unpack_sub<FMT>(row, sb, u);
        store32<O>(u, o);
    }
}

template <int FMT, typename O = float>
__global__ void dequant_r_k(const uint8_t * __restrict__ W, size_t row_bytes, ROff o, const int32_t * __restrict__ ids,
                            int64_t r0, int n, int64_t cols, O * __restrict__ out) {
    const int64_t nsb = cols / 32;
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (int64_t) n * nsb) return;
    const int i = (int) (gid / nsb);
    const int sb = (int) (gid % nsb);
    const int64_t r = ids ? ids[i] : r0 + i;
    Unpacked u;
    RT<FMT>::unpack(W + (size_t) r * row_bytes, o, sb, u);
    store32<O>(u, out + (int64_t) i * cols + sb * 32);
}

template <typename O>
static void dequant_launch(const QMat & W, const int32_t * ids, int64_t r0, int n, O * out, cudaStream_t s) {
    if (W.layout == 1) {
        const int64_t total = (int64_t) n * (W.cols / 32);
        const ROff o{W.r_off[0], W.r_off[1], W.r_off[2], W.r_off[3], W.r_off[4]};
        auto f = [&]<int FMT>() {
            dequant_r_k<FMT, O><<<(int) ((total + 127) / 128), 128, 0, s>>>((const uint8_t *) W.data, W.row_bytes, o,
                                                                           ids, r0, n, W.cols, out);
        };
        BNK_DISPATCH_R(W.type, f);
        return;
    }
    const int64_t total = (int64_t) n * (W.cols / 32);
    const int threads = 128;
    const int grid = (int) ((total + threads - 1) / threads);
    const uint8_t * w = (const uint8_t *) W.data;
    auto f = [&]<int FMT>() { dequant_k<FMT, O><<<grid, threads, 0, s>>>(w, W.row_bytes, ids, r0, n, W.cols, out); };
    switch (W.type) {
        case QT_F32: f.template operator()<QT_F32>(); break;
        case QT_F16: f.template operator()<QT_F16>(); break;
        case QT_BF16: f.template operator()<QT_BF16>(); break;
        default: BNK_DISPATCH_DP4A(W.type, f);
    }
}

void dequant_rows(const QMat & W, int64_t r0, int64_t n, float * out, cudaStream_t s) {
    dequant_launch(W, nullptr, r0, (int) n, out, s);
}

void dequant_gather(const QMat & W, const int32_t * ids, int n, float * out, cudaStream_t s) {
    dequant_launch(W, ids, 0, n, out, s);
}

void dequant_rows_f16(const QMat & W, int64_t r0, int64_t n, half * out, cudaStream_t s) {
    dequant_launch(W, nullptr, r0, (int) n, out, s);
    check_launch("dequant_f16");
}

}  // namespace bnk

namespace bnk {
QMat upload_matrix(const void * host, int type, int64_t rows, int64_t cols, size_t row_bytes, bool repack,
                   void ** dev_alloc) {
    QMat m;
    m.type = type;
    m.rows = rows;
    m.cols = cols;
    void * d = nullptr;
    const int64_t need = (type == QT_IQ4_NL || type == QT_Q8_0) ? 32 : type == QT_Q2_0 ? 64 : 256;
    if (repack && r_supported(type) && cols % need == 0) {
        const RLayout L = r_layout(type, cols);
        std::vector<uint8_t> buf(L.row_bytes * rows);
        r_repack(L, (const uint8_t *) host, row_bytes, rows, buf.data());
        if (cudaMalloc(&d, buf.size()) != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
        cudaMemcpy(d, buf.data(), buf.size(), cudaMemcpyHostToDevice);
        m.row_bytes = L.row_bytes;
        m.layout = 1;
        m.r_off[0] = L.off_a; m.r_off[1] = L.off_b; m.r_off[2] = L.off_c; m.r_off[3] = L.off_d; m.r_off[4] = L.off_e;
    } else {
        if (cudaMalloc(&d, row_bytes * rows) != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
        cudaMemcpy(d, host, row_bytes * rows, cudaMemcpyHostToDevice);
        m.row_bytes = row_bytes;
    }
    m.data = d;
    *dev_alloc = d;
    return m;
}
}  // namespace bnk
