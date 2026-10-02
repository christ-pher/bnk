// Host API of the decode-window matrix kernels (T <= 8 tokens).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace bnk {

constexpr int kMaxWindow = 8;  // tokens per decode/verify window

// Throws if the last kernel launch failed (bad configuration); no synchronization.
void check_launch(const char * what);

inline bool is_float_format(int t) { return t == 0 || t == 1 || t == 30; }  // F32, F16, BF16

// A weight matrix in its ggml format: `rows` output rows, each `cols` inputs wide.
struct QMat {
    const void * data = nullptr;  // device (or mapped host) pointer
    int type = 0;                 // ggml_type
    int64_t rows = 0, cols = 0;
    size_t row_bytes = 0;
    int layout = 0;               // 0: ggml blocks, 1: R layout (rfmt.h)
    uint32_t r_off[5] = {0, 0, 0, 0, 0};
    bool valid() const { return data != nullptr; }
};

// Activations quantized for dp4a: per token, `cols` int8 values plus one float scale per 32.
struct ActQ8 {
    int8_t * q = nullptr;   // [T][cols_pad]
    float * d = nullptr;    // [T][cols_pad/32]
    int64_t cols_pad = 0;   // row stride of q, multiple of 32
    int T = 0;
    int64_t cols = 0;
};

// Quantize x [T][ldx] (fp32) into `a` (whose buffers hold at least T*cols_pad values).
void quantize_act(const float * x, int64_t ldx, int T, int64_t cols, ActQ8 & a, cudaStream_t s);

// y[t*ldy + r] (+)= W[r,:] . x_t   for r < W.rows, t < T.
//   quantized W: x comes from `a` (quantize_act first); float W: x comes from `x`/ldx.
void gemv(const QMat & W, const ActQ8 * a, const float * x, int64_t ldx, int T,
          float * y, int64_t ldy, bool accumulate, cudaStream_t s);

// Convenience: quantizes into `scratch` when W needs it.
void gemv_auto(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy,
               bool accumulate, ActQ8 & scratch, cudaStream_t s);

// GEMV v2 kernels (gemv_r.cu): R-layout quant rows and float rows.
void gemv_r(const QMat & W, const ActQ8 & a, int T, float * y, int64_t ldy, bool accumulate, cudaStream_t s);
void gemv_float2(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy, bool accumulate,
                 cudaStream_t s);
// Upload a ggml-layout host matrix to the device, repacked to the R layout when supported.
QMat upload_matrix(const void * host, int type, int64_t rows, int64_t cols, size_t row_bytes, bool repack,
                   void ** dev_alloc);

// Dequantize rows [r0, r0+n) of W to fp32 (tests / embeddings).
void dequant_rows(const QMat & W, int64_t r0, int64_t n, float * out, cudaStream_t s);
// Dequantize the rows listed in `ids` (device int32) into out[i*cols].
void dequant_gather(const QMat & W, const int32_t * ids, int n, float * out, cudaStream_t s);

}  // namespace bnk
