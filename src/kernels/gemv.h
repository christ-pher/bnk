// Host API of the decode-window matrix kernels (T <= kMaxGemvRows tokens).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace bnk {

constexpr int kMaxWindow = 8;        // tokens per decode/verify window (the MoE engine; shared defaults)
constexpr int kMaxWindowDense = 16;  // the dense engine's windows (verifies up to 15 lookup drafts)
constexpr int kMaxGemvRows = 16;     // the GEMVs: the tensor-core kernel takes 16 tokens in one pass, the others two

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
    float scale = 1.f;            // multiplies every dequantized value (dequant_rows / gather only: a matrix-wide
                                  // global scale, e.g. NVFP4's)
    bool valid() const { return data != nullptr; }
};

// Activations quantized for dp4a: per token, `cols` int8 values plus one float scale per 32.
struct ActQ8 {
    int8_t * q = nullptr;   // [T][cols_pad]
    float * d = nullptr;    // [T][cols_pad/32]
    int16_t * s = nullptr;  // [T][cols_pad/16] sums of the codes per 16 (optional: kernels fall back to dp4a)
    half * h = nullptr;     // [T][cols_pad] the activations in fp16 (optional: the tensor-core GEMV)
    int64_t cols_pad = 0;   // row stride of q, multiple of 32
    int T = 0;
    int64_t cols = 0;
};

// Quantize x [T][ldx] (fp32) into `a` (whose buffers hold at least T*cols_pad values).
void quantize_act(const float * x, int64_t ldx, int T, int64_t cols, ActQ8 & a, cudaStream_t s);
// ... of silu(x * scale), and of silu(g) * u (the elementwise step folded into the quantization pass)
void quantize_act_silu(const float * x, float scale, int64_t ldx, int T, int64_t cols, ActQ8 & a, cudaStream_t s);
void quantize_act_silu_mul(const float * g, const float * u, int64_t ldx, int T, int64_t cols, ActQ8 & a, cudaStream_t s);
// quantize_act_silu_mul clamps up to [-limit, limit] and the gate to <= limit first (DeepSeek-V4; 0 = none, the default)
void quantize_set_swiglu_limit(float limit);
// ... of rmsnorm(o[t][h]) * w * sigmoid(z[t][h]) over T x H heads of S values (gated_rmsnorm folded in); false
// when S is not 128 (then run gated_rmsnorm and quantize_act)
bool quantize_gated_norm(const float * o, const float * z, const float * w, int T, int H, int S, float eps, ActQ8 & a,
                         cudaStream_t s);
// ... with the gate silu(z) instead of sigmoid(z) (the qwen35 family)
bool quantize_gated_norm_silu(const float * o, const float * z, const float * w, int T, int H, int S, float eps, ActQ8 & a,
                              cudaStream_t s);

// Two matrices that share an input are stacked into one at load time (Model: beta | alpha, the indexer's k | q,
// router | shared-expert gate, shared gate | up, HC down | inject, k | v): one launch, and the rows from r1 on go
// to a second output y2 (row stride ldy2), so each consumer still reads a contiguous array.
struct GemvSplit {
    int r1 = 0x7fffffff;
    float * y2 = nullptr;
    int64_t ldy2 = 0;
    float wscale = 1.f;   // the matrix's global scale (QMat::scale, NVFP4's), applied as each output is written
    // grp_rows > 0: a block-diagonal product in one launch. Rows [g grp_rows, (g+1) grp_rows) read the activation
    // columns from g * grp_cols on (the activation quantized over all the groups' columns at once, its `cols` set to
    // W.cols). DeepSeek's grouped output projection: 8 groups of o_lora rows over 8 slices of the heads. R layout only;
    // grp_rows a multiple of 16.
    int grp_rows = 0;
    int64_t grp_cols = 0;
};

// the same activations / split output from token t0 on (a window of more than 8 tokens as two calls)
inline ActQ8 act_rows(const ActQ8 & a, int t0) {
    ActQ8 r = a;
    r.q = a.q + (int64_t) t0 * a.cols_pad;
    r.d = a.d + (int64_t) t0 * (a.cols_pad / 32);
    if (a.s) r.s = a.s + (int64_t) t0 * (a.cols_pad / 16);
    if (a.h) r.h = a.h + (int64_t) t0 * a.cols_pad;
    r.T = a.T - t0;
    return r;
}
inline GemvSplit split_rows(const GemvSplit & sp, int t0) {
    GemvSplit r = sp;
    if (r.y2) r.y2 += (int64_t) t0 * r.ldy2;
    return r;
}

// y[t*ldy + r] (+)= W[r,:] . x_t   for r < W.rows, t < T (rows from sp.r1 on into sp.y2).
//   quantized W: x comes from `a` (quantize_act first); float W: x comes from `x`/ldx.
void gemv(const QMat & W, const ActQ8 * a, const float * x, int64_t ldx, int T,
          float * y, int64_t ldy, bool accumulate, cudaStream_t s, GemvSplit sp = {});

// BNK_GEMV_REF=1: the reference dp4a kernel for every quantized GEMV (debugging)
bool use_ref_gemv();

// Convenience: quantizes into `scratch` when W needs it.
void gemv_auto(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy,
               bool accumulate, ActQ8 & scratch, cudaStream_t s, GemvSplit sp = {});

// GEMV v2 kernels (gemv_r.cu): R-layout quant rows and float rows.
void gemv_f16_act(const QMat & W, const ActQ8 & a, int T, float * y, int64_t ldy, bool accumulate, cudaStream_t s,
                  GemvSplit sp = {});
void gemv_r(const QMat & W, const ActQ8 & a, int T, float * y, int64_t ldy, bool accumulate, cudaStream_t s,
            GemvSplit sp = {});
void gemv_float2(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy, bool accumulate,
                 cudaStream_t s, GemvSplit sp = {});
// rows [r0, r0 + n) of W as a matrix of their own (same storage)
inline QMat qmat_rows(const QMat & W, int64_t r0, int64_t n) {
    QMat v = W;
    v.data = (const uint8_t *) W.data + (size_t) r0 * W.row_bytes;
    v.rows = n;
    return v;
}
// Upload a ggml-layout host matrix to the device, repacked to the R layout when supported.
QMat upload_matrix(const void * host, int type, int64_t rows, int64_t cols, size_t row_bytes, bool repack,
                   void ** dev_alloc);

// Dequantize rows [r0, r0+n) of W to fp32 (tests / embeddings).
void dequant_rows(const QMat & W, int64_t r0, int64_t n, float * out, cudaStream_t s);
// ... to fp16 (prefill GEMM operands)
void dequant_rows_f16(const QMat & W, int64_t r0, int64_t n, half * out, cudaStream_t s);
// Dequantize the rows listed in `ids` (device int32) into out[i*cols].
void dequant_gather(const QMat & W, const int32_t * ids, int n, float * out, cudaStream_t s);

}  // namespace bnk
