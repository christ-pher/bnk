// Prompt-processing GEMMs: fp16 tensor-core products through cuBLAS, weights dequantized on the fly.
#pragma once

#include <cstdint>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/gemv.h"

namespace bnk {

void f32_to_f16(const float * x, half * y, int64_t n, cudaStream_t s);
// y[i] = (half) x[rows[i]] for a row gather (cols floats per row)
void gather_rows_f16(const float * x, int64_t cols, const int32_t * rows, int n, half * y, cudaStream_t s);
// h[p][f] = silu(gu[p][f]) * gu[p][F+f]   (fp16 rows of 2F -> F)
void swiglu_f16(const half * gu, half * h, int P, int F, cudaStream_t s);
// out[t][:] = base[t][:] * scale[t] + sum_j w[t*k+j] * D[inv[t*k+j]][:]   (base/scale optional)
void moe_gather_sum(const half * D, const int32_t * inv, const float * w, int N, int k, int E, const float * base,
                    const float * scale, const float * extra, float * out, cudaStream_t s);

class Gemm {
public:
    ~Gemm();
    // max_w_elems: the largest weight matrix dequantized at once; max_x_elems: the largest fp16 input
    void init(cudaStream_t s, size_t max_w_elems, size_t max_x_elems);
    // Y[N][M] (row stride ldy) = X[N][K] (fp32, row stride ldx) . W^T ; W is any QMat format
    void run(const QMat & W, const float * X, int64_t ldx, int N, float * Y, int64_t ldy, bool accumulate = false);
    // same with an fp16 input already prepared
    void run_h(const QMat & W, const half * X, int64_t ldx, int N, float * Y, int64_t ldy, bool accumulate = false);
    // raw fp16 product: Y[N][M] = X[N][K] . W[M][K]^T
    void raw(const half * W, int M, int K, const half * X, int64_t ldx, int N, float * Y, int64_t ldy, float beta);
    // grouped raw products (one cuBLAS call): Y_g[N_g][M] = X_g[N_g][K] . W_g[M][K]^T, fp16 outputs
    void grouped_h(const std::vector<const half *> & W, const std::vector<const half *> & X,
                   const std::vector<half *> & Y, const std::vector<int> & N, int M, int K);
    cublasHandle_t handle() const { return h_; }
    half * wbuf() const { return w16_; }
    half * xbuf() const { return x16_; }
    size_t wcap() const { return wcap_; }

private:
    cudaStream_t st_ = nullptr;
    cublasHandle_t h_ = nullptr;
    half * w16_ = nullptr, * x16_ = nullptr;
    size_t wcap_ = 0, xcap_ = 0;
    void ** ptrs_ = nullptr;  // device pointer arrays for grouped calls
    int ptr_cap_ = 0;
};

}  // namespace bnk
