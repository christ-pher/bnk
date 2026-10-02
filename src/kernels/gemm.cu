#include "kernels/gemm.h"

#include <stdexcept>
#include <algorithm>
#include <string>

namespace bnk {

#define CUBLAS_CHECK(x)                                                                                    \
    do {                                                                                                   \
        cublasStatus_t s_ = (x);                                                                           \
        if (s_ != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(s_)); \
    } while (0)

__global__ void f32_to_f16_k(const float * x, half * y, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = __float2half(x[i]);
}
void f32_to_f16(const float * x, half * y, int64_t n, cudaStream_t s) {
    const int64_t b = (n + 255) / 256;
    f32_to_f16_k<<<(int) (b < 65535 * 4 ? b : 65535 * 4), 256, 0, s>>>(x, y, n);
}

// grid.x = rows (grid.y is capped at 65535; a chunk of 8192 tokens routes 81920 rows)
__global__ void gather_rows_f16_k(const float * x, int64_t cols, const int32_t * rows, half * y) {
    const int i = blockIdx.x;
    const float * src = x + (int64_t) rows[i] * cols;
    for (int64_t c = blockIdx.y * blockDim.x + threadIdx.x; c < cols; c += gridDim.y * blockDim.x)
        y[(int64_t) i * cols + c] = __float2half(src[c]);
}
void gather_rows_f16(const float * x, int64_t cols, const int32_t * rows, int n, half * y, cudaStream_t s) {
    if (n <= 0) return;
    gather_rows_f16_k<<<dim3(n, (unsigned) ((cols + 255) / 256)), 256, 0, s>>>(x, cols, rows, y);
}

__global__ void swiglu_f16_k(const half * gu, half * h, int F) {
    const int p = blockIdx.y;
    for (int f = blockIdx.x * blockDim.x + threadIdx.x; f < F; f += gridDim.x * blockDim.x) {
        const float g = __half2float(gu[(int64_t) p * 2 * F + f]);
        const float u = __half2float(gu[(int64_t) p * 2 * F + F + f]);
        h[(int64_t) p * F + f] = __float2half(g / (1.f + __expf(-g)) * u);
    }
}
void swiglu_f16(const half * gu, half * h, int P, int F, cudaStream_t s) {
    if (P <= 0) return;
    swiglu_f16_k<<<dim3((F + 255) / 256, P), 256, 0, s>>>(gu, h, F);
}

__global__ void moe_gather_sum_k(const half * D, const int32_t * inv, const float * w, int k, int E, const float * base,
                                 const float * scale, const float * extra, float * out) {
    const int t = blockIdx.y;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < E; e += gridDim.x * blockDim.x) {
        float acc = base ? base[(int64_t) t * E + e] * (scale ? scale[t] : 1.f) : 0.f;
        if (extra) acc += extra[(int64_t) t * E + e];
        for (int j = 0; j < k; ++j) acc += w[t * k + j] * __half2float(D[(int64_t) inv[t * k + j] * E + e]);
        out[(int64_t) t * E + e] = acc;
    }
}
void moe_gather_sum(const half * D, const int32_t * inv, const float * w, int N, int k, int E, const float * base,
                    const float * scale, const float * extra, float * out, cudaStream_t s) {
    moe_gather_sum_k<<<dim3((E + 255) / 256, N), 256, 0, s>>>(D, inv, w, k, E, base, scale, extra, out);
}

Gemm::~Gemm() {
    if (ptrs_) cudaFree(ptrs_);
    if (h_) cublasDestroy(h_);
    if (w16_) cudaFree(w16_);
    if (x16_) cudaFree(x16_);
}

void Gemm::init(cudaStream_t s, size_t max_w_elems, size_t max_x_elems) {
    st_ = s;
    CUBLAS_CHECK(cublasCreate(&h_));
    CUBLAS_CHECK(cublasSetStream(h_, s));
    CUBLAS_CHECK(cublasSetMathMode(h_, CUBLAS_TENSOR_OP_MATH));
    wcap_ = max_w_elems;
    xcap_ = max_x_elems;
    if (cudaMalloc(&w16_, wcap_ * 2) != cudaSuccess || cudaMalloc(&x16_, xcap_ * 2) != cudaSuccess)
        throw std::runtime_error("gemm: workspace allocation failed");
}

void Gemm::raw(const half * W, int M, int K, const half * X, int64_t ldx, int N, float * Y, int64_t ldy, float beta) {
    const float alpha = 1.f;
    // column-major view: Y^T (M x N) = W (row-major M x K, i.e. K x M col-major, transposed) . X^T (K x N)
    CUBLAS_CHECK(cublasGemmEx(h_, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, W, CUDA_R_16F, K, X, CUDA_R_16F,
                              (int) ldx, &beta, Y, CUDA_R_32F, (int) ldy, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void Gemm::run_h(const QMat & W, const half * X, int64_t ldx, int N, float * Y, int64_t ldy, bool accumulate) {
    if ((size_t) W.rows * W.cols > wcap_) throw std::runtime_error("gemm: weight larger than the workspace");
    dequant_rows_f16(W, 0, W.rows, w16_, st_);
    raw(w16_, (int) W.rows, (int) W.cols, X, ldx, N, Y, ldy, accumulate ? 1.f : 0.f);
}

void Gemm::run(const QMat & W, const float * X, int64_t ldx, int N, float * Y, int64_t ldy, bool accumulate) {
    if ((size_t) N * W.cols > xcap_) throw std::runtime_error("gemm: input larger than the workspace");
    if (ldx == W.cols) {
        f32_to_f16(X, x16_, (int64_t) N * W.cols, st_);
    } else {
        for (int i = 0; i < N; ++i) f32_to_f16(X + i * ldx, x16_ + (size_t) i * W.cols, W.cols, st_);
    }
    run_h(W, x16_, W.cols, N, Y, ldy, accumulate);
}

void Gemm::grouped_h(const std::vector<const half *> & W, const std::vector<const half *> & X,
                     const std::vector<half *> & Y, const std::vector<int> & N, int M, int K) {
    const int G = (int) W.size();
    if (G == 0) return;
    std::vector<cublasOperation_t> ta(G, CUBLAS_OP_T), tb(G, CUBLAS_OP_N);
    std::vector<int> m(G, M), k(G, K), lda(G, K), ldb(G, K), ldc(G, M), gs(G, 1);
    std::vector<float> alpha(G, 1.f), beta(G, 0.f);
    if (G > ptr_cap_) {
        if (ptrs_) cudaFree(ptrs_);
        ptr_cap_ = std::max(G, 256);
        if (cudaMalloc(&ptrs_, (size_t) 3 * ptr_cap_ * sizeof(void *)) != cudaSuccess)
            throw std::runtime_error("gemm: pointer arrays");
    }
    std::vector<const void *> host(3 * (size_t) G);
    for (int i = 0; i < G; ++i) {
        host[i] = W[i];
        host[G + i] = X[i];
        host[2 * G + i] = Y[i];
    }
    if (cudaMemcpyAsync(ptrs_, host.data(), host.size() * sizeof(void *), cudaMemcpyHostToDevice, st_) != cudaSuccess)
        throw std::runtime_error("gemm: pointer upload");
    CUBLAS_CHECK(cublasGemmGroupedBatchedEx(h_, ta.data(), tb.data(), m.data(), (int *) N.data(), k.data(),
                                            alpha.data(), (const void * const *) ptrs_, CUDA_R_16F, lda.data(),
                                            (const void * const *) (ptrs_ + G), CUDA_R_16F, ldb.data(), beta.data(),
                                            (void * const *) (ptrs_ + 2 * G), CUDA_R_16F, ldc.data(), G, gs.data(),
                                            CUBLAS_COMPUTE_32F));
    // the pageable `host` array must outlive the copy: cudaMemcpyAsync from pageable memory has returned only
    // after staging it, so it is safe to let it go here
}

}  // namespace bnk
