// The ceiling for the prompt path's expert GEMMs on this card: cuBLAS fp16 tensor-core GEMMs, batched over experts,
// at the shapes moe_tc_k computes (gate|up: [M tokens] x [5120] -> [2 x 2304]; down: [M] x [2304] -> [5120]), for the
// token counts an expert sees in a group. Prints TFLOPS per shape (docs/PREFILL-PLAN.md step 0.3).
//   build/bench_expert_gemm [experts=64]
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define CK(x) do { auto e_ = (x); if (e_ != 0) { fprintf(stderr, "%s failed: %d\n", #x, (int) e_); exit(1); } } while (0)
int main(int argc, char ** argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 64;
    cublasHandle_t h;
    CK(cublasCreate(&h));
    CK(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));
    struct Shape { const char * name; int N, K; };
    const Shape shapes[] = {{"gate|up", 2 * 2304, 5120}, {"down", 5120, 2304}};
    const int Ms[] = {8, 16, 32, 64, 128, 256, 512};
    const size_t wmax = (size_t) 2 * 2304 * 5120, xmax = (size_t) 512 * 5120, ymax = (size_t) 512 * 2 * 2304;
    half *W, *X;
    float *Y;
    CK(cudaMalloc(&W, wmax * B * 2)); CK(cudaMalloc(&X, xmax * B * 2)); CK(cudaMalloc(&Y, ymax * B * 4));
    CK(cudaMemset(W, 0, wmax * B * 2)); CK(cudaMemset(X, 0, xmax * B * 2));
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    const float one = 1.f, zero = 0.f;
    printf("%-8s %5s  %8s  %7s\n", "shape", "M", "ms", "TFLOPS");
    for (const Shape & s : shapes)
        for (int M : Ms) {
            auto run = [&] {
                // Y[M][N] = X[M][K] W[N][K]^T, column-major view: Y^T (N x M) = W (N x K, ld K, transposed) * X^T
                CK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.N, M, s.K, &one, W, CUDA_R_16F, s.K, (long long) s.N * s.K,
                                              X, CUDA_R_16F, s.K, (long long) M * s.K, &zero, Y, CUDA_R_32F, s.N, (long long) M * s.N, B,
                                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            };
            for (int i = 0; i < 3; ++i) run();
            cudaEventRecord(a);
            const int R = 10;
            for (int i = 0; i < R; ++i) run();
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            ms /= R;
            printf("%-8s %5d  %8.3f  %7.1f\n", s.name, M, ms, 2.0 * M * s.N * s.K * B / (ms * 1e9));
        }
    return 0;
}
