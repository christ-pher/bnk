// The prompt path's dense GEMM shapes through cuBLAS: fp16 inputs with fp32 outputs (what Gemm::raw does) against
// fp16 outputs and cublasLt with a workspace, per shape at N tokens. Prints TFLOPS.
//   build/bench_gemm [N=4096]
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(x) do { auto e_ = (x); if (e_ != 0) { fprintf(stderr, "%s failed: %d\n", #x, (int) e_); exit(1); } } while (0)

int main(int argc, char ** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 4096;
    struct Shape { const char * name; int M, K; };
    const Shape shapes[] = {{"attn_qkv (GDN)", 10240, 2560}, {"wgate", 6144, 2560}, {"ssm_out", 2560, 6144},
                            {"attn_q", 12288, 2560}, {"hc_down", 320, 10240}, {"hc_up", 10240, 320},
                            {"router", 512, 2560}, {"sh_gate", 640, 2560}, {"sh_down", 2560, 640}};
    cublasHandle_t h;
    CK(cublasCreate(&h));
    CK(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));
    cublasLtHandle_t lt;
    CK(cublasLtCreate(&lt));
    void * ws;
    const size_t wsz = 64 << 20;
    CK(cudaMalloc(&ws, wsz));
    half * W; half * X; float * Y; half * Yh;
    CK(cudaMalloc(&W, (size_t) 12288 * 10240 * 2));
    CK(cudaMalloc(&X, (size_t) N * 10240 * 2));
    CK(cudaMalloc(&Y, (size_t) N * 12288 * 4));
    CK(cudaMalloc(&Yh, (size_t) N * 12288 * 2));
    CK(cudaMemset(W, 0, (size_t) 12288 * 10240 * 2));
    CK(cudaMemset(X, 0, (size_t) N * 10240 * 2));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const float alpha = 1.f, beta = 0.f;
    auto time = [&](auto && f) {
        for (int i = 0; i < 3; ++i) f();
        cudaEventRecord(e0);
        const int reps = 10;
        for (int i = 0; i < reps; ++i) f();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms / reps;
    };
    printf("N = %d tokens\n%-16s %8s %8s %8s   (ms, TFLOPS)\n", N, "shape", "f32 out", "f16 out", "Lt f32");
    for (const Shape & s : shapes) {
        const int M = s.M, K = s.K;
        const double flop = 2.0 * M * N * K;
        const float t32 = time([&]() {
            CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, W, CUDA_R_16F, K, X, CUDA_R_16F, K, &beta, Y,
                            CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        });
        const float t16 = time([&]() {
            CK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, W, CUDA_R_16F, K, X, CUDA_R_16F, K, &beta, Yh,
                            CUDA_R_16F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        });
        // cublasLt with the heuristic's best algorithm and a workspace
        cublasLtMatmulDesc_t op;
        CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
        CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
        CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
        cublasLtMatrixLayout_t la, lb, lc;
        CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, K, M, K));
        CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, N, K));
        CK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, M, N, M));
        cublasLtMatmulPreference_t pref;
        CK(cublasLtMatmulPreferenceCreate(&pref));
        CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
        cublasLtMatmulHeuristicResult_t res[8];
        int nres = 0;
        CK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, res, &nres));
        float tlt = -1.f;
        int best = -1;
        for (int i = 0; i < nres; ++i) {
            const float t = time([&]() {
                CK(cublasLtMatmul(lt, op, &alpha, W, la, X, lb, &beta, Y, lc, Y, lc, &res[i].algo, ws, wsz, 0));
            });
            if (tlt < 0 || t < tlt) { tlt = t; best = i; }
        }
        printf("%-16s %5.2f %5.0f  %5.2f %5.0f  %5.2f %5.0f  (Lt algo %d of %d)\n", s.name, t32, flop / t32 / 1e9, t16,
               flop / t16 / 1e9, tlt, flop / tlt / 1e9, best, nres);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
        cublasLtMatmulDescDestroy(op);
    }
    return 0;
}
