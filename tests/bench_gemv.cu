// GEMV bandwidth on real model matrices: bench_gemv MODEL.gguf [tensor names...]
#include <cstdio>
#include <string>
#include <vector>

#include "core/gguf_model.h"
#include "core/util.h"
#include "kernels/gemv.h"

using namespace bnk;

int main(int argc, char ** argv) {
    GgufModel g;
    g.open(argv[1]);
    std::vector<std::string> names;
    for (int i = 2; i < argc; ++i) names.push_back(argv[i]);
    if (names.empty())
        names = {"blk.0.attn_qkv.weight", "blk.0.ssm_out.weight", "blk.0.attn_gate.weight", "blk.0.hc_attn_down.weight",
                 "blk.0.hc_attn_up.weight", "blk.0.hc_attn_inject.weight", "blk.0.ffn_gate_inp.weight",
                 "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_down_shexp.weight", "output.weight"};
    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, 8 * 16384 * 4));
    CUDA_CHECK(cudaMalloc(&dy, 8 * 262144 * 4));
    CUDA_CHECK(cudaMemset(dx, 0, 8 * 16384 * 4));
    ActQ8 a;
    CUDA_CHECK(cudaMalloc(&a.q, 8 * 16384));
    CUDA_CHECK(cudaMalloc(&a.d, 8 * 16384 / 32 * 4));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (const auto & n : names) {
        const TensorRef * t = g.find(n);
        if (!t) continue;
        void * dw;
        QMat W = upload_matrix(t->data, (int) t->type, t->nrows(), t->ne[0], t->row_bytes(), !getenv("BNK_NO_R"), &dw);
        printf("%s%-32s %-6s %6lld x %-6lld %7.2f MB:", W.layout ? "R " : "  ", n.c_str(), ggml_type_name(t->type), (long long) W.rows,
               (long long) W.cols, t->nbytes / 1e6);
        for (int T : {1, 2, 4, 8}) {
            for (int w = 0; w < 3; ++w) gemv_auto(W, dx, W.cols, T, dy, W.rows, false, a, 0);
            const int reps = 50;
            cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) gemv_auto(W, dx, W.cols, T, dy, W.rows, false, a, 0);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            const double us = ms * 1000 / reps;
            printf("  T%d %6.1fus %4.0fGB/s", T, us, t->nbytes / (us * 1e3));
        }
        printf("\n");
        cudaFree(dw);
    }
    return 0;
}
