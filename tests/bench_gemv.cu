// GEMV bandwidth on real model matrices: bench_gemv MODEL.gguf [tensor names...]
// Also the error of each path against fp32 (dequantized weights x fp32 activations), at 8 tokens.
// BNK_GEMV_MMA_MIN=0: the dp4a kernels only (no tensor cores)
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "core/gguf_model.h"
#include "ggml.h"
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
    {   // activations like a normalized hidden state
        std::vector<float> hx(8 * 16384);
        std::mt19937 rng(3);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & v : hx) v = nd(rng);
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    }
    ActQ8 a;
    CUDA_CHECK(cudaMalloc(&a.q, 8 * 16384));
    CUDA_CHECK(cudaMalloc(&a.d, 8 * 16384 / 32 * 4));
    CUDA_CHECK(cudaMalloc(&a.s, 8 * 16384 / 16 * 2));
    CUDA_CHECK(cudaMalloc(&a.h, 8 * 16384 * 2));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (const auto & n : names) {
        const TensorRef * t = g.find(n);
        if (!t) continue;
        void * dw;
        QMat W;
        if (t->type == GGML_TYPE_BF16 && !getenv("BNK_KEEP_BF16")) {   // as the engine loads it: F16
            const int64_t n = t->nrows() * t->ne[0];
            std::vector<float> f(n);
            std::vector<ggml_fp16_t> h(n);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) t->data, f.data(), n);
            ggml_fp32_to_fp16_row(f.data(), h.data(), n);
            W = upload_matrix(h.data(), GGML_TYPE_F16, t->nrows(), t->ne[0], t->ne[0] * 2, false, &dw);
        } else {
            W = upload_matrix(t->data, (int) t->type, t->nrows(), t->ne[0], t->row_bytes(), !getenv("BNK_NO_R"), &dw);
        }
        printf("%s%-32s %-6s %6lld x %-6lld %7.2f MB:", W.layout ? "R " : "  ", n.c_str(), ggml_type_name(t->type), (long long) W.rows,
               (long long) W.cols, t->nbytes / 1e6);
        if (W.cols <= 16384 && W.rows <= 16384) {   // error at 8 tokens against fp32
            float * dref;
            CUDA_CHECK(cudaMalloc(&dref, W.rows * W.cols * 4));
            dequant_rows(W, 0, W.rows, dref, 0);
            std::vector<float> hw(W.rows * W.cols), hx(8 * W.cols), hy(8 * W.rows);
            CUDA_CHECK(cudaMemcpy(hw.data(), dref, hw.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hx.data(), dx, hx.size() * 4, cudaMemcpyDeviceToHost));
            gemv_auto(W, dx, W.cols, 8, dy, W.rows, false, a, 0);
            CUDA_CHECK(cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost));
            double num = 0, den = 0;
            for (int t = 0; t < 8; ++t)
                for (int64_t r = 0; r < W.rows; ++r) {
                    double ref = 0;
                    for (int64_t c = 0; c < W.cols; ++c) ref += (double) hw[r * W.cols + c] * hx[t * W.cols + c];
                    num += (hy[t * W.rows + r] - ref) * (hy[t * W.rows + r] - ref);
                    den += ref * ref;
                }
            printf(" err8 %.2e |", std::sqrt(num / den));
            cudaFree(dref);
        }
        for (int T : {1, 2, 3, 4, 5, 8}) {
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
