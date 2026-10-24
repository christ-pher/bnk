// The DeepSeek attention kernel on a prompt chunk's shape: T queries x 64 heads over a 128-key window plus 512 selected
// compressed rows (random indices), fp16 keys. Prints ms per call.   bench_ds_attn [T=2048] [reps=5]
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_fp16.h>
#include "deepseek/ds_ops.h"
#include "core/util.h"
using namespace bnk;
int main(int argc, char ** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 2048, reps = argc > 2 ? atoi(argv[2]) : 5;
    const int H = 64, D = 512, W = 128, R = 64, k_sel = 512, n_comp = 4096;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q((size_t) T * H * D), sinks(H, 0.f);
    for (auto & v : q) v = nd(rng);
    std::vector<half> win((size_t) (T + W) * D), comp((size_t) n_comp * D);
    for (auto & v : win) v = __float2half(nd(rng));
    for (auto & v : comp) v = __float2half(nd(rng));
    std::vector<int32_t> idx((size_t) T * k_sel);
    for (int t = 0; t < T; ++t) for (int j = 0; j < k_sel; ++j) idx[(size_t) t * k_sel + j] = rng() % n_comp;
    float * d_q, * d_out, * d_sinks, * d_freq; half * d_win, * d_comp; int32_t * d_idx; int * d_pos0;
    CUDA_CHECK(cudaMalloc(&d_q, q.size() * 4)); CUDA_CHECK(cudaMalloc(&d_out, q.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_sinks, H * 4)); CUDA_CHECK(cudaMalloc(&d_freq, R / 2 * 4));
    CUDA_CHECK(cudaMalloc(&d_win, win.size() * 2)); CUDA_CHECK(cudaMalloc(&d_comp, comp.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_idx, idx.size() * 4)); CUDA_CHECK(cudaMalloc(&d_pos0, 4));
    CUDA_CHECK(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sinks, sinks.data(), H * 4, cudaMemcpyHostToDevice));
    auto fr = ds_rope_inv_freq(R, 1e4f, 1.f, 0, 32.f, 1.f);
    CUDA_CHECK(cudaMemcpy(d_freq, fr.data(), fr.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_win, win.data(), win.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_comp, comp.data(), comp.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
    const int pos0 = 8192, origin = pos0 - W;
    CUDA_CHECK(cudaMemcpy(d_pos0, &pos0, 4, cudaMemcpyHostToDevice));
    cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
    for (int pass = 0; pass < 2; ++pass) {
        const bool sel = pass == 1;
        ds_attention(d_q, d_win, sel ? d_comp : nullptr, sel ? d_idx : nullptr, sel ? k_sel : 0, d_sinks, d_out, T, H, D, W, R, d_pos0, d_freq,
                     1.f / sqrtf((float) D), st, origin);
        CUDA_CHECK(cudaStreamSynchronize(st));
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, st);
        for (int i = 0; i < reps; ++i)
            ds_attention(d_q, d_win, sel ? d_comp : nullptr, sel ? d_idx : nullptr, sel ? k_sel : 0, d_sinks, d_out, T, H, D, W, R, d_pos0,
                         d_freq, 1.f / sqrtf((float) D), st, origin);
        cudaEventRecord(e1, st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        const double keys = sel ? W + k_sel : W, flops = 2.0 * 2 * T * H * keys * D;
        printf("T=%d, %s: %.2f ms per call (%.1f TFLOPS)\n", T, sel ? "window + 512 selected" : "window only", ms / reps,
               flops / (ms / reps / 1000) / 1e12);
    }
    return 0;
}
