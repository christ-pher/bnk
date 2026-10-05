// Sparse prompt attention (QSA) on real block selections: times the prompt kernels and checks them against the
// one-row reference.
//
//   build/bench_qsa_attn SEL_DUMP [rows=8192] [first_row=60000]
//
// SEL_DUMP is what BNK_DUMP_SEL writes during `bnk pdump` (per row: position, n_sel, top_blocks block ids). Queries,
// keys, values and gates are random; the selections and positions are the model's, so the gather pattern and the
// overlap between neighbouring rows are real.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "core/util.h"
#include "kernels/qsa.h"

using namespace bnk;

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: bench_qsa_attn SEL_DUMP [rows] [first_row]\n");
        return 1;
    }
    const int N = argc > 2 ? atoi(argv[2]) : 8192;
    const long first = argc > 3 ? atol(argv[3]) : 60000;
    // qwen4exp attention: 24 heads, 2 KV heads, head dim 256; QSA r = 4, top_k 2048 -> 512 blocks
    QsaShape sh{24, 2, 256, 4, 128, 4, 512, 64, 1e7f, 1e-6f};
    const int H = sh.H, Hkv = sh.Hkv, D = sh.D, K = sh.top_blocks;
    FILE * f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    const size_t rec = (size_t) (K + 2);
    std::vector<int32_t> raw((size_t) N * rec);
    fseek(f, (long) (first * rec * 4), SEEK_SET);
    if (fread(raw.data(), 4, raw.size(), f) != raw.size()) { fprintf(stderr, "dump too short\n"); return 1; }
    fclose(f);
    const int pos0 = raw[0];
    std::vector<int32_t> sel((size_t) N * K), nsel(N);
    for (int t = 0; t < N; ++t) {
        if (raw[t * rec] != pos0 + t) { fprintf(stderr, "rows are not consecutive at %d\n", t); return 1; }
        nsel[t] = raw[t * rec + 1];
        for (int k = 0; k < K; ++k) sel[(size_t) t * K + k] = raw[t * rec + 2 + k];
    }
    const int ctx = pos0 + N;
    printf("rows %d at positions %d..%d\n", N, pos0, ctx - 1);

    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<half> kv((size_t) ctx * Hkv * D);
    std::vector<float> q((size_t) N * H * D), qfull((size_t) N * H * 2 * D);
    half *dk, *dv;
    float *dq, *dqf, *dref, *dout;
    int32_t *dsel, *dnsel;
    CUDA_CHECK(cudaMalloc(&dk, kv.size() * 2));
    CUDA_CHECK(cudaMalloc(&dv, kv.size() * 2));
    for (auto & x : kv) x = __float2half(nd(rng));
    CUDA_CHECK(cudaMemcpy(dk, kv.data(), kv.size() * 2, cudaMemcpyHostToDevice));
    for (auto & x : kv) x = __float2half(nd(rng));
    CUDA_CHECK(cudaMemcpy(dv, kv.data(), kv.size() * 2, cudaMemcpyHostToDevice));
    for (auto & x : q) x = nd(rng) * 0.6f;   // scores with a realistic spread (some keys dominate)
    for (auto & x : qfull) x = nd(rng);
    CUDA_CHECK(cudaMalloc(&dq, q.size() * 4));
    CUDA_CHECK(cudaMalloc(&dqf, qfull.size() * 4));
    CUDA_CHECK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dqf, qfull.data(), qfull.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dsel, sel.size() * 4));
    CUDA_CHECK(cudaMalloc(&dnsel, nsel.size() * 4));
    CUDA_CHECK(cudaMemcpy(dsel, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dnsel, nsel.data(), nsel.size() * 4, cudaMemcpyHostToDevice));
    const size_t on = (size_t) N * H * D;
    CUDA_CHECK(cudaMalloc(&dref, on * 4));
    CUDA_CHECK(cudaMalloc(&dout, on * 4));
    const float scale = 1.f / sqrtf((float) D);
    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // the reference: the one-warp-per-(row, head) kernel
    qsa_attention_prefill_ref(dq, dk, dv, dqf, dsel, dnsel, dref, 0, N, sh, pos0, scale, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    std::vector<float> ref(on), out(on);
    CUDA_CHECK(cudaMemcpy(ref.data(), dref, on * 4, cudaMemcpyDeviceToHost));

    // 4 useful MACs per (row, head, key, dim): QK and PV
    double flops = 0;
    for (int t = 0; t < N; ++t) {
        const int qp = pos0 + t, ts = (qp + 1) / sh.ratio * sh.ratio;
        flops += 4.0 * H * D * (nsel[t] ? nsel[t] * sh.ratio + (qp + 1 - ts) : qp + 1);
    }
    auto run = [&](const char * name, auto && launch) {
        CUDA_CHECK(cudaMemset(dout, 0, on * 4));
        launch();
        CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(out.data(), dout, on * 4, cudaMemcpyDeviceToHost));
        double md = 0, mr = 0;
        for (size_t i = 0; i < on; ++i) {
            md = std::max(md, (double) fabsf(out[i] - ref[i]));
            mr = std::max(mr, (double) fabsf(ref[i]));
        }
        const int reps = 5;
        cudaEventRecord(e0, st);
        for (int i = 0; i < reps; ++i) launch();
        cudaEventRecord(e1, st);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        ms /= reps;
        printf("%-34s %8.2f ms  %5.2f TFLOP/s  %6.1f us/row  max |diff| %.2e (max |out| %.2f)\n", name, ms,
               flops / (ms * 1e9), ms * 1000 / N, md, mr);
    };
    run("tensor cores, 64 rows per launch", [&]() {
        for (int t0 = 0; t0 < N; t0 += 64) {
            const int t1 = std::min(N, t0 + 64);
            qsa_attention_prefill(dq, dk, dv, dqf, dsel + (size_t) t0 * K, dnsel + t0, dout, t0, t1, sh, pos0, scale, st);
        }
    });
    run("tensor cores, all rows in one launch", [&]() {
        qsa_attention_prefill(dq, dk, dv, dqf, dsel, dnsel, dout, 0, N, sh, pos0, scale, st);
    });
    for (int R : {1, 2, 4}) {
        char name[64];
        snprintf(name, sizeof name, "multi-row R=%d, one launch", R);
        run(name, [&]() { qsa_attention_prefill_multi(R, dq, dk, dv, dqf, dsel, dnsel, dout, 0, N, sh, pos0, scale, st); });
    }
    return 0;
}
