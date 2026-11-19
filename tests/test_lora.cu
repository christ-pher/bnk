// The dense engine's LoRA pieces against references, for windows of 1-16 rows:
//   lora_up (the fused B products) against a CPU fp64 product and against the float GEMV it replaces,
//   a base matrix with A's rows stacked under it (three-output GemvSplit) against the separate GEMVs,
//   A x through the quantized GEMV against fp64 on the same weights (the activation quantization's error).
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "dense/dense_ops.h"
#include "ggml.h"
#include "kernels/gemv.h"

using namespace bnk;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

static float * dev(const std::vector<float> & v) {
    float * d = nullptr;
    cudaMalloc(&d, v.size() * 4);
    cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
    return d;
}
static std::vector<float> host(const float * d, size_t n) {
    std::vector<float> v(n);
    cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost);
    return v;
}
static double rel(const std::vector<float> & a, const std::vector<double> & ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) num += (a[i] - ref[i]) * (a[i] - ref[i]), den += ref[i] * ref[i];
    return std::sqrt(num / std::max(den, 1e-30));
}
static double rel2(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]), den += (double) b[i] * b[i];
    return std::sqrt(num / std::max(den, 1e-30));
}

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    bool ok = true;

    // ---- lora_up: three B matrices (rows 640 / 128 / 128, rank 32) reading columns of a [T][96] product
    const int R = 32, W = 96, rows[3] = {640, 128, 128};
    QMat b[3];
    std::vector<std::vector<float>> bf(3);
    for (int i = 0; i < 3; ++i) {
        bf[i].resize((size_t) rows[i] * R);
        std::vector<ggml_fp16_t> h(bf[i].size());
        for (auto & v : bf[i]) v = nd(rng) * 0.05f;
        ggml_fp32_to_fp16_row(bf[i].data(), h.data(), (int64_t) h.size());
        ggml_fp16_to_fp32_row(h.data(), bf[i].data(), (int64_t) h.size());   // the reference uses the fp16 values
        void * d = nullptr;
        b[i] = upload_matrix(h.data(), GGML_TYPE_F16, rows[i], R, R * 2, false, &d);
    }
    const int off[3] = {0, 32, 64};
    for (int T = 1; T <= kMaxGemvRows; ++T) {
        std::vector<float> t((size_t) T * W), y0[3];
        for (auto & v : t) v = nd(rng);
        float * dt = dev(t);
        float * dy[3], * dg[3];
        int64_t ld[3];
        double worst = 0, worst_g = 0;
        for (int i = 0; i < 3; ++i) {
            ld[i] = rows[i] + 8;   // a row stride wider than the matrix, as the engine's outputs
            y0[i].resize((size_t) T * ld[i]);
            for (auto & v : y0[i]) v = nd(rng);
            dy[i] = dev(y0[i]);
            dg[i] = dev(y0[i]);
        }
        lora_up(b, dy, ld, off, 3, dt, T, W, s);
        for (int i = 0; i < 3; ++i) gemv(b[i], nullptr, dt + off[i], W, T, dg[i], ld[i], true, s);
        CK(cudaStreamSynchronize(s));
        for (int i = 0; i < 3; ++i) {
            std::vector<double> ref(y0[i].size());
            for (int tt = 0; tt < T; ++tt)
                for (int r = 0; r < ld[i]; ++r) {
                    double acc = y0[i][(size_t) tt * ld[i] + r];
                    if (r < rows[i])
                        for (int k = 0; k < R; ++k) acc += (double) bf[i][(size_t) r * R + k] * t[(size_t) tt * W + off[i] + k];
                    ref[(size_t) tt * ld[i] + r] = acc;
                }
            worst = std::max(worst, rel(host(dy[i], y0[i].size()), ref));
            worst_g = std::max(worst_g, rel(host(dg[i], y0[i].size()), ref));
            cudaFree(dy[i]);
            cudaFree(dg[i]);
        }
        cudaFree(dt);
        const bool good = worst < 1e-6;
        ok &= good;
        printf("lora_up T=%2d: rel err vs fp64 %.2e (float GEMV %.2e)%s\n", T, worst, worst_g, good ? "" : "  FAIL");
    }

    // ---- stacked: W [512 x 1024] | A [64 x 1024] (Q8_0), gate | up | A as three outputs vs separate GEMVs
    {
        const int K = 1024, M1 = 256, M2 = 256, MA = 64, M = M1 + M2 + MA;
        std::vector<float> w((size_t) M * K);
        for (int r = 0; r < M; ++r)
            for (int k = 0; k < K; ++k) w[(size_t) r * K + k] = nd(rng) * (r >= M1 + M2 ? 0.01f : 0.05f);
        const size_t rb = ggml_row_size(GGML_TYPE_Q8_0, K);
        std::vector<uint8_t> q(rb * M);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), q.data(), 0, M, K, nullptr);
        std::vector<float> wq((size_t) M * K);
        ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q.data(), wq.data(), (int64_t) M * K);
        void * d = nullptr;
        const QMat big = upload_matrix(q.data(), GGML_TYPE_Q8_0, M, K, rb, true, &d);
        printf("stacked matrix layout %d\n", big.layout);
        ActQ8 a;
        a.cols_pad = K;
        int8_t * aq; float * ad; int16_t * as; half * ah;
        CK(cudaMalloc(&aq, kMaxGemvRows * K)); CK(cudaMalloc(&ad, kMaxGemvRows * K / 32 * 4));
        CK(cudaMalloc(&as, kMaxGemvRows * K / 16 * 2)); CK(cudaMalloc(&ah, kMaxGemvRows * K * 2));
        a.q = aq; a.d = ad; a.s = as; a.h = ah;
        for (int T = 1; T <= kMaxGemvRows; ++T) {
            std::vector<float> x((size_t) T * K);
            for (auto & v : x) v = nd(rng);
            float * dx = dev(x);
            quantize_act(dx, K, T, K, a, s);
            std::vector<float> z((size_t) T * 512, 0.f);
            float * g = dev(z), * u = dev(z), * t = dev(z), * g2 = dev(z), * u2 = dev(z), * t2 = dev(z);
            GemvSplit sp{M1, u, M2};
            sp.r2 = M1 + M2;
            sp.y3 = t;
            sp.ldy3 = MA;
            gemv(big, &a, nullptr, 0, T, g, M1, false, s, sp);
            gemv(qmat_rows(big, 0, M1), &a, nullptr, 0, T, g2, M1, false, s);
            gemv(qmat_rows(big, M1, M2), &a, nullptr, 0, T, u2, M2, false, s);
            gemv(qmat_rows(big, M1 + M2, MA), &a, nullptr, 0, T, t2, MA, false, s);
            CK(cudaStreamSynchronize(s));
            const double eg = rel2(host(g, (size_t) T * M1), host(g2, (size_t) T * M1));
            const double eu = rel2(host(u, (size_t) T * M2), host(u2, (size_t) T * M2));
            const double et = rel2(host(t, (size_t) T * MA), host(t2, (size_t) T * MA));
            // A x against fp64 with the Q8_0 weights and the exact activations
            std::vector<double> ref((size_t) T * MA);
            for (int tt = 0; tt < T; ++tt)
                for (int r = 0; r < MA; ++r) {
                    double acc = 0;
                    for (int k = 0; k < K; ++k) acc += (double) wq[(size_t) (M1 + M2 + r) * K + k] * x[(size_t) tt * K + k];
                    ref[(size_t) tt * MA + r] = acc;
                }
            const double ea = rel(host(t, (size_t) T * MA), ref);
            const bool good = eg < 1e-6 && eu < 1e-6 && et < 1e-6;
            ok &= good;
            printf("stacked T=%2d: gate %.1e up %.1e A %.1e vs separate GEMVs; A x vs fp64 %.2e%s\n", T, eg, eu, et, ea,
                   good ? "" : "  FAIL");
            for (float * p : {dx, g, u, t, g2, u2, t2}) cudaFree(p);
        }
    }
    printf(ok ? "all passed\n" : "FAILED\n");
    return ok ? 0 : 1;
}
