// The GEMV paths the DeepSeek engine takes at every window size (1, 2, 4, 8 tokens), against a CPU reference on
// the dequantized weights: Q5_K and Q8_0 R-layout rows with int8 activations, F16 / F32 rows with fp32 input.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "core/util.h"
#include "ggml.h"
#include "kernels/gemv.h"

using namespace bnk;

static std::mt19937 rng(11);
static float rnd(float a = -1.f, float b = 1.f) { return std::uniform_real_distribution<float>(a, b)(rng); }
static int fails = 0;

static void run_case(const char * what, ggml_type type, int rows, int cols, bool via_act) {
    // a random fp32 matrix quantized to `type` (or kept float), dequantized back for the reference
    std::vector<float> w((size_t) rows * cols);
    for (auto & v : w) v = rnd();
    std::vector<uint8_t> q;
    std::vector<float> wd((size_t) rows * cols);
    const size_t rb = ggml_row_size(type, cols);
    q.resize(rb * rows);
    if (type == GGML_TYPE_F32) {
        memcpy(q.data(), w.data(), q.size());
        wd = w;
    } else if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(w.data(), (ggml_fp16_t *) q.data(), (int64_t) rows * cols);
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) q.data(), wd.data(), (int64_t) rows * cols);
    } else {
        ggml_quantize_chunk(type, w.data(), q.data(), 0, rows, cols, nullptr);
        ggml_get_type_traits(type)->to_float(q.data(), wd.data(), (int64_t) rows * cols);
    }
    void * dev = nullptr;
    QMat W = upload_matrix(q.data(), type, rows, cols, rb, true, &dev);
    const int Tmax = 8;
    std::vector<float> x((size_t) Tmax * cols);
    for (auto & v : x) v = rnd();
    float * d_x, * d_y;
    CUDA_CHECK(cudaMalloc(&d_x, x.size() * 4));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t) Tmax * rows * 4));
    // activations quantized (what the quantized paths read), or fp32 (the float paths)
    DevBuf<int8_t> aq; DevBuf<float> ad; DevBuf<int16_t> as; DevBuf<half> ah;
    const int64_t cp = (cols + 31) / 32 * 32;
    aq.alloc((size_t) Tmax * cp); ad.alloc((size_t) Tmax * cp / 32); as.alloc((size_t) Tmax * cp / 16); ah.alloc((size_t) Tmax * cp);
    ActQ8 a; a.q = aq; a.d = ad; a.s = as; a.h = ah;
    std::vector<double> ref((size_t) Tmax * rows);
    for (int t = 0; t < Tmax; ++t)
        for (int r = 0; r < rows; ++r) {
            double s = 0;
            for (int c = 0; c < cols; ++c) s += (double) wd[(size_t) r * cols + c] * x[(size_t) t * cols + c];
            ref[(size_t) t * rows + r] = s;
        }
    for (int T : {1, 2, 4, 8}) {
        CUDA_CHECK(cudaMemset(d_y, 0, (size_t) Tmax * rows * 4));
        if (via_act) {
            quantize_act(d_x, cols, T, cols, a, nullptr);
            gemv(W, &a, nullptr, 0, T, d_y, rows, false, nullptr);
        } else {
            gemv(W, nullptr, d_x, cols, T, d_y, rows, false, nullptr);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> y((size_t) T * rows);
        CUDA_CHECK(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost));
        double err = 0, mx = 0;
        for (size_t i = 0; i < y.size(); ++i) { err = std::max(err, std::fabs(y[i] - ref[i])); mx = std::max(mx, std::fabs(ref[i])); }
        // int8 activations: ~0.5% of the row scale; float paths: fp32-exact-ish
        const double tol = via_act ? 0.01 * mx : (type == GGML_TYPE_F16 ? 1e-3 : 1e-4) * mx;   // fp16 activations on the tensor cores from 4 tokens
        const bool ok = err <= tol;
        printf("%-10s %5dx%-5d T=%d  max err %.3e (ref max %.2f)%s\n", what, rows, cols, T, err, mx, ok ? "" : "  FAIL");
        if (!ok) ++fails;
    }
    cudaFree(dev); cudaFree(d_x); cudaFree(d_y);
}

// a stacked gate | up matrix read through GemvSplit: rows from r1 on go to a second output
static void run_split(int rows, int cols) {
    std::vector<float> w((size_t) 2 * rows * cols);
    for (auto & v : w) v = rnd();
    const size_t rb = ggml_row_size(GGML_TYPE_Q5_K, cols);
    std::vector<uint8_t> q(rb * 2 * rows);
    std::vector<float> wd((size_t) 2 * rows * cols);
    ggml_quantize_chunk(GGML_TYPE_Q5_K, w.data(), q.data(), 0, 2 * rows, cols, nullptr);
    ggml_get_type_traits(GGML_TYPE_Q5_K)->to_float(q.data(), wd.data(), (int64_t) 2 * rows * cols);
    void * dev = nullptr;
    QMat W = upload_matrix(q.data(), GGML_TYPE_Q5_K, 2 * rows, cols, rb, true, &dev);
    const int Tmax = 8;
    std::vector<float> x((size_t) Tmax * cols);
    for (auto & v : x) v = rnd();
    float * d_x, * d_g, * d_u;
    CUDA_CHECK(cudaMalloc(&d_x, x.size() * 4)); CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_g, (size_t) Tmax * rows * 4)); CUDA_CHECK(cudaMalloc(&d_u, (size_t) Tmax * rows * 4));
    DevBuf<int8_t> aq; DevBuf<float> ad; DevBuf<int16_t> as; DevBuf<half> ah;
    const int64_t cp = (cols + 31) / 32 * 32;
    aq.alloc((size_t) Tmax * cp); ad.alloc((size_t) Tmax * cp / 32); as.alloc((size_t) Tmax * cp / 16); ah.alloc((size_t) Tmax * cp);
    ActQ8 a; a.q = aq; a.d = ad; a.s = as; a.h = ah;
    for (int T : {1, 2, 4, 8}) {
        quantize_act(d_x, cols, T, cols, a, nullptr);
        gemv(W, &a, nullptr, 0, T, d_g, rows, false, nullptr, GemvSplit{rows, d_u, rows});
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> g((size_t) T * rows), u((size_t) T * rows);
        CUDA_CHECK(cudaMemcpy(g.data(), d_g, g.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(u.data(), d_u, u.size() * 4, cudaMemcpyDeviceToHost));
        double eg = 0, eu = 0, mx = 0;
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < rows; ++r) {
                double sg = 0, su = 0;
                for (int c = 0; c < cols; ++c) { sg += (double) wd[(size_t) r * cols + c] * x[(size_t) t * cols + c]; su += (double) wd[((size_t) rows + r) * cols + c] * x[(size_t) t * cols + c]; }
                eg = std::max(eg, std::fabs(g[(size_t) t * rows + r] - sg)); eu = std::max(eu, std::fabs(u[(size_t) t * rows + r] - su));
                mx = std::max({mx, std::fabs(sg), std::fabs(su)});
            }
        const bool ok = eg <= 0.01 * mx && eu <= 0.01 * mx;
        printf("Q5_K split %5dx%-5d T=%d  max err gate %.3e up %.3e (ref max %.2f)%s\n", rows, cols, T, eg, eu, mx, ok ? "" : "  FAIL");
        if (!ok) ++fails;
    }
    cudaFree(dev); cudaFree(d_x); cudaFree(d_g); cudaFree(d_u);
}

// the grouped output projection (GemvSplit::grp_rows): G groups of OL rows, group g over activation columns
// [g GW, (g+1) GW) of one quantized row of G GW columns, in one launch, against the reference
static void run_grouped(ggml_type type, int G, int OL, int GW) {
    const int rows = G * OL, cols = GW;
    std::vector<float> w((size_t) rows * cols);
    for (auto & v : w) v = rnd();
    const size_t rb = ggml_row_size(type, cols);
    std::vector<uint8_t> q(rb * rows);
    std::vector<float> wd((size_t) rows * cols);
    ggml_quantize_chunk(type, w.data(), q.data(), 0, rows, cols, nullptr);
    ggml_get_type_traits(type)->to_float(q.data(), wd.data(), (int64_t) rows * cols);
    void * dev = nullptr;
    QMat W = upload_matrix(q.data(), type, rows, cols, rb, true, &dev);
    const int Tmax = 8;
    const int64_t XC = (int64_t) G * GW;
    std::vector<float> x((size_t) Tmax * XC);
    for (auto & v : x) v = rnd();
    float * d_x, * d_y;
    CUDA_CHECK(cudaMalloc(&d_x, x.size() * 4)); CUDA_CHECK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t) Tmax * rows * 4));
    DevBuf<int8_t> aq; DevBuf<float> ad; DevBuf<int16_t> as; DevBuf<half> ah;
    aq.alloc((size_t) Tmax * XC); ad.alloc((size_t) Tmax * XC / 32); as.alloc((size_t) Tmax * XC / 16); ah.alloc((size_t) Tmax * XC);
    ActQ8 a; a.q = aq; a.d = ad; a.s = as; a.h = ah;
    for (int T : {1, 2, 4, 8}) {
        quantize_act(d_x, XC, T, XC, a, nullptr);
        ActQ8 av = a;
        av.cols = GW;
        GemvSplit sp;
        sp.grp_rows = OL;
        sp.grp_cols = GW;
        gemv(W, &av, nullptr, 0, T, d_y, rows, false, nullptr, sp);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> y((size_t) T * rows);
        CUDA_CHECK(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost));
        double err = 0, mx = 0;
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < rows; ++r) {
                const int g = r / OL;
                double s = 0;
                for (int c = 0; c < cols; ++c) s += (double) wd[(size_t) r * cols + c] * x[(size_t) t * XC + (size_t) g * GW + c];
                err = std::max(err, std::fabs(y[(size_t) t * rows + r] - s));
                mx = std::max(mx, std::fabs(s));
            }
        const bool ok = err <= 0.01 * mx;
        printf("%s grouped %dx%d rows x %d cols T=%d  max err %.3e (ref max %.2f)%s\n", ggml_type_name(type), G, OL, GW, T, err, mx,
               ok ? "" : "  FAIL");
        if (!ok) ++fails;
    }
    cudaFree(dev); cudaFree(d_x); cudaFree(d_y);
}

int main() {
    run_grouped(GGML_TYPE_Q8_0, 8, 64, 256);
    run_grouped(GGML_TYPE_Q5_K, 8, 48, 512);

    run_split(2304, 5120);
    run_case("Q5_K", GGML_TYPE_Q5_K, 1024, 5120, true);
    run_case("Q5_K", GGML_TYPE_Q5_K, 2048, 1280, true);
    run_case("Q5_K", GGML_TYPE_Q5_K, 512, 2304, true);
    run_case("Q8_0", GGML_TYPE_Q8_0, 1280, 5120, true);
    run_case("Q8_0", GGML_TYPE_Q8_0, 512, 5120, true);
    run_case("F16", GGML_TYPE_F16, 512, 5120, false);
    run_case("F16", GGML_TYPE_F16, 4096, 1280, false);
    run_case("F16", GGML_TYPE_F16, 32, 5120, false);
    run_case("F16", GGML_TYPE_F16, 128, 512, false);
    run_case("F32", GGML_TYPE_F32, 384, 5120, false);
    printf(fails ? "\n%d FAILURES\n" : "\nall ok\n", fails);
    return fails ? 1 : 0;
}
