// For every weight format in the given models: GPU dequantization must equal ggml's CPU to_float,
// and the dp4a/float GEMV must match a double-precision reference within activation-quantization error.
//   test_dequant model-00001-of-0000N.gguf [more.gguf ...]
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "core/gguf_model.h"
#include "ggml-cpu.h"
#include "kernels/gemv.h"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

using namespace bnk;

static int test_tensor(const TensorRef & t) {
    const int64_t cols = t.ne[0];
    const int64_t rows = std::min<int64_t>(t.nrows(), 512);
    const size_t rb = t.row_bytes();
    QMat W{nullptr, (int) t.type, rows, cols, rb};
    void * dw;
    CK(cudaMalloc(&dw, rows * rb));
    CK(cudaMemcpy(dw, t.data, rows * rb, cudaMemcpyHostToDevice));
    W.data = dw;

    // 1. dequantization vs ggml
    std::vector<float> ref(rows * cols), got(rows * cols);
    const auto * tr = ggml_get_type_traits(t.type);
    for (int64_t r = 0; r < rows; ++r) {
        if (t.type == GGML_TYPE_F32) memcpy(&ref[r * cols], t.data + r * rb, cols * 4);
        else tr->to_float(t.data + r * rb, &ref[r * cols], cols);
    }
    float * dout;
    CK(cudaMalloc(&dout, rows * cols * 4));
    dequant_rows(W, 0, rows, dout, 0);
    CK(cudaMemcpy(got.data(), dout, rows * cols * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0;
    int64_t bad = 0;
    for (int64_t i = 0; i < rows * cols; ++i) {
        double e = fabs((double) got[i] - ref[i]) / (fabs(ref[i]) + 1e-6);
        if (e > 1e-5) ++bad;
        maxrel = std::max(maxrel, e);
    }

    // 2. GEMV vs double reference on the dequantized weights
    const int T = 5;
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(T * cols);
    for (auto & v : x) v = nd(rng);
    float *dx, *dy;
    CK(cudaMalloc(&dx, x.size() * 4));
    CK(cudaMalloc(&dy, T * rows * 4));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    ActQ8 a;
    int64_t cp = (cols + 31) / 32 * 32;
    CK(cudaMalloc(&a.q, T * cp));
    CK(cudaMalloc(&a.d, T * cp / 32 * 4));
    gemv_auto(W, dx, cols, T, dy, rows, false, a, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> y(T * rows);
    CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (int tt = 0; tt < T; ++tt)
        for (int64_t r = 0; r < rows; ++r) {
            double s = 0;
            for (int64_t c = 0; c < cols; ++c) s += (double) ref[r * cols + c] * x[tt * cols + c];
            num += (y[tt * rows + r] - s) * (y[tt * rows + r] - s);
            den += s * s;
        }
    const double gerr = sqrt(num / (den + 1e-30));
    const bool ok = bad == 0 && gerr < 2e-2;
    printf("%-8s %-40s %6lld x %-6lld dequant max rel %.2e (%lld bad)  gemv rel rms %.2e  %s\n",
           ggml_type_name(t.type), t.name.c_str(), (long long) rows, (long long) cols, maxrel, (long long) bad, gerr,
           ok ? "OK" : "FAIL");
    cudaFree(dw); cudaFree(dout); cudaFree(dx); cudaFree(dy); cudaFree(a.q); cudaFree(a.d);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    ggml_cpu_init();
    int fails = 0;
    for (int i = 1; i < argc; ++i) {
        GgufModel m;
        m.open(argv[i]);
        std::map<int, const TensorRef *> one;  // first tensor of each format (skip PLE table / huge ones first)
        for (const auto & t : m.tensors()) {
            if (t.name.find("per_layer_token_embd") != std::string::npos) continue;
            if (t.ne[0] % 32 || t.ne[0] < 64) continue;
            if (!one.count(t.type)) one[t.type] = &t;
        }
        printf("== %s\n", argv[i]);
        for (auto & [ty, t] : one) fails += test_tensor(*t);
    }
    printf(fails ? "FAILURES: %d\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
