// Q2_0 AVX2 dot vs exact float math, and vs ggml's own Q2_0 x Q8_0 path.
#include <algorithm>
#include <cmath>
#include <string>
#include <cstdio>
#include <random>
#include <vector>

#include "core/gguf_model.h"
#include "cpu/kernels.h"
#include "ggml-cpu.h"

using namespace bnk;

int test_mdot(const char * path);
void bench_mdot(const char * path);
int main(int argc, char ** argv) {
    ggml_cpu_init();
    if (getenv("BENCH")) { for (int i = 1; i < argc; ++i) bench_mdot(argv[i]); return 0; }
    if (argc > 2) {
        int f = 0;
        for (int i = 2; i < argc; ++i) f += test_mdot(argv[i]);
        printf(f ? "mdot FAILURES\n" : "mdot all OK\n");
    }
    GgufModel g;
    g.open(argv[1]);
    const TensorRef & t = g.get("blk.1.ffn_down_exps.weight");
    const int n = (int) t.ne[0];
    const size_t rb = t.row_bytes();
    std::mt19937 rng(7);
    std::student_t_distribution<float> sd(3.f);  // heavy tails like silu(g)*u
    std::vector<float> x(n), wrow(n);
    double e_mine = 0, e_ggml = 0, den = 0;
    const auto * tr = ggml_get_type_traits_cpu(GGML_TYPE_Q2_0);
    std::vector<uint8_t> xq(ggml_row_size(tr->vec_dot_type, n));
    std::vector<A8P64> a(n / 64);
    for (int rep = 0; rep < 200; ++rep) {
        for (auto & v : x) v = sd(rng);
        quantize_a8p64(x.data(), n, a.data());
        ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(x.data(), xq.data(), n);
        for (int r = 0; r < 64; ++r) {
            const uint8_t * row = t.data + (size_t) (rep * 64 + r) * rb;
            ggml_get_type_traits(GGML_TYPE_Q2_0)->to_float(row, wrow.data(), n);
            double ex = 0;
            for (int i = 0; i < n; ++i) ex += (double) wrow[i] * x[i];
            const float mine = dot_q2_0(row, a.data(), n);
            float gg;
            tr->vec_dot(n, &gg, 0, row, 0, xq.data(), 0, 1);
            e_mine += (mine - ex) * (mine - ex);
            e_ggml += (gg - ex) * (gg - ex);
            den += ex * ex;
        }
    }
    printf("Q2_0 dot rel rms error: bnk A8P64 %.3e   ggml (%s) %.3e\n", sqrt(e_mine / den),
           ggml_type_name(tr->vec_dot_type), sqrt(e_ggml / den));
    return 0;
}

// multi-token kernels vs ggml vec_dot, on real expert rows of each supported format
#include "cpu/mdot.h"
int test_mdot(const char * path) {
    GgufModel g;
    g.open(path);
    int fails = 0;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd;
    for (const auto & t : g.tensors()) {
        if (t.name.find("_exps") == std::string::npos || !mdot_supported(t.type)) continue;
        static std::vector<int> seen;
        if (std::find(seen.begin(), seen.end(), (int) t.type) != seen.end()) continue;
        seen.push_back(t.type);
        const int n = (int) t.ne[0];
        const auto * tr = ggml_get_type_traits_cpu(t.type);
        const auto * ty = ggml_get_type_traits_cpu(tr->vec_dot_type);
        const size_t yrb = ggml_row_size(tr->vec_dot_type, n), rb = t.row_bytes();
        double maxd = 0;
        for (int T = 1; T <= 7; ++T) {
            std::vector<std::vector<uint8_t>> yq(T, std::vector<uint8_t>(yrb));
            std::vector<const void *> yp(T);
            std::vector<float> x(n);
            for (int k = 0; k < T; ++k) {
                for (auto & v : x) v = nd(rng);
                ty->from_float(x.data(), yq[k].data(), n);
                yp[k] = yq[k].data();
            }
            for (int r = 0; r < 32; ++r) {
                const uint8_t * row = t.data + (size_t) (r * 97 % t.nrows()) * rb;
                float out[8];
                mdot(t.type, n, row, yp.data(), T, out);
                for (int k = 0; k < T; ++k) {
                    float ref;
                    tr->vec_dot(n, &ref, 0, row, 0, yp[k], 0, 1);
                    maxd = std::max(maxd, (double) fabsf(out[k] - ref) / (fabsf(ref) + 1e-3));
                }
            }
        }
        const bool ok = maxd < 1e-4;
        printf("mdot %-8s %-36s max rel diff vs ggml %.2e %s\n", ggml_type_name(t.type), t.name.c_str(), maxd, ok ? "OK" : "FAIL");
        fails += !ok;
    }
    return fails;
}

#include <chrono>
void bench_mdot(const char * path) {
    GgufModel g;
    g.open(path);
    std::vector<int> seen;
    for (const auto & t : g.tensors()) {
        if (t.name.find("_exps") == std::string::npos || !mdot_supported(t.type)) continue;
        if (std::find(seen.begin(), seen.end(), (int) t.type) != seen.end()) continue;
        seen.push_back(t.type);
        const int n = (int) t.ne[0];
        const auto * tr = ggml_get_type_traits_cpu(t.type);
        const auto * ty = ggml_get_type_traits_cpu(tr->vec_dot_type);
        const size_t yrb = ggml_row_size(tr->vec_dot_type, n), rb = t.row_bytes();
        std::vector<std::vector<uint8_t>> yq(4, std::vector<uint8_t>(yrb));
        std::vector<const void *> yp(4);
        std::vector<float> x(n, 0.5f);
        for (int k = 0; k < 4; ++k) { ty->from_float(x.data(), yq[k].data(), n); yp[k] = yq[k].data(); }
        const int rows = 640;
        auto time = [&](auto fn) {
            fn();
            const auto a = std::chrono::steady_clock::now();
            for (int rep = 0; rep < 20; ++rep) fn();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count() / 20 / rows * 1000;
        };
        float out[4], acc = 0;
        const double g1 = time([&] { for (int r = 0; r < rows; ++r) { tr->vec_dot(n, out, 0, t.data + r * rb, 0, yp[0], 0, 1); acc += out[0]; } });
        const double m1 = time([&] { for (int r = 0; r < rows; ++r) { mdot(t.type, n, t.data + r * rb, yp.data(), 1, out); acc += out[0]; } });
        const double g4 = time([&] { for (int r = 0; r < rows; ++r) for (int k = 0; k < 4; ++k) { tr->vec_dot(n, out, 0, t.data + r * rb, 0, yp[k], 0, 1); acc += out[0]; } });
        const double m4 = time([&] { for (int r = 0; r < rows; ++r) { mdot(t.type, n, t.data + r * rb, yp.data(), 4, out); acc += out[0]; } });
        printf("%-8s n=%d ns/row: ggml T1 %.0f  mdot T1 %.0f | ggml T4 %.0f  mdot T4 %.0f (%g)\n", ggml_type_name(t.type), n, g1, m1, g4, m4, acc * 0);
    }
}
