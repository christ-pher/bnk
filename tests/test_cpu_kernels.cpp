// Q2_0 AVX2 dot vs exact float math, and vs ggml's own Q2_0 x Q8_0 path.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/gguf_model.h"
#include "cpu/kernels.h"
#include "ggml-cpu.h"

using namespace bnk;

int main(int argc, char ** argv) {
    ggml_cpu_init();
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
