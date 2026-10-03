// CPU expert kernels on this machine: ggml vec_dot speed per format and the pool's latency for a few experts.
//   bench_cpu_experts MODEL.gguf [threads]
#include <cstdio>
#include <random>
#include <vector>

#include "core/util.h"
#include "cpu/expert_pool.h"
#include "ggml-cpu.h"

using namespace bnk;

int main(int argc, char ** argv) {
    ggml_cpu_init();
    Model m;
    m.load(argv[1], true);
    const int threads = argc > 2 ? atoi(argv[2]) : 24;
    const Config & c = m.cfg;
    const int E = c.n_embd, F = c.n_ff_exp;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd;
    std::vector<float> x(8 * E);
    for (auto & v : x) v = nd(rng);

    // 1. single-thread vec_dot per format (rows straight from the mmap, warmed)
    std::vector<int> seen;
    for (int il = 0; il < c.n_layer; ++il) {
        const LayerWeights & L = m.layers[il];
        for (int which = 0; which < 2; ++which) {
            const int ty = which ? L.down_type : L.gate_type;
            if (std::find(seen.begin(), seen.end(), ty * 2 + which) != seen.end()) continue;
            seen.push_back(ty * 2 + which);
            const auto * tr = ggml_get_type_traits_cpu((ggml_type) ty);
            const int n = which ? F : E;
            const int rows = which ? E : F;
            const size_t rb = ggml_row_size((ggml_type) ty, n);
            const uint8_t * w = which ? L.down_src : L.gate_src;
            std::vector<uint8_t> xq(ggml_row_size(tr->vec_dot_type, n));
            ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(x.data(), xq.data(), n);
            float s = 0, acc = 0;
            for (int r = 0; r < rows; ++r) { tr->vec_dot(n, &s, 0, w + r * rb, 0, xq.data(), 0, 1); acc += s; }
            const double t0 = now_ms();
            const int reps = 20;
            for (int k = 0; k < reps; ++k)
                for (int r = 0; r < rows; ++r) { tr->vec_dot(n, &s, 0, w + r * rb, 0, xq.data(), 0, 1); acc += s; }
            const double dt = (now_ms() - t0) / reps;
            printf("%-8s %s: %4d rows x %4d: %.1f us/matrix, %.0f ns/row, %.2f GB/s single-thread (%g)\n",
                   ggml_type_name((ggml_type) ty), which ? "down   " : "gate/up", rows, n, dt * 1000,
                   dt * 1e6 / rows, rows * rb / (dt * 1e6), acc * 0);
        }
    }

    // 2. the pool: latency of k experts for one token
    ExpertStore st;
    st.build(m, 16, true);
    CpuExpertPool pool;
    pool.init(m, st, threads);
    std::vector<float> out(8 * E);
    // cold: what a decode step sees - a different layer each call, random experts whose weights come from DRAM
    {
        std::mt19937 r2(7);
        for (int nexp : {1, 2, 4, 8}) {
            const int reps = 480;
            double tot = 0;
            for (int k = 0; k < reps; ++k) {
                const int il = k % c.n_layer;
                std::vector<ExpertTask> tasks;
                for (int j = 0; j < nexp; ++j) tasks.push_back({j % 4, j, (int) (r2() % c.n_expert), 0.1f});
                const double t0 = now_ms();
                pool.run(il, 4, x.data(), tasks, out.data());
                tot += now_ms() - t0;
            }
            printf("cold pool %2d threads: %d expert-token pairs over a 4-token window: %.1f us/layer\n", threads, nexp,
                   tot / reps * 1000);
        }
    }
    if (getenv("BNK_COLD_ONLY")) return 0;
    for (int nexp : {1, 2, 4, 10}) {
        for (int il : {0, 1, 17}) {
            std::vector<ExpertTask> tasks;
            for (int j = 0; j < nexp; ++j) tasks.push_back({0, j, (j * 37 + 5) % c.n_expert, 0.1f});
            for (int k = 0; k < 5; ++k) pool.run(il, 1, x.data(), tasks, out.data());
            const double t0 = now_ms();
            const int reps = 50;
            for (int k = 0; k < reps; ++k) pool.run(il, 1, x.data(), tasks, out.data());
            printf("pool %2d threads: %2d expert(s) layer %2d (%s/%s): %.1f us\n", threads, nexp, il,
                   ggml_type_name((ggml_type) m.layers[il].gate_type), ggml_type_name((ggml_type) m.layers[il].down_type),
                   (now_ms() - t0) / reps * 1000);
        }
    }
    return 0;
}
