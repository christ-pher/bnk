// The CPU expert pool's speed on real experts of a deepseek41 GGUF, as decode uses it: T=1 windows with a few experts
// each (cold: the experts rotate through a set far larger than the caches; warm: the same experts every call), and
// verify-window shapes (T tokens per expert, the multi-token kernels).
//   bench_ds_cpu MODEL.gguf [layer] [experts in the store = 96] [experts per call = 3]
//   THREADS=24 BNK_CPU_CHUNKS=1 BNK_CPU_PIN=0 BNK_CPU_MDOT1=0 ...
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "core/gguf_model.h"
#include "cpu/expert_pool.h"
#include "deepseek/ds_model.h"
#include "ggml-cpu.h"

using namespace bnk;

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench_ds_cpu MODEL.gguf [layer] [n_store] [per_call]\n"); return 1; }
    ggml_cpu_init();
    const int il = argc > 2 ? atoi(argv[2]) : 10;
    const int n_store = argc > 3 ? atoi(argv[3]) : 96;
    const int per_call = argc > 4 ? atoi(argv[4]) : 3;
    GgufModel g;
    g.open(argv[1]);
    const DsConfig c = load_ds_config(g);
    Model ev;
    ev.cfg.n_layer = c.n_layer; ev.cfg.n_embd = c.n_embd; ev.cfg.n_expert = c.n_expert; ev.cfg.n_expert_used = c.n_used;
    ev.cfg.n_ff_exp = c.n_ff;
    ev.layers.assign(c.n_layer, LayerWeights{});
    for (int l = 0; l < c.n_layer; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const TensorRef & gt = g.get(b + "ffn_gate_exps.weight"), & ut = g.get(b + "ffn_up_exps.weight"), & dt = g.get(b + "ffn_down_exps.weight");
        LayerWeights & L = ev.layers[l];
        L.gate_type = gt.type; L.up_type = ut.type; L.down_type = dt.type;
        L.gate_bytes = gt.nbytes / c.n_expert; L.up_bytes = ut.nbytes / c.n_expert; L.down_bytes = dt.nbytes / c.n_expert;
        L.gate_src = gt.data; L.up_src = ut.data; L.down_src = dt.data;
    }
    const int E = c.n_embd;
    const LayerWeights & L = ev.layers[il];
    const double blob_mb = (L.gate_bytes + L.up_bytes + L.down_bytes) / 1048576.0;
    printf("layer %d: gate %s up %s down %s, %.2f MiB per expert, %d experts in the store\n", il, ggml_type_name((ggml_type) L.gate_type),
           ggml_type_name((ggml_type) L.up_type), ggml_type_name((ggml_type) L.down_type), blob_mb, n_store);
    std::vector<char> skip((size_t) c.n_layer * c.n_expert, 1);
    for (int e = 0; e < n_store; ++e) skip[(size_t) il * c.n_expert + e] = 0;
    ExpertStore st;
    st.build(ev, 8, false, &skip);
    const int threads = getenv("THREADS") ? atoi(getenv("THREADS")) : 24;
    CpuExpertPool pool;
    pool.init(ev, st, threads);
    cpu_experts_set_swiglu_limit(c.swiglu_limit);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int Tmax = 8;
    std::vector<float> x((size_t) Tmax * E), out((size_t) Tmax * E);
    for (auto & v : x) v = nd(rng);
    auto bench = [&](const char * what, int T, bool cold, int tokens_per_expert) {
        // tasks: per_call experts, each with tokens_per_expert tokens (tokens round-robin)
        std::vector<ExpertTask> tasks;
        int e0 = 0;
        auto make = [&]() {
            tasks.clear();
            for (int j = 0; j < per_call; ++j) {
                const int e = (e0 + j) % n_store;
                for (int t = 0; t < tokens_per_expert; ++t) tasks.push_back(ExpertTask{(t + j) % T, j, e, 0.3f});
            }
            if (cold) e0 = (e0 + per_call) % n_store;
        };
        make();
        for (int i = 0; i < 3; ++i) { pool.run(il, T, x.data(), tasks, out.data()); make(); }
        const int reps = cold ? std::max(8, 2 * n_store / per_call) : 30;
        const double t0 = now_ms();
        for (int i = 0; i < reps; ++i) { pool.run(il, T, x.data(), tasks, out.data()); make(); }
        const double ms = (now_ms() - t0) / reps;
        printf("%-44s %7.3f ms per call  (%5.1f GB/s of weights, %.3f ms per expert)\n", what, ms, per_call * blob_mb * 1.048576 / ms,
               ms / per_call);
    };
    char buf[128];
    snprintf(buf, sizeof buf, "T=1, %d experts, cold (decode)", per_call); bench(buf, 1, true, 1);
    snprintf(buf, sizeof buf, "T=1, %d experts, warm (same experts)", per_call); bench(buf, 1, false, 1);
    snprintf(buf, sizeof buf, "T=4, %d experts x 4 tokens, cold (verify)", per_call); bench(buf, 4, true, 4);
    snprintf(buf, sizeof buf, "T=8, %d experts x 8 tokens, cold", per_call); bench(buf, 8, true, 8);
    snprintf(buf, sizeof buf, "T=8, %d experts x 2 tokens, cold", per_call); bench(buf, 8, true, 2);
    return 0;
}
