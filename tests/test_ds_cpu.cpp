// The CPU expert pool on real experts of a deepseek41 GGUF: one token (ggml vec_dot per task) against several tokens
// (the multi-token kernels), both against a double reference on the dequantized rows. No GPU work beyond pinning.
//   test_ds_cpu MODEL.gguf [layer]
#include <algorithm>
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

static std::mt19937 rng(5);

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_ds_cpu MODEL.gguf [layer]\n"); return 1; }
    ggml_cpu_init();
    const int il = argc > 2 ? atoi(argv[2]) : 0;
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
    const int E = c.n_embd, F = c.n_ff;
    const LayerWeights & L = ev.layers[il];
    printf("layer %d: gate %s up %s down %s, clamp %.0f\n", il, ggml_type_name((ggml_type) L.gate_type),
           ggml_type_name((ggml_type) L.up_type), ggml_type_name((ggml_type) L.down_type), c.swiglu_limit);
    // the store holds three experts of this layer only
    const int ex[6] = {3, 200, 383, 17, 99, 301};
    std::vector<char> skip((size_t) c.n_layer * c.n_expert, 1);
    for (int e : ex) skip[(size_t) il * c.n_expert + e] = 0;
    ExpertStore st;
    st.build(ev, 4, true, &skip);
    CpuExpertPool pool;
    pool.init(ev, st, getenv("THREADS") ? atoi(getenv("THREADS")) : 24);
    cpu_experts_set_swiglu_limit(c.swiglu_limit);

    // reference: dequantized rows in double
    auto rows_of = [&](const uint8_t * src, int type, int64_t n, int64_t cols) {
        std::vector<float> f((size_t) n * cols);
        const size_t rb = ggml_row_size((ggml_type) type, cols);
        for (int64_t r = 0; r < n; ++r) ggml_get_type_traits((ggml_type) type)->to_float(src + r * rb, f.data() + r * cols, cols);
        return f;
    };
    auto expert_ref = [&](int e, const float * x, std::vector<double> & out) {
        auto gw = rows_of(L.gate_src + (size_t) e * L.gate_bytes, L.gate_type, F, E);
        auto uw = rows_of(L.up_src + (size_t) e * L.up_bytes, L.up_type, F, E);
        auto dw = rows_of(L.down_src + (size_t) e * L.down_bytes, L.down_type, E, F);
        std::vector<double> h(F);
        for (int r = 0; r < F; ++r) {
            double gs = 0, us = 0;
            for (int i = 0; i < E; ++i) { gs += (double) gw[(size_t) r * E + i] * x[i]; us += (double) uw[(size_t) r * E + i] * x[i]; }
            if (c.swiglu_limit > 0) { gs = std::min(gs, (double) c.swiglu_limit); us = std::min(std::max(us, -(double) c.swiglu_limit), (double) c.swiglu_limit); }
            h[r] = gs / (1 + exp(-gs)) * us;
        }
        out.assign(E, 0.0);
        for (int r = 0; r < E; ++r) { double s = 0; for (int i = 0; i < F; ++i) s += (double) dw[(size_t) r * F + i] * h[i]; out[r] = s; }
    };
    // replay mode: test_ds_cpu MODEL layer X.bin TASKS.txt OUT.bin -- the engine's mailbox (one row) against the
    // reference and against the pool run here
    if (argc > 5) {
        std::vector<float> xr(E), outr(E);
        FILE * fx = fopen(argv[3], "rb"), * fo = fopen(argv[5], "rb");
        if (fread(xr.data(), 4, E, fx) != (size_t) E || fread(outr.data(), 4, E, fo) != (size_t) E) { fprintf(stderr, "bad dump\n"); return 1; }
        fclose(fx); fclose(fo);
        std::vector<ExpertTask> tasks;
        FILE * ft = fopen(argv[4], "r");
        int t, e; float w;
        while (fscanf(ft, "%d %d %f", &t, &e, &w) == 3) tasks.push_back(ExpertTask{t, (int) tasks.size(), e, w});
        fclose(ft);
        double xr2 = 0; for (float v : xr) xr2 += (double) v * v;
        printf("x rms %.4g, %zu tasks\n", sqrt(xr2 / E), tasks.size());
        std::vector<char> sk((size_t) c.n_layer * c.n_expert, 1);
        for (auto & tk : tasks) sk[(size_t) il * c.n_expert + tk.expert] = 0;
        ExpertStore st2;
        st2.build(ev, 4, false, &sk);
        CpuExpertPool pool2;
        pool2.init(ev, st2, 24);
        cpu_experts_set_swiglu_limit(c.swiglu_limit);
        std::vector<double> sum(E, 0.0), r;
        for (auto & tk : tasks) {
            expert_ref(tk.expert, xr.data(), r);
            double rr = 0; for (int i = 0; i < E; ++i) rr += r[i] * r[i];
            printf("  expert %3d w %.4f: output rms %.4g\n", tk.expert, tk.w, sqrt(rr / E));
            for (int i = 0; i < E; ++i) sum[i] += tk.w * r[i];
        }
        double sr = 0, so = 0, d = 0;
        for (int i = 0; i < E; ++i) { sr += sum[i] * sum[i]; so += (double) outr[i] * outr[i]; d += (outr[i] - sum[i]) * (outr[i] - sum[i]); }
        printf("reference routed sum rms %.4g; engine's mailbox out rms %.4g; rel diff %.4g\n", sqrt(sr / E), sqrt(so / E), sqrt(d / sr));
        std::vector<float> here(E);
        pool2.run(il, 1, xr.data(), tasks, here.data());
        double h2 = 0, dh = 0; for (int i = 0; i < E; ++i) { h2 += (double) here[i] * here[i]; dh += (here[i] - sum[i]) * (here[i] - sum[i]); }
        printf("pool here: out rms %.4g, rel diff vs reference %.4g\n", sqrt(h2 / E), sqrt(dh / sr));
        return 0;
    }
    const int T = 8;
    std::vector<float> x((size_t) T * E);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : x) v = nd(rng);
    std::vector<double> ref0, ref1;
    expert_ref(ex[0], x.data(), ref0);           // token 0 through expert A
    expert_ref(ex[1], x.data() + E, ref1);       // token 1 through expert B
    auto report = [&](const char * what, const float * got, const std::vector<double> & ref, double w) {
        double num = 0, den = 0;
        for (int i = 0; i < E; ++i) { num += (got[i] - w * ref[i]) * (got[i] - w * ref[i]); den += (w * ref[i]) * (w * ref[i]); }
        printf("%-44s rel rms err %.4e%s\n", what, sqrt(num / den), sqrt(num / den) > 0.02 ? "  FAIL" : "");
        return sqrt(num / den) > 0.02;
    };
    int fails = 0;
    std::vector<float> out((size_t) T * E);
    // T = 1: one task
    pool.run(il, 1, x.data(), {ExpertTask{0, 0, ex[0], 1.f}}, out.data());
    fails += report("T=1, one expert (vec_dot path)", out.data(), ref0, 1.f);
    // T = 1 with a weight
    pool.run(il, 1, x.data(), {ExpertTask{0, 0, ex[0], 0.37f}}, out.data());
    fails += report("T=1, weight 0.37", out.data(), ref0, 0.37);
    // T = 2: two tokens, two experts (vec_dot per task)
    pool.run(il, 2, x.data(), {ExpertTask{0, 0, ex[0], 1.f}, ExpertTask{1, 0, ex[1], 1.f}}, out.data());
    fails += report("T=2, token 0 expert A", out.data(), ref0, 1.f);
    fails += report("T=2, token 1 expert B", out.data() + E, ref1, 1.f);
    // T = 8: tokens 0..7 all through expert A (the multi-token kernels), token 1 through B as well
    {
        std::vector<ExpertTask> tasks;
        for (int t = 0; t < T; ++t) tasks.push_back(ExpertTask{t, 0, ex[0], 1.f});
        tasks.push_back(ExpertTask{1, 1, ex[1], 1.f});
        pool.run(il, T, x.data(), tasks, out.data());
        fails += report("T=8, token 0 expert A (mdot path)", out.data(), ref0, 1.f);
        std::vector<double> sum(E);
        std::vector<double> r1a;
        expert_ref(ex[0], x.data() + E, r1a);
        for (int i = 0; i < E; ++i) sum[i] = r1a[i] + ref1[i];
        fails += report("T=8, token 1 expert A + B", out.data() + E, sum, 1.f);
    }
    // T = 1 through six experts at once (a decode step's misses), weights as the router gives them
    {
        std::vector<ExpertTask> tasks;
        const float ws[6] = {0.4f, 0.3f, 0.12f, 0.1f, 0.05f, 0.03f};
        std::vector<double> sum(E, 0.0), r;
        for (int i = 0; i < 6; ++i) {
            tasks.push_back(ExpertTask{0, i, ex[i], ws[i]});
            expert_ref(ex[i], x.data(), r);
            for (int j = 0; j < E; ++j) sum[j] += ws[i] * r[j];
        }
        for (int rep = 0; rep < 3; ++rep) {
            pool.run(il, 1, x.data(), tasks, out.data());
            fails += report("T=1, six experts (a decode step)", out.data(), sum, 1.f);
        }
    }
    printf(fails ? "\n%d FAILURES\n" : "\nall ok\n", fails);
    return fails ? 1 : 0;
}
