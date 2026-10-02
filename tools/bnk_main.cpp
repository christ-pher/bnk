// bnk command line: run a tokenized prompt, greedy decode, and compare against llama.cpp dumps.
//   bnk run   --model M.gguf --tokens-file F [--max-new N] [--ctx N] [--threads N]
//   bnk check --model M.gguf --tokens-file F --ref PREFIX   (per-layer residual + logits vs llama_ref)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "engine/engine.h"

using namespace bnk;

static std::vector<int32_t> read_tokens(const std::string & path) {
    std::ifstream in(path);
    std::string s((std::istreambuf_iterator<char>(in)), {});
    for (char & ch : s) if (ch == ',') ch = ' ';
    std::istringstream is(s);
    std::vector<int32_t> v;
    long x;
    while (is >> x) v.push_back((int32_t) x);
    return v;
}

static bool read_floats(const std::string & path, std::vector<float> & v) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize(n / 4);
    size_t r = fread(v.data(), 4, v.size(), f);
    fclose(f);
    return r == v.size();
}

static double rel_rms(const float * a, const float * b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        num += (double) (a[i] - b[i]) * (a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return sqrt(num / (den + 1e-30));
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: bnk run|check --model M --tokens-file F [--max-new N] [--ctx N] [--threads N] [--ref P]\n");
        return 1;
    }
    std::string mode = argv[1], model, tokfile, ref;
    int max_new = 32;
    EngineOptions opt;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model = next();
        else if (a == "--tokens-file") tokfile = next();
        else if (a == "--max-new") max_new = std::stoi(next());
        else if (a == "--ctx") opt.max_ctx = std::stoi(next());
        else if (a == "--threads") opt.cpu_threads = std::stoi(next());
        else if (a == "--ref") ref = next();
        else if (a == "--profile") opt.profile = next();
        else if (a == "--counts") opt.counts_out = next();
        else if (a == "--cache-gib") opt.expert_cache_gib = std::stod(next());
        else if (a == "--no-graphs") opt.use_graphs = false;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    auto prompt = read_tokens(tokfile);
    Engine eng;
    eng.load(model, opt);
    const Config & c = eng.cfg();

    if (mode == "check") {
        // window by window, compare every layer's residual and the logits at every position
        std::vector<std::vector<float>> ref_layers(c.n_layer);
        for (int il = 0; il < c.n_layer; ++il) read_floats(ref + ".l_last-" + std::to_string(il), ref_layers[il]);
        std::vector<float> ref_logits;
        read_floats(ref + ".logits", ref_logits);
        const size_t HC = c.hc_dim();
        std::vector<double> lerr(c.n_layer, 0);
        double logit_err = 0, kl_sum = 0, kl_max = 0;
        int agree = 0;
        eng.dump_all = true;
        for (size_t p = 0; p < prompt.size(); p += kMaxWindow) {
            const int T = (int) std::min<size_t>(kMaxWindow, prompt.size() - p);
            eng.forward(prompt.data() + p, T);
            for (int il = 0; il < c.n_layer; ++il) {
                if (ref_layers[il].size() < (p + T) * HC) continue;
                lerr[il] = std::max(lerr[il], rel_rms(eng.dumped_layers[il].data(), ref_layers[il].data() + p * HC, T * HC));
            }
            for (int t = 0; t < T; ++t) {
                auto lg = eng.logits_host(t);
                const float * rl = ref_logits.data() + (p + t) * c.n_vocab;
                logit_err = std::max(logit_err, rel_rms(lg.data(), rl, c.n_vocab));
                int a = 0, b = 0;
                for (int j = 1; j < c.n_vocab; ++j) { if (lg[j] > lg[a]) a = j; if (rl[j] > rl[b]) b = j; }
                agree += a == b;
                // KL(ref || ours)
                double mr = rl[b], mo = lg[a], zr = 0, zo = 0;
                for (int j = 0; j < c.n_vocab; ++j) { zr += exp(rl[j] - mr); zo += exp(lg[j] - mo); }
                double kl = 0;
                for (int j = 0; j < c.n_vocab; ++j) {
                    const double pr = exp(rl[j] - mr) / zr;
                    if (pr < 1e-12) continue;
                    const double lpo = (lg[j] - mo) - log(zo);
                    kl += pr * (log(pr) - lpo);
                }
                kl_sum += kl;
                kl_max = std::max(kl_max, kl);
            }
        }
        for (int il = 0; il < c.n_layer; ++il) printf("layer %2d  max rel rms vs llama.cpp %.3e\n", il, lerr[il]);
        printf("logits max rel rms %.3e, argmax agreement %d/%zu, KL(ref||bnk) mean %.4f max %.4f\n", logit_err, agree,
               prompt.size(), kl_sum / prompt.size(), kl_max);
        return 0;
    }

    const double t0 = now_ms();
    eng.prefill(prompt);
    const double t1 = now_ms();
    int tok = eng.argmax(eng.last_T - 1);
    std::vector<int> out{tok};
    eng.times = StageTimes{};
    const double t2 = now_ms();
    for (int i = 1; i < max_new; ++i) {
        eng.forward(&tok, 1);
        tok = eng.argmax(0);
        out.push_back(tok);
        if (tok == c.eos_token) break;
    }
    const double t3 = now_ms();
    printf("output:");
    for (int t : out) printf(" %d", t);
    printf("\nprefill %zu tokens in %.1f ms (%.1f tok/s)\n", prompt.size(), t1 - t0, prompt.size() / ((t1 - t0) / 1000));
    const auto & tm = eng.times;
    printf("decode %zu tokens in %.1f ms (%.2f tok/s); per token: total %.2f ms, CPU experts %.2f ms, PLE host %.2f ms\n",
           out.size() - 1, t3 - t2, (out.size() - 1) / ((t3 - t2) / 1000), tm.total_ms / tm.calls,
           tm.cpu_experts_ms / tm.calls, tm.ple_ms / tm.calls);
    printf("expert misses %.2f%% (%.2f per token)\n", 100.0 * tm.misses / std::max<int64_t>(1, tm.routed),
           (double) tm.misses / tm.calls);
    return 0;
}
