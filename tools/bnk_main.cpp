// bnk command line: run a tokenized prompt, greedy decode, and compare against llama.cpp dumps.
//   bnk run   --model M.gguf --tokens-file F [--max-new N] [--ctx N] [--threads N]
//   bnk check --model M.gguf --tokens-file F --ref PREFIX   (per-layer residual + logits vs llama_ref)
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>

#include <cuda_profiler_api.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "dense/dense_engine.h"
#include "deepseek/ds_engine.h"
#include "engine/engine.h"
#include "engine/engine_api.h"
#include "engine/generate.h"
#include "engine/mtp.h"
#include "server/serve.h"
#include <memory>

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

// "0,1" -> {0, 1}
static std::vector<int> int_list(const std::string & s) {
    std::vector<int> v;
    for (size_t a = 0, b; a < s.size(); a = b + 1) {
        b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        if (b > a) v.push_back(std::stoi(s.substr(a, b - a)));
    }
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

// each pipeline stage's GPU time per decode forward
static void print_stages(const EngineApi & eng) {
    for (int i = 0; i < eng.stages(); ++i) {
        const StageInfo s = eng.stage_info(i);
        printf("stage %d (GPU %d, layers %d-%d): %.2f ms of GPU time per forward over %lld forwards", i, s.device,
               s.layer_begin, s.layer_end - 1, s.gpu_ms / std::max<int64_t>(1, s.gpu_forwards), (long long) s.gpu_forwards);
        if (s.prompt_tokens)
            printf("; prompt chunks %.1f s for %lld tokens (%.0f tok/s)", s.prompt_ms / 1000, (long long) s.prompt_tokens,
                   s.prompt_tokens / (s.prompt_ms / 1000));
        printf("\n");
    }
}

// The engine for the model's architecture: the MoE engine (qwen4exp) or the dense one (qwen35).
static std::unique_ptr<EngineApi> make_engine(const std::string & model, const EngineOptions & opt) {
    std::string arch;
    {
        GgufModel g;
        g.open(model);
        arch = g.get_str("general.architecture");
    }
    if (arch == "qwen35") {
        auto e = std::make_unique<DenseEngine>();
        e->load(model, opt);
        return e;
    }
    if (arch == "deepseek41") {   // docs/DEEPSEEK.md; --ple-gguf names the Engram file (default: beside the model)
        auto e = std::make_unique<DsEngine>();
        e->load(model, opt);
        return e;
    }
    auto e = std::make_unique<Engine>();
    e->load(model, opt);
    return e;
}

int main(int argc, char ** argv) {
    // GPU numbers as nvidia-smi shows them (CUDA's default order puts the fastest first)
    setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0);
    if (argc < 2) {
        fprintf(stderr, "usage: bnk run|check|nll --model M --tokens-file F [--max-new N] [--ctx N] [--threads N] [--ref P]\n"
                        "           [--gpus 0,1 [--split L]]   (several GPUs: a pipeline, layers split between them)\n"
                        "           [--kv8 1]   (8-bit K/V cache: less VRAM at long contexts; BNK_KV8=1 likewise)\n");
        return 1;
    }
    std::string mode = argv[1], model, tokfile, ref, mtp_path;
    bool solo = false, adapt_set = false, pipe = false;
    int max_new = 32;
    GenOptions gopt;
    EngineOptions opt;
    opt.kv8 = env_int("BNK_KV8", 0) != 0;   // fp16 K/V by default; --kv8 wins
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
        else if (a == "--mtp") opt.mtp = next();
        else if (a == "--dflash") opt.dflash = next();
        else if (a == "--ple-gguf") opt.ple_gguf = next();
        else if (a == "--cold-experts") opt.cold_experts = next();
        else if (a == "--mmproj") opt.mmproj = next();
        else if (a == "--vision-cache-gib") opt.vision_cache_gib = std::stod(next());
        else if (a == "--draft-vocab") opt.draft_vocab = next();
        else if (a == "--prefill-chunk-max") opt.prefill_chunk_max = std::stoi(next());
        else if (a == "--prefill-small") opt.prefill_small = std::stoi(next());
        else if (a == "--stage-small-mib") opt.stage_small_mib = std::stoi(next());
        else if (a == "--stage-full-min") opt.stage_full_min = std::stoi(next());
        else if (a == "--draft") gopt.max_draft = std::stoi(next());
        else if (a == "--min-p") gopt.min_p = std::stof(next());
        else if (a == "--counts") opt.counts_out = next();
        else if (a == "--cache-gib") {   // one size, or one per GPU (comma-separated)
            const std::string v = next();
            if (v.find(',') == std::string::npos) opt.expert_cache_gib = std::stod(v);
            else for (size_t a0 = 0, b; a0 < v.size(); a0 = b + 1) {
                b = v.find(',', a0);
                if (b == std::string::npos) b = v.size();
                opt.stage_cache_gib.push_back(std::stod(v.substr(a0, b - a0)));
            }
        }
        else if (a == "--vram-reserve-gib") opt.vram_reserve_gib = std::stod(next());
        else if (a == "--no-graphs") opt.use_graphs = false;
        else if (a == "--prefill-chunk") opt.prefill_chunk = std::stoi(next());
        else if (a == "--gemm-min-tokens") opt.gemm_min_tokens = std::stoi(next());
        else if (a == "--adapt-every") opt.adapt_every = std::stoi(next()), adapt_set = true;
        else if (a == "--adapt-swaps") opt.adapt_swaps = std::stoi(next()), adapt_set = true;
        else if (a == "--park-gib") opt.park_gib = std::stod(next());
        else if (a == "--park-min") opt.park_min = std::stoi(next());
        else if (a == "--kv8") opt.kv8 = std::stoi(next()) != 0;
        else if (a == "--slots") opt.slots = std::stoi(next());
        else if (a == "--gpus") { const std::string v = next(); opt.gpus = v == "all" ? std::vector<int>{-1} : int_list(v); }
        else if (a == "--split") opt.split = int_list(next());
        else if (a == "--tp") opt.tp = std::stoi(next());
        else if (a == "--solo") solo = true;
        else if (a == "--pipe") pipe = true;   // multi on two GPUs: two lanes in flight (Engine::forward_pipe)
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (mode == "serve") {
        std::unique_ptr<EngineApi> engp = make_engine(model, opt);
        return serve_main(*engp, gopt, engp->model_name());
    }
    // several conversations share the expert cache: it has to follow a mix of routings, so it adapts every forward
    // and swaps more at a time (3 batched agents: 72-76 -> 84 tok/s; one conversation: no measurable change)
    if (opt.slots > 1 && !adapt_set) {
        opt.adapt_every = 1;
        opt.adapt_swaps = 32;
    }
    auto prompt = mode == "multi" ? std::vector<int32_t>{} : read_tokens(tokfile);
    std::unique_ptr<EngineApi> engp = make_engine(model, opt);
    EngineApi & eng = *engp;
    const Config & c = eng.cfg();

    if (mode == "nll") {
        // teacher-forced negative log-likelihood of the tokens through the decode path, in windows of BNK_NLL_T
        // (default 5) rows: how well the model predicts a text, for comparing kernel variants. BNK_NLL_SKIP=N reads
        // the first N tokens as a prompt first and scores the rest (the predictions at that depth)
        const int T = std::max(1, std::min(env_int("BNK_NLL_T", 5), eng.max_window()));
        const int V = c.n_vocab;
        std::vector<float> lg((size_t) T * V);
        double nll = 0;
        int64_t n = 0, top1 = 0;
        const size_t skip = std::min(prompt.size() - 1, (size_t) env_int("BNK_NLL_SKIP", 0));
        if (skip) eng.prefill(std::vector<int32_t>(prompt.begin(), prompt.begin() + skip));
        // BNK_NLL_DUMP=F writes every row's top 64 log-probabilities; BNK_NLL_REF=F reads such a file and reports the
        // KL divergence from it (over those 64 tokens), a finer measure of a variant than the NLL
        constexpr int KT = 64;
        FILE * dump = getenv("BNK_NLL_DUMP") ? fopen(getenv("BNK_NLL_DUMP"), "wb") : nullptr;
        FILE * reff = getenv("BNK_NLL_REF") ? fopen(getenv("BNK_NLL_REF"), "rb") : nullptr;
        double kl_sum = 0, kl_max = 0;
        int64_t top1_same = 0;
        std::vector<int> kl_hist(6, 0);
        double fwd_ms = 0;   // forward plus the logits copy, full windows only
        int64_t fwd_n = 0;
        for (size_t i = skip; i + 1 < prompt.size(); i += T) {
            const int t = (int) std::min<size_t>(T, prompt.size() - 1 - i);
            const auto f0 = std::chrono::steady_clock::now();
            eng.forward(prompt.data() + i, t);
            eng.logits_rows_host(t, lg.data());
            if (t == T) {
                fwd_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - f0).count();
                ++fwd_n;
            }
            for (int r = 0; r < t; ++r) {
                const float * row = lg.data() + (size_t) r * V;
                const int want = prompt[i + r + 1];
                float mx = row[0];
                int am = 0;
                for (int v = 1; v < V; ++v)
                    if (row[v] > mx) mx = row[v], am = v;
                double z = 0;
                for (int v = 0; v < V; ++v) z += std::exp((double) row[v] - mx);
                nll += std::log(z) - ((double) row[want] - mx);
                top1 += am == want;
                const double lz = std::log(z) + mx;
                if (dump) {
                    std::vector<int> id(V);
                    for (int v = 0; v < V; ++v) id[v] = v;
                    std::partial_sort(id.begin(), id.begin() + KT, id.end(), [&](int a, int b) { return row[a] > row[b]; });
                    for (int j = 0; j < KT; ++j) {
                        const float lp = (float) (row[id[j]] - lz);
                        fwrite(&id[j], 4, 1, dump);
                        fwrite(&lp, 4, 1, dump);
                    }
                }
                if (reff) {
                    int id[KT];
                    float lp[KT];
                    for (int j = 0; j < KT; ++j)
                        if (fread(&id[j], 4, 1, reff) != 1 || fread(&lp[j], 4, 1, reff) != 1) id[j] = -1;
                    double kl = 0;
                    for (int j = 0; j < KT; ++j)
                        if (id[j] >= 0) kl += std::exp((double) lp[j]) * ((double) lp[j] - ((double) row[id[j]] - lz));
                    kl_sum += kl;
                    kl_max = std::max(kl_max, kl);
                    top1_same += id[0] == am;
                    const double edges[5] = {1e-4, 1e-3, 1e-2, 1e-1, 1};
                    int b = 0;
                    while (b < 5 && kl >= edges[b]) ++b;
                    kl_hist[b]++;
                }
                ++n;
            }
        }
        printf("nll %.5f per token over %lld tokens (windows of %d), ppl %.4f, top-1 %.2f%%\n", nll / n, (long long) n, T,
               std::exp(nll / n), 100.0 * top1 / n);
        if (fwd_n) printf("forward of %d rows: %.2f ms (with the logits copy), over %lld windows\n", T, fwd_ms / fwd_n, (long long) fwd_n);
        if (dump) fclose(dump);
        if (reff) {
            fclose(reff);
            printf("KL from the reference: mean %.5f, max %.4f; same top-1 %.2f%%; rows by KL <1e-4 %d, <1e-3 %d, <1e-2 %d, "
                   "<0.1 %d, <1 %d, more %d\n", kl_sum / n, kl_max, 100.0 * top1_same / n, kl_hist[0], kl_hist[1],
                   kl_hist[2], kl_hist[3], kl_hist[4], kl_hist[5]);
        }
        return 0;
    }

    if (mode == "pdump") {
        // batched prompt pass; the last position's logits to --ref (raw float32), for comparing engine variants.
        // BNK_SPLIT=N: process the first N tokens untimed, then time the rest (a follow-up turn at depth)
        const size_t split = std::min(prompt.size() - 1, (size_t) env_int("BNK_SPLIT", 0));
        if (split) eng.prefill(std::vector<int32_t>(prompt.begin(), prompt.begin() + split));
        const double t0 = now_ms();
        eng.prefill(std::vector<int32_t>(prompt.begin() + split, prompt.end()));
        const double t1 = now_ms();
        if (split) printf("timed part: %zu tokens after %zu\n", prompt.size() - split, split);
        auto lg = eng.logits_host(eng.last_T - 1);
        FILE * f = fopen(ref.c_str(), "wb");
        if (!f || fwrite(lg.data(), 4, lg.size(), f) != lg.size()) throw std::runtime_error("cannot write " + ref);
        fclose(f);
        printf("prefill %zu tokens in %.1f ms (%.1f tok/s) -> %s\n", prompt.size(), t1 - t0,
               prompt.size() / ((t1 - t0) / 1000), ref.c_str());
        return 0;
    }

    if (mode == "pcheck") {
        // batched prompt pass: the last position's logits vs llama.cpp
        std::vector<float> ref_logits;
        read_floats(ref + ".logits", ref_logits);
        const double t0 = now_ms();
        eng.prefill(prompt);
        const double t1 = now_ms();
        auto lg = eng.logits_host(eng.last_T - 1);
        const float * rl = ref_logits.data() + (prompt.size() - 1) * c.n_vocab;
        int a = 0, b = 0;
        for (int j = 1; j < c.n_vocab; ++j) { if (lg[j] > lg[a]) a = j; if (rl[j] > rl[b]) b = j; }
        double mr = rl[b], mo = lg[a], zr = 0, zo = 0, kl = 0;
        for (int j = 0; j < c.n_vocab; ++j) { zr += exp(rl[j] - mr); zo += exp(lg[j] - mo); }
        for (int j = 0; j < c.n_vocab; ++j) {
            const double pr = exp(rl[j] - mr) / zr;
            if (pr > 1e-12) kl += pr * (log(pr) - ((lg[j] - mo) - log(zo)));
        }
        printf("prefill %zu tokens in %.1f ms (%.1f tok/s): last-token argmax bnk %d ref %d, KL %.4f\n", prompt.size(),
               t1 - t0, prompt.size() / ((t1 - t0) / 1000), a, b, kl);
        return 0;
    }

    if (mode == "hdump") {
        // teacher-forced pass in full windows; per position: the residual after each layer of BNK_HDUMP_LAYERS
        // (default 5,19,33,47,61: DFlash2's taps) as fp32, concatenated, to BNK_HDUMP (.h), and the argmax to .ids
        std::vector<int> layers;
        {
            const char * s = getenv("BNK_HDUMP_LAYERS");
            std::string v = s ? s : "5,19,33,47,61";
            for (size_t a = 0; a < v.size();) {
                size_t b = v.find(',', a);
                if (b == std::string::npos) b = v.size();
                layers.push_back(std::stoi(v.substr(a, b - a)));
                a = b + 1;
            }
        }
        const char * out = getenv("BNK_HDUMP");
        if (!out) throw std::runtime_error("hdump: set BNK_HDUMP to the output path prefix");
        FILE * fh = fopen((std::string(out) + ".h").c_str(), "wb");
        FILE * fi = fopen((std::string(out) + ".ids").c_str(), "wb");
        const size_t HC = eng.res_dim();
        eng.dump_all = true;
        std::vector<int32_t> am(kMaxWindow);
        const int hdT = std::max(1, std::min(env_int("BNK_HDUMP_T", kMaxWindow), kMaxWindow));   // rows per window
        const int hdT2 = env_int("BNK_HDUMP_T2", 0);   // a second pass with this window size, to <out>_2.h / .ids
        for (int pass = 0; pass < (hdT2 > 0 ? 2 : 1); ++pass) {
        const int wT = pass == 0 ? hdT : hdT2;
        if (pass == 1) {
            fclose(fh); fclose(fi);
            fh = fopen((std::string(out) + "_2.h").c_str(), "wb");
            fi = fopen((std::string(out) + "_2.ids").c_str(), "wb");
            eng.reset();
        }
        for (size_t p = 0; p < prompt.size(); p += wT) {
            const int T = (int) std::min<size_t>(wT, prompt.size() - p);
            eng.forward(prompt.data() + p, T);
            eng.argmax_all(T, am.data());
            for (int t = 0; t < T; ++t)
                for (int il : layers) fwrite(eng.dumped_layers[il].data() + t * HC, 4, HC, fh);
            fwrite(am.data(), 4, T, fi);
        }
        }
        fclose(fh);
        fclose(fi);
        printf("hdump: %zu positions x %zu layers x %zu floats\n", prompt.size(), layers.size(), HC);
        return 0;
    }

    if (mode == "check") {
        // window by window, compare every layer's residual and the logits at every position
        // the residual after every layer: llama_ref names it l_last-N for the HC models, l_out-N otherwise
        std::vector<std::vector<float>> ref_layers(c.n_layer);
        for (int il = 0; il < c.n_layer; ++il)
            read_floats(ref + (c.hc > 1 ? ".l_last-" : ".l_out-") + std::to_string(il), ref_layers[il]);
        std::vector<float> ref_logits;
        read_floats(ref + ".logits", ref_logits);
        const size_t HC = eng.res_dim();
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
                static const bool lt = getenv("BNK_LAYER_TRACE") != nullptr;
                if (lt)
                    for (int t = 0; t < T; ++t)
                        fprintf(stderr, "LE %d %zu %.5f\n", il, p + t,
                                rel_rms(eng.dumped_layers[il].data() + t * HC, ref_layers[il].data() + (p + t) * HC, HC));
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
                static const char * dl = getenv("BNK_DUMP_LOGITS");
                if (dl) {
                    FILE * f = fopen(dl, (p + t) == 0 ? "wb" : "ab");
                    fwrite(lg.data(), 4, c.n_vocab, f);
                    fclose(f);
                }
                static const bool trace = getenv("BNK_KL_TRACE") != nullptr;
                if (trace) fprintf(stderr, "KL %zu %.5f\n", p + t, kl);
            }
        }
        for (int il = 0; il < c.n_layer; ++il) printf("layer %2d  max rel rms vs llama.cpp %.3e\n", il, lerr[il]);
        printf("logits max rel rms %.3e, argmax agreement %d/%zu, KL(ref||bnk) mean %.4f max %.4f\n", logit_err, agree,
               prompt.size(), kl_sum / prompt.size(), kl_max);
        return 0;
    }

    if (mode == "multi") {
        // several prompts (comma-separated --tokens-file), each in its own slot, decoded together in batched rounds
        // (the rows of a round shared out: kMaxWindow / active); --solo decodes them one after another instead
        std::vector<std::vector<int32_t>> prompts;
        for (size_t a = 0, b; a <= tokfile.size(); a = b + 1) {
            b = tokfile.find(',', a);
            if (b == std::string::npos) b = tokfile.size();
            prompts.push_back(read_tokens(tokfile.substr(a, b - a)));
        }
        const int n = (int) prompts.size();
        if (n > eng.slots()) {
            fprintf(stderr, "multi: %d prompts but %d slots (--slots)\n", n, eng.slots());
            return 1;
        }
        std::vector<std::unique_ptr<Generator>> gens;
        std::vector<std::vector<int>> outs(n);
        const double p0 = now_ms();
        for (int i = 0; i < n; ++i) {
            gens.push_back(std::make_unique<Generator>(eng, eng.mtp(), gopt, i));
            outs[i].push_back(gens[i]->prefill(prompts[i]));
        }
        const double p1 = now_ms();
        eng.times = StageTimes{};
        auto done = [&](int i) { return (int) outs[i].size() >= max_new || outs[i].back() == c.eos_token; };
        int64_t rounds = 0, forwards = 0;
        double fwd_ms = 0, fin_ms = 0;
        const double t2 = now_ms();
        if (pipe) {
            // the prompts alternate between two lanes; each step runs one lane on the first GPU while the other
            // (from the step before) runs on the second, then verifies, commits and drafts that one
            std::vector<int> lane_of(n);
            for (int i = 0; i < n; ++i) lane_of[i] = i & 1;
            std::vector<EngineApi::BatchWin> mid_wins;
            std::vector<int> mid_ids;
            int mid = -1;
            auto finish = [&](const std::vector<int> & ids) {
                std::vector<Generator *> ga;
                std::vector<std::vector<int32_t>> got(ids.size());
                for (size_t j = 0; j < ids.size(); ++j) {
                    got[j] = gens[ids[j]]->finish(false);
                    ga.push_back(gens[ids[j]].get());
                }
                Generator::draft_batch(ga);
                for (size_t j = 0; j < ids.size(); ++j)
                    for (int t : got[j]) {
                        outs[ids[j]].push_back(t);
                        if (t == c.eos_token || (int) outs[ids[j]].size() >= max_new) break;
                    }
            };
            while (true) {
                const int x = mid == 0 ? 1 : 0;
                std::vector<int> ids;
                for (int i = 0; i < n; ++i)
                    if (lane_of[i] == x && !done(i)) ids.push_back(i);
                if (ids.empty() && mid < 0) {   // nothing in flight and this lane is done: the rest join it
                    bool any = false;
                    for (int i = 0; i < n; ++i) any |= !done(i);
                    if (!any) break;
                    for (int i = 0; i < n; ++i) lane_of[i] = x;
                    continue;
                }
                std::vector<Generator *> ga;
                for (int i : ids) ga.push_back(gens[i].get());
                const std::vector<int> rows = Generator::share_rows(ga);
                std::vector<EngineApi::BatchWin> wins;
                for (size_t j = 0; j < ids.size(); ++j) {
                    const auto & w = gens[ids[j]]->window(rows[j]);
                    wins.push_back({ids[j], w.data(), (int) w.size(), w.size() == 1});
                }
                const double f0 = now_ms();
                eng.forward_pipe(wins, x, mid_wins, mid >= 0 ? mid : 1 - x);
                fwd_ms += now_ms() - f0;
                ++forwards;
                const double f1 = now_ms();
                if (mid >= 0) finish(mid_ids);
                fin_ms += now_ms() - f1;
                mid = wins.empty() ? -1 : x;
                mid_wins = wins;
                mid_ids = ids;
                ++rounds;
            }
        } else if (solo) {
            for (int i = 0; i < n; ++i)
                while (!done(i)) {
                    for (int t : gens[i]->next()) {
                        outs[i].push_back(t);
                        if (t == c.eos_token) break;
                    }
                    ++forwards;
                }
        } else {
            while (true) {
                std::vector<int> act;
                for (int i = 0; i < n; ++i)
                    if (!done(i)) act.push_back(i);
                if (act.empty()) break;
                std::vector<Generator *> ga;
                for (int i : act) ga.push_back(gens[i].get());
                static const bool even = getenv("BNK_EVEN_ROWS") != nullptr;
                std::vector<int> rows = Generator::share_rows(ga);
                if (even) rows.assign(act.size(), std::max(1, eng.max_window() / (int) act.size()));
                std::vector<EngineApi::BatchWin> wins;
                for (size_t j = 0; j < act.size(); ++j) {
                    const auto & w = gens[act[j]]->window(rows[j]);
                    wins.push_back({act[j], w.data(), (int) w.size(), w.size() == 1});
                }
                const double f0 = now_ms();
                eng.forward_batch(wins);
                fwd_ms += now_ms() - f0;
                ++forwards;
                const double f1 = now_ms();
                static const bool draft_batch = !getenv("BNK_DRAFT_BATCH") || atoi(getenv("BNK_DRAFT_BATCH")) != 0;
                std::vector<std::vector<int32_t>> got(act.size());
                for (size_t j = 0; j < act.size(); ++j) got[j] = gens[act[j]]->finish(!draft_batch);
                if (draft_batch) Generator::draft_batch(ga);
                for (size_t j = 0; j < act.size(); ++j)
                    for (int t : got[j]) {
                        outs[act[j]].push_back(t);
                        if (t == c.eos_token || (int) outs[act[j]].size() >= max_new) break;
                    }
                fin_ms += now_ms() - f1;
                ++rounds;
            }
        }
        const double t3 = now_ms();
        size_t total = 0;
        for (int i = 0; i < n; ++i) {
            printf("output[%d]:", i);
            for (int t : outs[i]) printf(" %d", t);
            printf("\n");
            total += outs[i].size() - 1;
        }
        const auto & tm = eng.times;
        printf("prefill %d prompts in %.1f ms\n", n, p1 - p0);
        printf("decode %s: %zu tokens in %.1f ms (%.2f tok/s total, %.2f per conversation); %lld forwards; "
               "expert misses %.2f%% (%.2f per forward)\n", solo ? "one after another" : pipe ? "in two lanes" : "batched", total, t3 - t2,
               total / ((t3 - t2) / 1000), total / ((t3 - t2) / 1000) / (solo ? 1 : n), (long long) forwards,
               100.0 * tm.misses / std::max<int64_t>(1, tm.routed), (double) tm.misses / std::max(1, tm.calls));
        if (!solo)
            printf("per round: batched forward %.2f ms, verify + commit + drafting of every conversation %.2f ms\n",
                   fwd_ms / std::max<int64_t>(1, rounds), fin_ms / std::max<int64_t>(1, rounds));
        printf("forward graphs: %lld captured, %.1f MiB of device memory at capture\n", (long long) eng.graph_captures(),
               eng.graph_bytes() / 1048576.0);
        print_stages(eng);
        return 0;
    }

    if (mode == "pbench") {
        // prompt reads as the server does them (a snapshot where the last turn begins), after a warm-up read of
        // --tokens-file. BNK_PB_FILES: "cold.txt|follow.txt,..." (follow: cold's tokens plus a new turn; optional);
        // each pair is read from scratch, then its follow-up with the context cached, then --max-new tokens decoded.
        const int32_t im = env_int("BNK_PB_IM_START", 248045);
        auto ckpt = [&](const std::vector<int32_t> & p) {
            for (size_t i = p.size(); i-- > 1;)
                if (p[i] == im) return (int) i;
            return -1;
        };
        Generator gen(eng, eng.mtp(), gopt);
        gen.prefill(prompt);   // warm-up: workspaces, graphs, cuBLAS
        const char * files = getenv("BNK_PB_FILES");
        std::string fl = files ? files : "";
        for (size_t a0 = 0, b; a0 < fl.size(); a0 = b + 1) {
            b = fl.find(',', a0);
            if (b == std::string::npos) b = fl.size();
            const std::string item = fl.substr(a0, b - a0);
            const size_t bar = item.find('|');
            const auto cold = read_tokens(item.substr(0, bar));
            eng.reset();
            Generator g1(eng, eng.mtp(), gopt);
            // BNK_PB_PROF=cold: the profiler's capture range covers the cold read instead of the follow-up
            static const bool pcold = getenv("BNK_PB_PROF") && !strcmp(getenv("BNK_PB_PROF"), "cold");
            if (pcold) cudaProfilerStart();
            double t0 = now_ms();
            g1.start(cold, SamplingParams{}, ckpt(cold));
            double t1 = now_ms();
            if (pcold) cudaProfilerStop();
            printf("cold   %7zu tokens: %8.1f ms  %7.1f tok/s\n", cold.size(), t1 - t0, cold.size() / ((t1 - t0) / 1000));
            fflush(stdout);
            if (bar == std::string::npos) continue;
            const auto fu = read_tokens(item.substr(bar + 1));
            Generator g2(eng, eng.mtp(), gopt);
            static const bool pprof = getenv("BNK_PB_PROF") && !pcold;   // nsys --capture-range=cudaProfilerApi
            if (pprof) cudaProfilerStart();
            t0 = now_ms();
            int tok = g2.start(fu, SamplingParams{}, ckpt(fu));
            t1 = now_ms();
            if (pprof) cudaProfilerStop();
            printf("follow %7zu new:    %8.1f ms  %7.1f tok/s\n", fu.size() - cold.size(), t1 - t0,
                   (fu.size() - cold.size()) / ((t1 - t0) / 1000));
            std::vector<int> out{tok};
            const int64_t m0 = eng.times.misses, sw0 = eng.cache_swaps();
            const double t2 = now_ms();
            while ((int) out.size() < max_new && out.back() != c.eos_token)
                for (int t : g2.next()) out.push_back(t);
            const double t3 = now_ms();
            const auto & gs = g2.stats;
            if (max_new > 1)
                printf("decode %7zu tokens: %8.1f ms  %7.1f tok/s  %.2f tok/round, verify %.2f draft %.2f ms/round\n",
                       out.size() - 1, t3 - t2, (out.size() - 1) / ((t3 - t2) / 1000),
                       (double) gs.emitted / std::max<int64_t>(1, gs.rounds), gs.verify_ms / std::max<int64_t>(1, gs.rounds),
                       gs.draft_ms / std::max<int64_t>(1, gs.rounds));
            printf("       expert misses %lld, cache swaps %lld\n", (long long) (eng.times.misses - m0),
                   (long long) (eng.cache_swaps() - sw0));
            eng.save_counts();   // the routing counts rank the next load's expert placement (--counts)
            fflush(stdout);
            // BNK_PB_DRAFTS="4:0.8,6:0.5": the same turn decoded again with other speculation settings (from the
            // snapshot where it starts)
            const char * dr = getenv("BNK_PB_DRAFTS");
            for (std::string ds = dr ? dr : ""; !ds.empty();) {
                const size_t cm = ds.find(',');
                const std::string one = ds.substr(0, cm);
                ds = cm == std::string::npos ? "" : ds.substr(cm + 1);
                GenOptions go = gopt;
                go.max_draft = std::stoi(one.substr(0, one.find(':')));
                go.min_p = std::stof(one.substr(one.find(':') + 1));
                Generator g3(eng, eng.mtp(), go);
                std::vector<int> o3{g3.start(fu, SamplingParams{}, ckpt(fu))};
                const int64_t n0 = eng.times.misses;
                const double a0 = now_ms();
                while ((int) o3.size() < max_new && o3.back() != c.eos_token)
                    for (int t : g3.next()) o3.push_back(t);
                const double a1 = now_ms();
                const auto & s3 = g3.stats;
                printf("  draft %d / %.2f: %7.1f tok/s  %.2f tok/round, verify %.2f draft %.2f ms/round%s\n", go.max_draft,
                       go.min_p, (o3.size() - 1) / ((a1 - a0) / 1000), (double) s3.emitted / std::max<int64_t>(1, s3.rounds),
                       s3.verify_ms / std::max<int64_t>(1, s3.rounds), s3.draft_ms / std::max<int64_t>(1, s3.rounds),
                       o3 == out ? "" : " (tokens differ)");
                printf("       expert misses %lld\n", (long long) (eng.times.misses - n0));
                fflush(stdout);
            }
        }
        print_stages(eng);
        return 0;
    }

    Generator gen(eng, eng.mtp(), gopt);
    const double t0 = now_ms();
    int tok = gen.prefill(prompt);
    gen.set_lookahead(!getenv("BNK_LOOKAHEAD") || atoi(getenv("BNK_LOOKAHEAD")) != 0);
    const double t1 = now_ms();
    std::vector<int> out{tok};
    eng.times = StageTimes{};
    // BNK_PROFILE_DECODE: limit a profiler's capture to the decode phase (nsys --capture-range=cudaProfilerApi)
    static const bool prof = getenv("BNK_PROFILE_DECODE") != nullptr;
    if (prof) cudaProfilerStart();
    const double t2 = now_ms();
    while ((int) out.size() < max_new && out.back() != c.eos_token) {
        for (int t : gen.next_lookahead()) {
            out.push_back(t);
            if (t == c.eos_token) break;
        }
    }
    const double t3 = now_ms();
    if (prof) cudaProfilerStop();
    printf("output:");
    for (int t : out) printf(" %d", t);
    printf("\nprefill %zu tokens in %.1f ms (%.1f tok/s)\n", prompt.size(), t1 - t0, prompt.size() / ((t1 - t0) / 1000));
    const auto & tm = eng.times;
    const auto & gs = gen.stats;
    printf("decode %zu tokens in %.1f ms (%.2f tok/s); %lld rounds, %.2f tokens/round, drafts accepted %lld/%lld"
           " (lookup %lld/%lld in %lld rounds, MTP %lld/%lld in %lld)\n",
           out.size() - 1, t3 - t2, (out.size() - 1) / ((t3 - t2) / 1000), (long long) gs.rounds,
           (double) gs.emitted / std::max<int64_t>(1, gs.rounds), (long long) gs.accepted, (long long) gs.drafted,
           (long long) gs.lookup_accepted, (long long) gs.lookup_drafted, (long long) gs.lookup_rounds,
           (long long) gs.mtp_accepted, (long long) gs.mtp_drafted, (long long) gs.mtp_rounds);
    if (gs.la_rounds) printf("lookahead: %lld of %lld rounds with a speculative next window confirmed (%.0f%%)\n",
                             (long long) gs.la_hits, (long long) gs.la_rounds, 100.0 * gs.la_hits / gs.la_rounds);
    printf("per round: verify %.2f ms, commit %.2f ms, draft %.2f ms; per forward: CPU experts %.2f ms, n-gram rows %.2f ms, "
           "mailbox wait %.2f ms\n",
           gs.verify_ms / std::max<int64_t>(1, gs.rounds), gs.commit_ms / std::max<int64_t>(1, gs.rounds),
           gs.draft_ms / std::max<int64_t>(1, gs.rounds), tm.cpu_experts_ms / std::max(1, tm.calls), tm.ple_ms / std::max(1, tm.calls),
           tm.wait_ms / std::max(1, tm.calls));
    printf("drafting: run %.2f ms per round, %.2f further steps per round at %.2f ms each\n",
           gs.draft_run_ms / std::max<int64_t>(1, gs.rounds), (double) gs.draft_steps / std::max<int64_t>(1, gs.rounds),
           (gs.draft_ms - gs.draft_run_ms) / std::max<int64_t>(1, gs.draft_steps));
    print_stages(eng);
    printf("expert misses %.2f%% (%.2f per forward), %lld cache swaps\n",
           100.0 * tm.misses / std::max<int64_t>(1, tm.routed), (double) tm.misses / std::max(1, tm.calls),
           (long long) eng.cache_swaps());
    return 0;
}
