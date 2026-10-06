// bnk command line: run a tokenized prompt, greedy decode, and compare against llama.cpp dumps.
//   bnk run   --model M.gguf --tokens-file F [--max-new N] [--ctx N] [--threads N]
//   bnk check --model M.gguf --tokens-file F --ref PREFIX   (per-layer residual + logits vs llama_ref)
#include <cmath>
#include <cstdio>

#include <cuda_profiler_api.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "engine/engine.h"
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
    std::string mode = argv[1], model, tokfile, ref, mtp_path;
    bool solo = false, adapt_set = false;
    int max_new = 32;
    GenOptions gopt;
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
        else if (a == "--mtp") opt.mtp = next();
        else if (a == "--ple-gguf") opt.ple_gguf = next();
        else if (a == "--draft-vocab") opt.draft_vocab = next();
        else if (a == "--prefill-chunk-max") opt.prefill_chunk_max = std::stoi(next());
        else if (a == "--prefill-small") opt.prefill_small = std::stoi(next());
        else if (a == "--stage-small-mib") opt.stage_small_mib = std::stoi(next());
        else if (a == "--stage-full-min") opt.stage_full_min = std::stoi(next());
        else if (a == "--draft") gopt.max_draft = std::stoi(next());
        else if (a == "--min-p") gopt.min_p = std::stof(next());
        else if (a == "--counts") opt.counts_out = next();
        else if (a == "--cache-gib") opt.expert_cache_gib = std::stod(next());
        else if (a == "--no-graphs") opt.use_graphs = false;
        else if (a == "--prefill-chunk") opt.prefill_chunk = std::stoi(next());
        else if (a == "--adapt-every") opt.adapt_every = std::stoi(next()), adapt_set = true;
        else if (a == "--adapt-swaps") opt.adapt_swaps = std::stoi(next()), adapt_set = true;
        else if (a == "--park-gib") opt.park_gib = std::stod(next());
        else if (a == "--park-min") opt.park_min = std::stoi(next());
        else if (a == "--slots") opt.slots = std::stoi(next());
        else if (a == "--solo") solo = true;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (mode == "serve") {
        Engine eng;
        eng.load(model, opt);
        return serve_main(eng, gopt, eng.model().gguf.get_str("general.name", "bnk"));
    }
    // several conversations share the expert cache: it has to follow a mix of routings, so it adapts every forward
    // and swaps more at a time (3 batched agents: 72-76 -> 84 tok/s; one conversation: no measurable change)
    if (opt.slots > 1 && !adapt_set) {
        opt.adapt_every = 1;
        opt.adapt_swaps = 32;
    }
    auto prompt = mode == "multi" ? std::vector<int32_t>{} : read_tokens(tokfile);
    Engine eng;
    eng.load(model, opt);
    const Config & c = eng.cfg();

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
        if (solo) {
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
                if (even) rows.assign(act.size(), std::max(1, kMaxWindow / (int) act.size()));
                std::vector<Engine::BatchWin> wins;
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
               "expert misses %.2f%% (%.2f per forward)\n", solo ? "one after another" : "batched", total, t3 - t2,
               total / ((t3 - t2) / 1000), total / ((t3 - t2) / 1000) / (solo ? 1 : n), (long long) forwards,
               100.0 * tm.misses / std::max<int64_t>(1, tm.routed), (double) tm.misses / std::max(1, tm.calls));
        if (!solo)
            printf("per round: batched forward %.2f ms, verify + commit + drafting of every conversation %.2f ms\n",
                   fwd_ms / std::max<int64_t>(1, rounds), fin_ms / std::max<int64_t>(1, rounds));
        printf("forward graphs: %lld captured, %.1f MiB of device memory at capture\n", (long long) eng.graph_captures(),
               eng.graph_bytes() / 1048576.0);
        return 0;
    }

    Generator gen(eng, eng.mtp(), gopt);
    const double t0 = now_ms();
    int tok = gen.prefill(prompt);
    const double t1 = now_ms();
    std::vector<int> out{tok};
    eng.times = StageTimes{};
    // BNK_PROFILE_DECODE: limit a profiler's capture to the decode phase (nsys --capture-range=cudaProfilerApi)
    static const bool prof = getenv("BNK_PROFILE_DECODE") != nullptr;
    if (prof) cudaProfilerStart();
    const double t2 = now_ms();
    while ((int) out.size() < max_new && out.back() != c.eos_token) {
        for (int t : gen.next()) {
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
    printf("decode %zu tokens in %.1f ms (%.2f tok/s); %lld rounds, %.2f tokens/round, drafts accepted %lld/%lld\n",
           out.size() - 1, t3 - t2, (out.size() - 1) / ((t3 - t2) / 1000), (long long) gs.rounds,
           (double) gs.emitted / std::max<int64_t>(1, gs.rounds), (long long) gs.accepted, (long long) gs.drafted);
    printf("per round: verify %.2f ms, commit %.2f ms, draft %.2f ms; per forward: CPU experts %.2f ms\n",
           gs.verify_ms / std::max<int64_t>(1, gs.rounds), gs.commit_ms / std::max<int64_t>(1, gs.rounds),
           gs.draft_ms / std::max<int64_t>(1, gs.rounds), tm.cpu_experts_ms / std::max(1, tm.calls));
    printf("drafting: run %.2f ms per round, %.2f further steps per round at %.2f ms each\n",
           gs.draft_run_ms / std::max<int64_t>(1, gs.rounds), (double) gs.draft_steps / std::max<int64_t>(1, gs.rounds),
           (gs.draft_ms - gs.draft_run_ms) / std::max<int64_t>(1, gs.draft_steps));
    printf("expert misses %.2f%% (%.2f per forward), %lld cache swaps\n",
           100.0 * tm.misses / std::max<int64_t>(1, tm.routed), (double) tm.misses / std::max(1, tm.calls),
           (long long) eng.cache().swaps_done);
    return 0;
}
