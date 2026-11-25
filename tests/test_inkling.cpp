// test_inkling MODEL TOKENS REF [prefill|decode|verify] [P]
// Compares the Inkling engine's logits with a reference dump (llama.cpp: every position's logits as f32 rows of the
// unpadded vocabulary, for the token ids in TOKENS). prefill: the prompt path over every token; decode: the first P
// tokens as a prompt, the rest one token at a time; verify: the rest in windows of 8, each committed in part
// (as a speculative decode would) and the remainder read again.
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "inkling/ink_engine.h"

using namespace bnk;

static std::vector<std::pair<double, int>> g_rows;   // (KL, position) of every row (WORST=n prints the n worst)
static int g_pos = 0, g_next = -1;
struct Stats;
static Stats * g_all = nullptr;
struct Stats {
    int n = 0, top1 = 0;
    double kl = 0, maxd = 0, nll_o = 0, nll_r = 0;
    int nn = 0;
    void add(const float * ours, const float * ref, int V) {
        int a = 0, b = 0;
        double mo = -1e30, mr = -1e30, d = 0;
        for (int i = 0; i < V; ++i) {
            if (ours[i] > ours[a]) a = i;
            if (ref[i] > ref[b]) b = i;
            mo = std::max(mo, (double) ours[i]);
            mr = std::max(mr, (double) ref[i]);
            d = std::max(d, (double) std::fabs(ours[i] - ref[i]));
        }
        double zo = 0, zr = 0;
        for (int i = 0; i < V; ++i) { zo += std::exp(ours[i] - mo); zr += std::exp(ref[i] - mr); }
        double k = 0;
        for (int i = 0; i < V; ++i) {
            const double pr = std::exp(ref[i] - mr) / zr;
            if (pr > 1e-12) k += pr * ((ref[i] - mr - std::log(zr)) - (ours[i] - mo - std::log(zo)));
        }
        if (g_next >= 0) {
            nll_o += -(ours[g_next] - mo - std::log(zo));
            nll_r += -(ref[g_next] - mr - std::log(zr));
            ++nn;
        }
        ++n;
        if (this == g_all) g_rows.push_back({k, g_pos});
        top1 += a == b;
        kl += k;
        maxd = std::max(maxd, d);
    }
    void print(const char * what) const {
        if (n) printf("%-22s %5d rows: top-1 agreement %.2f%%, mean KL %.5f, max |diff| %.3f; next-token NLL ours %.4f, ref %.4f\n",
                      what, n, 100.0 * top1 / n, kl / n, maxd, nn ? nll_o / nn : 0., nn ? nll_r / nn : 0.);
    }
};

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: test_inkling MODEL TOKENS REF [prefill|decode|verify] [P]\n");
        return 1;
    }
    const std::string mode = argc > 4 ? argv[4] : "prefill";
    std::vector<int32_t> toks;
    { std::ifstream f(argv[2]); int t; while (f >> t) toks.push_back(t); }
    const int N = (int) toks.size();
    const int P = argc > 5 ? atoi(argv[5]) : std::min(N - 1, 16);
    int fd = open(argv[3], O_RDONLY);
    if (fd < 0) { perror("ref"); return 1; }
    const size_t bytes = lseek(fd, 0, SEEK_END);
    const float * ref = (const float *) mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    const int V = (int) (bytes / 4 / N);

    EngineOptions opt;
    opt.max_ctx = N + 64;
    InkEngine eng;
    eng.load(argv[1], opt);
    if (eng.cfg().n_vocab != V) { fprintf(stderr, "vocabulary %d vs reference %d\n", eng.cfg().n_vocab, V); return 1; }
    std::vector<float> host((size_t) kInkWin * V);
    Stats near, far, all;
    g_all = &all;
    std::vector<Stats> bucket(64);   // per 256 positions
    auto check = [&](int pos0, int n, const float * dev) {
        cudaMemcpy(host.data(), dev, (size_t) n * V * 4, cudaMemcpyDeviceToHost);
        for (int r = 0; r < n; ++r) {
            const int p = pos0 + r;
            g_pos = p;
            g_next = p + 1 < N ? toks[p + 1] : -1;
            (p < 1024 ? near : far).add(host.data() + (size_t) r * V, ref + (size_t) p * V, V);
            all.add(host.data() + (size_t) r * V, ref + (size_t) p * V, V);
            bucket[std::min(63, p / 256)].add(host.data() + (size_t) r * V, ref + (size_t) p * V, V);
        }
    };
    const double t0 = now_ms();
    if (mode == "prefill") {
        eng.logits_hook = check;
        eng.prefill(toks);
    } else {
        eng.prefill(std::vector<int32_t>(toks.begin(), toks.begin() + P));
        eng.logits_rows_host(1, host.data(), eng.last_T - 1);
        all.add(host.data(), ref + (size_t) (P - 1) * V, V);
        if (mode == "decode") {
            for (int p = P; p < N; ++p) {
                eng.forward(&toks[p], 1, true);
                eng.logits_rows_host(1, host.data(), 0);
                (p < 1024 ? near : far).add(host.data(), ref + (size_t) p * V, V);
                all.add(host.data(), ref + (size_t) p * V, V);
            }
        } else {   // verify: windows of 8, commit 1 + (window index % 8) rows, the rest again next window
            int p = P, wi = 0;
            while (p < N) {
                const int T = std::min(8, N - p);
                eng.forward(&toks[p], T, false);
                eng.logits_rows_host(T, host.data(), 0);
                const int keep = std::min(T, 1 + (wi++ % 8));
                for (int r = 0; r < keep; ++r) {
                    (p + r < 1024 ? near : far).add(host.data() + (size_t) r * V, ref + (size_t) (p + r) * V, V);
                    all.add(host.data() + (size_t) r * V, ref + (size_t) (p + r) * V, V);
                }
                eng.commit(keep);
                p += keep;
            }
        }
    }
    const double ms = now_ms() - t0;
    printf("%s: %d tokens (P %d) in %.1f s\n", mode.c_str(), N, P, ms / 1000);
    near.print("positions < 1024");
    far.print("positions >= 1024");
    all.print("all");
    if (const char * w = getenv("WORST")) {
        std::sort(g_rows.rbegin(), g_rows.rend());
        for (int i = 0; i < std::min<int>(atoi(w), g_rows.size()); ++i)
            printf("  KL %.4f at position %d (token %d -> next %d)\n", g_rows[i].first, g_rows[i].second, toks[g_rows[i].second],
                   g_rows[i].second + 1 < N ? toks[g_rows[i].second + 1] : -1);
    }
    if (getenv("BUCKETS"))
        for (int b = 0; b < 64; ++b)
            if (bucket[b].n) { char nm[32]; snprintf(nm, sizeof nm, "  [%d, %d)", b * 256, b * 256 + 256); bucket[b].print(nm); }
    return 0;
}
