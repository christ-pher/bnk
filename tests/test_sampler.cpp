// Sampler::distribution against the reference (sort every candidate, then min-p and top-p over the sorted list):
// the same tokens in the same order, the same probabilities, over random rows and parameters.
//   build/test_sampler
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "engine/sampler.h"

using namespace bnk;
using Dist = std::vector<std::pair<int32_t, float>>;

// the sampler before the head-only sort (2026-10-14), without penalties
static Dist reference(const std::vector<float> & l, const SamplingParams & sp) {
    const int V = (int) l.size();
    const float temp = std::max(sp.temperature, 1e-4f);
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    int k = sp.top_k > 0 ? std::min(sp.top_k, V) : V;
    if (k < V) {
        std::nth_element(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
        idx.resize(k);
    } else {
        const float mx = *std::max_element(l.begin(), l.end());
        size_t w = 0;
        for (int i = 0; i < V; ++i)
            if ((l[i] - mx) / temp > -30.f) idx[w++] = i;
        idx.resize(w);
    }
    Dist cand;
    for (int i : idx) cand.emplace_back(i, l[i]);
    std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) {
        return a.second > b.second || (a.second == b.second && a.first < b.first);
    });
    const float mx = cand[0].second;
    double z = 0;
    Dist out;
    for (const auto & c : cand) {
        const float p = std::exp((c.second - mx) / temp);
        out.emplace_back(c.first, p);
        z += p;
    }
    for (auto & c : out) c.second = (float) (c.second / z);
    if (sp.min_p > 0.f) {
        const float thr = sp.min_p * out[0].second;
        size_t w = 1;
        while (w < out.size() && out[w].second >= thr) ++w;
        out.resize(w);
    }
    if (sp.top_p < 1.f) {
        double c = 0;
        size_t w = 0;
        while (w < out.size()) {
            c += out[w].second;
            ++w;
            if (c >= sp.top_p) break;
        }
        out.resize(std::max<size_t>(w, 1));
    }
    double s = 0;
    for (auto & c : out) s += c.second;
    for (auto & c : out) c.second = (float) (c.second / s);
    return out;
}

int main() {
    std::mt19937 rng(7);
    const int V = 248320;
    std::vector<float> l(V);
    Sampler smp;
    Dist got;
    const std::vector<int32_t> recent;
    int cases = 0, bad = 0;
    for (int row = 0; row < 40; ++row) {
        // flat to peaked rows, some with ties (rounded logits)
        std::normal_distribution<float> nd(0.f, 0.5f + (row % 8));
        for (auto & x : l) x = nd(rng);
        if (row % 3 == 0) l[rng() % V] += 10.f + row;
        if (row % 5 == 0)
            for (auto & x : l) x = std::round(x * 4.f) / 4.f;
        for (float temp : {0.3f, 0.6f, 1.0f, 1.5f})
            for (int top_k : {0, 20, 1000})
                for (float top_p : {1.0f, 0.95f, 0.8f, 0.999f})
                    for (float min_p : {0.0f, 0.05f, 1e-4f}) {
                        if (top_p == 1.0f && min_p == 0.0f && top_k == 0 && row % 4) continue;   // full sorts: some
                        SamplingParams sp;
                        sp.temperature = temp;
                        sp.top_k = top_k;
                        sp.top_p = top_p;
                        sp.min_p = min_p;
                        Dist want = reference(l, sp);
                        smp.distribution(l.data(), V, sp, recent, got);
                        ++cases;
                        if (top_p == 1.0f && min_p == 0.0f) {   // nothing cut: the same tokens, in any order
                            std::sort(want.begin(), want.end());
                            std::sort(got.begin(), got.end());
                        }
                        bool ok = want.size() == got.size();
                        float err = 0;
                        for (size_t i = 0; ok && i < want.size(); ++i) {
                            ok = want[i].first == got[i].first;
                            err = std::max(err, std::fabs(want[i].second - got[i].second));
                        }
                        if (!ok || err > 1e-6f) {
                            if (++bad <= 10)
                                printf("MISMATCH row %d temp %.2f top_k %d top_p %.3f min_p %g: %zu vs %zu tokens, err %g\n",
                                       row, temp, top_k, top_p, min_p, want.size(), got.size(), err);
                        }
                    }
    }
    printf("%d cases, %d mismatches: %s\n", cases, bad, bad ? "FAIL" : "PASS");

    // draw_head_tail (the GPU's head with the rest of the row fetched only when a draw lands there) is distributed as
    // draw() / draw_residual() over the whole row: histograms of N draws, total variation against the exact
    // distribution, beside the same for the whole-row draw (the sampling noise)
    {
        std::mt19937 g(7);
        std::normal_distribution<float> nd(0.f, 2.f);
        const int V = 2000, K = 64, N = 400000;
        std::vector<float> l(V);
        for (auto & x : l) x = nd(g);
        SamplingParams sp;
        sp.temperature = 1.f;
        Sampler sm;
        sm.seed(11);
        Dist full;
        sm.distribution(l.data(), V, sp, {}, full);
        Dist sorted = full;
        std::sort(sorted.begin(), sorted.end(), [](const auto & a, const auto & b) { return a.second > b.second; });
        Dist head(sorted.begin(), sorted.begin() + K);
        double hs = 0;
        for (auto & c : head) hs += c.second;
        const double tail = 1.0 - hs;
        Dist q{{head[0].first, 0.5f}, {head[3].first, 0.3f}, {head[10].first, 0.2f}};   // a draft's distribution
        const int32_t excl = head[1].first;
        int whole_calls = 0;
        auto whole = [&]() -> const Dist & { ++whole_calls; return full; };
        // the exact targets
        auto target = [&](int mode) {
            std::vector<double> p(V, 0.0);
            double tot = 0;
            for (auto & c : full) {
                double v = c.second;
                if (mode == 1 && c.first == excl) v = 0;
                if (mode == 2) {
                    double qc = 0;
                    for (auto & d : q) if (d.first == c.first) qc = d.second;
                    v = std::max(0.0, v - qc);
                }
                p[c.first] = v;
                tot += v;
            }
            for (auto & v : p) v /= tot;
            return p;
        };
        auto tv = [&](const std::vector<int> & h, const std::vector<double> & p) {
            double d = 0;
            for (int t = 0; t < V; ++t) d += std::fabs((double) h[t] / N - p[t]);
            return d / 2;
        };
        const char * names[3] = {"plain", "excluding a token", "residual vs a draft"};
        for (int mode = 0; mode < 3; ++mode) {
            std::vector<int> ht(V, 0), hw(V, 0);
            for (int n = 0; n < N; ++n) {
                ht[sm.draw_head_tail(head, tail, mode == 1 ? excl : -1, mode == 2 ? &q : nullptr, whole)]++;
                hw[mode == 2 ? sm.draw_residual(full, q) : sm.draw(full, mode == 1 ? excl : -1)]++;
            }
            const auto p = target(mode);
            const double a = tv(ht, p), b = tv(hw, p);
            const bool ok = a < 1.5 * b + 0.002;
            bad += !ok;
            printf("head/tail draw, %s: total variation %.4f (whole-row draw %.4f); the row fetched for %.1f%% of draws "
                   "(tail mass %.3f) -> %s\n", names[mode], a, b, 100.0 * whole_calls / N, tail, ok ? "PASS" : "FAIL");
            whole_calls = 0;
        }
    }
    return bad ? 1 : 0;
}
