#include "engine/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace bnk {

void Sampler::distribution(const float * logits, int n_vocab, const SamplingParams & sp,
                           const std::vector<int32_t> & recent, std::vector<std::pair<int32_t, float>> & out) {
    out.clear();
    scratch_.assign(logits, logits + n_vocab);
    float * l = scratch_.data();
    if (sp.repetition_penalty != 1.f || sp.presence_penalty != 0.f) {
        std::unordered_set<int32_t> seen(recent.begin(), recent.end());
        for (int32_t t : seen) {
            if (t < 0 || t >= n_vocab) continue;
            if (sp.repetition_penalty != 1.f) l[t] = l[t] > 0 ? l[t] / sp.repetition_penalty : l[t] * sp.repetition_penalty;
            l[t] -= sp.presence_penalty;
        }
    }
    const float temp = std::max(sp.temperature, 1e-4f);
    // candidates: top-k first (bounded), else everything within reach of the max
    int k = sp.top_k > 0 ? std::min(sp.top_k, n_vocab) : n_vocab;
    idx_.resize(n_vocab);
    std::iota(idx_.begin(), idx_.end(), 0);
    if (k < n_vocab) {
        std::nth_element(idx_.begin(), idx_.begin() + k, idx_.end(), [&](int a, int b) { return l[a] > l[b]; });
        idx_.resize(k);
    } else {
        // without top-k: keep tokens within 30 nats of the max at this temperature (the rest underflow)
        const float mx = *std::max_element(l, l + n_vocab);
        size_t w = 0;
        for (int i = 0; i < n_vocab; ++i)
            if ((l[i] - mx) / temp > -30.f) idx_[w++] = i;
        idx_.resize(w);
    }
    cand_.clear();
    for (int32_t i : idx_) cand_.emplace_back(i, l[i]);
    finish(sp, out);
}

void Sampler::distribution_topk(const int32_t * ids, const float * logits, int k, const SamplingParams & sp,
                                std::vector<std::pair<int32_t, float>> & out) {
    out.clear();
    cand_.clear();
    for (int i = 0; i < k; ++i) cand_.emplace_back(ids[i], logits[i]);
    finish(sp, out);
}

bool Sampler::distribution_head(const int32_t * ids, const float * logits, int k, double z,
                                const SamplingParams & sp, std::vector<std::pair<int32_t, float>> & out) {
    out.clear();
    cand_.clear();
    for (int i = 0; i < k; ++i) cand_.emplace_back(ids[i], logits[i]);
    return finish(sp, out, z);
}

bool Sampler::finish(const SamplingParams & sp, std::vector<std::pair<int32_t, float>> & out, double z_all) {
    const float temp = std::max(sp.temperature, 1e-4f);
    // most likely first (ties: lower token id first, so both candidate paths agree)
    const auto before = [](const std::pair<int32_t, float> & a, const std::pair<int32_t, float> & b) {
        return a.second > b.second || (a.second == b.second && a.first < b.first);
    };
    float mx = cand_[0].second;
    for (const auto & c : cand_) mx = std::max(mx, c.second);
    double z = z_all;
    if (z <= 0)
        for (const auto & c : cand_) z += std::exp((c.second - mx) / temp);
    // min-p relative to the top probability, then top-p, keep a prefix of the sorted candidates: only a head long
    // enough to hold the cut is sorted, grown while the cut lies beyond it (without top-k there are ~100K
    // candidates, and sorting all of them cost ~20 ms per row). With z_all, cand_ is already only such a head.
    const float thr = sp.min_p * (float) (1.0f / z);
    const size_t n = cand_.size();
    const bool trunc = sp.min_p > 0.f || sp.top_p < 1.f;
    size_t head = z_all <= 0 && trunc ? std::min<size_t>(n, 64) : n;
    bool cut = false;
    for (;;) {
        if (head < n) std::nth_element(cand_.begin(), cand_.begin() + (long) head, cand_.end(), before);
        // without min-p and top-p every candidate is kept: the order does not matter (draw() walks any order)
        if (trunc || z_all > 0) std::sort(cand_.begin(), cand_.begin() + (long) head, before);
        out.clear();
        double c = 0;
        cut = false;
        for (size_t w = 0; w < head && !cut; ++w) {
            // past the host's candidates (30 nats below the max): the head held all of them
            if (z_all > 0 && (cand_[w].second - mx) / temp <= -30.f) { cut = true; break; }
            const float p = (float) (std::exp((cand_[w].second - mx) / temp) / z);
            if (sp.min_p > 0.f && w > 0 && p < thr) { cut = true; break; }
            out.emplace_back(cand_[w].first, p);
            c += p;
            cut = sp.top_p < 1.f && c >= sp.top_p;
        }
        if (cut || head == n) break;
        head = std::min(n, head * 8);
    }
    if (z_all > 0 && !cut) return false;   // the cut lies beyond the head
    double s = 0;
    for (auto & c : out) s += c.second;
    for (auto & c : out) c.second = (float) (c.second / s);
    return true;
}

int32_t Sampler::draw(const std::vector<std::pair<int32_t, float>> & dist, int32_t exclude) {
    double total = 0;
    for (auto & c : dist)
        if (c.first != exclude) total += c.second;
    if (total <= 0) return dist.empty() ? 0 : dist[0].first;
    double u = std::uniform_real_distribution<double>(0.0, total)(rng_);
    for (auto & c : dist) {
        if (c.first == exclude) continue;
        u -= c.second;
        if (u <= 0) return c.first;
    }
    for (auto it = dist.rbegin(); it != dist.rend(); ++it)
        if (it->first != exclude) return it->first;
    return dist[0].first;
}

}  // namespace bnk
