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

void Sampler::finish(const SamplingParams & sp, std::vector<std::pair<int32_t, float>> & out) {
    const float temp = std::max(sp.temperature, 1e-4f);
    // most likely first (ties: lower token id first, so both candidate paths agree)
    std::sort(cand_.begin(), cand_.end(), [](const auto & a, const auto & b) {
        return a.second > b.second || (a.second == b.second && a.first < b.first);
    });
    const float mx = cand_[0].second;
    double z = 0;
    out.reserve(cand_.size());
    for (const auto & c : cand_) {
        const float p = std::exp((c.second - mx) / temp);
        out.emplace_back(c.first, p);
        z += p;
    }
    for (auto & c : out) c.second = (float) (c.second / z);
    // min-p relative to the top probability, then top-p
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
