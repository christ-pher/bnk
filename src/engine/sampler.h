// Token sampling on the host: temperature, top-k, top-p, min-p, repetition/presence penalties.
#pragma once

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace bnk {

struct SamplingParams {
    float temperature = 0.f;  // 0 = greedy
    int top_k = 0;            // 0 = off
    float top_p = 1.f;
    float min_p = 0.f;
    float presence_penalty = 0.f;
    float repetition_penalty = 1.f;
    uint64_t seed = 0;
    bool greedy() const { return temperature <= 0.f; }
};

class Sampler {
public:
    void seed(uint64_t s) { rng_.seed(s); }
    // The truncated, renormalized distribution of one logits row: (token, prob), most likely first (in no particular
    // order without min-p and top-p, when nothing is cut).
    void distribution(const float * logits, int n_vocab, const SamplingParams & sp, const std::vector<int32_t> & recent,
                      std::vector<std::pair<int32_t, float>> & out);
    // The same from a row's top-k candidates (token, logit) chosen elsewhere (on the GPU): only valid without
    // penalties (they change which tokens are the top k) and with top_k > 0.
    void distribution_topk(const int32_t * ids, const float * logits, int k, const SamplingParams & sp,
                           std::vector<std::pair<int32_t, float>> & out);
    // The same without top-k (top_k 0) from a row's k most likely candidates and z, the sum of exp((logit - max) /
    // temperature) over every candidate (within 30 nats of the max): false when min-p and top-p keep more than
    // these k, and the row needs distribution() over the whole vocabulary.
    bool distribution_head(const int32_t * ids, const float * logits, int k, double z, const SamplingParams & sp,
                           std::vector<std::pair<int32_t, float>> & out);
    // the top-k candidates come from the GPU (no penalties; top_k 0 through distribution_head)
    static bool gpu_ok(const SamplingParams & sp, int max_k) {
        return sp.top_k >= 0 && sp.top_k <= max_k && sp.repetition_penalty == 1.f && sp.presence_penalty == 0.f;
    }
    static bool topk_ok(const SamplingParams & sp, int max_k) {
        return sp.top_k > 0 && sp.top_k <= max_k && sp.repetition_penalty == 1.f && sp.presence_penalty == 0.f;
    }
    int32_t draw(const std::vector<std::pair<int32_t, float>> & dist, int32_t exclude = -1);
    float uniform() { return std::uniform_real_distribution<float>(0.f, 1.f)(rng_); }

private:
    std::mt19937_64 rng_{std::random_device{}()};  // per process; a request's "seed" makes it reproducible
    std::vector<float> scratch_;
    std::vector<int32_t> idx_;
    std::vector<std::pair<int32_t, float>> cand_;
    // candidates (token, logit) in cand_ -> the distribution (temperature, min-p, top-p); z_all > 0: cand_ is the
    // head of the candidates and z_all their sum, false when the cut lies beyond the head
    bool finish(const SamplingParams & sp, std::vector<std::pair<int32_t, float>> & out, double z_all = 0);
};

}  // namespace bnk
