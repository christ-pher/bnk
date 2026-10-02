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
    // The truncated, renormalized distribution of one logits row: (token, prob), most likely first.
    void distribution(const float * logits, int n_vocab, const SamplingParams & sp, const std::vector<int32_t> & recent,
                      std::vector<std::pair<int32_t, float>> & out);
    int32_t draw(const std::vector<std::pair<int32_t, float>> & dist, int32_t exclude = -1);
    float uniform() { return std::uniform_real_distribution<float>(0.f, 1.f)(rng_); }

private:
    std::mt19937_64 rng_{std::random_device{}()};  // per process; a request's "seed" makes it reproducible
    std::vector<float> scratch_;
    std::vector<int32_t> idx_;
};

}  // namespace bnk
