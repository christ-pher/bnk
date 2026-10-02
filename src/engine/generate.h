// Token generation on top of Engine: greedy or sampled, plain or speculative with the MTP drafter.
// Greedy verification keeps exactly the tokens plain greedy decoding of the same windows produces; sampled
// verification accepts a (greedy) draft d with probability p(d) and otherwise draws from p without d, which
// keeps the target distribution.
#pragma once

#include <vector>

#include "engine/engine.h"
#include "engine/mtp.h"
#include "engine/sampler.h"

namespace bnk {

struct GenOptions {
    int max_draft = 3;      // drafts per verify window (window = 1 + drafts)
    float min_p = 0.5f;     // stop drafting when the drafter's own probability falls below this
};

struct GenStats {
    int64_t rounds = 0, drafted = 0, accepted = 0, emitted = 0;
    int64_t prompt_tokens = 0, reused_tokens = 0;
    double prefill_ms = 0, verify_ms = 0, commit_ms = 0, draft_ms = 0, sample_ms = 0;
};

class Generator {
public:
    Generator(Engine & eng, MtpLayer * mtp, GenOptions opt = {}) : eng_(eng), mtp_(mtp), opt_(opt) {}

    // Starts a request: reuses the engine state when its history is a prefix of `prompt` (only the rest is
    // processed), otherwise starts over. Returns the first generated token.
    int32_t start(const std::vector<int32_t> & prompt, const SamplingParams & sp);
    // Emits the next tokens: 1 + accepted drafts.
    std::vector<int32_t> next();
    // Back-compat for the CLI: start with greedy sampling on a fresh state.
    int32_t prefill(const std::vector<int32_t> & prompt) {
        eng_.reset();
        return start(prompt, SamplingParams{});
    }
    void set_draft(int n) { opt_.max_draft = n; }
    GenStats stats;

private:
    int32_t pick(const float * logits_row, std::vector<std::pair<int32_t, float>> & dist);
    void draft_from(const float * R_row, int32_t tok, int cell);
    void recent_tail(std::vector<int32_t> & out) const;

    Engine & eng_;
    MtpLayer * mtp_;
    GenOptions opt_;
    SamplingParams sp_;
    Sampler sampler_;
    int32_t pending_ = -1;              // emitted, not yet processed by the main model
    std::vector<int32_t> drafts_;       // drafts for the next window
    std::vector<int32_t> argmax_buf_;
    std::vector<float> logits_host_;
    std::vector<int32_t> emitted_tail_;
    std::vector<std::pair<int32_t, float>> dist_;
};

}  // namespace bnk
