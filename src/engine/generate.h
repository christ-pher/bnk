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
    double draft_run_ms = 0;   // the drafter over the kept rows (and the first draft)
    int64_t draft_steps = 0;   // further one-row draft steps
};

class Generator {
public:
    // slot: the engine's conversation slot this generator decodes in (-1: whichever is current)
    Generator(Engine & eng, MtpLayer * mtp, GenOptions opt = {}, int slot = -1)
        : eng_(eng), mtp_(mtp), opt_(opt), slot_(slot) {}

    // Starts a request: reuses the engine state when its history is a prefix of `prompt` (only the rest is
    // processed), otherwise starts over. Returns the first generated token.
    // `checkpoint_at`: where to snapshot the state for later requests to resume from (the start of the prompt's
    // last turn, which the next request will share); -1 = the end of the prompt.
    int32_t start(const std::vector<int32_t> & prompt, const SamplingParams & sp, int checkpoint_at = -1);
    // Emits the next tokens: 1 + accepted drafts.
    std::vector<int32_t> next();
    // The same round in two halves around Engine::forward_batch (several generators in one forward):
    // window() is this round's tokens (the pending one plus up to max_rows-1 drafts), finish() verifies them from
    // this slot's rows of the forward, commits, drafts the next ones and returns the emitted tokens.
    const std::vector<int32_t> & window(int max_rows = kMaxWindow);
    std::vector<int32_t> finish();
    int slot() const { return slot_; }
    int drafts() const { return (int) drafts_.size(); }
    // The rows of one batched forward shared out among generators: one each for the pending token, then the
    // drafts, one at a time round-robin, while rows are left (a generator gets no more rows than it has drafts).
    static std::vector<int> share_rows(const std::vector<Generator *> & gens, int total = kMaxWindow);
    // Back-compat for the CLI: start with greedy sampling on a fresh state.
    int32_t prefill(const std::vector<int32_t> & prompt) {
        use_slot();
        eng_.reset();
        return start(prompt, SamplingParams{});
    }
    void set_draft(int n) { opt_.max_draft = n; }
    void set_draft_min_p(float p) { opt_.min_p = p; }
    GenStats stats;

private:
    int32_t pick(const float * logits_row, std::vector<std::pair<int32_t, float>> & dist);
    void draft_from(const float * R_row, int32_t tok, int cell);
    void recent_tail(std::vector<int32_t> & out) const;

    Engine & eng_;
    MtpLayer * mtp_;
    GenOptions opt_;
    int slot_ = -1;
    std::vector<int32_t> win_;   // this round's window
    void use_slot() { if (slot_ >= 0) eng_.select(slot_); }
    SamplingParams sp_;
    Sampler sampler_;
    int32_t pending_ = -1;              // emitted, not yet processed by the main model
    std::vector<int32_t> drafts_;       // drafts for the next window
    std::vector<int32_t> argmax_buf_;
    std::vector<float> logits_host_;
    std::vector<int32_t> topk_ids_;
    std::vector<float> topk_vals_;
    // row `row`'s distribution: top-k on the GPU when the parameters allow it, else the whole row on the host
    void row_distribution(int row, std::vector<std::pair<int32_t, float>> & dist);
    std::vector<int32_t> emitted_tail_;
    std::vector<std::pair<int32_t, float>> dist_;
};

}  // namespace bnk
