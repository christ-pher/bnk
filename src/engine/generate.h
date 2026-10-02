// Token generation on top of Engine: plain greedy, or speculative with the MTP drafter (greedy verification:
// the output equals what plain greedy decoding of the same windows produces).
#pragma once

#include <functional>
#include <vector>

#include "engine/engine.h"
#include "engine/mtp.h"

namespace bnk {

struct GenOptions {
    int max_draft = 3;      // drafts per verify window (window = 1 + drafts)
    float min_p = 0.5f;     // stop drafting when the drafter's own probability falls below this
};

struct GenStats {
    int64_t rounds = 0, drafted = 0, accepted = 0, emitted = 0;
    double verify_ms = 0, commit_ms = 0, draft_ms = 0;
};

class Generator {
public:
    Generator(Engine & eng, MtpLayer * mtp, GenOptions opt = {}) : eng_(eng), mtp_(mtp), opt_(opt) {}

    // Processes the prompt; returns the first generated token (greedy).
    int prefill(const std::vector<int32_t> & prompt);
    // Emits the next tokens: 1 + accepted drafts. Call after prefill(); `last` is the token emitted last.
    std::vector<int32_t> next();
    GenStats stats;

private:
    Engine & eng_;
    MtpLayer * mtp_;
    GenOptions opt_;
    int32_t pending_ = -1;              // emitted, not yet processed by the main model
    std::vector<int32_t> drafts_;       // drafts for the next window
    float * last_R_ = nullptr;          // device: the main residual of the last processed row
    int last_cell_ = -1;
    std::vector<int32_t> argmax_buf_;
};

}  // namespace bnk
