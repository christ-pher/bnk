// Token generation on top of Engine: greedy or sampled, plain or speculative with the MTP drafter.
// Greedy verification keeps exactly the tokens plain greedy decoding of the same windows produces; sampled
// verification accepts a (greedy) draft d with probability p(d) and otherwise draws from p without d, which
// keeps the target distribution.
#pragma once

#include <unordered_map>
#include <vector>

#include "engine/engine_api.h"
#include "engine/sampler.h"

namespace bnk {

struct GenOptions {
    int max_draft = 3;      // drafts per verify window (window = 1 + drafts)
    float min_p = 0.5f;     // stop drafting when the drafter's own probability falls below this
    // Prompt-lookup drafting: when the sequence's last `lookup_n` tokens occurred earlier in it, what followed
    // them then is proposed as the drafts (up to lookup_max, the window's other rows). Agents re-emit file
    // contents, identifiers and tool-call scaffolding; the MTP drafter proposes 4 tokens at ~75% each, a lookup
    // match proposes 7. The MTP drafter still runs every round (its K/V must follow the conversation) and its
    // first token decides between the two when the match is short (lookup_agree). 0 turns the lookup off.
    int lookup_n = 3;
    int lookup_max = 7;
    int lookup_agree = 6;   // a match with fewer than this many tokens of context needs the MTP's first token to agree
};

struct GenStats {
    int64_t rounds = 0, drafted = 0, accepted = 0, emitted = 0;
    int64_t prompt_tokens = 0, reused_tokens = 0;
    double prefill_ms = 0, verify_ms = 0, commit_ms = 0, draft_ms = 0, sample_ms = 0;
    double draft_run_ms = 0;   // the drafter over the kept rows (and the first draft)
    int64_t draft_steps = 0;   // further one-row draft steps
    int64_t host_rows = 0;     // rows sampled from the whole vocabulary on the host after the GPU candidates fell short
    // per draft source: rounds whose drafts came from the lookup / the MTP drafter, their drafts and accepted ones
    int64_t lookup_rounds = 0, lookup_drafted = 0, lookup_accepted = 0;
    int64_t mtp_rounds = 0, mtp_drafted = 0, mtp_accepted = 0;
    // lookahead probe (BNK_LA_PROBE=1): rounds with drafts, and those where every draft was accepted and the
    // bonus token equalled the drafter's next chained token (a speculative next window would have been right)
    int64_t la_rounds = 0, la_hits = 0;
};

// The n-grams of a token sequence, for prompt-lookup drafting: where the latest occurrence of each `n`-gram
// ends. Extended as the sequence grows; rebuilt when it shrinks (a rollback).
class LookupIndex {
public:
    explicit LookupIndex(int n = 3) : n_(n) {}
    void sync(const std::vector<int32_t> & seq);   // index every n-gram of seq (incrementally when seq extends the last one)
    // the latest earlier occurrence of seq's last n tokens: the position after it (the first token it predicts),
    // or -1; *context = tokens before the n-gram that agree with the sequence too (up to 32)
    int find(const std::vector<int32_t> & seq, int * context) const;
    size_t size() const { return done_; }
private:
    uint64_t key(const int32_t * t) const;
    int n_;
    size_t done_ = 0;                                   // seq tokens indexed so far
    std::vector<int32_t> head_;                         // seq[0, min(done_, 8)) to notice a replaced sequence
    std::unordered_map<uint64_t, int32_t> table_;       // n-gram -> the position after its latest occurrence
};

class Generator {
public:
    // slot: the engine's conversation slot this generator decodes in (-1: whichever is current)
    Generator(EngineApi & eng, Drafter * mtp, GenOptions opt = {}, int slot = -1)
        : eng_(eng), mtp_(mtp), opt_(opt), slot_(slot) {}

    // Starts a request: reuses the engine state when its history is a prefix of `prompt` (only the rest is
    // processed), otherwise starts over. Returns the first generated token.
    // `checkpoint_at`: where to snapshot the state for later requests to resume from (the start of the prompt's
    // last turn, which the next request will share); -1 = the end of the prompt.
    int32_t start(const std::vector<int32_t> & prompt, const SamplingParams & sp, int checkpoint_at = -1) {
        return start(prompt, sp, std::vector<int>{checkpoint_at});
    }
    // `checkpoints`: positions to snapshot at, ascending (the end of a system prompt other conversations share,
    // the start of the last turn); -1 = the end of the prompt
    int32_t start(const std::vector<int32_t> & prompt, const SamplingParams & sp, const std::vector<int> & checkpoints);
    // Emits the next tokens: 1 + accepted drafts.
    std::vector<int32_t> next();
    // The same with lookahead (EngineApi::forward_lookahead): while the second GPU verifies this window, the first
    // runs the window that follows if every draft is accepted; a confirmed speculation skips the first GPU's
    // half of the next round. Same tokens as next() in greedy mode. Falls back to next() when unsupported.
    std::vector<int32_t> next_lookahead();
    bool lookahead_on() const { return la_on_; }
    void set_lookahead(bool on) { la_on_ = on; }
    // a confirmed speculation left the next window's first-stage half done: the next round must be a
    // next_lookahead() round (with lookahead off it runs only the second stage, no new speculation)
    bool lookahead_pending() const { return la_stage0_done_; }
    // the request ends: a half-done next window is dropped (the engine keeps what the verified windows committed)
    void lookahead_abort() {
        if (la_stage0_done_) { use_slot(); eng_.spec_abort(); }
        la_stage0_done_ = false;
        la_spec_.clear();
    }
    // The same round in two halves around Engine::forward_batch (several generators in one forward):
    // window() is this round's tokens (the pending one plus up to max_rows-1 drafts), finish() verifies them from
    // this slot's rows of the forward, commits, drafts the next ones and returns the emitted tokens.
    const std::vector<int32_t> & window(int max_rows = kMaxWindow);
    // draft = false: verify and commit only; the next drafts come from draft_batch() (several generators' drafter
    // passes in one forward)
    std::vector<int32_t> finish(bool draft = true);
    static void draft_batch(const std::vector<Generator *> & gens);
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
    void set_lookup(int n, int max_drafts, int agree) { opt_.lookup_n = n; opt_.lookup_max = max_drafts; opt_.lookup_agree = agree; index_ = LookupIndex(std::max(n, 1)); }
    GenStats stats;

private:
    int32_t pick(const float * logits_row, std::vector<std::pair<int32_t, float>> & dist);
    void draft_from(const float * R_row, int32_t tok, int cell);
    // the lookup's proposal for what follows `bonus` (the token the next window starts with), into lookup_
    void lookup_drafts(int32_t bonus);
    // after the MTP's first draft of a round: picks the lookup's chain or the MTP's as this round's drafts; true
    // when the MTP chain is to be continued (step())
    bool choose_drafts(int32_t mtp_first, float mtp_prob);
    void count_round(int accepted, int drafted);
    void recent_tail(std::vector<int32_t> & out) const;

    EngineApi & eng_;
    Drafter * mtp_;
    GenOptions opt_;
    int slot_ = -1;
    std::vector<int32_t> win_;   // this round's window
    // the drafter's job after a round, left for draft_batch(): the kept rows' residuals, their next tokens, cells
    struct DraftJob { bool pending = false; const float * R = nullptr; std::vector<int32_t> next; int cell0 = 0; };
    DraftJob job_;
    LookupIndex index_{3};
    std::vector<int32_t> lookup_;       // the lookup's proposal for the next window (empty: none)
    int lookup_ctx_ = 0;                // its context length
    bool drafts_from_lookup_ = false;   // the current drafts_ came from the lookup
    int32_t probe_ = -1;                // the drafter's token after the window's last draft (the probe)
    bool probing_ = false;
    // lookahead
    bool la_on_ = false;
    int la_lane_ = 0;                   // the lane this round's window runs in
    bool la_stage0_done_ = false;       // the window's first-stage half ran as last round's confirmed speculation
    std::vector<int32_t> la_spec_;      // the speculative window run last round (the chain-ahead tokens)
    int la_cell_ = 0;                   // the drafter's next cell after the window's last draft
    int32_t la_last_ = -1;              // the window's last draft token (the chain starts from it)
    // speculate only when it is likely to hold: a missed speculation delays the next round by the speculative
    // forward (GPU 0 is busy with it), a hit saves GPU 0's half, so the break-even is about half the rounds.
    // Lookup drafts hold 85-92% of the time; MTP drafts follow the recent rate of full acceptance.
    double la_full_ = 0.5;              // EMA of "every draft accepted" over recent rounds
    double la_conf_ = 0.0;              // the drafter's own confidence in the window's drafts (the product of its probabilities)
    double la_threshold_ = 0.8;         // BNK_LA_MIN: speculate when the confidence is at least this
    int verify_window(int T, int r0, int32_t * bonus);   // accepted drafts of the window just run; *bonus = the next token
    std::vector<int32_t> seq_;          // scratch: history + the pending token
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
    // false: top_k 0 and min-p / top-p keep more than the GPU's candidates (the row needs the host sampler)
    bool row_distribution(int row, std::vector<std::pair<int32_t, float>> & dist);
    int gpu_k() const;                        // candidates per row from the GPU: top_k, or kTopkMax without top-k
    void gpu_candidates(int row0, int T);     // rows row0..row0+T-1: their candidates (and normalizers) from the GPU
    std::vector<double> topk_z_;
    std::vector<int32_t> emitted_tail_;
    std::vector<std::pair<int32_t, float>> dist_;
};

}  // namespace bnk
