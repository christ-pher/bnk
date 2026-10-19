#include "engine/generate.h"

#include "core/util.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace bnk {

void Generator::recent_tail(std::vector<int32_t> & out) const {
    out.clear();
    const auto & h = eng_.history();
    const size_t n = std::min<size_t>(h.size(), 64);
    out.insert(out.end(), h.end() - n, h.end());
    out.insert(out.end(), emitted_tail_.begin(), emitted_tail_.end());
}

int Generator::gpu_k() const { return sp_.top_k > 0 ? sp_.top_k : EngineApi::kTopkMax; }

void Generator::gpu_candidates(int row0, int T) {
    const int K = gpu_k();
    topk_ids_.resize((size_t) T * K);
    topk_vals_.resize((size_t) T * K);
    topk_z_.resize(T);
    eng_.topk_rows_host(row0, T, K, topk_ids_.data(), topk_vals_.data(), sp_.top_k > 0 ? nullptr : topk_z_.data(),
                        std::max(sp_.temperature, 1e-4f));
}

bool Generator::row_distribution(int row, std::vector<std::pair<int32_t, float>> & dist) {
    const int K = gpu_k();
    const int32_t * ids = topk_ids_.data() + (size_t) row * K;
    const float * vals = topk_vals_.data() + (size_t) row * K;
    if (sp_.top_k > 0) {
        sampler_.distribution_topk(ids, vals, K, sp_, dist);
        return true;
    }
    return sampler_.distribution_head(ids, vals, K, topk_z_[row], sp_, dist);
}

int32_t Generator::pick(const float * row, std::vector<std::pair<int32_t, float>> & dist) {
    std::vector<int32_t> recent;
    recent_tail(recent);
    sampler_.distribution(row, eng_.cfg().n_vocab, sp_, recent, dist);
    return sampler_.draw(dist);
}

// ---------------------------------------------------------------------------------------------- lookup index
uint64_t LookupIndex::key(const int32_t * t) const {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n_; ++i) {
        h ^= (uint64_t) (uint32_t) t[i];
        h *= 1099511628211ull;
    }
    return h;
}

// Indexes the n-grams that end before seq's last token (the one ending at it is what find() looks up, so it
// must not be its own latest occurrence); covered tokens: seq[0, seq.size() - 1).
void LookupIndex::sync(const std::vector<int32_t> & seq) {
    const size_t target = seq.empty() ? 0 : seq.size() - 1;
    const size_t hn = std::min<size_t>(head_.size(), 8);
    bool same = target >= done_ && hn == std::min<size_t>(done_, 8);
    for (size_t i = 0; same && i < hn; ++i) same = head_[i] == seq[i];
    if (!same) {
        table_.clear();
        done_ = 0;
    }
    if (done_ == 0) {
        table_.reserve(seq.size() + seq.size() / 2);
        head_.assign(seq.begin(), seq.begin() + std::min<size_t>(target, 8));
    }
    for (size_t i = std::max<size_t>(done_ + 1, (size_t) n_); i <= target; ++i)   // the n-gram seq[i - n, i) -> i
        table_[key(seq.data() + i - n_)] = (int32_t) i;
    done_ = target;
}

int LookupIndex::find(const std::vector<int32_t> & seq, int * context) const {
    *context = 0;
    if (seq.size() < (size_t) n_ + 1) return -1;
    // the n-gram ending the sequence is in the table as its own latest occurrence only after sync(seq): the
    // table holds the sequence up to done_, so look the tail up against what precedes it
    auto it = table_.find(key(seq.data() + seq.size() - n_));
    if (it == table_.end()) return -1;
    const int p = it->second;
    if (p >= (int) seq.size()) return -1;
    for (int i = 0; i < n_; ++i)
        if (seq[p - n_ + i] != seq[seq.size() - n_ + i]) return -1;   // a hash collision
    int ctx = 0;
    while (ctx < 32 && p - n_ - 1 - ctx >= 0 && seq[p - n_ - 1 - ctx] == seq[(int) seq.size() - n_ - 1 - ctx]) ++ctx;
    *context = ctx;
    return p;
}

// ---------------------------------------------------------------------------------------------- drafting
void Generator::lookup_drafts(int32_t bonus) {
    lookup_.clear();
    lookup_ctx_ = 0;
    probing_ = getenv("BNK_LA_PROBE") != nullptr;
    if (opt_.lookup_n <= 0 || opt_.lookup_max <= 0) return;
    const auto & h = eng_.history();
    seq_.assign(h.begin(), h.end());
    seq_.push_back(bonus);
    index_.sync(seq_);   // the n-gram ending at the bonus token is now the latest occurrence of itself: find() looks before it
    int ctx = 0;
    const int p = index_.find(seq_, &ctx);
    if (p < 0) return;
    // BNK_LOOKUP_WIDE: 0 never wider than kMaxWindow, 1 (default) during a streak or when the match shares at least
    // BNK_LOOKUP_WIDE_CTX tokens of context before its n-gram (a verbatim copy), 2 always
    static const int wide = getenv("BNK_LOOKUP_WIDE") ? atoi(getenv("BNK_LOOKUP_WIDE")) : 1;
    static const int wide_ctx = getenv("BNK_LOOKUP_WIDE_CTX") ? atoi(getenv("BNK_LOOKUP_WIDE_CTX")) : 16;
    const bool w = wide == 2 || (wide == 1 && (lookup_streak_ || ctx >= wide_ctx));
    const int rows = w ? eng_.max_window() : std::min(eng_.max_window(), kMaxWindow);
    const int L = std::min<int>({opt_.lookup_max + (probing_ ? 1 : 0), (int) seq_.size() - p, rows - 1 + (probing_ ? 1 : 0)});
    if (L <= 0) return;
    lookup_.assign(seq_.begin() + p, seq_.begin() + p + L);
    lookup_ctx_ = ctx;
}

bool Generator::choose_drafts(int32_t mtp_first, float mtp_prob) {
    drafts_.clear();
    drafts_from_lookup_ = false;
    if (!lookup_.empty() && (lookup_ctx_ >= opt_.lookup_agree || lookup_[0] == mtp_first)) {
        drafts_ = lookup_;
        if (probing_ && (int) drafts_.size() > opt_.lookup_max) {   // the extra token is the probe, not a draft
            probe_ = drafts_.back();
            drafts_.pop_back();
        }
        drafts_from_lookup_ = true;
        return false;
    }
    if (opt_.max_draft <= 0) return false;
    drafts_.push_back(mtp_first);
    return (int) drafts_.size() < opt_.max_draft && mtp_prob >= opt_.min_p;
}

void Generator::count_round(int accepted, int drafted) {
    lookup_streak_ = drafts_from_lookup_ && drafted > 0 && accepted == drafted;
    if (drafts_from_lookup_) {
        stats.lookup_rounds++;
        stats.lookup_drafted += drafted;
        stats.lookup_accepted += accepted;
    } else if (drafted > 0) {
        stats.mtp_rounds++;
        stats.mtp_drafted += drafted;
        stats.mtp_accepted += accepted;
    }
}

void Generator::draft_from(const float * R_row, int32_t tok, int cell) {
    drafts_.clear();
    drafts_from_lookup_ = false;
    if (!mtp_ || opt_.max_draft <= 0) return;
    const double t0 = now_ms();
    float pr = 0.f;
    lookup_drafts(tok);
    int d = mtp_->run(R_row, &tok, 1, cell, true, &pr);
    if (choose_drafts(d, pr)) {
        while ((int) drafts_.size() < opt_.max_draft && pr >= opt_.min_p) {
            d = mtp_->step(d, ++cell, &pr);
            drafts_.push_back(d);
        }
    }
    stats.draft_ms += now_ms() - t0;
}

int32_t Generator::start(const std::vector<int32_t> & prompt, const SamplingParams & sp, const std::vector<int> & checkpoints) {
    static const char * en = getenv("BNK_LOOKUP_N"), * em = getenv("BNK_LOOKUP_MAX"), * ea = getenv("BNK_LOOKUP_AGREE");
    if (en || em || ea)   // experiments: the lookup's parameters for every request
        set_lookup(en ? atoi(en) : opt_.lookup_n, em ? atoi(em) : opt_.lookup_max, ea ? atoi(ea) : opt_.lookup_agree);
    use_slot();
    sp_ = sp;
    if (sp.seed) sampler_.seed(sp.seed);
    emitted_tail_.clear();
    drafts_.clear();
    eng_.select_conversation(prompt);   // another conversation's request: swap its parked state in
    // reuse what is already processed: all of it when the prompt extends the history, else rewind to the last
    // snapshot inside the shared prefix (a chat client re-renders earlier turns, e.g. without their reasoning)
    const auto & h = eng_.history();
    size_t common = 0;
    while (common < h.size() && common < prompt.size() && h[common] == prompt[common]) ++common;
    size_t keep = 0;
    if (!h.empty() && common == h.size() && h.size() < prompt.size()) {
        keep = h.size();
    } else if (common > 0) {
        const int r = eng_.rollback((int) std::min(common, prompt.size() - 1));
        if (r > 0) keep = (size_t) r;
    }
    if (keep == 0) eng_.reset();
    const double t0 = now_ms();
    static const bool prof = getenv("BNK_START_PROF") != nullptr;
    // the prompt in segments, a snapshot after each checkpoint position inside it (ascending, past what is kept)
    std::vector<size_t> cps;
    bool at_end = false;
    for (int c : checkpoints) {
        if (c < 0 || (size_t) c >= prompt.size()) { at_end = true; continue; }
        if ((size_t) c > keep && (cps.empty() || (size_t) c > cps.back())) cps.push_back((size_t) c);
    }
    size_t from = keep;
    for (size_t cp : cps) {
        const double a = now_ms();
        eng_.prefill(std::vector<int32_t>(prompt.begin() + from, prompt.begin() + cp));
        const double b = now_ms();
        eng_.checkpoint();
        if (prof)
            fprintf(stderr, "start: %zu new tokens %.0f ms (rebalance %.0f ms), snapshot at %zu %.0f ms\n",
                    cp - from, b - a, eng_.last_rebalance_ms, cp, now_ms() - b);
        from = cp;
    }
    if (from < prompt.size()) {
        const double a = now_ms();
        eng_.prefill(std::vector<int32_t>(prompt.begin() + from, prompt.end()));
        if (prof) fprintf(stderr, "start: last %zu tokens %.0f ms\n", prompt.size() - from, now_ms() - a);
    }
    if (at_end) eng_.checkpoint();
    stats.prefill_ms += now_ms() - t0;
    stats.prompt_tokens += (int64_t) prompt.size();
    stats.reused_tokens += (int64_t) keep;
    int32_t b;
    if (sp_.greedy()) {
        b = eng_.argmax(eng_.last_T - 1);
    } else {
        bool done = false;
        if (Sampler::gpu_ok(sp_, EngineApi::kTopkMax)) {
            gpu_candidates(eng_.last_T - 1, 1);
            if ((done = row_distribution(0, dist_))) b = sampler_.draw(dist_);
        }
        if (!done) {
            logits_host_ = eng_.logits_host(eng_.last_T - 1);
            b = pick(logits_host_.data(), dist_);
        }
    }
    pending_ = b;
    emitted_tail_.push_back(b);
    if (eng_.mtp_pending_cell() >= 0 && eng_.mtp_pending_cell() == eng_.pos() - 1)
        draft_from(eng_.mtp_pending_R(), b, eng_.mtp_pending_cell());
    return b;
}

std::vector<int32_t> Generator::next() {
    if (la_stage0_done_) fprintf(stderr, "bnk: BUG: a plain round with a half-done lookahead window (la_on %d)\n", (int) la_on_);
    window();
    const int T = (int) win_.size();
    use_slot();
    const double t0 = now_ms();
    eng_.forward_batch({EngineApi::BatchWin{eng_.current_slot(), win_.data(), T, T == 1}});
    stats.verify_ms += now_ms() - t0;
    return finish();
}

const std::vector<int32_t> & Generator::window(int max_rows) {
    if (max_rows <= 0) max_rows = eng_.max_window();
    use_slot();
    win_.clear();
    win_.push_back(pending_);
    for (int d : drafts_) {
        if ((int) win_.size() >= max_rows) break;
        win_.push_back(d);
    }
    if (eng_.pos() + (int) win_.size() > eng_.max_ctx()) win_.resize(1);
    return win_;
}

std::vector<int> Generator::share_rows(const std::vector<Generator *> & gens, int total) {
    const int n = (int) gens.size();
    if (n && total <= 0) total = gens[0]->eng_.max_window();
    std::vector<int> rows(n, 1);
    int left = total - n;
    for (bool more = true; left > 0 && more;) {
        more = false;
        for (int i = 0; i < n && left > 0; ++i)
            if (rows[i] - 1 < gens[i]->drafts()) {
                ++rows[i];
                --left;
                more = true;
            }
    }
    return rows;
}

int Generator::verify_window(int T, int r0, int32_t * bonus_out) {
    const std::vector<int32_t> & win = win_;
    double t0 = now_ms();
    int a = 0;
    int32_t bonus;
    if (sp_.greedy()) {
        argmax_buf_.resize(T);
        eng_.argmax_all(T, argmax_buf_.data(), r0);
        while (a < T - 1 && argmax_buf_[a] == win[a + 1]) ++a;
        bonus = argmax_buf_[a];
        stats.verify_ms += now_ms() - t0;
    } else {
        stats.verify_ms += now_ms() - t0;
        t0 = now_ms();
        const int V = eng_.cfg().n_vocab;
        // the top-k candidates of every row in one GPU pass (K values per row cross PCIe, not the vocabulary)
        // (without top-k: the 1,024 most likely and each row's normalizer, enough unless min-p and top-p keep more)
        const bool gpu = Sampler::gpu_ok(sp_, EngineApi::kTopkMax) && !getenv("BNK_HOST_SAMPLER");
        if (gpu) {
            gpu_candidates(r0, T);
        } else {
            logits_host_.resize((size_t) T * V);
            eng_.logits_rows_host(T, logits_host_.data(), r0);
        }
        bonus = -1;
        for (int i = 0; i < T; ++i) {
            if (!gpu) {
                std::vector<int32_t> recent;
                recent_tail(recent);
                sampler_.distribution(logits_host_.data() + (size_t) i * V, V, sp_, recent, dist_);
            } else if (!row_distribution(i, dist_)) {   // the cut lies beyond the GPU's candidates: the whole row
                logits_host_.resize(V);
                eng_.logits_rows_host(1, logits_host_.data(), r0 + i);
                std::vector<int32_t> recent;
                sampler_.distribution(logits_host_.data(), V, sp_, recent, dist_);
                ++stats.host_rows;
            }
            if (i == T - 1) { bonus = sampler_.draw(dist_); break; }
            const int32_t d = win[i + 1];
            float pd = 0.f;
            for (auto & c : dist_)
                if (c.first == d) { pd = c.second; break; }
            if (sampler_.uniform() < pd) {
                emitted_tail_.push_back(d);
                ++a;
                continue;
            }
            bonus = sampler_.draw(dist_, d);
            break;
        }
        stats.sample_ms += now_ms() - t0;
    }
    *bonus_out = bonus;
    return a;
}

std::vector<int32_t> Generator::finish(bool draft) {
    use_slot();
    const std::vector<int32_t> & win = win_;
    const int T = (int) win.size();
    const int P = eng_.pos() - T;          // the window's first position
    const int r0 = eng_.row0_of(eng_.current_slot());
    int32_t bonus = -1;
    const int a = verify_window(T, r0, &bonus);
    double t0 = now_ms();
    if (T > 1) eng_.commit(a + 1);
    stats.commit_ms += now_ms() - t0;

    std::vector<int32_t> out(win.begin() + 1, win.begin() + 1 + a);
    out.push_back(bonus);
    if (sp_.greedy()) emitted_tail_.insert(emitted_tail_.end(), out.begin(), out.end());
    else emitted_tail_.push_back(bonus);
    if (emitted_tail_.size() > 256) emitted_tail_.erase(emitted_tail_.begin(), emitted_tail_.end() - 128);
    stats.rounds++;
    stats.drafted += T - 1;
    stats.accepted += a;
    stats.emitted += (int64_t) out.size();
    count_round(a, T - 1);
    if (probing_ && T > 1) {
        stats.la_rounds++;
        if (a == T - 1 && probe_ == bonus) stats.la_hits++;
    }
    probe_ = -1;
    pending_ = bonus;
    drafts_.clear();
    drafts_from_lookup_ = false;
    job_.pending = false;
    if (mtp_ && opt_.max_draft > 0) lookup_drafts(bonus);   // what the lookup proposes after the bonus token
    if (mtp_ && !draft) {   // left for draft_batch(): the residual rows stay valid until the next forward
        std::vector<int32_t> nxt(win.begin() + 1, win.begin() + 1 + a);
        nxt.push_back(bonus);
        const float * R = eng_.residual_dev() + (size_t) r0 * eng_.res_dim();
        eng_.set_mtp_pending(R + (size_t) a * eng_.res_dim(), P + a);
        job_ = DraftJob{opt_.max_draft > 0, R, nxt, P};
        return out;
    }
    if (mtp_) {
        t0 = now_ms();
        // cells P..P+a: each kept row's main residual with the token that follows it
        std::vector<int32_t> nxt(win.begin() + 1, win.begin() + 1 + a);
        nxt.push_back(bonus);
        float pr = 0.f;
        const float * R = eng_.residual_dev() + (size_t) r0 * eng_.res_dim();
        int d = mtp_->run(R, nxt.data(), a + 1, P, opt_.max_draft > 0, &pr);
        stats.draft_run_ms += now_ms() - t0;
        eng_.set_mtp_pending(R + (size_t) a * eng_.res_dim(), P + a);
        int cell = P + a + 1;
        la_conf_ = drafts_from_lookup_ ? 1.0 : 0.0;
        if (opt_.max_draft > 0 && choose_drafts(d, pr)) {
            la_conf_ = pr;
            while ((int) drafts_.size() < opt_.max_draft && pr >= opt_.min_p) {
                d = mtp_->step(d, cell++, &pr);
                drafts_.push_back(d);
                stats.draft_steps++;
                la_conf_ *= pr;
            }
            if (probing_) probe_ = mtp_->step(d, cell++, &pr);
        }
        la_cell_ = cell;   // the chain-ahead continues from the last MTP step (lookup drafts: from their last token)
        la_last_ = drafts_.empty() ? bonus : drafts_.back();
        stats.draft_ms += now_ms() - t0;
    }
    return out;
}

// ------------------------------------------------------------------------------------------------- lookahead
std::vector<int32_t> Generator::next_lookahead() {
    if (!la_stage0_done_ && (!la_on_ || !mtp_ || !eng_.lookahead_ok())) return next();
    window();
    const int T = (int) win_.size();
    use_slot();
    if (la_stage0_done_ && (int) la_spec_.size() != T) la_stage0_done_ = false;   // (the context limit truncated it)
    // the window's first position: after a confirmed speculation the engine's history already holds the window
    const int P = eng_.pos() - (la_stage0_done_ ? T : 0);
    const int lane = la_lane_;
    // the chain-ahead: what the drafter proposes after this window's last draft, T' = the window length the next
    // round would use (the lookup's continuation when the drafts came from it, else the MTP chain)
    std::vector<int32_t> la_lookup;
    static const double la_min = getenv("BNK_LA_MIN") ? atof(getenv("BNK_LA_MIN")) : la_threshold_;
    // not worth a speculation (the drafter doubts its own drafts) and nothing pending: the plain round, whose
    // CPU experts are served in sequence rather than beside a chain
    if (!la_stage0_done_ && !drafts_from_lookup_ && la_conf_ < la_min) return next();
    if (drafts_from_lookup_) {
        // the lookup had its match: the tokens that followed the window's drafts in the earlier occurrence
        // (after a confirmed speculation the engine's history already holds this window)
        const auto & h = eng_.history();
        seq_.assign(h.begin(), h.end());
        if (!la_stage0_done_) seq_.insert(seq_.end(), win_.begin(), win_.end());
        int ctx = 0;   // the table holds the history's n-grams (synced by lookup_drafts): the tail may end in the window
        const int p = index_.find(seq_, &ctx);
        if (p >= 0) {
            const int L = std::min<int>({opt_.lookup_max + 1, (int) seq_.size() - p, eng_.max_window()});
            if (L > 0) la_lookup.assign(seq_.begin() + p, seq_.begin() + p + L);
        }
    }
    const int chain_T = std::min(eng_.max_window(), opt_.max_draft + 1);
    const bool spec_from_lookup = !la_lookup.empty();
    const bool worth = spec_from_lookup || la_conf_ >= la_min;
    auto chain = [&](std::vector<int32_t> & out) -> int {
        if (T <= 1 || !la_on_ || !worth) return 0;   // a window with no drafts: a speculation would need the bonus token
        if (!la_lookup.empty()) {
            out = la_lookup;
            return (int) out.size();
        }
        if (drafts_from_lookup_) return 0;   // the drafter's chain does not know the lookup's tokens
        int cell = la_cell_;
        float pr = 0.f;
        int32_t d = la_last_;
        for (int i = 0; i < chain_T; ++i) {
            d = mtp_->step(d, cell++, &pr);
            out.push_back(d);
        }
        return (int) out.size();
    };
    const double t0 = now_ms();
    eng_.forward_lookahead(EngineApi::BatchWin{eng_.current_slot(), win_.data(), T, T == 1}, lane, chain, la_stage0_done_);
    stats.verify_ms += now_ms() - t0;
    use_slot();
    const int r0 = eng_.row0_of(eng_.current_slot());
    int32_t bonus = -1;
    const int a = verify_window(T, r0, &bonus);
    std::vector<int32_t> out(win_.begin() + 1, win_.begin() + 1 + a);
    out.push_back(bonus);
    if (sp_.greedy()) emitted_tail_.insert(emitted_tail_.end(), out.begin(), out.end());
    else emitted_tail_.push_back(bonus);
    if (emitted_tail_.size() > 256) emitted_tail_.erase(emitted_tail_.begin(), emitted_tail_.end() - 128);
    stats.rounds++;
    stats.drafted += T - 1;
    stats.accepted += a;
    stats.emitted += (int64_t) out.size();
    count_round(a, T - 1);
    if (T > 1) la_full_ = 0.85 * la_full_ + 0.15 * (a == T - 1 ? 1.0 : 0.0);
    const bool spec = eng_.spec_pending();
    static const bool la_dbg = getenv("BNK_LA_DEBUG") != nullptr;
    std::vector<int32_t> spec_tokens;   // what the first stage ran (the lane buffers hold its rows)
    const int spec_lane = eng_.spec_lane();
    // the speculative window's tokens: the chain's output (kept by the engine's history: the last spec_T tokens)
    if (spec) {
        const auto & h = eng_.history();
        // history = ... + window + spec tokens
        const int nspec = (int) h.size() - (P + T);
        spec_tokens.assign(h.end() - nspec, h.end());
    }
    if (la_dbg)
        fprintf(stderr, "la-round: emitted %lld T %d accepted %d bonus %d spec %zu [%d ...] from_lookup %d conf %.2f stage0_done %d -> %s\n", (long long) stats.emitted, T, a, bonus,
                spec_tokens.size(), spec_tokens.empty() ? -1 : spec_tokens[0], (int) spec_from_lookup, la_conf_, (int) la_stage0_done_,
                !spec || spec_tokens.empty() ? "no spec" : (a == T - 1 && spec_tokens[0] == bonus) ? "HIT" : (a == T - 1 ? "miss: bonus" : "miss: rejected"));
    const std::vector<int32_t> nxt_rows(win_.begin() + 1, win_.begin() + 1 + a);
    const float * R = eng_.residual_dev() + (size_t) r0 * eng_.res_dim();
    if (spec && a == T - 1 && !spec_tokens.empty() && spec_tokens[0] == bonus) {
        // the speculation held: the second stage commits this window, the first already holds the next one
        stats.la_hits++;
        stats.la_rounds++;
        double t1 = now_ms();
        eng_.spec_confirm();
        stats.commit_ms += now_ms() - t1;
        t1 = now_ms();
        // the drafter: K/V for this window's rows from the main residuals, then through the speculative tokens
        // so its chain continues after them
        std::vector<int32_t> nxt = nxt_rows;
        nxt.push_back(bonus);
        float pr = 0.f;
        int d = mtp_->run(R, nxt.data(), a + 1, P, true, &pr);
        eng_.set_mtp_pending(R + (size_t) a * eng_.res_dim(), P + a);
        int cell = P + a + 1;
        la_conf_ = spec_from_lookup ? 1.0 : 1.0;
        for (size_t i = 1; i < spec_tokens.size(); ++i) {
            d = mtp_->step(spec_tokens[i], cell++, &pr);
            if (!spec_from_lookup) la_conf_ *= pr;   // how sure the drafter is of the chained drafts it is now verifying
        }
        (void) d;
        la_cell_ = cell;
        la_last_ = spec_tokens.back();
        stats.draft_ms += now_ms() - t1;
        pending_ = spec_tokens[0];
        drafts_.assign(spec_tokens.begin() + 1, spec_tokens.end());
        drafts_from_lookup_ = spec_from_lookup;
        la_spec_ = spec_tokens;
        la_lane_ = spec_lane;
        la_stage0_done_ = true;
        probe_ = -1;
        return out;
    }
    if (spec) {
        stats.la_rounds++;
        eng_.spec_discard();
    }
    double t1 = now_ms();
    if (T > 1) eng_.commit(a + 1);
    stats.commit_ms += now_ms() - t1;
    pending_ = bonus;
    drafts_.clear();
    drafts_from_lookup_ = false;
    probe_ = -1;
    la_stage0_done_ = false;
    la_spec_.clear();
    if (mtp_ && opt_.max_draft > 0) lookup_drafts(bonus);
    if (mtp_) {
        t1 = now_ms();
        std::vector<int32_t> nxt = nxt_rows;
        nxt.push_back(bonus);
        float pr = 0.f;
        int d = mtp_->run(R, nxt.data(), a + 1, P, opt_.max_draft > 0, &pr);
        stats.draft_run_ms += now_ms() - t1;
        eng_.set_mtp_pending(R + (size_t) a * eng_.res_dim(), P + a);
        int cell = P + a + 1;
        la_conf_ = drafts_from_lookup_ ? 1.0 : 0.0;
        if (opt_.max_draft > 0 && choose_drafts(d, pr)) {
            la_conf_ = pr;
            while ((int) drafts_.size() < opt_.max_draft && pr >= opt_.min_p) {
                d = mtp_->step(d, cell++, &pr);
                drafts_.push_back(d);
                stats.draft_steps++;
                la_conf_ *= pr;
            }
        }
        la_cell_ = cell;
        la_last_ = drafts_.empty() ? bonus : drafts_.back();
        stats.draft_ms += now_ms() - t1;
    }
    return out;
}

// Every generator's drafter pass in one forward, then further steps together for those still confident. Each
// generator's drafts are what its own run()/step() sequence would give (the rows are computed independently).
void Generator::draft_batch(const std::vector<Generator *> & gens) {
    std::vector<Generator *> g;
    for (Generator * x : gens)
        if (x->job_.pending && x->mtp_) g.push_back(x);
    if (g.empty()) return;
    Drafter * mtp = g[0]->mtp_;
    const double t0 = now_ms();
    std::vector<Drafter::RunItem> items;
    for (Generator * x : g)
        items.push_back({x->slot_, x->job_.R, x->job_.next.data(), (int) x->job_.next.size(), x->job_.cell0});
    std::vector<int32_t> tok(g.size());
    std::vector<float> pr(g.size());
    mtp->run_multi(items, tok.data(), pr.data());
    const double t1 = now_ms();
    std::vector<int> cell(g.size()), idx(g.size());   // idx: each generator's item index in the last call
    for (size_t i = 0; i < g.size(); ++i) {
        const bool more = g[i]->choose_drafts(tok[i], pr[i]);
        cell[i] = g[i]->job_.cell0 + (int) g[i]->job_.next.size();
        idx[i] = more ? (int) i : -1;
        g[i]->stats.draft_run_ms += (t1 - t0) / g.size();
    }
    while (true) {
        std::vector<Drafter::StepItem> st;
        std::vector<size_t> who;
        for (size_t i = 0; i < g.size(); ++i) {
            Generator * x = g[i];
            if (idx[i] >= 0 && (int) x->drafts_.size() < x->opt_.max_draft && pr[i] >= x->opt_.min_p) {
                st.push_back({idx[i], x->drafts_.back(), cell[i]++});
                who.push_back(i);
            } else {
                idx[i] = -1;   // done drafting
            }
        }
        if (st.empty()) break;
        std::vector<int32_t> t2(st.size());
        std::vector<float> p2(st.size());
        mtp->step_multi(st, t2.data(), p2.data());
        for (size_t j = 0; j < who.size(); ++j) {
            Generator * x = g[who[j]];
            x->drafts_.push_back(t2[j]);
            pr[who[j]] = p2[j];
            idx[who[j]] = (int) j;
            x->stats.draft_steps++;
        }
    }
    const double t2 = now_ms();
    for (Generator * x : g) {
        x->stats.draft_ms += (t2 - t0) / g.size();
        x->job_.pending = false;
    }
}

}  // namespace bnk
