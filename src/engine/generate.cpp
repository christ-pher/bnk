#include "engine/generate.h"

#include <algorithm>
#include <cstring>

namespace bnk {

void Generator::recent_tail(std::vector<int32_t> & out) const {
    out.clear();
    const auto & h = eng_.history();
    const size_t n = std::min<size_t>(h.size(), 64);
    out.insert(out.end(), h.end() - n, h.end());
    out.insert(out.end(), emitted_tail_.begin(), emitted_tail_.end());
}

int32_t Generator::pick(const float * row, std::vector<std::pair<int32_t, float>> & dist) {
    std::vector<int32_t> recent;
    recent_tail(recent);
    sampler_.distribution(row, eng_.cfg().n_vocab, sp_, recent, dist);
    return sampler_.draw(dist);
}

void Generator::draft_from(const float * R_row, int32_t tok, int cell) {
    drafts_.clear();
    if (!mtp_ || opt_.max_draft <= 0) return;
    const double t0 = now_ms();
    float pr = 0.f;
    int d = mtp_->run(R_row, &tok, 1, cell, true, &pr);
    drafts_.push_back(d);
    while ((int) drafts_.size() < opt_.max_draft && pr >= opt_.min_p) {
        d = mtp_->step(d, ++cell, &pr);
        drafts_.push_back(d);
    }
    stats.draft_ms += now_ms() - t0;
}

int32_t Generator::start(const std::vector<int32_t> & prompt, const SamplingParams & sp, int checkpoint_at) {
    sp_ = sp;
    if (sp.seed) sampler_.seed(sp.seed);
    emitted_tail_.clear();
    drafts_.clear();
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
    const size_t cp = checkpoint_at < 0 ? prompt.size() : (size_t) checkpoint_at;
    if (cp > keep && cp < prompt.size()) {
        static const bool prof = getenv("BNK_START_PROF") != nullptr;
        const double a = now_ms();
        eng_.prefill(std::vector<int32_t>(prompt.begin() + keep, prompt.begin() + cp));
        const double b = now_ms();
        eng_.checkpoint();
        const double c = now_ms();
        eng_.prefill(std::vector<int32_t>(prompt.begin() + cp, prompt.end()));
        if (prof)
            fprintf(stderr, "start: %zu new tokens %.0f ms (rebalance %.0f ms), snapshot %.0f ms, last %zu tokens %.0f ms\n",
                    cp - keep, b - a, eng_.last_rebalance_ms, c - b, prompt.size() - cp, now_ms() - c);
    } else {
        eng_.prefill(std::vector<int32_t>(prompt.begin() + keep, prompt.end()));
        if (cp >= prompt.size()) eng_.checkpoint();
    }
    stats.prefill_ms += now_ms() - t0;
    stats.prompt_tokens += (int64_t) prompt.size();
    stats.reused_tokens += (int64_t) keep;
    int32_t b;
    if (sp_.greedy()) {
        b = eng_.argmax(eng_.last_T - 1);
    } else {
        logits_host_ = eng_.logits_host(eng_.last_T - 1);
        b = pick(logits_host_.data(), dist_);
    }
    pending_ = b;
    emitted_tail_.push_back(b);
    if (eng_.mtp_pending_cell() >= 0 && eng_.mtp_pending_cell() == eng_.pos() - 1)
        draft_from(eng_.mtp_pending_R(), b, eng_.mtp_pending_cell());
    return b;
}

std::vector<int32_t> Generator::next() {
    std::vector<int32_t> win;
    win.push_back(pending_);
    for (int d : drafts_) win.push_back(d);
    if (eng_.pos() + (int) win.size() > eng_.max_ctx()) win.resize(1);
    const int T = (int) win.size();
    const int P = eng_.pos();
    double t0 = now_ms();
    eng_.forward(win.data(), T, T == 1);
    int a = 0;
    int32_t bonus;
    if (sp_.greedy()) {
        argmax_buf_.resize(T);
        eng_.argmax_all(T, argmax_buf_.data());
        while (a < T - 1 && argmax_buf_[a] == win[a + 1]) ++a;
        bonus = argmax_buf_[a];
        stats.verify_ms += now_ms() - t0;
    } else {
        stats.verify_ms += now_ms() - t0;
        t0 = now_ms();
        const int V = eng_.cfg().n_vocab;
        logits_host_.resize((size_t) T * V);
        eng_.logits_rows_host(T, logits_host_.data());
        bonus = -1;
        for (int i = 0; i < T; ++i) {
            std::vector<int32_t> recent;
            recent_tail(recent);
            sampler_.distribution(logits_host_.data() + (size_t) i * V, V, sp_, recent, dist_);
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
    t0 = now_ms();
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
    pending_ = bonus;
    drafts_.clear();
    if (mtp_) {
        t0 = now_ms();
        // cells P..P+a: each kept row's main residual with the token that follows it
        std::vector<int32_t> nxt(win.begin() + 1, win.begin() + 1 + a);
        nxt.push_back(bonus);
        float pr = 0.f;
        int d = mtp_->run(eng_.residual_dev(), nxt.data(), a + 1, P, opt_.max_draft > 0, &pr);
        eng_.set_mtp_pending(eng_.residual_dev() + (size_t) a * eng_.cfg().hc_dim(), P + a);
        if (opt_.max_draft > 0) {
            drafts_.push_back(d);
            int cell = P + a + 1;
            while ((int) drafts_.size() < opt_.max_draft && pr >= opt_.min_p) {
                d = mtp_->step(d, cell++, &pr);
                drafts_.push_back(d);
            }
        }
        stats.draft_ms += now_ms() - t0;
    }
    return out;
}

}  // namespace bnk
