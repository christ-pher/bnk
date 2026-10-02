#include "engine/generate.h"

#include <algorithm>

namespace bnk {

int Generator::prefill(const std::vector<int32_t> & prompt) {
    drafts_.clear();
    eng_.prefill(prompt);
    const int b = eng_.argmax(eng_.last_T - 1);
    pending_ = b;
    if (mtp_ && eng_.mtp_pending_cell() >= 0) {
        const double t0 = now_ms();
        float pr = 0.f;
        int d = mtp_->run(eng_.mtp_pending_R(), &b, 1, eng_.mtp_pending_cell(), true, &pr);
        drafts_.push_back(d);
        int cell = eng_.mtp_pending_cell() + 1;
        while ((int) drafts_.size() < opt_.max_draft && pr >= opt_.min_p) {
            d = mtp_->step(d, cell++, &pr);
            drafts_.push_back(d);
        }
        stats.draft_ms += now_ms() - t0;
    }
    return b;
}

std::vector<int32_t> Generator::next() {
    std::vector<int32_t> win;
    win.push_back(pending_);
    for (int d : drafts_) win.push_back(d);
    const int T = (int) win.size();
    const int P = eng_.pos();
    double t0 = now_ms();
    eng_.forward(win.data(), T, T == 1);
    argmax_buf_.resize(T);
    eng_.argmax_all(T, argmax_buf_.data());
    int a = 0;
    while (a < T - 1 && argmax_buf_[a] == win[a + 1]) ++a;
    const int32_t bonus = argmax_buf_[a];
    stats.verify_ms += now_ms() - t0;
    t0 = now_ms();
    if (T > 1) eng_.commit(a + 1);
    stats.commit_ms += now_ms() - t0;

    std::vector<int32_t> out(win.begin() + 1, win.begin() + 1 + a);
    out.push_back(bonus);
    stats.rounds++;
    stats.drafted += T - 1;
    stats.accepted += a;
    stats.emitted += (int64_t) out.size();
    pending_ = bonus;
    drafts_.clear();
    if (mtp_) {
        t0 = now_ms();
        // cells P..P+a: the main residual of each kept row with the token that follows it
        std::vector<int32_t> nxt(win.begin() + 1, win.begin() + 1 + a);
        nxt.push_back(bonus);
        float pr = 0.f;
        int d = mtp_->run(eng_.residual_dev(), nxt.data(), a + 1, P, true, &pr);
        drafts_.push_back(d);
        int cell = P + a + 1;
        while ((int) drafts_.size() < opt_.max_draft && pr >= opt_.min_p) {
            d = mtp_->step(d, cell++, &pr);
            drafts_.push_back(d);
        }
        // keep the residual of the last kept row for nothing else: the next round recomputes from the window
        stats.draft_ms += now_ms() - t0;
    }
    return out;
}

}  // namespace bnk
