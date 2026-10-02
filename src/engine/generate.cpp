#include "engine/generate.h"

#include <algorithm>

namespace bnk {

int Generator::prefill(const std::vector<int32_t> & prompt) {
    const Config & c = eng_.cfg();
    const int HC = c.hc_dim();
    if (!last_R_) CUDA_CHECK(cudaMalloc(&last_R_, (size_t) HC * 4));
    last_cell_ = -1;
    drafts_.clear();
    for (size_t i = 0; i < prompt.size(); i += kMaxWindow) {
        const int T = (int) std::min<size_t>(kMaxWindow, prompt.size() - i);
        const int p = eng_.pos();
        eng_.forward(prompt.data() + i, T, true);
        if (mtp_) {
            // the previous window's last row pairs with this window's first token
            if (last_cell_ >= 0) mtp_->run(last_R_, prompt.data() + i, 1, last_cell_, false, nullptr);
            if (T > 1) mtp_->run(eng_.residual_dev(), prompt.data() + i + 1, T - 1, p, false, nullptr);
            CUDA_CHECK(cudaMemcpyAsync(last_R_, eng_.residual_dev() + (size_t) (T - 1) * HC, (size_t) HC * 4,
                                       cudaMemcpyDeviceToDevice, eng_.stream()));
            last_cell_ = p + T - 1;
        }
    }
    const int b = eng_.argmax(eng_.last_T - 1);
    pending_ = b;
    if (mtp_) {
        const double t0 = now_ms();
        float pr = 0.f;
        int d = mtp_->run(last_R_, &b, 1, last_cell_, true, &pr);
        drafts_.push_back(d);
        int cell = last_cell_ + 1;
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
