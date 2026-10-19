// DFlash2 (z-lab/dflash, incoai/Qwen3.8-27B-DFlash2) as the dense engine's speculative drafter (engine_api.h Drafter).
//
// A 5-layer model that drafts a whole block in one pass: the target's residuals after layers 5/19/33/47/61 of every
// verified position, concatenated and projected (fc, then hidden_norm), are the K/V context of each draft layer
// (sliding window 2048, own K/V per conversation slot); the block is the last verified token's successor (the
// "anchor", known) and 7 mask tokens at the next 8 positions, attending to the context and to each other (not
// causal). Each layer wraps its attention and its FFN in a grouped two-tap convolution over the block's rows whose
// per-row kernel comes from the row itself. The block's last 7 rows go through the target's head; a rank-256
// selector then walks the top-16 candidates of each position, scoring each against the token chosen before it.
//
// fc is linear in the five taps, so the target adds fc_i . h_i into an accumulator as each tap's residual is ready
// (DenseEngine: tap()); the drafter reads the accumulator's rows of the verified tokens. Drafts only steer
// speculation: the verify window decides every emitted token.
#pragma once

#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "core/util.h"
#include "dense/dense_dflash_ops.h"
#include "dense/dense_model.h"
#include "engine/engine_api.h"
#include "kernels/gemv.h"

namespace bnk {

class DenseDFlash : public Drafter {
public:
    static constexpr int kTaps = 5, kBlock = 8, kRing = 4096;
    ~DenseDFlash();
    // `main`: the target (its token table and head; both must be on this GPU); `acc`: the target's decode tap
    // accumulator [kMaxWindowDense][E] and `res`: its residual rows (run() locates its rows by R_rows - res)
    void load(const std::string & path, const DenseModel & main, int slots, cudaStream_t st, bool verbose);
    void bind(const float * acc, const float * res) { acc_ = acc; res_ = res; }
    bool loaded() const { return loaded_; }
    size_t vram_bytes() const;
    const std::vector<int> & taps() const { return taps_; }
    const QMat & fc(int i) const { return fc_[i]; }
    int tap_index(int il) const;   // which tap layer il is, or -1

    void select(int slot) { cur_ = slot; }
    // the context is valid up to position `pos` (a new or rolled-back conversation in the current slot)
    // (positions mod kRing: after a rollback of more than kRing - window positions the slots below `pos` may hold
    // the old history's later positions, so the context starts at `pos` then)
    void restart(int pos) {
        if (fed_[cur_] - pos > kRing - win_) valid_lo_[cur_] = pos;
        else valid_lo_[cur_] = std::min(valid_lo_[cur_], pos);
        fed_[cur_] = std::min(fed_[cur_], pos);
    }
    // context rows from a tap accumulator (pre-norm): positions pos0..pos0+n-1 of the current slot
    void feed(const float * acc_rows, int n, int pos0);

    int run(const float * R_rows, const int32_t * next_tokens, int n, int cell0, bool draft, float * prob) override;
    int step(int32_t tok, int cell, float * prob) override;
    void run_multi(const std::vector<RunItem> & items, int32_t * tok_out, float * prob_out) override;
    void step_multi(const std::vector<StepItem> & items, int32_t * tok_out, float * prob_out) override;

private:
    struct Layer {
        QMat in_norm, post_norm, wq, wk, wv, wo, q_norm, k_norm, gate, up, down;
        QMat aconv_proj, mconv_proj;
        float * aconv_base = nullptr, * mconv_base = nullptr;   // [2 sides][2 taps][E], f32
    };

    void draft_block(int anchor, int s);   // the block anchored at `anchor` (position s): drafts_, probs_
    void enqueue_block();
    void ctx_rows(const float * acc_rows, int n, int pos0);

    bool loaded_ = false;
    int dev_ = 0;
    cudaStream_t st_ = nullptr;
    DenseModel w_;   // the drafter's GGUF and its uploads
    const DenseModel * main_ = nullptr;
    std::vector<int> taps_;
    QMat fc_[kTaps], hidden_norm_, norm_, sel_proj_;
    half * sel_pre_ = nullptr, * sel_succ_ = nullptr;   // [vocab][rank] f16
    std::vector<Layer> layers_;
    int E = 0, FF = 0, H = 0, HKV = 0, D = 0, win_ = 0, mask_ = 0, gsize_ = 0, rank_ = 0, topk_ = 0, nvocab_ = 0;
    float eps_ = 1e-6f, theta_ = 1e7f;
    const float * acc_ = nullptr, * res_ = nullptr;

    // per slot: K/V of the context and the block, by position mod kRing [layer][kRing][HKV][D] f16
    std::vector<DevBuf<half>> kc_, vc_;
    std::vector<int> fed_;        // positions fed so far (the context is valid below)
    std::vector<int> valid_lo_;   // and from here on
    int cur_ = 0;

    // block buffers [kBlock] rows; context buffers [kMaxWindowDense] rows
    DevBuf<float> x_, xn_, xc_, dyn_, q_, k_, v_, o_, a_, g_, u_, h_, scores_, logits_, hp_, cvals_, cn_, ck_, cv_;
    DevBuf<int32_t> cids_, tok_, out_;
    DevBuf<float> prob_;
    DevBuf<DFlashParams> par_;
    DFlashParams * h_par_ = nullptr;
    DevBuf<int> cpos_;   // the context rows' first position
    int * h_cpos_ = nullptr;
    int32_t * h_out_ = nullptr;
    float * h_prob_ = nullptr;
    int32_t * h_tok_ = nullptr;
    DevBuf<int8_t> actq_;
    DevBuf<float> actd_;
    DevBuf<int16_t> acts_;
    DevBuf<half> acth_;
    ActQ8 act_;
    std::vector<cudaGraphExec_t> graphs_;   // per slot: the block pass
    int drafts_[kBlock - 1] = {}, ndrafts_ = 0, next_ = 0;
    float probs_[kBlock - 1] = {};
    std::vector<std::vector<int>> multi_drafts_;
    std::vector<std::vector<float>> multi_probs_;
    std::vector<int> multi_next_;
    std::vector<int> mmap_;   // the last multi call's items: their run_multi index
};

}  // namespace bnk
