// The dense model's MTP block as the speculative drafter (engine_api.h Drafter): cell i pairs the main model's
// final residual R_i with the token at position i+1 and predicts the token at i+2.
//   x = eh_proj(cat(rms(embed(tok)) * enorm, rms(R) * hnorm)) -> attention layer (own K/V) -> FFN
//   -> rms(x) * shared_head_norm -> the main model's head
// Drafts only steer speculation: the verify window decides every emitted token.
#pragma once

#include <map>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "core/util.h"
#include "core/vmem.h"
#include "dense/dense_model.h"
#include "engine/engine_api.h"
#include "kernels/gemv.h"

namespace bnk {

class DenseMtp : public Drafter {
public:
    ~DenseMtp();
    // The block's weights are the main model's (DenseModel::mtp, loaded on the last stage's GPU).
    void load(const DenseModel & main, int max_ctx, cudaStream_t st, bool verbose, const std::string & draft_vocab,
              VramBudget * budget, int slots);
    void select(int slot) { cur_ = &slots_.at(slot); }
    void ensure_ctx(int cells);
    void release_ctx(int cells);
    size_t ctx_bytes_needed(int cells) const;
    size_t ctx_bytes_mapped() const { return cur_ ? cur_->kv_k.mapped() + cur_->kv_v.mapped() : 0; }
    bool loaded() const { return loaded_; }
    size_t vram_bytes() const { return vram_; }

    int run(const float * R_rows, const int32_t * next_tokens, int n, int cell0, bool draft, float * prob) override;
    int step(int32_t tok, int cell, float * prob) override;
    void run_multi(const std::vector<RunItem> & items, int32_t * tok_out, float * prob_out) override;
    void step_multi(const std::vector<StepItem> & items, int32_t * tok_out, float * prob_out) override;

private:
    int forward(int n, int cell0, bool draft, float * prob);
    void enqueue(int n, bool draft);
    void inputs(int N);   // the rows' eh_proj input from Rin_ and tok_dev_ -> R_
    struct MItem { int slot, n, cell0; };
    void forward_multi(const std::vector<MItem> & items, int32_t * tok_out, float * prob_out);
    void enqueue_multi(const std::vector<MItem> & items);
    void ensure_multi();
    std::vector<int> multi_slots_;
    struct MGraph { cudaGraphExec_t exec = nullptr; uint64_t used = 0; };
    std::map<std::vector<int>, MGraph> mgraphs_;
    uint64_t mgraph_age_ = 0;

    bool loaded_ = false;
    int dev_ = 0;
    const DenseModel * main_ = nullptr;
    cudaStream_t st_ = nullptr;
    size_t vram_ = 0;
    std::vector<void *> allocs_;
    int max_ctx_ = 0;
    size_t kv_cell_bytes_ = 0;

    // buffers (kMaxWindow rows unless noted)
    DevBuf<float> Rin_, emb_, cat_, R_, xn_, qfull_, k_, v_, q_, attn_o_, attn_scratch_, bo_, sg_, su_, y_, sample_,
        logits_, prob_;
    DevBuf<int32_t> tok_dev_, pos_dev_, out_dev_;
    DevBuf<int8_t> actq_;
    DevBuf<float> actd_;
    DevBuf<int16_t> acts_;
    DevBuf<half> acth_;
    ActQ8 act_;
    // batched drafting: the draft rows gathered
    DevBuf<float> Dres_, dattn_, dbo_, dxn_, dsg_, dsu_, dy_, dsample_, dlogits_, dprob_;
    DevBuf<int32_t> dout_, mpos_;
    int32_t * h_multi_ = nullptr;   // pinned: [0, 8) tokens in, [8, 24) cells (first, last) per item, [24, 32) out
    float * h_mprob_ = nullptr;
    int32_t * h_io_ = nullptr;      // pinned: [0] token out, [8..15] tokens in, [16..17] cells
    float * h_prob_ = nullptr;
    struct Slot {
        ElasticBuf kv_k, kv_v;
        DevBuf<half> kc, vc;
        cudaGraphExec_t graphs[kMaxWindow + 1][2] = {};
    };
    std::vector<Slot> slots_;
    Slot * cur_ = nullptr;
    // draft head over a token subset (rows of the main head), and the ids its rows stand for
    QMat dhead_;
    int32_t * dvocab_ = nullptr;
    int n_dvocab_ = 0;
};

}  // namespace bnk
