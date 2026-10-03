// The model's multi-token-prediction layer, used as the speculative drafter.
//
// Cell i pairs the main model's final multi-stream residual R_i with the token at position i+1 (rope position
// i) and predicts the token at i+2; its own output residual feeds the next draft step (vLLM qwen4_exp MTP):
//   e = fc_embedding(rms(embed(tok)) * (1+w))      h = fc_hidden per stream(rms_10240(R) * (1+w))
//   R = h + e -> attention HC -> attention (own K/V) -> MLP HC -> MoE (512 experts, top-k, shared) -> write
//   -> final mixer -> the main model's head
// Drafts only steer speculation: the verify window decides every emitted token.
#pragma once

#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "core/model.h"
#include "core/util.h"
#include "core/vmem.h"
#include "kernels/gemv.h"
#include "kernels/moe.h"
#include "kernels/qsa.h"

namespace bnk {

class MtpLayer {
public:
    ~MtpLayer();
    // Loads the MTP GGUF (tools/build_mtp.py: the checkpoint's mtp.* tensors) for the given main model.
    void load(const std::string & path, const Model & main, int max_ctx, cudaStream_t st, bool verbose,
              const std::string & draft_vocab = "", VramBudget * budget = nullptr);
    // K/V cells [0, cells) backed by VRAM (the address space covers the whole context)
    void ensure_ctx(int cells);
    void release_ctx(int cells);
    size_t ctx_bytes_needed(int cells) const;
    bool loaded() const { return loaded_; }
    size_t vram_bytes() const { return vram_; }

    // K/V for cells cell0..cell0+n-1 from main-model residual rows R (device, [n][hc*E]) and the tokens that
    // follow them; with `draft`, the full layer on the last row: returns its greedy draft (prob in *prob)
    // and keeps its residual for step(). Returns -1 when !draft.
    int run(const float * R_rows, const int32_t * next_tokens, int n, int cell0, bool draft, float * prob);
    // One more draft from the last residual, pairing it with `tok` at `cell`.
    int step(int32_t tok, int cell, float * prob);
    double ms = 0;
    int64_t calls = 0;

private:
    int forward(int n, int cell0, bool draft, float * prob);
    void enqueue(int n, bool draft);
    void hc_pre(const HcWeights & w, const float * res, int T, bool inject, float * mixed, float * inj);

    bool loaded_ = false;
    const Model * main_ = nullptr;
    cudaStream_t st_ = nullptr;
    size_t vram_ = 0;
    std::vector<void *> allocs_;
    int max_ctx_ = 0;

    QMat fc_emb_, fc_hid_, wq_, wk_, wv_, wo_, router_, sh_gate_, sh_up_, sh_down_, sh_gate_inp_;
    HcWeights hc_attn_, hc_mlp_, hc_mix_;
    float * w_emb_ = nullptr, * w_hid_ = nullptr, * q_norm_ = nullptr, * k_norm_ = nullptr;
    MoeLayerDesc moe_{};
    int32_t * slot_id_ = nullptr;

    // buffers ([kMaxWindow] rows unless noted)
    DevBuf<float> Rin_, emb_, en_, e2_, hn_, h2_, R_, xn_, lo_, gpre_, mixed_, inj_, inj2_;
    DevBuf<float> qfull_, k_, v_, q_, attn_o_, attn_scratch_, bo_, rlog_, rw_, sg_, su_, sh_, sgate_, shared_, y_;
    DevBuf<float> sample_, logits_, prob_;
    DevBuf<int32_t> ids_, tok_dev_, pos_dev_, out_dev_;
    DevBuf<int8_t> actq_;
    DevBuf<float> actd_;
    ActQ8 act_;
    ElasticBuf kv_k_, kv_v_, kv_raw_, kv_pool_;
    // sparse attention (the layer's own indexer, as the main model's full-attention layers)
    bool sparse_ = false;
    QsaShape qsh_{};
    int max_blocks_ = 0;
    QMat idx_q_, idx_k_;
    float * idx_q_norm_ = nullptr, * idx_k_norm_ = nullptr;
    DevBuf<half> kraw_;
    DevBuf<float> pooled_, ik_, iq_, scores_;
    DevBuf<int32_t> sel_, nsel_;
    size_t kv_cell_bytes_ = 0;
    DevBuf<half> kc_, vc_;
    MoeScratch moes_;
    DevBuf<uint8_t> hits_buf_;
    DevBuf<float> gu_buf_, hd_buf_, part_buf_;
    DevBuf<int8_t> hq_buf_;
    DevBuf<uint32_t> seq_dev_;
    MoeMsg * msg_ = nullptr;
    int32_t * h_io_ = nullptr;  // pinned: [0] token out, [8..15] tokens in, [16..17] cells
    float * h_prob_ = nullptr;
    // draft head over a token subset (rows of the main head), and the ids its rows stand for
    QMat dhead_;
    int32_t * dvocab_ = nullptr;
    int n_dvocab_ = 0;
    cudaGraphExec_t graphs_[kMaxWindow + 1][2] = {};
};

}  // namespace bnk
