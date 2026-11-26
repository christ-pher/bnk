// Inkling 2's DSpark drafter (dspark-Inkling-2-Mini-*.gguf, architecture "dflash" with a Markov context head) as the
// Inkling engine's Drafter (engine_api.h). The reference is llama.cpp's src/models/dflash.cpp (build_inject_ctx_mix,
// build_layers, build_logits) and src/models/dspark.cpp (build_dspark_markov_head with the rnn state, f_rnn), driven
// as common/speculative.cpp's common_speculative_impl_draft_dspark does.
//
// The target's features of every committed token (its residual at the input of layer visits 5, 23, 41 and its
// normed output, 4 x E) are injected: each of the 3 layers mixes them (ctx_mix, RMS norm, ctx_norm) into its own
// context K (RMS-normed per head, NeoX RoPE) and V; a fifth mix (markov_ctx_mix) is the Markov head's context state
// f_rnn. A draft block is [the last token, 7 mask rows] at positions pos0..: three dense layers (q / k RMS-normed
// per head, RoPE, attention over the context and the whole block, non-causal; layers 0 and 2 a 1,025-token window
// from the block's first row), the drafter's output norm, the target's head; then the Markov chain: a gated state
// over [state, markov_w1[previous token], the row's hidden, f_rnn] (joint_proj) biases each row's logits
// (markov_w2), soft-capped at 30, and the row's token is drawn from the top 64 at the target's temperature (greedy:
// the argmax); the drawn token conditions the next row.
#pragma once

#include <random>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "core/gguf_model.h"
#include "core/util.h"
#include "engine/engine_api.h"
#include "kernels/gemm.h"
#include "kernels/gemv.h"

namespace bnk {

class InkSpark : public Drafter {
public:
    static constexpr int kMaxBlock = 8;
    static constexpr int kCand = 64;     // the draws' candidates (q for the verifier)
    static constexpr int kTaps = 4;
    static constexpr int kMaxSlots = 4;
    ~InkSpark();
    // the vocabulary's share of a GPU: the target's head rows [v0, v0 + head.rows) on device dev (its stream st)
    struct HeadPart { int dev; cudaStream_t st; QMat head; int v0; };
    // the drafter GGUF; the target's token table; the head parts (the first on the current device, where the drafter
    // runs; one part: the whole vocabulary); max_ctx: the full layer's cache; chunk: the longest feed
    // slots: conversations (each its own context caches, Markov context and last block)
    void load(const std::string & path, const QMat & tok_embd, int n_vocab, const std::vector<HeadPart> & heads,
              int max_ctx, int chunk, bool verbose, int slots = 1);
    // the conversation the single-slot calls (feed, restart, save, load_state, run, step) work on
    void select(int slot);
    int current_slot() const { return cur_; }
    bool loaded() const { return loaded_; }
    const std::vector<int> & target_visits() const { return targets_; }   // 5, 23, 41, 48 (48: the normed output)
    size_t vram_bytes() const { return vram_bytes_; }

    // injection: n rows of target features (device, [n][4 E]) at positions pos0.. (contiguous with what is fed)
    void feed(const float * feats, int n, int pos0);
    // the context is valid below `pos` (a reset or a rollback)
    void restart(int pos) { fed_hi_ = pos; blk_ = Block{}; }
    int fed() const { return fed_hi_; }
    // snapshots (the window rings, f_rnn) for the engine's checkpoints
    void save(std::vector<uint8_t> & out) const;
    void load_state(const std::vector<uint8_t> & in);

    // Drafter
    int run(const float * R_rows, const int32_t * next_tokens, int n, int cell0, bool draft, float * prob) override;
    int step(int32_t tok, int cell, float * prob) override;
    void run_multi(const std::vector<RunItem> & items, int32_t * tok_out, float * prob_out) override;
    void step_multi(const std::vector<StepItem> & items, int32_t * tok_out, float * prob_out) override;
    void set_temperature(float t) override { temperature_ = t; }
    void set_max_drafts(int n) override { n_use_ = n; }
    const std::vector<std::pair<int32_t, float>> * draft_dist(int slot, int i) const override;
    int block_drafts() const { return n_draft_; }

private:
    bool loaded_ = false;
    GgufModel gguf_;
    cudaStream_t st_ = nullptr;
    int dev_ = 0;
    int E_ = 0, H_ = 0, KVH_ = 0, D_ = 128, F_ = 0, n_layer_ = 0, block_ = 8, rank_ = 0, n_vocab_ = 0, max_ctx_ = 0;
    int n_swa_ = 0, ring_swa_ = 2048, chunk_ = 512, n_draft_ = 7, n_use_ = 7;   // the block's drafts, those used
    float eps_ = 1e-6f, rope_base_ = 1e6f, softcap_ = 30.f, temperature_ = 0.f;
    std::vector<bool> swa_;
    std::vector<int> targets_;
    size_t vram_bytes_ = 0;
    std::vector<void *> allocs_;

    struct Layer {
        QMat qkv, kvctx, wo, gu, down;           // stacked q | k | v, k_ctx | v_ctx, gate | up (Q8_0)
        const float * attn_norm = nullptr, * q_norm = nullptr, * k_norm = nullptr, * k_ctx_norm = nullptr;
        const float * ctx_mix = nullptr, * ctx_norm = nullptr, * ffn_norm = nullptr;
        DevBuf<half> K, V;                        // context + block K / V, cells pos % ring
        int ring = 0, window = 0;
    };
    std::vector<Layer> layers_;
    QMat tok_embd_, w1_, joint_;
    const float * tok_norm_ = nullptr, * mask_embd_ = nullptr, * out_norm_ = nullptr;
    // per vocabulary part (GPU): the target's head rows, markov_w2's rows and bias, the draft rows' logits, a step's
    // bias / column / top 64; the second part also a copy of the block's hidden rows and of the step's Markov vector
    struct Part {
        int dev = 0, v0 = 0, vn = 0;
        cudaStream_t st = nullptr;
        QMat head, w2;
        const float * w2b = nullptr;
        DevBuf<float> logits, bias, col, topv, hid, markov;
        DevBuf<int32_t> topi;
        cudaEvent_t ev = nullptr;
        cudaEvent_t evs[kMaxBlock + 1] = {};   // per step: the input is ready (part 0) / the step is done (part 1)
        ActQ8 act;
    };
    std::vector<Part> parts_;
    const float * joint_b_ = nullptr, * mctx_mix_ = nullptr, * mctx_norm_ = nullptr;
    std::vector<float> h_mix_[5];                 // the mixes' weights on the host (layers, then the Markov head's)

    int fed_hi_ = 0;                              // positions [0, fed_hi_) are in the context
    Gemm gemm_;
    ActQ8 act_;
    DevBuf<int8_t> actq_;
    DevBuf<float> actd_;
    DevBuf<int16_t> acts_;
    DevBuf<half> acth_;
    DevBuf<float> mixed_, kv_, x_, xn_, qkv_, attn_, o_, gu_, h_, hid_, logits_, f_rnn_, state_, joint_in_, joint_out_,
        markov_, attn_scratch_, draw_p_, draw_u_;
    DevBuf<int32_t> tok_dev_, prev_, draw_tok_, draw_ids_, pos_dev_, in_dev_;   // in_dev_: the anchor, its position
    int32_t * h_in_ = nullptr;
    cudaGraphExec_t graphs_[kMaxSlots][kMaxBlock + 1] = {};   // the draft per slot and block rows
    bool warm_[kMaxBlock + 1] = {};
    void enqueue_draft(int nb);
    void finish_draft(int32_t anchor, int pos0, int nb, int V);
    float * h_u_ = nullptr;                       // pinned: the draws' uniforms and 1/temperature
    int32_t * h_out_ = nullptr;                   // pinned: tokens [kMaxBlock], then ids [kMaxBlock][kCand]
    float * h_probs_ = nullptr;                   // pinned: [kMaxBlock][kCand]
    std::mt19937 rng_{12345};

    // the block drafted last
    struct Block {
        std::vector<int32_t> toks;
        std::vector<float> prob;
        std::vector<std::vector<std::pair<int32_t, float>>> dist;
        int next = 0;
    };
    Block blk_;

    // conversations: the current one's state is live (layers_[].K / V, f_rnn_, fed_hi_, blk_); the others' are parked
    // here (the current slot's entry holds nothing): select swaps them
    struct Parked { std::vector<DevBuf<half>> K, V; DevBuf<float> f_rnn; int fed_hi = 0; Block blk; };
    std::vector<Parked> parked_;
    int cur_ = 0, n_slots_ = 1;
    std::vector<int> item_slot_;   // run_multi / step_multi: each item's slot (the next call's prev indices)

    void mm(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy);
    QMat upload(const std::string & name);
    QMat upload_stack(const std::vector<std::string> & names);
    QMat upload_rows_of(const GgufModel & g, const std::string & name, int64_t r0, int64_t r1);
    const float * upload_f32(const std::string & name, std::vector<float> * host = nullptr);
    void draft_block(int32_t anchor, int pos0);
    std::string dump_dir_;   // BNK_INK_SPARK_DUMP=dir: every fed row and the first drafts (tools: the NumPy reference)
    int dumps_ = 0;
};

}  // namespace bnk
