// DSpark: DeepSeek-V4.1-Flash's own draft head as the DeepSeek engine's drafter (engine_api.h Drafter). docs/DSPARK.md.
//
// Three extra V4.1 blocks (mtp.0-2 of the checkpoint; llama.cpp's DFLASH GGUF layout from tools/dspark_gguf.py:
// stages blk.0-2, fc = main_proj, enc.output_norm = main_norm, output_norm, markov_w1/w2, conf_proj) draft a block of
// 5 tokens in one forward: the anchor (the target's last token) and 4 noise tokens, embedded with the target's token
// table, attend over a ring of the last 128 target positions' latents (kv_norm(wkv(main_norm(main_proj(h)))) where h
// is the mean over the hyper-connection copies of the target's residual at the attention input of layers 37-39) and
// over each other, run the stage's MoE (128 routed experts, top-3, the backbone's expert shape, plus the shared one),
// and go through the target's head; a rank-256 Markov head biases each position's logits by the token chosen before
// it, and a confidence head scores each position's acceptance. Routed experts live in RAM and run on the CPU expert
// pool (MXFP4 bit-exact); the dense weights on this GPU.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "core/util.h"
#include "cpu/expert_pool.h"
#include "deepseek/ds_model.h"
#include "engine/engine_api.h"
#include "kernels/gemv.h"

namespace bnk {

class DsSpark : public Drafter {
public:
    static constexpr int kMaxBlock = 8;
    ~DsSpark();
    // Loads the drafter GGUF for the target `m` (its config, token table and head; the stage holding the head and the
    // target layers). Dense weights to the current device; the routed experts into their own pinned host store, run
    // by `pool` (the engine's CPU expert pool). `slots`: conversation slots, each with its own rings.
    void load(const std::string & path, DsModel & m, CpuExpertPool & pool, int slots, int threads, cudaStream_t st, bool verbose);
    bool loaded() const { return loaded_; }
    int block() const { return block_; }                                  // drafts a forward (5)
    const std::vector<int> & target_layers() const { return targets_; }   // the backbone layers whose attention input it reads
    int n_targets() const { return (int) targets_.size(); }
    size_t vram_bytes() const { return vram_bytes_; }
    size_t host_bytes() const { return store_.total_bytes(); }

    // the target's rows: `targets` [kMaxWindow][n_targets * E] on this device (the engine's capture of the window's
    // rows at the target layers' attention input, mean over the hc copies; row i of the window), `res` / `res_dim` the
    // engine's residual rows (run() finds its window rows by R_rows - res), `rope` the plain rotary table, `tok_embd`
    // the target's token table (host-mapped, portable: readable from this device)
    // `targets`: the engine's position-indexed capture ring [nslots][n_targets * E] on this device (the window rows at
    // the target layers' attention input, mean over the hc copies, at slot position % nslots); `rope` the plain rotary
    // table; `tok_embd` the target's token table (host-mapped, portable: readable from this device)
    void bind(const float * targets, int nslots, const float * rope, const QMat & tok_embd) {
        targets_dev_ = targets; nslots_ = nslots; rope_ = rope; tok_embd_ = tok_embd;
    }
    // the positions [p_from, p_to) are captured (a prompt's tail): their latents into the rings
    void feed_range(int p_from, int p_to);
    void select(int slot) { cur_ = slot; }
    // the ring is valid up to position `pos` (a new or rolled-back conversation in the current slot)
    void restart(int pos);

    // Drafter: run() feeds the n verified rows' latents into the rings at cells cell0.. and, with `draft`, drafts a
    // block after the last of them (returns draft 0); step() hands out drafts 1.. of the block (-1 when the block
    // is spent or the confidence says stop).
    int run(const float * R_rows, const int32_t * next_tokens, int n, int cell0, bool draft, float * prob) override;
    int step(int32_t tok, int cell, float * prob) override;
    void run_multi(const std::vector<RunItem> & items, int32_t * tok_out, float * prob_out) override;
    void step_multi(const std::vector<StepItem> & items, int32_t * tok_out, float * prob_out) override;
    void set_temperature(float t) override { temperature_ = t; }
    const std::vector<std::pair<int32_t, float>> * draft_dist(int slot, int i) const override;

private:
    bool loaded_ = false;
    DsModel * main_ = nullptr;
    CpuExpertPool * pool_ = nullptr;
    GgufModel gguf_;
    DsConfig c_;                 // the drafter's hyper-parameters (the target's, with its own expert counts)
    std::vector<DsLayer> stages_;
    Model ev_;                   // the stages' routed experts as the expert store / CPU pool see them
    ExpertStore store_;
    QMat main_proj_, main_norm_, out_norm_, markov_w1_, markov_w2_, conf_proj_;
    std::vector<int> targets_;
    int block_ = 0;
    int32_t noise_id_ = 0;
    float temperature_ = 0.f;
    // the schedule (docs/DSPARK.md): draft i is kept while its survival probability s_i (the confidences' product) adds
    // tokens per millisecond above the round's running average: s_i >= row_ms * (1 + sum s_j) / (round_ms + draft_ms +
    // row_ms * (i - 1)). round_ms / row_ms: a plain round and a verify row (BNK_DS_SPARK_ROUND_MS / _ROW_MS, defaults
    // 40 / 10: the exchange's warm cache on this machine); draft_ms: this drafter's own time, measured. p_min_ (>= 0,
    // BNK_DS_SPARK_PMIN) is a floor on the product on top of that.
    float p_min_ = 0.f;
    float round_ms_ = 40.f, row_ms_ = 10.f, draft_ms_ = 12.f;
    size_t vram_bytes_ = 0;
    const float * targets_dev_ = nullptr;
    int nslots_ = 0;
    const float * rope_ = nullptr;
    QMat tok_embd_;
    cudaStream_t st_ = nullptr;
    int dev_ = 0;                 // the GPU holding the drafter (the stage with the head)
    std::vector<float> conf_w_;   // the confidence projection on the host [E + rank]

    // work buffers (T = block_ rows; the feed up to kMaxWindow rows)
    DevBuf<float> x_, res_blk_, mixed_, xn_, out_, pre_a_, pre_b_, attn_pre_, post_, comb_, mix_scratch_;
    DevBuf<float> qr_, qrn_, q_, kvr_, attn_o_, oa_;
    DevBuf<float> rlogits_, rw_, sg_, su_, shared_out_, cpu_out_, moe_out_, logits_, bias_, w1_row_, main_x_, main_xn_;
    DevBuf<int32_t> rids_, ids_, pos_;
    DevBuf<int8_t> actq_, actq2_;
    DevBuf<half> acth_, acth2_;
    DevBuf<float> actd_, actd2_;
    ActQ8 act_, act2_;
    std::vector<ExpertTask> tasks_;
    std::vector<float> h_x_, h_out_, h_logits_, h_bias_, h_mixed_, h_w1_;
    std::vector<int32_t> h_ids_;
    void stage_forward(int il, int T);      // one stage on the block rows (res_blk_, pre_)
    // BNK_DS_SPARK_DUMP=prefix: the first dumps_left_ draft calls' inputs and intermediates to prefix.N.* (the NumPy
    // reference tools/dspark_ref.py checks them against the checkpoint's math)
    int dumps_left_ = 0, dump_n_ = 0;
    std::string dump_prefix_;
    void dump(const char * what, const void * dev, size_t bytes);
    void dump_host(const char * what, const void * host, size_t bytes);
    float * pre_ = nullptr, * pre_next_ = nullptr;

    // per conversation slot: each stage's ring of the target positions' latents, and how far it is valid
    struct Slot {
        std::vector<DevBuf<half>> ring;   // [stage][kDsRing][D]
        int fed_lo = 0, fed_hi = 0;       // the ring holds positions [fed_lo, fed_hi) (contiguous; a gap restarts it)
    };
    std::vector<Slot> slots_;
    int cur_ = 0;

    // the block drafted last (per slot): tokens, their distributions, confidences
    struct Block {
        std::vector<int32_t> toks;
        std::vector<float> conf;
        std::vector<std::vector<std::pair<int32_t, float>>> dist;
        int next = 0;
    };
    std::vector<Block> blocks_;

    QMat upload(const std::string & name, bool required = true);
    void feed(int pos0, int n);   // the capture ring's rows at positions pos0..pos0+n-1 into the stages' rings
    void draft_block(int32_t anchor, int pos);                  // the block after the anchor at position pos
};

}  // namespace bnk
