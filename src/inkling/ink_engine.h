// The Inkling 2 engine (inkling2_loop): one conversation slot, the forward over a window of tokens (decode /
// verify, up to kInkWin) or a prompt chunk. A token visits 48 layers (the core three times); each visit has its own
// K/V cache (window layers: a ring of window + chunk cells; full layers: the context) and its short-convolution
// state. docs/INKLING-PLAN.md.
//
// Two GPUs (EngineOptions::gpus, the default when there are two): tensor parallel. Each rank holds half the query
// and KV heads (their caches and conv channels), half of every expert's FFN width and half the vocabulary; the
// depth mixes, norms, loop controls, routers and the post-attention / post-FFN convolutions run on both (the same
// inputs, the same results). Two all-reduces a visit: the attention's output projection and the MoE, each rank
// pushing its partial into the peer's inbox. Rank 0 is this object and drives both GPUs; a decode window is one
// CUDA graph over both devices.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "core/util.h"
#include "engine/engine.h"
#include "engine/engine_api.h"
#include "inkling/ink_model.h"
#include "inkling/ink_moe.h"
#include "inkling/ink_spark.h"
#include "vision/images.h"
#include "kernels/gemm.h"
#include "kernels/gemv.h"
#include "kernels/moe.h"

namespace bnk {

constexpr int kInkWin = 16;   // rows of a decode / verify window
constexpr int kInkAttnSplits = 32;   // most key splits of the attention
constexpr int kInkMaxSlots = 4;      // conversation slots (EngineOptions::slots)

class InkEngine : public EngineApi {
public:
    ~InkEngine();
    void load(const std::string & path, const EngineOptions & opt);

    // ---- EngineApi
    std::vector<int> devices() const override {
        return rank1_ ? std::vector<int>{dev_, rank1_->dev_} : std::vector<int>{dev_};
    }
    int device() const override { return dev_; }
    int last_device() const override { return dev_; }
    int stages() const override { return 1; }
    StageInfo stage_info(int i) const override;
    std::string model_name() const override { return model_.gguf.get_str("general.name", "bnk"); }
    const Config & cfg() const override { return model_.base; }
    // with the drafter: the rows it reads are the target's features of the last forward ([T][4 E], residual_dev)
    int res_dim() const override { return spark_ ? InkSpark::kTaps * model_.cfg.E : model_.cfg.E; }
    Drafter * mtp() override { return spark_ ? spark_.get() : nullptr; }
    int max_ctx() const override { return opt_.max_ctx; }
    int max_window() const override { return kInkWin; }
    int snap_every() const override { return 0; }
    int cpu_threads() const override { return 0; }
    int64_t graph_captures() const override { return captures_; }
    size_t graph_bytes() const override { return 0; }

    // slots: each conversation its own caches and conv states on both GPUs, history, checkpoints, drafter state and
    // graphs; select swaps the current one's buffers with the slot's (nothing is copied). A batched round runs the
    // slots' windows one after the other, their logits / features rows side by side (row0_of)
    int slots() const override { return n_slots_; }
    int current_slot() const override { return cur_; }
    void select(int slot) override;
    void reset() override;
    int pos() const override { return (int) history_.size(); }
    const std::vector<int32_t> & history() const override { return history_; }
    void checkpoint() override;
    int rollback(int max_pos) override;
    void select_conversation(const std::vector<int32_t> &) override {}
    int reusable(const std::vector<int32_t> & prompt) const override;
    void set_busy(const std::vector<int> &) override {}
    int parked() const override { return 0; }
    size_t parked_bytes() const override { return 0; }
    void drop_parked_all() override {}
    void save_counts() override {}

    void prefill(const std::vector<int32_t> & tokens) override;
    void forward(const int32_t * tokens, int T, bool commit_all = true) override;
    void forward_batch(const std::vector<BatchWin> & wins) override;
    void forward_pipe(const std::vector<BatchWin> &, int, const std::vector<BatchWin> &, int) override {
        throw std::runtime_error("inkling: no pipelined lanes");
    }
    int row0_of(int slot) const override { return slot >= 0 && slot < kInkMaxSlots ? row0_[slot] : 0; }
    void commit(int c) override;
    const float * residual_dev() const override { return spark_ ? feat_out_ : nullptr; }
    void set_mtp_pending(const float *, int) override {}
    const float * mtp_pending_R() const override { return nullptr; }
    int mtp_pending_cell() const override { return -1; }

    int argmax(int t) override;
    void argmax_all(int T, int32_t * out, int row0 = 0) override;
    std::vector<float> logits_host(int t) override;
    void logits_rows_host(int T, float * out, int row0 = 0) override;
    void topk_rows_host(int row0, int T, int K, int32_t * ids, float * vals, double * z = nullptr, float temp = 1.f) override;

    int64_t cache_swaps() const override { return 0; }
    int cache_slots(int) const override { return 0; }
    int cache_resident() const override { return 0; }
    size_t cache_bytes() const override { return 0; }
    size_t kv_bytes_mapped() const override { return kv_bytes_; }
    ImageStore * images() override { return images_ && images_->enabled() ? images_.get() : nullptr; }

    // tests: when set, every prompt row's logits (device, [n][n_vocab]) for rows pos0.. as they are computed
    std::function<void(int pos0, int n, const float * logits)> logits_hook;

private:
    int dev_ = 0;
    // ---- tensor parallel
    int tp_n_ = 1, tp_rank_ = 0;
    InkEngine * peer_ = nullptr;           // the other rank
    std::unique_ptr<InkEngine> rank1_;     // owned by rank 0
    bool p2p_ = false;
    // the DSpark drafter (opt.mtp: dspark-Inkling-2-Mini-*.gguf) on rank 0, and the target features it reads: the
    // stream at the input of visits 5, 23, 41 (in visit order) and the normed output, [chunk][4 E]
    std::unique_ptr<InkSpark> spark_;
    // images (opt.mmproj: the inkling2v encoder) on rank 0's GPU; their rows replace the embedding's (unnormed)
    std::unique_ptr<ImageStore> images_;
    std::vector<int32_t> seq_;          // the sequence with the window (the image rows' placement)
    bool window_images_ = false;
    DevBuf<float> feat_;
    int tap_of_visit_[64];
    DevBuf<float> inbox_[2][2];            // [point: attention / MoE][visit parity]: the peer's partial [W][E]
    std::vector<cudaEvent_t> ev_[2];       // per point and visit: this rank's partial is in the peer's inbox
    cudaEvent_t ev_head_ = nullptr, ev_fork_ = nullptr, ev_join_ = nullptr;
    DevBuf<float> head_part_;              // rank 1: its vocabulary rows' logits [kInkWin][vocab_n]
    std::vector<InkEngine *> ranks() { return rank1_ ? std::vector<InkEngine *>{this, rank1_.get()} : std::vector<InkEngine *>{this}; }
    void load_rank(const std::string & path, const EngineOptions & opt, int dev, int tp_n, int tp_rank);
    void send(int point, int v, const float * buf, int T);
    void recv(int point, int v, float * buf, int T);
    // the visit in three phases around the exchanges (layer() runs them back to back on one GPU)
    std::vector<const float *> blocks_, prelude_blocks_, bank_;
    float * prelude_ = nullptr, * bank_out_ = nullptr;
    void fwd_in(int T);
    void pass_begin(int pass, int T);
    void visit_a(int v, int ip, int T, int pos0, const InkLoop * loop, bool keep);   // to the out projection's partial
    void visit_b(int v, int ip, int T, const InkLoop * loop, bool keep);             // to the MoE's partial
    void visit_c(int v, int T, bool keep);                                           // the FFN's conv and residual
    void head_prep(int T);
    void head_rows(int r0, int n);
    std::pair<int, int> hook_last_{0, 0};
    void reset_here();
    void save_ckpt(int slot, int pos);
    void load_ckpt(int slot);
    void commit_here(int cnt);
    void controls(const InkLoop * loop, int j0, int n, const float * const * cur, const float * const * msg,
                  float * const * out, const float * const * norm, int T);
    EngineOptions opt_;
    InkModel model_;
    cudaStream_t st_ = nullptr;
    Gemm gemm_;
    ActQ8 act_;
    DevBuf<int8_t> actq_;
    DevBuf<float> actd_;
    DevBuf<int16_t> acts_;
    DevBuf<half> acth_;
    int chunk_ = 512;        // prompt chunk rows (the buffers' capacity)
    int ring_ = 0;           // window layers' ring cells
    size_t kv_bytes_ = 0;
    int conv_ch_ = 0;        // channels of a visit's conv state: k | v | attention out | FFN out

    // per visit (cache slot): K/V cache, conv state [3][conv_ch_], the pending window's conv inputs [kInkWin][conv_ch_]
    struct Visit {
        DevBuf<half> K, V;
        int ring = 0, window = 0;
        DevBuf<float> conv, win_in;
    };
    std::vector<Visit> visits_;
    std::vector<int32_t> history_;
    int pending_T_ = 0;      // rows of an uncommitted verify window
    void alloc_visits(std::vector<Visit> & vs);

    // checkpoints: conv states and the window rings on the host
    struct Ckpt { int pos = -1; std::vector<float> conv; std::vector<half> ring; std::vector<uint8_t> spark; int64_t age = 0; };
    std::vector<Ckpt> ckpts_;
    int64_t ckpt_age_ = 0;

    // the other slots' conversations, parked (the current slot's entry is empty); rank 1 parks its visits and
    // checkpoints only
    struct Parked { std::vector<Visit> visits; std::vector<int32_t> history; int pending_T = 0; std::vector<Ckpt> ckpts; int64_t ckpt_age = 0; };
    std::vector<Parked> parked_;
    int n_slots_ = 1, cur_ = 0;
    void select_here(int slot);
    int row0_[kInkMaxSlots] = {};
    float * lg_ = nullptr;             // the logits the readers see: logits_, or a batched round's blogits_
    const float * feat_out_ = nullptr; // the features the drafter reads: feat_, or a batched round's bfeat_
    DevBuf<float> blogits_, bfeat_;

    // activations (capacity chunk_ rows)
    DevBuf<float> x_, ffn_inp_, mixed_, ins_[3], attn_, o_, tmp_,
        moe_, rlog_, conv_in_, kc_, vc_, ctrl_in_, ctrl_g_, ctrl_gf_, proj_, msg_, tau_, gx_, gu_, gh_, gd_, rb_;
    DevBuf<float> mix_sc_;       // the depth mix's scores
    DevBuf<float> conv_scratch_; // prompt chunks' conv inputs
    const float * peer_in_[2] = {nullptr, nullptr};   // the peer's partials (the inbox) for the next conv
    DevBuf<float> blocks_buf_;   // [kMaxBlocks][chunk][E]
    static constexpr int kMaxBlocks = 12;
    DevBuf<float> logits_;       // [kInkWin][n_vocab_pad] (all_logits: [chunk])
    DevBuf<int32_t> tok_dev_, ids_dev_, rows_dev_, argmax_dev_, topk_ids_;
    DevBuf<float> w_dev_, wg_dev_, topk_vals_;
    DevBuf<int32_t> inv_dev_;
    DevBuf<double> topk_z_;
    DevBuf<float> attn_scratch_;
    int32_t * h_tok_ = nullptr;
    int * h_pos_ = nullptr;          // pinned: the window's first position (the graphs copy it to d_pos_)
    float * h_tau_ = nullptr;        // pinned: the full layers' log-scaled positions
    DevBuf<int> d_pos_;
    DevBuf<uint8_t> plan_buf_;       // InkMoePlan
    DevBuf<float> mgu_, mh_, my_;    // the device MoE's gate | up, h and per-pair outputs
    DevBuf<int8_t> mq_;              // ... its int8 activations
    // the prompt path's tensor-core experts (moe_tc): fp16 input, h and outputs, the grouping
    DevBuf<half> xh_, hh_, dh_;
    DevBuf<TcItem> tc_items_gu_, tc_items_dn_;
    DevBuf<TcBlock> tc_blocks_;
    DevBuf<float> mqd_;
    bool graphs_on_ = true;
    cudaGraphExec_t graphs_[kInkMaxSlots][kInkWin + 1][2] = {};   // per slot, window rows and commit (keep) mode
    int64_t captures_ = 0;
    void enqueue(int T, bool keep);
    std::vector<int32_t> h_ids_;
    std::vector<float> h_w_;
    int logits_rows_ = 0;
    int blk_next_ = 0;       // block buffers handed out in this forward
    float * fresh_block();

    void run(const int32_t * tokens, int T, bool commit_all);
    void mm(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy, bool acc = false);

    // norms: each output row's RMS-norm weight (null: none)
    void mix(const std::vector<const float *> & blocks, const float * cur, const float * query, int nq, int T,
             float * out, const float * const * norms = nullptr);
    void control(float * cur, const float * msg, const InkLoop & loop, int j, int T);
    void moe(const InkLayer & L, const float * x, int T, float * out);
    void conv(int v, int off, int C, const float * x_rows, int64_t ldx, float * y, int T, const float * w,
              const float * dyn, int64_t ld_dyn, int H, bool keep);
};

}  // namespace bnk
