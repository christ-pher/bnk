// The Inkling 2 engine (inkling2_loop): every weight on one GPU, one conversation slot, the forward over a window
// of tokens (decode / verify, up to kInkWin) or a prompt chunk. A token visits 48 layers (the core three times);
// each visit has its own K/V cache (window layers: a ring of window + chunk cells; full layers: the context) and its
// short-convolution state. docs/INKLING-PLAN.md.
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
#include "kernels/gemm.h"
#include "kernels/gemv.h"

namespace bnk {

constexpr int kInkWin = 16;   // rows of a decode / verify window

class InkEngine : public EngineApi {
public:
    ~InkEngine();
    void load(const std::string & path, const EngineOptions & opt);

    // ---- EngineApi
    std::vector<int> devices() const override { return {dev_}; }
    int device() const override { return dev_; }
    int last_device() const override { return dev_; }
    int stages() const override { return 1; }
    StageInfo stage_info(int i) const override;
    std::string model_name() const override { return model_.gguf.get_str("general.name", "bnk"); }
    const Config & cfg() const override { return model_.base; }
    int res_dim() const override { return model_.cfg.E; }
    Drafter * mtp() override { return nullptr; }
    int max_ctx() const override { return opt_.max_ctx; }
    int max_window() const override { return kInkWin; }
    int snap_every() const override { return 0; }
    int cpu_threads() const override { return 0; }
    int64_t graph_captures() const override { return 0; }
    size_t graph_bytes() const override { return 0; }

    int slots() const override { return 1; }
    int current_slot() const override { return 0; }
    void select(int slot) override {
        if (slot != 0) throw std::runtime_error("inkling: one conversation slot");
    }
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
    int row0_of(int) const override { return 0; }
    void commit(int c) override;
    const float * residual_dev() const override { return nullptr; }
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

    // tests: when set, every prompt row's logits (device, [n][n_vocab]) for rows pos0.. as they are computed
    std::function<void(int pos0, int n, const float * logits)> logits_hook;

private:
    int dev_ = 0;
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

    // checkpoints: conv states and the window rings on the host
    struct Ckpt { int pos = -1; std::vector<float> conv; std::vector<half> ring; int64_t age = 0; };
    std::vector<Ckpt> ckpts_;
    int64_t ckpt_age_ = 0;

    // activations (capacity chunk_ rows)
    DevBuf<float> x_, inp_sa_, ffn_inp_, mixed_, ins_[3], q_, r_, sink_, dyn_k_, dyn_v_, attn_, gate_, o_, tmp_,
        moe_, rlog_, conv_in_, kc_, vc_, ctrl_in_, ctrl_g_, ctrl_gain_, ctrl_forget_, msg_, tau_, gx_, gu_, gh_, gd_;
    DevBuf<float> blocks_buf_;   // [kMaxBlocks][chunk][E]
    static constexpr int kMaxBlocks = 12;
    DevBuf<float> logits_;       // [kInkWin][n_vocab_pad] (all_logits: [chunk])
    DevBuf<int32_t> tok_dev_, ids_dev_, rows_dev_, argmax_dev_, topk_ids_;
    DevBuf<float> w_dev_, wg_dev_, topk_vals_;
    DevBuf<int32_t> inv_dev_;
    DevBuf<double> topk_z_;
    DevBuf<float> attn_scratch_;
    int32_t * h_tok_ = nullptr;
    std::vector<int32_t> h_ids_;
    std::vector<float> h_w_;
    int logits_rows_ = 0;
    int blk_next_ = 0;       // block buffers handed out in this forward
    float * fresh_block();

    void run(const int32_t * tokens, int T, bool commit_all);
    void mm(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy, bool acc = false);
    // one layer visit: slot v of physical layer ip; blocks the depth-mix list; msgs [4][T][E] for passes > 0
    void layer(int v, int ip, int T, int pos0, std::vector<const float *> & blocks, const float * msgs,
               const InkLoop * loop, bool keep);
    void mix(const std::vector<const float *> & blocks, const float * cur, const float * query, int nq, int T,
             float * out);
    void control(float * cur, const float * msg, const InkLoop & loop, int j, int T);
    void moe(const InkLayer & L, const float * x, int T, float * out);
    void conv(int v, int off, int C, const float * x_rows, int64_t ldx, float * y, int T, const float * w,
              const float * dyn, int H, bool keep);
};

}  // namespace bnk
