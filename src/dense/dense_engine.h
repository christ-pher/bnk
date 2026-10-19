// The dense Qwen3.5-family forward (qwen35: Qwen3.8-27B) over a window of up to kMaxWindowDense tokens, and its prompt
// path. Every weight lives in VRAM: one forward is one CUDA graph launch per pipeline stage (captured per window
// layout), nothing crosses to the CPU but the window's inputs and the residual rows between GPUs.
//
// Several GPUs (EngineOptions::gpus) make a pipeline as in engine.h: each GPU holds a contiguous range of layers
// with their weights, KV cache and recurrent state; the first stage embeds the tokens, the last one holds the head
// and the MTP drafter. The public API (EngineApi) covers the whole pipeline from the first stage.
#pragma once

#include <condition_variable>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <functional>
#include <thread>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "core/util.h"
#include "core/vmem.h"
#include "dense/dense_dflash.h"
#include "dense/dense_model.h"
#include "dense/dense_mtp.h"
#include "engine/engine.h"
#include "engine/engine_api.h"
#include "kernels/gemm.h"
#include "kernels/gemv.h"

namespace bnk {

class DenseEngine : public EngineApi {
public:
    ~DenseEngine();
    void load(const std::string & path, const EngineOptions & opt);
    // the layers of each stage when the model is split over `gpus` (balanced by the bytes each stage holds)
    static std::vector<int> plan_split(const std::string & path, const EngineOptions & opt, const std::vector<int> & gpus);

    // ---- EngineApi
    std::vector<int> devices() const override;
    int device() const override { return dev_; }
    int last_device() const override { return next_ ? next_->last_device() : dev_; }
    int stages() const override { return tp() ? 1 : next_ ? 1 + next_->stages() : 1; }
    StageInfo stage_info(int i) const override;
    std::string model_name() const override { return model_.gguf.get_str("general.name", "bnk"); }
    const Config & cfg() const override { return model_.cfg; }
    int res_dim() const override { return model_.cfg.n_embd; }
    Drafter * mtp() override {
        return next_ && !tp() ? next_->mtp() : dflash_.loaded() ? (Drafter *) &dflash_ : mtp_.loaded() ? &mtp_ : nullptr;
    }
    int max_ctx() const override { return opt_.max_ctx; }
    int max_window() const override { return kMaxWindowDense; }
    int cpu_threads() const override { return 0; }
    int64_t graph_captures() const override { return graph_captures_ + (next_ ? next_->graph_captures() : 0); }
    size_t graph_bytes() const override { return graph_bytes_ + (next_ ? next_->graph_bytes() : 0); }

    int slots() const override { return (int) seqs_.size(); }
    int current_slot() const override;
    void select(int slot) override;
    void reset() override;
    int pos() const override { return (int) cur_->history.size(); }
    const std::vector<int32_t> & history() const override { return cur_->history; }
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
    static constexpr int kLanes = 2;
    void forward_pipe(const std::vector<BatchWin> & first, int first_lane, const std::vector<BatchWin> & second,
                      int second_lane) override;
    int row0_of(int slot) const override { return seqs_.at(slot)->last_r0; }
    void commit(int c) override;
    const float * residual_dev() const override { return next_ && !tp() ? next_->residual_dev() : res_.p; }
    void set_mtp_pending(const float * R_row_dev, int cell) override;
    const float * mtp_pending_R() const override { return next_ && !tp() ? next_->mtp_pending_R() : cur_->mtp_R; }
    int mtp_pending_cell() const override { return next_ && !tp() ? next_->mtp_pending_cell() : cur_->mtp_cell; }

    int argmax(int t) override;
    void argmax_all(int T, int32_t * out, int row0 = 0) override;
    std::vector<float> logits_host(int t) override;
    void logits_rows_host(int T, float * out, int row0 = 0) override;
    void topk_rows_host(int row0, int T, int K, int32_t * ids, float * vals, double * z = nullptr, float temp = 1.f) override;

    int64_t cache_swaps() const override { return 0; }
    int cache_slots(int) const override { return 0; }
    int cache_resident() const override { return 0; }
    size_t cache_bytes() const override { return 0; }
    size_t kv_bytes_mapped() const override;

    // this stage's timings (stage_info)
    double gpu_ms() const { return gpu_ms_; }
    int64_t gpu_forwards() const { return gpu_forwards_; }
    double prompt_ms() const { return pf_ms_; }
    int64_t prompt_tokens() const { return pf_tokens_; }
    int layer_begin() const { return l0_; }
    int layer_end() const { return l1_; }
    const DenseEngine * stage(int i) const { return i == 0 ? this : next_ ? next_->stage(i - 1) : nullptr; }
    struct PrefillStats { double ms = 0; int64_t tokens = 0, chunks = 0; } pstats;

private:
    // ---- pipeline stages
    int dev_ = 0, l0_ = 0, l1_ = 0;
    bool last_ = true;
    DenseEngine * head_ = this;
    std::unique_ptr<DenseEngine> next_;
    DenseEngine * tail() { return next_ && !tp() ? next_->tail() : this; }
    // ---- tensor parallel: the chain's objects are the ranks, every layer on each (sliced); rank 0 drives both
    // GPUs' streams in lockstep and captures one graph over both devices
    int tp_n_ = 1, tp_rank_ = 0;
    bool tp() const { return tp_n_ > 1; }
    DenseEngine * peer_ = nullptr;
    // the all-reduce with the peer through pinned host memory, per exchange point (0: after the attention /
    // DeltaNet, 1: after the FFN) and layer parity (a layer's copy out must not overwrite what the peer still reads)
    float * tp_h_[2][2] = {};           // decode: this rank's partial [W][E] (fp32, exact: both ranks add the same)
    half * tp_pfh_[2][2] = {};          // prompt chunks: [pf_max][E] as fp16 (both ranks add the two rounded halves)
    std::vector<cudaEvent_t> tp_ev_[2], tp_pfev_[2];   // per point and layer (a capture keeps every record apart)
    DevBuf<float> tp_tmp_;
    DevBuf<half> tp_mine16_, tp_peer16_;
    cudaEvent_t tp_fork_ = nullptr, tp_join_ = nullptr;
    // peer copies instead (needs the host's IOMMU in passthrough, docs/DENSE.md), plain device copies across devices
    // (unified addressing; cudaMemcpyPeerAsync does not capture). Per point and parity: decode pushes this rank's
    // partial into the peer's tp_d_ (an inbox); prompt chunks convert into this rank's tp_pfd_ and the peer pulls it.
    bool tp_p2p_ = false;
    DevBuf<float> tp_d_[2][2];
    DevBuf<half> tp_pfd_[2][2];
    static void tp_enable_p2p(DenseEngine * a, DenseEngine * b, bool verbose);
    void tp_send(const float * buf, int rows, int point, int il);
    void tp_recv(float * buf, int rows, int point, int il);
    void tp_pf_send(const float * partial, int rows, int point, int il);   // as fp16, from partial
    void tp_pf_recv(float * res, int rows, int point, int il);             // res += mine16 + peer16
    void tp_enqueue_forward(int T);
    void tp_prefill_chunk(const int32_t * tokens, int N);
    // the per-rank phases of a layer, driven in lockstep over the ranks
    void fwd_in(int T);
    void layer_block(int il, int T);
    void layer_ffn(int il, int T);
    void pf_in(const int32_t * tokens, int N, int buf);
    void pf_layer_a(int il, int N);
    void pf_layer_b(int il, int N);
    void pf_out(int N, int buf);
    bool first() const { return l0_ == 0; }
    bool owns(int il) const { return il >= l0_ && il < l1_; }
    void load_stage(const std::string & path, const EngineOptions & opt);
    // the residual rows between stages: pinned, portable host memory written by this stage, read by the next
    float * xfer_in_[kLanes] = {}, * pf_xfer_in_[2] = {nullptr, nullptr};
    float * xfer_out_[kLanes] = {}, * pf_xfer_out_[2] = {nullptr, nullptr};
    int lane_ = 0;
    struct Worker {
        std::thread th;
        std::mutex mu;
        std::condition_variable cv;
        std::function<void()> job;
        bool busy = false, quit = false;
        std::exception_ptr err;
        Worker();
        ~Worker();
        void post(std::function<void()> f);
        void wait();
    };
    std::unique_ptr<Worker> worker_;
    std::mutex stats_mu_;
    cudaEvent_t ev_out_ = nullptr, ev_t0_ = nullptr, ev_t1_ = nullptr;
    double gpu_ms_ = 0, pf_ms_ = 0;
    int64_t gpu_forwards_ = 0, pf_tokens_ = 0;

    void reset_here();
    void select_here(int slot);
    void checkpoint_here();
    int rollback_here(int max_pos);
    void commit_here(int c);
    void join_commit();
    void join_commit_here();
    void ensure_ctx(int cells);
    void ensure_ctx_here(int cells);
    void map_ctx(int cells);
    void fwd_prepare(const std::vector<BatchWin> & batch, int lane = 0);
    void fwd_launch(cudaEvent_t after);
    void fwd_end();
    bool fwd_graph_ = true;
    cudaGraphExec_t fwd_exec_ = nullptr;
    double fwd_t0_ = 0;

    DenseModel model_;
    DenseMtp mtp_;
    // the DFlash2 drafter (opt.dflash; tensor parallel rank 0 or a single stage): the target adds each tap layer's
    // fc_i . residual into tap_acc_ (decode windows) / pf_tap_ (prompt chunks)
    DenseDFlash dflash_;
    DevBuf<float> tap_acc_, tap_tmp_, pf_tap_;
    void tap(int il, int T);                 // decode: after layer il's FFN (res_ + pend_ is its residual)
    void pf_tap(int il, int N);              // prompt chunk: after layer il (pf_.res)
    EngineOptions opt_;
    VramBudget budget_;
    cudaStream_t st_ = nullptr, st_commit_ = nullptr;
    cudaStream_t st_side_[2] = {nullptr, nullptr};
    cudaEvent_t ev_fork_[2] = {nullptr, nullptr}, ev_join_[2] = {nullptr, nullptr};
    bool side_ = true;
    cudaStream_t fork(int i);
    void join(int i);
    cudaEvent_t ev_verified_ = nullptr, ev_committed_ = nullptr;
    bool commit_inflight_ = false;

    // the decode forward
    void enqueue_forward(int T);
    void layer_forward(int il, int T);
    void gdn(int il, int T);
    void attn(int il, int T);
    void ffn(int il, int T);
    void head(int T);
    void enqueue_commit(int c);
    const float * pend_ = nullptr;   // a residual add left to the next norm's pass

    struct Checkpoint { int pos = -1, mtp_cell = -1; uint64_t age = 0; float * host = nullptr; };
    uint64_t ckpt_age_ = 0;
    static constexpr int kMaxCheckpoints = 6;
    // One conversation's state on this stage: history, KV cache of its attention layers, DeltaNet recurrences and
    // conv histories, snapshots, and the graphs that bake its buffers' addresses.
    struct Seq {
        std::vector<int32_t> history;
        std::vector<Checkpoint> ckpts;
        float * mtp_R = nullptr;
        int mtp_cell = -1;
        int pending_T = 0;
        std::vector<DevBuf<float>> conv_buf;    // [(K-1)+W][C] per DeltaNet layer
        DevBuf<float *> conv_ptrs;
        std::vector<DevBuf<float>> ssm_state;   // [nv][S][S]
        std::vector<DevBuf<half>> kc, vc;       // [max_ctx][Hkv][D] per attention layer
        std::vector<ElasticBuf> kv_k, kv_v;
        int ctx_mapped = 0;
        cudaGraphExec_t commit_graphs[kLanes][kMaxWindowDense][kMaxWindowDense + 1] = {};
        int last_r0 = 0, last_lane = 0;
        bool used = false;
    };
    std::vector<std::unique_ptr<Seq>> seqs_;
    Seq * cur_ = nullptr;
    struct Win { Seq * q; int r0, T, pos0; bool commit; };
    std::vector<Win> wins_;
    struct FwdGraph { cudaGraphExec_t exec = nullptr; uint64_t used = 0; size_t bytes = 0; };
    std::map<std::vector<int>, FwdGraph> fwd_graphs_;
    uint64_t fwd_graph_age_ = 0;
    int64_t graph_captures_ = 0;
    size_t graph_bytes_ = 0;
    static constexpr int kMaxFwdGraphs = 32;
    static constexpr size_t kGraphBytesEst = 24ull << 20;
    static constexpr int kMaxSlots = 8;
    static constexpr int kCtxStep = 8192, kCtxBaseline = 16384;
    void alloc_seq(Seq & q);
    void use(Seq * q);
    template <typename F> void each_state(F && f);
    size_t state_floats();
    uint32_t seq_ = 0;

    // window inputs (pinned host) and their device copy
    WinParams * h_par_ = nullptr;
    WinParams * d_par_ = nullptr;
    int32_t * h_tok_ = nullptr;

    // device activations (kMaxWindowDense rows)
    DevBuf<float> res_, xn_, out_, ffn_out_;
    DevBuf<float> conv_out_, z_, gdn_o_, gdn_n_;
    DevBuf<float> qfull_, k_, v_, q_, attn_o_, attn_scratch_;
    DevBuf<float> sg_, su_, logits_;
    DevBuf<int32_t> argmax_dev_, topk_ids_;
    DevBuf<float> topk_vals_;
    DevBuf<double> topk_z_;
    DevBuf<int8_t> actq_;
    DevBuf<half> acth_;
    DevBuf<float> actd_;
    DevBuf<int16_t> acts_;
    ActQ8 act_;
    int n_gdn_ = 0;
    std::vector<DevBuf<float>> gdn_co_[kLanes], gdn_g_[kLanes], gdn_b_[kLanes];

    // ---- batched prompt processing
    Gemm gemm_;
    int pf_max_ = 0;
    void prefill_alloc(int N);
    void prefill_chunk(const int32_t * tokens, int N);
    void prefill_chunk_here(const int32_t * tokens, int N, int buf = 0);
    void pf_gdn(int il, int N);
    void pf_attn(int il, int N);
    void pf_ffn(int il, int N);
    void mtp_feed(const float * R_rows, const int32_t * tokens, int n, int pos0);
    struct {
        DevBuf<float> res, xn, out, conv, co, z, g, b, o, n, qfull, k, v, q, ao, sg, su, x;
        DevBuf<int32_t> tok;
    } pf_;
};

template <typename F> void DenseEngine::each_state(F && f) {
    const Config & c = model_.cfg;
    for (int il = l0_; il < l1_; ++il)
        if (!c.is_attn(il)) {
            f(cur_->conv_buf[il].p, cur_->conv_buf[il].n);
            f(cur_->ssm_state[il].p, cur_->ssm_state[il].n);
        }
    if (cur_->mtp_R) f(cur_->mtp_R, (size_t) c.n_embd);
}

}  // namespace bnk
