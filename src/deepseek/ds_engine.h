// The DeepSeek-V4.1-Flash forward over a window of up to kMaxWindow tokens (docs/DEEPSEEK.md).
//
// Built like the two Qwen engines: one forward is one CUDA graph launch per pipeline stage (captured per window
// layout and context bucket); the routed experts resident in each GPU's cache run there, the others go through the
// per-layer mailboxes to the CPU pool, which the calling thread services while the graphs run. Experts are placed
// disjointly: a resident expert has no host copy. Prompts go through decode windows for now.
#pragma once

#include <atomic>
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

#include "core/expert_cache.h"
#include "core/util.h"
#include "core/vmem.h"
#include "cpu/expert_pool.h"
#include "deepseek/ds_model.h"
#include "deepseek/ds_spark.h"
#include "deepseek/ds_ops.h"
#include "engine/engine.h"
#include "engine/engine_api.h"
#include "kernels/gemv.h"
#include "kernels/moe.h"

namespace bnk {

// A drafter with no drafts of its own: the generator drafts only when it has a drafter, and with this one its
// prompt-lookup proposals (what followed the sequence's last tokens earlier: copied text, code, tool scaffolding)
// become the drafts. DSpark, the model's own draft head (3 MoE layers of 128 experts), is not ported yet: a verified
// window row cost 0.43-0.48 of a plain round with the static placement (the CPU expert reads of the extra token),
// 0.13-0.19 with the adaptive exchange, which makes the port worth it (docs/DEEPSEEK-NEXT.md, 2026-10-26).
class LookupOnlyDrafter : public Drafter {
public:
    int run(const float *, const int32_t *, int, int, bool, float * prob) override { if (prob) *prob = 0.f; return -1; }
    int step(int32_t, int, float * prob) override { if (prob) *prob = 0.f; return -1; }
    void run_multi(const std::vector<RunItem> & items, int32_t * tok_out, float * prob_out) override {
        for (size_t i = 0; i < items.size(); ++i) { tok_out[i] = -1; prob_out[i] = 0.f; }
    }
    void step_multi(const std::vector<StepItem> & items, int32_t * tok_out, float * prob_out) override {
        for (size_t i = 0; i < items.size(); ++i) { tok_out[i] = -1; prob_out[i] = 0.f; }
    }
};

class DsEngine : public EngineApi {
public:
    ~DsEngine();
    void load(const std::string & path, const EngineOptions & opt);

    // ---- EngineApi
    std::vector<int> devices() const override;
    int device() const override { return dev_; }
    int last_device() const override { return next_ ? next_->last_device() : dev_; }
    int stages() const override { return next_ ? 1 + next_->stages() : 1; }
    StageInfo stage_info(int i) const override;
    std::string model_name() const override { return model_.gguf.get_str("general.name", "bnk"); }
    const Config & cfg() const override { return model_.cfg; }
    int res_dim() const override { return model_.ds.hc_dim(); }
    // the DSpark drafter when loaded (--mtp FILE; it lives on the stage with the head and the target layers), else
    // the lookup-only drafter
    Drafter * mtp() override {
        for (const DsEngine * s = this; s; s = s->next_.get()) if (s->spark_) return s->spark_.get();
        return lookup_drafter_.get();
    }
    int max_ctx() const override { return opt_.max_ctx; }
    int cpu_threads() const override { return cpu_->threads(); }
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
    void save_counts() override;

    void prefill(const std::vector<int32_t> & tokens) override;
    void forward(const int32_t * tokens, int T, bool commit_all = true) override;
    void forward_batch(const std::vector<BatchWin> & wins) override;
    static constexpr int kLanes = 2;
    void forward_pipe(const std::vector<BatchWin> & first, int first_lane, const std::vector<BatchWin> & second,
                      int second_lane) override;
    int row0_of(int slot) const override { return seqs_.at(slot)->last_r0; }
    void commit(int c) override;
    const float * residual_dev() const override { return next_ ? next_->residual_dev() : res_.p; }
    void set_mtp_pending(const float *, int) override {}
    const float * mtp_pending_R() const override { return nullptr; }
    int mtp_pending_cell() const override { return -1; }

    int argmax(int t) override;
    void argmax_all(int T, int32_t * out, int row0 = 0) override;
    std::vector<float> logits_host(int t) override;
    void logits_rows_host(int T, float * out, int row0 = 0) override;
    void topk_rows_host(int row0, int T, int K, int32_t * ids, float * vals, double * z = nullptr, float temp = 1.f) override;

    int64_t cache_swaps() const override { return cache_.swaps_done + (next_ ? next_->cache_swaps() : 0); }
    int cache_slots(int il) const override;
    int cache_resident() const override { return cache_.resident() + (next_ ? next_->cache_resident() : 0); }
    size_t cache_bytes() const override { return cache_.bytes() + (next_ ? next_->cache_bytes() : 0); }
    size_t kv_bytes_mapped() const override;

    double gpu_ms() const { return gpu_ms_; }
    int64_t gpu_forwards() const { return gpu_forwards_; }
    int layer_begin() const { return l0_; }
    int layer_end() const { return l1_; }
    const DsEngine * stage(int i) const { return i == 0 ? this : next_ ? next_->stage(i - 1) : nullptr; }

private:
    // ---- pipeline stages: this stage's device and layers [l0_, l1_); head_ is the first stage
    int dev_ = 0, l0_ = 0, l1_ = 0;
    bool last_ = true;
    DsEngine * head_ = this;
    std::unique_ptr<DsEngine> next_;
    bool first() const { return l0_ == 0; }
    bool owns(int il) const { return il >= l0_ && il < l1_; }
    void load_stage(const std::string & path, const EngineOptions & opt);
    void init_cache(const Ranking & rank);
    // the residual rows (hc * E floats) and the pending pre-mix (hc floats) per row between stages: pinned host
    float * xfer_in_[kLanes] = {}, * xfer_out_[kLanes] = {};
    int lane_ = 0;
    size_t xfer_floats() const { return (size_t) kMaxWindow * (model_.ds.hc_dim() + model_.ds.hc); }
    std::mutex stats_mu_;
    cudaEvent_t ev_out_ = nullptr, ev_t0_ = nullptr, ev_t1_ = nullptr;
    double gpu_ms_ = 0;
    int64_t gpu_forwards_ = 0;

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
    int ctx_bucket(int cells) const;
    void fwd_prepare(const std::vector<BatchWin> & batch, int lane = 0);
    void fwd_launch(cudaEvent_t after);
    void fwd_service();
    void fwd_end();
    bool fwd_graph_ = true;
    cudaGraphExec_t fwd_exec_ = nullptr;
    double fwd_t0_ = 0;
    void save_counts_here();

    std::unique_ptr<LookupOnlyDrafter> lookup_drafter_;   // BNK_DS_LOOKUP=0 turns the lookup drafts off
    std::unique_ptr<DsSpark> spark_;     // docs/DSPARK.md; on the last stage
    DevBuf<float> tgt_;                  // its view of the rows at the target layers by position: [kDsRing][n_targets * E]
    std::vector<int> tgt_index_;         // per layer: which target it is, -1 none
    DsModel model_;
    ExpertStore own_store_;
    ExpertStore * store_ = &own_store_;
    ExpertCache cache_;
    CpuExpertPool own_cpu_;
    CpuExpertPool * cpu_ = &own_cpu_;
    EngineOptions opt_;
    VramBudget budget_;
    cudaStream_t st_ = nullptr, st_commit_ = nullptr;
    cudaEvent_t ev_verified_ = nullptr, ev_committed_ = nullptr;
    bool commit_inflight_ = false;

    // the decode forward
    void enqueue_forward(int T);
    void layer_forward(int il, int T);
    void engram(int il, int T);
    void attn(int il, int T);
    void moe(int il, int T);
    void head(int T);
    void service_cpu(uint32_t seq, int T);
    void enqueue_commit(int c);
    void engram_gather();
    void engram_hash(int e, const std::vector<int32_t> & hist, int p, uint64_t * rows) const;
    void quantize_rows(const float * x, int64_t ldx, int T, int64_t cols, ActQ8 & a) { quantize_act(x, ldx, T, cols, a, st_); }

    struct Checkpoint { int pos = -1; uint64_t age = 0; float * host = nullptr; };
    uint64_t ckpt_age_ = 0;
    static constexpr int kMaxCheckpoints = 8;
    // One conversation's state on this stage: history, the window ring and (at source layers) the compressed
    // latents and index keys of every layer, the compressor's open group, snapshots, and the graphs that bake its
    // buffers' addresses.
    struct Seq {
        std::vector<int32_t> history;
        std::vector<Checkpoint> ckpts;
        int pending_T = 0;
        std::vector<DevBuf<half>> ring;          // [kDsRing][D] per layer
        std::vector<ElasticBuf> ckv_buf, ikc_buf;   // kv-source layers: [max_ctx / ratio][D] and [..][ID]
        std::vector<DevBuf<half>> ckc, ikc;      // views
        std::vector<DevBuf<float>> st_kv, st_sc; // kv-source layers with ratio > 1: the open group's raw rows
        int ctx_mapped = 0;
        cudaGraphExec_t commit_graphs[kLanes][kMaxWindow][kMaxWindow + 1] = {};
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
    int ctx_grid_ = 0;   // the context bucket the graph being captured / run sizes its grids for (cells)
    void alloc_seq(Seq & q);
    void use(Seq * q);
    template <typename F> void each_state(F && f);
    size_t state_floats();
    uint32_t seq_ = 0;

    // window inputs (pinned host) and their device copy
    WinParams * h_par_ = nullptr;
    WinParams * d_par_ = nullptr;
    int32_t * h_tok_ = nullptr;
    std::vector<uint8_t *> h_eg_rows_;   // per Engram layer: [W][cols] table rows, mapped

    // rotary tables: window layers (plain) and compressed layers (YaRN)
    DevBuf<float> rope_win_, rope_comp_;
    const float * rope_of(int il) const { return model_.ds.ratio[il] ? rope_comp_.p : rope_win_.p; }

    // device activations (kMaxWindow rows)
    DevBuf<float> x_, res_, mixed_, xn_, out_, pre_a_, pre_b_, attn_pre_, post_, comb_, mix_scratch_;
    float * pre_ = nullptr, * pre_next_ = nullptr;   // the pre-mix the next sublayer collapses with / the one after
    DevBuf<float> qr_, qrn_, q_, kvr_, attn_o_, oa_;
    DevBuf<float> iq_, iw_, scores_, cand_scratch_;
    DevBuf<int32_t> idx_;
    DevBuf<uint8_t> cand_;
    int64_t lds_ = 0;        // row stride of scores_ (compressed rows of the longest context)
    int n_blocks_max_ = 0;
    DevBuf<float> eg_emb_, eg_kv_;
    DevBuf<float> rlogits_, rw_, sg_, su_, shared_out_, moe_out_, ones_, logits_;
    DevBuf<int32_t> rids_, argmax_dev_, topk_ids_;
    DevBuf<float> topk_vals_;
    DevBuf<double> topk_z_;
    DevBuf<int8_t> actq_, actq2_;
    DevBuf<half> acth_, acth2_;
    DevBuf<float> actd_, actd2_;
    DevBuf<int16_t> acts_, acts2_;
    ActQ8 act_, act2_;
    // per lane and kv-source layer: the window's raw compressor projections (kept for the commit)
    std::vector<DevBuf<float>> ckv_raw_[kLanes], sc_raw_[kLanes];
    // The PCIe lane of decode (BNK_DS_PULL=P experts a layer; off by default, see pull_init): in layer il, layer il + 1's router is
    // applied to il's FFN input (Strata's guess: the top expert right 97% of the time), its non-resident picks are
    // pulled from their host copies (the store's memory, mapped) into il + 1's staging slots on a low-priority side
    // stream while il's FFN and il + 1's attention run, and il + 1's slot table is patched so the plan kernel finds
    // them resident: the CPU is spared them. The staging slots are P spare slots at the end of each layer's cache
    // region (ExpertCache::reserve_spare), R-layout like the rest, so the hit kernels need no extra launch: the blob
    // goes host -> pull_raw_ -> repacked into the slot. After il + 1's FFN the table is restored.
    int pull_P_ = 0;
    cudaStream_t st_pull_ = nullptr;
    cudaEvent_t ev_pull_fork_ = nullptr, ev_pull_join_ = nullptr;
    DevBuf<uint64_t> pull_blob_ptr_;         // [n_layer][n_expert] device aliases of the host blobs (0: none)
    DevBuf<uint8_t> pull_raw_;               // P blobs in their file layout, on their way to the slots
    DevBuf<uint8_t *> pull_dst_;             // [layers of this stage][P] the spare slots' addresses, for the repack
    DevBuf<int32_t> pull_picks_, pull_staged_;   // [n_layer][P]
    DevBuf<float> pull_logits_, pull_w_;
    DevBuf<int32_t> pull_ids_;
    DevBuf<uint32_t> pull_stats_;            // staged, used, late
    uint32_t pull_stats_seen_[3] = {0, 0, 0};
    DevBuf<int32_t> pull_state_;             // [n_layer]: 0 idle, 1 a pull in flight, 2 committed (slots patched)
    bool pull_wait_ = false;                 // BNK_DS_PULL_WAIT=1: the plan waits for the pull (the first design)
    void pull_init();
    void pull_predict(int il, int T);        // in layer il, before its FFN: il + 1's routing guessed and its picks made
    void pull_launch(int il);                // after il's plan kernel: the picks' pull forked onto st_pull_ (the plan
                                             // kernel's mailbox writes and a pull's reads of host memory on the same
                                             // link ran the plan 10x slower when they overlapped)
    int pull_pending_ = -1;                  // the layer whose picks await their pull_launch (-1 none)
    // MoE
    MoeScratch moes_;
    DevBuf<uint8_t> hits_buf_;
    DevBuf<float> gu_buf_, hd_buf_, part_buf_;
    DevBuf<int8_t> hq_buf_;
    DevBuf<uint32_t> counts_;
    std::vector<uint32_t> saved_counts_;
    // the adaptive exchange (ExpertCache::adapt_exchange): every adapt_every_ decode forwards, up to adapt_swaps_
    // exchanges in flight per stage; BNK_DS_ADAPT=N sets the interval (0 off), BNK_DS_ADAPT_SWAPS the count
    int adapt_every_ = 0, adapt_swaps_ = 0;
    int64_t fwd_count_ = 0;
    uint8_t * mail_ = nullptr;
    size_t mail_stride_ = 0;
    MoeMsg * msg(int il) const { return (MoeMsg *) (mail_ + (size_t) il * mail_stride_); }
    std::vector<ExpertTask> tasks_;

    // ---- batched prompt processing (ds_prefill.cpp): a group of up to pf_group_ tokens (BNK_DS_PF_CHUNKS chunks of
    // pf_max_) runs layer-major: every layer goes over all of the group's chunks (attention and the dense projections
    // per chunk through fp16 tensor-core GEMMs, weights dequantized on the fly), then the layer's routed experts over the
    // whole group at once on the tensor cores: resident ones from the cache, the others streamed from the host store
    // into two staging buffers in waves (the copy engine busy throughout, the next layer's first wave prefetched by
    // rank), the rest on the CPU pool (an adaptive cost model). So every non-resident expert crosses PCIe once per
    // layer per group, not once per chunk. Groups are pipelined over the stages (a worker thread runs group k through
    // the later stages while the first stage reads group k+1).
    Gemm gemm_;
    int pf_max_ = 0;     // tokens per chunk (the attention buffers)
    int pf_group_ = 0;   // tokens per group (the MoE buffers): pf_max_ * chunks
    void prefill_alloc(int N, int G);
    void prefill_group(const int32_t * tokens, int N);
    void prefill_group_here(const int32_t * tokens, int N, int buf);
    void pf_engram(int il, int c, int N);
    void pf_attn(int il, int c, int N, int pos0);
    void pf_shared(int il, int c, int N);
    void pf_moe(int il, int N);
    void pf_prefetch(int il, const std::vector<int> & order, int nbuf);
    void pf_head(int N);
    void pf_engram_gather(int N, int e_only = -1);
    double pf_eg_ms_ = 0;   // (under head_->stats_mu_)
    void pf_update_costs();
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
    std::unique_ptr<Worker> pf_worker_;
    float * pf_xfer_[2] = {nullptr, nullptr};   // pinned: a group's residual rows + pre-mix between stages (two in flight)
    std::vector<std::vector<int>> pf_rank_;      // per layer: this stage's non-resident experts, hottest first
    double pf_ms_ = 0;
    int64_t pf_tokens_ = 0;
    // the cost model's constants, corrected from measurements as a read goes on (per stage: its own PCIe link; the
    // CPU pool is shared, its scale lives on the first stage)
    double pf_pcie_us_mb_ = 0;
    double pf_cpu_scale_ = 1.0;
    static constexpr int kPfWaves = 32;
    struct {
        // group-sized (pf_group_ rows)
        DevBuf<float> res, xn, pre_a, pre_b, post, comb, shared, moe, cpu, rlog, w, w2;
        DevBuf<int32_t> ids, perm, inv, ids2;    // ids2: the next layer's routing guessed from this layer's input
        DevBuf<half> xg, hh, dd;
        // chunk-sized (pf_max_ rows)
        DevBuf<float> mixed, xnc, out, attn_pre, postc, combc;
        DevBuf<float> qr, qrn, q, kvr, attn_o, oa, ckv, sc, iq, iw, scores, cand_scratch, sg, sh;
        DevBuf<int32_t> idx, tok;
        DevBuf<uint8_t> cand, items;
        DevBuf<float> eg_emb, eg_kv;
        DevBuf<half> kwin;                       // kwin: the window keys of the chunk and the 128 positions before it
        DevBuf<int32_t> pos;                     // the first position of every score sub-chunk (device), per chunk
        WinParams * h_par = nullptr;             // pinned, per chunk
        WinParams * d_par = nullptr;
        int n_chunks_max = 0;
        uint8_t * eg_rows = nullptr;             // pinned host, per Engram layer in sequence
        size_t eg_rows_per_layer = 0;
        int32_t * h_ids = nullptr, * h_ids2 = nullptr, * h_perm = nullptr, * h_inv = nullptr, * h_cpu_map = nullptr;
        int32_t * h_cpu_tok = nullptr;           // the tokens with CPU tasks, compact (their rows go out in that order)
        int xc_rows = 0;                         // rows the compact copy holds (more: every row goes out, as before)
        float * h_w = nullptr, * h_x = nullptr, * h_cpu = nullptr;
        uint8_t * h_items = nullptr;
        // device aliases of the mapped host buffers (read by kernels instead of copy-engine transfers)
        int32_t * d_h_perm = nullptr, * d_h_inv = nullptr, * d_h_cpu_map = nullptr, * d_h_cpu_tok = nullptr;
        float * d_h_cpu = nullptr;
        DevBuf<float> xc;                        // [xc_rows][E]: the CPU tokens' FFN input rows, compact
        uint8_t * d_h_items = nullptr;
        size_t items_bytes = 0;
        int max_blocks = 0;
        ElasticBuf stage_b[2];
        uint8_t * stage[2] = {nullptr, nullptr};
        size_t stage_bytes = 0;
        std::vector<int> stage_idx;              // [n_expert] -> index in its stage buffer (-1: not staged), this layer
        std::vector<char> stage_buf;             // [n_expert] -> which buffer
        int toggle = 0;                          // the buffer the next wave takes
        int pre_layer = -1;                      // the layer whose first waves are prefetched
        std::vector<int> pre_list, pre_pos;      // its experts, each one's index in its buffer
        std::vector<char> pre_bufs;              // ... and the buffer
        cudaStream_t copy = nullptr;
        cudaEvent_t copied[2] = {nullptr, nullptr}, used[2] = {nullptr, nullptr}, xready = nullptr;
        cudaEvent_t wt0[kPfWaves] = {}, wt1[kPfWaves] = {};   // timing of each wave's copies
        int n_timed = 0;
        double timed_mb = 0;
        double cpu_pred_ms = 0, cpu_meas_ms = 0;
        int score_rows = 0;                      // rows of the indexer's score buffer (the chunk in pieces)
        int n_sub = 0;                           // score sub-chunks per chunk
    } pf_;
};

template <typename F> void DsEngine::each_state(F && f) {
    const DsConfig & c = model_.ds;
    for (int il = l0_; il < l1_; ++il)
        if (c.is_kv_src(il) && c.ratio[il] > 1) {
            f(cur_->st_kv[il].p, cur_->st_kv[il].n);
            f(cur_->st_sc[il].p, cur_->st_sc[il].n);
        }
}

}  // namespace bnk
