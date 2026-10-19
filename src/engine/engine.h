// The qwen4exp forward pass over a window of up to kMaxWindow tokens.
//
// One forward = one CUDA graph launch (captured per window size). Window inputs (token ids, position,
// sequence number, PLE rows) live in pinned host memory that the graph reads. Routed experts resident
// in the VRAM cache run on the GPU; the others are sent through per-layer mailboxes to the CPU pool,
// which the calling thread services while the graph runs.
//
// Several GPUs (EngineOptions::gpus) make a pipeline: each GPU is a stage holding a contiguous range of layers
// with their dense weights, expert cache, KV and recurrent state; the first stage also embeds the tokens, the last
// one holds the head and the MTP drafter. The Engine the caller holds is the first stage and owns the others
// (next_); its public API covers the whole pipeline (a forward runs every stage, logits come from the last one,
// state operations apply to every stage, parking decisions are made once). Between stages only the residual rows
// cross, through pinned host memory. The routed experts in host RAM and the CPU pool are shared.
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

#include "core/expert_cache.h"
#include "core/pinned.h"
#include "core/vmem.h"
#include "core/model.h"
#include "core/util.h"
#include "cpu/expert_pool.h"
#include "engine/engine_api.h"
#include "engine/mtp.h"
#include "kernels/gemm.h"
#include "kernels/gemv.h"
#include "kernels/moe.h"
#include "kernels/qsa.h"

namespace bnk {

struct EngineOptions {
    int max_ctx = 8192;
    int cpu_threads = 24;
    bool verbose = true;
    bool use_graphs = true;
    double vram_reserve_gib = 0.75;  // left free after the expert cache
    double expert_cache_gib = -1;    // < 0: everything that fits
    std::string profile;             // ranking for the initial cache fill (STRP or BNKC)
    std::string counts_out;          // where routing counts are saved (BNKC), empty = off
    std::string mtp;                 // MTP draft layer GGUF (empty = no speculation)
    std::string ple_gguf;            // GGUF holding the PLE table when the model file has none
    std::string cold_experts;        // the least-routed experts in a smaller format (tools/requant_experts), all kept resident
    // chunk size for long prompts (borrows VRAM from the expert cache meanwhile). 4,096 since the experts moved to
    // the tensor cores: 8,192-token chunks evicted experts that then streamed back every chunk, and left the
    // pipeline fewer chunks to overlap (two V100s, cold reads: 64K 2,587 -> 2,901 tok/s, 128K 1,375 -> 2,905)
    int prefill_chunk_max = 4096;
    // Between prompts the prompt path keeps a layout for prefill_small tokens and up to stage_small_mib of PCIe
    // staging; more is borrowed from the expert cache for bigger reads. Smaller values give decoding a bigger cache
    // but slow down each turn: on an 80K-token agent conversation with Orca, 512 / 256 MiB gave +1.6% decode speed
    // (miss rate 7.9% -> 7.5%) for 1.2 s more per turn, so the defaults keep everything mapped.
    int prefill_small = 2048;        // layout kept between prompts
    int stage_small_mib = 1 << 20;   // staging kept between prompts (capped at a whole layer's outside experts)
    int stage_full_min = 512;        // reads this long borrow a stage for a whole layer's outside experts
    std::string draft_vocab;         // int32 token ids the drafter may propose (empty = the whole vocabulary)
    int prefill_chunk = 2048;        // tokens per batched prompt pass (0 = decode windows only)
    int prefill_min = 24;            // fewer tokens than this go through decode windows
    int prefetch_min = 1024;         // chunks this long stream every non-resident expert one layer ahead
    double pcie_us_per_mb = 78.0;    // cost model for the prompt path's non-resident experts: PCIe per MB of blob
    double cpu_us_per_expert_token = 46.0;  // ... and the CPU pool per (expert, token)
    int gemm_min_tokens = 32;        // prompt-path experts with fewer tokens run on their quantized weights
    int adapt_every = 4;             // forwards between adaptive cache updates (0 = static cache)
    int adapt_swaps = 16;            // max expert swaps started per update
    // Parked conversations: when a request belongs to another conversation, the live one's whole state goes to
    // host RAM (up to park_gib, least recently used dropped first) and comes back when its next request arrives,
    // instead of its prompt being read again. 0 = off (one conversation at a time, as before).
    double park_gib = 0;
    int park_min = 2048;             // conversations shorter than this are not worth parking
    // K/V cache cells in 8 bits (a fp16 scale per 32 values, as q8_0): 1,088 instead of 2,048 bytes per cell and
    // attention layer, so long contexts leave the expert cache more VRAM
    bool kv8 = false;
    int slots = 1;                   // conversations held on the GPU at once (batched decoding when > 1)
    // Pipeline over several GPUs: their CUDA device ids, one stage each (empty: the current device alone). The
    // same id twice runs two stages on one GPU (for testing the split on a single card; cap expert_cache_gib).
    std::vector<int> gpus;           // {-1}: every GPU of compute capability 7.0+
    std::vector<int> split;          // first layer of every stage after the first (empty: balanced by memory)
    // Tensor parallel instead of a pipeline (the dense engine): every layer split across the GPUs, their partial
    // sums exchanged through host memory twice per layer. 0: a pipeline, 1: tensor parallel over `gpus`.
    int tp = 0;
    std::vector<double> stage_cache_gib;   // expert_cache_gib per stage (empty: expert_cache_gib for every one)
};

struct WinParams {     // pinned host; copied to the device by the graph's first node
    int32_t pos0;
    int32_t T;
    uint32_t seq;
    int32_t pad;
};

class Engine : public EngineApi {
public:
    ~Engine();
    void load(const std::string & path, const EngineOptions & opt);
    void reset();
    // the pipeline: stages (GPUs), each one's device and layers
    int stages() const { return next_ ? 1 + next_->stages() : 1; }
    std::vector<int> devices() const;
    int device() const { return dev_; }
    int last_device() const { return next_ ? next_->last_device() : dev_; }   // the last stage's GPU
    int layer_begin() const { return l0_; }
    int layer_end() const { return l1_; }
    const Engine * stage(int i) const { return i == 0 ? this : next_ ? next_->stage(i - 1) : nullptr; }
    // this stage's GPU time in decode forwards (from its graph's start to its end, waiting for the previous stage
    // excluded), summed, and the forwards counted
    double gpu_ms() const { return gpu_ms_; }
    int64_t gpu_forwards() const { return gpu_forwards_; }
    // this stage's wall time in prompt chunks (its layers, start to finish), and the tokens
    double prompt_ms() const { return pf_ms_; }
    int64_t prompt_tokens() const { return pf_tokens_; }
    // the expert caches of every stage together (telemetry)
    int cache_slots(int il) const;
    int cache_resident() const;
    size_t cache_bytes() const;
    int64_t cache_swaps() const;
    StageInfo stage_info(int i) const {
        const Engine * s = stage(i);
        return StageInfo{s->device(), s->layer_begin(), s->layer_end(), s->cache().resident(), s->cache().bytes(),
                         s->gpu_ms(), s->gpu_forwards(), s->prompt_ms(), s->prompt_tokens()};
    }
    std::string model_name() const { return model_.gguf.get_str("general.name", "bnk"); }
    int res_dim() const { return model_.cfg.hc_dim(); }
    // the layers of each stage when the model is split over `gpus` (balanced by the memory each needs and has)
    static std::vector<int> plan_split(const std::string & path, const EngineOptions & opt, const std::vector<int> & gpus);

    // Runs tokens[0..T) at positions pos()..pos()+T-1; logits of every window position stay on the device
    // and the final residual of every row in residual_dev(). With commit_all the window is committed;
    // otherwise (a verify window) nothing is until commit(c) keeps its first c rows.
    void forward(const int32_t * tokens, int T, bool commit_all = true);
    // One forward over several slots' windows at once (batched decoding): their rows are concatenated in the
    // given order (at most kMaxWindow in all); the row-wise work (norms, GEMVs, MoE, head) runs once over all of
    // them, attention / DeltaNet / PLE per slot on its own state. row0(i) is where window i's rows start (logits,
    // residuals); a slot's commit() afterwards keeps rows of its own window.
    int64_t graph_captures() const { return graph_captures_ + (next_ ? next_->graph_captures() : 0); }
    size_t graph_bytes() const { return graph_bytes_ + (next_ ? next_->graph_bytes() : 0); }
    void forward_batch(const std::vector<BatchWin> & wins);
    // Two batches in flight on a two-GPU pipeline: `first` runs on the first GPU while `second` runs on the second
    // (`second` went through the first GPU in the previous call). Either may be empty. Each batch keeps to its lane
    // (0 or 1) from the first GPU to its commit: the lanes have their own window buffers, so one batch's forward
    // on the first GPU does not overwrite what the other's commit still needs. forward_batch() uses lane 0.
    static constexpr int kLanes = 2;
    void forward_pipe(const std::vector<BatchWin> & first, int first_lane, const std::vector<BatchWin> & second,
                      int second_lane);
    int row0_of(int slot) const;   // where that slot's rows started in the last forward
    void commit(int c);
    // The commit (the DeltaNet recurrence over the kept rows) runs on its own stream, overlapping the drafter;
    // everything that touches the recurrent state or the window buffers first joins it (on the main stream).
    void join_commit();
    // the last stage's final residual rows (and its stream: the drafter's)
    const float * residual_dev() const { return next_ ? next_->residual_dev() : res_.p; }
    cudaStream_t stream() const { return next_ ? next_->stream() : st_; }
    // Processes a prompt (batched passes for long stretches, decode windows for short tails) and commits it;
    // logits of its last token are in logits row 0... last_T-1 (argmax(last_T - 1)). With `mtp` set, the
    // draft layer's K/V are filled for every cell except the last token's (which pairs with the next one).
    void prefill(const std::vector<int32_t> & tokens);
    // VRAM for the context and the prompt path is elastic: KV for cells [0, cells) is mapped on demand
    // (the expert cache gives the bytes up), and rebalance() hands spare bytes back to the cache.
    void ensure_ctx(int cells);
    void rebalance();
    size_t vram_budget() const { return budget_.limit() + (next_ ? next_->vram_budget() : 0); }
    size_t kv_bytes_mapped() const;
    // One batched prompt pass over N tokens (commits; logits of the last token in row 0).
    void prefill_chunk(const int32_t * tokens, int N);
    struct PrefillStats { double ms = 0, copy_wait_ms = 0; int64_t tokens = 0, chunks = 0; } pstats;
    // After prefill with an MTP layer: the last prompt row's residual and cell, which pair with the first
    // generated token (Generator drafts from them).
    const float * mtp_pending_R() const { return next_ ? next_->mtp_pending_R() : cur_->mtp_R; }
    int mtp_pending_cell() const { return next_ ? next_->mtp_pending_cell() : cur_->mtp_cell; }
    int argmax(int t);
    void argmax_all(int T, int32_t * out, int row0 = 0);  // argmax of rows row0..row0+T-1 of the last forward
    std::vector<float> logits_host(int t);
    const float * logits_dev() const { return next_ ? next_->logits_dev() : logits_.p; }

    int pos() const { return (int) cur_->history.size(); }
    // conversation slots (EngineOptions::slots): select() makes slot i the current one
    int slots() const { return (int) seqs_.size(); }
    int current_slot() const;
    void select(int slot);
    // Snapshots of the per-position-independent state (DeltaNet recurrences, conv and n-gram histories, the
    // drafter's last residual) so a later prompt that shares only part of the history can resume from the last
    // snapshot inside the shared part instead of starting over. The KV cache needs none: it is per position.
    void checkpoint();
    // Restores the latest snapshot at or before max_pos; returns its position (the history is cut there), or -1.
    int rollback(int max_pos);
    int checkpoints() const;
    // Makes the state that can reuse the most of `prompt` live: the live one, or a parked one (parking the live
    // conversation first when it would lose at least park_min tokens). No-op unless park_gib > 0.
    void select_conversation(const std::vector<int32_t> & prompt);
    // tokens of `prompt` the current slot's state can keep (its history, or its last snapshot inside the prefix)
    int reusable(const std::vector<int32_t> & prompt) const;
    bool can_fork() const override;
    bool lookahead_ok() const override { return first() && !last_ && next_ && next_->last_ && !seqs_.empty() && !seqs_[0]->ssm_shadow.empty(); }
    void forward_lookahead(const BatchWin & win, int lane, const std::function<int(std::vector<int32_t> &)> & chain,
                           bool stage0_done) override;
    bool spec_pending() const override { return spec_inflight_ || cur_->spec_T > 0; }
    int spec_lane() const override { return spec_inflight_ || cur_->spec_T > 0 ? cur_->spec_lane : -1; }
    void spec_confirm() override;
    void spec_discard() override;
    void spec_abort() override;
    int park_min() const override { return opt_.park_min; }
    // slots with a request in flight (decoding, or the prompt being read): never moved off the GPU to make room
    void set_busy(const std::vector<int> & slots);
    int parked() const { return (int) parked_.size(); }
    void drop_parked_all() {
        for (Engine * s = this; s; s = s->next_.get()) {
            DeviceScope ds(s->dev_);
            while (!s->parked_.empty()) s->drop_parked(s->parked_.size() - 1);
        }
    }
    size_t parked_bytes() const;
    size_t spare_bytes() const;   // pinned buffers kept for reuse, every stage
    const std::vector<int32_t> & history() const { return cur_->history; }
    int max_ctx() const { return opt_.max_ctx; }
    int cpu_threads() const { return cpu_->threads(); }
    void logits_rows_host(int T, float * out, int row0 = 0);   // rows row0..row0+T-1 of the last forward's logits
    // the K largest logits of rows row0..row0+T-1 of the last window, selected on the GPU: ids/vals [T][K] (host)
    // z (optional): each row's sum of exp((logit - max) / temp) within 30 nats of its max (Sampler::distribution_head)
    void topk_rows_host(int row0, int T, int K, int32_t * ids, float * vals, double * z = nullptr, float temp = 1.f);
    void set_mtp_pending(const float * R_row_dev, int cell);
    const Config & cfg() const { return model_.cfg; }
    const Model & model() const { return model_; }
    const ExpertCache & cache() const { return cache_; }   // this stage's (cache_*() above: every stage's)
    MtpLayer * mtp() { return next_ ? next_->mtp() : mtp_.loaded() ? &mtp_ : nullptr; }
    void save_counts();

private:
    // ---- pipeline stages: this stage's device and layers [l0_, l1_); head_ is the first stage (which holds the
    // shared counters), next_ the following one. The *_here functions act on this stage alone; the public ones on
    // the first stage run every stage.
    int dev_ = 0, l0_ = 0, l1_ = 0;
    bool last_ = true;
    Engine * head_ = this;
    std::unique_ptr<Engine> next_;
    Engine * tail() { return next_ ? next_->tail() : this; }
    bool first() const { return l0_ == 0; }
    bool owns(int il) const { return il >= l0_ && il < l1_; }
    void load_stage(const std::string & path, const EngineOptions & opt);
    // the residual rows between stages: pinned, portable host memory written by this stage, read by the next one
    // (xfer_out_ is the next stage's xfer_in_); decode windows and prompt chunks have their own, prompt chunks two
    // (the first stage writes the next chunk's while the later stages read this one's)
    float * xfer_in_[kLanes] = {}, * pf_xfer_in_[2] = {nullptr, nullptr};
    float * xfer_out_[kLanes] = {}, * pf_xfer_out_[2] = {nullptr, nullptr};
    int lane_ = 0;   // the lane of the forward this stage is preparing / running
    // Prompt reads pipelined over the stages: the first stage works on chunk k+1 (on the caller's thread) while
    // this worker runs chunk k through the later ones. Between chunks both are idle (the caller's progress hook
    // may decode other conversations then).
    struct Worker {
        std::thread th;
        std::mutex mu;
        std::condition_variable cv;
        std::function<void()> job;
        bool busy = false, quit = false;
        std::exception_ptr err;
        Worker();
        ~Worker();
        void post(std::function<void()> f);   // after the previous job
        void wait();                          // until idle; rethrows the job's exception
    };
    std::unique_ptr<Worker> worker_;
    std::unique_ptr<Worker> svc_;   // forward_pipe: answers the second stage's CPU experts meanwhile
    std::unique_ptr<Worker> la_worker_;   // lookahead: the drafter's chain-ahead (the service must run beside it)
    std::unique_ptr<Worker> spec_worker_;  // lookahead: the speculative forward's CPU experts and its end, while the verdict is processed
    bool spec_inflight_ = false;
    void spec_join();                      // waits for the speculative forward (its end records spec_T)
    std::mutex stats_mu_;   // the first stage's counters, updated by the prompt path of every stage
    cudaEvent_t ev_out_ = nullptr;   // this stage's part of the forward is done (the next stage's stream waits)
    cudaEvent_t ev_t0_ = nullptr, ev_t1_ = nullptr;   // timing of this stage's part
    double gpu_ms_ = 0, pf_ms_ = 0;
    int64_t gpu_forwards_ = 0, pf_tokens_ = 0;
    void reset_here();
    void select_here(int slot);
    void checkpoint_here();
    int rollback_here(int max_pos);
    void commit_here(int c);
    void join_commit_here();
    void ensure_ctx_here(int cells);
    void rebalance_here();
    void save_counts_here();
    // forward_batch in four steps per stage: book the windows (and capture the graph); launch it after the previous
    // stage's; service the CPU experts; wait
    void fwd_prepare(const std::vector<BatchWin> & batch, int lane = 0, bool spec = false);
    bool spec_ = false;   // the forward being prepared / ended is a speculative one
    struct Seq;
    void swap_shadow(Seq & q);   // the slot's recurrent state pointers <-> its shadows (around a speculative forward)
    void spec_begin(Seq * q, const std::vector<int32_t> & tokens, int lane);   // first stage: shadow := real + the pending window committed in full, then the speculative forward launched
    void fwd_launch(cudaEvent_t after);
    bool fwd_graph_ = true;   // the forward in flight runs as a graph (else eagerly, the CPU part inside)
    cudaGraphExec_t fwd_exec_ = nullptr;
    double fwd_t0_ = 0;
    void fwd_service();
    void fwd_end();
    void prefill_chunk_here(const int32_t * tokens, int N, int buf = 0);
    // the VRAM budget shared by the expert cache, the context (KV) and the prompt path
    VramBudget budget_;
    void enqueue_forward(int T);
    void layer_forward(int il, int T);
    void hc_pre(const HcWeights & w, int T, bool want_inject);
    void ple_gather();
    void ple(int il, int T);
    void gdn(int il, int T);
    void attn(int il, int T);
    void moe(int il, int T);
    void head(int T);
    void service_cpu(uint32_t seq, int T);
    void enqueue_commit(int c, bool shadow = false);
    void prefill_alloc(int max_n);
    void prefill_staging_alloc();
    void pf_hc_pre(const HcWeights & w, int N, bool inject);
    void pf_ple(int il, int N);
    void pf_gdn(int il, int N);
    void pf_attn(int il, int N);
    void pf_moe(int il, int N);
    void pf_stage_layer(int il);
    void pf_stage_list(int il, const std::vector<int> & experts);
    void mtp_feed(const float * R_rows, const int32_t * tokens, int n, int pos0);

    Model model_;
    ExpertStore own_store_;          // the first stage's (every stage uses it)
    ColdStore own_cold_;             // the cold experts' blobs (docs/EXPERT-MIX.md), the first stage's too
    ColdStore * cold_store_ = &own_cold_;
    ExpertStore * store_ = &own_store_;
    ExpertCache cache_;
    MtpLayer mtp_;
    CpuExpertPool own_cpu_;
    CpuExpertPool * cpu_ = &own_cpu_;
    EngineOptions opt_;
    cudaStream_t st_ = nullptr;
    cudaStream_t st_commit_ = nullptr;
    // Independent work of a layer on side streams (captured into the forward graph as parallel branches): the
    // shared expert beside the routed ones, the DeltaNet gate/beta/alpha projections beside wqkv, the QSA indexer
    // beside q/k/v. fork(i) starts side stream i after what st_ has queued; join(i) makes st_ wait for it.
    // BNK_SIDE_STREAMS=0: everything on st_.
    cudaStream_t st_side_[2] = {nullptr, nullptr};
    cudaEvent_t ev_fork_[2] = {nullptr, nullptr}, ev_join_[2] = {nullptr, nullptr};
    bool side_ = true;
    cudaStream_t fork(int i);
    const float * pend_out_ = nullptr;   // an hc_combine(res_, pend_out_, inj_) left to the next hc_pre's norm
    bool mix_quantized_ = false;   // mixact_ holds quantize(mixed_) (hc_pre's fused hc_mix_q), for the next user
    void quantize_mixed(int T);
    void join(int i);
    cudaEvent_t ev_verified_ = nullptr, ev_committed_ = nullptr;
    bool commit_inflight_ = false;
    struct Checkpoint { int pos = -1, mtp_cell = -1; uint64_t age = 0; float * host = nullptr; };
    uint64_t ckpt_age_ = 0;
    static constexpr int kMaxCheckpoints = 6;

    // One conversation's state on the GPU: its history, the KV cache and indexer keys of every attention layer,
    // the DeltaNet recurrences and conv histories, the PLE n-gram history, its snapshots, and the CUDA graphs that
    // bake its buffers' addresses. The engine holds opt.slots of them; cur_ is the one the single-sequence API
    // (prefill, forward, commit, checkpoint, park ...) works on.
    struct Seq {
        std::vector<int32_t> history;
        std::vector<Checkpoint> ckpts;
        float * mtp_R = nullptr;
        int mtp_cell = -1;
        int pending_T = 0;   // rows of the last verify window not yet committed
        std::vector<DevBuf<float>> conv_buf;    // [(K-1)+W][C] per GDN layer
        DevBuf<float *> conv_ptrs;              // the GDN layers' conv_buf pointers (one shift kernel at commit)
        std::vector<DevBuf<float>> ssm_state;   // [nv][S][S]
        std::vector<DevBuf<half>> kc, vc;       // [max_ctx][Hkv][D] per attention layer
        std::vector<DevBuf<half>> kraw;         // QSA raw indexer keys and pooled block keys per attention layer
        std::vector<DevBuf<float>> pooled;
        std::vector<ElasticBuf> kv_k, kv_v, kv_raw, kv_pool;
        int ctx_mapped = 0;   // cells of context whose KV is mapped
        DevBuf<float> ple_hist;
        cudaGraphExec_t commit_graphs[kLanes][kMaxWindow][kMaxWindow + 1] = {};   // [lane][first row][kept rows]
        int last_r0 = 0;     // where its rows were in the last forward (for its commit)
        int last_lane = 0;   // ... and in which lane's buffers
        bool used = false;   // ever held a conversation (an unused slot maps no context)
        // lookahead (the first stage of a two-stage pipeline): the speculative window's intermediates live in
        // the other lane, its recurrent state in these shadow copies (promoted when the speculation held)
        std::vector<DevBuf<float>> ssm_shadow, conv_shadow;
        DevBuf<float> ple_shadow;
        DevBuf<float *> conv_ptrs_shadow;
        int spec_T = 0, spec_r0 = 0, spec_lane = -1;   // the speculative window run (spec_T 0: none)
    };
    std::vector<std::unique_ptr<Seq>> seqs_;
    Seq * cur_ = nullptr;
    // the forward in flight: each window's slot, first row, rows, position and whether it commits at once
    struct Win { Seq * q; int r0, T, pos0; bool commit; };
    std::vector<Win> wins_;
    // per window layout (slot, T, commit ... in slot order); least recently used ones are dropped beyond
    // kMaxFwdGraphs: every instantiated graph holds device memory, and batched layouts vary round to round
    struct FwdGraph { cudaGraphExec_t exec = nullptr; uint64_t used = 0; size_t bytes = 0; };
    std::map<std::vector<int>, FwdGraph> fwd_graphs_;
    uint64_t fwd_graph_age_ = 0;
    int64_t graph_captures_ = 0;
    size_t graph_bytes_ = 0;   // device memory the instantiated graphs took (measured at capture)
    int max_fwd_graphs_ = 32;   // 48 with lookahead (its layouts in the other lane; BNK_FWD_GRAPHS)
    static constexpr size_t kGraphBytesEst = 24ull << 20;   // ~18 MiB measured per instantiated forward graph
    static constexpr int kMaxSlots = 8;
    void alloc_seq(Seq & q);
    void use(Seq * q);   // cur_ = q, and the drafter's slot with it
    template <typename F> void each_state(F && f);   // f(device ptr, floats) over every saved buffer
    size_t state_floats();
    // f(device ptr, bytes) over every buffer holding per-position state for cells [0, cells) (KV, indexer keys
    // and their pooled blocks, the drafter's)
    template <typename F> void each_ctx(int cells, F && f);
    template <typename F> void each_ctx_of(Seq & q, int cells, F && f);   // the same over another slot's buffers
    int slot_of(const Seq * q) const;
    // Forking: the live slot takes the first `pos` tokens of another conversation's state (its KV cells and the
    // snapshot it holds at pos) while that conversation keeps its own: a new conversation that starts with a
    // system prompt another one has read. Decided on the first stage, carried out on every stage.
    void fork_parked(uint64_t id, int pos);
    void fork_slot(int src, int pos);
    void fork_copy_parked(uint64_t id, int pos, bool finish);
    void fork_copy_slot(int src, int pos, bool finish);
    void fork_done_here(const std::vector<int32_t> & history, int pos, int mtp_cell);
    int fork_min_ = 64;   // tokens a fork must save (BNK_FORK_MIN; 0 turns forking off)
    struct Parked {
        std::vector<int32_t> history;
        int mtp_cell = -1;
        std::vector<Checkpoint> ckpts;   // own pinned buffers
        uint8_t * host = nullptr;        // pinned: the context ranges, then the recurrent state
        size_t cap = 0;                  // host's size (from the pool, with room to grow)
        size_t bytes = 0;
        uint64_t used = 0;
    };
    std::vector<Parked> parked_;
    uint64_t park_age_ = 0;
    // Pinned buffers (pin_host) are kept for reuse: pinning still costs ~50 ms per GiB, and conversations taking
    // turns hand the same few buffers around (a restore frees what the next park needs).
    struct Pinned { uint8_t * p; size_t cap; };
    std::vector<Pinned> spare_;          // conversation buffers
    std::vector<float *> ck_spare_;      // snapshot buffers (all state_floats() long)
    uint8_t * pinned_take(size_t bytes, size_t * cap);
    void pinned_give(uint8_t * p, size_t cap);
    float * ck_take();
    float * ck_alloc();   // a new snapshot buffer (throws when RAM cannot be pinned)
    void ck_free(float * p);
    void ck_give(float * p);
    void trim_spares(size_t extra = 0);   // spares beyond the parking budget (less `extra`) go back to the OS
    // move_ckpts: the live snapshots go along instead of being copied (the live state is replaced next).
    // Decided on the first stage (park), carried out on every stage (park_copy: one entry each, in step)
    void park(uint64_t keep = 0, bool move_ckpts = false);
    bool park_copy(uint64_t id, bool move_ckpts);
    size_t park_bytes_here(int n);   // this stage's live state for n cells (context + recurrent)
    void restore_copy(uint64_t id, bool finish);
    // VRAM for `need` more bytes on this stage when its expert cache has nothing left to give: idle conversations
    // (largest first) are parked in host RAM and their context unmapped, on every stage. False when that is not
    // enough either.
    bool evict_idle(size_t need);
    bool evict_idle_for(Engine * stage, size_t need);   // on the first stage
    std::vector<char> busy_;
    uint64_t restoring_ = 0;   // the parked entry restore() is bringing back: evict_idle's parking keeps it
    // the prompt path's layout for chunks of N tokens; false when VRAM is short (the old layout stays)
    bool try_layout(int N);
    // the expert staging for this read; a whole layer's worth falls back to the small stage when VRAM is short
    void staging_fit();
    void restore(size_t i);
    void drop_parked(size_t i);
    void drop_parked_id(uint64_t id);   // every stage's entry of that conversation
    uint32_t seq_ = 0;
    int64_t fwd_count_ = 0;

    // window inputs (pinned host) and their device copy
    WinParams * h_par_ = nullptr;
    WinParams * d_par_ = nullptr;
    int32_t * h_tok_ = nullptr;
    uint8_t * h_ple_rows_ = nullptr;

    // device activations (sized for kMaxWindow)
    DevBuf<float> x_, res_, xn_, lo_, gpre_, mixed_, inj_, out_;
    DevBuf<float> conv_out_, z_, alpha_, beta_, gdn_o_, gdn_n_;
    DevBuf<float> qfull_, k_, v_, q_, attn_o_, attn_scratch_;
    DevBuf<float> rlogits_, rw_, sg_, su_, sh_, sgate_, shared_out_, moe_out_, logits_;
    DevBuf<int32_t> rids_, argmax_dev_, topk_ids_;
    DevBuf<float> topk_vals_;
    DevBuf<double> topk_z_;
    DevBuf<float> ple_emb_, ple_key_, ple_val_, ple_gated_;
    DevBuf<int8_t> actq_, mixq_;
    DevBuf<half> acth_, mixh_;   // the activations in fp16 too (tensor-core GEMV for several tokens)
    DevBuf<float> actd_, mixd_;
    DevBuf<int16_t> acts_, mixs_;
    ActQ8 act_, mixact_;
    // MoE
    MoeScratch moes_;
    DevBuf<uint8_t> hits_buf_;
    DevBuf<float> gu_buf_, hd_buf_, part_buf_;
    DevBuf<int8_t> hq_buf_;
    DevBuf<uint32_t> counts_;
    std::vector<uint32_t> saved_counts_;
    uint8_t * mail_ = nullptr;  // host-mapped mailboxes, one per layer
    size_t mail_stride_ = 0;
    MoeMsg * msg(int il) const { return (MoeMsg *) (mail_ + (size_t) il * mail_stride_); }
    std::vector<ExpertTask> tasks_;
    // per-layer recurrent state
    int n_gdn_ = 0;
    // per lane and GDN layer: the window's conv outputs, decay, beta (kept for the commit)
    std::vector<DevBuf<float>> gdn_co_[kLanes], gdn_g_[kLanes], gdn_b_[kLanes];
    // QSA: raw indexer keys and pooled block keys per attention layer, selection scratch
    DevBuf<float> ik_, iq_, qsa_scores_;
    DevBuf<int32_t> qsa_sel_, qsa_nsel_;
    QsaShape qsh_{};
    int max_blocks_ = 0;
    bool qsa_on_ = false;

    // ---- batched prompt processing
    Gemm gemm_;
    int pf_max_ = 0;   // chunk size the prompt buffers are laid out for now
    int pf_cap_ = 0;   // the largest chunk the host-side buffers take
    // elastic VRAM (budget_ is declared first so it outlives every buffer charged to it)
    ElasticBuf pf_arena_, pf_stage_b_[2];
    void prefill_layout(int N);
    void predict_stats(int il, int T);
    // BNK_ROUTE_LOG=file: per decode forward, every row's routed experts per layer (negative: not resident), the
    // per-layer GPU timestamps and CPU expert times (tools/batch_sim.py reads it)
    FILE * route_log_ = nullptr;
    DevBuf<int32_t> route_dev_;          // [n_layer][kMaxWindow * k]
    std::vector<float> cpu_us_;          // this forward's CPU expert time per layer
    void write_route_log(int T, int pos0, bool commit_all, double t0, double t1);
    DevBuf<float> pred_logits_, pred_w_;
    DevBuf<int32_t> pred_ids_;
    size_t prefill_arena_bytes(int N);
    size_t prefill_carve(int N, bool dry_only);
    void map_ctx(int cells);
    int pf_small_ = 0, pf_base_ = 0, pf_big_ = 0;   // chunk layouts: between prompts, normal, long prompts
    bool cache_ready_ = false;
    static constexpr int kCtxStep = 8192, kCtxBaseline = 16384;
    void prefill_staging_ensure(bool full);
    bool pf_full_ = false;   // this read stages whole layers (checked again before every chunk)
    bool pf_tc_ = true;      // the routed experts of a chunk on the tensor cores (moe_tc; BNK_PF_TC=0: cuBLAS + list)
    void prefill_staging_shrink();
    struct {
        DevBuf<float> res, xn, lo, gpre, mixed, inj, out, conv, co, z, g, b, o, n, qfull, k, v, q, ao, rlog, w, sg, su,
            sh, sgate, shared, moe, ple_emb, ple_key, ple_val, ple_gated, ple_hist, x;
        DevBuf<int32_t> ids, perm, inv, tok;
        DevBuf<half> xg, gu, hh, dd, wexp;
        uint8_t * ple_rows = nullptr;            // pinned host
        int32_t * h_ids = nullptr, * h_perm = nullptr, * h_inv = nullptr;
        float * h_w = nullptr;
        uint8_t * stage[2] = {nullptr, nullptr};  // non-resident experts of a layer, streamed over PCIe
        size_t stage_bytes = 0;
        std::vector<std::vector<int>> stage_idx;   // [layer][expert] -> index in the stage (-1: resident)
        cudaStream_t copy = nullptr;
        cudaEvent_t copied[2] = {nullptr, nullptr}, used[2] = {nullptr, nullptr}, xready = nullptr;
        int wexp_experts = 0;                     // experts dequantized per grouped call
        bool prefetch = false;                    // this chunk streams every non-resident expert ahead
        float * h_x = nullptr, * h_cpu = nullptr; // pinned: the chunk's MoE input, the CPU experts' output
        DevBuf<float> cpu;                        // [N][E] the CPU part on the device
        DevBuf<float> ik, iq, scores;             // QSA for prompt rows
        DevBuf<int32_t> sel, nsel;
        DevBuf<int8_t> xq, hq;                    // list path: the chunk's input quantized; h per pair
        DevBuf<float> xd, hd, gu32;
        DevBuf<uint8_t> items;                    // PfItem[]
        PfItem * h_items = nullptr;
        int max_items = 0;
    } pf_;
};

// (here so that park.cpp can use it as well)
template <typename F> void Engine::each_state(F && f) {
    const Config & c = model_.cfg;
    for (int il = l0_; il < l1_; ++il)
        if (!c.is_attn(il)) {
            f(cur_->conv_buf[il].p, cur_->conv_buf[il].n);
            f(cur_->ssm_state[il].p, cur_->ssm_state[il].n);
        }
    if (cur_->ple_hist.p) f(cur_->ple_hist.p, cur_->ple_hist.n);
    if (cur_->mtp_R) f(cur_->mtp_R, (size_t) c.hc_dim());
}

}  // namespace bnk
