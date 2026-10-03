// The qwen4exp forward pass over a window of up to kMaxWindow tokens.
//
// One forward = one CUDA graph launch (captured per window size). Window inputs (token ids, position,
// sequence number, PLE rows) live in pinned host memory that the graph reads. Routed experts resident
// in the VRAM cache run on the GPU; the others are sent through per-layer mailboxes to the CPU pool,
// which the calling thread services while the graph runs.
#pragma once

#include <memory>
#include <functional>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "core/expert_cache.h"
#include "core/vmem.h"
#include "core/model.h"
#include "core/util.h"
#include "cpu/expert_pool.h"
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
    int prefill_chunk_max = 8192;    // chunk size for long prompts (borrows VRAM from the expert cache meanwhile)
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
};

struct StageTimes {
    double total_ms = 0, cpu_experts_ms = 0, ple_ms = 0, wait_ms = 0;
    int64_t misses = 0, routed = 0;
    int calls = 0;
};

struct WinParams {     // pinned host; copied to the device by the graph's first node
    int32_t pos0;
    int32_t T;
    uint32_t seq;
    int32_t pad;
};

class Engine {
public:
    ~Engine();
    void load(const std::string & path, const EngineOptions & opt);
    void reset();

    // Runs tokens[0..T) at positions pos()..pos()+T-1; logits of every window position stay on the device
    // and the final residual of every row in residual_dev(). With commit_all the window is committed;
    // otherwise (a verify window) nothing is until commit(c) keeps its first c rows.
    void forward(const int32_t * tokens, int T, bool commit_all = true);
    void commit(int c);
    const float * residual_dev() const { return res_; }
    cudaStream_t stream() const { return st_; }
    // Processes a prompt (batched passes for long stretches, decode windows for short tails) and commits it;
    // logits of its last token are in logits row 0... last_T-1 (argmax(last_T - 1)). With `mtp` set, the
    // draft layer's K/V are filled for every cell except the last token's (which pairs with the next one).
    void prefill(const std::vector<int32_t> & tokens);
    // VRAM for the context and the prompt path is elastic: KV for cells [0, cells) is mapped on demand
    // (the expert cache gives the bytes up), and rebalance() hands spare bytes back to the cache.
    void ensure_ctx(int cells);
    void rebalance();
    double last_rebalance_ms = 0;
    size_t vram_budget() const { return budget_.limit(); }
    size_t kv_bytes_mapped() const;
    // One batched prompt pass over N tokens (commits; logits of the last token in row 0).
    void prefill_chunk(const int32_t * tokens, int N);
    struct PrefillStats { double ms = 0, copy_wait_ms = 0; int64_t tokens = 0, chunks = 0; } pstats;
    // After prefill with an MTP layer: the last prompt row's residual and cell, which pair with the first
    // generated token (Generator drafts from them).
    const float * mtp_pending_R() const { return mtp_R_; }
    int mtp_pending_cell() const { return mtp_cell_; }
    int argmax(int t);
    void argmax_all(int T, int32_t * out);  // argmax of every row of the last window
    std::vector<float> logits_host(int t);
    const float * logits_dev() const { return logits_; }

    int pos() const { return (int) history_.size(); }
    // Snapshots of the per-position-independent state (DeltaNet recurrences, conv and n-gram histories, the
    // drafter's last residual) so a later prompt that shares only part of the history can resume from the last
    // snapshot inside the shared part instead of starting over. The KV cache needs none: it is per position.
    void checkpoint();
    // Restores the latest snapshot at or before max_pos; returns its position (the history is cut there), or -1.
    int rollback(int max_pos);
    int checkpoints() const;
    const std::vector<int32_t> & history() const { return history_; }
    int max_ctx() const { return opt_.max_ctx; }
    int cpu_threads() const { return cpu_.threads(); }
    void logits_rows_host(int T, float * out);   // rows 0..T-1 of the last window's logits
    void set_mtp_pending(const float * R_row_dev, int cell);
    const Config & cfg() const { return model_.cfg; }
    const Model & model() const { return model_; }
    const ExpertCache & cache() const { return cache_; }
    MtpLayer * mtp() { return mtp_.loaded() ? &mtp_ : nullptr; }
    StageTimes times;
    // lifetime routing counters per layer (expert uses, and those served by the CPU)
    std::vector<int64_t> layer_routed, layer_misses;
    // called after every prefill chunk/window with (tokens done, tokens total)
    std::function<void(size_t, size_t)> on_prefill_progress;
    void save_counts();

    // debugging (disables graphs): HC residual after every layer, dumped_layers[il] = [T][hc*E]
    bool dump_all = false;
    std::vector<std::vector<float>> dumped_layers;
    int last_T = 0;

private:
    // the VRAM budget shared by the expert cache, the context (KV) and the prompt path
    VramBudget budget_;
    void enqueue_forward(int T);
    void layer_forward(int il, int T);
    void hc_pre(const HcWeights & w, int T, bool want_inject);
    void ple_gather(int T);
    void ple(int il, int T);
    void gdn(int il, int T);
    void attn(int il, int T);
    void moe(int il, int T);
    void head(int T);
    void service_cpu(uint32_t seq, int T);
    void enqueue_commit(int c);
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
    float * mtp_R_ = nullptr;
    int mtp_cell_ = -1;

    Model model_;
    ExpertStore store_;
    ExpertCache cache_;
    MtpLayer mtp_;
    CpuExpertPool cpu_;
    EngineOptions opt_;
    cudaStream_t st_ = nullptr;
    std::vector<int32_t> history_;
    struct Checkpoint { int pos = -1, mtp_cell = -1; uint64_t age = 0; float * host = nullptr; };
    std::vector<Checkpoint> ckpts_;
    uint64_t ckpt_age_ = 0;
    static constexpr int kMaxCheckpoints = 6;
    template <typename F> void each_state(F && f);   // f(device ptr, floats) over every saved buffer
    uint32_t seq_ = 0;
    int64_t fwd_count_ = 0;
    cudaGraphExec_t graphs_[2][kMaxWindow + 1] = {};   // [commit_all][T]
    cudaGraphExec_t commit_graphs_[kMaxWindow + 1] = {};
    bool commit_all_ = true;
    int pending_T_ = 0;  // rows of the last verify window not yet committed

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
    DevBuf<int32_t> rids_, argmax_dev_;
    DevBuf<float> ple_emb_, ple_key_, ple_val_, ple_gated_, ple_hist_;
    DevBuf<int8_t> actq_, mixq_;
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
    std::vector<DevBuf<float>> conv_buf_;   // [(K-1)+W][C] per GDN layer
    std::vector<DevBuf<float>> ssm_state_;  // [nv][S][S]
    std::vector<DevBuf<float>> gdn_co_, gdn_g_, gdn_b_;  // per GDN layer: the window's conv outputs, decay, beta
    std::vector<DevBuf<half>> kc_, vc_;     // [max_ctx][Hkv][D] per attention layer
    // QSA: raw indexer keys and pooled block keys per attention layer, selection scratch
    std::vector<DevBuf<half>> kraw_;
    std::vector<DevBuf<float>> pooled_;
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
    std::vector<ElasticBuf> kv_k_, kv_v_, kv_raw_, kv_pool_;
    int ctx_mapped_ = 0;   // cells of context whose KV is mapped
    void prefill_layout(int N);
    void predict_stats(int il, int T);
    DevBuf<float> pred_logits_, pred_w_;
    DevBuf<int32_t> pred_ids_;
    size_t prefill_arena_bytes(int N);
    size_t prefill_carve(int N, bool dry_only);
    void map_ctx(int cells);
    int pf_small_ = 0, pf_base_ = 0, pf_big_ = 0;   // chunk layouts: between prompts, normal, long prompts
    bool cache_ready_ = false;
    static constexpr int kCtxStep = 8192, kCtxBaseline = 16384;
    void prefill_staging_ensure(bool full);
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

}  // namespace bnk
