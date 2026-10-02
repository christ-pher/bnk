// The qwen4exp forward pass over a window of up to kMaxWindow tokens.
//
// One forward = one CUDA graph launch (captured per window size). Window inputs (token ids, position,
// sequence number, PLE rows) live in pinned host memory that the graph reads. Routed experts resident
// in the VRAM cache run on the GPU; the others are sent through per-layer mailboxes to the CPU pool,
// which the calling thread services while the graph runs.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "core/expert_cache.h"
#include "core/model.h"
#include "core/util.h"
#include "cpu/expert_pool.h"
#include "engine/mtp.h"
#include "kernels/gemv.h"
#include "kernels/moe.h"

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
    void prefill(const std::vector<int32_t> & tokens);
    int argmax(int t);
    void argmax_all(int T, int32_t * out);  // argmax of every row of the last window
    std::vector<float> logits_host(int t);
    const float * logits_dev() const { return logits_; }

    int pos() const { return (int) history_.size(); }
    const Config & cfg() const { return model_.cfg; }
    const Model & model() const { return model_; }
    const ExpertCache & cache() const { return cache_; }
    MtpLayer * mtp() { return mtp_.loaded() ? &mtp_ : nullptr; }
    StageTimes times;
    void save_counts();

    // debugging (disables graphs): HC residual after every layer, dumped_layers[il] = [T][hc*E]
    bool dump_all = false;
    std::vector<std::vector<float>> dumped_layers;
    int last_T = 0;

private:
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

    Model model_;
    ExpertStore store_;
    ExpertCache cache_;
    MtpLayer mtp_;
    CpuExpertPool cpu_;
    EngineOptions opt_;
    cudaStream_t st_ = nullptr;
    std::vector<int32_t> history_;
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
    bool qsa_warned_ = false;
};

}  // namespace bnk
