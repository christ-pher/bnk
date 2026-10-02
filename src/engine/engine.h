// The qwen4exp forward pass over a window of up to kMaxWindow tokens.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "core/model.h"
#include "core/util.h"
#include "cpu/expert_pool.h"
#include "kernels/gemv.h"

namespace bnk {

struct EngineOptions {
    int max_ctx = 8192;
    int cpu_threads = 24;
    bool verbose = true;
};

struct StageTimes {
    double gpu_ms = 0, cpu_experts_ms = 0, ple_ms = 0, total_ms = 0;
    int calls = 0;
};

class Engine {
public:
    void load(const std::string & path, const EngineOptions & opt);
    void reset();

    // Runs tokens[0..T) at positions pos()..pos()+T-1 and commits them. Logits of every window
    // position stay on the device (logits_dev()); pass `want_logits` false to skip the head.
    void forward(const int32_t * tokens, int T, bool want_logits = true);
    // Prompt processing in windows of kMaxWindow; logits of the last token only.
    void prefill(const std::vector<int32_t> & tokens);
    int argmax(int t);
    std::vector<float> logits_host(int t);
    const float * logits_dev() const { return logits_; }

    int pos() const { return (int) history_.size(); }
    const Config & cfg() const { return model_.cfg; }
    const Model & model() const { return model_; }
    StageTimes times;

    // debugging: copy of the HC residual after the last forward's layer `il` (set dump_layer first)
    int dump_layer = -1;
    std::vector<float> dumped;
    bool dump_all = false;                      // res after every layer: dumped_layers[il] = [T][hc*E]
    std::vector<std::vector<float>> dumped_layers;
    int last_T = 0;

private:
    void layer_forward(int il, int T);
    void hc_pre(const HcWeights & w, int T, bool want_inject);
    void ple(int il, int T);
    void gdn(int il, int T);
    void attn(int il, int T);
    void moe(int il, int T);
    void head(int T);

    Model model_;
    ExpertStore store_;
    CpuExpertPool cpu_;
    EngineOptions opt_;
    cudaStream_t st_ = nullptr;
    std::vector<int32_t> history_;
    int pos0_ = 0;  // position of window token 0 during a forward

    // device activations (sized for kMaxWindow)
    DevBuf<float> x_, res_, xn_, lo_, gpre_, mixed_, inj_, out_;
    DevBuf<float> conv_out_, z_, alpha_, beta_, gdn_o_, gdn_n_;
    DevBuf<float> qfull_, k_, v_, q_, attn_o_, attn_scratch_;
    DevBuf<float> rlogits_, rw_, sg_, su_, sh_, sgate_, moe_out_, logits_;
    DevBuf<int32_t> rids_, tok_dev_, argmax_dev_;
    DevBuf<float> ple_emb_, ple_key_, ple_val_, ple_gated_, ple_hist_;
    DevBuf<uint8_t> ple_rows_dev_;
    DevBuf<int8_t> actq_;
    DevBuf<float> actd_;
    ActQ8 act_;
    // per-layer recurrent state
    std::vector<DevBuf<float>> conv_buf_;   // [(K-1)+W][C] per GDN layer
    std::vector<DevBuf<float>> ssm_state_;  // [nv][S][S]
    std::vector<DevBuf<half>> kc_, vc_;     // [max_ctx][Hkv][D] per attention layer
    // host staging (pinned)
    float * h_x_ = nullptr, * h_out_ = nullptr, * h_w_ = nullptr;
    int32_t * h_ids_ = nullptr;
    uint8_t * h_ple_rows_ = nullptr;
    std::vector<ExpertTask> tasks_;
    bool qsa_warned_ = false;
};

}  // namespace bnk
