// DeepSeek-V4.1-Flash (GGUF architecture "deepseek41"): hyper-parameters, the dense weights of a pipeline stage in
// VRAM, the routed experts' sources in the memory-mapped file, and the Engram tables (a second GGUF, mapped, read
// by rows). docs/DEEPSEEK.md describes the model and the memory plan.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "core/gguf_model.h"
#include "core/model.h"
#include "kernels/gemv.h"

namespace bnk {

struct DsConfig {
    int n_layer = 0, n_embd = 0, n_vocab = 0;
    int n_head = 0, head_dim = 0, n_rot = 0, q_lora = 0, o_groups = 0, o_lora = 0;
    int window = 0;
    std::vector<int> ratio;      // per layer: 0 = window only, r = compressed KV r-to-1 shared from kv_src
    std::vector<int> kv_src;     // per layer: the layer whose compressed KV it reads (-1: none)
    std::vector<int> topk_src;   // per layer: the layer whose indexer selection it reads (-1: none)
    int cand_src = -1, cand_blocks = 0, cand_bs = 0;
    int idx_heads = 0, idx_dim = 0, idx_topk = 0;
    float rope_base = 1e4f, comp_rope_base = 1.6e5f;
    float yarn_factor = 1.f, yarn_beta_fast = 32.f, yarn_beta_slow = 1.f;
    int yarn_orig = 0;
    int hc = 0, sinkhorn = 20;
    float hc_eps = 1e-6f, eps = 1e-20f;
    int n_expert = 0, n_used = 0, n_ff = 0;
    float route_scale = 1.f, swiglu_limit = 0.f;
    bool route_norm = true;
    // Engram: n-gram hash tables at a few layers
    std::vector<int> engram_layers;
    int eg_ngram = 0, eg_heads = 0, eg_dim = 0, eg_pad = 0;
    std::vector<int32_t> eg_token_map;              // token id -> compressed id
    std::vector<std::vector<uint64_t>> eg_primes;   // [engram layer][(ngram-1) * heads]
    std::vector<std::vector<uint64_t>> eg_offsets;  // row offsets inside that layer's table
    std::vector<std::vector<uint64_t>> eg_mult;     // [engram layer][ngram]
    std::vector<int64_t> eg_rows;                   // rows of each layer's table
    int eg_cols() const { return (eg_ngram - 1) * eg_heads; }
    int engram_index(int il) const {
        for (size_t i = 0; i < engram_layers.size(); ++i) if (engram_layers[i] == il) return (int) i;
        return -1;
    }
    bool is_kv_src(int il) const { return kv_src[il] == il; }
    bool is_idx_src(int il) const { return topk_src[il] == il; }
    int hc_dim() const { return hc * n_embd; }
    int mix_dim() const { return (2 + hc) * hc; }
};

struct DsLayer {
    QMat attn_norm, ffn_norm, sinks;                         // f32
    QMat q_a, q_a_norm, q_b, kv, kv_norm, o_a, o_b;          // o_a: [o_groups * o_lora] rows of n_head*head_dim/o_groups
    QMat comp_kv, comp_gate, comp_norm, idx_k, idx_k_norm;   // kv-source layers (comp_gate only for ratio > 1)
    QMat idx_q_b, idx_proj;                                  // index-source layers
    QMat hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;   // f32
    QMat eg_wkv, eg_gate;                                    // Engram layers; eg_gate = q_weight * k_weight, f32 [hc][E]
    QMat router, probs_b;                                    // f32
    QMat sh_gate, sh_up, sh_down, sh_gu;                     // the shared expert (gate | up stacked when possible)
};

struct DsModel {
    Config cfg;        // the fields the shared code reads (vocabulary, experts, eos ...)
    DsConfig ds;
    GgufModel gguf, engram_gguf;
    std::vector<DsLayer> layers;
    QMat tok_embd, output_norm, output;
    std::vector<const TensorRef *> engram_tab;   // per Engram layer (host, mapped)
    // the routed experts as the shared expert machinery (ExpertStore, ExpertCache, CpuExpertPool) sees them
    Model ev;
    int l0 = 0, l1 = 0;
    size_t vram_dense_bytes = 0;

    // dense weights of layers [l0, l1) to the current device; the token table with `embd`, the head with `head`;
    // every layer's expert sources either way. engram_path: the tables' GGUF (empty: beside the model, named
    // by deepseek41.engram.file).
    void load(const std::string & path, const std::string & engram_path, bool verbose, int l0 = 0, int l1 = -1,
              bool embd = true, bool head = true);
    ~DsModel();

private:
    std::vector<void *> device_allocs_;
    std::vector<void *> host_allocs_;
    QMat upload(const std::string & name, bool required = true);
    QMat upload_tensor(const TensorRef & t, const std::string & name);
    QMat upload_f32(const std::vector<float> & v, int64_t rows, int64_t cols);
    bool stack(const std::string & na, const std::string & nb, QMat & a, QMat & b, QMat & ab);
    void expert_src(int il);
};

DsConfig load_ds_config(const GgufModel & g);

}  // namespace bnk
