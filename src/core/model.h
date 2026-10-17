// qwen4exp model description: hyper-parameters from GGUF metadata and the placement of every weight.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "core/gguf_model.h"
#include "kernels/gemv.h"
#include "kernels/moe.h"

namespace bnk {

enum class Arch { QWEN4EXP, GLM5NEXT };

struct Config {
    Arch arch = Arch::QWEN4EXP;
    int n_layer = 0, n_embd = 0, n_vocab = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0;           // full attention
    int n_rot = 0;                                         // rotary dims per head (NeoX pairing)
    float rope_base = 1e7f;
    float rms_eps = 1e-6f;
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
    float expert_weights_scale = 0.f;                      // 0 / 1 = none
    // gated delta net
    int ssm_conv = 0, ssm_state = 0, ssm_groups = 0, ssm_vheads = 0, ssm_inner = 0;
    int full_attn_interval = 4;
    // hyper-connections
    int hc = 0, hc_rank = 0;
    // QSA indexer
    int idx_heads = 0, idx_dim = 0, idx_top_k = 0;
    std::vector<int> compress_ratio;                       // per layer, 0 = dense attention
    // PLE n-gram embeddings
    int ple_layer = -1, ple_ngram = 0, ple_heads_per_ngram = 0, ple_conv = 0, ple_dim = 0;
    int ple_eos = 0, ple_image = 0;
    std::vector<uint64_t> ple_mult, ple_offset, ple_vocab;
    int eos_token = 0, pad_token = 0;
    std::vector<int> eos_tokens;                           // every id that ends a turn (glm5next: 3 of them)

    // ---- glm5next (GLM-5.3-Flash): hybrid KDA linear attention / MLA + DSA sparse attention, mHC
    // hyper-connections, sigmoid-routed experts with a bias, leading dense layers. The Qwen fields above double
    // for what the shared code needs: ssm_conv / ssm_state / ssm_vheads / ssm_groups describe the KDA conv and
    // state (the conv channels are q | k | v heads, as DeltaNet's), n_head_kv = 1 and head_dim = kv_lora the latent
    // cache row, idx_* / compress_ratio the DSA indexer (ratio = kpool).
    std::vector<int> layer_type;   // per layer: 0 KDA, 1 MLA
    std::vector<int> mlp_type;     // per layer: 0 dense MLP, 1 MoE
    int n_ff = 0;                  // the dense MLP width
    float hc_eps = 1e-6f;
    int hc_sinkhorn = 20;
    float kda_lower = -5.f;        // the forget gate's lower bound (g = lower * sigmoid(...))
    int q_lora = 0, kv_lora = 0, v_dim = 0;   // MLA: query rank, latent width, value head dim (head_dim: K nope dim)
    int qk_dim = 0;                // qk head dim (nope; rope 0)
    int idx_kpool = 0;             // DSA: tokens per indexer pool
    bool idx_tail = true;          // the incomplete tail pool is always selected
    float swiglu_limit = 0.f;      // clamp on the gate (max) and up (abs) before silu * up (0: none)
    bool router_sigmoid = false;   // sigmoid scores + bias for the choice (noaux_tc), weights normalized and scaled
    bool router_norm = true;

    bool is_glm() const { return arch == Arch::GLM5NEXT; }
    int ple_heads() const { return (ple_ngram - 1) * ple_heads_per_ngram; }
    bool is_attn(int il) const { return is_glm() ? layer_type[il] == 1 : (il + 1) % full_attn_interval == 0; }
    bool is_moe(int il) const { return !is_glm() || mlp_type[il] == 1; }
    int conv_channels() const { return 2 * ssm_groups * ssm_state + ssm_vheads * ssm_state; }
    int hc_dim() const { return hc * n_embd; }
    int kda_qkv() const { return ssm_vheads * ssm_state; }       // q (or k or v) width = heads x head dim
    int idx_raw_dim() const { return is_glm() ? 2 * idx_dim : idx_dim; }   // the raw indexer row: key | pool gate
    int mla_q_dim() const { return n_head * qk_dim; }
};

Config load_config(const GgufModel & g);

struct HcWeights {
    QMat norm, down, up, inject;  // norm: f32 [hc*n_embd]; inject absent for the head mixer
    QMat down_inject;             // down | inject stacked (the decode path's one launch); down / inject are its row views
};

struct LayerWeights {
    HcWeights hc_attn, hc_ffn;
    bool attn = false;
    // full attention
    QMat wq, wk, wv, wo, q_norm, k_norm;
    QMat idx_q, idx_k, idx_q_norm, idx_k_norm;
    // gated delta net
    QMat wqkv, wgate, ssm_beta, ssm_alpha, ssm_out, ssm_conv1d, ssm_dt, ssm_a, ssm_norm;
    // PLE
    QMat ple_key, ple_value, ple_norm_key, ple_norm_query, ple_norm_conv, ple_conv1d;
    // MoE
    QMat router, shexp_gate_inp, sh_gate, sh_up, sh_down;
    // Stacked pairs for the decode path (one launch, a split output); the parts above are row views into them
    // when the pair exists (same format and input width): beta | alpha, indexer k | q, router | shared gate, shared
    // gate | up, k | v
    QMat ssm_ba, idx_kq, router_sg, sh_gu, wkv;
    // routed experts: format of each matrix (the bytes live in the expert store)
    int gate_type = 0, up_type = 0, down_type = 0;
    size_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;   // per expert
    const uint8_t * gate_src = nullptr, * up_src = nullptr, * down_src = nullptr;  // mmap, expert 0
    // NVFP4 experts: fp32 global scale per expert matrix (`blk.N.ffn_*_exps.scale`, written by llama.cpp's
    // converter; moe.h ExpScales). Host copies for the CPU expert path, device arrays for the kernels.
    std::vector<float> exp_sgate, exp_sup, exp_sdown;
    ExpScales exp_scales;
    bool moe = true;               // false: a dense MLP layer (glm5next's leading layers)

    // ---- glm5next
    QMat attn_norm, ffn_norm;      // F32 [E]
    QMat hc_attn_fn, hc_ffn_fn;    // mHC weight projections [24, hc * E] (F16); their bases [24] and scales [3] F32
    QMat hc_attn_base, hc_attn_scale, hc_ffn_base, hc_ffn_scale;
    // KDA: q | k | v (Q8_0 [3 qkv, E]), their conv taps F32 [3 qkv, K], f_a | g_a | b (F16 [2 S + H, E]), f_b / g_b
    // (F16 [qkv, S]), dt_bias F32 [qkv], A_log F32 [H], the gated output norm F32 [S]; ssm_out = o_proj (Q8_0)
    QMat kda_fgb_a, kda_f_b, kda_g_b;
    // MLA: q_a | kv_a (Q8_0 [q_lora + kv_lora, E]), their norms, q_b | the indexer's wq_b (Q8_0 [H qk + ih id,
    // q_lora]), kv_b F16 [H (qk + v), kv_lora] (row-major: the absorbed products read it by head), o_proj (wo);
    // the indexer's wk | weights_proj | pool gate (Q8_0 [id + ih + id, E]), its LayerNorm (F32 [id] x 2), ape F32
    // [kpool, id]
    QMat attn_qkv_a, attn_q_a_norm, attn_kv_a_norm, attn_q_b_idx, attn_kv_b, idx_x, idx_k_norm_w, idx_k_norm_b, idx_ape;
    QMat router_bias;              // F32 [n_expert]: e_score_correction_bias
    QMat ffn_gate_up, ffn_down;    // the dense MLP layers (Q8_0)
    bool experts_present = true;   // false: the GGUF holds no (converted) experts for this layer (a partial file)
};

struct Model {
    Config cfg;
    GgufModel gguf;
    std::vector<LayerWeights> layers;
    HcWeights hc_head;
    QMat output, tok_embd;
    GgufModel ple_gguf;                                    // --ple-gguf: holds the PLE table when the model has none
    const TensorRef * ple_table = nullptr;                 // host (mmap), [ple_dim, rows]
    float ple_scale = 1.f;                                 // its global scale (NVFP4: per_layer_token_embd.scale)
    size_t vram_dense_bytes = 0;

    // dense weights to VRAM (the current device); ple_path supplies per_layer_token_embd.weight when the model file
    // has none. A pipeline stage (several GPUs) uploads only layers [l0, l1), the embeddings with `embd` and the
    // head with `head`; the routed experts' placement is described for every layer either way.
    void load(const std::string & path, bool verbose = true, const std::string & ple_path = "", int l0 = 0, int l1 = -1,
              bool embd = true, bool head = true);
    int l0 = 0, l1 = 0;   // the layers whose dense weights are on this device
    ~Model();

    bool repack_ = true;  // R layouts for dense quant matrices
    bool hc_q8_ = true;   // BF16 hyper-connection / PLE projections as Q8_0

private:
    void load_glm(bool verbose, bool embd, bool head);   // model_glm.cpp: the glm5next weights
    friend Config load_config(const GgufModel & g);
    friend Config load_config_glm(const GgufModel & g);
    std::vector<void *> device_allocs_;
    std::vector<void *> host_allocs_;   // page-locked host copies read by the GPU over PCIe (the token table)
    QMat upload(const std::string & name, bool required = true);
    QMat upload_data(const std::string & name, const uint8_t * data, int type, int64_t rows, int64_t cols,
                     size_t row_bytes);
    // two matrices of the same format and width stacked (a | b): one upload; a and b become row views of it
    bool stack(const std::string & na, const std::string & nb, QMat & a, QMat & b, QMat & ab);
    void expert_src(int il);
};

}  // namespace bnk
