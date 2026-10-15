// qwen4exp model description: hyper-parameters from GGUF metadata and the placement of every weight.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "core/gguf_model.h"
#include "kernels/gemv.h"

namespace bnk {

struct Config {
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

    int ple_heads() const { return (ple_ngram - 1) * ple_heads_per_ngram; }
    bool is_attn(int il) const { return (il + 1) % full_attn_interval == 0; }
    int conv_channels() const { return 2 * ssm_groups * ssm_state + ssm_vheads * ssm_state; }
    int hc_dim() const { return hc * n_embd; }
};

Config load_config(const GgufModel & g);

struct HcWeights {
    QMat norm, down, up, inject;  // norm: f32 [hc*n_embd]; inject absent for the head mixer
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
    // routed experts: format of each matrix (the bytes live in the expert store)
    int gate_type = 0, up_type = 0, down_type = 0;
    size_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;   // per expert
    const uint8_t * gate_src = nullptr, * up_src = nullptr, * down_src = nullptr;  // mmap, expert 0
};

struct Model {
    Config cfg;
    GgufModel gguf;
    std::vector<LayerWeights> layers;
    HcWeights hc_head;
    QMat output, tok_embd;
    GgufModel ple_gguf;                                    // --ple-gguf: holds the PLE table when the model has none
    const TensorRef * ple_table = nullptr;                 // host (mmap), [ple_dim, rows]
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
    // Tensor parallelism: this device holds slice tp_rank of tp_n of every layer. Row-split projections keep rows
    // [rank * rows / n, +rows / n) (per head / head group / shared-expert column), column-split ones (the
    // projections back to the model width: attn_output, ssm_out, ffn_down_shexp, output's rows) the matching
    // input columns, so every device's product is a partial sum of the full one. The hyper-connections, router,
    // norms, embeddings and PLE are whole on every device. tp_slices() below lists each matrix's rule.
    int tp_n = 1, tp_rank = 0;
    // local shapes under the split (equal to the model's when tp_n == 1)
    int n_head_l() const { return cfg.n_head / tp_n; }
    int n_head_kv_l() const { return cfg.n_head_kv / tp_n; }
    int ssm_vheads_l() const { return cfg.ssm_vheads / tp_n; }
    int ssm_groups_l() const { return cfg.ssm_groups / tp_n; }
    int conv_channels_l() const { return 2 * ssm_groups_l() * cfg.ssm_state + ssm_vheads_l() * cfg.ssm_state; }
    int n_ff_shexp_l() const { return cfg.n_ff_shexp / tp_n; }
    int n_vocab_l() const { return (cfg.n_vocab + tp_n - 1) / tp_n; }   // the output head's rows on this device
    int vocab0() const { return tp_rank * n_vocab_l(); }
    // this device's experts: e % tp_n == tp_rank
    bool expert_local(int e) const { return tp_n == 1 || e % tp_n == tp_rank; }

private:
    std::vector<void *> device_allocs_;
    struct Slice {   // row ranges and column ranges of a tensor (empty: every row / column), in order
        std::vector<std::pair<int64_t, int64_t>> rows, cols;
    };
    QMat upload(const std::string & name, bool required = true, const Slice * slice = nullptr);
    QMat upload_data(const std::string & name, const uint8_t * data, int type, int64_t rows, int64_t cols,
                     size_t row_bytes);
    Slice slice_rows(int64_t rows_per_rank) const;   // rows [rank * r, +r)
    Slice slice_cols(int64_t cols_per_rank) const;   // columns [rank * c, +c)
    // the DeltaNet value heads of this rank's key groups (head h belongs to group h % groups): heads
    // {j * groups + rank * gl + i}, as ranges of `width` rows (or columns) each
    std::vector<std::pair<int64_t, int64_t>> gdn_head_ranges(int64_t width) const;
    void expert_src(int il);
};

}  // namespace bnk
