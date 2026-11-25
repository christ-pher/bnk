// Inkling 2 (GGUF architecture "inkling2_loop": Thinking Machines' Inkling-2-Mini): hyper-parameters and every
// weight on one GPU, as the file stores them (BF16 stays BF16, the experts' NVFP4 blocks stay NVFP4 with their
// per-expert fp32 scales). docs/INKLING-PLAN.md describes the model; the reference is llama.cpp's
// src/models/inkling2.cpp and inkling2-loop.cpp.
#pragma once

#include <string>
#include <vector>

#include "core/gguf_model.h"
#include "core/model.h"
#include "kernels/gemv.h"

namespace bnk {

struct InkConfig {
    int n_layer = 0;            // physical layers (24)
    int E = 0;                  // n_embd (2304)
    int n_head = 0, n_head_kv = 0, head_dim = 0;
    int n_expert = 0, n_used = 0, n_ff = 0;   // routed experts, used per token, expert FFN width (gate | up = 2 n_ff)
    int n_vocab = 0, n_vocab_pad = 0;         // the tokenizer's and the tables' (padded) rows
    float rms_eps = 1e-6f;
    float qk_clip = 0, rel_clip = 0;          // L2 clip thresholds: q and k (22.6), the relative-position queries (8)
    int n_rel = 0, rel_extent = 0;            // relative-position channels per head (16), the full layers' extent (1024)
    int n_swa = 0;                            // window of the sliding layers (1024)
    std::vector<bool> swa;                    // per physical layer
    int n_conv = 4;                           // short convolution taps
    int res_block = 6;                        // block attention residuals: a new block every res_block layers
    float log_floor = 8192, log_alpha = 0.1f; // the full layers' log-scaled positions
    float situ_beta = 4, situ_lin = 25;       // SITU activation
    float w_scale = 1;                        // routing weights' scale (8)
    float softcap = 0;                        // final logit soft cap (30)
    int loop_start = 0, loop_end = 0, loop_count = 1, gate_rank = 0;   // core [start, end) runs loop_count times
    int eos = 0, pad = 0;
    // tensor parallel: this rank's share (n_head, n_head_kv and n_ff above are the local ones)
    int tp_n = 1, tp_rank = 0;
    int vocab0 = 0, vocab_n = 0;              // the head's rows on this rank

    int n_core() const { return loop_end - loop_start; }
    int n_visits() const { return n_layer + (loop_count - 1) * n_core(); }   // cache slots (48)
    int kv_dim() const { return n_head_kv * head_dim; }
    int q_dim() const { return n_head * head_dim; }
    int extent(int il) const { return swa[il] ? n_swa : rel_extent; }
};

struct InkLayer {
    bool swa = false;
    // attention (BF16 matrices): q, k, v, out, the output gate, the relative-position queries, the sink logits;
    // full layers: the input-dependent conv taps of k and v
    QMat wq, wk, wv, wo, wgate, wrel, wsink, k_conv_proj, v_conv_proj;
    // f32 vectors (device)
    const float * attn_norm = nullptr;                                     // window layers
    const float * q_norm = nullptr, * k_norm = nullptr, * v_norm = nullptr;  // full layers: per input
    const float * q_clip = nullptr;    // [n_head][head_dim]
    const float * rel_clip = nullptr;  // [n_head][n_rel]
    const float * rel_bias = nullptr;  // [extent][n_rel]
    const float * k_conv = nullptr, * v_conv = nullptr;     // [kv_dim][n_conv]
    const float * attn_post_conv = nullptr, * ffn_post_conv = nullptr;   // [E][n_conv]
    const float * attn_res_query = nullptr;   // [nq][E] (layers > 0: 1 query, full layers 3), nullptr for layer 0
    int attn_res_nq = 0;
    const float * ffn_res_query = nullptr;    // [E]
    const float * ffn_norm = nullptr;
    // MoE: router (F32 [n_expert][E]), selection bias, gate | up and down of every expert (one matrix each, expert
    // e's rows from e * rows on), NVFP4 global scales per expert (1 for float experts)
    QMat router;
    const float * exp_bias = nullptr;
    QMat gate_up, down;
    std::vector<float> gu_scale, down_scale;
    const float * d_gu_scale = nullptr, * d_down_scale = nullptr;   // device copies (null: float experts)
};

// a core layer's loop controls: the four message queries (q, k, v, ffn) and per control down / gain / forget
struct InkLoop {
    const float * msg_query = nullptr;   // [4][E]
    QMat down[4], gain[4], forget[4];
};

struct InkModel {
    InkConfig cfg;
    Config base;          // the shared config the generator and server read (vocabulary, eos, dims)
    GgufModel gguf;
    std::vector<InkLayer> layers;
    std::vector<InkLoop> loops;          // [n_core]
    QMat tok_embd, output;               // BF16 [n_vocab_pad][E]
    const float * tok_norm = nullptr, * output_norm = nullptr, * output_res_query = nullptr;
    size_t vram_bytes = 0;

    // tp_n > 1: rank tp_rank's share of the heads, of every expert's FFN width and of the vocabulary
    void load(const std::string & path, bool verbose, int tp_n = 1, int tp_rank = 0);
    ~InkModel();

    // expert e's rows of a stacked expert matrix (n experts)
    static QMat expert(const QMat & m, int e, int n) {
        const int64_t rows = m.rows / n;
        return qmat_rows(m, (int64_t) e * rows, rows);
    }

private:
    std::vector<void *> allocs_;
    using Ranges = std::vector<std::pair<int64_t, int64_t>>;
    QMat upload_bytes(const std::string & name, const std::vector<uint8_t> & host, int64_t rows, int64_t cols, int type);
    // the rows [r0, r1) of each range, or the columns [c0, c1) of every row (whole quant blocks), as one matrix
    QMat upload_rows(const std::string & name, const Ranges & rows);
    QMat upload_cols(const std::string & name, const Ranges & cols);
    const float * upload_f32_range(const std::string & name, int64_t off, int64_t n);
    QMat upload(const std::string & name, bool required = true);
    const float * upload_vec(const std::vector<float> & v);
    const float * upload_f32(const std::string & name, size_t expect, bool required = true);
};

}  // namespace bnk
