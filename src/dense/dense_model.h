// The dense Qwen3.5 family (GGUF architecture "qwen35": Qwen3.8-27B and the smaller ones): hyper-parameters and
// every weight of a pipeline stage in VRAM.
//
// A layer is a Gated DeltaNet block or (every full_attention_interval-th) a gated full-attention block, then a
// dense SwiGLU FFN, with plain residual adds and an RMSNorm before each half. The model's MTP block (the "nextn"
// layer after the trunk) is loaded too when asked: it drafts for speculative decoding.
#pragma once

#include <string>
#include <vector>

#include "core/gguf_model.h"
#include "core/model.h"
#include "kernels/gemv.h"

namespace bnk {

// A LoRA adapter at one input of a layer: y_i += B_i (A x) for the projections that share the input x (q | k | v,
// gate | up, or o / down alone). A is the projections' A matrices stacked (Q8_0, read with the activations the
// base GEMV already quantized), the B_i are F16 with alpha / r folded in. Under tensor parallelism the slices
// follow the base matrix: row-split projections keep the whole A and their rows of B, column-split ones (o, down)
// their columns of A and the whole B, so the product lands in the partial sum before the all-reduce.
struct DenseLora {
    QMat a;            // [n * r][cols] (a row view of `host` when stacked)
    // decode: the base matrix (or gate | up) with A's rows stacked under it (Q8_0 bases): one GEMV computes both,
    // rows from host_rows on going to the A product; the base QMats are row views of it. Invalid: A on its own.
    QMat host;
    int host_rows = 0;
    QMat b[3];         // [rows_i][r] (invalid: that projection has no adapter)
    int off[3] = {0, 0, 0};   // B_i reads columns [off_i, off_i + r) of the A product
    int width = 0;     // a.rows
    bool valid() const { return a.valid(); }
};

struct DenseLayer {
    bool attn = false;
    QMat attn_norm, post_norm;   // f32 [E]: before the attention / DeltaNet, before the FFN
    // full attention: q | gate per head, k | v stacked (one launch), out
    QMat wq, wk, wv, wkv, wo, q_norm, k_norm;
    // gated delta net: qkv | gate, beta | alpha stacked, out
    QMat wqkv, wgate, ssm_beta, ssm_alpha, ssm_ba, ssm_out, ssm_conv1d, ssm_dt, ssm_a, ssm_norm;
    // FFN: gate | up stacked (one launch, split output), down
    QMat gate, up, gu, down;
    // a LoRA adapter (--lora): at the attention's input (q | k | v), its output projection, the FFN's input
    // (gate | up) and down
    DenseLora lora_qkv, lora_o, lora_gu, lora_down;
};

// the MTP block: eh_proj(cat(enorm(embed(tok)), hnorm(h))) through a full-attention layer, then its own final norm
// and the main model's head
struct DenseMtpWeights {
    bool present = false;
    DenseLayer layer;
    QMat eh_proj, enorm, hnorm, head_norm;
};

struct DenseModel {
    Config cfg;
    // Tensor parallel (tp_n > 1): this rank's slice of every layer. lc is the config of the slice: its heads, KV
    // heads, DeltaNet key groups and value heads, FFN width (the kernels run on those). Attention heads, KV heads,
    // DeltaNet groups and the FFN are split evenly; a rank's value heads are those whose key group it holds (the
    // GGUF's tiled order keeps a rank's heads in runs of nk/tp_n), so the local head h belongs to local group
    // h % (nk/tp_n) as before.
    int tp_n = 1, tp_rank = 0;
    Config lc;
    GgufModel gguf;
    std::vector<DenseLayer> layers;   // every layer; weights only for [l0, l1)
    QMat tok_embd;                    // page-locked host memory, gathered by row over PCIe
    QMat output_norm, output;
    DenseMtpWeights mtp;
    int l0 = 0, l1 = 0;
    size_t vram_bytes = 0;

    // Uploads layers [l0, l1) (l1 < 0: all), the token table with `embd`, the head with `head`, the MTP block
    // with `mtp` (the block after the trunk, when the file has one); tp_n > 1: rank tp_rank's slice of each layer.
    void load(const std::string & path, bool verbose, int l0 = 0, int l1 = -1, bool embd = true, bool head = true,
              bool mtp = false, int tp_n = 1, int tp_rank = 0);
    // A LoRA adapter GGUF (llama.cpp's convert_lora_to_gguf.py: <tensor>.lora_a [r][in], <tensor>.lora_b [out][r],
    // adapter.lora.alpha) for layers [l0, l1), this rank's slices; scale multiplies alpha / r. Call after load.
    // Returns the widest A product (rows of the scratch the forward needs per token), 0 for none.
    int load_lora(const std::string & path, float scale, bool verbose);
    int lora_width = 0;
    ~DenseModel();
    using Ranges = std::vector<std::pair<int64_t, int64_t>>;

    // one tensor of `gguf` to VRAM, as the kernels read it (BF16 matrices as Q8_0, R layout); also the DFlash2
    // drafter's weights (DenseDFlash keeps a DenseModel for its own file)
    QMat upload(const std::string & name, bool required = true);
    // slices: the given row ranges [r0, r1) concatenated, or the given column ranges of every row (ranges in
    // whole quant blocks)
    QMat upload_rows(const std::string & name, const Ranges & rows);
    QMat upload_cols(const std::string & name, const Ranges & cols);

private:
    std::vector<void *> device_allocs_;
    std::vector<void *> host_allocs_;
    QMat upload_tensor(const TensorRef & t, bool repack);
    void release(const QMat & m);   // frees m's allocation when m is one (not a view into another)
    QMat upload_tensor_raw(const TensorRef & t, bool repack);
    float global_scale(const std::string & wname) const;
    QMat upload_rows2(const std::string & na, const std::string & nb, const Ranges & ra, const Ranges & rb, QMat & a, QMat & b);
    // two matrices of the same format and width stacked (a | b): one upload; a and b become row views of it
    bool stack(const std::string & na, const std::string & nb, QMat & a, QMat & b, QMat & ab);
    void load_layer(DenseLayer & L, const std::string & prefix, bool attn);
};

Config load_dense_config(const GgufModel & g);

}  // namespace bnk
