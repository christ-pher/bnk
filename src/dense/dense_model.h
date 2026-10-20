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

struct DenseLayer {
    bool attn = false;
    QMat attn_norm, post_norm;   // f32 [E]: before the attention / DeltaNet, before the FFN
    // full attention: q | gate per head, k | v stacked (one launch), out
    QMat wq, wk, wv, wkv, wo, q_norm, k_norm;
    // gated delta net: qkv | gate, beta | alpha stacked, out
    QMat wqkv, wgate, ssm_beta, ssm_alpha, ssm_ba, ssm_out, ssm_conv1d, ssm_dt, ssm_a, ssm_norm;
    // FFN: gate | up stacked (one launch, split output), down
    QMat gate, up, gu, down;
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
    QMat upload_tensor_raw(const TensorRef & t, bool repack);
    float global_scale(const std::string & wname) const;
    QMat upload_rows2(const std::string & na, const std::string & nb, const Ranges & ra, const Ranges & rb, QMat & a, QMat & b);
    // two matrices of the same format and width stacked (a | b): one upload; a and b become row views of it
    bool stack(const std::string & na, const std::string & nb, QMat & a, QMat & b, QMat & ab);
    void load_layer(DenseLayer & L, const std::string & prefix, bool attn);
};

Config load_dense_config(const GgufModel & g);

}  // namespace bnk
