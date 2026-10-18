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
    GgufModel gguf;
    std::vector<DenseLayer> layers;   // every layer; weights only for [l0, l1)
    QMat tok_embd;                    // page-locked host memory, gathered by row over PCIe
    QMat output_norm, output;
    DenseMtpWeights mtp;
    int l0 = 0, l1 = 0;
    size_t vram_bytes = 0;

    // Uploads layers [l0, l1) (l1 < 0: all), the token table with `embd`, the head with `head`, the MTP block
    // with `mtp` (the block after the trunk, when the file has one).
    void load(const std::string & path, bool verbose, int l0 = 0, int l1 = -1, bool embd = true, bool head = true,
              bool mtp = false);
    ~DenseModel();

private:
    std::vector<void *> device_allocs_;
    std::vector<void *> host_allocs_;
    QMat upload(const std::string & name, bool required = true);
    QMat upload_tensor(const TensorRef & t, bool repack);
    // two matrices of the same format and width stacked (a | b): one upload; a and b become row views of it
    bool stack(const std::string & na, const std::string & nb, QMat & a, QMat & b, QMat & ab);
    void load_layer(DenseLayer & L, const std::string & prefix, bool attn);
};

Config load_dense_config(const GgufModel & g);

}  // namespace bnk
