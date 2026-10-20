// The vision encoder of the Qwen3.8 models (the mmproj file, projector qwen3vl_merger): a 27-layer ViT over
// 16x16 patches with 2D rotary positions and a learned position table resized to the image, then a merger that
// turns every 2x2 block of patches into one embedding of the language model's width. Images come in already
// resized (sides multiples of patch * merge); one encode() returns their embeddings in the model's token order
// (row-major over the merged grid).
//
// The weights stay in page-locked host memory as fp16 and stream to the GPU one layer ahead (about 0.9 GB per
// image over PCIe), so an idle encoder holds no VRAM; the activations come from the engine's VRAM budget for the
// length of the encode (borrowed from the expert cache, given back after).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "core/gguf_model.h"
#include "core/vmem.h"

namespace bnk {

struct VisionConfig {
    int n_embd = 0, n_ff = 0, n_layer = 0, n_head = 0, patch = 16, merge = 2, out_dim = 0, pos_side = 0;
    float eps = 1e-6f;
    float mean[3] = {0.5f, 0.5f, 0.5f}, std[3] = {0.5f, 0.5f, 0.5f};
    int head_dim() const { return n_embd / n_head; }
    int unit() const { return patch * merge; }   // image sides are multiples of this
};

class VisionEncoder {
public:
    ~VisionEncoder();
    // Opens the mmproj file and prepares the weights (host). `budget`: the engine's VRAM budget on `device`
    // (nullptr: plain allocations, for tools and tests).
    void load(const std::string & mmproj, int device, VramBudget * budget, bool verbose = true);
    bool loaded() const { return loaded_; }
    const VisionConfig & cfg() const { return cfg_; }
    // rgb: H x W x 3 bytes (row-major), H and W multiples of cfg().unit(). out: (H / unit) * (W / unit) rows of
    // cfg().out_dim floats (host).
    void encode(const uint8_t * rgb, int H, int W, float * out);
    double ms = 0;
    int64_t images = 0, tokens = 0;

private:
    struct Stage {               // one streamed unit of weights: fp16 matrices, then fp32 vectors, page-locked
        uint8_t * host = nullptr;
        size_t bytes = 0;
        std::vector<size_t> off;  // byte offset of each part
    };
    VisionConfig cfg_;
    GgufModel gguf_;
    bool loaded_ = false;
    int dev_ = 0;
    VramBudget * budget_ = nullptr;
    std::vector<Stage> stages_;  // 0: patch embedding, 1..n_layer: the blocks, n_layer + 1: post-norm and merger
    size_t stage_max_ = 0;
    float * pos_table_ = nullptr;   // device, [pos_side^2][n_embd] (fp32, kept: 10.6 MB)
    cudaStream_t st_ = nullptr, copy_ = nullptr;
    cudaEvent_t copied_[2] = {nullptr, nullptr}, used_[2] = {nullptr, nullptr};
    cublasHandle_t blas_ = nullptr;
    uint8_t * wbuf_[2] = {nullptr, nullptr};   // device stage buffers (in the scratch, like the activations)
    ElasticBuf arena_;                          // the scratch from the budget ...
    void * plain_ = nullptr;                    // ... or a plain allocation (no budget)
    size_t plain_bytes_ = 0;
    void * scratch(size_t bytes);
    void release_scratch();
    void add_stage(const std::vector<std::pair<const TensorRef *, bool>> & parts);
    void fetch(int s, int buf);
};

}  // namespace bnk
