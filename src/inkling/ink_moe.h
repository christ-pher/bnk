// Inkling 2's MoE for decode / verify windows, entirely on the device (no host round trip: graph-capturable).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace bnk {

constexpr int kInkWinMax = 16;                 // tokens of a window
constexpr int kInkMaxExperts = 256;
constexpr int kInkMaxPairs = kInkWinMax * 8;   // T * k

// the window's routed pairs grouped by expert (device)
struct InkMoePlan {
    int n_u, k;
    int expert[kInkMaxPairs];          // the unique experts, ascending
    int start[kInkMaxPairs + 1];       // their pairs' range in the grouped order
    int tok[kInkMaxPairs];             // grouped: the pair's token
    float wt[kInkMaxPairs];            // grouped: its routing weight
    int pair_slot[kInkMaxPairs];       // pair t * k + j -> its grouped slot
    int slot_e[kInkMaxPairs];          // grouped: the pair's expert
};

struct InkMoeArgs {
    const int32_t * ids = nullptr;   // [T][k] routed experts
    const float * w = nullptr;       // [T][k] routing weights
    // or the router's logits [T][n_exp] (routed here with the plan, one launch): sigmoid, the top k of
    // sigmoid + bias, the weights normalized and scaled
    const float * logits = nullptr, * bias = nullptr;
    float w_scale = 1;
    int T = 0, k = 0, n_exp = 0, E = 0, F = 0;
    const float * x = nullptr;       // [T][E] the FFN input (normed)
    const uint8_t * gate_up = nullptr; int gu_type = 0; size_t gu_row_bytes = 0; const float * gu_scale = nullptr;
    const uint8_t * down = nullptr; int down_type = 0; size_t down_row_bytes = 0; const float * down_scale = nullptr;
    float beta = 4, lb = 25;         // SITU
    InkMoePlan * plan = nullptr;     // scratch
    float * gu = nullptr;            // [T*k][2F] scratch
    int8_t * q = nullptr;            // [T*k][max(E, F)] int8 activations (NVFP4 experts: dp4a)
    float * qd = nullptr;            // [T*k][max(E, F) / 32] their scales
    float * h = nullptr;             // [T*k][F] scratch
    float * y = nullptr;             // [T*k][E] scratch
    float * out = nullptr;           // [T][E]
};

void ink_moe_window(const InkMoeArgs & a, cudaStream_t s);

}  // namespace bnk
