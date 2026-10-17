// Hybrid MoE step: experts resident in the VRAM cache run on the GPU; the rest go through a host-mapped
// mailbox to the CPU pool, concurrently.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/gemv.h"

namespace bnk {

constexpr int kMaxRouted = kMaxWindow * 16;  // T * k entries per layer

// Per-layer mailbox in host-mapped memory. The GPU fills the request and bumps seq_req; the CPU answers
// in `out` and bumps seq_done. x/out follow the header: x [W][E], out [W][E] floats.
struct alignas(64) MoeMsg {
    volatile uint32_t seq_req;
    uint32_t pad0[15];
    volatile uint32_t seq_done;
    uint32_t pad1[15];
    int32_t n_miss;
    int32_t T;
    int32_t miss_t[kMaxRouted];
    int32_t miss_e[kMaxRouted];
    float miss_w[kMaxRouted];
    static size_t bytes(int E) { return (sizeof(MoeMsg) + 2 * (size_t) kMaxWindow * E * 4 + 63) / 64 * 64; }
    __host__ __device__ float * x() { return (float *) (this + 1); }
    __host__ __device__ float * out(int E) { return x() + (size_t) kMaxWindow * E; }
};

// The hit list the plan kernel builds on the device.
struct HitEntry {
    int32_t expert, slot;
    uint32_t mask;            // tokens of the window routed to this expert
    float w[kMaxWindow];      // their routing weights
};
struct HitList {
    int32_t n;
    int32_t n_miss;           // copy of the mailbox's, so device kernels need not read host memory
    int32_t pad[2];
    HitEntry e[kMaxRouted];
    int32_t pair_ent[kMaxRouted];   // routed pair (t * k + slot) -> its hit entry, -1 for a miss (moe_reduce adds a
                                    // token's experts in its own routing order, whatever else the window holds)
};

// NVFP4 experts: the fp32 global scale of each expert's gate / up / down matrix (device [n_expert]; all null for
// formats without one). ggml's NVFP4 blocks hold the weights without it, ~16,000x their true size, so the kernels
// apply it where an expert's output is formed: gate before the silu, up with it, down on the int8 h (its fp32
// per-32 scale carries the magnitude) or on the tensor-core down product before its fp16 store.
struct ExpScales {
    const float * gate = nullptr, * up = nullptr, * down = nullptr;
    float limit = 0.f;   // > 0: h = silu(min(gate, limit)) * clamp(up, -limit, limit) (glm5next's swiglu_limit)
};

struct MoeLayerDesc {
    const int32_t * slot_of;  // [n_expert] device, -1 = not resident
    const uint8_t * base;     // the layer's slot region (device)
    size_t blob;              // bytes per slot
    size_t gate_bytes, up_bytes;
    int gate_type, down_type;
    size_t grow, drow;        // row bytes of gate/up rows and of down rows
    int rlay = 0;             // slots hold R-layout rows (rfmt.h) rather than ggml blocks
    ExpScales scales;         // NVFP4 experts' per-matrix global scales (null when the format has none)
    uint32_t go[5] = {0, 0, 0, 0, 0}, dof[5] = {0, 0, 0, 0, 0};  // R field offsets of gate/up and down rows
};

// Repacks n ggml expert blobs (src, stride src_stride) into R-layout slots (dst pointers device array).
void moe_repack_blobs(const uint8_t * src, size_t src_stride, uint8_t * const * dst, int n, int F, int E,
                      int gate_type, size_t g_rb, int down_type, size_t d_rb, cudaStream_t st);

struct MoeScratch {
    HitList * hits = nullptr;  // device
    float * gu = nullptr;      // [kMaxRouted][W][2F]
    int8_t * hq = nullptr;     // [kMaxRouted][W][F]
    float * hd = nullptr;      // [kMaxRouted][W][F/32]
    float * part = nullptr;    // [kMaxRouted][W][E]
    uint32_t * counts = nullptr;  // [n_layer][n_expert] routing frequency (device)
};

// route_topk (logits -> ids / w) and moe_plan in one launch
void moe_route_plan(const float * logits, int n_exp, float w_scale, int32_t * ids, float * w, int T, int k,
                    const MoeLayerDesc & d, MoeScratch & s, MoeMsg * msg, const float * x, int E, const uint32_t * seq,
                    uint32_t * counts_layer, cudaStream_t st);
// Builds the hit list and the CPU request (mailbox `msg`, host-mapped), then signals seq.
void moe_plan(const int32_t * ids, const float * w, int T, int k, const MoeLayerDesc & d, MoeScratch & s,
              MoeMsg * msg, const float * x, int E, const uint32_t * seq, uint32_t * counts_layer, cudaStream_t st);
// Hit experts on the GPU: writes s.part.
void moe_hits(const MoeLayerDesc & d, MoeScratch & s, const ActQ8 & xq, int T, int k, int E, int F, cudaStream_t st);
// Waits for the CPU's answer when the layer had misses.
void moe_wait(const MoeScratch & s, MoeMsg * msg, const uint32_t * seq, cudaStream_t st);
// out[t] = sum hits + shared * sgate[t] + cpu answer
// wait_seq: moe_wait's work (and its layer counter) in the same launch
// raw_gate: sgate holds the gate's logit (the sigmoid is applied here, one launch fewer)
void moe_reduce(const MoeScratch & s, MoeMsg * msg, const float * shared, const float * sgate, float * out, int T,
                int k, int E, cudaStream_t st, const uint32_t * wait_seq = nullptr, bool raw_gate = false);
void moe_debug_times(uint64_t (*out)[3], int n);
// restarts the per-layer timestamps at layer 0 (on `st`, before a forward: the drafter's MoE calls advance them)
void moe_debug_reset(cudaStream_t st);
// ---- prompt path: experts with few tokens, straight on their quantized weights
struct PfItem {
    const uint8_t * blob;     // the expert's blob on the device (cache slot or stage)
    int32_t n;                // tokens in this item (<= 8)
    int32_t expert;
    int32_t tok[kMaxWindow];  // rows of the chunk's quantized input
    int32_t pair[kMaxWindow]; // output rows (sorted pair positions)
};
// gu[pair][2F] = blob gate/up rows . x[tok]; h quantized per pair; D[pair][E] (fp16) = down . h
void moe_list(const PfItem * items, int n_items, const MoeLayerDesc & d, const ActQ8 & xq, int E, int F,
              float * gu, int8_t * hq, float * hd, half * D, cudaStream_t st);

// ---- prompt path, tensor cores (moe_tc.cu): an expert's tokens against its quantized rows, the weights unpacked
// once per 64-token tile into fp16 fragments. Pairs (token, slot) are sorted by expert: an item is one expert's run
// of pairs [p0, p0 + n), a block one 64-token tile of it (t0 = 0, 64, ...). The gate/up kernel reads pair p's
// activation at x[perm[p]] (fp16) and writes h = silu(gate) * up to h[p][F]; the down kernel reads h[p] and writes
// D[p][E] (fp16, as the grouped GEMM path did).
struct TcItem {
    const uint8_t * blob;     // the expert's blob on the device (cache slot or stage)
    int32_t p0, n;            // its pairs
    int32_t expert;
};
struct TcBlock {
    int32_t item, t0;
};
constexpr int kTcTile = 64;   // tokens per block
void moe_tc_gu(const TcItem * items, const TcBlock * blocks, int nblocks, const MoeLayerDesc & d, const half * x,
               int64_t ldx, const int32_t * perm, half * h, int E, int F, cudaStream_t st);
void moe_tc_down(const TcItem * items, const TcBlock * blocks, int nblocks, const MoeLayerDesc & d, const half * h,
                 half * D, int E, int F, cudaStream_t st);

// seq += 1 (first kernel of a forward)
void bump_seq(uint32_t * seq, cudaStream_t st);

}  // namespace bnk
