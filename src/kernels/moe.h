// Hybrid MoE step: experts resident in the VRAM cache run on the GPU; the rest go through a host-mapped
// mailbox to the CPU pool, concurrently.
#pragma once

#include <cstddef>
#include <cstdint>
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
};

struct MoeLayerDesc {
    const int32_t * slot_of;  // [n_expert] device, -1 = not resident
    const uint8_t * base;     // the layer's slot region (device)
    size_t blob;              // bytes per slot
    size_t gate_bytes, up_bytes;
    int gate_type, down_type;
    size_t grow, drow;        // row bytes of gate/up rows and of down rows
};

struct MoeScratch {
    HitList * hits = nullptr;  // device
    float * gu = nullptr;      // [kMaxRouted][W][2F]
    int8_t * hq = nullptr;     // [kMaxRouted][W][F]
    float * hd = nullptr;      // [kMaxRouted][W][F/32]
    float * part = nullptr;    // [kMaxRouted][W][E]
    uint32_t * counts = nullptr;  // [n_layer][n_expert] routing frequency (device)
};

// Builds the hit list and the CPU request (mailbox `msg`, host-mapped), then signals seq.
void moe_plan(const int32_t * ids, const float * w, int T, int k, const MoeLayerDesc & d, MoeScratch & s,
              MoeMsg * msg, const float * x, int E, const uint32_t * seq, uint32_t * counts_layer, cudaStream_t st);
// Hit experts on the GPU: writes s.part.
void moe_hits(const MoeLayerDesc & d, MoeScratch & s, const ActQ8 & xq, int T, int k, int E, int F, cudaStream_t st);
// Waits for the CPU's answer when the layer had misses.
void moe_wait(const MoeScratch & s, MoeMsg * msg, const uint32_t * seq, cudaStream_t st);
// out[t] = sum hits + shared * sgate[t] + cpu answer
void moe_reduce(const MoeScratch & s, MoeMsg * msg, const float * shared, const float * sgate, float * out, int T,
                int k, int E, cudaStream_t st);
void moe_debug_times(uint64_t (*out)[3], int n);
// seq += 1 (first kernel of a forward)
void bump_seq(uint32_t * seq, cudaStream_t st);

}  // namespace bnk
