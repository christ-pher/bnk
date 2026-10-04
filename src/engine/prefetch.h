// Decode-time expert prefetch. While layer L runs, the GPU ranks layer L+1's non-resident experts by its router on
// L's MoE input and posts the best few to a host-mapped mailbox; a host thread copies their blobs over PCIe into
// prefetch slots (two sets, by layer parity) and marks them. Layer L+1's plan kernel takes a missed expert from
// its slot when the copy landed in time; otherwise the CPU computes it as before, so a late copy costs nothing.
#pragma once

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "core/model.h"
#include "cpu/expert_pool.h"
#include "kernels/moe.h"

namespace bnk {

class ExpertPrefetch {
public:
    ~ExpertPrefetch();
    // per_layer experts per layer (0 = off, at most kMaxPrefetch). Allocates slots, staging and the mailboxes.
    void init(const Model & m, const ExpertStore & st, int per_layer, bool verbose);
    bool on() const { return B_ > 0; }
    int per_layer() const { return B_; }
    PfMsg * msg(int il) const { return msgs_ + il; }
    PfLayer * table(int il) const { return on() ? tables_ + il : nullptr; }
    // A forward with this seq was launched: serve its layers' mailboxes.
    void begin(uint32_t seq) { target_.store(seq, std::memory_order_release); }
    int64_t issued() const { return issued_.load(std::memory_order_relaxed); }
    int64_t used();   // prefetched experts the plan kernels took, over all layers

private:
    void loop();
    void issue(int tl, const PfMsg & m, uint32_t seq);
    const Model * m_ = nullptr;
    const ExpertStore * st_ = nullptr;
    int B_ = 0, n_layer_ = 0;
    PfMsg * msgs_ = nullptr;          // host-mapped [n_layer]
    PfLayer * tables_ = nullptr;      // device [n_layer]
    uint8_t * slots_ = nullptr;       // device: [2][B] R-layout slots
    uint8_t * stage_ = nullptr;       // device: [B] ggml blobs on their way to a slot
    uint8_t ** dst_ = nullptr;        // device: [2][B] slot pointers for the repack kernel
    size_t slot_bytes_ = 0, stage_bytes_ = 0;
    cudaStream_t copy_ = nullptr;
    std::thread thread_;
    std::atomic<uint32_t> target_{0};
    std::atomic<bool> stop_{false};
    std::atomic<int64_t> issued_{0};
};

}  // namespace bnk
