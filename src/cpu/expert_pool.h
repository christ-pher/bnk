// Routed experts in host RAM and the CPU worker pool that computes them.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#include "core/model.h"

namespace bnk {

// Every routed expert of every layer as one contiguous blob [gate rows | up rows | down rows] in one
// page-locked arena, so the GPU can DMA blobs into its cache and the CPU computes them in place.
class ExpertStore {
public:
    ~ExpertStore();
    void build(const Model & m, int threads, bool verbose = true);
    const uint8_t * blob(int il, int e) const { return base_ + layer_off_[il] + (size_t) e * blob_[il]; }
    size_t blob_bytes(int il) const { return blob_[il]; }
    size_t max_blob_bytes() const { return max_blob_; }
    size_t total_bytes() const { return total_; }
    bool pinned() const { return pinned_; }
    uint8_t * base() const { return base_; }
    size_t layer_offset(int il) const { return layer_off_[il]; }

private:
    uint8_t * base_ = nullptr;
    size_t total_ = 0, max_blob_ = 0;
    bool pinned_ = false;
    std::vector<size_t> layer_off_, blob_;
};

// A fixed set of spinning workers that run index-ranged jobs with low dispatch latency.
class SpinPool {
public:
    explicit SpinPool(int n_workers = 0, int first_cpu = -1);
    ~SpinPool();
    int size() const { return (int) workers_.size() + 1; }  // the caller participates
    // Runs fn(i, worker) for i in [0, n); returns when all are done.
    void run(int n, const std::function<void(int, int)> & fn);

private:
    void loop(int id);
    std::vector<std::thread> workers_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> next_{0}, done_{0};
    std::atomic<bool> stop_{false};
    int n_ = 0;
    const std::function<void(int, int)> * fn_ = nullptr;
};

struct ExpertTask {
    int t;       // token in window
    int slot;    // which of the k selections
    int expert;
    float w;     // routing weight
};

class CpuExpertPool {
public:
    void init(const Model & m, const ExpertStore & st, int n_threads);
    // out[t][E] = sum over tasks of w * expert(x[t]); `out` is zeroed first. x: [T][E] host.
    void run(int il, int T, const float * x, const std::vector<ExpertTask> & tasks, float * out);
    int threads() const { return pool_ ? pool_->size() : 0; }
    double last_ms = 0;

private:
    const Model * m_ = nullptr;
    const ExpertStore * st_ = nullptr;
    std::unique_ptr<SpinPool> pool_;
    // scratch
    std::vector<uint8_t> xq_;      // per token: x quantized for the gate/up vec_dot type
    std::vector<float> gu_;        // per task: [2*n_ff]
    std::vector<uint8_t> hq_;      // per task: h quantized for the down vec_dot type
    std::vector<float> dout_;      // per task: [E]
};

}  // namespace bnk
