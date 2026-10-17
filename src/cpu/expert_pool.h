// Routed experts in host RAM and the CPU worker pool that computes them.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "core/model.h"

namespace bnk {

using Ranking = std::vector<std::pair<int, int>>;  // (layer, expert), hottest first

// The routed experts in host RAM: each one a blob [gate rows | up rows | down rows] in a page-locked arena, so the
// GPU can DMA blobs into its cache and the CPU computes them in place. When the arena is capped (a model whose
// experts exceed the RAM: glm5next's 160 GiB), the hottest experts of the ranking get the arena and the rest stay
// in the model file's mapping (three separate matrices, read through the page cache and the disk).
class ExpertStore {
public:
    ~ExpertStore();
    // cap: bytes of arena (0: every expert); rank: which experts take it, hottest first (empty: in order)
    void build(const Model & m, int threads, bool verbose = true, size_t cap = 0, const Ranking & rank = {});
    struct Parts { const uint8_t * gate, * up, * down; bool pinned; };
    Parts parts(int il, int e) const;
    // the expert's blob (gate | up | down, blob_bytes(il)) to device memory on `s`
    void copy_to_device(int il, int e, uint8_t * dst, cudaStream_t s) const;
    bool in_arena(int il, int e) const { return arena_slot_[(size_t) il * n_expert_ + e] >= 0; }
    size_t blob_bytes(int il) const { return blob_[il]; }
    size_t max_blob_bytes() const { return max_blob_; }
    size_t total_bytes() const { return total_; }     // the arena
    size_t model_bytes() const { return model_; }     // every expert
    int arena_experts() const { return n_arena_; }
    bool pinned() const { return pinned_; }
    uint8_t * base() const { return base_; }

private:
    const Model * m_ = nullptr;
    uint8_t * base_ = nullptr;
    size_t total_ = 0, model_ = 0, max_blob_ = 0;
    int n_expert_ = 0, n_arena_ = 0;
    bool pinned_ = false;
    std::vector<size_t> blob_;
    std::vector<int32_t> arena_slot_;   // [n_layer * n_expert] -> arena slot, -1 = the file mapping
    std::vector<size_t> slot_off_;      // arena byte offset of each slot
};

// A fixed set of spinning workers. run() calls fn(part, nparts) once on every thread (the caller is
// part 0); each part takes a static share of the work, so dispatch needs no shared counter.
class SpinPool {
public:
    explicit SpinPool(int n_workers = 0, int first_cpu = -1);
    ~SpinPool();
    int size() const { return (int) workers_.size() + 1; }
    void run(const std::function<void(int, int)> & fn);

private:
    struct alignas(64) Flag { std::atomic<uint64_t> v{0}; };
    void loop(int id);
    std::vector<std::thread> workers_;
    alignas(64) std::atomic<uint64_t> gen_{0};
    std::unique_ptr<Flag[]> done_;
    std::atomic<bool> stop_{false};
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
    // out[t][E] = sum over tasks of w * expert(x[t]); `out` is zeroed first. x: [T][E] host. One caller at a
    // time (pipeline stages reading a prompt run on two threads).
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
    std::vector<uint8_t> hq_;      // per task: h quantized for the down product
    std::unique_ptr<std::atomic<int>[]> rows_done_;
    std::vector<std::vector<int>> groups_;   // tasks of the same expert
    int chunks_per_thread_ = 1;
    int rows_done_n_ = 0;
    std::mutex mu_;
};

}  // namespace bnk
