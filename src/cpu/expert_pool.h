// Routed experts in host RAM and the CPU worker pool that computes them.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <shared_mutex>
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
    // cap: bytes of arena (0: every expert); rank: which experts take it, hottest first (empty: in order);
    // skip(il, e): experts that do not need the arena (resident in a GPU cache already). Until build() runs,
    // every expert is served from the model file's mapping (prepare() alone), so the GPU caches can fill first.
    void prepare(const Model & m);
    void build(const Model & m, int threads, bool verbose = true, size_t cap = 0, const Ranking & rank = {},
               const std::function<bool(int, int)> & skip = {});
    struct Parts { const uint8_t * gate, * up, * down; bool pinned; };
    Parts parts(int il, int e) const;
    // reads an expert's blob (gate | up | down) from the model file into dst (one outside the arena: a page fault
    // per 4 KiB otherwise)
    void read_blob(int il, int e, uint8_t * dst) const;
    void read_blob_part(int il, int e, uint8_t * dst, int part, int nparts) const;
    // the expert's blob (gate | up | down, blob_bytes(il)) to device memory on `s`
    void copy_to_device(int il, int e, uint8_t * dst, cudaStream_t s);
    // the blobs of `experts` of layer il to dst + i * stride on `s`: arena experts by DMA; the others read from the
    // model file by a few threads into a pinned stage, wave by wave, and DMA'd from there (the host waits for each
    // wave's copies before the stage is reused)
    void copy_batch_to_device(int il, const std::vector<int> & experts, uint8_t * dst, size_t stride, cudaStream_t s);
    int disk_threads = 16;   // the NVMe gives 2 GB/s to one reader, 5.3 GB/s to sixteen
    bool in_arena(int il, int e) const { return arena_slot_[(size_t) il * n_expert_ + e] >= 0; }
    // The arena as the second level of the GPU caches: an expert a GPU evicts is written into it (read from the
    // model file on a background thread, into the arena's tail ring: its coldest tenth, cycled), so later misses
    // on it cost the pool 0.3 ms rather than a file read. A no-op when it is there already, or when the layer's
    // blobs are another size than the victim's. Readers of arena pointers hold arena_mu_ shared meanwhile.
    void put(int il, int e);
    void drain_puts();
    void set_puts(bool on) { puts_on_ = on; }   // off while the GPUs read the arena directly (their table is static)
    int64_t puts() const { return puts_; }
    std::shared_mutex arena_mu_;
    // the expert's blob as the GPUs see it (the arena is mapped for them), 0 when it is not in the arena
    unsigned long long host_dev_ptr(int il, int e) const {
        const int32_t sl = arena_slot_[(size_t) il * n_expert_ + e];
        return sl < 0 || !mapped_ ? 0ull : (unsigned long long) (dev_base_ + slot_off_[sl]);
    }
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
    bool pinned_ = false, mapped_ = false;
    uint8_t * dev_base_ = nullptr;      // the arena's device address (UVA: the host one)
    std::vector<size_t> blob_;
    std::vector<int32_t> arena_slot_;   // [n_layer * n_expert] -> arena slot, -1 = the file mapping
    std::vector<size_t> slot_off_;      // arena byte offset of each slot
    std::vector<std::pair<int, int>> slot_expert_;   // slot -> (layer, expert)
    // puts: the queue, its thread, the tail ring it writes into
    std::deque<std::pair<int, int>> put_q_;
    std::mutex put_mu_;
    std::condition_variable put_cv_;
    std::thread put_thread_;
    bool put_stop_ = false, puts_on_ = true;
    int ring_ = 0, ring_cursor_ = 0;
    int64_t puts_ = 0;
    std::vector<uint8_t> put_scratch_;
    void put_loop();
    // the last DMA from the arena on each device: a put waits for them before a slot changes hands
    std::vector<cudaEvent_t> dma_ev_;
    uint8_t * dstage_ = nullptr;        // pinned stage for the file-read blobs on their way to a GPU
    int dstage_slots_ = 0;
    std::mutex dstage_mu_;
    void dstage_ensure();
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
    int64_t disk_reads_ = 0;   // experts read from the model file (outside the arena)
    int64_t disk_reads() const { return disk_reads_; }

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
    std::vector<ExpertStore::Parts> gparts_; // each group's matrices (the arena, or the scratch below)
    std::vector<uint8_t> dscratch_;          // file-read blobs for this call
    int chunks_per_thread_ = 1;
    int rows_done_n_ = 0;
    std::mutex mu_;
};

}  // namespace bnk
