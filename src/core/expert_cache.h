// The VRAM tier of the routed experts: per-layer slot regions holding the most-used experts' blobs.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/model.h"
#include "core/vmem.h"
#include "cpu/expert_pool.h"
#include "kernels/moe.h"

namespace bnk {

using Ranking = std::vector<std::pair<int, int>>;  // (layer, expert), hottest first

// The expert ranking from bnk's routing-counts file (BNKC); empty when unreadable.
Ranking load_ranking(const std::string & path, int n_layer, int n_expert);
// Ranking from routing counts [n_layer][n_expert]: most routed first, ties interleaved across layers.
Ranking ranking_from_counts(const std::vector<uint32_t> & counts, int n_layer, int n_expert);
bool save_counts(const std::string & path, const std::vector<uint32_t> & counts, int n_layer, int n_expert);
std::vector<uint32_t> load_counts(const std::string & path, int n_layer, int n_expert);

class ExpertCache {
public:
    ~ExpertCache();
    // Fills up to `budget` bytes of VRAM following `rank` (completed with every remaining pair). Each layer's
    // slots live in an elastic region charged to `vb`, so the cache can later shrink() and grow().
    void init(const Model & m, const ExpertStore & st, size_t budget, Ranking rank, cudaStream_t s, bool verbose,
              VramBudget * vb);
    // The per-layer slot counts init() would choose, without allocating.
    static std::vector<int> plan(const Model & m, const ExpertStore & st, size_t budget, const Ranking & rank);
    // Gives back at least `need` bytes: drops the coldest slots (moving any hot expert out of a layer's tail
    // first) and unmaps the freed pages. Synchronizes `s`. Returns the bytes freed.
    size_t shrink(size_t need, cudaStream_t s);
    // Takes up to `avail` bytes: new slots for the hottest non-resident experts, filled now. Returns bytes used.
    size_t grow(size_t avail, cudaStream_t s);
    // experts of layer il that are resident (slots may also be empty after a grow, until filled)
    int resident(int il) const { return resident_l_[il]; }
    // experts of layer il without a mapped slot right now (also those whose swap into a slot is still in flight:
    // the old expert is unmapped at once, the new one only when its copy finishes)
    int unmapped(int il) const {
        int n = 0;
        for (int e = 0; e < n_expert_; ++e) n += slot_of_host_[(size_t) il * n_expert_ + e] < 0;
        return n;
    }
    // VRAM bytes of one expert of layer il (R-layout rows)
    static size_t slot_bytes(const Model & m, int il);
    MoeLayerDesc desc(int il) const;
    int resident() const { return resident_; }
    size_t bytes() const;
    int slots(int il) const { return (int) slot_expert_[il].size(); }

    // Adaptive tier. Call between forwards on the compute stream: finishes copies that are done, then
    // plans new swaps from routing counts (device [n_layer][n_expert], cumulative) and starts their
    // copies on a side stream. Returns the number of swaps started.
    int adapt(const uint32_t * counts_dev, cudaStream_t s, int max_swaps);
    int64_t swaps_done = 0;
    const std::vector<int> & slot_experts(int il) const { return slot_expert_[il]; }
    int slot_of(int il, int e) const { return slot_of_host_[(size_t) il * n_expert_ + e]; }

private:
    const Model * m_ = nullptr;
    const ExpertStore * st_ = nullptr;
    int n_layer_ = 0, n_expert_ = 0, resident_ = 0;
    std::vector<ElasticBuf> layer_buf_;   // each layer's slots, slot i at i * slot_bytes
    std::vector<uint8_t *> layer_base_;
    std::vector<size_t> sb_;               // slot bytes per layer
    std::vector<int> resident_l_;
    std::vector<float> rank_value_;        // [n_layer * n_expert]: from the initial ranking, hottest highest
    float value(int il, int e) const;
    void finish_pending(cudaStream_t s, bool wait);
    std::vector<std::vector<int>> slot_expert_;
    std::vector<int32_t> slot_of_host_;
    int32_t * slot_of_dev_ = nullptr;
    // adaptive state
    cudaStream_t copy_ = nullptr;
    std::vector<uint32_t> last_counts_;
    std::vector<float> score_;
    struct Pending { int il, slot, expert; bool was_empty; cudaEvent_t done; };
    std::vector<Pending> pending_;
    uint8_t * stage_ = nullptr;       // ggml blobs on their way to an R slot
    size_t stage_bytes_ = 0;
    uint8_t ** dst_ptrs_ = nullptr;   // device array of slot pointers for the repack kernel
    uint8_t * swap_stage_ = nullptr;  // adaptive swaps: one stage per in-flight swap
    uint8_t ** swap_ptrs_ = nullptr;
    int swap_n_ = 0;
    int stage_n_ = 0;
    void fill(int il, const std::vector<int> & slots, const std::vector<int> & experts, cudaStream_t s);
    std::vector<char> busy_;   // [n_layer * n_expert]: expert involved in an in-flight swap
};

}  // namespace bnk
