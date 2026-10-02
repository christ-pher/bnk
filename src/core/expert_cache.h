// The VRAM tier of the routed experts: per-layer slot regions holding the most-used experts' blobs.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "core/model.h"
#include "cpu/expert_pool.h"
#include "kernels/moe.h"

namespace bnk {

using Ranking = std::vector<std::pair<int, int>>;  // (layer, expert), hottest first

// Strata's STRP profile, or bnk's own counts file (BNKC); empty when unreadable.
Ranking load_ranking(const std::string & path, int n_layer, int n_expert);
// Ranking from routing counts [n_layer][n_expert]: most routed first, ties interleaved across layers.
Ranking ranking_from_counts(const std::vector<uint32_t> & counts, int n_layer, int n_expert);
bool save_counts(const std::string & path, const std::vector<uint32_t> & counts, int n_layer, int n_expert);
std::vector<uint32_t> load_counts(const std::string & path, int n_layer, int n_expert);

class ExpertCache {
public:
    ~ExpertCache();
    // Fills up to `budget` bytes of VRAM following `rank` (completed with every remaining pair).
    void init(const Model & m, const ExpertStore & st, size_t budget, Ranking rank, cudaStream_t s, bool verbose);
    MoeLayerDesc desc(int il) const;
    int resident() const { return resident_; }
    size_t bytes() const { return bytes_; }
    int slots(int il) const { return (int) slot_expert_[il].size(); }
    // Replace the expert in `slot` of layer il by `expert` (stream-ordered; host map updated too).
    void swap(int il, int slot, int expert, cudaStream_t s);

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
    size_t bytes_ = 0;
    uint8_t * region_ = nullptr;
    std::vector<uint8_t *> layer_base_;
    std::vector<std::vector<int>> slot_expert_;
    std::vector<int32_t> slot_of_host_;
    int32_t * slot_of_dev_ = nullptr;
    // adaptive state
    cudaStream_t copy_ = nullptr;
    std::vector<uint32_t> last_counts_;
    std::vector<float> score_;
    struct Pending { int il, slot, expert; cudaEvent_t done; };
    std::vector<Pending> pending_;
    std::vector<char> busy_;   // [n_layer * n_expert]: expert involved in an in-flight swap
};

}  // namespace bnk
