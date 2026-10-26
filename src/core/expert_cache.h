// The VRAM tier of the routed experts: per-layer slot regions holding the most-used experts' blobs.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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
// `rank` restricted to layers [l0, l1) and completed with every missing pair (the order plan()/init() fill in)
Ranking complete_ranking_public(const Ranking & rank, int n_layer, int n_expert, int l0, int l1);
std::vector<uint32_t> load_counts(const std::string & path, int n_layer, int n_expert);
// delta += old, then halved (all of it) while the total exceeds the cap (BNK_COUNTS_CAP routings, default 2^28): the
// file ranks by a sliding window of the recent workload rather than everything ever served
void merge_counts(std::vector<uint32_t> & delta, const std::vector<uint32_t> & old);

class ExpertCache {
public:
    ~ExpertCache();
    // Fills up to `budget` bytes of VRAM following `rank` (completed with every remaining pair). Each layer's
    // slots live in an elastic region charged to `vb`, so the cache can later shrink() and grow().
    // A pipeline stage (several GPUs) caches only layers [l0, l1) (l1 < 0: every layer).
    // `cs`: the cold experts' blobs (model.h ColdGroup); every cold expert gets a slot of its own region first,
    // the hot ranking fills what is left of the budget.
    void init(const Model & m, const ExpertStore & st, size_t budget, Ranking rank, cudaStream_t s, bool verbose,
              VramBudget * vb, int l0 = 0, int l1 = -1, const ColdStore * cs = nullptr);
    // The per-layer slot counts init() would choose, without allocating.
    static std::vector<int> plan(const Model & m, const ExpertStore & st, size_t budget, const Ranking & rank,
                                 int l0 = 0, int l1 = -1);
    bool owns(int il) const { return il >= l0_ && il < l1_; }
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
    static size_t slot_bytes_for(const Model & m, int gate_type, int down_type);
    MoeLayerDesc desc(int il) const;          // the hot region (slots [0, n_expert))
    int n_cold_groups(int il) const { return (int) cold_sb_[il].size(); }
    MoeLayerDesc desc_cold(int il, int g) const;   // cold region g (slots [n_expert * (1 + g), + its experts))
    int resident() const { return resident_; }
    size_t bytes() const;
    int slots(int il) const { return (int) slot_expert_[il].size(); }
    // n slots past the layer's resident ones, mapped (the budget charged) but unassigned: a decode-time staging area
    // whose slots the hit kernels address like any other (base + slot * blob). spare_slot(il, p) is slot p of them.
    void reserve_spare(int il, int n);
    int spare_slot(int il, int p) const { return (int) slot_expert_[il].size() + p; }

    // Adaptive tier. Call between forwards on the compute stream: finishes copies that are done, then
    // plans new swaps from routing counts (device [n_layer][n_expert], cumulative) and starts their
    // copies on a side stream. Returns the number of swaps started.
    int adapt(const uint32_t * counts_dev, cudaStream_t s, int max_swaps);
    // The adaptive tier for a disjoint placement (a resident expert has no host copy, docs/DEEPSEEK.md): the hottest
    // non-resident expert of a layer takes the slot of its coldest resident one, which first gets a host copy in one
    // of the store's spare blobs (a helper thread writes it from the mapped model file). Between forwards on the
    // compute stream: (1) maps the copies that finished and releases the host blobs of the experts now resident,
    // (2) for exchanges whose host copy is written, unmaps the leaving expert and starts the incoming one's copy,
    // (3) plans new exchanges from the routing counts. Every expert is resident or in the store at every moment,
    // and often both for a moment. Returns the exchanges started. Needs the store built with spare blobs.
    int adapt_exchange(const uint32_t * counts_dev, cudaStream_t s, int max_swaps, ExpertStore & store);
    int exchanges_pending() const { return (int) xs_.size(); }
    int64_t swaps_done = 0;
    const std::vector<int> & slot_experts(int il) const { return slot_expert_[il]; }
    int slot_of(int il, int e) const { return slot_of_host_[(size_t) il * n_expert_ + e]; }

private:
    const Model * m_ = nullptr;
    const ExpertStore * st_ = nullptr;
    int n_layer_ = 0, n_expert_ = 0, resident_ = 0;
    int l0_ = 0, l1_ = 0;   // the layers this cache serves
    std::vector<ElasticBuf> layer_buf_;   // each layer's slots, slot i at i * slot_bytes
    std::vector<uint8_t *> layer_base_;
    std::vector<size_t> sb_;               // slot bytes per layer
    const ColdStore * cs_ = nullptr;
    std::vector<std::vector<ElasticBuf>> cold_buf_;   // [layer][group]: the cold bands' regions (never shrunk)
    std::vector<std::vector<uint8_t *>> cold_base_;
    std::vector<std::vector<size_t>> cold_sb_;
    size_t cold_bytes_ = 0;
    // ggml blobs (src, `bb` bytes each, of types gate_type / down_type) repacked into R slots at dst
    void fill_blobs(const std::vector<uint8_t *> & dst, const std::vector<const uint8_t *> & src, size_t bb, int gate_type,
                    int down_type, cudaStream_t s);
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
    // shared by adapt() and adapt_exchange(): the copy stream and staging, the decayed routing scores, the plan
    void adapt_init(int max_swaps);
    void update_scores(const uint32_t * counts_dev, cudaStream_t s);
    struct Cand { float gain; int il, slot, in; };
    std::vector<Cand> plan_swaps() const;   // per layer at most one: the hottest non-resident vs the coldest resident
    // the exchange: a leaving expert's host copy (the helper thread), then the GPU copy of the incoming one
    struct Exchange {
        int il, slot, in, out, blob, stage;
        std::atomic<int> host_done{0};
        bool gpu_started = false;
        cudaEvent_t done = nullptr;
    };
    std::vector<std::unique_ptr<Exchange>> xs_;
    std::vector<int> stage_free_;          // swap_stage_ slots not in use
    std::thread copier_;
    std::mutex hq_mu_;
    std::condition_variable hq_cv_;
    std::deque<Exchange *> hq_;
    bool hq_stop_ = false;
    ExpertStore * xstore_ = nullptr;
    void copier_main();
};

}  // namespace bnk
