#include "core/expert_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>

#include "core/util.h"
#include "ggml.h"
#include "kernels/rfmt.h"

namespace bnk {

// The expert ranking from a routing-counts file (bnk's BNKC; empty when there is none).
Ranking load_ranking(const std::string & path, int n_layer, int n_expert) {
    Ranking r;
    if (path.empty()) return r;
    auto counts = load_counts(path, n_layer, n_expert);
    if (!counts.empty()) r = ranking_from_counts(counts, n_layer, n_expert);
    return r;
}

Ranking ranking_from_counts(const std::vector<uint32_t> & counts, int n_layer, int n_expert) {
    std::vector<int> idx((size_t) n_layer * n_expert);
    std::iota(idx.begin(), idx.end(), 0);
    // stable on (count desc, rank-within-layer asc, layer asc): equal counts interleave across layers
    std::vector<int> within((size_t) n_layer * n_expert);
    for (int il = 0; il < n_layer; ++il) {
        std::vector<int> e(n_expert);
        std::iota(e.begin(), e.end(), 0);
        std::stable_sort(e.begin(), e.end(), [&](int a, int b) {
            return counts[(size_t) il * n_expert + a] > counts[(size_t) il * n_expert + b];
        });
        for (int i = 0; i < n_expert; ++i) within[(size_t) il * n_expert + e[i]] = i;
    }
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (counts[a] != counts[b]) return counts[a] > counts[b];
        if (within[a] != within[b]) return within[a] < within[b];
        return a < b;
    });
    Ranking r;
    for (int i : idx) r.emplace_back(i / n_expert, i % n_expert);
    return r;
}

bool save_counts(const std::string & path, const std::vector<uint32_t> & counts, int n_layer, int n_expert) {
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const uint32_t h[3] = {1, (uint32_t) n_layer, (uint32_t) n_expert};
    fwrite("BNKC", 1, 4, f);
    fwrite(h, 4, 3, f);
    fwrite(counts.data(), 4, counts.size(), f);
    fclose(f);
    return rename(tmp.c_str(), path.c_str()) == 0;
}

std::vector<uint32_t> load_counts(const std::string & path, int n_layer, int n_expert) {
    std::vector<uint32_t> c;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return c;
    char magic[4];
    uint32_t h[3];
    if (fread(magic, 1, 4, f) == 4 && !memcmp(magic, "BNKC", 4) && fread(h, 4, 3, f) == 3 &&
        (int) h[1] == n_layer && (int) h[2] == n_expert) {
        c.resize((size_t) n_layer * n_expert);
        if (fread(c.data(), 4, c.size(), f) != c.size()) c.clear();
    }
    fclose(f);
    return c;
}

// DMA the ggml blobs of (slot, expert) pairs into the stage and repack them into their R slots, in batches.
void ExpertCache::fill(int il, const std::vector<int> & slots, const std::vector<int> & experts, cudaStream_t s) {
    const LayerWeights & L = m_->layers[il];
    const Config & c = m_->cfg;
    const size_t bb = st_->blob_bytes(il);
    if (bb > stage_bytes_) throw std::runtime_error("expert cache: a blob larger than the stage");
    const size_t grb = ggml_row_size((ggml_type) L.gate_type, c.n_embd), drb = ggml_row_size((ggml_type) L.down_type, c.n_ff_exp);
    std::vector<uint8_t *> dst(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) dst[i] = layer_base_[il] + (size_t) slots[i] * sb_[il];
    for (size_t i0 = 0; i0 < dst.size(); i0 += stage_n_) {
        const int n = (int) std::min<size_t>(stage_n_, dst.size() - i0);
        for (int j = 0; j < n; ++j) {
            const int e = experts[i0 + j];
            uint8_t * sg = stage_ + (size_t) j * bb;
            if (const uint8_t * src = st_->blob(il, e)) {
                CUDA_CHECK(cudaMemcpyAsync(sg, src, bb, cudaMemcpyHostToDevice, s));
            } else {   // no host copy (a VRAM-only expert): the three matrices straight from the mapped file
                CUDA_CHECK(cudaMemcpyAsync(sg, L.gate_src + (size_t) e * L.gate_bytes, L.gate_bytes, cudaMemcpyHostToDevice, s));
                CUDA_CHECK(cudaMemcpyAsync(sg + L.gate_bytes, L.up_src + (size_t) e * L.up_bytes, L.up_bytes, cudaMemcpyHostToDevice, s));
                CUDA_CHECK(cudaMemcpyAsync(sg + L.gate_bytes + L.up_bytes, L.down_src + (size_t) e * L.down_bytes, L.down_bytes,
                                           cudaMemcpyHostToDevice, s));
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(dst_ptrs_, dst.data() + i0, n * sizeof(uint8_t *), cudaMemcpyHostToDevice, s));
        moe_repack_blobs(stage_, bb, dst_ptrs_, n, c.n_ff_exp, c.n_embd, L.gate_type, grb, L.down_type, drb, s);
        CUDA_CHECK(cudaStreamSynchronize(s));
    }
}

void ExpertCache::fill_blobs(const std::vector<uint8_t *> & dst, const std::vector<const uint8_t *> & src, size_t bb,
                             int gate_type, int down_type, cudaStream_t s) {
    const Config & c = m_->cfg;
    if (bb > stage_bytes_) throw std::runtime_error("expert cache: a blob larger than the stage");
    const size_t grb = ggml_row_size((ggml_type) gate_type, c.n_embd), drb = ggml_row_size((ggml_type) down_type, c.n_ff_exp);
    for (size_t i0 = 0; i0 < dst.size(); i0 += stage_n_) {
        const int n = (int) std::min<size_t>(stage_n_, dst.size() - i0);
        for (int j = 0; j < n; ++j)
            CUDA_CHECK(cudaMemcpyAsync(stage_ + (size_t) j * bb, src[i0 + j], bb, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(dst_ptrs_, dst.data() + i0, n * sizeof(uint8_t *), cudaMemcpyHostToDevice, s));
        moe_repack_blobs(stage_, bb, dst_ptrs_, n, c.n_ff_exp, c.n_embd, gate_type, grb, down_type, drb, s);
        // the stage is reused by the next batch: let this one finish first
        CUDA_CHECK(cudaStreamSynchronize(s));
    }
}

ExpertCache::~ExpertCache() {
    if (swap_stage_) cudaFree(swap_stage_);
    if (swap_ptrs_) cudaFree(swap_ptrs_);
    if (stage_) cudaFree(stage_);
    if (dst_ptrs_) cudaFree(dst_ptrs_);
    for (auto & p : pending_) cudaEventDestroy(p.done);
    if (copy_) cudaStreamDestroy(copy_);
    if (slot_of_dev_) cudaFree(slot_of_dev_);
}

size_t ExpertCache::bytes() const {
    size_t b = 0;
    for (const auto & lb : layer_buf_) b += lb.mapped();
    return b + cold_bytes_;
}

float ExpertCache::value(int il, int e) const {
    const size_t i = (size_t) il * n_expert_ + e;
    const float r = rank_value_.empty() ? 0.f : rank_value_[i];
    return score_.empty() ? r : score_[i] + 1e-3f * r;  // routing frequency, the initial ranking breaking ties
}

// Maps the experts whose swap copies finished (with `wait`, waits for every in-flight copy first).
void ExpertCache::finish_pending(cudaStream_t s, bool wait) {
    if (wait && copy_) CUDA_CHECK(cudaStreamSynchronize(copy_));
    for (size_t i = 0; i < pending_.size();) {
        Pending & p = pending_[i];
        if (cudaEventQuery(p.done) != cudaSuccess) { ++i; continue; }
        int32_t sl = p.slot;
        CUDA_CHECK(cudaMemcpyAsync(slot_of_dev_ + (size_t) p.il * n_expert_ + p.expert, &sl, 4, cudaMemcpyHostToDevice, s));
        slot_of_host_[(size_t) p.il * n_expert_ + p.expert] = p.slot;
        busy_[(size_t) p.il * n_expert_ + p.expert] = 0;
        if (p.was_empty) {
            ++resident_;
            ++resident_l_[p.il];
        }
        cudaEventDestroy(p.done);
        pending_[i] = pending_.back();
        pending_.pop_back();
        ++swaps_done;
    }
}

int ExpertCache::adapt(const uint32_t * counts_dev, cudaStream_t s, int max_swaps) {
    const size_t NN = (size_t) n_layer_ * n_expert_;
    if (!copy_) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
        // the staging for swaps in flight is allocated outside the VRAM budget: with little free memory take
        // fewer of them rather than fail (64 did not fit next to three conversations' context)
        swap_n_ = std::max(max_swaps, 1);
        while (cudaMalloc(&swap_stage_, stage_bytes_ * swap_n_) != cudaSuccess) {
            cudaGetLastError();
            if (swap_n_ == 1) throw std::runtime_error("expert cache: no VRAM for swap staging");
            swap_n_ = std::max(1, swap_n_ / 2);
        }
        CUDA_CHECK(cudaMalloc(&swap_ptrs_, sizeof(uint8_t *) * swap_n_));
        last_counts_.assign(NN, 0);
        score_.assign(NN, 0.f);
        busy_.assign(NN, 0);
    }
    // 1. map the experts whose copies finished
    finish_pending(s, false);
    // 2. decayed routing frequency
    std::vector<uint32_t> now(NN);
    CUDA_CHECK(cudaMemcpyAsync(now.data(), counts_dev, NN * 4, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    for (size_t i = 0; i < NN; ++i) {
        const uint32_t d = now[i] - last_counts_[i];
        score_[i] = score_[i] * 0.97f + (float) d;
    }
    last_counts_.swap(now);
    if (max_swaps <= 0 || !pending_.empty()) return 0;
    // 3. per layer: the hottest non-resident vs the coldest resident (an empty slot first)
    struct Cand { float gain; int il, slot, in; };
    std::vector<Cand> cands;
    for (int il = 0; il < n_layer_; ++il) {
        if (slot_expert_[il].empty()) continue;
        const float * sc = score_.data() + (size_t) il * n_expert_;
        int best = -1, worst_slot = -1;
        float worst_v = 0.f;
        for (int e = 0; e < n_expert_; ++e)
            if (slot_of_host_[(size_t) il * n_expert_ + e] < 0 && !busy_[(size_t) il * n_expert_ + e] &&
                (best < 0 || sc[e] > sc[best]))
                best = e;
        for (int sl = 0; sl < (int) slot_expert_[il].size(); ++sl) {
            const int e = slot_expert_[il][sl];
            if (e >= 0 && busy_[(size_t) il * n_expert_ + e]) continue;
            const float v = e < 0 ? -1.f : sc[e];
            if (worst_slot < 0 || v < worst_v) {
                worst_slot = sl;
                worst_v = v;
            }
        }
        if (best < 0 || worst_slot < 0) continue;
        const bool empty = slot_expert_[il][worst_slot] < 0;
        const float gain = sc[best] - (empty ? 0.f : worst_v);
        if (gain > 1.5f || (empty && sc[best] > 0.f)) cands.push_back({gain + (empty ? 1e6f : 0.f), il, worst_slot, best});
    }
    std::sort(cands.begin(), cands.end(), [](const Cand & a, const Cand & b) { return a.gain > b.gain; });
    int started = 0;
    for (const Cand & c : cands) {
        if (started >= max_swaps || started >= swap_n_) break;
        const int old = slot_expert_[c.il][c.slot];
        if (old >= 0) {
            int32_t minus1 = -1;
            CUDA_CHECK(cudaMemcpyAsync(slot_of_dev_ + (size_t) c.il * n_expert_ + old, &minus1, 4, cudaMemcpyHostToDevice, s));
            slot_of_host_[(size_t) c.il * n_expert_ + old] = -1;
        }
        cudaEvent_t unmapped;
        CUDA_CHECK(cudaEventCreateWithFlags(&unmapped, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(unmapped, s));
        CUDA_CHECK(cudaStreamWaitEvent(copy_, unmapped, 0));
        cudaEventDestroy(unmapped);
        {
            const Config & cf = m_->cfg;
            const LayerWeights & L = m_->layers[c.il];
            const size_t bb = st_->blob_bytes(c.il);
            uint8_t * stg = swap_stage_ + (size_t) started * stage_bytes_;
            CUDA_CHECK(cudaMemcpyAsync(stg, st_->blob(c.il, c.in), bb, cudaMemcpyHostToDevice, copy_));
            uint8_t * dst = layer_base_[c.il] + (size_t) c.slot * sb_[c.il];
            CUDA_CHECK(cudaMemcpyAsync(swap_ptrs_ + started, &dst, sizeof(uint8_t *), cudaMemcpyHostToDevice, copy_));
            moe_repack_blobs(stg, bb, swap_ptrs_ + started, 1, cf.n_ff_exp, cf.n_embd, L.gate_type,
                             ggml_row_size((ggml_type) L.gate_type, cf.n_embd), L.down_type,
                             ggml_row_size((ggml_type) L.down_type, cf.n_ff_exp), copy_);
        }
        Pending p{c.il, c.slot, c.in, old < 0, nullptr};
        CUDA_CHECK(cudaEventCreateWithFlags(&p.done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(p.done, copy_));
        pending_.push_back(p);
        slot_expert_[c.il][c.slot] = c.in;
        busy_[(size_t) c.il * n_expert_ + c.in] = 1;
        ++started;
    }
    return started;
}

// the ranking's pairs of layers [l0, l1), completed with every remaining pair of those layers
static Ranking complete_ranking(const Ranking & rank, int n_layer, int n_expert, int l0 = 0, int l1 = -1) {
    if (l1 < 0) l1 = n_layer;
    std::vector<char> seen((size_t) n_layer * n_expert, 0);
    Ranking full;
    for (auto [l, e] : rank)
        if (l >= l0 && l < l1 && e >= 0 && e < n_expert && !seen[(size_t) l * n_expert + e]) {
            seen[(size_t) l * n_expert + e] = 1;
            full.emplace_back(l, e);
        }
    for (int e = 0; e < n_expert; ++e)
        for (int l = l0; l < l1; ++l)
            if (!seen[(size_t) l * n_expert + e]) full.emplace_back(l, e);
    return full;
}

size_t ExpertCache::slot_bytes(const Model & m, int il) {
    const LayerWeights & L = m.layers[il];
    return slot_bytes_for(m, L.gate_type, L.down_type);
}

size_t ExpertCache::slot_bytes_for(const Model & m, int gate_type, int down_type) {
    const RLayout gl = r_layout(gate_type, m.cfg.n_embd), dl = r_layout(down_type, m.cfg.n_ff_exp);
    return ((size_t) 2 * m.cfg.n_ff_exp * gl.row_bytes + (size_t) m.cfg.n_embd * dl.row_bytes + 63) / 64 * 64;
}

Ranking complete_ranking_public(const Ranking & rank, int n_layer, int n_expert, int l0, int l1) {
    return complete_ranking(rank, n_layer, n_expert, l0, l1);
}

// pairs in ranking order while they fit; a layer's region is mapped in whole granules
static std::vector<int> plan_slots(const Model & m, size_t budget, const Ranking & full) {
    std::vector<int> slots(m.cfg.n_layer, 0);
    std::vector<size_t> sb(m.cfg.n_layer);
    for (int l = 0; l < m.cfg.n_layer; ++l) sb[l] = ExpertCache::slot_bytes(m, l);
    size_t used = 0;
    for (auto [l, e] : full) {
        (void) e;
        const size_t add = vmem_round((slots[l] + 1) * sb[l]) - vmem_round(slots[l] * sb[l]);
        if (used + add > budget) continue;
        used += add;
        slots[l]++;
    }
    return slots;
}

std::vector<int> ExpertCache::plan(const Model & m, const ExpertStore & st, size_t budget, const Ranking & rank,
                                   int l0, int l1) {
    (void) st;
    return plan_slots(m, budget, complete_ranking(rank, m.cfg.n_layer, m.cfg.n_expert, l0, l1));
}

void ExpertCache::init(const Model & m, const ExpertStore & st, size_t budget, Ranking rank, cudaStream_t s,
                       bool verbose, VramBudget * vb, int l0, int l1, const ColdStore * cs) {
    m_ = &m;
    st_ = &st;
    cs_ = cs && !cs->empty() ? cs : nullptr;
    n_layer_ = m.cfg.n_layer;
    n_expert_ = m.cfg.n_expert;
    l0_ = l0;
    l1_ = l1 < 0 ? n_layer_ : l1;
    // the cold bands: their regions come off the budget first, every cold expert resident; the hot ranking
    // (the other experts) fills the rest
    cold_buf_.assign(n_layer_, {});
    cold_base_.assign(n_layer_, {});
    cold_sb_.assign(n_layer_, {});
    cold_bytes_ = 0;
    int n_cold = 0;
    if (cs_) {   // a budget too small for the bands (a CPU-only tier, --cache-gib 0): the file is ignored, the store has every expert
        size_t need = 0;
        for (int l = l0_; l < l1_; ++l)
            for (const auto & g : m.layers[l].cold) need += vmem_round(slot_bytes_for(m, g.gate_type, g.down_type) * g.experts.size());
        if (need > budget) {
            if (verbose) fprintf(stderr, "bnk: the cold experts need %.2f GiB of VRAM, the cache has %.2f: running without them\n", need / 1073741824.0, budget / 1073741824.0);
            cs_ = nullptr;
        }
    }
    if (cs_) {
        for (int l = l0_; l < l1_; ++l) {
            const LayerWeights & L = m.layers[l];
            for (const auto & g : L.cold) {
                const size_t sb = slot_bytes_for(m, g.gate_type, g.down_type);
                cold_sb_[l].push_back(sb);
                const size_t region = vmem_round(sb * g.experts.size());
                budget -= region;
                cold_bytes_ += region;
                n_cold += (int) g.experts.size();
            }
        }
        // the hot ranking leaves the cold experts out
        Ranking hot;
        for (auto [l, e] : rank)
            if (l < 0 || l >= n_layer_ || m.layers[l].cold_of.empty() || m.layers[l].cold_of[e] < 0) hot.push_back({l, e});
        rank = std::move(hot);
    }
    Ranking full = complete_ranking(rank, n_layer_, n_expert_, l0_, l1_);
    if (cs_) {
        Ranking hot;
        for (auto [l, e] : full)
            if (m.layers[l].cold_of.empty() || m.layers[l].cold_of[e] < 0) hot.push_back({l, e});
        full = std::move(hot);
    }
    rank_value_.assign((size_t) n_layer_ * n_expert_, 0.f);
    for (size_t i = 0; i < full.size(); ++i)
        rank_value_[(size_t) full[i].first * n_expert_ + full[i].second] = (float) (full.size() - i) / full.size();

    const std::vector<int> nslots = plan_slots(m, budget, full);
    slot_expert_.assign(n_layer_, {});
    {
        std::vector<int> left = nslots;
        for (auto [l, e] : full)
            if (left[l] > 0) {
                slot_expert_[l].push_back(e);
                --left[l];
            }
    }
    sb_.resize(n_layer_);
    layer_buf_ = std::vector<ElasticBuf>(n_layer_);
    layer_base_.assign(n_layer_, nullptr);
    resident_l_.assign(n_layer_, 0);
    slot_of_host_.assign((size_t) n_layer_ * n_expert_, -1);
    // staging for ggml blobs on their way into R slots
    stage_n_ = 64;
    stage_bytes_ = std::max(st.max_blob_bytes(), cs_ ? cs_->max_blob_bytes() : (size_t) 0);   // a cold band's blobs may be larger (Q8_0)
    CUDA_CHECK(cudaMalloc(&stage_, stage_bytes_ * stage_n_));
    CUDA_CHECK(cudaMalloc(&dst_ptrs_, sizeof(uint8_t *) * stage_n_));
    resident_ = 0;
    if (cs_) {   // the cold regions: mapped whole, filled from the cold store, numbered after the hot slots
        for (int l = l0_; l < l1_; ++l) {
            const LayerWeights & L = m.layers[l];
            cold_buf_[l] = std::vector<ElasticBuf>(L.cold.size());
            cold_base_[l].assign(L.cold.size(), nullptr);
            for (size_t g = 0; g < L.cold.size(); ++g) {
                const auto & cg = L.cold[g];
                const size_t sb = cold_sb_[l][g];
                cold_buf_[l][g].reserve(sb * cg.experts.size(), vb, "cold experts", false);
                cold_buf_[l][g].ensure(sb * cg.experts.size());
                cold_base_[l][g] = cold_buf_[l][g].as<uint8_t>();
                std::vector<uint8_t *> dst(cg.experts.size());
                std::vector<const uint8_t *> src(cg.experts.size());
                for (size_t i = 0; i < cg.experts.size(); ++i) {
                    slot_of_host_[(size_t) l * n_expert_ + cg.experts[i]] = n_expert_ * (1 + (int) g) + (int) i;
                    dst[i] = cold_base_[l][g] + i * sb;
                    src[i] = cs_->blob(l, (int) g, (int) i);
                }
                fill_blobs(dst, src, cs_->blob_bytes(l, (int) g), cg.gate_type, cg.down_type, s);
                resident_l_[l] += (int) cg.experts.size();
                resident_ += (int) cg.experts.size();
            }
        }
    }
    for (int l = 0; l < n_layer_; ++l) {
        sb_[l] = slot_bytes(m, l);
        if (!owns(l)) continue;
        layer_buf_[l].reserve((size_t) n_expert_ * sb_[l], vb, "expert cache", false);
        layer_buf_[l].ensure(slot_expert_[l].size() * sb_[l]);
        layer_base_[l] = layer_buf_[l].as<uint8_t>();
        std::vector<int> sl, ex;
        for (size_t i = 0; i < slot_expert_[l].size(); ++i) {
            const int e = slot_expert_[l][i];
            slot_of_host_[(size_t) l * n_expert_ + e] = (int) i;
            sl.push_back((int) i);
            ex.push_back(e);
        }
        resident_l_[l] += (int) ex.size();
        resident_ += (int) ex.size();
        fill(l, sl, ex, s);
    }
    CUDA_CHECK(cudaMalloc(&slot_of_dev_, slot_of_host_.size() * 4));
    CUDA_CHECK(cudaMemcpyAsync(slot_of_dev_, slot_of_host_.data(), slot_of_host_.size() * 4, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    const int n = (l1_ - l0_) * n_expert_;
    if (verbose)
        fprintf(stderr, "bnk: VRAM expert cache%s: %d of %d experts (%.1f%%), %.2f GiB%s\n",
                n == n_layer_ * n_expert_ ? "" : (" for layers " + std::to_string(l0_) + "-" + std::to_string(l1_ - 1)).c_str(),
                resident_, n, 100.0 * resident_ / n, bytes() / 1073741824.0,
                n_cold ? (" (" + std::to_string(n_cold) + " cold in " + std::to_string(cold_bytes_ >> 20) + " MiB)").c_str() : "");
}

MoeLayerDesc ExpertCache::desc_cold(int il, int g) const {
    const LayerWeights::ColdGroup & cg = m_->layers[il].cold.at(g);
    const Config & c = m_->cfg;
    const RLayout gl = r_layout(cg.gate_type, c.n_embd), dl = r_layout(cg.down_type, c.n_ff_exp);
    MoeLayerDesc d;
    d.slot_of = slot_of_dev_ + (size_t) il * n_expert_;
    d.base = cold_base_[il][g];
    d.blob = cold_sb_[il][g];
    d.gate_bytes = (size_t) c.n_ff_exp * gl.row_bytes;
    d.up_bytes = (size_t) c.n_ff_exp * gl.row_bytes;
    d.gate_type = cg.gate_type;
    d.down_type = cg.down_type;
    d.grow = gl.row_bytes;
    d.drow = dl.row_bytes;
    d.rlay = 1;
    d.slot_lo = n_expert_ * (1 + g);
    d.slot_hi = d.slot_lo + (int) cg.experts.size();
    d.scales = ExpScales{};   // the global scale is folded into the cold blocks
    const uint32_t g5[5] = {gl.off_a, gl.off_b, gl.off_c, gl.off_d, gl.off_e}, d5[5] = {dl.off_a, dl.off_b, dl.off_c, dl.off_d, dl.off_e};
    for (int i = 0; i < 5; ++i) { d.go[i] = g5[i]; d.dof[i] = d5[i]; }
    return d;
}

size_t ExpertCache::shrink(size_t need, cudaStream_t s) {
    finish_pending(s, true);
    const int L = n_layer_;
    // per layer: its slots, coldest first (empty slots before any expert)
    std::vector<std::vector<std::pair<float, int>>> order(L);
    for (int l = 0; l < L; ++l) {
        for (int sl = 0; sl < (int) slot_expert_[l].size(); ++sl) {
            const int e = slot_expert_[l][sl];
            order[l].push_back({e < 0 ? -1e30f : value(l, e), sl});
        }
        std::sort(order[l].begin(), order[l].end());
    }
    std::vector<int> cut(L, 0);
    auto freed_of = [&](int l) {
        const size_t n = slot_expert_[l].size() - cut[l];
        return layer_buf_[l].mapped() - vmem_round(n * sb_[l]);
    };
    size_t freed = 0;
    while (freed < need) {
        int bl = -1;
        float bv = 0.f;
        for (int l = 0; l < L; ++l)
            if (cut[l] < (int) order[l].size() && (bl < 0 || order[l][cut[l]].first < bv)) {
                bl = l;
                bv = order[l][cut[l]].first;
            }
        if (bl < 0) break;
        freed -= freed_of(bl);
        cut[bl]++;
        freed += freed_of(bl);
    }
    for (int l = 0; l < L; ++l) {
        if (!cut[l]) continue;
        const int n = (int) slot_expert_[l].size(), nn = n - cut[l];
        std::vector<char> evict(n, 0);
        for (int i = 0; i < cut[l]; ++i) evict[order[l][i].second] = 1;
        // kept experts past the new end move into evicted slots below it (there are as many of each)
        std::vector<int> holes, movers;
        for (int sl = 0; sl < nn; ++sl)
            if (evict[sl]) holes.push_back(sl);
        for (int sl = nn; sl < n; ++sl)
            if (!evict[sl]) movers.push_back(sl);
        for (int sl = 0; sl < n; ++sl) {
            const int e = slot_expert_[l][sl];
            if (evict[sl] && e >= 0) {
                slot_of_host_[(size_t) l * n_expert_ + e] = -1;
                --resident_;
                --resident_l_[l];
            }
        }
        for (size_t i = 0; i < movers.size(); ++i) {
            const int from = movers[i], to = holes[i], e = slot_expert_[l][from];
            CUDA_CHECK(cudaMemcpyAsync(layer_base_[l] + (size_t) to * sb_[l], layer_base_[l] + (size_t) from * sb_[l],
                                       sb_[l], cudaMemcpyDeviceToDevice, s));
            slot_expert_[l][to] = e;
            if (e >= 0) slot_of_host_[(size_t) l * n_expert_ + e] = to;
        }
        slot_expert_[l].resize(nn);
    }
    CUDA_CHECK(cudaMemcpyAsync(slot_of_dev_, slot_of_host_.data(), slot_of_host_.size() * 4, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    size_t done = 0;
    for (int l = 0; l < L; ++l) {
        const size_t before = layer_buf_[l].mapped();
        layer_buf_[l].shrink_to(slot_expert_[l].size() * sb_[l]);
        done += before - layer_buf_[l].mapped();
    }
    if (getenv("BNK_VMEM_LOG"))
        fprintf(stderr, "bnk: expert cache shrank by %.2f GiB to %d experts\n", done / 1073741824.0, resident_);
    return done;
}

size_t ExpertCache::grow(size_t avail, cudaStream_t s) {
    finish_pending(s, true);
    const int L = n_layer_;
    // per layer: its non-resident experts, hottest first
    std::vector<std::vector<std::pair<float, int>>> cand(L);
    for (int l = l0_; l < l1_; ++l) {
        for (int e = 0; e < n_expert_; ++e)
            if (slot_of_host_[(size_t) l * n_expert_ + e] < 0 && (busy_.empty() || !busy_[(size_t) l * n_expert_ + e]))
                cand[l].push_back({-value(l, e), e});
        std::sort(cand[l].begin(), cand[l].end());
    }
    std::vector<int> take(L, 0);
    size_t used = 0;
    for (;;) {
        int bl = -1;
        float bv = 0.f;
        for (int l = 0; l < L; ++l) {
            if (take[l] >= (int) cand[l].size()) continue;
            const size_t n = slot_expert_[l].size() + take[l];
            const size_t add = vmem_round((n + 1) * sb_[l]) - std::max(vmem_round(n * sb_[l]), layer_buf_[l].mapped());
            if (used + add > avail) continue;
            const float v = -cand[l][take[l]].first;
            if (bl < 0 || v > bv) {
                bl = l;
                bv = v;
            }
        }
        if (bl < 0) break;
        const size_t n = slot_expert_[bl].size() + take[bl];
        used += vmem_round((n + 1) * sb_[bl]) - std::max(vmem_round(n * sb_[bl]), layer_buf_[bl].mapped());
        take[bl]++;
    }
    size_t done = 0;
    for (int l = 0; l < L; ++l) {
        if (!take[l]) continue;
        const int n0 = (int) slot_expert_[l].size();
        const size_t before = layer_buf_[l].mapped();
        layer_buf_[l].ensure((size_t) (n0 + take[l]) * sb_[l]);
        done += layer_buf_[l].mapped() - before;
        std::vector<int> sl, ex;
        for (int i = 0; i < take[l]; ++i) {
            const int e = cand[l][i].second;
            slot_expert_[l].push_back(e);
            slot_of_host_[(size_t) l * n_expert_ + e] = n0 + i;
            sl.push_back(n0 + i);
            ex.push_back(e);
        }
        fill(l, sl, ex, s);
        resident_l_[l] += take[l];
        resident_ += take[l];
    }
    CUDA_CHECK(cudaMemcpyAsync(slot_of_dev_, slot_of_host_.data(), slot_of_host_.size() * 4, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    if (getenv("BNK_VMEM_LOG") && done)
        fprintf(stderr, "bnk: expert cache grew by %.2f GiB to %d experts\n", done / 1073741824.0, resident_);
    return done;
}

void ExpertCache::reserve_spare(int il, int n) {
    if (!owns(il) || n <= 0) return;
    if (slot_expert_[il].size() + n > (size_t) n_expert_) throw std::runtime_error("expert cache: no room for spare slots");
    layer_buf_[il].ensure((slot_expert_[il].size() + n) * sb_[il]);
}

MoeLayerDesc ExpertCache::desc(int il) const {
    const LayerWeights & L = m_->layers[il];
    const Config & c = m_->cfg;
    const RLayout gl = r_layout(L.gate_type, c.n_embd), dl = r_layout(L.down_type, c.n_ff_exp);
    MoeLayerDesc d;
    d.slot_of = slot_of_dev_ + (size_t) il * n_expert_;
    d.base = layer_base_[il];
    d.blob = slot_bytes(*m_, il);
    d.gate_bytes = (size_t) c.n_ff_exp * gl.row_bytes;
    d.up_bytes = (size_t) c.n_ff_exp * gl.row_bytes;
    d.gate_type = L.gate_type;
    d.down_type = L.down_type;
    d.grow = gl.row_bytes;
    d.drow = dl.row_bytes;
    d.rlay = 1;
    d.slot_lo = 0;
    d.slot_hi = n_expert_;
    d.scales = L.exp_scales;
    const uint32_t g5[5] = {gl.off_a, gl.off_b, gl.off_c, gl.off_d, gl.off_e}, d5[5] = {dl.off_a, dl.off_b, dl.off_c, dl.off_d, dl.off_e};
    for (int i = 0; i < 5; ++i) { d.go[i] = g5[i]; d.dof[i] = d5[i]; }
    return d;
}

}  // namespace bnk
