#include "core/expert_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>

#include "core/util.h"
#include "ggml.h"
#include "kernels/rfmt.h"

namespace bnk {

Ranking load_ranking(const std::string & path, int n_layer, int n_expert) {
    Ranking r;
    if (path.empty()) return r;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return r;
    char magic[4];
    if (fread(magic, 1, 4, f) != 4) { fclose(f); return r; }
    if (!memcmp(magic, "STRP", 4)) {
        uint32_t h[5];
        if (fread(h, 4, 5, f) == 5 && (int) h[1] == n_layer && (int) h[2] == n_expert) {
            for (uint32_t i = 0; i < h[4]; ++i) {
                uint16_t le[2];
                if (fread(le, 2, 2, f) != 2) break;
                r.emplace_back(le[0], le[1]);
            }
        }
        fclose(f);
        return r;
    }
    fclose(f);
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
    const Config & c = m_->cfg;
    const LayerWeights & L = m_->layers[il];
    const size_t sb = slot_bytes(*m_, il), bb = st_->blob_bytes(il);
    const size_t grb = ggml_row_size((ggml_type) L.gate_type, c.n_embd), drb = ggml_row_size((ggml_type) L.down_type, c.n_ff_exp);
    for (size_t i0 = 0; i0 < slots.size(); i0 += stage_n_) {
        const int n = (int) std::min<size_t>(stage_n_, slots.size() - i0);
        std::vector<uint8_t *> dst(n);
        for (int j = 0; j < n; ++j) {
            CUDA_CHECK(cudaMemcpyAsync(stage_ + (size_t) j * bb, st_->blob(il, experts[i0 + j]), bb, cudaMemcpyHostToDevice, s));
            dst[j] = layer_base_[il] + (size_t) slots[i0 + j] * sb;
        }
        CUDA_CHECK(cudaMemcpyAsync(dst_ptrs_, dst.data(), n * sizeof(uint8_t *), cudaMemcpyHostToDevice, s));
        moe_repack_blobs(stage_, bb, dst_ptrs_, n, c.n_ff_exp, c.n_embd, L.gate_type, grb, L.down_type, drb, s);
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
    if (region_) cudaFree(region_);
    if (slot_of_dev_) cudaFree(slot_of_dev_);
}

int ExpertCache::adapt(const uint32_t * counts_dev, cudaStream_t s, int max_swaps) {
    const size_t NN = (size_t) n_layer_ * n_expert_;
    if (!copy_) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
        swap_n_ = std::max(max_swaps, 1);
        CUDA_CHECK(cudaMalloc(&swap_stage_, stage_bytes_ * swap_n_));
        CUDA_CHECK(cudaMalloc(&swap_ptrs_, sizeof(uint8_t *) * swap_n_));
        last_counts_.assign(NN, 0);
        score_.assign(NN, 0.f);
        busy_.assign(NN, 0);
    }
    // 1. map the experts whose copies finished
    for (size_t i = 0; i < pending_.size();) {
        Pending & p = pending_[i];
        if (cudaEventQuery(p.done) != cudaSuccess) { ++i; continue; }
        int32_t sl = p.slot;
        CUDA_CHECK(cudaMemcpyAsync(slot_of_dev_ + (size_t) p.il * n_expert_ + p.expert, &sl, 4, cudaMemcpyHostToDevice, s));
        slot_of_host_[(size_t) p.il * n_expert_ + p.expert] = p.slot;
        busy_[(size_t) p.il * n_expert_ + p.expert] = 0;
        cudaEventDestroy(p.done);
        pending_[i] = pending_.back();
        pending_.pop_back();
        ++swaps_done;
    }
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
    // 3. per layer: the hottest non-resident vs the coldest resident
    struct Cand { float gain; int il, slot, in; };
    std::vector<Cand> cands;
    for (int il = 0; il < n_layer_; ++il) {
        if (slot_expert_[il].empty()) continue;
        const float * sc = score_.data() + (size_t) il * n_expert_;
        int best = -1, worst_slot = -1;
        for (int e = 0; e < n_expert_; ++e)
            if (slot_of_host_[(size_t) il * n_expert_ + e] < 0 && !busy_[(size_t) il * n_expert_ + e] &&
                (best < 0 || sc[e] > sc[best]))
                best = e;
        for (int sl = 0; sl < (int) slot_expert_[il].size(); ++sl) {
            const int e = slot_expert_[il][sl];
            if (busy_[(size_t) il * n_expert_ + e]) continue;
            if (worst_slot < 0 || sc[e] < sc[slot_expert_[il][worst_slot]]) worst_slot = sl;
        }
        if (best < 0 || worst_slot < 0) continue;
        const float gain = sc[best] - sc[slot_expert_[il][worst_slot]];
        if (gain > 1.5f) cands.push_back({gain, il, worst_slot, best});
    }
    std::sort(cands.begin(), cands.end(), [](const Cand & a, const Cand & b) { return a.gain > b.gain; });
    int started = 0;
    for (const Cand & c : cands) {
        if (started >= max_swaps || started >= swap_n_) break;
        const int old = slot_expert_[c.il][c.slot];
        int32_t minus1 = -1;
        CUDA_CHECK(cudaMemcpyAsync(slot_of_dev_ + (size_t) c.il * n_expert_ + old, &minus1, 4, cudaMemcpyHostToDevice, s));
        slot_of_host_[(size_t) c.il * n_expert_ + old] = -1;
        cudaEvent_t unmapped;
        CUDA_CHECK(cudaEventCreateWithFlags(&unmapped, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(unmapped, s));
        CUDA_CHECK(cudaStreamWaitEvent(copy_, unmapped, 0));
        cudaEventDestroy(unmapped);
        {
            const Config & cf = m_->cfg;
            const LayerWeights & L = m_->layers[c.il];
            const size_t bb = st_->blob_bytes(c.il), sbytes = slot_bytes(*m_, c.il);
            uint8_t * stg = swap_stage_ + (size_t) started * stage_bytes_;
            CUDA_CHECK(cudaMemcpyAsync(stg, st_->blob(c.il, c.in), bb, cudaMemcpyHostToDevice, copy_));
            uint8_t * dst = layer_base_[c.il] + (size_t) c.slot * sbytes;
            CUDA_CHECK(cudaMemcpyAsync(swap_ptrs_ + started, &dst, sizeof(uint8_t *), cudaMemcpyHostToDevice, copy_));
            moe_repack_blobs(stg, bb, swap_ptrs_ + started, 1, cf.n_ff_exp, cf.n_embd, L.gate_type,
                             ggml_row_size((ggml_type) L.gate_type, cf.n_embd), L.down_type,
                             ggml_row_size((ggml_type) L.down_type, cf.n_ff_exp), copy_);
        }
        Pending p{c.il, c.slot, c.in, nullptr};
        CUDA_CHECK(cudaEventCreateWithFlags(&p.done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(p.done, copy_));
        pending_.push_back(p);
        slot_expert_[c.il][c.slot] = c.in;
        busy_[(size_t) c.il * n_expert_ + c.in] = 1;
        ++started;
    }
    return started;
}

static Ranking complete_ranking(const Ranking & rank, int n_layer, int n_expert) {
    std::vector<char> seen((size_t) n_layer * n_expert, 0);
    Ranking full;
    for (auto [l, e] : rank)
        if (l >= 0 && l < n_layer && e >= 0 && e < n_expert && !seen[(size_t) l * n_expert + e]) {
            seen[(size_t) l * n_expert + e] = 1;
            full.emplace_back(l, e);
        }
    for (int e = 0; e < n_expert; ++e)
        for (int l = 0; l < n_layer; ++l)
            if (!seen[(size_t) l * n_expert + e]) full.emplace_back(l, e);
    return full;
}

size_t ExpertCache::slot_bytes(const Model & m, int il) {
    const LayerWeights & L = m.layers[il];
    const RLayout gl = r_layout(L.gate_type, m.cfg.n_embd), dl = r_layout(L.down_type, m.cfg.n_ff_exp);
    return ((size_t) 2 * m.cfg.n_ff_exp * gl.row_bytes + (size_t) m.cfg.n_embd * dl.row_bytes + 63) / 64 * 64;
}

std::vector<int> ExpertCache::plan(const Model & m, const ExpertStore & st, size_t budget, const Ranking & rank) {
    (void) st;
    std::vector<int> slots(m.cfg.n_layer, 0);
    size_t used = 0;
    for (auto [l, e] : complete_ranking(rank, m.cfg.n_layer, m.cfg.n_expert)) {
        const size_t b = slot_bytes(m, l);
        if (used + b > budget) continue;
        used += b;
        slots[l]++;
    }
    return slots;
}

void ExpertCache::init(const Model & m, const ExpertStore & st, size_t budget, Ranking rank, cudaStream_t s,
                       bool verbose) {
    m_ = &m;
    st_ = &st;
    n_layer_ = m.cfg.n_layer;
    n_expert_ = m.cfg.n_expert;
    const Ranking full = complete_ranking(rank, n_layer_, n_expert_);

    // take pairs in order while they fit
    slot_expert_.assign(n_layer_, {});
    size_t used = 0;
    for (auto [l, e] : full) {
        const size_t b = slot_bytes(m, l);
        if (used + b > budget) continue;
        used += b;
        slot_expert_[l].push_back(e);
    }
    bytes_ = used;
    if (used) CUDA_CHECK(cudaMalloc(&region_, used));
    layer_base_.assign(n_layer_, nullptr);
    slot_of_host_.assign((size_t) n_layer_ * n_expert_, -1);
    // staging for ggml blobs on their way into R slots
    stage_n_ = 64;
    stage_bytes_ = st.max_blob_bytes();
    CUDA_CHECK(cudaMalloc(&stage_, stage_bytes_ * stage_n_));
    CUDA_CHECK(cudaMalloc(&dst_ptrs_, sizeof(uint8_t *) * stage_n_));
    size_t off = 0;
    resident_ = 0;
    for (int l = 0; l < n_layer_; ++l) {
        layer_base_[l] = region_ + off;
        std::vector<int> sl, ex;
        for (size_t i = 0; i < slot_expert_[l].size(); ++i) {
            const int e = slot_expert_[l][i];
            slot_of_host_[(size_t) l * n_expert_ + e] = (int) i;
            sl.push_back((int) i);
            ex.push_back(e);
            ++resident_;
        }
        fill(l, sl, ex, s);
        off += slot_expert_[l].size() * slot_bytes(m, l);
    }
    CUDA_CHECK(cudaMalloc(&slot_of_dev_, slot_of_host_.size() * 4));
    CUDA_CHECK(cudaMemcpyAsync(slot_of_dev_, slot_of_host_.data(), slot_of_host_.size() * 4, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    if (verbose)
        fprintf(stderr, "bnk: VRAM expert cache: %d of %d experts (%.1f%%), %.2f GiB\n", resident_,
                n_layer_ * n_expert_, 100.0 * resident_ / (n_layer_ * n_expert_), used / 1073741824.0);
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
    const uint32_t g5[5] = {gl.off_a, gl.off_b, gl.off_c, gl.off_d, gl.off_e}, d5[5] = {dl.off_a, dl.off_b, dl.off_c, dl.off_d, dl.off_e};
    for (int i = 0; i < 5; ++i) { d.go[i] = g5[i]; d.dof[i] = d5[i]; }
    return d;
}

}  // namespace bnk
