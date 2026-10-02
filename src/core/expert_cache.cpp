#include "core/expert_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>

#include "core/util.h"
#include "ggml.h"

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

ExpertCache::~ExpertCache() {
    if (region_) cudaFree(region_);
    if (slot_of_dev_) cudaFree(slot_of_dev_);
}

void ExpertCache::init(const Model & m, const ExpertStore & st, size_t budget, Ranking rank, cudaStream_t s,
                       bool verbose) {
    m_ = &m;
    st_ = &st;
    n_layer_ = m.cfg.n_layer;
    n_expert_ = m.cfg.n_expert;
    // complete the ranking with every pair not yet listed, interleaved across layers
    std::vector<char> seen((size_t) n_layer_ * n_expert_, 0);
    Ranking full;
    for (auto [l, e] : rank)
        if (l >= 0 && l < n_layer_ && e >= 0 && e < n_expert_ && !seen[(size_t) l * n_expert_ + e]) {
            seen[(size_t) l * n_expert_ + e] = 1;
            full.emplace_back(l, e);
        }
    for (int e = 0; e < n_expert_; ++e)
        for (int l = 0; l < n_layer_; ++l)
            if (!seen[(size_t) l * n_expert_ + e]) full.emplace_back(l, e);

    // take pairs in order while they fit
    slot_expert_.assign(n_layer_, {});
    size_t used = 0;
    for (auto [l, e] : full) {
        const size_t b = st.blob_bytes(l);
        if (used + b > budget) continue;
        used += b;
        slot_expert_[l].push_back(e);
    }
    bytes_ = used;
    if (used) CUDA_CHECK(cudaMalloc(&region_, used));
    layer_base_.assign(n_layer_, nullptr);
    slot_of_host_.assign((size_t) n_layer_ * n_expert_, -1);
    size_t off = 0;
    resident_ = 0;
    for (int l = 0; l < n_layer_; ++l) {
        layer_base_[l] = region_ + off;
        for (size_t i = 0; i < slot_expert_[l].size(); ++i) {
            const int e = slot_expert_[l][i];
            slot_of_host_[(size_t) l * n_expert_ + e] = (int) i;
            CUDA_CHECK(cudaMemcpyAsync(layer_base_[l] + i * st.blob_bytes(l), st.blob(l, e), st.blob_bytes(l),
                                       cudaMemcpyHostToDevice, s));
            ++resident_;
        }
        off += slot_expert_[l].size() * st.blob_bytes(l);
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
    MoeLayerDesc d;
    d.slot_of = slot_of_dev_ + (size_t) il * n_expert_;
    d.base = layer_base_[il];
    d.blob = st_->blob_bytes(il);
    d.gate_bytes = L.gate_bytes;
    d.up_bytes = L.up_bytes;
    d.gate_type = L.gate_type;
    d.down_type = L.down_type;
    d.grow = ggml_row_size((ggml_type) L.gate_type, m_->cfg.n_embd);
    d.drow = ggml_row_size((ggml_type) L.down_type, m_->cfg.n_ff_exp);
    return d;
}

void ExpertCache::swap(int il, int slot, int expert, cudaStream_t s) {
    const int old = slot_expert_[il][slot];
    const size_t b = st_->blob_bytes(il);
    int32_t minus1 = -1, sl = slot;
    // unmap the old expert, copy, map the new one (all stream-ordered)
    CUDA_CHECK(cudaMemcpyAsync(slot_of_dev_ + (size_t) il * n_expert_ + old, &minus1, 4, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(layer_base_[il] + (size_t) slot * b, st_->blob(il, expert), b, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(slot_of_dev_ + (size_t) il * n_expert_ + expert, &sl, 4, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaStreamSynchronize(s));  // the pageable sources above must outlive the copies
    slot_of_host_[(size_t) il * n_expert_ + old] = -1;
    slot_of_host_[(size_t) il * n_expert_ + expert] = slot;
    slot_expert_[il][slot] = expert;
}

}  // namespace bnk
