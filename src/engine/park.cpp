// Parked conversations: several conversations (e.g. an agent's workers) taking turns on one engine.
//
// The engine holds one conversation's state: the history, the KV cache and indexer keys of every attention
// layer (per position), the recurrent state of the DeltaNet layers and the PLE n-gram history (not per position,
// hence the snapshots), and the drafter's own KV. A request from another conversation used to throw all of that
// away and read its prompt from the start (20-45 s at 20-50K tokens). Parking copies the live state to pinned host
// RAM (28 KiB per token: ~0.15 s each way for 50K tokens over PCIe) and brings it back when that conversation's
// next request arrives; the request then proceeds exactly as if it had never been away.
#include "engine/engine.h"

#include <algorithm>
#include <cstring>

namespace bnk {

namespace {

// How much of `prompt` a state with this history and these snapshots can keep (Generator::start's rule): all
// of the history when the prompt extends it, else the last snapshot inside the shared prefix.
template <typename C>
int reuse_of(const std::vector<int32_t> & h, const std::vector<C> & ckpts, const std::vector<int32_t> & prompt) {
    size_t common = 0;
    while (common < h.size() && common < prompt.size() && h[common] == prompt[common]) ++common;
    if (!h.empty() && common == h.size() && h.size() < prompt.size()) return (int) h.size();
    if (common == 0) return 0;
    const int lim = (int) std::min(common, prompt.size() - 1);
    int best = 0;
    for (const auto & k : ckpts)
        if (k.pos >= 0 && k.pos <= lim && k.pos <= (int) h.size()) best = std::max(best, k.pos);
    return best;
}

}  // namespace

size_t Engine::state_floats() {
    size_t floats = 0;
    each_state([&](float *, size_t n) { floats += n; });
    return floats;
}

template <typename F> void Engine::each_ctx(int cells, F && f) {
    const Config & c = model_.cfg;
    for (int il = 0; il < c.n_layer; ++il) {
        if (!kc_[il].p) continue;
        const size_t kv = (size_t) cells * c.n_head_kv * c.head_dim * sizeof(half);
        f((void *) kc_[il].p, kv);
        f((void *) vc_[il].p, kv);
        if (kraw_[il].p) {
            f((void *) kraw_[il].p, (size_t) cells * c.idx_dim * sizeof(half));
            f((void *) pooled_[il].p, (size_t) (cells / c.compress_ratio[il] + 1) * c.idx_dim * sizeof(float));
        }
    }
    if (mtp_.loaded()) mtp_.each_ctx(cells, f);
}

size_t Engine::parked_bytes() const {
    size_t b = 0;
    for (const auto & p : parked_) b += p.bytes;
    return b;
}

void Engine::drop_parked(size_t i) {
    Parked & p = parked_[i];
    for (auto & k : p.ckpts) CUDA_CHECK(cudaFreeHost(k.host));
    CUDA_CHECK(cudaFreeHost(p.host));
    parked_.erase(parked_.begin() + (long) i);
}

// Copies the live conversation into host RAM (the live state is left as it is). Never drops the entry stamped
// `keep` (one about to be restored).
void Engine::park(uint64_t keep) {
    const double t0 = now_ms();
    const int n = pos();
    // one entry per conversation: an older copy whose history this one extends is superseded
    for (size_t i = parked_.size(); i-- > 0;) {
        const auto & h = parked_[i].history;
        if (parked_[i].used != keep && h.size() <= history_.size() && std::equal(h.begin(), h.end(), history_.begin()))
            drop_parked(i);
    }
    const size_t sfloats = state_floats();
    size_t bytes = sfloats * 4;
    each_ctx(n, [&](void *, size_t b) { bytes += b; });
    size_t ck_bytes = 0;
    for (const auto & k : ckpts_) ck_bytes += k.pos >= 0 ? sfloats * 4 : 0;
    const size_t limit = (size_t) (opt_.park_gib * 1073741824.0);
    if (bytes + ck_bytes > limit) return;
    while (parked_bytes() + bytes + ck_bytes > limit) {   // least recently used first
        size_t lru = parked_.size();
        for (size_t i = 0; i < parked_.size(); ++i)
            if (parked_[i].used != keep && (lru == parked_.size() || parked_[i].used < parked_[lru].used)) lru = i;
        if (lru == parked_.size()) return;
        drop_parked(lru);
        park_stats.evictions++;
    }
    Parked p;
    if (cudaHostAlloc((void **) &p.host, bytes, 0) != cudaSuccess) {
        cudaGetLastError();
        if (opt_.verbose) fprintf(stderr, "bnk: could not pin %.2f GiB to park a conversation\n", bytes / 1073741824.0);
        return;
    }
    size_t off = 0;
    each_ctx(n, [&](void * d, size_t b) {
        CUDA_CHECK(cudaMemcpyAsync(p.host + off, d, b, cudaMemcpyDeviceToHost, st_));
        off += b;
    });
    each_state([&](float * d, size_t f) {
        CUDA_CHECK(cudaMemcpyAsync(p.host + off, d, f * 4, cudaMemcpyDeviceToHost, st_));
        off += f * 4;
    });
    for (const auto & k : ckpts_) {   // the snapshots, copied host to host meanwhile
        if (k.pos < 0) continue;
        Checkpoint c = k;
        CUDA_CHECK(cudaHostAlloc((void **) &c.host, sfloats * 4, 0));
        memcpy(c.host, k.host, sfloats * 4);
        p.ckpts.push_back(c);
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    p.history = history_;
    p.mtp_cell = mtp_cell_;
    p.bytes = bytes + ck_bytes;
    p.used = ++park_age_;
    parked_.push_back(std::move(p));
    park_stats.parks++;
    park_stats.park_ms += now_ms() - t0;
    if (opt_.verbose)
        fprintf(stderr, "bnk: parked a conversation of %d tokens (%.2f GiB) in %.0f ms; %zu parked, %.2f GiB\n", n,
                (bytes + ck_bytes) / 1073741824.0, now_ms() - t0, parked_.size(), parked_bytes() / 1073741824.0);
}

// Makes parked conversation i the live one (the live state is overwritten: park it first to keep it).
void Engine::restore(size_t i) {
    const double t0 = now_ms();
    Parked & p = parked_[i];
    const int n = (int) p.history.size();
    ensure_ctx(n + kMaxWindow);
    size_t off = 0;
    each_ctx(n, [&](void * d, size_t b) {
        CUDA_CHECK(cudaMemcpyAsync(d, p.host + off, b, cudaMemcpyHostToDevice, st_));
        off += b;
    });
    each_state([&](float * d, size_t f) {
        CUDA_CHECK(cudaMemcpyAsync(d, p.host + off, f * 4, cudaMemcpyHostToDevice, st_));
        off += f * 4;
    });
    CUDA_CHECK(cudaStreamSynchronize(st_));
    history_ = std::move(p.history);
    mtp_cell_ = p.mtp_cell;
    // its snapshots replace the live ones (the slots keep their pinned buffers)
    for (auto & k : ckpts_) k.pos = -1;
    for (auto & c : p.ckpts) {
        Checkpoint * slot = nullptr;
        for (auto & k : ckpts_)
            if (k.pos < 0) { slot = &k; break; }
        if (!slot && (int) ckpts_.size() < kMaxCheckpoints) {
            ckpts_.emplace_back();
            slot = &ckpts_.back();
            CUDA_CHECK(cudaHostAlloc((void **) &slot->host, state_floats() * 4, 0));
        }
        if (!slot) break;
        std::swap(slot->host, c.host);   // the parked entry frees the old buffer
        slot->pos = c.pos;
        slot->mtp_cell = c.mtp_cell;
        slot->age = ++ckpt_age_;
    }
    drop_parked(i);
    park_stats.restores++;
    park_stats.restore_ms += now_ms() - t0;
    if (opt_.verbose)
        fprintf(stderr, "bnk: resumed a parked conversation of %d tokens in %.0f ms; %zu parked\n", n, now_ms() - t0,
                parked_.size());
}

void Engine::select_conversation(const std::vector<int32_t> & prompt) {
    if (opt_.park_gib <= 0 || prompt.empty()) return;
    const int live = reuse_of(history_, ckpts_, prompt);
    int best = -1, best_r = live;
    for (size_t i = 0; i < parked_.size(); ++i) {
        const int r = reuse_of(parked_[i].history, parked_[i].ckpts, prompt);
        if (r > best_r) best = (int) i, best_r = r;
    }
    // the live conversation is kept when this request would throw much of it away (a parked one comes in, or
    // the live state rewinds / starts over)
    const int lost = pos() - (best < 0 ? live : 0);
    if (lost > 0 && lost >= opt_.park_min) {
        const uint64_t keep = best >= 0 ? parked_[best].used : 0;
        park(keep);
        for (size_t i = 0; i < parked_.size() && best >= 0; ++i)   // park() may have dropped others: find it again
            if (parked_[i].used == keep) best = (int) i;
    }
    if (best >= 0) restore((size_t) best);
}

}  // namespace bnk
