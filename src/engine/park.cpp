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
        if (!cur_->kc[il].p) continue;
        const size_t kv = (size_t) cells * c.n_head_kv * c.head_dim * sizeof(half);
        f((void *) cur_->kc[il].p, kv);
        f((void *) cur_->vc[il].p, kv);
        if (cur_->kraw[il].p) {
            f((void *) cur_->kraw[il].p, (size_t) cells * c.idx_dim * sizeof(half));
            f((void *) cur_->pooled[il].p, (size_t) (cells / c.compress_ratio[il] + 1) * c.idx_dim * sizeof(float));
        }
    }
    if (mtp_.loaded()) mtp_.each_ctx(cells, f);
}

size_t Engine::parked_bytes() const {
    size_t b = 0;
    for (const auto & p : parked_) b += p.bytes;
    return b;
}

uint8_t * Engine::pinned_take(size_t bytes, size_t * cap) {
    size_t best = spare_.size();
    for (size_t i = 0; i < spare_.size(); ++i)   // the smallest that fits
        if (spare_[i].cap >= bytes && (best == spare_.size() || spare_[i].cap < spare_[best].cap)) best = i;
    if (best < spare_.size()) {
        uint8_t * p = spare_[best].p;
        *cap = spare_[best].cap;
        spare_.erase(spare_.begin() + (long) best);
        return p;
    }
    // room for the conversation to grow before the buffer stops fitting it
    const size_t step = 256ull << 20;
    *cap = (bytes + bytes / 4 + step - 1) / step * step;
    uint8_t * p = nullptr;
    while (cudaHostAlloc((void **) &p, *cap, 0) != cudaSuccess) {
        cudaGetLastError();
        if (spare_.empty()) return nullptr;
        CUDA_CHECK(cudaFreeHost(spare_.back().p));   // too little RAM to pin: give the spares back first
        spare_.pop_back();
    }
    return p;
}

void Engine::pinned_give(uint8_t * p, size_t cap) {
    spare_.push_back({p, cap});
    if (spare_.size() > 2) {   // keep the two biggest
        auto it = std::min_element(spare_.begin(), spare_.end(), [](const Pinned & a, const Pinned & b) { return a.cap < b.cap; });
        CUDA_CHECK(cudaFreeHost(it->p));
        spare_.erase(it);
    }
}

float * Engine::ck_take() {
    if (!ck_spare_.empty()) {
        float * p = ck_spare_.back();
        ck_spare_.pop_back();
        return p;
    }
    float * p = nullptr;
    CUDA_CHECK(cudaHostAlloc((void **) &p, state_floats() * 4, 0));
    return p;
}

void Engine::ck_give(float * p) {
    if (!p) return;
    if (ck_spare_.size() < (size_t) kMaxCheckpoints) ck_spare_.push_back(p);
    else CUDA_CHECK(cudaFreeHost(p));
}

void Engine::drop_parked(size_t i) {
    Parked & p = parked_[i];
    for (auto & k : p.ckpts) ck_give(k.host);
    pinned_give(p.host, p.cap);
    parked_.erase(parked_.begin() + (long) i);
}

// Copies the live conversation into host RAM (the live state is left as it is). Never drops the entry stamped
// `keep` (one about to be restored).
void Engine::park(uint64_t keep, bool move_ckpts) {
    join_commit();
    const double t0 = now_ms();
    const int n = pos();
    // one entry per conversation: an older copy whose history this one extends is superseded
    for (size_t i = parked_.size(); i-- > 0;) {
        const auto & h = parked_[i].history;
        if (parked_[i].used != keep && h.size() <= cur_->history.size() && std::equal(h.begin(), h.end(), cur_->history.begin()))
            drop_parked(i);
    }
    const size_t sfloats = state_floats();
    size_t bytes = sfloats * 4;
    each_ctx(n, [&](void *, size_t b) { bytes += b; });
    size_t ck_bytes = 0;
    for (const auto & k : cur_->ckpts) ck_bytes += k.pos >= 0 ? sfloats * 4 : 0;
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
    p.host = pinned_take(bytes, &p.cap);
    if (!p.host) {
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
    for (auto & k : cur_->ckpts) {   // the snapshots: moved, or copied host to host meanwhile
        if (k.pos < 0) continue;
        Checkpoint c = k;
        if (move_ckpts) {
            k.host = ck_take();
            k.pos = -1;
        } else {
            c.host = ck_take();
            memcpy(c.host, k.host, sfloats * 4);
        }
        p.ckpts.push_back(c);
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    p.history = cur_->history;
    p.mtp_cell = cur_->mtp_cell;
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
    join_commit();
    const double t0 = now_ms();
    // making room may park idle conversations (evict_idle): that may drop other entries and move this one
    const uint64_t id = parked_[i].used;
    const int n = (int) parked_[i].history.size();
    restoring_ = id;
    try {
        ensure_ctx(n + kMaxWindow);
    } catch (...) {
        restoring_ = 0;
        throw;
    }
    restoring_ = 0;
    for (size_t j = 0; j < parked_.size(); ++j)
        if (parked_[j].used == id) i = j;
    Parked & p = parked_[i];
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
    cur_->history = std::move(p.history);
    cur_->mtp_cell = p.mtp_cell;
    // its snapshots replace the live ones (the slots keep their pinned buffers)
    for (auto & k : cur_->ckpts) k.pos = -1;
    for (auto & c : p.ckpts) {
        Checkpoint * slot = nullptr;
        for (auto & k : cur_->ckpts)
            if (k.pos < 0) { slot = &k; break; }
        if (!slot && (int) cur_->ckpts.size() < kMaxCheckpoints) {
            cur_->ckpts.emplace_back();
            slot = &cur_->ckpts.back();
            slot->host = ck_take();
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

void Engine::set_busy(const std::vector<int> & slots) {
    busy_.assign(seqs_.size(), 0);
    for (int s : slots)
        if (s >= 0 && s < (int) busy_.size()) busy_[s] = 1;
}

bool Engine::evict_idle(size_t need) {
    Seq * keep = cur_;
    for (;;) {
        if (budget_.make_room(need)) break;
        int bi = -1;
        for (int i = 0; i < (int) seqs_.size(); ++i) {
            const Seq * q = seqs_[i].get();
            if (q == keep || !q->used || (i < (int) busy_.size() && busy_[i]) || q->ctx_mapped <= kCtxBaseline) continue;
            if (bi < 0 || q->ctx_mapped > seqs_[bi]->ctx_mapped) bi = i;
        }
        if (bi < 0) break;
        use(seqs_[bi].get());
        const int n = pos();
        const size_t before = parked_.size();
        if (opt_.park_gib > 0 && n >= opt_.park_min) park(restoring_);
        if (opt_.verbose)
            fprintf(stderr, "bnk: VRAM is short: moved an idle conversation of %d tokens off the GPU (slot %d, %s)\n", n, bi,
                    parked_.size() > before ? "parked in RAM" : "not parked: its next turn reads it again");
        reset();
    }
    use(keep);
    return budget_.make_room(need);
}

int Engine::reusable(const std::vector<int32_t> & prompt) const {
    return reuse_of(cur_->history, cur_->ckpts, prompt);
}

void Engine::select_conversation(const std::vector<int32_t> & prompt) {
    if (opt_.park_gib <= 0 || prompt.empty()) return;
    const int live = reuse_of(cur_->history, cur_->ckpts, prompt);
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
        park(keep, best >= 0);
        for (size_t i = 0; i < parked_.size() && best >= 0; ++i)   // park() may have dropped others: find it again
            if (parked_[i].used == keep) best = (int) i;
    }
    if (best >= 0) restore((size_t) best);
}

}  // namespace bnk
