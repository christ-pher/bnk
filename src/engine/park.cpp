// Parked conversations: several conversations (e.g. an agent's workers) taking turns on one engine.
//
// The engine holds one conversation's state: the history, the KV cache and indexer keys of every attention
// layer (per position), the recurrent state of the DeltaNet layers and the PLE n-gram history (not per position,
// hence the snapshots), and the drafter's own KV. A request from another conversation used to throw all of that
// away and read its prompt from the start (20-45 s at 20-50K tokens). Parking copies the live state to pinned host
// RAM (28 KiB per token: ~0.15 s each way for 50K tokens over PCIe) and brings it back when that conversation's
// next request arrives; the request then proceeds exactly as if it had never been away.
#include "engine/engine.h"
#include "kernels/ops.h"

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

template <typename F> void Engine::each_ctx(int cells, F && f) { each_ctx_of(*cur_, cells, f); }

template <typename F> void Engine::each_ctx_of(Seq & q, int cells, F && f) {
    const Config & c = model_.cfg;
    for (int il = 0; il < c.n_layer; ++il) {
        if (!q.kc[il].p) continue;
        const size_t kv = (size_t) cells * c.n_head_kv * kv_row_halves(c.head_dim) * sizeof(half);
        f((void *) q.kc[il].p, kv);
        f((void *) q.vc[il].p, kv);
        if (q.kraw[il].p) {
            f((void *) q.kraw[il].p, (size_t) cells * c.idx_dim * sizeof(half));
            f((void *) q.pooled[il].p, (size_t) (cells / c.compress_ratio[il] + 1) * c.idx_dim * sizeof(float));
        }
    }
    if (mtp_.loaded()) mtp_.each_ctx_of(slot_of(&q), cells, f);
}

int Engine::slot_of(const Seq * q) const {
    for (size_t i = 0; i < seqs_.size(); ++i)
        if (seqs_[i].get() == q) return (int) i;
    throw std::runtime_error("slot_of: not a slot of this engine");
}

size_t Engine::parked_bytes() const {
    size_t b = 0;
    for (const auto & p : parked_) b += p.bytes;
    return b + (next_ ? next_->parked_bytes() : 0);
}

size_t Engine::spare_bytes() const {
    size_t b = ck_spare_.size() * const_cast<Engine *>(this)->state_floats() * 4;
    for (const auto & s : spare_) b += s.cap;
    return b + (next_ ? next_->spare_bytes() : 0);
}

// Pinned RAM handed back stays pinned for the next park while parked and spare buffers together are within the
// parking budget (pinning in a request costs ~0.2 s per GiB in the engine): conversations taking turns, and new
// ones replacing old, then park without pinning anything.
void Engine::trim_spares(size_t extra) {
    const size_t limit = (size_t) (opt_.park_gib * 1073741824.0);
    while (!spare_.empty() || ck_spare_.size() > (size_t) kMaxCheckpoints) {
        if (head_->parked_bytes() + head_->spare_bytes() + extra <= limit) return;
        if (ck_spare_.size() > (size_t) kMaxCheckpoints) {
            ck_free(ck_spare_.back());
            ck_spare_.pop_back();
            continue;
        }
        auto it = std::min_element(spare_.begin(), spare_.end(), [](const Pinned & a, const Pinned & b) { return a.cap < b.cap; });
        unpin_host(it->p, it->cap);
        spare_.erase(it);
    }
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
    trim_spares(*cap);   // none fits: the smallest spares make way within the budget
    uint8_t * p = nullptr;
    while (!(p = (uint8_t *) pin_host(*cap))) {
        if (spare_.empty()) return nullptr;
        unpin_host(spare_.back().p, spare_.back().cap);   // too little RAM to pin: give the spares back first
        spare_.pop_back();
    }
    return p;
}

void Engine::pinned_give(uint8_t * p, size_t cap) {
    spare_.push_back({p, cap});
    trim_spares();
}

float * Engine::ck_take() {
    if (!ck_spare_.empty()) {
        float * p = ck_spare_.back();
        ck_spare_.pop_back();
        return p;
    }
    return ck_alloc();
}

float * Engine::ck_alloc() {
    float * p = (float *) pin_host(state_floats() * 4);
    if (!p) throw std::runtime_error("cannot pin host RAM for a snapshot");
    return p;
}

void Engine::ck_free(float * p) {
    if (p) unpin_host(p, state_floats() * 4);
}

void Engine::ck_give(float * p) {
    if (!p) return;
    ck_spare_.push_back(p);
    trim_spares();
}

void Engine::drop_parked(size_t i) {
    Parked & p = parked_[i];
    for (auto & k : p.ckpts) ck_give(k.host);
    pinned_give(p.host, p.cap);
    parked_.erase(parked_.begin() + (long) i);
}

void Engine::drop_parked_id(uint64_t id) {
    for (Engine * s = this; s; s = s->next_.get()) {
        DeviceScope ds(s->dev_);
        for (size_t i = 0; i < s->parked_.size(); ++i)
            if (s->parked_[i].used == id) {
                s->drop_parked(i);
                break;
            }
    }
}

size_t Engine::park_bytes_here(int n) {
    size_t bytes = state_floats() * 4;
    each_ctx(n, [&](void *, size_t b) { bytes += b; });
    return bytes;
}

// Copies the live conversation into host RAM (the live state is left as it is), on every stage. Never drops the
// entry stamped `keep` (one about to be restored).
void Engine::park(uint64_t keep, bool move_ckpts) {
    join_commit();
    const double t0 = now_ms();
    const int n = pos();
    // one entry per conversation: an older copy whose history this one extends is superseded
    for (size_t i = parked_.size(); i-- > 0;) {
        const auto & h = parked_[i].history;
        if (parked_[i].used != keep && h.size() <= cur_->history.size() && std::equal(h.begin(), h.end(), cur_->history.begin()))
            drop_parked_id(parked_[i].used);
    }
    size_t bytes = 0, ck_bytes = 0;
    for (Engine * s = this; s; s = s->next_.get()) {
        DeviceScope ds(s->dev_);
        bytes += s->park_bytes_here(n);
        const size_t sfloats = s->state_floats();
        for (const auto & k : s->cur_->ckpts) ck_bytes += k.pos >= 0 ? sfloats * 4 : 0;
    }
    const size_t limit = (size_t) (opt_.park_gib * 1073741824.0);
    if (bytes + ck_bytes > limit) return;
    while (parked_bytes() + bytes + ck_bytes > limit) {   // least recently used first
        size_t lru = parked_.size();
        for (size_t i = 0; i < parked_.size(); ++i)
            if (parked_[i].used != keep && (lru == parked_.size() || parked_[i].used < parked_[lru].used)) lru = i;
        if (lru == parked_.size()) return;
        drop_parked_id(parked_[lru].used);
        park_stats.evictions++;
    }
    const uint64_t id = ++park_age_;
    // every stage's copies in flight together (each GPU on its own link), then waited for
    bool ok = true;
    for (Engine * s = this; s && ok; s = s->next_.get()) ok = s->park_copy(id, move_ckpts);
    for (Engine * s = this; s; s = s->next_.get()) {
        DeviceScope ds(s->dev_);
        CUDA_CHECK(cudaStreamSynchronize(s->st_));
    }
    if (!ok) {
        drop_parked_id(id);   // the stages that did park it
        if (opt_.verbose) fprintf(stderr, "bnk: could not pin %.2f GiB to park a conversation\n", bytes / 1073741824.0);
        return;
    }
    park_stats.parks++;
    park_stats.park_ms += now_ms() - t0;
    if (opt_.verbose)
        fprintf(stderr, "bnk: parked a conversation of %d tokens (%.2f GiB) in %.0f ms; %zu parked, %.2f GiB\n", n,
                (bytes + ck_bytes) / 1073741824.0, now_ms() - t0, parked_.size(), parked_bytes() / 1073741824.0);
}

// This stage's copy of the live conversation, entry `id` (false: no pinned memory for it). The device-to-host
// copies are left in flight on st_: the caller synchronizes.
bool Engine::park_copy(uint64_t id, bool move_ckpts) {
    DeviceScope ds(dev_);
    join_commit_here();
    const int n = pos();
    const size_t sfloats = state_floats();
    const size_t bytes = park_bytes_here(n);
    size_t ck_bytes = 0;
    for (const auto & k : cur_->ckpts) ck_bytes += k.pos >= 0 ? sfloats * 4 : 0;
    Parked p;
    p.host = pinned_take(bytes, &p.cap);
    if (!p.host) return false;
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
    p.history = cur_->history;
    p.mtp_cell = cur_->mtp_cell;
    p.bytes = bytes + ck_bytes;
    p.used = id;
    parked_.push_back(std::move(p));
    return true;
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
    for (Engine * s = this; s; s = s->next_.get()) s->restore_copy(id, false);   // every stage's copies in flight
    for (Engine * s = this; s; s = s->next_.get()) s->restore_copy(id, true);
    park_stats.restores++;
    park_stats.restore_ms += now_ms() - t0;
    if (opt_.verbose)
        fprintf(stderr, "bnk: resumed a parked conversation of %d tokens in %.0f ms; %zu parked\n", n, now_ms() - t0,
                parked_.size());
}

// finish false: starts the host-to-device copies; true: waits for them and makes the entry the live state
void Engine::restore_copy(uint64_t id, bool finish) {
    DeviceScope ds(dev_);
    size_t i = parked_.size();
    for (size_t j = 0; j < parked_.size(); ++j)
        if (parked_[j].used == id) i = j;
    if (i == parked_.size()) throw std::runtime_error("restore: a pipeline stage lost the parked conversation");
    Parked & p = parked_[i];
    if (!finish) {
        const int n = (int) p.history.size();
        size_t off = 0;
        each_ctx(n, [&](void * d, size_t b) {
            CUDA_CHECK(cudaMemcpyAsync(d, p.host + off, b, cudaMemcpyHostToDevice, st_));
            off += b;
        });
        each_state([&](float * d, size_t f) {
            CUDA_CHECK(cudaMemcpyAsync(d, p.host + off, f * 4, cudaMemcpyHostToDevice, st_));
            off += f * 4;
        });
        return;
    }
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
}

void Engine::set_busy(const std::vector<int> & slots) {
    busy_.assign(seqs_.size(), 0);
    for (int s : slots)
        if (s >= 0 && s < (int) busy_.size()) busy_[s] = 1;
}

bool Engine::evict_idle(size_t need) { return head_->evict_idle_for(this, need); }

// Room for `need` bytes on `stage`: idle conversations go to host RAM (parked when worth it) on every stage, the
// one using the most context on that stage first.
bool Engine::evict_idle_for(Engine * stage, size_t need) {
    // the conversation that stage is making room for stays; every stage's current one is restored afterwards (a
    // stage preparing a forward has its window's conversation current, which may not be the first stage's)
    const int keep = stage->current_slot();
    std::vector<Seq *> was;
    for (Engine * s = this; s; s = s->next_.get()) was.push_back(s->cur_);
    auto room = [&]() {
        DeviceScope ds(stage->dev_);
        return stage->budget_.make_room(need);
    };
    for (;;) {
        if (room()) break;
        int bi = -1;
        for (int i = 0; i < (int) stage->seqs_.size(); ++i) {
            const Seq * q = stage->seqs_[i].get();
            if (i == keep || !q->used || (i < (int) busy_.size() && busy_[i]) || q->ctx_mapped <= kCtxBaseline) continue;
            if (bi < 0 || q->ctx_mapped > stage->seqs_[bi]->ctx_mapped) bi = i;
        }
        if (bi < 0) break;
        for (Engine * s = this; s; s = s->next_.get()) s->use(s->seqs_[bi].get());
        const int n = pos();
        const size_t before = parked_.size();
        if (opt_.park_gib > 0 && n >= opt_.park_min) park(restoring_);
        if (opt_.verbose)
            fprintf(stderr, "bnk: VRAM is short%s: moved an idle conversation of %d tokens off the GPU (slot %d, %s)\n",
                    next_ ? (" on GPU " + std::to_string(stage->dev_)).c_str() : "", n, bi,
                    parked_.size() > before ? "parked in RAM" : "not parked: its next turn reads it again");
        for (Engine * s = this; s; s = s->next_.get()) s->reset_here();
    }
    size_t k = 0;
    for (Engine * s = this; s; s = s->next_.get()) s->use(was[k++]);
    return room();
}

int Engine::reusable(const std::vector<int32_t> & prompt) const {
    return reuse_of(cur_->history, cur_->ckpts, prompt);
}

static int fork_min_env(int dflt) {
    static const int v = getenv("BNK_FORK_MIN") ? atoi(getenv("BNK_FORK_MIN")) : dflt;
    return v;
}

bool Engine::can_fork() const { return fork_min_env(fork_min_) > 0; }

namespace {

// the prompt continues this history (every token of it, and more)
bool extends(const std::vector<int32_t> & h, const std::vector<int32_t> & prompt) {
    return !h.empty() && h.size() < prompt.size() && std::equal(h.begin(), h.end(), prompt.begin());
}

}  // namespace

// The state this request starts from: the live slot's own (Generator::start continues or rewinds it), a parked
// conversation it continues (restored), or the prefix another conversation shares with it, parked or on the
// GPU (forked: copied in, the other conversation untouched). The live conversation is parked first when the
// request would throw much of it away.
void Engine::select_conversation(const std::vector<int32_t> & prompt) {
    if (prompt.empty()) return;
    const int fork_min = fork_min_env(fork_min_);
    const int live = reuse_of(cur_->history, cur_->ckpts, prompt);
    int bestp = -1, bestp_r = live;
    bool bestp_ext = false;
    if (opt_.park_gib > 0)
        for (size_t i = 0; i < parked_.size(); ++i) {
            const int r = reuse_of(parked_[i].history, parked_[i].ckpts, prompt);
            if (r > bestp_r) bestp = (int) i, bestp_r = r, bestp_ext = extends(parked_[i].history, prompt);
        }
    int bests = -1, bests_r = std::max(live, bestp_r);
    if (fork_min > 0)
        for (size_t i = 0; i < seqs_.size(); ++i) {
            Seq * q = seqs_[i].get();
            if (q == cur_ || !q->used || (i < busy_.size() && busy_[i]) || extends(q->history, prompt)) continue;
            const int r = reuse_of(q->history, q->ckpts, prompt);   // a snapshot inside the shared prefix
            if (r > bests_r) bests = (int) i, bests_r = r;
        }
    const bool fork_s = bests >= 0 && bests_r >= fork_min;
    const bool take_p = !fork_s && bestp >= 0 && (bestp_ext || (fork_min > 0 && bestp_r >= fork_min));
    const bool replace = fork_s || take_p;   // else the live state is the best start (Generator::start continues or rewinds it)
    // the live conversation is kept when this request would throw much of it away: all of it when another comes
    // in, the part past the reusable prefix when it rewinds or starts over
    const uint64_t keep = take_p ? parked_[bestp].used : 0;
    const int lost = replace ? pos() : pos() - live;
    if (lost > 0 && lost >= opt_.park_min && opt_.park_gib > 0) {
        park(keep, replace);   // the live snapshots go along when the live state is replaced below
        for (size_t i = 0; i < parked_.size() && take_p; ++i)   // park() may have dropped others: find it again
            if (parked_[i].used == keep) bestp = (int) i;
    }
    if (!replace) return;
    if (fork_s) fork_slot(bests, bests_r);
    else if (bestp_ext) restore((size_t) bestp);
    else fork_parked(keep, bestp_r);
}

void Engine::fork_parked(uint64_t id, int pos) {
    join_commit();
    const double t0 = now_ms();
    restoring_ = id;   // evictions while room is made keep the source
    try {
        ensure_ctx(pos + kMaxWindow);
    } catch (...) {
        restoring_ = 0;
        throw;
    }
    restoring_ = 0;
    for (Engine * s = this; s; s = s->next_.get()) s->fork_copy_parked(id, pos, false);
    for (Engine * s = this; s; s = s->next_.get()) s->fork_copy_parked(id, pos, true);
    checkpoint();   // the prefix is a snapshot of the new conversation too (its own rewinds and forks)
    park_stats.forks++;
    park_stats.fork_ms += now_ms() - t0;
    if (opt_.verbose)
        fprintf(stderr, "bnk: forked a conversation from a parked one's first %d tokens in %.0f ms\n", pos, now_ms() - t0);
}

void Engine::fork_slot(int src, int pos) {
    join_commit();
    const double t0 = now_ms();
    std::vector<char> was = busy_;   // the source stays on the GPU while room is made
    if (busy_.size() < seqs_.size()) busy_.resize(seqs_.size(), 0);
    busy_[src] = 1;
    try {
        ensure_ctx(pos + kMaxWindow);
    } catch (...) {
        busy_ = was;
        throw;
    }
    busy_ = was;
    for (Engine * s = this; s; s = s->next_.get()) s->fork_copy_slot(src, pos, false);
    for (Engine * s = this; s; s = s->next_.get()) s->fork_copy_slot(src, pos, true);
    checkpoint();
    park_stats.forks++;
    park_stats.fork_ms += now_ms() - t0;
    if (opt_.verbose)
        fprintf(stderr, "bnk: forked a conversation from slot %d's first %d tokens in %.0f ms\n", src, pos, now_ms() - t0);
}

// This stage's part of a fork from parked entry `id` (finish false: the copies, left in flight on st_; true:
// waited for, the live conversation becomes the prefix)
void Engine::fork_copy_parked(uint64_t id, int pos, bool finish) {
    DeviceScope ds(dev_);
    size_t i = parked_.size();
    for (size_t j = 0; j < parked_.size(); ++j)
        if (parked_[j].used == id) i = j;
    if (i == parked_.size()) throw std::runtime_error("fork: a pipeline stage lost the parked conversation");
    Parked & p = parked_[i];
    const Checkpoint * c = nullptr;
    for (const auto & k : p.ckpts)
        if (k.pos == pos) c = &k;
    if (!c) throw std::runtime_error("fork: the parked conversation has no snapshot at " + std::to_string(pos));
    if (!finish) {
        join_commit_here();
        const int n = (int) p.history.size();
        std::vector<size_t> full;   // the entry's layout: every buffer at its full length
        each_ctx(n, [&](void *, size_t b) { full.push_back(b); });
        size_t off = 0, k = 0;
        each_ctx(pos, [&](void * d, size_t b) {
            CUDA_CHECK(cudaMemcpyAsync(d, p.host + off, b, cudaMemcpyHostToDevice, st_));
            off += full[k++];
        });
        size_t so = 0;
        each_state([&](float * d, size_t f) {
            CUDA_CHECK(cudaMemcpyAsync(d, c->host + so, f * 4, cudaMemcpyHostToDevice, st_));
            so += f;
        });
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    fork_done_here(p.history, pos, c->mtp_cell);
}

// The same from an idle slot on this GPU (device-to-device copies)
void Engine::fork_copy_slot(int src, int pos, bool finish) {
    DeviceScope ds(dev_);
    Seq & q = *seqs_.at(src);
    const Checkpoint * c = nullptr;
    for (const auto & k : q.ckpts)
        if (k.pos == pos) c = &k;
    if (!c) throw std::runtime_error("fork: slot " + std::to_string(src) + " has no snapshot at " + std::to_string(pos));
    if (!finish) {
        join_commit_here();
        std::vector<void *> dst;
        each_ctx(pos, [&](void * d, size_t) { dst.push_back(d); });
        size_t k = 0;
        each_ctx_of(q, pos, [&](void * s, size_t b) {
            CUDA_CHECK(cudaMemcpyAsync(dst.at(k++), s, b, cudaMemcpyDeviceToDevice, st_));
        });
        size_t so = 0;
        each_state([&](float * d, size_t f) {
            CUDA_CHECK(cudaMemcpyAsync(d, c->host + so, f * 4, cudaMemcpyHostToDevice, st_));
            so += f;
        });
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    fork_done_here(q.history, pos, c->mtp_cell);
}

void Engine::fork_done_here(const std::vector<int32_t> & h, int pos, int mtp_cell) {
    cur_->history.assign(h.begin(), h.begin() + pos);
    cur_->mtp_cell = mtp_cell;
    cur_->pending_T = 0;
    for (auto & k : cur_->ckpts) k.pos = -1;
}

}  // namespace bnk
