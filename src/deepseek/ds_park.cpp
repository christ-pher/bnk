// Parked conversations on the DeepSeek engine (docs/DS-PARK-PLAN.md): several conversations (an agent's workers,
// two agents) taking turns on one slot.
//
// A conversation's state on a stage is its history, the per-position rows of every kv-source layer (the compressed
// latents and the indexer keys: pos / ratio rows each, under 1 GB at 256K tokens), the window rings and the
// compressor's open groups (not per position: each_snap, which the snapshots hold too), its snapshots, and on the
// stage with the DSpark drafter that drafter's rings. Without parking a request from another conversation threw all
// of it away and read its prompt from the start (190K tokens: 260-276 s). Parking copies it to pinned host RAM and
// brings it back when the conversation's next request arrives; the request then proceeds exactly as if it had never
// been away. The design is the Qwen engine's (src/engine/park.cpp) without slot forking (this engine runs one slot).
#include "deepseek/ds_engine.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <sys/stat.h>

namespace bnk {

namespace {

// How much of `prompt` a state with this history and these snapshots can keep (Generator::start's rule): all of the
// history when the prompt extends it, else the last snapshot inside the shared prefix.
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

// the prompt continues this history (every token of it, and more)
bool extends(const std::vector<int32_t> & h, const std::vector<int32_t> & prompt) {
    return !h.empty() && h.size() < prompt.size() && std::equal(h.begin(), h.end(), prompt.begin());
}

// MemAvailable in bytes (0 when unknown)
size_t mem_available() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
    fclose(f);
    return kb * 1024;
}

// RAM a park must leave free: the Engram tables are read through the page cache, and the expert pin already holds
// most of the machine (BNK_DS_PARK_FREE_GIB, default 16)
size_t park_free_floor() {
    static const size_t v = (size_t) (env_double("BNK_DS_PARK_FREE_GIB", 16.0) * 1073741824.0);
    return v;
}

int fork_min() {
    static const int v = env_int("BNK_FORK_MIN", 64);
    return v;
}

}  // namespace

template <typename F> void DsEngine::each_ctx(int n, F && f) {
    const DsConfig & c = model_.ds;
    for (int il = l0_; il < l1_; ++il) {
        if (!c.is_kv_src(il)) continue;
        const size_t rows = (size_t) n / std::max(1, c.ratio[il]) + 1;   // the rows positions [0, n) wrote
        f((void *) cur_->ckc[il].p, rows * c.head_dim * sizeof(half));
        f((void *) cur_->ikc[il].p, rows * c.idx_dim * sizeof(half));
    }
}

size_t DsEngine::park_bytes_here(int n) {
    size_t b = snap_bytes();
    each_ctx(n, [&](void *, size_t k) { b += k; });
    if (spark_) spark_->each_ring([&](void *, size_t k) { b += k; });
    return b;
}

size_t DsEngine::parked_bytes() const {
    size_t b = 0;
    for (const DsEngine * s = this; s; s = s->next_.get())
        for (const auto & p : s->parked_) b += p.bytes;
    return b;
}

// a pinned buffer of at least `bytes` (a spare one when it fits), nullptr when RAM is short
uint8_t * DsEngine::pinned_take(size_t bytes, size_t * cap) {
    size_t best = pin_spare_.size();
    for (size_t i = 0; i < pin_spare_.size(); ++i)
        if (pin_spare_[i].second >= bytes && (best == pin_spare_.size() || pin_spare_[i].second < pin_spare_[best].second)) best = i;
    if (best < pin_spare_.size()) {
        uint8_t * p = pin_spare_[best].first;
        *cap = pin_spare_[best].second;
        pin_spare_.erase(pin_spare_.begin() + (long) best);
        return p;
    }
    const size_t want = (bytes + (64ull << 20) - 1) / (64ull << 20) * (64ull << 20);
    const size_t avail = mem_available();
    if (avail && avail < want + park_free_floor()) {
        // spares that do not fit go back first
        for (auto & s : pin_spare_) cudaFreeHost(s.first);
        pin_spare_.clear();
        if (mem_available() < want + park_free_floor()) return nullptr;
    }
    uint8_t * p = nullptr;
    if (cudaHostAlloc((void **) &p, want, cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    *cap = want;
    return p;
}

// keeps two spares (the buffers two alternating conversations hand back and forth), frees the rest
void DsEngine::pinned_give(uint8_t * p, size_t cap) {
    if (!p) return;
    pin_spare_.emplace_back(p, cap);
    while (pin_spare_.size() > 2) {
        auto it = std::min_element(pin_spare_.begin(), pin_spare_.end(), [](auto & a, auto & b) { return a.second < b.second; });
        cudaFreeHost(it->first);
        pin_spare_.erase(it);
    }
}

uint8_t * DsEngine::ck_take() {
    if (!ck_spare_.empty()) {
        uint8_t * p = ck_spare_.back();
        ck_spare_.pop_back();
        return p;
    }
    uint8_t * p = nullptr;
    CUDA_CHECK(cudaHostAlloc((void **) &p, std::max<size_t>(snap_bytes(), 64), cudaHostAllocPortable));
    return p;
}

void DsEngine::ck_give(uint8_t * p) {
    if (!p) return;
    if (ck_spare_.size() < (size_t) kMaxCheckpoints) ck_spare_.push_back(p);
    else cudaFreeHost(p);
}

int DsEngine::find_parked(uint64_t id) const {
    for (size_t i = 0; i < parked_.size(); ++i)
        if (parked_[i].used == id) return (int) i;
    return -1;
}

void DsEngine::drop_parked(size_t i) {
    Parked & p = parked_[i];
    for (auto & k : p.ckpts) ck_give(k.host);
    pinned_give(p.host, p.cap);
    parked_.erase(parked_.begin() + (long) i);
}

void DsEngine::drop_parked_id(uint64_t id) {
    for (DsEngine * s = this; s; s = s->next_.get()) {
        DeviceScope ds(s->dev_);
        const int i = s->find_parked(id);
        if (i >= 0) s->drop_parked((size_t) i);
    }
}

void DsEngine::drop_parked_all() {
    for (DsEngine * s = this; s; s = s->next_.get()) {
        DeviceScope ds(s->dev_);
        while (!s->parked_.empty()) s->drop_parked(s->parked_.size() - 1);
        for (auto & b : s->pin_spare_) cudaFreeHost(b.first);
        s->pin_spare_.clear();
    }
    disk_drop_all();   // a clean slate: the disk states go too
}

// Copies the live conversation into host RAM (the live state is left as it is) on every stage. Never drops the entry
// stamped `keep` (one about to be restored). move_ckpts: the live snapshots go along instead of being copied (the
// live state is replaced next).
void DsEngine::park(uint64_t keep, bool move_ckpts) {
    join_commit();
    const double t0 = now_ms();
    const int n = pos();
    if (n == 0) return;
    // one entry per conversation: an older copy whose history this one extends is superseded
    for (size_t i = parked_.size(); i-- > 0;) {
        const auto & h = parked_[i].history;
        if (parked_[i].used != keep && h.size() <= cur_->history.size() && std::equal(h.begin(), h.end(), cur_->history.begin()))
            drop_parked_id(parked_[i].used);
    }
    size_t bytes = 0;
    for (DsEngine * s = this; s; s = s->next_.get()) {
        DeviceScope ds(s->dev_);
        bytes += s->park_bytes_here(n);
        for (const auto & k : s->cur_->ckpts) bytes += k.pos >= 0 ? s->snap_bytes() : 0;
    }
    const size_t limit = (size_t) (opt_.park_gib * 1073741824.0);
    if (bytes > limit) return;
    while (parked_bytes() + bytes > limit) {   // least recently used first
        int lru = -1;
        for (size_t i = 0; i < parked_.size(); ++i)
            if (parked_[i].used != keep && (lru < 0 || parked_[i].used < parked_[lru].used)) lru = (int) i;
        if (lru < 0) return;
        if (!park_dir_.empty()) disk_write(parked_[lru].used, true);   // out of RAM onto disk
        else drop_parked_id(parked_[lru].used);
        park_stats.evictions++;
    }
    const uint64_t id = ++park_age_;
    bool ok = true;   // every stage's copies in flight together (each GPU on its own link), then waited for
    for (DsEngine * s = this; s && ok; s = s->next_.get()) ok = s->park_copy(id, move_ckpts);
    for (DsEngine * s = this; s; s = s->next_.get()) {
        DeviceScope ds(s->dev_);
        CUDA_CHECK(cudaStreamSynchronize(s->st_));
    }
    if (!ok) {
        drop_parked_id(id);   // the stages that did park it
        if (opt_.verbose) fprintf(stderr, "bnk: could not pin %.2f GiB to park a conversation (RAM short)\n", bytes / 1073741824.0);
        return;
    }
    park_stats.parks++;
    park_stats.park_ms += now_ms() - t0;
    if (opt_.verbose)
        fprintf(stderr, "bnk: parked a conversation of %d tokens (%.2f GiB) in %.0f ms; %zu parked, %.2f GiB\n", n,
                bytes / 1073741824.0, now_ms() - t0, parked_.size(), parked_bytes() / 1073741824.0);
}

// This stage's copy of the live conversation as entry `id` (false: no pinned memory for it). The device-to-host
// copies are left in flight on st_: the caller synchronizes.
bool DsEngine::park_copy(uint64_t id, bool move_ckpts) {
    DeviceScope ds(dev_);
    join_commit_here();
    const int n = pos();
    const size_t bytes = park_bytes_here(n);
    Parked p;
    p.host = pinned_take(bytes, &p.cap);
    if (!p.host) return false;
    size_t off = 0;
    auto out = [&](void * d, size_t b) {
        CUDA_CHECK(cudaMemcpyAsync(p.host + off, d, b, cudaMemcpyDeviceToHost, st_));
        off += b;
    };
    each_ctx(n, out);
    each_snap(out);
    if (spark_) {
        spark_->each_ring(out);
        spark_->fed(&p.spark_lo, &p.spark_hi);
    }
    const size_t sb = snap_bytes();
    size_t ck_bytes = 0;
    for (auto & k : cur_->ckpts) {   // the snapshots: moved, or copied host to host meanwhile
        if (k.pos < 0) continue;
        Checkpoint c = k;
        if (move_ckpts) {
            k.host = ck_take();
            k.pos = -1;
        } else {
            c.host = ck_take();
            memcpy(c.host, k.host, sb);
        }
        p.ckpts.push_back(c);
        ck_bytes += sb;
    }
    p.history = cur_->history;
    p.bytes = bytes + ck_bytes;
    p.used = id;
    parked_.push_back(std::move(p));
    return true;
}

// BNK_DS_PARK_POISON=1 (tools/ds_park_check.py --poison): before a restore or a fork, every buffer of the live
// slot's state is filled with NaN (all bits set, fp16 and fp32 alike), so a part the restore does not bring back,
// or a kernel reading past what is valid, turns the logits into NaN instead of hiding in the run-to-run noise.
static bool poison_on() {
    static const bool v = env_int("BNK_DS_PARK_POISON", 0) != 0;
    return v;
}

void DsEngine::poison_here() {
    DeviceScope ds(dev_);
    for (int il = l0_; il < l1_; ++il) {
        if (cur_->ckv_buf[il].mapped()) CUDA_CHECK(cudaMemsetAsync(cur_->ckv_buf[il].ptr(), 0xff, cur_->ckv_buf[il].mapped(), st_));
        if (cur_->ikc_buf[il].mapped()) CUDA_CHECK(cudaMemsetAsync(cur_->ikc_buf[il].ptr(), 0xff, cur_->ikc_buf[il].mapped(), st_));
    }
    each_snap([&](void * d, size_t b) { CUDA_CHECK(cudaMemsetAsync(d, 0xff, b, st_)); });
    if (spark_) spark_->each_ring([&](void * d, size_t b) { CUDA_CHECK(cudaMemsetAsync(d, 0xff, b, st_)); });
}

// Makes parked conversation `id` the live one (the live state is overwritten: park it first to keep it).
void DsEngine::restore(uint64_t id) {
    join_commit();
    const double t0 = now_ms();
    const int n = (int) parked_[find_parked(id)].history.size();
    ensure_ctx(n + kMaxWindow);
    if (poison_on()) for (DsEngine * s = this; s; s = s->next_.get()) s->poison_here();
    for (DsEngine * s = this; s; s = s->next_.get()) s->restore_copy(id, false);   // every stage's copies in flight
    for (DsEngine * s = this; s; s = s->next_.get()) s->restore_copy(id, true);
    park_stats.restores++;
    park_stats.restore_ms += now_ms() - t0;
    if (opt_.verbose)
        fprintf(stderr, "bnk: resumed a parked conversation of %d tokens in %.0f ms; %zu parked\n", n, now_ms() - t0,
                parked_.size());
}

// finish false: starts the host-to-device copies; true: waits for them and makes the entry the live state
void DsEngine::restore_copy(uint64_t id, bool finish) {
    DeviceScope ds(dev_);
    const int i = find_parked(id);
    if (i < 0) throw std::runtime_error("restore: a pipeline stage lost the parked conversation");
    Parked & p = parked_[i];
    if (!finish) {
        size_t off = 0;
        auto in = [&](void * d, size_t b) {
            CUDA_CHECK(cudaMemcpyAsync(d, p.host + off, b, cudaMemcpyHostToDevice, st_));
            off += b;
        };
        each_ctx((int) p.history.size(), in);
        // BNK_DS_PARK_BREAK=1 (tools/ds_park_check.py's control): the open groups are not brought back, to show what
        // a restore that misses part of the state looks like
        static const bool brk = env_int("BNK_DS_PARK_BREAK", 0) != 0;
        if (brk) {
            each_state([&](float *, size_t f) { off += f * 4; });
            for (int il = l0_; il < l1_; ++il) in(cur_->ring[il].p, cur_->ring[il].n * sizeof(half));
        } else
            each_snap(in);
        if (spark_) spark_->each_ring(in);
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    cur_->history = std::move(p.history);
    cur_->pending_T = 0;
    if (spark_) spark_->set_fed(p.spark_lo, p.spark_hi);
    // its snapshots replace the live ones (the slots keep their pinned buffers)
    for (auto & k : cur_->ckpts) k.pos = -1;
    size_t j = 0;
    for (auto & c : p.ckpts) {
        if (j >= cur_->ckpts.size()) break;
        Checkpoint & slot = cur_->ckpts[j++];
        std::swap(slot.host, c.host);   // the parked entry frees the old buffer
        slot.pos = c.pos;
        slot.age = ++ckpt_age_;
    }
    drop_parked((size_t) i);
}

// The first `pos` tokens of parked conversation `id` become the live state (pos: one of its snapshots, or its whole
// history); the entry stays. Another conversation that shares a long prefix with it (the same system prompt and
// repository dump) starts from there instead of reading it again.
void DsEngine::fork_parked(uint64_t id, int pos) {
    join_commit();
    const double t0 = now_ms();
    ensure_ctx(pos + kMaxWindow);
    if (poison_on()) for (DsEngine * s = this; s; s = s->next_.get()) s->poison_here();
    for (DsEngine * s = this; s; s = s->next_.get()) s->fork_copy(id, pos, false);
    for (DsEngine * s = this; s; s = s->next_.get()) s->fork_copy(id, pos, true);
    park_stats.forks++;
    park_stats.fork_ms += now_ms() - t0;
    if (opt_.verbose) fprintf(stderr, "bnk: took %d tokens from a parked conversation in %.0f ms\n", pos, now_ms() - t0);
}

void DsEngine::fork_copy(uint64_t id, int pos, bool finish) {
    DeviceScope ds(dev_);
    const int i = find_parked(id);
    if (i < 0) throw std::runtime_error("fork: a pipeline stage lost the parked conversation");
    Parked & p = parked_[i];
    const bool whole = pos == (int) p.history.size();
    const Checkpoint * snap = nullptr;
    for (const auto & k : p.ckpts)
        if (k.pos == pos) snap = &k;
    if (!whole && !snap) throw std::runtime_error("fork: no snapshot of the parked conversation at that position");
    if (!finish) {
        // each buffer's first rows (the entry holds every buffer's rows for its whole history, in each_ctx's order),
        // then the state: the snapshot at pos, or the entry's own state when pos is its end
        std::vector<size_t> parts;
        each_ctx((int) p.history.size(), [&](void *, size_t k) { parts.push_back(k); });
        size_t src = 0, part = 0, full = 0;
        for (size_t k : parts) full += k;
        each_ctx(pos, [&](void * d, size_t b) {
            CUDA_CHECK(cudaMemcpyAsync(d, p.host + src, b, cudaMemcpyHostToDevice, st_));
            src += parts[part++];
        });
        const uint8_t * state = whole ? p.host + full : snap->host;
        size_t so = 0;
        each_snap([&](void * d, size_t b) {
            CUDA_CHECK(cudaMemcpyAsync(d, state + so, b, cudaMemcpyHostToDevice, st_));
            so += b;
        });
        if (spark_) {
            size_t ro = full + snap_bytes();
            spark_->each_ring([&](void * d, size_t b) {
                CUDA_CHECK(cudaMemcpyAsync(d, p.host + ro, b, cudaMemcpyHostToDevice, st_));
                ro += b;
            });
        }
        return;
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    cur_->history.assign(p.history.begin(), p.history.begin() + pos);
    cur_->pending_T = 0;
    if (spark_) {
        spark_->set_fed(p.spark_lo, p.spark_hi);
        spark_->restart(pos);   // its rings hold the entry's positions: valid below pos while the ring still holds them
    }
    // the entry's snapshots at or before pos, copied
    for (auto & k : cur_->ckpts) k.pos = -1;
    const size_t sb = snap_bytes();
    size_t j = 0;
    for (const auto & c : p.ckpts) {
        if (c.pos > pos || j >= cur_->ckpts.size()) continue;
        Checkpoint & slot = cur_->ckpts[j++];
        memcpy(slot.host, c.host, sb);
        slot.pos = c.pos;
        slot.age = ++ckpt_age_;
    }
}

int DsEngine::reusable(const std::vector<int32_t> & prompt) const {
    return reuse_of(cur_->history, cur_->ckpts, prompt);
}

// The state this request starts from: the live one (Generator::start continues or rewinds it), a parked conversation
// it continues (restored), or the prefix a parked conversation shares with it (copied out, the entry untouched). The
// live conversation is parked first when the request would throw much of it away.
void DsEngine::select_conversation(const std::vector<int32_t> & prompt) {
    if (prompt.empty() || opt_.park_gib <= 0) return;
    const int live = reuse_of(cur_->history, cur_->ckpts, prompt);
    int bestp = -1, bestp_r = live;
    bool bestp_ext = false;
    for (size_t i = 0; i < parked_.size(); ++i) {
        const int r = reuse_of(parked_[i].history, parked_[i].ckpts, prompt);
        if (r > bestp_r) bestp = (int) i, bestp_r = r, bestp_ext = extends(parked_[i].history, prompt);
    }
    // a state on disk that does better (after a restart, or one that left RAM)
    int bestd = -1;
    {
        std::lock_guard<std::mutex> lk(disk_mu_);
        for (size_t i = 0; i < disk_.size(); ++i) {
            const int r = reuse_of(disk_[i].history, disk_[i].ckpts, prompt);
            if (r > bestp_r) bestd = (int) i, bestp_r = r, bestp_ext = extends(disk_[i].history, prompt);
        }
    }
    if (bestd >= 0) bestp = -1;
    const bool take = (bestp >= 0 || bestd >= 0) && (bestp_ext || (fork_min() > 0 && bestp_r - live >= fork_min()));
    uint64_t keep = take && bestp >= 0 ? parked_[bestp].used : 0;
    // the live conversation is kept when this request would throw much of it away: all of it when another comes in,
    // the part past the reusable prefix when it rewinds or starts over
    const int lost = take ? pos() : pos() - live;
    if (lost > 0 && lost >= opt_.park_min) park(keep, take);
    if (!take) return;
    if (bestd >= 0) {
        const double t0 = now_ms();
        keep = disk_load((size_t) bestd);
        if (!keep) return;   // unreadable or RAM short: the prompt is read as usual
        if (opt_.verbose) fprintf(stderr, "bnk: read a conversation of %zu tokens from disk in %.0f ms\n",
                                  parked_[find_parked(keep)].history.size(), now_ms() - t0);
    }
    if (bestp_ext) restore(keep);
    else fork_parked(keep, bestp_r);
}

// ------------------------------------------------------------------------------------------------ on disk
// One file a conversation, <park_dir>/ds-<identity>/<stamp>.bnkpark:
//   header: magic "BNKDSPK1", identity (u64), stages, history length, snapshots (u32 each), the snapshot
//   positions, the history (i32), then per stage: its entry's bytes (u64) and the DSpark range (i32 x2);
//   then every stage's entry (the per-position rows, the state, the drafter's rings), then every stage's snapshots.
// The identity covers the model files, the drafter, the stage layout and every part's size, so a file is only read
// by an engine whose state has the same shape.

namespace {

constexpr char kMagic[8] = {'B', 'N', 'K', 'D', 'S', 'P', 'K', '1'};

uint64_t fnv(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

uint64_t fnv_file(uint64_t h, const std::string & path) {
    if (path.empty()) return fnv(h, "-", 1);
    const std::string base = std::filesystem::path(path).filename().string();
    h = fnv(h, base.data(), base.size());
    struct stat st{};
    const int64_t size = stat(path.c_str(), &st) == 0 ? (int64_t) st.st_size : -1;
    return fnv(h, &size, sizeof size);
}

struct Header {
    std::vector<int32_t> history, ckpt_pos;
    std::vector<uint64_t> stage_bytes;
    std::vector<int32_t> spark_lo, spark_hi;
    uint64_t identity = 0;
};

bool read_header(FILE * f, Header & h) {
    char magic[8];
    uint32_t stages = 0, n_hist = 0, n_ck = 0;
    if (fread(magic, 8, 1, f) != 1 || memcmp(magic, kMagic, 8) != 0) return false;
    if (fread(&h.identity, 8, 1, f) != 1 || fread(&stages, 4, 1, f) != 1 || fread(&n_hist, 4, 1, f) != 1 ||
        fread(&n_ck, 4, 1, f) != 1)
        return false;
    if (stages == 0 || stages > 8 || n_ck > 64 || n_hist > (1u << 24)) return false;
    h.ckpt_pos.resize(n_ck);
    h.history.resize(n_hist);
    h.stage_bytes.resize(stages);
    h.spark_lo.resize(stages);
    h.spark_hi.resize(stages);
    if (n_ck && fread(h.ckpt_pos.data(), 4, n_ck, f) != n_ck) return false;
    if (n_hist && fread(h.history.data(), 4, n_hist, f) != n_hist) return false;
    for (uint32_t s = 0; s < stages; ++s)
        if (fread(&h.stage_bytes[s], 8, 1, f) != 1 || fread(&h.spark_lo[s], 4, 1, f) != 1 || fread(&h.spark_hi[s], 4, 1, f) != 1)
            return false;
    return true;
}

}  // namespace

uint64_t DsEngine::disk_identity() {
    uint64_t h = 1469598103934665603ull;
    h = fnv(h, kMagic, 8);
    h = fnv_file(h, model_path_);
    h = fnv_file(h, opt_.mmproj);
    h = fnv_file(h, opt_.mtp);
    for (DsEngine * s = this; s; s = s->next_.get()) {
        const int lay[2] = {s->l0_, s->l1_};
        h = fnv(h, lay, sizeof lay);
        const size_t sizes[2] = {s->snap_bytes(), s->park_bytes_here(64)};   // the state's and a 64-token entry's shape
        h = fnv(h, sizes, sizeof sizes);
    }
    return h;
}

void DsEngine::disk_init() {
    std::string dir = opt_.park_dir;
    if (dir.empty() && getenv("BNK_PARK_DIR")) dir = getenv("BNK_PARK_DIR");
    const double gib = env_double("BNK_PARK_DISK_GIB", opt_.park_disk_gib);
    if (dir.empty() || gib <= 0 || opt_.park_gib <= 0) return;
    char name[32];
    snprintf(name, sizeof name, "ds-%016llx", (unsigned long long) disk_identity());
    std::error_code ec;
    const std::filesystem::path d = std::filesystem::path(dir) / name;
    std::filesystem::create_directories(d, ec);
    if (ec) {
        fprintf(stderr, "bnk: conversations on disk off: cannot create %s (%s)\n", d.c_str(), ec.message().c_str());
        return;
    }
    park_dir_ = d.string();
    disk_limit_ = (size_t) (gib * 1073741824.0);
    disk_worker_ = std::make_unique<Worker>();
    size_t total = 0;
    for (const auto & e : std::filesystem::directory_iterator(d, ec)) {
        const std::string p = e.path().string();
        if (e.path().extension() == ".tmp") {   // a write the process did not finish
            std::filesystem::remove(e.path(), ec);
            continue;
        }
        if (e.path().extension() != ".bnkpark") continue;
        FILE * f = fopen(p.c_str(), "rb");
        Header h;
        const bool ok = f && read_header(f, h) && h.identity == disk_identity();
        if (f) fclose(f);
        if (!ok) {
            std::filesystem::remove(e.path(), ec);
            continue;
        }
        DiskEntry de;
        de.path = p;
        de.history = std::move(h.history);
        for (int32_t c : h.ckpt_pos) de.ckpts.push_back(Checkpoint{c, 0, nullptr});
        de.bytes = (size_t) e.file_size(ec);
        de.stamp = atoll(e.path().stem().c_str());
        disk_stamp_ = std::max(disk_stamp_, de.stamp);
        total += de.bytes;
        disk_.push_back(std::move(de));
    }
    if (opt_.verbose)
        fprintf(stderr, "bnk: conversations on disk: %s, %zu kept (%.2f GiB of %.0f)\n", park_dir_.c_str(), disk_.size(),
                total / 1073741824.0, gib);
    disk_trim();
}

// Takes entry `id` out of RAM on every stage and writes it (async: on the writer thread, which frees the buffers).
void DsEngine::disk_write(uint64_t id, bool async) {
    if (park_dir_.empty()) return drop_parked_id(id);
    std::vector<Parked> e;
    std::vector<size_t> snap;
    for (DsEngine * s = this; s; s = s->next_.get()) {
        const int i = s->find_parked(id);
        if (i < 0) throw std::runtime_error("disk: a pipeline stage lost the parked conversation");
        e.push_back(std::move(s->parked_[i]));
        s->parked_.erase(s->parked_.begin() + i);
        snap.push_back(s->snap_bytes());
    }
    std::vector<size_t> bytes;
    for (DsEngine * s = this; s; s = s->next_.get())
        bytes.push_back(s->park_bytes_here((int) e[0].history.size()));
    const int64_t stamp = ++disk_stamp_;
    const uint64_t ident = disk_identity();
    auto job = [this, e = std::move(e), snap, bytes, stamp, ident]() mutable {
        char name[32];
        snprintf(name, sizeof name, "%012lld.bnkpark", (long long) stamp);
        const std::string path = park_dir_ + "/" + name, tmp = path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        bool ok = f != nullptr;
        const uint32_t stages = (uint32_t) e.size(), n_hist = (uint32_t) e[0].history.size(), n_ck = (uint32_t) e[0].ckpts.size();
        auto put = [&](const void * p, size_t n) { ok = ok && (n == 0 || fwrite(p, 1, n, f) == n); };
        put(kMagic, 8);
        put(&ident, 8);
        put(&stages, 4);
        put(&n_hist, 4);
        put(&n_ck, 4);
        for (const auto & k : e[0].ckpts) put(&k.pos, 4);
        put(e[0].history.data(), (size_t) n_hist * 4);
        for (uint32_t s = 0; s < stages; ++s) {
            const uint64_t b = bytes[s];
            put(&b, 8);
            put(&e[s].spark_lo, 4);
            put(&e[s].spark_hi, 4);
        }
        for (uint32_t s = 0; s < stages; ++s) put(e[s].host, bytes[s]);
        for (uint32_t s = 0; s < stages; ++s)
            for (const auto & k : e[s].ckpts) put(k.host, snap[s]);
        if (f && fclose(f) != 0) ok = false;
        std::error_code ec;
        if (ok) std::filesystem::rename(tmp, path, ec);
        if (!ok || ec) {
            std::filesystem::remove(tmp, ec);
            fprintf(stderr, "bnk: could not write a conversation to %s\n", path.c_str());
        }
        DiskEntry de;
        de.path = path;
        de.history = e[0].history;
        for (const auto & k : e[0].ckpts) de.ckpts.push_back(Checkpoint{k.pos, 0, nullptr});
        de.stamp = stamp;
        de.bytes = 0;
        for (size_t b : bytes) de.bytes += b;
        for (size_t s = 0; s < e.size(); ++s) de.bytes += e[s].ckpts.size() * snap[s];
        for (auto & p : e) {
            for (auto & k : p.ckpts) cudaFreeHost(k.host);
            cudaFreeHost(p.host);
        }
        if (!ok || ec) return;
        {
            std::lock_guard<std::mutex> lk(disk_mu_);
            // one file a conversation: an older one whose history this one extends is superseded
            for (size_t i = disk_.size(); i-- > 0;) {
                const auto & h = disk_[i].history;
                if (h.size() <= de.history.size() && std::equal(h.begin(), h.end(), de.history.begin())) {
                    std::filesystem::remove(disk_[i].path, ec);
                    disk_.erase(disk_.begin() + (long) i);
                }
            }
            disk_.push_back(std::move(de));
        }
        disk_trim();
    };
    if (async) disk_worker_->post(std::move(job));
    else {
        disk_worker_->wait();
        job();
    }
}

// the oldest files go while the directory is over its budget
void DsEngine::disk_trim() {
    std::lock_guard<std::mutex> lk(disk_mu_);
    size_t total = 0;
    for (const auto & d : disk_) total += d.bytes;
    while (total > disk_limit_ && !disk_.empty()) {
        auto it = std::min_element(disk_.begin(), disk_.end(), [](auto & a, auto & b) { return a.stamp < b.stamp; });
        std::error_code ec;
        std::filesystem::remove(it->path, ec);
        total -= it->bytes;
        disk_.erase(it);
    }
}

void DsEngine::disk_drop_all() {
    if (park_dir_.empty()) return;
    disk_worker_->wait();
    std::lock_guard<std::mutex> lk(disk_mu_);
    std::error_code ec;
    for (const auto & d : disk_) std::filesystem::remove(d.path, ec);
    disk_.clear();
}

// Reads disk entry i into RAM as a parked entry of every stage (the file stays: a later state of the conversation
// supersedes it). Returns the entry's id, 0 when the file cannot be read or RAM is short.
uint64_t DsEngine::disk_load(size_t i) {
    disk_worker_->wait();   // a write in flight may be this very entry
    std::string path;
    {
        std::lock_guard<std::mutex> lk(disk_mu_);
        if (i >= disk_.size()) return 0;
        path = disk_[i].path;
    }
    FILE * f = fopen(path.c_str(), "rb");
    Header h;
    if (!f || !read_header(f, h) || h.identity != disk_identity()) {
        if (f) fclose(f);
        return 0;
    }
    std::vector<DsEngine *> st;
    for (DsEngine * s = this; s; s = s->next_.get()) st.push_back(s);
    bool ok = h.stage_bytes.size() == st.size();
    for (size_t s = 0; ok && s < st.size(); ++s) ok = h.stage_bytes[s] == st[s]->park_bytes_here((int) h.history.size());
    std::vector<Parked> e(st.size());
    for (size_t s = 0; ok && s < st.size(); ++s) {
        DeviceScope ds(st[s]->dev_);
        Parked & p = e[s];
        p.host = st[s]->pinned_take(h.stage_bytes[s], &p.cap);
        ok = p.host != nullptr && fread(p.host, 1, h.stage_bytes[s], f) == h.stage_bytes[s];
        p.history = h.history;
        p.spark_lo = h.spark_lo[s];
        p.spark_hi = h.spark_hi[s];
        p.bytes = h.stage_bytes[s] + h.ckpt_pos.size() * st[s]->snap_bytes();
    }
    for (size_t s = 0; ok && s < st.size(); ++s) {
        DeviceScope ds(st[s]->dev_);
        const size_t sb = st[s]->snap_bytes();
        for (int32_t c : h.ckpt_pos) {
            Checkpoint k{c, 0, st[s]->ck_take()};
            e[s].ckpts.push_back(k);
            if (fread(k.host, 1, sb, f) != sb) { ok = false; break; }
        }
    }
    fclose(f);
    if (!ok) {
        for (size_t s = 0; s < st.size(); ++s) {
            DeviceScope ds(st[s]->dev_);
            for (auto & k : e[s].ckpts) st[s]->ck_give(k.host);
            st[s]->pinned_give(e[s].host, e[s].cap);
        }
        fprintf(stderr, "bnk: could not read the conversation in %s\n", path.c_str());
        return 0;
    }
    const uint64_t id = ++park_age_;
    for (size_t s = 0; s < st.size(); ++s) {
        e[s].used = id;
        st[s]->parked_.push_back(std::move(e[s]));
    }
    return id;
}

// At shutdown: the live conversation and every parked one to disk, for the next start.
void DsEngine::persist() {
    if (park_dir_.empty()) return;
    const double t0 = now_ms();
    if (pos() >= std::max(1, opt_.park_min)) park(0, false);
    std::vector<uint64_t> ids;
    for (const auto & p : parked_) ids.push_back(p.used);
    for (uint64_t id : ids) disk_write(id, false);
    disk_worker_->wait();
    if (opt_.verbose && !ids.empty())
        fprintf(stderr, "bnk: wrote %zu conversations to disk in %.0f ms\n", ids.size(), now_ms() - t0);
}

}  // namespace bnk
