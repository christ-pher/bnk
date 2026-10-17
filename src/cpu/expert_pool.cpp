#include "cpu/expert_pool.h"

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "core/util.h"
#include "cpu/kernels.h"
#include "cpu/mdot.h"
#include "ggml-cpu.h"

namespace bnk {

// ------------------------------------------------------------------------------------------ store
ExpertStore::~ExpertStore() {
    if (base_) {
        if (pinned_) cudaHostUnregister(base_);
        munmap(base_, std::max<size_t>(total_, 4096));
    }
    if (dstage_) cudaFreeHost(dstage_);
}

static void pread_all(int fd, uint8_t * dst, size_t n, uint64_t off) {
    while (n) {
        const ssize_t r = pread(fd, dst, n, (off_t) off);
        if (r <= 0) throw std::runtime_error("pread of the model file failed");
        dst += r; n -= (size_t) r; off += (uint64_t) r;
    }
}

void ExpertStore::read_blob(int il, int e, uint8_t * dst) const { read_blob_part(il, e, dst, 0, 1); }

// slice `part` of `nparts` of the blob (gate | up | down: three ranges of the file), so a 14 MB expert is read
// by every thread of a pool at once rather than one thread each
void ExpertStore::read_blob_part(int il, int e, uint8_t * dst, int part, int nparts) const {
    const LayerWeights & L = m_->layers[il];
    const size_t total = L.gate_bytes + L.up_bytes + L.down_bytes;
    const size_t chunk = (total + nparts - 1) / nparts / 4096 * 4096 + 4096;
    const size_t a = std::min(total, chunk * part), b = std::min(total, a + chunk);
    if (a >= b) return;
    const size_t seg_len[3] = {L.gate_bytes, L.up_bytes, L.down_bytes};
    const uint64_t seg_off[3] = {L.gate_off + (uint64_t) e * L.gate_bytes, L.up_off + (uint64_t) e * L.up_bytes,
                                 L.down_off + (uint64_t) e * L.down_bytes};
    size_t s0 = 0;
    for (int i = 0; i < 3; ++i) {
        const size_t s1 = s0 + seg_len[i];
        const size_t lo = std::max(a, s0), hi = std::min(b, s1);
        if (lo < hi) pread_all(L.exp_fd, dst + lo, hi - lo, seg_off[i] + (lo - s0));
        s0 = s1;
    }
}

void ExpertStore::dstage_ensure() {
    if (dstage_) return;
    dstage_slots_ = std::max(1, env_int("BNK_DISK_STAGE", 32));
    if (cudaHostAlloc((void **) &dstage_, (size_t) dstage_slots_ * max_blob_, cudaHostAllocPortable) != cudaSuccess)
        throw std::runtime_error("cannot allocate the disk stage");
}

void ExpertStore::copy_batch_to_device(int il, const std::vector<int> & experts, uint8_t * dst, size_t stride,
                                       cudaStream_t s) {
    const size_t bb = blob_[il];
    std::vector<int> disk;
    for (size_t i = 0; i < experts.size(); ++i) {
        const Parts p = parts(il, experts[i]);
        if (p.pinned) CUDA_CHECK(cudaMemcpyAsync(dst + i * stride, p.gate, bb, cudaMemcpyHostToDevice, s));
        else disk.push_back((int) i);
    }
    if (disk.empty()) return;
    std::lock_guard<std::mutex> lk(dstage_mu_);
    dstage_ensure();
    for (size_t w0 = 0; w0 < disk.size(); w0 += dstage_slots_) {
        const int n = (int) std::min<size_t>(dstage_slots_, disk.size() - w0);
        const int nth = std::min(disk_threads, n);
        std::vector<std::thread> th;
        for (int t = 0; t < nth; ++t)
            th.emplace_back([&, t]() {   // every thread reads its slice of every expert of the wave
                for (int j = 0; j < n; ++j) read_blob_part(il, experts[disk[w0 + j]], dstage_ + (size_t) j * max_blob_, t, nth);
            });
        for (auto & x : th) x.join();
        for (int j = 0; j < n; ++j)
            CUDA_CHECK(cudaMemcpyAsync(dst + (size_t) disk[w0 + j] * stride, dstage_ + (size_t) j * max_blob_, bb,
                                       cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaStreamSynchronize(s));   // (an event would belong to one GPU; the stage serves every stage)
    }
}

void ExpertStore::prepare(const Model & m) {
    const Config & c = m.cfg;
    m_ = &m;
    n_expert_ = c.n_expert;
    blob_.resize(c.n_layer);
    model_ = 0;
    for (int il = 0; il < c.n_layer; ++il) {
        const LayerWeights & L = m.layers[il];
        blob_[il] = (L.gate_bytes + L.up_bytes + L.down_bytes + 63) / 64 * 64;
        model_ += blob_[il] * c.n_expert;
        max_blob_ = std::max(max_blob_, blob_[il]);
    }
    arena_slot_.assign((size_t) c.n_layer * c.n_expert, -1);
    slot_off_.clear();
    total_ = 0;
}

void ExpertStore::build(const Model & m, int threads, bool verbose, size_t cap, const Ranking & rank,
                        const std::function<bool(int, int)> & skip) {
    const Config & c = m.cfg;
    prepare(m);
    // which experts get the arena: the ranking's order (completed with the rest: expert by expert across the
    // layers, as the GPU caches complete theirs), those a GPU holds skipped, while they fit
    std::vector<std::pair<int, int>> order;
    {
        std::vector<char> seen(arena_slot_.size(), 0);
        for (auto [l, e] : rank)
            if (l >= 0 && l < c.n_layer && e >= 0 && e < c.n_expert && blob_[l] && !seen[(size_t) l * c.n_expert + e]) {
                seen[(size_t) l * c.n_expert + e] = 1;
                order.emplace_back(l, e);
            }
        for (int e = 0; e < c.n_expert; ++e)
            for (int l = 0; l < c.n_layer; ++l)
                if (blob_[l] && !seen[(size_t) l * c.n_expert + e]) order.emplace_back(l, e);
    }
    const size_t limit = cap ? cap : model_;
    int skipped = 0;
    for (auto [l, e] : order) {
        if (skip && skip(l, e)) { ++skipped; continue; }
        if (total_ + blob_[l] > limit) continue;
        arena_slot_[(size_t) l * c.n_expert + e] = (int32_t) slot_off_.size();
        slot_off_.push_back(total_);
        total_ += blob_[l];
    }
    n_arena_ = (int) slot_off_.size();
    const double t0 = now_ms();
    void * p = mmap(nullptr, std::max<size_t>(total_, 4096), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("cannot reserve the expert arena");
    madvise(p, total_, MADV_HUGEPAGE);
    base_ = (uint8_t *) p;

    // copy in parallel, each job a run of arena slots (in file order when there is no ranking)
    std::vector<std::pair<int, int>> slots;   // slot -> (layer, expert)
    slots.resize(n_arena_);
    for (int l = 0; l < c.n_layer; ++l)
        for (int e = 0; e < c.n_expert; ++e) {
            const int32_t sl = arena_slot_[(size_t) l * c.n_expert + e];
            if (sl >= 0) slots[sl] = {l, e};
        }
    constexpr int CHUNK = 16;
    std::atomic<int> next{0};
    const int jobs = (n_arena_ + CHUNK - 1) / CHUNK;
    auto worker = [&]() {
        for (int j; (j = next.fetch_add(1)) < jobs;) {
            for (int sl = j * CHUNK; sl < std::min(n_arena_, (j + 1) * CHUNK); ++sl) {
                const auto [il, e] = slots[sl];
                const LayerWeights & L = m.layers[il];
                uint8_t * dst = base_ + slot_off_[sl];
                memcpy(dst, L.gate_src + (size_t) e * L.gate_bytes, L.gate_bytes);
                memcpy(dst + L.gate_bytes, L.up_src + (size_t) e * L.up_bytes, L.up_bytes);
                memcpy(dst + L.gate_bytes + L.up_bytes, L.down_src + (size_t) e * L.down_bytes, L.down_bytes);
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < threads; ++i) th.emplace_back(worker);
    for (auto & t : th) t.join();
    const double t1 = now_ms();
    pinned_ = total_ > 0 && cudaHostRegister(base_, total_, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess;
    if (!pinned_) cudaGetLastError();
    if (pinned_) {   // mapped for the GPUs: the hit kernels read non-resident experts straight from it
        void * dp = nullptr;
        mapped_ = cudaHostGetDevicePointer(&dp, base_, 0) == cudaSuccess;
        if (!mapped_) cudaGetLastError();
        dev_base_ = (uint8_t *) dp;
    }
    if (verbose) {
        fprintf(stderr, "bnk: experts: %.2f GiB in RAM (copied in %.1f s, %.2f GiB/s), %s\n", total_ / 1073741824.0,
                (t1 - t0) / 1000, total_ / 1073741824.0 / ((t1 - t0) / 1000),
                pinned_ ? "page-locked for DMA" : "NOT page-locked (DMA falls back to staging)");
        if (total_ < model_)
            fprintf(stderr, "bnk: experts: %d of %zu in RAM, %d on the GPUs; the other %.2f GiB stay in the model file (page cache / disk)\n",
                    n_arena_, order.size(), skipped, (model_ - total_ - (size_t) skipped * max_blob_) / 1073741824.0);
    }
}

ExpertStore::Parts ExpertStore::parts(int il, int e) const {
    const LayerWeights & L = m_->layers[il];
    const int32_t sl = arena_slot_[(size_t) il * n_expert_ + e];
    if (sl >= 0) {
        const uint8_t * b = base_ + slot_off_[sl];
        return Parts{b, b + L.gate_bytes, b + L.gate_bytes + L.up_bytes, pinned_};
    }
    return Parts{L.gate_src + (size_t) e * L.gate_bytes, L.up_src + (size_t) e * L.up_bytes,
                 L.down_src + (size_t) e * L.down_bytes, false};
}

void ExpertStore::copy_to_device(int il, int e, uint8_t * dst, cudaStream_t s) {
    copy_batch_to_device(il, {e}, dst, 0, s);
}

// ------------------------------------------------------------------------------------------ pool
SpinPool::SpinPool(int n_workers, int first_cpu) {
    done_.reset(new Flag[n_workers + 1]);
    for (int i = 0; i < n_workers; ++i) {
        workers_.emplace_back([this, i]() { loop(i + 1); });
        if (first_cpu >= 0) {
            cpu_set_t cs;
            CPU_ZERO(&cs);
            CPU_SET(first_cpu + i, &cs);
            pthread_setaffinity_np(workers_.back().native_handle(), sizeof cs, &cs);
        }
    }
}

SpinPool::~SpinPool() {
    stop_ = true;
    gen_.fetch_add(1);
    for (auto & t : workers_) t.join();
}

void SpinPool::loop(int id) {
    uint64_t seen = 0;
    while (true) {
        int spins = 0;
        uint64_t g;
        while ((g = gen_.load(std::memory_order_acquire)) == seen) {
            if (++spins > 2000000) usleep(20);
            else __builtin_ia32_pause();
        }
        seen = g;
        if (stop_) return;
        (*fn_)(id, size());
        done_[id].v.store(g, std::memory_order_release);
    }
}

void SpinPool::run(const std::function<void(int, int)> & fn) {
    fn_ = &fn;
    const uint64_t g = gen_.fetch_add(1, std::memory_order_acq_rel) + 1;
    fn(0, size());
    for (size_t i = 1; i <= workers_.size(); ++i)
        while (done_[i].v.load(std::memory_order_acquire) != g) __builtin_ia32_pause();
}

// ------------------------------------------------------------------------------------------ experts
void CpuExpertPool::init(const Model & m, const ExpertStore & st, int n_threads) {
    m_ = &m;
    st_ = &st;
    // BNK_CPU_PIN=1: worker i on core i + 1 (the calling thread stays free: threads it creates later would
    // inherit a single-core mask); BNK_CPU_CHUNKS=n: n work ranges per thread, claimed dynamically
    const bool pin = env_int("BNK_CPU_PIN", 0) != 0;
    chunks_per_thread_ = std::max(1, env_int("BNK_CPU_CHUNKS", 1));
    pool_ = std::make_unique<SpinPool>(n_threads - 1, pin ? 1 : -1);
}

static inline float silu(float x) { return x / (1.f + expf(-x)); }

static inline void split(int n, int part, int nparts, int & b, int & e) {
    b = (int) ((int64_t) n * part / nparts);
    e = (int) ((int64_t) n * (part + 1) / nparts);
}

void CpuExpertPool::run(int il, int T, const float * x, const std::vector<ExpertTask> & tasks, float * out) {
    std::lock_guard<std::mutex> lk(mu_);
    const double t0 = now_ms();
    const Config & c = m_->cfg;
    const LayerWeights & L = m_->layers[il];
    const int E = c.n_embd, F = c.n_ff_exp;
    const int nt = (int) tasks.size();
    if (nt == 0) {
        memset(out, 0, sizeof(float) * T * E);
        return;
    }
    const auto * tg = ggml_get_type_traits_cpu((ggml_type) L.gate_type);
    const auto * tu = ggml_get_type_traits_cpu((ggml_type) L.up_type);
    const auto * td = ggml_get_type_traits_cpu((ggml_type) L.down_type);
    if (tg->vec_dot_type != tu->vec_dot_type) throw std::runtime_error("gate/up vec_dot types differ");
    const ggml_type xt = tg->vec_dot_type;
    const bool down_q2 = L.down_type == GGML_TYPE_Q2_0;
    const bool gu_multi = mdot_supported(L.gate_type) && L.gate_type == L.up_type;
    const bool dn_multi = mdot_supported(L.down_type);
    const size_t xrow = ggml_row_size(xt, E);
    const size_t hrow = down_q2 ? (F / 64) * sizeof(A8P64) : ggml_row_size(td->vec_dot_type, F);
    const size_t grow = ggml_row_size((ggml_type) L.gate_type, E), urow = ggml_row_size((ggml_type) L.up_type, E);
    const size_t drow = ggml_row_size((ggml_type) L.down_type, F);
    auto xfrom = ggml_get_type_traits_cpu(xt)->from_float;
    auto hfrom = ggml_get_type_traits_cpu(td->vec_dot_type)->from_float;

    // group the tasks by expert: one decode of each weight row serves all of the expert's tokens
    groups_.clear();
    std::vector<int> gidx(nt);
    for (int i = 0; i < nt; ++i) {
        int g = 0;
        while (g < (int) groups_.size() && tasks[groups_[g][0]].expert != tasks[i].expert) ++g;
        if (g == (int) groups_.size()) groups_.emplace_back();
        groups_[g].push_back(i);
        gidx[i] = g;
    }
    const int ng = (int) groups_.size();
    // experts outside the page-locked arena are read from the model file into a scratch first (the pool's threads
    // each read some), rather than faulted in a page at a time by the dot products
    gparts_.resize(ng);
    {
        std::vector<int> disk;
        for (int gi = 0; gi < ng; ++gi) {
            gparts_[gi] = st_->parts(il, tasks[groups_[gi][0]].expert);
            if (!gparts_[gi].pinned) disk.push_back(gi);
        }
        if (!disk.empty()) {
            const size_t bb = st_->blob_bytes(il);
            if (dscratch_.size() < disk.size() * bb) dscratch_.resize(disk.size() * bb);
            for (size_t j = 0; j < disk.size(); ++j) {
                uint8_t * d = dscratch_.data() + j * bb;
                gparts_[disk[j]] = ExpertStore::Parts{d, d + L.gate_bytes, d + L.gate_bytes + L.up_bytes, false};
            }
            pool_->run([&](int part, int nparts) {   // every thread reads its slice of every expert
                for (size_t j = 0; j < disk.size(); ++j)
                    st_->read_blob_part(il, tasks[groups_[disk[j]][0]].expert, dscratch_.data() + j * bb, part, nparts);
            });
            disk_reads_ += (int64_t) disk.size();
        }
    }

    xq_.resize(xrow * T);
    gu_.resize((size_t) nt * 2 * F);
    hq_.resize(hrow * nt + 64);
    if (rows_done_n_ < ng) {
        rows_done_.reset(new std::atomic<int>[ng]);
        rows_done_n_ = ng;
    }
    for (int i = 0; i < ng; ++i) rows_done_[i].store(0, std::memory_order_relaxed);
    for (int t = 0; t < T; ++t) xfrom(x + (size_t) t * E, xq_.data() + t * xrow, E);

    // phase 1: the gate/up rows of every expert group, split evenly
    const int R = ng * 2 * F;
    // work is cut into chunks_per_thread x threads ranges claimed from a counter (1: one fixed slice each)
    const int nchunk1 = std::max(1, std::min(R, chunks_per_thread_ * pool_->size()));
    std::atomic<int> next1{0};
    pool_->run([&](int part, int nparts) {
      for (int c = chunks_per_thread_ > 1 ? next1.fetch_add(1) : part; c < (chunks_per_thread_ > 1 ? nchunk1 : nparts);
           c = chunks_per_thread_ > 1 ? next1.fetch_add(1) : nparts) {
        int b, e;
        split(R, c, chunks_per_thread_ > 1 ? nchunk1 : nparts, b, e);
        thread_local std::vector<const void *> ysv;
        thread_local std::vector<float> valsv;
        ysv.resize(nt);
        valsv.resize(nt);
        const void ** ys = ysv.data();
        float * vals = valsv.data();
        int g = b;
        while (g < e) {
            const int gi = g / (2 * F);
            const int r_end = std::min(e, (gi + 1) * 2 * F);
            const std::vector<int> & grp = groups_[gi];
            const int n = (int) grp.size();
            const ExpertStore::Parts pt = gparts_[gi];
            for (int k = 0; k < n; ++k) ys[k] = xq_.data() + tasks[grp[k]].t * xrow;
            for (; g < r_end; ++g) {
                const int r = g - gi * 2 * F;
                const uint8_t * w = r < F ? pt.gate + (size_t) r * grow : pt.up + (size_t) (r - F) * urow;
                if (gu_multi && n > 1) {
                    mdot(L.gate_type, E, w, ys, n, vals);
                } else {
                    for (int k = 0; k < n; ++k) (r < F ? tg : tu)->vec_dot(E, &vals[k], 0, w, 0, ys[k], 0, 1);
                }
                for (int k = 0; k < n; ++k) gu_[(size_t) grp[k] * 2 * F + r] = vals[k];
            }
            const int mine = r_end - std::max(b, gi * 2 * F);
            if (rows_done_[gi].fetch_add(mine, std::memory_order_acq_rel) + mine == 2 * F) {
                // NVFP4 global scales (model.h): gate and up here, down on the output rows below
                const int ex0 = tasks[grp[0]].expert;
                const float sg = L.exp_sgate.empty() ? 1.f : L.exp_sgate[ex0], su = L.exp_sup.empty() ? 1.f : L.exp_sup[ex0];
                for (int k = 0; k < n; ++k) {
                    float * gu = gu_.data() + (size_t) grp[k] * 2 * F;
                    const float lim = L.exp_scales.limit;
                    for (int r = 0; r < F; ++r) {
                        float gv = gu[r] * sg, uv = gu[F + r] * su;
                        if (lim > 0.f) { gv = std::min(gv, lim); uv = std::min(std::max(uv, -lim), lim); }
                        gu[r] = silu(gv) * uv;
                    }
                    if (down_q2) quantize_a8p64(gu, F, (A8P64 *) (hq_.data() + grp[k] * hrow));
                    else hfrom(gu, hq_.data() + grp[k] * hrow, F);
                }
            }
        }
      }
    });

    // phase 2: output rows, split evenly; each row decodes every group's down row once, then each token sums its
    // experts in task order (the same order a one-token window uses, so speculation reproduces plain decoding)
    const int nchunk2 = std::max(1, std::min(E, chunks_per_thread_ * pool_->size()));
    std::atomic<int> next2{0};
    pool_->run([&](int part, int nparts) {
      for (int c = chunks_per_thread_ > 1 ? next2.fetch_add(1) : part; c < (chunks_per_thread_ > 1 ? nchunk2 : nparts);
           c = chunks_per_thread_ > 1 ? next2.fetch_add(1) : nparts) {
        int b, e;
        split(E, c, chunks_per_thread_ > 1 ? nchunk2 : nparts, b, e);
        thread_local std::vector<const void *> hsv;
        thread_local std::vector<float> valsv, tvv;
        hsv.resize(nt);
        valsv.resize(nt);
        tvv.resize(nt);
        const void ** hs = hsv.data();
        float * vals = valsv.data();
        float * tv = tvv.data();
        for (int r = b; r < e; ++r) {
            for (int gi = 0; gi < ng; ++gi) {
                const std::vector<int> & grp = groups_[gi];
                const int n = (int) grp.size();
                const uint8_t * w = gparts_[gi].down + (size_t) r * drow;
                for (int k = 0; k < n; ++k) hs[k] = hq_.data() + grp[k] * hrow;
                if (dn_multi && n > 1) {
                    mdot(L.down_type, F, w, hs, n, vals);
                } else if (down_q2) {
                    for (int k = 0; k < n; ++k) vals[k] = dot_q2_0(w, (const A8P64 *) hs[k], F);
                } else {
                    for (int k = 0; k < n; ++k) td->vec_dot(F, &vals[k], 0, w, 0, hs[k], 0, 1);
                }
                const float sd = L.exp_sdown.empty() ? 1.f : L.exp_sdown[tasks[grp[0]].expert];
                for (int k = 0; k < n; ++k) tv[grp[k]] = vals[k] * sd;
            }
            for (int t = 0; t < T; ++t) out[(size_t) t * E + r] = 0.f;
            for (int i = 0; i < nt; ++i) out[(size_t) tasks[i].t * E + r] += tasks[i].w * tv[i];
        }
      }
    });
    last_ms = now_ms() - t0;
}

}  // namespace bnk
