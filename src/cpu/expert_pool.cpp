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
        munmap(base_, total_);
    }
}

void ExpertStore::build(const Model & m, int threads, bool verbose) {
    const Config & c = m.cfg;
    layer_off_.resize(c.n_layer);
    blob_.resize(c.n_layer);
    total_ = 0;
    for (int il = 0; il < c.n_layer; ++il) {
        const LayerWeights & L = m.layers[il];
        blob_[il] = (L.gate_bytes + L.up_bytes + L.down_bytes + 63) / 64 * 64;
        layer_off_[il] = total_;
        total_ += blob_[il] * c.n_expert;
        max_blob_ = std::max(max_blob_, blob_[il]);
    }
    const double t0 = now_ms();
    void * p = mmap(nullptr, total_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("cannot reserve the expert arena");
    madvise(p, total_, MADV_HUGEPAGE);
    base_ = (uint8_t *) p;

    // copy in parallel; each job streams one contiguous range of one source tensor (a matrix of a run of
    // experts), so a cold page cache reads the files sequentially
    constexpr int PARTS = 4;
    std::atomic<int> next{0};
    const int jobs = c.n_layer * 3 * PARTS;
    auto worker = [&]() {
        for (int j; (j = next.fetch_add(1)) < jobs;) {
            const int il = j / (3 * PARTS), mat = (j / PARTS) % 3, part = j % PARTS;
            const LayerWeights & L = m.layers[il];
            const int e0 = part * c.n_expert / PARTS, e1 = (part + 1) * c.n_expert / PARTS;
            const uint8_t * src = mat == 0 ? L.gate_src : mat == 1 ? L.up_src : L.down_src;
            const size_t sz = mat == 0 ? L.gate_bytes : mat == 1 ? L.up_bytes : L.down_bytes;
            const size_t dst_off = mat == 0 ? 0 : mat == 1 ? L.gate_bytes : L.gate_bytes + L.up_bytes;
            const uint8_t * s0 = src + (size_t) e0 * sz;
            const size_t len = (size_t) (e1 - e0) * sz;
            const uintptr_t pa = (uintptr_t) s0 & ~(uintptr_t) 4095;
            madvise((void *) pa, len + ((uintptr_t) s0 - pa), MADV_SEQUENTIAL);
            madvise((void *) pa, len + ((uintptr_t) s0 - pa), MADV_WILLNEED);
            for (int e = e0; e < e1; ++e)
                memcpy(base_ + layer_off_[il] + (size_t) e * blob_[il] + dst_off, src + (size_t) e * sz, sz);
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < threads; ++i) th.emplace_back(worker);
    for (auto & t : th) t.join();
    const double t1 = now_ms();
    pinned_ = cudaHostRegister(base_, total_, cudaHostRegisterPortable) == cudaSuccess;
    if (!pinned_) cudaGetLastError();
    if (verbose)
        fprintf(stderr, "bnk: experts: %.2f GiB in RAM (copied in %.1f s, %.2f GiB/s), %s\n", total_ / 1073741824.0,
                (t1 - t0) / 1000, total_ / 1073741824.0 / ((t1 - t0) / 1000),
                pinned_ ? "page-locked for DMA" : "NOT page-locked (DMA falls back to staging)");
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
            const uint8_t * blob = st_->blob(il, tasks[grp[0]].expert);
            for (int k = 0; k < n; ++k) ys[k] = xq_.data() + tasks[grp[k]].t * xrow;
            for (; g < r_end; ++g) {
                const int r = g - gi * 2 * F;
                const uint8_t * w = r < F ? blob + (size_t) r * grow : blob + L.gate_bytes + (size_t) (r - F) * urow;
                if (gu_multi && n > 1) {
                    mdot(L.gate_type, E, w, ys, n, vals);
                } else {
                    for (int k = 0; k < n; ++k) (r < F ? tg : tu)->vec_dot(E, &vals[k], 0, w, 0, ys[k], 0, 1);
                }
                for (int k = 0; k < n; ++k) gu_[(size_t) grp[k] * 2 * F + r] = vals[k];
            }
            const int mine = r_end - std::max(b, gi * 2 * F);
            if (rows_done_[gi].fetch_add(mine, std::memory_order_acq_rel) + mine == 2 * F) {
                for (int k = 0; k < n; ++k) {
                    float * gu = gu_.data() + (size_t) grp[k] * 2 * F;
                    for (int r = 0; r < F; ++r) gu[r] = silu(gu[r]) * gu[F + r];
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
                const uint8_t * w = st_->blob(il, tasks[grp[0]].expert) + L.gate_bytes + L.up_bytes + (size_t) r * drow;
                for (int k = 0; k < n; ++k) hs[k] = hq_.data() + grp[k] * hrow;
                if (dn_multi && n > 1) {
                    mdot(L.down_type, F, w, hs, n, vals);
                } else if (down_q2) {
                    for (int k = 0; k < n; ++k) vals[k] = dot_q2_0(w, (const A8P64 *) hs[k], F);
                } else {
                    for (int k = 0; k < n; ++k) td->vec_dot(F, &vals[k], 0, w, 0, hs[k], 0, 1);
                }
                for (int k = 0; k < n; ++k) tv[grp[k]] = vals[k];
            }
            for (int t = 0; t < T; ++t) out[(size_t) t * E + r] = 0.f;
            for (int i = 0; i < nt; ++i) out[(size_t) tasks[i].t * E + r] += tasks[i].w * tv[i];
        }
      }
    });
    last_ms = now_ms() - t0;
}

}  // namespace bnk
