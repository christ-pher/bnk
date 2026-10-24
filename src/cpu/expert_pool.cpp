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

ColdStore::~ColdStore() {
    if (pinned_) cudaHostUnregister(base_);
    if (base_) munmap(base_, total_);
}

void ColdStore::build(const Model & m, int threads, bool verbose) {
    const Config & c = m.cfg;
    off_.assign(c.n_layer, {});
    blob_.assign(c.n_layer, {});
    total_ = 0;
    int n = 0;
    for (int il = 0; il < c.n_layer; ++il) {
        const LayerWeights & L = m.layers[il];
        for (const auto & g : L.cold) {
            const size_t b = (g.gate_bytes + g.up_bytes + g.down_bytes + 63) / 64 * 64;
            off_[il].push_back(total_);
            blob_[il].push_back(b);
            total_ += b * g.experts.size();
            max_blob_ = std::max(max_blob_, b);
            n += (int) g.experts.size();
        }
    }
    if (!total_) return;
    const double t0 = now_ms();
    void * p = mmap(nullptr, total_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("cannot reserve the cold expert arena");
    madvise(p, total_, MADV_HUGEPAGE);
    base_ = (uint8_t *) p;
    std::atomic<int> next{0};
    std::vector<std::pair<int, int>> jobs;   // (layer, group)
    for (int il = 0; il < c.n_layer; ++il)
        for (size_t g = 0; g < m.layers[il].cold.size(); ++g) jobs.push_back({il, (int) g});
    auto worker = [&]() {
        for (int j; (j = next.fetch_add(1)) < (int) jobs.size();) {
            const auto [il, gi] = jobs[j];
            const auto & g = m.layers[il].cold[gi];
            uint8_t * dst = base_ + off_[il][gi];
            const size_t b = blob_[il][gi];
            for (size_t i = 0; i < g.experts.size(); ++i) {
                memcpy(dst + i * b, g.gate_src + i * g.gate_bytes, g.gate_bytes);
                memcpy(dst + i * b + g.gate_bytes, g.up_src + i * g.up_bytes, g.up_bytes);
                memcpy(dst + i * b + g.gate_bytes + g.up_bytes, g.down_src + i * g.down_bytes, g.down_bytes);
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < std::max(1, threads); ++i) th.emplace_back(worker);
    for (auto & t : th) t.join();
    const double t1 = now_ms();
    pinned_ = cudaHostRegister(base_, total_, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess;
    if (!pinned_) cudaGetLastError();
    if (verbose)
        fprintf(stderr, "bnk: cold experts: %d in RAM, %.2f GiB (copied in %.1f s), %s\n", n, total_ / 1073741824.0,
                (t1 - t0) / 1000, pinned_ ? "page-locked for DMA" : "not page-locked");
}

void ExpertStore::layout(const Model & m) {
    const Config & c = m.cfg;
    layer_off_.resize(c.n_layer);
    blob_.resize(c.n_layer);
    total_ = 0;
    max_blob_ = 0;
    for (int il = 0; il < c.n_layer; ++il) {
        const LayerWeights & L = m.layers[il];
        blob_[il] = (L.gate_bytes + L.up_bytes + L.down_bytes + 63) / 64 * 64;
        layer_off_[il] = total_;
        total_ += blob_[il] * c.n_expert;
        max_blob_ = std::max(max_blob_, blob_[il]);
    }
    total_ = 0;
}

void ExpertStore::build(const Model & m, int threads, bool verbose, const std::vector<char> * skip) {
    const Config & c = m.cfg;
    layer_off_.resize(c.n_layer);
    blob_.resize(c.n_layer);
    total_ = 0;
    // with a skip list the arena holds only the experts not skipped (those resident in VRAM without a host copy:
    // docs/DEEPSEEK.md); index_[il][e] is the expert's blob number in its layer, -1 when skipped
    index_.assign(c.n_layer, {});
    int kept = 0;
    for (int il = 0; il < c.n_layer; ++il) {
        const LayerWeights & L = m.layers[il];
        blob_[il] = (L.gate_bytes + L.up_bytes + L.down_bytes + 63) / 64 * 64;
        layer_off_[il] = total_;
        index_[il].assign(c.n_expert, -1);
        int n = 0;
        for (int e = 0; e < c.n_expert; ++e)
            if (!skip || !(*skip)[(size_t) il * c.n_expert + e]) index_[il][e] = n++;
        kept += n;
        total_ += blob_[il] * n;
        max_blob_ = std::max(max_blob_, blob_[il]);
    }
    n_kept_ = kept;
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
            for (int e = e0; e < e1; ++e) {
                const int b = index_[il][e];
                if (b >= 0) memcpy(base_ + layer_off_[il] + (size_t) b * blob_[il] + dst_off, src + (size_t) e * sz, sz);
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < threads; ++i) th.emplace_back(worker);
    for (auto & t : th) t.join();
    const double t1 = now_ms();
    pinned_ = cudaHostRegister(base_, total_, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess;
    if (!pinned_) cudaGetLastError();
    if (verbose)
        fprintf(stderr, "bnk: experts: %d in RAM, %.2f GiB (copied in %.1f s, %.2f GiB/s), %s\n", n_kept_, total_ / 1073741824.0,
                (t1 - t0) / 1000, total_ / 1073741824.0 / ((t1 - t0) / 1000),
                pinned_ ? "page-locked for DMA" : "NOT page-locked (DMA falls back to staging)");
}

// ------------------------------------------------------------------------------------------ pool
// the CPUs this process may run on, one per physical core first (by core_id), then the SMT siblings: a container's
// cpuset is often a scattered subset of the host's threads, and two workers on one core share its execution units
static std::vector<int> cpu_order() {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if (sched_getaffinity(0, sizeof cs, &cs) != 0) return {};
    std::vector<std::pair<int, int>> cpus;   // (core_id, cpu)
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &cs)) continue;
        int core = c;
        char path[96];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", c);
        if (FILE * f = fopen(path, "r")) {
            if (fscanf(f, "%d", &core) != 1) core = c;
            fclose(f);
        }
        cpus.push_back({core, c});
    }
    std::vector<int> order;
    std::vector<char> used(cpus.size(), 0);
    for (int round = 0; (int) order.size() < (int) cpus.size(); ++round) {
        std::vector<int> seen_core;
        for (size_t i = 0; i < cpus.size(); ++i) {
            if (used[i]) continue;
            if (std::find(seen_core.begin(), seen_core.end(), cpus[i].first) != seen_core.end()) continue;
            seen_core.push_back(cpus[i].first);
            used[i] = 1;
            order.push_back(cpus[i].second);
        }
    }
    return order;
}

SpinPool::SpinPool(int n_workers, int first_cpu) {
    done_.reset(new Flag[n_workers + 1]);
    const std::vector<int> order = first_cpu >= 0 ? cpu_order() : std::vector<int>{};
    for (int i = 0; i < n_workers; ++i) {
        workers_.emplace_back([this, i]() { loop(i + 1); });
        if (first_cpu >= 0 && (size_t) (first_cpu + i) < order.size()) {   // (the caller's thread takes order[0..first_cpu))
            cpu_set_t cs;
            CPU_ZERO(&cs);
            CPU_SET(order[first_cpu + i], &cs);
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
void CpuExpertPool::init(const Model & m, const ExpertStore & st, int n_threads, bool pin_default) {
    m_ = &m;
    st_ = &st;
    // BNK_CPU_PIN=1: worker i on core i + 1 (the calling thread stays free: threads it creates later would
    // inherit a single-core mask); BNK_CPU_CHUNKS=n: n work ranges per thread, claimed dynamically
    const bool pin = env_int("BNK_CPU_PIN", pin_default ? 1 : 0) != 0;
    chunks_per_thread_ = std::max(1, env_int("BNK_CPU_CHUNKS", 1));
    chunks_env_ = getenv("BNK_CPU_CHUNKS") != nullptr;
    pool_ = std::make_unique<SpinPool>(n_threads - 1, pin ? 1 : -1);
}

void CpuExpertPool::set_chunks(int n) {
    if (!chunks_env_) chunks_per_thread_ = std::max(1, n);
}

static inline float silu(float x) { return x / (1.f + expf(-x)); }
static float g_cpu_limit = 0.f;   // SwiGLU clamp (DeepSeek-V4), 0 = none
void cpu_experts_set_swiglu_limit(float limit) { g_cpu_limit = limit; }

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
    // the multi-token kernels from this many tokens on (mdot_min_tokens: 1 where they beat ggml's one-token dot)
    const bool gu_multi = mdot_supported(L.gate_type) && L.gate_type == L.up_type;
    const bool dn_multi = mdot_supported(L.down_type);
    const int gu_min = gu_multi ? mdot_min_tokens(L.gate_type) : 2, dn_min = dn_multi ? mdot_min_tokens(L.down_type) : 2;
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

    gu_.resize((size_t) nt * 2 * F);
    hq_.resize(hrow * nt + 64);
    if (rows_done_n_ < ng) {
        rows_done_.reset(new std::atomic<int>[ng]);
        rows_done_n_ = ng;
    }
    for (int i = 0; i < ng; ++i) rows_done_[i].store(0, std::memory_order_relaxed);
    // only the tokens the tasks name are quantized (a prompt group hands in thousands of rows for a few tasks), on
    // every thread; the output is zeroed the same way (the rows of unnamed tokens stay zero)
    std::vector<int> tok_used;
    {
        std::vector<int> mark(T, -1);
        for (int i = 0; i < nt; ++i)
            if (mark[tasks[i].t] < 0) { mark[tasks[i].t] = (int) tok_used.size(); tok_used.push_back(tasks[i].t); }
        xq_.resize(xrow * tok_used.size());
        tok_slot_.assign(mark.begin(), mark.end());
    }
    const int nu = (int) tok_used.size();
    pool_->run([&](int part, int nparts) {
        int b, e;
        split(nu, part, nparts, b, e);
        for (int u = b; u < e; ++u) xfrom(x + (size_t) tok_used[u] * E, xq_.data() + (size_t) u * xrow, E);
        split(T, part, nparts, b, e);
        if (e > b) memset(out + (size_t) b * E, 0, sizeof(float) * (size_t) (e - b) * E);
    });

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
            for (int k = 0; k < n; ++k) ys[k] = xq_.data() + (size_t) tok_slot_[tasks[grp[k]].t] * xrow;
            for (; g < r_end; ++g) {
                const int r = g - gi * 2 * F;
                const uint8_t * w = r < F ? blob + (size_t) r * grow : blob + L.gate_bytes + (size_t) (r - F) * urow;
                if (gu_multi && n >= gu_min) {
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
                    if (g_cpu_limit > 0.f) {
                        const float lim = g_cpu_limit;
                        for (int r = 0; r < F; ++r)
                            gu[r] = silu(std::min(gu[r] * sg, lim)) * std::min(std::max(gu[F + r] * su, -lim), lim);
                    } else {
                        for (int r = 0; r < F; ++r) gu[r] = silu(gu[r] * sg) * gu[F + r] * su;
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
                const uint8_t * w = st_->blob(il, tasks[grp[0]].expert) + L.gate_bytes + L.up_bytes + (size_t) r * drow;
                for (int k = 0; k < n; ++k) hs[k] = hq_.data() + grp[k] * hrow;
                if (dn_multi && n >= dn_min) {
                    mdot(L.down_type, F, w, hs, n, vals);
                } else if (down_q2) {
                    for (int k = 0; k < n; ++k) vals[k] = dot_q2_0(w, (const A8P64 *) hs[k], F);
                } else {
                    for (int k = 0; k < n; ++k) td->vec_dot(F, &vals[k], 0, w, 0, hs[k], 0, 1);
                }
                const float sd = L.exp_sdown.empty() ? 1.f : L.exp_sdown[tasks[grp[0]].expert];
                for (int k = 0; k < n; ++k) tv[grp[k]] = vals[k] * sd;
            }
            for (int i = 0; i < nt; ++i) out[(size_t) tasks[i].t * E + r] += tasks[i].w * tv[i];
        }
      }
    });
    last_ms = now_ms() - t0;
}

}  // namespace bnk
