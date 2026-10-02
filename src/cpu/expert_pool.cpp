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

    // copy blobs in parallel (layer-major so each thread streams through the source files)
    std::atomic<int> next{0};
    const int jobs = c.n_layer * 8;
    auto worker = [&]() {
        for (int j; (j = next.fetch_add(1)) < jobs;) {
            const int il = j / 8, part = j % 8;
            const LayerWeights & L = m.layers[il];
            const int e0 = part * c.n_expert / 8, e1 = (part + 1) * c.n_expert / 8;
            for (int e = e0; e < e1; ++e) {
                uint8_t * dst = base_ + layer_off_[il] + (size_t) e * blob_[il];
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
    pinned_ = cudaHostRegister(base_, total_, cudaHostRegisterPortable) == cudaSuccess;
    if (!pinned_) cudaGetLastError();
    if (verbose)
        fprintf(stderr, "bnk: experts: %.2f GiB in RAM (copied in %.1f s, %.2f GiB/s), %s\n", total_ / 1073741824.0,
                (t1 - t0) / 1000, total_ / 1073741824.0 / ((t1 - t0) / 1000),
                pinned_ ? "page-locked for DMA" : "NOT page-locked (DMA falls back to staging)");
}

// ------------------------------------------------------------------------------------------ pool
SpinPool::SpinPool(int n_workers, int first_cpu) {
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
            if (++spins > 200000) usleep(50);
            else __builtin_ia32_pause();
        }
        seen = g;
        if (stop_) return;
        const auto & fn = *fn_;
        const int n = n_;
        for (int i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < n;) fn(i, id);
        done_.fetch_add(1, std::memory_order_acq_rel);
    }
}

void SpinPool::run(int n, const std::function<void(int, int)> & fn) {
    if (n <= 0) return;
    fn_ = &fn;
    n_ = n;
    next_.store(0, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_acq_rel);
    for (int i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < n;) fn(i, 0);
    while (done_.load(std::memory_order_acquire) < (int) workers_.size()) __builtin_ia32_pause();
}

// ------------------------------------------------------------------------------------------ experts
void CpuExpertPool::init(const Model & m, const ExpertStore & st, int n_threads) {
    m_ = &m;
    st_ = &st;
    pool_ = std::make_unique<SpinPool>(n_threads - 1, -1);
}

static inline float silu(float x) { return x / (1.f + expf(-x)); }

void CpuExpertPool::run(int il, int T, const float * x, const std::vector<ExpertTask> & tasks, float * out) {
    const double t0 = now_ms();
    const Config & c = m_->cfg;
    const LayerWeights & L = m_->layers[il];
    const int E = c.n_embd, F = c.n_ff_exp;
    const int nt = (int) tasks.size();
    memset(out, 0, sizeof(float) * T * E);
    if (nt == 0) return;

    const auto * tg = ggml_get_type_traits_cpu((ggml_type) L.gate_type);
    const auto * tu = ggml_get_type_traits_cpu((ggml_type) L.up_type);
    const auto * td = ggml_get_type_traits_cpu((ggml_type) L.down_type);
    if (tg->vec_dot_type != tu->vec_dot_type) throw std::runtime_error("gate/up vec_dot types differ");
    const ggml_type xt = tg->vec_dot_type, ht = td->vec_dot_type;
    const size_t xrow = ggml_row_size(xt, E), hrow = ggml_row_size(ht, F);
    const size_t grow = ggml_row_size((ggml_type) L.gate_type, E), urow = ggml_row_size((ggml_type) L.up_type, E);
    const size_t drow = ggml_row_size((ggml_type) L.down_type, F);
    auto xfrom = ggml_get_type_traits_cpu(xt)->from_float;
    auto hfrom = ggml_get_type_traits_cpu(ht)->from_float;

    xq_.resize(xrow * T);
    gu_.resize((size_t) nt * 2 * F);
    hq_.resize(hrow * nt);
    dout_.resize((size_t) nt * E);

    // 1. quantize the inputs, one job per token
    pool_->run(T, [&](int t, int) { xfrom(x + (size_t) t * E, xq_.data() + t * xrow, E); });

    // 2. gate and up rows: tasks x chunks of 64 rows over the 2F rows
    constexpr int GU_CHUNK = 64;
    const int gu_chunks = (2 * F + GU_CHUNK - 1) / GU_CHUNK;
    pool_->run(nt * gu_chunks, [&](int j, int) {
        const int ti = j / gu_chunks, ch = j % gu_chunks;
        const ExpertTask & tk = tasks[ti];
        const uint8_t * b = st_->blob(il, tk.expert);
        const uint8_t * xq = xq_.data() + tk.t * xrow;
        float * gu = gu_.data() + (size_t) ti * 2 * F;
        const int r0 = ch * GU_CHUNK, r1 = std::min(2 * F, r0 + GU_CHUNK);
        for (int r = r0; r < r1; ++r) {
            if (r < F) tg->vec_dot(E, &gu[r], 0, b + (size_t) r * grow, 0, xq, 0, 1);
            else tu->vec_dot(E, &gu[r], 0, b + L.gate_bytes + (size_t) (r - F) * urow, 0, xq, 0, 1);
        }
    });

    // 3. h = silu(g)*u, quantized; then the down rows in chunks of 256
    pool_->run(nt, [&](int ti, int) {
        float * gu = gu_.data() + (size_t) ti * 2 * F;
        for (int r = 0; r < F; ++r) gu[r] = silu(gu[r]) * gu[F + r];
        hfrom(gu, hq_.data() + ti * hrow, F);
    });
    constexpr int D_CHUNK = 256;
    const int d_chunks = (E + D_CHUNK - 1) / D_CHUNK;
    pool_->run(nt * d_chunks, [&](int j, int) {
        const int ti = j / d_chunks, ch = j % d_chunks;
        const ExpertTask & tk = tasks[ti];
        const uint8_t * b = st_->blob(il, tk.expert) + L.gate_bytes + L.up_bytes;
        const uint8_t * hq = hq_.data() + ti * hrow;
        float * d = dout_.data() + (size_t) ti * E;
        const int r0 = ch * D_CHUNK, r1 = std::min(E, r0 + D_CHUNK);
        for (int r = r0; r < r1; ++r) td->vec_dot(F, &d[r], 0, b + (size_t) r * drow, 0, hq, 0, 1);
    });

    // 4. weighted sum per token (in task order, for determinism)
    pool_->run(T * d_chunks, [&](int j, int) {
        const int t = j / d_chunks, ch = j % d_chunks;
        const int r0 = ch * D_CHUNK, r1 = std::min(E, r0 + D_CHUNK);
        float * o = out + (size_t) t * E;
        for (int ti = 0; ti < nt; ++ti) {
            if (tasks[ti].t != t) continue;
            const float w = tasks[ti].w;
            const float * d = dout_.data() + (size_t) ti * E;
            for (int r = r0; r < r1; ++r) o[r] += w * d[r];
        }
    });
    last_ms = now_ms() - t0;
}

}  // namespace bnk
