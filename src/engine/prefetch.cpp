#include "engine/prefetch.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>

#include "core/expert_cache.h"
#include "core/util.h"
#include "ggml.h"

namespace bnk {

ExpertPrefetch::~ExpertPrefetch() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    if (copy_) { cudaStreamSynchronize(copy_); cudaStreamDestroy(copy_); }
    if (msgs_) cudaFreeHost(msgs_);
    cudaFree(tables_);
    cudaFree(slots_);
    cudaFree(stage_);
    cudaFree(dst_);
}

void ExpertPrefetch::init(const Model & m, const ExpertStore & st, int per_layer, bool verbose) {
    B_ = std::clamp(per_layer, 0, kMaxPrefetch);
    if (!B_) return;
    m_ = &m;
    st_ = &st;
    n_layer_ = m.cfg.n_layer;
    for (int il = 0; il < n_layer_; ++il) slot_bytes_ = std::max(slot_bytes_, ExpertCache::slot_bytes(m, il));
    stage_bytes_ = st.max_blob_bytes();
    CUDA_CHECK(cudaHostAlloc((void **) &msgs_, sizeof(PfMsg) * n_layer_, cudaHostAllocMapped));
    memset(msgs_, 0, sizeof(PfMsg) * n_layer_);
    CUDA_CHECK(cudaMalloc(&slots_, 2 * B_ * slot_bytes_));
    CUDA_CHECK(cudaMalloc(&stage_, B_ * stage_bytes_));
    CUDA_CHECK(cudaMalloc(&dst_, 2 * B_ * sizeof(uint8_t *)));
    CUDA_CHECK(cudaMalloc(&tables_, sizeof(PfLayer) * n_layer_));
    std::vector<uint8_t *> dst(2 * B_);
    for (int i = 0; i < 2 * B_; ++i) dst[i] = slots_ + (size_t) i * slot_bytes_;
    CUDA_CHECK(cudaMemcpy(dst_, dst.data(), dst.size() * sizeof(uint8_t *), cudaMemcpyHostToDevice));
    std::vector<PfLayer> t(n_layer_);
    for (int il = 0; il < n_layer_; ++il) {
        memset(&t[il], 0, sizeof(PfLayer));
        for (int j = 0; j < kMaxPrefetch; ++j) {
            t[il].expert[j] = -1;
            t[il].blob[j] = j < B_ ? dst[(il & 1) * B_ + j] : nullptr;
        }
    }
    CUDA_CHECK(cudaMemcpy(tables_, t.data(), t.size() * sizeof(PfLayer), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
    if (verbose)
        fprintf(stderr, "bnk: expert prefetch: up to %d experts per layer one layer ahead (%.0f MiB of slots and staging)\n",
                B_, (2 * B_ * slot_bytes_ + B_ * stage_bytes_) / 1048576.0);
    thread_ = std::thread([this] { loop(); });
}

int64_t ExpertPrefetch::used() {
    if (!on()) return 0;
    std::vector<PfLayer> t(n_layer_);
    CUDA_CHECK(cudaMemcpy(t.data(), tables_, t.size() * sizeof(PfLayer), cudaMemcpyDeviceToHost));
    int64_t u = 0;
    for (auto & l : t) u += l.used;
    return u;
}

// Waits for each layer's prediction of the target forward and issues its copies; a newer forward (or one whose
// layers post nothing, e.g. eager debugging) moves the thread on.
void ExpertPrefetch::loop() {
    uint32_t served = 0;
    int idle = 0;
    while (!stop_.load(std::memory_order_relaxed)) {
        const uint32_t s = target_.load(std::memory_order_acquire);
        if (s == served) {
            if (++idle > 20000) {   // between turns: stop spinning a core
                timespec ts{0, 50000};
                nanosleep(&ts, nullptr);
            } else {
                __builtin_ia32_pause();
            }
            continue;
        }
        idle = 0;
        for (int il = 0; il + 1 < n_layer_; ++il) {
            const PfMsg * m = msg(il);
            bool moved = false;
            while (__atomic_load_n(&m->seq, __ATOMIC_ACQUIRE) != s) {
                if (stop_.load(std::memory_order_relaxed) || target_.load(std::memory_order_acquire) != s) {
                    moved = true;
                    break;
                }
                __builtin_ia32_pause();
            }
            if (moved) break;
            issue(il + 1, *m, s);
        }
        served = s;
    }
}

// Copies the predicted blobs of layer tl into its parity's slots, repacks them and marks them for forward seq. The
// slots were last read by layer tl - 2, whose kernels finished before layer tl - 1 posted this prediction.
void ExpertPrefetch::issue(int tl, const PfMsg & m, uint32_t seq) {
    static const bool nocopy = getenv("BNK_PF_NOCOPY") != nullptr;   // diagnostics: predictions only
    const int n = std::min(m.n, B_);
    if (n <= 0 || nocopy) return;
    const Config & c = m_->cfg;
    const LayerWeights & L = m_->layers[tl];
    const size_t bb = st_->blob_bytes(tl);
    PfMark mk{};
    mk.n = n;
    for (int j = 0; j < n; ++j) {
        mk.expert[j] = m.expert[j];
        CUDA_CHECK(cudaMemcpyAsync(stage_ + (size_t) j * bb, st_->blob(tl, m.expert[j]), bb, cudaMemcpyHostToDevice, copy_));
    }
    moe_repack_blobs(stage_, bb, dst_ + (tl & 1) * B_, n, c.n_ff_exp, c.n_embd, L.gate_type,
                     ggml_row_size((ggml_type) L.gate_type, c.n_embd), L.down_type,
                     ggml_row_size((ggml_type) L.down_type, c.n_ff_exp), copy_);
    moe_pf_mark(tables_ + tl, mk, seq, copy_);
    issued_.fetch_add(n, std::memory_order_relaxed);
}

}  // namespace bnk
