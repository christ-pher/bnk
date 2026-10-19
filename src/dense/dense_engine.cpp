#include "dense/dense_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "dense/dense_attn.h"
#include "dense/dense_ops.h"
#include "kernels/ops.h"

namespace bnk {

static const float * F(const QMat & m) { return (const float *) m.data; }

// Debug dumps of one layer's intermediates (BNK_DUMP_DIR, BNK_DUMP_LAYER; eager mode only): appended per window.
static void dbg(cudaStream_t st, int rank, int il, const char * name, const float * dev, size_t n) {
    static const char * dir = getenv("BNK_DUMP_DIR");
    static int layer = env_int("BNK_DUMP_LAYER", 0);
    if (!dir || il != layer) return;
    std::vector<float> h(n);
    CUDA_CHECK(cudaStreamSynchronize(st));
    CUDA_CHECK(cudaMemcpy(h.data(), dev, n * 4, cudaMemcpyDeviceToHost));
    FILE * f = fopen((std::string(dir) + "/" + name + (rank ? ".r" + std::to_string(rank) : "")).c_str(), "ab");
    fwrite(h.data(), 4, n, f);
    fclose(f);
}

// ------------------------------------------------------------------------------------------------ worker
DenseEngine::Worker::Worker() {
    th = std::thread([this]() {
        for (;;) {
            std::function<void()> f;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&]() { return busy || quit; });
                if (quit && !busy) return;
                f = std::move(job);
            }
            try {
                f();
            } catch (...) {
                std::lock_guard<std::mutex> lk(mu);
                err = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                busy = false;
            }
            cv.notify_all();
        }
    });
}

DenseEngine::Worker::~Worker() {
    {
        std::lock_guard<std::mutex> lk(mu);
        quit = true;
    }
    cv.notify_all();
    th.join();
}

void DenseEngine::Worker::post(std::function<void()> f) {
    wait();
    std::lock_guard<std::mutex> lk(mu);
    job = std::move(f);
    busy = true;
    cv.notify_all();
}

void DenseEngine::Worker::wait() {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&]() { return !busy; });
    if (err) {
        auto e = err;
        err = nullptr;
        std::rethrow_exception(e);
    }
}

DenseEngine::~DenseEngine() {
    worker_.reset();
    next_.reset();
    cudaSetDevice(dev_);
    for (auto & q : seqs_) {
        for (auto & k : q->ckpts)
            if (k.host) cudaFreeHost(k.host);
        for (auto & gl : q->commit_graphs) for (auto & gm : gl) for (auto & g : gm) if (g) cudaGraphExecDestroy(g);
    }
    for (auto & kv : fwd_graphs_) cudaGraphExecDestroy(kv.second.exec);
    if (h_par_) cudaFreeHost(h_par_);
    if (h_tok_) cudaFreeHost(h_tok_);
    if (d_par_) cudaFree(d_par_);
    for (float * b : xfer_in_)
        if (b) cudaFreeHost(b);
    for (float * b : pf_xfer_in_)
        if (b) cudaFreeHost(b);
    if (ev_out_) cudaEventDestroy(ev_out_);
    if (ev_t0_) cudaEventDestroy(ev_t0_);
    if (ev_t1_) cudaEventDestroy(ev_t1_);
    for (int pt = 0; pt < 2; ++pt) {
        for (int par = 0; par < 2; ++par) {
            if (tp_h_[pt][par]) cudaFreeHost(tp_h_[pt][par]);
            if (tp_pfh_[pt][par]) cudaFreeHost(tp_pfh_[pt][par]);
        }
        for (cudaEvent_t e : tp_ev_[pt]) cudaEventDestroy(e);
        for (cudaEvent_t e : tp_pfev_[pt]) cudaEventDestroy(e);
    }
    if (tp_fork_) cudaEventDestroy(tp_fork_);
    if (tp_join_) cudaEventDestroy(tp_join_);
}

cudaStream_t DenseEngine::fork(int i) {
    if (!side_) return st_;
    CUDA_CHECK(cudaEventRecord(ev_fork_[i], st_));
    CUDA_CHECK(cudaStreamWaitEvent(st_side_[i], ev_fork_[i], 0));
    return st_side_[i];
}

void DenseEngine::join(int i) {
    if (!side_) return;
    CUDA_CHECK(cudaEventRecord(ev_join_[i], st_side_[i]));
    CUDA_CHECK(cudaStreamWaitEvent(st_, ev_join_[i], 0));
}

std::vector<int> DenseEngine::devices() const {
    std::vector<int> d;
    for (const DenseEngine * s = this; s; s = s->next_.get())
        if (std::find(d.begin(), d.end(), s->dev_) == d.end()) d.push_back(s->dev_);
    return d;
}

StageInfo DenseEngine::stage_info(int i) const {
    const DenseEngine * s = tp() ? (i == 0 ? this : nullptr) : stage(i);
    if (!s) throw std::runtime_error("no stage " + std::to_string(i));
    return StageInfo{s->dev_, s->l0_, s->l1_, 0, 0, s->gpu_ms_, s->gpu_forwards_, s->pf_ms_, s->pf_tokens_};
}

// ------------------------------------------------------------------------------------------------ loading
// Layers balanced by the bytes each stage holds (the head counts on the last stage, the drafter's block too).
std::vector<int> DenseEngine::plan_split(const std::string & path, const EngineOptions & opt, const std::vector<int> & gpus) {
    const int n = (int) gpus.size();
    GgufModel g;
    g.open(path);
    const Config c = load_dense_config(g);
    std::vector<int> bounds{0};
    if (!opt.split.empty()) {
        if ((int) opt.split.size() != n - 1) throw std::runtime_error("--split: give one layer per stage after the first");
        for (int l : opt.split) {
            if (l <= bounds.back() || l >= c.n_layer) throw std::runtime_error("--split: layers out of order");
            bounds.push_back(l);
        }
        bounds.push_back(c.n_layer);
        return bounds;
    }
    std::vector<double> bytes(c.n_layer, 0);
    double head = 0;
    for (const TensorRef & t : g.tensors()) {
        if (t.name.rfind("blk.", 0) == 0) {
            const int il = atoi(t.name.c_str() + 4);
            if (il < c.n_layer) bytes[il] += (double) t.nbytes;
            else if (!opt.mtp.empty()) head += (double) t.nbytes;   // the MTP block, on the last stage
        } else if (t.name == "output.weight" || t.name == "output_norm.weight") {
            head += (double) t.nbytes;
        }
    }
    double total = head;
    for (double b : bytes) total += b;
    const double per = total / n;
    double acc = 0;
    for (int il = 0; il < c.n_layer && (int) bounds.size() < n; ++il) {
        acc += bytes[il];
        if (acc >= per * bounds.size()) bounds.push_back(il + 1);
    }
    while ((int) bounds.size() < n) bounds.push_back(bounds.back());
    bounds.push_back(c.n_layer);
    for (int i = 1; i < (int) bounds.size(); ++i)
        if (bounds[i] <= bounds[i - 1]) throw std::runtime_error("plan_split: a stage without layers");
    return bounds;
}

void DenseEngine::load(const std::string & path, const EngineOptions & opt) {
    std::vector<int> gpus = opt.gpus;
    if (gpus.empty()) {
        int d = 0;
        CUDA_CHECK(cudaGetDevice(&d));
        gpus = {d};
    }
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (gpus.size() == 1 && gpus[0] < 0) {
        gpus.clear();
        for (int d = 0; d < count; ++d) {
            cudaDeviceProp pr;
            CUDA_CHECK(cudaGetDeviceProperties(&pr, d));
            if (pr.major >= 7) gpus.push_back(d);
        }
        if (gpus.empty()) throw std::runtime_error("no CUDA device of compute capability 7.0 or newer");
    }
    for (int d : gpus)
        if (d < 0 || d >= count) throw std::runtime_error("no CUDA device " + std::to_string(d) + " (" + std::to_string(count) + " visible)");
    const bool tpar = opt.tp > 0 && gpus.size() > 1;
    if (tpar && gpus.size() != 2) throw std::runtime_error("tensor parallel: two GPUs (--gpus 0,1)");
    const std::vector<int> bounds = gpus.size() > 1 && !tpar ? plan_split(path, opt, gpus) : std::vector<int>{0, -1, -1};
    DenseEngine * s = this, * prev = nullptr;
    for (size_t k = 0; k < gpus.size(); ++k) {
        s->head_ = this;
        s->dev_ = gpus[k];
        if (tpar) {   // every layer on each rank, sliced; rank 0 holds the head and drafts
            s->tp_n_ = (int) gpus.size();
            s->tp_rank_ = (int) k;
            s->l0_ = 0;
            s->l1_ = -1;
            s->last_ = k == 0;
            if (opt.verbose) fprintf(stderr, "bnk: tensor parallel slice %zu of %zu: GPU %d\n", k + 1, gpus.size(), s->dev_);
        } else {
            s->l0_ = bounds[k];
            s->l1_ = bounds[k + 1];
            s->last_ = k + 1 == gpus.size();
            if (opt.verbose && gpus.size() > 1)
                fprintf(stderr, "bnk: stage %zu of %zu: GPU %d, layers %d-%d\n", k + 1, gpus.size(), s->dev_, s->l0_, s->l1_ - 1);
        }
        if (prev) {
            prev->peer_ = s;
            s->peer_ = prev;
        }
        s->load_stage(path, opt);
        if (prev && tpar) tp_enable_p2p(prev, s, opt.verbose);
        if (prev && !tpar) {
            for (int l = 0; l < kLanes; ++l) prev->xfer_out_[l] = s->xfer_in_[l];
            prev->pf_xfer_out_[0] = s->pf_xfer_in_[0];
            prev->pf_xfer_out_[1] = s->pf_xfer_in_[1];
        }
        if (k + 1 < gpus.size()) {
            prev = s;
            s->next_ = std::make_unique<DenseEngine>();
            s = s->next_.get();
        }
    }
    CUDA_CHECK(cudaSetDevice(dev_));
}

// Both ranks loaded: peer access both ways if the driver allows it (BNK_TP_P2P=0 keeps the pinned host path).
// With the host IOMMU translating, peer writes are blocked and a copy takes ~50 ms, so check the exchange time
// (build/bench_xfer) before trusting it.
void DenseEngine::tp_enable_p2p(DenseEngine * a, DenseEngine * b, bool verbose) {
    if (getenv("BNK_TP_P2P") && atoi(getenv("BNK_TP_P2P")) == 0) return;
    int ab = 0, ba = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&ab, a->dev_, b->dev_));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&ba, b->dev_, a->dev_));
    if (!ab || !ba) return;
    for (DenseEngine * r : {a, b}) {
        DeviceScope ds(r->dev_);
        const cudaError_t e = cudaDeviceEnablePeerAccess((r == a ? b : a)->dev_, 0);
        if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else CUDA_CHECK(e);
        for (int pt = 0; pt < 2; ++pt) {
            for (int par = 0; par < 2; ++par) {
                r->tp_d_[pt][par].alloc(r->tp_tmp_.n);
                if (r->tp_mine16_.n) r->tp_pfd_[pt][par].alloc(r->tp_mine16_.n);
            }
        }
        r->tp_p2p_ = true;
    }
    if (verbose) fprintf(stderr, "bnk: tensor parallel exchange: peer copies GPU %d <-> GPU %d\n", a->dev_, b->dev_);
}

void DenseEngine::load_stage(const std::string & path, const EngineOptions & opt) {
    opt_ = opt;
    kv_set_q8(opt.kv8);
    const bool head = head_ == this;
    CUDA_CHECK(cudaSetDevice(dev_));
    if (cudaSetDeviceFlags(cudaDeviceMapHost) != cudaSuccess) cudaGetLastError();
    CUDA_CHECK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&st_commit_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_verified_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_committed_, cudaEventDisableTiming));
    side_ = !getenv("BNK_SIDE_STREAMS") || atoi(getenv("BNK_SIDE_STREAMS")) != 0;
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&st_side_[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_fork_[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_join_[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_out_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&ev_t0_));
    CUDA_CHECK(cudaEventCreate(&ev_t1_));
    // --mtp (any value: the block lives in the model file) makes the last stage load the drafter; it embeds its
    // tokens and shares the head, so that stage holds the token table too
    const bool want_mtp = last_ && !opt.mtp.empty();
    model_.load(path, opt.verbose, l0_, l1_, first() || want_mtp, last_, want_mtp, tp_n_, tp_rank_);
    if (l1_ < 0) l1_ = model_.cfg.n_layer;
    const Config & c = model_.lc;   // this slice's dimensions (the whole model's unless tensor parallel)
    if (head) {
        layer_routed.assign(c.n_layer, 0);
        layer_misses.assign(c.n_layer, 0);
    }

    const int W = kMaxWindowDense, E = c.n_embd, FF = c.n_ff;
    res_.alloc(W * E); xn_.alloc(W * E); out_.alloc(W * E); ffn_out_.alloc(W * E);
    const int C = c.conv_channels(), VI = c.ssm_vheads * c.ssm_state;
    conv_out_.alloc(W * C); z_.alloc(W * VI); gdn_o_.alloc(W * VI); gdn_n_.alloc(W * VI);
    qfull_.alloc(W * c.n_head * c.head_dim * 2); k_.alloc(W * c.n_head_kv * c.head_dim);
    v_.alloc(W * c.n_head_kv * c.head_dim); q_.alloc(W * c.n_head * c.head_dim);
    attn_o_.alloc(W * c.n_head * c.head_dim);
    attn_scratch_.alloc(std::max(attention_scratch_floats(W, c.n_head, c.head_dim, opt.max_ctx),
                                 attention_decode_tc_scratch_floats(W, c.n_head, c.head_dim)));
    sg_.alloc(W * FF); su_.alloc(W * FF);
    if (last_) logits_.alloc((size_t) W * c.n_vocab);
    argmax_dev_.alloc(W);
    const int64_t maxcols = std::max<int64_t>({(int64_t) E, (int64_t) FF, (int64_t) VI, (int64_t) c.n_head * c.head_dim});
    actq_.alloc(W * maxcols); actd_.alloc(W * maxcols / 32); acts_.alloc(W * maxcols / 16); acth_.alloc(W * maxcols);
    act_.q = actq_; act_.d = actd_; act_.s = acts_; act_.h = acth_;

    CUDA_CHECK(cudaHostAlloc((void **) &h_par_, kMaxSlots * sizeof(WinParams), cudaHostAllocMapped));
    CUDA_CHECK(cudaMalloc(&d_par_, kMaxSlots * sizeof(WinParams)));
    CUDA_CHECK(cudaHostAlloc((void **) &h_tok_, W * 4, cudaHostAllocMapped));
    pf_max_ = std::max(opt.prefill_chunk, opt.prefill_chunk_max);
    if (!first()) {
        for (float *& b : xfer_in_) CUDA_CHECK(cudaHostAlloc((void **) &b, (size_t) W * E * 4, cudaHostAllocPortable));
        if (pf_max_ > 0)
            for (float *& b : pf_xfer_in_) CUDA_CHECK(cudaHostAlloc((void **) &b, (size_t) pf_max_ * E * 4, cudaHostAllocPortable));
    }
    if (tp()) {   // the exchange buffers with the peer
        for (int pt = 0; pt < 2; ++pt) {
            for (int par = 0; par < 2; ++par) {
                CUDA_CHECK(cudaHostAlloc((void **) &tp_h_[pt][par], (size_t) W * E * 4, cudaHostAllocPortable));
                if (pf_max_ > 0)
                    CUDA_CHECK(cudaHostAlloc((void **) &tp_pfh_[pt][par], (size_t) pf_max_ * E * 2, cudaHostAllocPortable));
            }
            tp_ev_[pt].resize(c.n_layer);
            tp_pfev_[pt].resize(c.n_layer);
            for (int il = 0; il < c.n_layer; ++il) {
                CUDA_CHECK(cudaEventCreateWithFlags(&tp_ev_[pt][il], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&tp_pfev_[pt][il], cudaEventDisableTiming));
            }
        }
        tp_tmp_.alloc((size_t) W * E);
        if (pf_max_ > 0) {
            tp_mine16_.alloc((size_t) pf_max_ * E);
            tp_peer16_.alloc((size_t) pf_max_ * E);
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&tp_fork_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&tp_join_, cudaEventDisableTiming));
    }
    for (int l = 0; l < kLanes; ++l) {
        gdn_co_[l].resize(c.n_layer);
        gdn_g_[l].resize(c.n_layer);
        gdn_b_[l].resize(c.n_layer);
    }
    size_t state_bytes = 0, kv_cell_bytes = 0;
    for (int il = l0_; il < l1_; ++il) {
        if (c.is_attn(il)) {
            kv_cell_bytes += 2 * (size_t) c.n_head_kv * kv_row_halves(c.head_dim) * sizeof(half);
        } else {
            for (int l = 0; l < (first() && last_ ? 1 : kLanes); ++l) {
                gdn_co_[l][il].alloc((size_t) W * C);
                gdn_g_[l][il].alloc((size_t) W * c.ssm_vheads);
                gdn_b_[l][il].alloc((size_t) W * c.ssm_vheads);
            }
            state_bytes += ((size_t) (c.ssm_conv - 1 + W) * C + (size_t) c.ssm_vheads * c.ssm_state * c.ssm_state) * 4;
        }
    }
    // Everything elastic (the context, the drafter's K/V) draws on one budget: what is free now less a reserve
    // for cuBLAS and graph workspaces.
    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    const size_t reserve = (size_t) (opt.vram_reserve_gib * 1073741824.0);
    budget_.set_limit(free_b > reserve ? free_b - reserve : 0);
    budget_.set_reserve(reserve);
    seqs_.clear();
    for (int i = 0; i < std::max(1, opt.slots); ++i) {
        seqs_.push_back(std::make_unique<Seq>());
        alloc_seq(*seqs_.back());
    }
    cur_ = seqs_[0].get();
    cur_->used = true;
    if (want_mtp && model_.mtp.present && opt.dflash.empty())
        mtp_.load(model_, opt.max_ctx, st_, opt.verbose, opt.draft_vocab, &budget_, (int) seqs_.size());
    if (!opt.dflash.empty() && last_) {
        if (next_ && !tp()) throw std::runtime_error("dflash: needs one stage or tensor parallel (its taps span the layers)");
        dflash_.load(opt.dflash, model_, (int) seqs_.size(), st_, opt.verbose);
        tap_acc_.alloc((size_t) kMaxWindowDense * E);
        tap_tmp_.alloc((size_t) kMaxWindowDense * E);
        dflash_.bind(tap_acc_, res_);
    }
    if (opt.verbose)
        fprintf(stderr, "bnk: %scontext %d: %.0f KiB of %sKV per token (%.2f GiB when full, mapped as used), "
                        "recurrent state %.2f GiB%s\n",
                tp() ? ("slice " + std::to_string(tp_rank_ + 1) + ": ").c_str()
                     : first() && last_ ? "" : ("layers " + std::to_string(l0_) + "-" + std::to_string(l1_ - 1) + ": ").c_str(),
                opt.max_ctx, kv_cell_bytes / 1024.0, opt.kv8 ? "8-bit " : "", (double) kv_cell_bytes * opt.max_ctx / 1073741824.0,
                state_bytes / 1073741824.0, seqs_.size() > 1 ? (", " + std::to_string(seqs_.size()) + " conversation slots").c_str() : "");
    if (pf_max_ > 0) prefill_alloc(pf_max_);
    ensure_ctx_here(std::min(opt.max_ctx, kCtxBaseline));
    reset_here();
    if (mtp_.loaded())
        for (auto & q : seqs_)
            if (!q->mtp_R) CUDA_CHECK(cudaMalloc(&q->mtp_R, (size_t) E * 4));
    // every slot's snapshot buffers pinned now (pinning during a request would cost its turn)
    const size_t ck = state_floats() * 4;
    for (auto & q : seqs_)
        while ((int) q->ckpts.size() < kMaxCheckpoints) {
            q->ckpts.emplace_back();
            CUDA_CHECK(cudaHostAlloc((void **) &q->ckpts.back().host, ck, cudaHostAllocPortable));
        }
    if (opt.verbose)
        fprintf(stderr, "bnk: %s: snapshots %d x %d slots x %.0f MiB pinned\n",
                tp() ? ("slice " + std::to_string(tp_rank_ + 1)).c_str() : ("layers " + std::to_string(l0_) + "-" + std::to_string(l1_ - 1)).c_str(),
                kMaxCheckpoints, (int) seqs_.size(), ck / 1048576.0);
}

void DenseEngine::alloc_seq(Seq & q) {
    const Config & c = model_.lc;
    const int W = kMaxWindowDense, C = c.conv_channels();
    q.conv_buf.resize(c.n_layer);
    q.ssm_state.resize(c.n_layer);
    q.kc.resize(c.n_layer);
    q.vc.resize(c.n_layer);
    q.kv_k = std::vector<ElasticBuf>(c.n_layer);
    q.kv_v = std::vector<ElasticBuf>(c.n_layer);
    for (int il = l0_; il < l1_; ++il) {
        if (c.is_attn(il)) {
            const size_t kv = (size_t) opt_.max_ctx * c.n_head_kv * kv_row_halves(c.head_dim);
            q.kv_k[il].reserve(kv * sizeof(half), &budget_, "KV cache");
            q.kv_v[il].reserve(kv * sizeof(half), &budget_, "KV cache");
            q.kc[il].view(q.kv_k[il].as<half>(), kv);
            q.vc[il].view(q.kv_v[il].as<half>(), kv);
        } else {
            q.conv_buf[il].alloc((size_t) (c.ssm_conv - 1 + W) * C);
            q.ssm_state[il].alloc((size_t) c.ssm_vheads * c.ssm_state * c.ssm_state);
        }
    }
}

size_t DenseEngine::state_floats() {
    size_t floats = 0;
    each_state([&](float *, size_t n) { floats += n; });
    return floats;
}

void DenseEngine::use(Seq * q) {
    cur_ = q;
    for (size_t i = 0; i < seqs_.size(); ++i)
        if (seqs_[i].get() == q && mtp_.loaded()) mtp_.select((int) i);
}

int DenseEngine::current_slot() const {
    for (size_t i = 0; i < seqs_.size(); ++i)
        if (seqs_[i].get() == cur_) return (int) i;
    return -1;
}

void DenseEngine::select(int slot) {
    for (DenseEngine * s = this; s; s = s->next_.get()) s->select_here(slot);
}

void DenseEngine::select_here(int slot) {
    DeviceScope ds(dev_);
    if (slot < 0 || slot >= (int) seqs_.size()) throw std::runtime_error("select: no slot " + std::to_string(slot));
    use(seqs_[slot].get());
    if (dflash_.loaded()) dflash_.select(slot);
    if (!cur_->used) {
        cur_->used = true;
        reset_here();
    }
}

// ------------------------------------------------------------------------------------------------ context
void DenseEngine::ensure_ctx(int cells) {
    for (DenseEngine * s = this; s; s = s->next_.get()) s->ensure_ctx_here(cells);
}

void DenseEngine::ensure_ctx_here(int cells) {
    DeviceScope ds(dev_);
    cells = std::min(opt_.max_ctx, (cells + kCtxStep - 1) / kCtxStep * kCtxStep);
    if (cells <= cur_->ctx_mapped) return;
    map_ctx(cells);
}

void DenseEngine::map_ctx(int cells) {
    const Config & c = model_.lc;
    const bool grow = cells > cur_->ctx_mapped;
    for (int il = 0; il < c.n_layer; ++il) {
        if (!cur_->kv_k[il].reserved()) continue;
        const size_t kv = (size_t) cells * c.n_head_kv * kv_row_halves(c.head_dim) * sizeof(half);
        if (grow) {
            cur_->kv_k[il].ensure(kv);
            cur_->kv_v[il].ensure(kv);
        } else {
            cur_->kv_k[il].shrink_to(kv);
            cur_->kv_v[il].shrink_to(kv);
        }
    }
    if (mtp_.loaded()) {
        if (grow) mtp_.ensure_ctx(cells);
        else mtp_.release_ctx(cells);
    }
    cur_->ctx_mapped = cells;
}

size_t DenseEngine::kv_bytes_mapped() const {
    size_t b = 0;
    for (size_t il = 0; il < cur_->kv_k.size(); ++il) b += cur_->kv_k[il].mapped() + cur_->kv_v[il].mapped();
    if (mtp_.loaded()) b += mtp_.ctx_bytes_mapped();
    return b + (next_ ? next_->kv_bytes_mapped() : 0);
}

// ------------------------------------------------------------------------------------------------ state
void DenseEngine::checkpoint() {
    for (DenseEngine * s = this; s; s = s->next_.get()) s->checkpoint_here();
}

void DenseEngine::checkpoint_here() {
    DeviceScope ds(dev_);
    join_commit_here();
    const int p = pos();
    if (p == 0) return;
    for (auto & k : cur_->ckpts)
        if (k.pos == p) { k.age = ++ckpt_age_; return; }
    Checkpoint * slot = nullptr;
    for (auto & k : cur_->ckpts)
        if (k.pos < 0) { slot = &k; break; }
    if (!slot)
        slot = &*std::min_element(cur_->ckpts.begin(), cur_->ckpts.end(), [](const Checkpoint & a, const Checkpoint & b) { return a.age < b.age; });
    size_t off = 0;
    each_state([&](float * d, size_t n) {
        CUDA_CHECK(cudaMemcpyAsync(slot->host + off, d, n * 4, cudaMemcpyDeviceToHost, st_));
        off += n;
    });
    CUDA_CHECK(cudaStreamSynchronize(st_));
    slot->pos = p;
    slot->mtp_cell = cur_->mtp_cell;
    slot->age = ++ckpt_age_;
}

int DenseEngine::rollback(int max_pos) {
    const int r = rollback_here(max_pos);
    for (DenseEngine * s = next_.get(); s; s = s->next_.get())
        if (s->rollback_here(max_pos) != r) throw std::runtime_error("rollback: the pipeline stages disagree");
    return r;
}

int DenseEngine::rollback_here(int max_pos) {
    DeviceScope ds(dev_);
    join_commit_here();
    Checkpoint * best = nullptr;
    for (auto & k : cur_->ckpts)
        if (k.pos >= 0 && k.pos <= max_pos && k.pos <= pos() && (!best || k.pos > best->pos)) best = &k;
    if (!best) return -1;
    size_t off = 0;
    each_state([&](float * d, size_t n) {
        CUDA_CHECK(cudaMemcpyAsync(d, best->host + off, n * 4, cudaMemcpyHostToDevice, st_));
        off += n;
    });
    CUDA_CHECK(cudaStreamSynchronize(st_));
    cur_->history.resize(best->pos);
    cur_->mtp_cell = best->mtp_cell;
    if (dflash_.loaded()) dflash_.restart(best->pos);
    best->age = ++ckpt_age_;
    for (auto & k : cur_->ckpts)
        if (k.pos > best->pos) k.pos = -1;
    return best->pos;
}

// How much of `prompt` the current slot's state can keep (Generator::start's rule): all of the history when the
// prompt extends it, else the last snapshot inside the shared prefix.
int DenseEngine::reusable(const std::vector<int32_t> & prompt) const {
    const auto & h = cur_->history;
    size_t common = 0;
    while (common < h.size() && common < prompt.size() && h[common] == prompt[common]) ++common;
    if (!h.empty() && common == h.size() && h.size() < prompt.size()) return (int) h.size();
    if (common == 0) return 0;
    const int lim = (int) std::min(common, prompt.size() - 1);
    int best = 0;
    for (const auto & k : cur_->ckpts)
        if (k.pos >= 0 && k.pos <= lim && k.pos <= (int) h.size()) best = std::max(best, k.pos);
    return best;
}

void DenseEngine::reset() {
    for (DenseEngine * s = this; s; s = s->next_.get()) s->reset_here();
}

void DenseEngine::reset_here() {
    DeviceScope ds(dev_);
    join_commit_here();
    const Config & c = model_.lc;
    cur_->history.clear();
    for (auto & k : cur_->ckpts) k.pos = -1;
    cur_->mtp_cell = -1;
    if (dflash_.loaded()) dflash_.restart(0);
    for (int il = l0_; il < l1_; ++il) {
        if (!c.is_attn(il)) {
            CUDA_CHECK(cudaMemsetAsync(cur_->conv_buf[il].p, 0, cur_->conv_buf[il].n * 4, st_));
            CUDA_CHECK(cudaMemsetAsync(cur_->ssm_state[il].p, 0, cur_->ssm_state[il].n * 4, st_));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    if (cur_->ctx_mapped > kCtxBaseline) map_ctx(std::min(opt_.max_ctx, kCtxBaseline));
}

// ------------------------------------------------------------------------------------------------ the forward
void DenseEngine::gdn(int il, int T) {
    const Config & c = model_.lc;
    const DenseLayer & L = model_.layers[il];
    const int E = c.n_embd, C = c.conv_channels(), K = c.ssm_conv, S = c.ssm_state, nv = c.ssm_vheads;
    float * co = gdn_co_[lane_][il], * gg = gdn_g_[lane_][il], * gb = gdn_b_[lane_][il];
    {   // beta / alpha (needed at the prep) and the gate (at the norm) beside wqkv
        cudaStream_t sb = fork(0);
        if (L.ssm_ba.valid()) gemv(L.ssm_ba, &act_, nullptr, 0, T, gb, nv, false, sb, GemvSplit{nv, gg, nv});
        else {
            gemv(L.ssm_beta, &act_, nullptr, 0, T, gb, nv, false, sb);
            gemv(L.ssm_alpha, &act_, nullptr, 0, T, gg, nv, false, sb);
        }
        gemv(L.wgate, &act_, nullptr, 0, T, z_, nv * S, false, sb);
    }
    // the new conv inputs go behind each window's own K-1 history rows (one window: straight there)
    const bool one = wins_.size() == 1;
    float * qkv = one ? wins_[0].q->conv_buf[il].p + (size_t) (K - 1) * C : conv_out_.p;
    gemv(L.wqkv, &act_, nullptr, 0, T, qkv, C, false, st_);
    if (!one)
        for (const Win & w : wins_)
            CUDA_CHECK(cudaMemcpyAsync(w.q->conv_buf[il].p + (size_t) (K - 1) * C, conv_out_.p + (size_t) w.r0 * C,
                                       (size_t) w.T * C * 4, cudaMemcpyDeviceToDevice, st_));
    join(0);
    dbg(st_, tp_rank_, il, "qkv", qkv, (size_t) T * C);
    dbg(st_, tp_rank_, il, "z", z_, (size_t) T * nv * S);
    dbg(st_, tp_rank_, il, "beta", gb, (size_t) T * nv);
    dbg(st_, tp_rank_, il, "alpha", gg, (size_t) T * nv);
    for (const Win & w : wins_) {
        float * cw = co + (size_t) w.r0 * C, * gw = gg + (size_t) w.r0 * nv, * bw = gb + (size_t) w.r0 * nv;
        gdn_conv(w.q->conv_buf[il], F(L.ssm_conv1d), cw, w.T, C, K, st_);
        dbg(st_, tp_rank_, il, "conv_silu", cw, (size_t) w.T * C);
        gdn_prep(cw, w.T, C, c.ssm_groups, nv, S, gw, bw, F(L.ssm_dt), F(L.ssm_a), c.rms_eps, st_);
        gdn_recurrence(cw, C, gw, bw, w.q->ssm_state[il], gdn_o_.p + (size_t) w.r0 * nv * S, w.T, c.ssm_groups, nv, S,
                       w.commit ? w.T : 0, st_);
    }
    // the gated norm inside the quantization pass of ssm_out's input
    if (!use_ref_gemv() && quantize_gated_norm_silu(gdn_o_, z_, F(L.ssm_norm), T, nv, S, c.rms_eps, act_, st_)) {
        gemv(L.ssm_out, &act_, nullptr, 0, T, out_, E, false, st_);
    } else {
        gated_rmsnorm_silu(gdn_o_, z_, F(L.ssm_norm), gdn_n_, T, nv, S, c.rms_eps, st_);
        quantize_act(gdn_n_, nv * S, T, nv * S, act_, st_);
        gemv(L.ssm_out, &act_, nullptr, 0, T, out_, E, false, st_);
    }
    for (const Win & w : wins_)
        if (w.commit) shift_rows(w.q->conv_buf[il], C, w.T, K - 1, st_);
}

void DenseEngine::attn(int il, int T) {
    const Config & c = model_.lc;
    const DenseLayer & L = model_.layers[il];
    const int E = c.n_embd, H = c.n_head, Hkv = c.n_head_kv, D = c.head_dim;
    gemv(L.wq, &act_, nullptr, 0, T, qfull_, H * D * 2, false, st_);
    if (L.wkv.valid()) gemv(L.wkv, &act_, nullptr, 0, T, k_, Hkv * D, false, st_, GemvSplit{Hkv * D, v_, (int64_t) Hkv * D});
    else {
        gemv(L.wk, &act_, nullptr, 0, T, k_, Hkv * D, false, st_);
        gemv(L.wv, &act_, nullptr, 0, T, v_, Hkv * D, false, st_);
    }
    dbg(st_, tp_rank_, il, "Qcur_full", qfull_, (size_t) T * H * D * 2);
    dbg(st_, tp_rank_, il, "Vcur", v_, (size_t) T * Hkv * D);
    for (size_t wi = 0; wi < wins_.size(); ++wi) {
        const Win & w = wins_[wi];
        const int * pos0 = &d_par_[wi].pos0;
        const size_t r = (size_t) w.r0;
        float * qw = q_.p + r * H * D, * qf = qfull_.p + r * H * D * 2, * ao = attn_o_.p + r * H * D;
        attn_prep(qf, k_.p + r * Hkv * D, v_.p + r * Hkv * D, F(L.q_norm), F(L.k_norm), qw, w.q->kc[il], w.q->vc[il],
                  w.T, H, Hkv, D, c.n_rot, c.rope_base, pos0, c.rms_eps, st_);
        // the tensor-core split-KV kernel (one pass over the cache for the window's rows and the KV head's
        // query heads); BNK_ATTN_TC=0: the one-row-per-block kernel
        static const bool tc = !getenv("BNK_ATTN_TC") || atoi(getenv("BNK_ATTN_TC")) != 0;
        if (!(tc && attention_decode_tc(qw, w.q->kc[il], w.q->vc[il], qf, ao, w.T, H, Hkv, D, pos0, 1.f / sqrtf((float) D),
                                        attn_scratch_, st_)))
            attention(qw, w.q->kc[il], w.q->vc[il], qf, ao, w.T, H, Hkv, D, pos0, 1.f / sqrtf((float) D), attn_scratch_, st_);
    }
    dbg(st_, tp_rank_, il, "attn_gated", attn_o_, (size_t) T * H * D);
    quantize_act(attn_o_, H * D, T, H * D, act_, st_);
    gemv(L.wo, &act_, nullptr, 0, T, out_, E, false, st_);
}

void DenseEngine::ffn(int il, int T) {
    const Config & c = model_.lc;
    const DenseLayer & L = model_.layers[il];
    const int E = c.n_embd, FF = c.n_ff;
    if (L.gu.valid()) gemv(L.gu, &act_, nullptr, 0, T, sg_, FF, false, st_, GemvSplit{FF, su_, FF});
    else {
        gemv(L.gate, &act_, nullptr, 0, T, sg_, FF, false, st_);
        gemv(L.up, &act_, nullptr, 0, T, su_, FF, false, st_);
    }
    // silu(gate) * up inside the quantization pass of down's input
    quantize_act_silu_mul(sg_, su_, FF, T, FF, act_, st_);
    gemv(L.down, &act_, nullptr, 0, T, ffn_out_, E, false, st_);
}

void DenseEngine::head(int T) {
    const Config & c = model_.cfg;
    dense_norm_q(res_, pend_, F(model_.output_norm), xn_, T, c.n_embd, c.rms_eps, act_, st_);
    pend_ = nullptr;
    gemv(model_.output, &act_, nullptr, 0, T, logits_, c.n_vocab, false, st_);
}

// res (+ pending) -> norm -> attention / DeltaNet -> out_ (a partial sum under tensor parallel)
void DenseEngine::layer_block(int il, int T) {
    const Config & c = model_.lc;
    const DenseLayer & L = model_.layers[il];
    const int E = c.n_embd;
    dense_norm_q(res_, pend_, F(L.attn_norm), xn_, T, E, c.rms_eps, act_, st_);
    pend_ = nullptr;
    dbg(st_, tp_rank_, il, "input", res_, (size_t) T * E);
    dbg(st_, tp_rank_, il, "attn_norm", xn_, (size_t) T * E);
    if (L.attn) attn(il, T); else gdn(il, T);
    dbg(st_, tp_rank_, il, "attn_output", out_, (size_t) T * E);
}

// res += out_ -> norm -> FFN -> ffn_out_ (a partial sum under tensor parallel)
void DenseEngine::layer_ffn(int il, int T) {
    const Config & c = model_.lc;
    const DenseLayer & L = model_.layers[il];
    const int E = c.n_embd;
    dense_norm_q(res_, out_, F(L.post_norm), xn_, T, E, c.rms_eps, act_, st_);
    dbg(st_, tp_rank_, il, "attn_residual", res_, (size_t) T * E);
    dbg(st_, tp_rank_, il, "attn_post_norm", xn_, (size_t) T * E);
    ffn(il, T);
    dbg(st_, tp_rank_, il, "ffn_out", ffn_out_, (size_t) T * E);
}

// res -> norm -> attention / DeltaNet -> (+) -> norm -> FFN -> (+): each add rides in the next norm's pass; the
// stage's last layer adds explicitly (its residual leaves the stage)
void DenseEngine::layer_forward(int il, int T) {
    const Config & c = model_.lc;
    const int E = c.n_embd;
    layer_block(il, T);
    layer_ffn(il, T);
    pend_ = ffn_out_;
    tap(il, T);
    if (il + 1 == l1_ && !last_) {
        add_rows(res_, pend_, (int64_t) T * E, st_);
        pend_ = nullptr;
    }
    if (head_->dump_all) {
        if (pend_) {
            add_rows(res_, pend_, (int64_t) T * E, st_);
            pend_ = nullptr;
        }
        auto & dl = head_->dumped_layers;
        dl.resize(c.n_layer);
        dl[il].resize((size_t) T * E);
        CUDA_CHECK(cudaMemcpyAsync(dl[il].data(), res_.p, dl[il].size() * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
    }
}

// the window's inputs: its parameters, and the token embeddings (first stage) or the previous stage's rows
void DenseEngine::fwd_in(int T) {
    const Config & c = model_.cfg;
    const size_t rows = (size_t) T * c.n_embd * 4;
    CUDA_CHECK(cudaMemcpyAsync(d_par_, h_par_, wins_.size() * sizeof(WinParams), cudaMemcpyHostToDevice, st_));
    if (first()) dequant_gather(model_.tok_embd, h_tok_, T, res_, st_);
    else CUDA_CHECK(cudaMemcpyAsync(res_.p, xfer_in_[lane_], rows, cudaMemcpyHostToDevice, st_));
    pend_ = nullptr;
}

void DenseEngine::enqueue_forward(int T) {
    const size_t rows = (size_t) T * model_.cfg.n_embd * 4;
    fwd_in(T);
    for (int il = l0_; il < l1_; ++il) layer_forward(il, T);
    if (last_) head(T);
    else CUDA_CHECK(cudaMemcpyAsync(xfer_out_[lane_], res_.p, rows, cudaMemcpyDeviceToHost, st_));
}

// ---- tensor parallel: the all-reduce with the peer in two phases, so that each rank records its event before
// the other waits for it (a capture may not wait on an event recorded later). tp_send copies this rank's partial
// sum out to the pinned buffer of (point, layer parity) and marks it; tp_recv, queued after the peer's tp_send,
// waits for the peer's mark, brings its partial in and adds it. Both ranks add the same two fp32 values, so their
// residuals stay bit for bit identical.
void DenseEngine::tp_send(const float * buf, int rows, int point, int il) {
    const int par = il & 1;
    const size_t bytes = (size_t) rows * model_.cfg.n_embd * 4;
    if (tp_p2p_)   // pushed straight into the peer's inbox (it adds from there; one copy instead of two)
        CUDA_CHECK(cudaMemcpyAsync(peer_->tp_d_[point][par].p, buf, bytes, cudaMemcpyDeviceToDevice, st_));
    else
        CUDA_CHECK(cudaMemcpyAsync(tp_h_[point][par], buf, bytes, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaEventRecord(tp_ev_[point][il], st_));
}

void DenseEngine::tp_recv(float * buf, int rows, int point, int il) {
    const int par = il & 1;
    const size_t n = (size_t) rows * model_.cfg.n_embd;
    CUDA_CHECK(cudaStreamWaitEvent(st_, peer_->tp_ev_[point][il], 0));
    if (tp_p2p_) {
        add_rows(buf, tp_d_[point][par], (int64_t) n, st_);
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(tp_tmp_.p, peer_->tp_h_[point][par], n * 4, cudaMemcpyHostToDevice, st_));
    add_rows(buf, tp_tmp_, (int64_t) n, st_);
}

// prompt chunks exchange fp16 partials (half the PCIe bytes): res += mine16 + peer16 on both ranks
void DenseEngine::tp_pf_send(const float * partial, int rows, int point, int il) {
    const int par = il & 1;
    const size_t n = (size_t) rows * model_.cfg.n_embd;
    if (tp_p2p_) {   // converted straight into the staging buffer the peer pulls from
        f32_to_f16(partial, tp_pfd_[point][par], (int64_t) n, st_);
    } else {
        f32_to_f16(partial, tp_mine16_, (int64_t) n, st_);
        CUDA_CHECK(cudaMemcpyAsync(tp_pfh_[point][par], tp_mine16_.p, n * 2, cudaMemcpyDeviceToHost, st_));
    }
    CUDA_CHECK(cudaEventRecord(tp_pfev_[point][il], st_));
}

void DenseEngine::tp_pf_recv(float * res, int rows, int point, int il) {
    const int par = il & 1;
    const size_t n = (size_t) rows * model_.cfg.n_embd;
    CUDA_CHECK(cudaStreamWaitEvent(st_, peer_->tp_pfev_[point][il], 0));
    if (tp_p2p_) {
        CUDA_CHECK(cudaMemcpyAsync(tp_peer16_.p, peer_->tp_pfd_[point][par].p, n * 2, cudaMemcpyDeviceToDevice, st_));
        add_rows_f16x2(res, tp_pfd_[point][par], tp_peer16_, (int64_t) n, st_);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(tp_peer16_.p, peer_->tp_pfh_[point][par], n * 2, cudaMemcpyHostToDevice, st_));
        add_rows_f16x2(res, tp_mine16_, tp_peer16_, (int64_t) n, st_);
    }
}

// Rank 0 drives both ranks' streams layer by layer (each phase on each rank, then the exchange): captured into
// one graph over both devices, or run eagerly.
void DenseEngine::tp_enqueue_forward(int T) {
    DenseEngine * rk[2] = {this, peer_};
    const Config & c = model_.cfg;
    for (DenseEngine * r : rk) {
        DeviceScope ds(r->dev_);
        r->fwd_in(T);
    }
    for (int il = 0; il < c.n_layer; ++il) {
        for (DenseEngine * r : rk) {
            DeviceScope ds(r->dev_);
            r->layer_block(il, T);
            r->tp_send(r->out_, T, 0, il);
        }
        for (DenseEngine * r : rk) {
            DeviceScope ds(r->dev_);
            r->tp_recv(r->out_, T, 0, il);
        }
        for (DenseEngine * r : rk) {
            DeviceScope ds(r->dev_);
            r->layer_ffn(il, T);
            r->tp_send(r->ffn_out_, T, 1, il);
        }
        for (DenseEngine * r : rk) {
            DeviceScope ds(r->dev_);
            r->tp_recv(r->ffn_out_, T, 1, il);
            r->pend_ = r->ffn_out_;
            r->tap(il, T);
        }
        if (head_->dump_all) {   // the residual after the layer (rank 0's; the peer's is the same)
            DeviceScope ds(dev_);
            add_rows(res_, pend_, (int64_t) T * c.n_embd, st_);
            pend_ = nullptr;
            auto & dl = head_->dumped_layers;
            dl.resize(c.n_layer);
            dl[il].resize((size_t) T * c.n_embd);
            CUDA_CHECK(cudaMemcpyAsync(dl[il].data(), res_.p, dl[il].size() * 4, cudaMemcpyDeviceToHost, st_));
            CUDA_CHECK(cudaStreamSynchronize(st_));
            {
                DeviceScope d2(peer_->dev_);
                add_rows(peer_->res_, peer_->pend_, (int64_t) T * c.n_embd, peer_->st_);
                peer_->pend_ = nullptr;
            }
        }
    }
    {
        DeviceScope ds(dev_);
        head(T);   // rank 0: the head (its norm adds the last FFN)
    }
    {   // the peer's residual complete too (its drafter-free copy; BNK_TP_DEBUG compares the two)
        DeviceScope ds(peer_->dev_);
        if (peer_->pend_) add_rows(peer_->res_, peer_->pend_, (int64_t) T * c.n_embd, peer_->st_);
        peer_->pend_ = nullptr;
    }
}

void DenseEngine::forward(const int32_t * tokens, int T, bool commit_all) {
    forward_batch({BatchWin{current_slot(), tokens, T, commit_all}});
}

void DenseEngine::forward_batch(const std::vector<BatchWin> & batch_in) {
    std::vector<BatchWin> batch = batch_in;
    std::sort(batch.begin(), batch.end(), [](const BatchWin & a, const BatchWin & b) { return a.slot < b.slot; });
    if (batch.empty() || batch.size() > (size_t) kMaxSlots) throw std::runtime_error("forward: bad batch");
    const double t0 = now_ms();
    // tensor parallel: the peer books its windows first (rank 0's capture bakes the peer's window buffers)
    if (tp()) peer_->fwd_prepare(batch);
    for (DenseEngine * s = this; s; s = s->next_.get())
        if (!(tp() && s == peer_)) s->fwd_prepare(batch);
    cudaEvent_t after = nullptr;
    for (DenseEngine * s = this; s; s = s->next_.get()) {
        s->fwd_launch(after);
        after = s->ev_out_;
    }
    for (DenseEngine * s = this; s; s = s->next_.get()) s->fwd_end();
    times.total_ms += now_ms() - t0;
    times.calls++;
}

void DenseEngine::fwd_prepare(const std::vector<BatchWin> & batch, int lane) {
    DeviceScope ds(dev_);
    join_commit_here();
    if (lane < 0 || lane >= kLanes || (lane > 0 && first() && last_))
        throw std::runtime_error("forward: no lane " + std::to_string(lane));
    lane_ = lane;
    Seq * keep = cur_;
    int R = 0;
    for (const auto & b : batch) {
        if (b.T < 1) throw std::runtime_error("forward: bad window");
        R += b.T;
    }
    if (R > kMaxWindowDense) throw std::runtime_error("forward: more than kMaxWindowDense rows");
    fwd_t0_ = now_ms();
    wins_.clear();
    ++seq_;
    std::vector<int> key{(int) batch.size()};
    for (const auto & b : batch) {
        Seq * q = seqs_.at(b.slot).get();
        if (q->pending_T) throw std::runtime_error("forward: the previous verify window was not committed");
        for (const Win & w : wins_)
            if (w.q == q) throw std::runtime_error("forward: a slot twice in one batch");
        use(q);
        if (!q->used) { q->used = true; reset_here(); }
        if (pos() + b.T > opt_.max_ctx) { use(keep); throw std::runtime_error("context full"); }
        ensure_ctx_here(pos() + b.T + kMaxWindowDense);
        const int r0 = wins_.empty() ? 0 : wins_.back().r0 + wins_.back().T;
        wins_.push_back(Win{q, r0, b.T, pos(), b.commit_all});
        q->history.insert(q->history.end(), b.tokens, b.tokens + b.T);
        q->last_r0 = r0;
        q->last_lane = lane;
        memcpy(h_tok_ + r0, b.tokens, b.T * 4);
        h_par_[wins_.size() - 1] = WinParams{wins_.back().pos0, b.T, seq_, 0};
        key.insert(key.end(), {b.slot, b.T, b.commit_all ? 1 : 0});
    }
    key.push_back(lane);
    use(keep);
    last_T = R;
    fwd_graph_ = opt_.use_graphs && !head_->dump_all;
    fwd_exec_ = nullptr;
    if (tp() && tp_rank_ > 0) return;   // rank 0's graph covers this rank's work
    if (fwd_graph_) {
        FwdGraph & fg = fwd_graphs_[key];
        if (!fg.exec) {
            while ((int) fwd_graphs_.size() > kMaxFwdGraphs) {
                auto lru = fwd_graphs_.end();
                for (auto it = fwd_graphs_.begin(); it != fwd_graphs_.end(); ++it)
                    if (it->second.exec && (lru == fwd_graphs_.end() || it->second.used < lru->second.used)) lru = it;
                if (lru == fwd_graphs_.end()) break;
                cudaGraphExecDestroy(lru->second.exec);
                fwd_graphs_.erase(lru);
            }
            budget_.make_room(kGraphBytesEst);
            size_t f0 = 0, f1 = 0, tot = 0;
            cudaMemGetInfo(&f0, &tot);
            cudaGraph_t g;
            CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
            if (tp()) {   // the peer's stream joins the capture
                CUDA_CHECK(cudaEventRecord(tp_fork_, st_));
                {
                    DeviceScope d2(peer_->dev_);
                    CUDA_CHECK(cudaStreamWaitEvent(peer_->st_, tp_fork_, 0));
                }
                tp_enqueue_forward(R);
                {   // (an event records only on its own device's streams: the peer's join event)
                    DeviceScope d2(peer_->dev_);
                    CUDA_CHECK(cudaEventRecord(peer_->tp_join_, peer_->st_));
                }
                CUDA_CHECK(cudaStreamWaitEvent(st_, peer_->tp_join_, 0));
            } else {
                enqueue_forward(R);
            }
            CUDA_CHECK(cudaStreamEndCapture(st_, &g));
            CUDA_CHECK(cudaGraphInstantiate(&fg.exec, g, 0));
            cudaGraphDestroy(g);
            cudaMemGetInfo(&f1, &tot);
            fg.bytes = f0 > f1 ? f0 - f1 : 0;
            budget_.charge(fg.bytes);
            graph_bytes_ += fg.bytes;
            ++graph_captures_;
        }
        fg.used = ++fwd_graph_age_;
        fwd_exec_ = fg.exec;
    }
}

void DenseEngine::fwd_launch(cudaEvent_t after) {
    DeviceScope ds(dev_);
    if (tp() && tp_rank_ > 0) {   // ordered after rank 0's launch, which holds this rank's work
        CUDA_CHECK(cudaStreamWaitEvent(st_, peer_->ev_out_, 0));
        return;
    }
    if (after) CUDA_CHECK(cudaStreamWaitEvent(st_, after, 0));
    CUDA_CHECK(cudaEventRecord(ev_t0_, st_));
    if (fwd_graph_) {
        CUDA_CHECK(cudaGraphLaunch(fwd_exec_, st_));
    } else if (tp()) {
        CUDA_CHECK(cudaEventRecord(tp_fork_, st_));
        {
            DeviceScope d2(peer_->dev_);
            CUDA_CHECK(cudaStreamWaitEvent(peer_->st_, tp_fork_, 0));
        }
        tp_enqueue_forward(last_T);
        {
            DeviceScope d2(peer_->dev_);
            CUDA_CHECK(cudaEventRecord(peer_->tp_join_, peer_->st_));
        }
        CUDA_CHECK(cudaStreamWaitEvent(st_, peer_->tp_join_, 0));
    } else {
        enqueue_forward(last_T);
    }
    CUDA_CHECK(cudaEventRecord(ev_out_, st_));
    CUDA_CHECK(cudaEventRecord(ev_t1_, st_));
}

void DenseEngine::fwd_end() {
    DeviceScope ds(dev_);
    CUDA_CHECK(cudaStreamSynchronize(st_));
    for (const Win & w : wins_) w.q->pending_T = w.commit ? 0 : w.T;
    if (tp() && tp_rank_ > 0) return;
    static const bool tpdbg = getenv("BNK_TP_DEBUG") != nullptr;
    if (tp() && tpdbg) {   // the two ranks' residual rows must agree bit for bit
        const size_t n = (size_t) last_T * model_.cfg.n_embd;
        std::vector<float> a(n), b(n);
        CUDA_CHECK(cudaMemcpy(a.data(), res_.p, n * 4, cudaMemcpyDeviceToHost));
        {
            DeviceScope d2(peer_->dev_);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(b.data(), peer_->res_.p, n * 4, cudaMemcpyDeviceToHost));
        }
        double sa = 0, sb = 0, sd = 0;
        for (size_t i = 0; i < n; ++i) { sa += a[i] * a[i]; sb += b[i] * b[i]; sd += (a[i] - b[i]) * (a[i] - b[i]); }
        fprintf(stderr, "tp: residual rms rank0 %.5f rank1 %.5f diff %.3e (%s)\n", sqrt(sa / n), sqrt(sb / n), sqrt(sd / n),
                fwd_graph_ ? "graph" : "eager");
    }
    float ms = 0.f;
    if (cudaEventElapsedTime(&ms, ev_t0_, ev_t1_) == cudaSuccess) {
        gpu_ms_ += ms;
        gpu_forwards_++;
    } else {
        cudaGetLastError();
    }
}

// Two batches in flight on a two-GPU pipeline: `first` runs on the first GPU while `second` (which went through
// the first GPU in the previous call) runs on the second. Each batch keeps to its lane from the first GPU to its
// commit: the lanes have their own window buffers.
void DenseEngine::forward_pipe(const std::vector<BatchWin> & first_in, int first_lane, const std::vector<BatchWin> & second_in,
                               int second_lane) {
    if (stages() != 2 || tp()) throw std::runtime_error("forward_pipe: needs a pipeline of two GPUs");
    if (!first_in.empty() && !second_in.empty() && first_lane == second_lane)
        throw std::runtime_error("forward_pipe: the two batches need different lanes");
    auto sorted = [](std::vector<BatchWin> b) {
        std::sort(b.begin(), b.end(), [](const BatchWin & x, const BatchWin & y) { return x.slot < y.slot; });
        return b;
    };
    const std::vector<BatchWin> a = sorted(first_in), b = sorted(second_in);
    for (const auto & w : a)
        for (const auto & v : b)
            if (w.slot == v.slot) throw std::runtime_error("forward_pipe: a slot in both batches");
    DenseEngine * s1 = next_.get();
    const double t0 = now_ms();
    if (!a.empty()) fwd_prepare(a, first_lane);
    if (!b.empty()) s1->fwd_prepare(b, second_lane);
    if (!a.empty()) fwd_launch(nullptr);
    if (!b.empty()) s1->fwd_launch(nullptr);
    if (!a.empty()) fwd_end();
    if (!b.empty()) s1->fwd_end();
    times.total_ms += now_ms() - t0;
    times.calls++;
}

// ------------------------------------------------------------------------------------------------ commit
// Keep the first c rows of the last verify window: replay the delta rule over them and shift the histories.
void DenseEngine::enqueue_commit(int cnt) {
    const Config & c = model_.lc;
    const int C = c.conv_channels(), K = c.ssm_conv, S = c.ssm_state, nv = c.ssm_vheads;
    const size_t r0 = (size_t) cur_->last_r0;
    for (int il = l0_; il < l1_; ++il) {
        if (c.is_attn(il)) continue;
        const int ln = cur_->last_lane;
        gdn_recurrence(gdn_co_[ln][il].p + r0 * C, C, gdn_g_[ln][il].p + r0 * nv, gdn_b_[ln][il].p + r0 * nv, cur_->ssm_state[il],
                       nullptr, cnt, c.ssm_groups, nv, S, cnt, st_);
    }
    if (n_gdn_) shift_rows_multi(cur_->conv_ptrs, n_gdn_, C, cnt, K - 1, st_);
}

void DenseEngine::join_commit() {
    for (DenseEngine * s = this; s; s = s->next_.get()) s->join_commit_here();
}

void DenseEngine::join_commit_here() {
    if (!commit_inflight_) return;
    DeviceScope ds(dev_);
    CUDA_CHECK(cudaStreamWaitEvent(st_, ev_committed_, 0));
    commit_inflight_ = false;
}

void DenseEngine::commit(int cnt) {
    for (DenseEngine * s = this; s; s = s->next_.get()) s->commit_here(cnt);
}

void DenseEngine::commit_here(int cnt) {
    DeviceScope ds(dev_);
    if (!cur_->conv_ptrs.p) {   // the conv buffers never move: their pointer table once, before any capture
        const Config & c = model_.lc;
        std::vector<float *> ptrs;
        for (int il = l0_; il < l1_; ++il)
            if (!c.is_attn(il)) ptrs.push_back(cur_->conv_buf[il].p);
        n_gdn_ = (int) ptrs.size();
        cur_->conv_ptrs.alloc(std::max<size_t>(1, ptrs.size()));
        if (!ptrs.empty())
            CUDA_CHECK(cudaMemcpy(cur_->conv_ptrs.p, ptrs.data(), ptrs.size() * sizeof(float *), cudaMemcpyHostToDevice));
    }
    if (cur_->pending_T == 0) throw std::runtime_error("commit: no verify window pending");
    if (cnt < 1 || cnt > cur_->pending_T) throw std::runtime_error("commit: count out of range");
    cur_->history.resize(cur_->history.size() - (cur_->pending_T - cnt));
    cur_->pending_T = 0;
    if (opt_.use_graphs) {
        cudaGraphExec_t & cg = cur_->commit_graphs[cur_->last_lane][cur_->last_r0][cnt];
        if (!cg) {
            cudaGraph_t g;
            CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
            enqueue_commit(cnt);
            CUDA_CHECK(cudaStreamEndCapture(st_, &g));
            CUDA_CHECK(cudaGraphInstantiate(&cg, g, 0));
            cudaGraphDestroy(g);
        }
        // on the commit stream, after the verify window: the drafter (main stream) runs meanwhile
        CUDA_CHECK(cudaEventRecord(ev_verified_, st_));
        CUDA_CHECK(cudaStreamWaitEvent(st_commit_, ev_verified_, 0));
        CUDA_CHECK(cudaGraphLaunch(cg, st_commit_));
        CUDA_CHECK(cudaEventRecord(ev_committed_, st_commit_));
        commit_inflight_ = true;
    } else {
        enqueue_commit(cnt);
    }
}

// ------------------------------------------------------------------------------------------------ drafter
void DenseEngine::tap(int il, int T) {
    if (!dflash_.loaded()) return;
    const int i = dflash_.tap_index(il);
    if (i < 0) return;
    const int64_t n = (int64_t) T * model_.cfg.n_embd;
    copy_f32(tap_tmp_, res_, n, st_);
    if (pend_) add_rows(tap_tmp_, pend_, n, st_);
    gemv_auto(dflash_.fc(i), tap_tmp_, model_.cfg.n_embd, T, tap_acc_, model_.cfg.n_embd, i > 0, act_, st_);
}

void DenseEngine::pf_tap(int il, int N) {
    if (!dflash_.loaded()) return;
    const int i = dflash_.tap_index(il);
    if (i < 0) return;
    gemm_.run(dflash_.fc(i), pf_.res, model_.cfg.n_embd, N, pf_tap_, model_.cfg.n_embd, i > 0);
}

void DenseEngine::mtp_feed(const float * R_rows, const int32_t * tokens, int n, int pos0) {
    if (dflash_.loaded()) {   // the same rows' tap accumulator (prompt chunk or decode window)
        dflash_.feed(R_rows == pf_.res.p ? pf_tap_.p : tap_acc_.p, n, pos0);
        return;
    }
    if (!mtp_.loaded()) return;
    DeviceScope ds(dev_);
    const int E = model_.cfg.n_embd;
    if (!cur_->mtp_R) CUDA_CHECK(cudaMalloc(&cur_->mtp_R, (size_t) E * 4));
    if (cur_->mtp_cell >= 0 && cur_->mtp_cell == pos0 - 1) mtp_.run(cur_->mtp_R, tokens, 1, cur_->mtp_cell, false, nullptr);
    for (int r0 = 0; r0 < n - 1; r0 += kMaxWindowDense) {
        const int rr = std::min(kMaxWindowDense, n - 1 - r0);
        mtp_.run(R_rows + (size_t) r0 * E, tokens + r0 + 1, rr, pos0 + r0, false, nullptr);
    }
    CUDA_CHECK(cudaMemcpyAsync(cur_->mtp_R, R_rows + (size_t) (n - 1) * E, (size_t) E * 4, cudaMemcpyDeviceToDevice, st_));
    cur_->mtp_cell = pos0 + n - 1;
}

void DenseEngine::set_mtp_pending(const float * R_row_dev, int cell) {
    if (next_ && !tp()) return next_->set_mtp_pending(R_row_dev, cell);
    if (!mtp_.loaded()) return;
    DeviceScope ds(dev_);
    if (!cur_->mtp_R) CUDA_CHECK(cudaMalloc(&cur_->mtp_R, (size_t) model_.cfg.n_embd * 4));
    CUDA_CHECK(cudaMemcpyAsync(cur_->mtp_R, R_row_dev, (size_t) model_.cfg.n_embd * 4, cudaMemcpyDeviceToDevice, st_));
    cur_->mtp_cell = cell;
}

// ------------------------------------------------------------------------------------------------ logits
int DenseEngine::argmax(int t) {
    if (next_ && !tp()) return next_->argmax(t);
    DeviceScope ds(dev_);
    argmax_rows(logits_.p + (size_t) t * model_.cfg.n_vocab, 1, model_.cfg.n_vocab, argmax_dev_, st_);
    int32_t r;
    CUDA_CHECK(cudaMemcpyAsync(&r, argmax_dev_.p, 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    return r;
}

void DenseEngine::argmax_all(int T, int32_t * out, int row0) {
    if (next_ && !tp()) return next_->argmax_all(T, out, row0);
    DeviceScope ds(dev_);
    argmax_rows(logits_.p + (size_t) row0 * model_.cfg.n_vocab, T, model_.cfg.n_vocab, argmax_dev_, st_);
    CUDA_CHECK(cudaMemcpyAsync(out, argmax_dev_.p, T * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void DenseEngine::topk_rows_host(int row0, int T, int K, int32_t * ids, float * vals, double * z, float temp) {
    if (next_ && !tp()) return next_->topk_rows_host(row0, T, K, ids, vals, z, temp);
    DeviceScope ds(dev_);
    if (K > kTopkMax || T > kMaxWindowDense) throw std::runtime_error("topk_rows_host: too many");
    if (!topk_ids_.p) {
        topk_ids_.alloc((size_t) kMaxWindowDense * kTopkMax);
        topk_vals_.alloc((size_t) kMaxWindowDense * kTopkMax);
    }
    const int V = model_.cfg.n_vocab;
    topk_rows(logits_.p + (size_t) row0 * V, T, V, K, topk_ids_, topk_vals_, st_);
    CUDA_CHECK(cudaMemcpyAsync(ids, topk_ids_.p, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(vals, topk_vals_.p, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st_));
    if (z) {
        if (!topk_z_.p) topk_z_.alloc(kMaxWindowDense);
        row_expsum(logits_.p + (size_t) row0 * V, T, V, temp, topk_z_.p, st_);
        CUDA_CHECK(cudaMemcpyAsync(z, topk_z_.p, (size_t) T * sizeof(double), cudaMemcpyDeviceToHost, st_));
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void DenseEngine::logits_rows_host(int T, float * out, int row0) {
    if (next_ && !tp()) return next_->logits_rows_host(T, out, row0);
    DeviceScope ds(dev_);
    CUDA_CHECK(cudaMemcpy(out, logits_.p + (size_t) row0 * model_.cfg.n_vocab, (size_t) T * model_.cfg.n_vocab * 4,
                          cudaMemcpyDeviceToHost));
}

std::vector<float> DenseEngine::logits_host(int t) {
    if (next_ && !tp()) return next_->logits_host(t);
    DeviceScope ds(dev_);
    std::vector<float> v(model_.cfg.n_vocab);
    CUDA_CHECK(cudaMemcpy(v.data(), logits_.p + (size_t) t * model_.cfg.n_vocab, v.size() * 4, cudaMemcpyDeviceToHost));
    return v;
}

}  // namespace bnk
