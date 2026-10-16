#include "engine/engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "ggml-cpu.h"
#include "ggml.h"
#include "kernels/ops.h"

namespace bnk {

static const float * F(const QMat & m) { return (const float *) m.data; }

// Debug dumps of one layer's intermediates (BNK_DUMP_DIR, BNK_DUMP_LAYER; eager mode only): appended per window.
static void dbg(cudaStream_t st, int il, const char * name, const float * dev, size_t n) {
    static const char * dir = getenv("BNK_DUMP_DIR");
    static int layer = env_int("BNK_DUMP_LAYER", 0);
    if (!dir || il != layer) return;
    std::vector<float> h(n);
    CUDA_CHECK(cudaStreamSynchronize(st));
    CUDA_CHECK(cudaMemcpy(h.data(), dev, n * 4, cudaMemcpyDeviceToHost));
    FILE * f = fopen((std::string(dir) + "/" + name).c_str(), "ab");
    fwrite(h.data(), 4, n, f);
    fclose(f);
}

// y = W x, with x already quantized in `xq` when W is a quant format.
static void gemv_q(const QMat & W, const ActQ8 & xq, const float * x, int64_t ldx, int T, float * y, int64_t ldy,
                   cudaStream_t s, GemvSplit sp = {}) {
    if (is_float_format(W.type)) gemv(W, nullptr, x, ldx, T, y, ldy, false, s, sp);
    else gemv(W, &xq, nullptr, 0, T, y, ldy, false, s, sp);
}

Engine::Worker::Worker() {
    th = std::thread([this]() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv.wait(lk, [&]() { return quit || job; });
            if (!job) return;
            auto f = std::move(job);
            job = nullptr;
            lk.unlock();
            try {
                f();
            } catch (...) {
                lk.lock();
                err = std::current_exception();
                lk.unlock();
            }
            lk.lock();
            busy = false;
            cv.notify_all();
        }
    });
}

Engine::Worker::~Worker() {
    {
        std::lock_guard<std::mutex> lk(mu);
        quit = true;
    }
    cv.notify_all();
    th.join();
}

void Engine::Worker::post(std::function<void()> f) {
    wait();
    std::lock_guard<std::mutex> lk(mu);
    job = std::move(f);
    busy = true;
    cv.notify_all();
}

void Engine::Worker::wait() {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&]() { return !busy; });
    if (err) {
        auto e = err;
        err = nullptr;
        std::rethrow_exception(e);
    }
}

Engine::~Engine() {
    worker_.reset();
    next_.reset();   // the later stages first, each on its own device
    cudaSetDevice(dev_);
    for (auto & q : seqs_) {
        for (auto & k : q->ckpts)
            ck_free(k.host);
        for (auto & gl : q->commit_graphs) for (auto & gm : gl) for (auto & g : gm) if (g) cudaGraphExecDestroy(g);
    }
    for (auto & kv : fwd_graphs_) cudaGraphExecDestroy(kv.second.exec);
    if (counts_.p) save_counts_here();
    if (mail_) cudaFreeHost(mail_);
    if (h_par_) cudaFreeHost(h_par_);
    if (h_tok_) cudaFreeHost(h_tok_);
    if (h_ple_rows_) cudaFreeHost(h_ple_rows_);
    if (d_par_) cudaFree(d_par_);
    for (float * b : xfer_in_)
        if (b) cudaFreeHost(b);
    for (float * b : pf_xfer_in_)
        if (b) cudaFreeHost(b);
    if (ev_out_) cudaEventDestroy(ev_out_);
    if (ev_t0_) cudaEventDestroy(ev_t0_);
    if (ev_t1_) cudaEventDestroy(ev_t1_);
}

// mixact_ = quantize(mixed_), unless the hc_mix that just wrote mixed_ quantized it already
void Engine::quantize_mixed(int T) {
    const int E = model_.cfg.n_embd;
    if (!mix_quantized_) quantize_act(mixed_, E, T, E, mixact_, st_);
    mix_quantized_ = false;
}

cudaStream_t Engine::fork(int i) {
    if (!side_) return st_;
    CUDA_CHECK(cudaEventRecord(ev_fork_[i], st_));
    CUDA_CHECK(cudaStreamWaitEvent(st_side_[i], ev_fork_[i], 0));
    return st_side_[i];
}

void Engine::join(int i) {
    if (!side_) return;
    CUDA_CHECK(cudaEventRecord(ev_join_[i], st_side_[i]));
    CUDA_CHECK(cudaStreamWaitEvent(st_, ev_join_[i], 0));
}

std::vector<int> Engine::devices() const {
    std::vector<int> d;
    for (const Engine * s = this; s; s = s->next_.get())
        if (std::find(d.begin(), d.end(), s->dev_) == d.end()) d.push_back(s->dev_);
    return d;
}

int Engine::cache_slots(int il) const {
    for (const Engine * s = this; s; s = s->next_.get())
        if (s->owns(il)) return s->cache_.slots(il);
    return 0;
}

int Engine::cache_resident() const { return cache_.resident() + (next_ ? next_->cache_resident() : 0); }
size_t Engine::cache_bytes() const { return cache_.bytes() + (next_ ? next_->cache_bytes() : 0); }
int64_t Engine::cache_swaps() const { return cache_.swaps_done + (next_ ? next_->cache_swaps() : 0); }

// Contiguous layer ranges, one per GPU. Each layer would hold its dense weights, all of its experts and the KV of
// a full context; the first stage adds the token embeddings, the last the head and the drafter. The split minimizes
// the largest share of a GPU's free memory, so with room for everything each GPU keeps its layers' experts
// resident, and otherwise both fall short alike (the CPU computes the rest as on one GPU).
std::vector<int> Engine::plan_split(const std::string & path, const EngineOptions & opt, const std::vector<int> & gpus) {
    kv_set_q8(opt.kv8);
    const int S = (int) gpus.size();
    GgufModel g;
    g.open(path);
    const Config c = load_config(g);
    const int L = c.n_layer;
    std::vector<int> b{0};
    if (!opt.split.empty()) {
        if ((int) opt.split.size() != S - 1)
            throw std::runtime_error("--split: give " + std::to_string(S - 1) + " layer(s) for " + std::to_string(S) + " GPUs");
        for (int x : opt.split) {
            if (x <= b.back() || x >= L)
                throw std::runtime_error("--split: the first layers of the stages must increase within 1.." + std::to_string(L - 1));
            b.push_back(x);
        }
        b.push_back(L);
        return b;
    }
    if (S > L) throw std::runtime_error("more GPUs than layers");
    std::vector<double> lay(L, 0.0);
    double embd = 0, head = 0;
    for (const TensorRef & t : g.tensors()) {
        if (t.name.rfind("blk.", 0) == 0) {
            const int il = atoi(t.name.c_str() + 4);
            if (il >= 0 && il < L) lay[il] += (double) t.nbytes;
        } else if (t.name == "token_embd.weight") {
            embd = (double) t.nbytes;
        } else if (t.name.rfind("output", 0) == 0) {
            head += (double) t.nbytes;
        }
    }
    for (int il = 0; il < L; ++il)
        if (c.is_attn(il)) {
            double cell = 4.0 * c.n_head_kv * kv_row_halves(c.head_dim);
            if (c.compress_ratio[il] > 0) cell += 2.0 * c.idx_dim + 4.0 * c.idx_dim / c.compress_ratio[il];
            lay[il] += cell * opt.max_ctx;
        }
    // per GPU: fixed costs (CUDA context, workspaces, prompt buffers) and the free memory; one GPU listed twice
    // (a test of the split on one card) shares it
    const double fixed = 2.0 * 1073741824.0;
    double mtp_bytes = 0;
    if (!opt.mtp.empty())
        if (FILE * f = fopen(opt.mtp.c_str(), "rb")) {
            fseek(f, 0, SEEK_END);
            mtp_bytes = (double) ftell(f);
            fclose(f);
        }
    std::vector<double> cap(S);
    for (int k = 0; k < S; ++k) {
        DeviceScope ds(gpus[k]);
        size_t fr = 0, tot = 0;
        CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
        const int same = (int) std::count(gpus.begin(), gpus.end(), gpus[k]);
        cap[k] = std::max(1.0, (double) fr / same - fixed);
    }
    auto extra = [&](int k) {
        double e = 0;
        if (k == 0) e += embd;
        if (k == S - 1) e += head + (mtp_bytes > 0 ? embd + mtp_bytes : 0);
        return e;
    };
    // best[k][i]: the smallest worst load ratio of stages k.. over layers i..L-1 (and where stage k ends)
    std::vector<std::vector<double>> best(S + 1, std::vector<double>(L + 1, 1e300));
    std::vector<std::vector<int>> cut(S + 1, std::vector<int>(L + 1, -1));
    best[S][L] = 0;
    for (int k = S - 1; k >= 0; --k)
        for (int i = 0; i < L; ++i) {
            double sum = extra(k);
            for (int j = i + 1; j <= L - (S - 1 - k); ++j) {   // stage k takes layers [i, j)
                sum += lay[j - 1];
                if (k == S - 1 && j != L) continue;
                const double v = std::max(sum / cap[k], best[k + 1][j]);
                if (v < best[k][i]) {
                    best[k][i] = v;
                    cut[k][i] = j;
                }
            }
        }
    for (int k = 0, i = 0; k < S; ++k) {
        i = cut[k][i];
        if (i < 0) throw std::runtime_error("cannot split the layers over the GPUs");
        b.push_back(i);
    }
    if (opt.verbose) {
        fprintf(stderr, "bnk: %d GPUs, layers split", S);
        for (int k = 0; k < S; ++k) {
            double sum = extra(k);
            for (int il = b[k]; il < b[k + 1]; ++il) sum += lay[il];
            fprintf(stderr, "%s %d-%d on GPU %d (all resident: %.1f of %.1f GiB)", k ? "," : "", b[k], b[k + 1] - 1, gpus[k],
                    sum / 1073741824.0, cap[k] / 1073741824.0);
        }
        fprintf(stderr, "\n");
    }
    return b;
}

void Engine::load(const std::string & path, const EngineOptions & opt) {
    ggml_cpu_init();
    std::vector<int> gpus = opt.gpus;
    if (gpus.empty()) {
        int d = 0;
        CUDA_CHECK(cudaGetDevice(&d));
        gpus = {d};
    }
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (gpus.size() == 1 && gpus[0] < 0) {   // "all": every GPU bnk can run on (sm_70 and newer)
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
    const std::vector<int> bounds = gpus.size() > 1 ? plan_split(path, opt, gpus) : std::vector<int>{0, -1};
    Engine * s = this, * prev = nullptr;
    for (size_t k = 0; k < gpus.size(); ++k) {
        s->head_ = this;
        s->dev_ = gpus[k];
        s->l0_ = bounds[k];
        s->l1_ = bounds[k + 1];
        s->last_ = k + 1 == gpus.size();
        s->store_ = &own_store_;
        s->cpu_ = &own_cpu_;
        if (opt.verbose && gpus.size() > 1)
            fprintf(stderr, "bnk: stage %zu of %zu: GPU %d, layers %d-%d\n", k + 1, gpus.size(), s->dev_, s->l0_, s->l1_ - 1);
        EngineOptions so = opt;
        if (!opt.stage_cache_gib.empty()) {
            if (opt.stage_cache_gib.size() != gpus.size())
                throw std::runtime_error("expert cache sizes: give one, or one per GPU (" + std::to_string(gpus.size()) + ")");
            so.expert_cache_gib = opt.stage_cache_gib[k];
        }
        s->load_stage(path, so);
        if (prev) {
            for (int l = 0; l < kLanes; ++l) prev->xfer_out_[l] = s->xfer_in_[l];
            prev->pf_xfer_out_[0] = s->pf_xfer_in_[0];
            prev->pf_xfer_out_[1] = s->pf_xfer_in_[1];
        }
        if (k + 1 < gpus.size()) {
            prev = s;
            s->next_ = std::make_unique<Engine>();
            s = s->next_.get();
        }
    }
    CUDA_CHECK(cudaSetDevice(dev_));
}

void Engine::load_stage(const std::string & path, const EngineOptions & opt) {
    opt_ = opt;
    kv_set_q8(opt.kv8);
    const bool head = head_ == this;
    CUDA_CHECK(cudaSetDevice(dev_));
    // a second stage on the same GPU finds its context already made (the flag is the default on 64-bit anyway)
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
    // the drafter embeds its tokens and shares the head: the last stage holds the embeddings too when it drafts
    model_.load(path, opt.verbose, opt.ple_gguf, l0_, l1_, first() || (last_ && !opt.mtp.empty()), last_);
    if (l1_ < 0) l1_ = model_.cfg.n_layer;
    if (head) {
        layer_routed.assign(model_.cfg.n_layer, 0);
        layer_misses.assign(model_.cfg.n_layer, 0);
        store_->build(model_, std::min(opt.cpu_threads, 16), opt.verbose);
        cpu_->init(model_, *store_, opt.cpu_threads);
    }

    const Config & c = model_.cfg;
    const int W = kMaxWindow, E = c.n_embd, HC = c.hc_dim();
    x_.alloc(W * E); res_.alloc(W * HC); xn_.alloc(W * HC); lo_.alloc(W * c.hc_rank);
    gpre_.alloc(W * HC); mixed_.alloc(W * E); inj_.alloc(W * c.hc); out_.alloc(W * E);
    const int C = c.conv_channels(), VI = c.ssm_vheads * c.ssm_state;
    conv_out_.alloc(W * C); z_.alloc(W * VI); alpha_.alloc(W * c.ssm_vheads); beta_.alloc(W * c.ssm_vheads);
    gdn_o_.alloc(W * VI); gdn_n_.alloc(W * VI);
    qfull_.alloc(W * c.n_head * c.head_dim * 2); k_.alloc(W * c.n_head_kv * c.head_dim);
    v_.alloc(W * c.n_head_kv * c.head_dim); q_.alloc(W * c.n_head * c.head_dim);
    attn_o_.alloc(W * c.n_head * c.head_dim);
    attn_scratch_.alloc(attention_scratch_floats(W, c.n_head, c.head_dim, opt.max_ctx));
    rlogits_.alloc(W * c.n_expert); rw_.alloc(W * c.n_expert_used); rids_.alloc(W * c.n_expert_used);
    sg_.alloc(W * c.n_ff_shexp); su_.alloc(W * c.n_ff_shexp); sh_.alloc(W * c.n_ff_shexp); sgate_.alloc(W);
    shared_out_.alloc(W * E); moe_out_.alloc(W * E);
    if (last_) logits_.alloc((size_t) W * c.n_vocab);
    argmax_dev_.alloc(W);
    const int64_t maxcols = std::max<int64_t>(HC, 16384);
    actq_.alloc(W * maxcols); actd_.alloc(W * maxcols / 32);
    acts_.alloc(W * maxcols / 16);
    acth_.alloc(W * maxcols);
    act_.q = actq_; act_.d = actd_; act_.s = acts_; act_.h = acth_;
    mixq_.alloc(W * E + 64); mixd_.alloc(W * E / 32 + 8); mixs_.alloc(W * E / 16 + 16); mixh_.alloc(W * E + 64);
    mixact_.q = mixq_; mixact_.d = mixd_; mixact_.s = mixs_; mixact_.h = mixh_;

    // MoE scratch
    const int F = c.n_ff_exp;
    hits_buf_.alloc(sizeof(HitList));
    gu_buf_.alloc((size_t) kMaxRouted * W * 2 * F);
    hq_buf_.alloc((size_t) kMaxRouted * W * F);
    hd_buf_.alloc((size_t) kMaxRouted * W * F / 32);
    part_buf_.alloc((size_t) kMaxRouted * W * E);
    counts_.alloc((size_t) c.n_layer * c.n_expert);
    moes_.hits = (HitList *) hits_buf_.p;
    moes_.gu = gu_buf_; moes_.hq = hq_buf_; moes_.hd = hd_buf_; moes_.part = part_buf_; moes_.counts = counts_;
    mail_stride_ = MoeMsg::bytes(E);
    CUDA_CHECK(cudaHostAlloc((void **) &mail_, mail_stride_ * c.n_layer, cudaHostAllocMapped));
    memset(mail_, 0, mail_stride_ * c.n_layer);

    CUDA_CHECK(cudaHostAlloc((void **) &h_par_, kMaxSlots * sizeof(WinParams), cudaHostAllocMapped));
    CUDA_CHECK(cudaMalloc(&d_par_, kMaxSlots * sizeof(WinParams)));
    CUDA_CHECK(cudaHostAlloc((void **) &h_tok_, W * 4, cudaHostAllocMapped));
    if (!first()) {   // the previous stage's residual rows come through here
        for (float *& b : xfer_in_) CUDA_CHECK(cudaHostAlloc((void **) &b, (size_t) W * HC * 4, cudaHostAllocPortable));
        if (opt.prefill_chunk > 0)
            for (float *& b : pf_xfer_in_)
                CUDA_CHECK(cudaHostAlloc((void **) &b, (size_t) std::max(opt.prefill_chunk, opt.prefill_chunk_max) * HC * 4,
                                         cudaHostAllocPortable));
    }

    if (c.ple_layer >= 0 && owns(c.ple_layer)) {
        const int ph = c.ple_heads();
        ple_emb_.alloc(W * ph * c.ple_dim); ple_key_.alloc(W * HC); ple_val_.alloc(W * E);
        ple_gated_.alloc(W * HC);
        CUDA_CHECK(cudaHostAlloc((void **) &h_ple_rows_, (size_t) W * ph * model_.ple_table->row_bytes(),
                                 cudaHostAllocMapped));
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
            if (model_.layers[il].idx_q.valid() && c.compress_ratio[il] > 0 && !getenv("BNK_NO_QSA")) {
                kv_cell_bytes += c.idx_dim * sizeof(half) + c.idx_dim * sizeof(float) / c.compress_ratio[il];
                qsa_on_ = true;
            }
        } else {
            // the second lane only for a pipeline (two batches in flight)
            for (int l = 0; l < (first() && last_ ? 1 : kLanes); ++l) {
                gdn_co_[l][il].alloc((size_t) W * C);
                gdn_g_[l][il].alloc((size_t) W * c.ssm_vheads);
                gdn_b_[l][il].alloc((size_t) W * c.ssm_vheads);
            }
            state_bytes += ((size_t) (c.ssm_conv - 1 + W) * C + (size_t) c.ssm_vheads * c.ssm_state * c.ssm_state) * 4;
        }
    }
    // the conversation slots: address space for every one, VRAM for the KV only as a conversation uses it
    seqs_.clear();
    for (int i = 0; i < std::max(1, opt.slots); ++i) {
        seqs_.push_back(std::make_unique<Seq>());
        alloc_seq(*seqs_.back());
    }
    cur_ = seqs_[0].get();
    cur_->used = true;
    if (qsa_on_) {
        int r = 0;
        for (int il = 0; il < c.n_layer; ++il) r = std::max(r, c.compress_ratio[il]);
        qsh_ = QsaShape{c.n_head, c.n_head_kv, c.head_dim, c.idx_heads, c.idx_dim, r, c.idx_top_k / r, c.n_rot,
                        c.rope_base, c.rms_eps};
        max_blocks_ = opt.max_ctx / r + 1;
        ik_.alloc(W * c.idx_dim); iq_.alloc(W * c.idx_heads * c.idx_dim);
        qsa_scores_.alloc((size_t) W * max_blocks_); qsa_sel_.alloc((size_t) W * (c.idx_top_k / r)); qsa_nsel_.alloc(W);
    }
    if (opt.verbose)
        fprintf(stderr, "bnk: %scontext %d: %.0f KiB of %sKV per token (%.2f GiB when full, mapped as used), "
                        "recurrent state %.2f GiB, %d CPU expert threads%s\n",
                first() && last_ ? "" : ("layers " + std::to_string(l0_) + "-" + std::to_string(l1_ - 1) + ": ").c_str(),
                opt.max_ctx, kv_cell_bytes / 1024.0, opt.kv8 ? "8-bit " : "", (double) kv_cell_bytes * opt.max_ctx / 1073741824.0,
                state_bytes / 1073741824.0, cpu_->threads(),
                seqs_.size() > 1 ? (", " + std::to_string(seqs_.size()) + " conversation slots").c_str() : "");

    if (const char * rl = getenv("BNK_ROUTE_LOG")) {
        if (!first() || !last_) throw std::runtime_error("BNK_ROUTE_LOG needs a single GPU");
        route_log_ = fopen(rl, "wb");
        if (!route_log_) throw std::runtime_error(std::string("cannot open ") + rl);
        route_dev_.alloc((size_t) c.n_layer * kMaxWindow * c.n_expert_used);
        cpu_us_.assign(c.n_layer, 0.f);
    }
    if (last_ && !opt.mtp.empty())
        mtp_.load(opt.mtp, model_, opt.max_ctx, st_, opt.verbose, opt.draft_vocab, &budget_, (int) seqs_.size());
    pf_tc_ = !getenv("BNK_PF_TC") || atoi(getenv("BNK_PF_TC")) != 0;
    // NVFP4 experts carry their global scales outside the blocks; only the tensor-core prompt path applies them
    // (the cuBLAS path's fp16 copies of the unscaled weights would overflow)
    for (const auto & L : model_.layers)
        if (L.exp_scales.gate && !pf_tc_) { fprintf(stderr, "bnk: NVFP4 experts: BNK_PF_TC=0 ignored\n"); pf_tc_ = true; break; }
    pf_base_ = opt.prefill_chunk;
    pf_small_ = std::min(opt.prefill_chunk, std::max(64, opt.prefill_small));
    pf_big_ = std::max(opt.prefill_chunk, opt.prefill_chunk_max);
    if (opt.prefill_chunk > 0) {
        prefill_alloc(pf_big_);
        pf_arena_.reserve(prefill_arena_bytes(pf_big_), &budget_, "prompt buffers");
    }

    // Everything elastic draws on one budget: what is free now, less a reserve for cuBLAS and graph workspaces.
    // The context and the prompt buffers take their baseline; the expert tier takes the rest, less the prompt
    // path's expert staging (which depends on how many experts each layer keeps: iterate to a fixed point).
    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    const size_t reserve = (size_t) (opt.vram_reserve_gib * 1073741824.0);
    budget_.set_limit(free_b > reserve ? free_b - reserve : 0);
    budget_.set_reserve(reserve);
    budget_.reclaim = [this](size_t need) { return cache_ready_ ? cache_.shrink(need, st_) : (size_t) 0; };
    ensure_ctx_here(std::min(opt.max_ctx, kCtxBaseline));
    if (opt.prefill_chunk > 0) prefill_layout(pf_small_);
    Ranking rank = load_ranking(opt.profile, c.n_layer, c.n_expert);
    if (rank.empty() && !opt.counts_out.empty()) rank = load_ranking(opt.counts_out, c.n_layer, c.n_expert);
    if (opt.verbose && head) fprintf(stderr, "bnk: expert ranking: %s\n", rank.empty() ? "none (uniform)" : "loaded");
    const size_t avail = budget_.free();
    size_t budget = avail;
    if (opt.expert_cache_gib >= 0) budget = std::min(budget, (size_t) (opt.expert_cache_gib * 1073741824.0));
    if (opt.prefill_chunk > 0) {
        // the expert stage kept between prompts (two of them) depends on how many experts each layer leaves
        // outside the cache: iterate to a fixed point
        const size_t small = (size_t) opt.stage_small_mib << 20;
        for (int it = 0; it < 6; ++it) {
            const auto slots = ExpertCache::plan(model_, *store_, budget, rank, l0_, l1_);
            size_t need = 0;
            for (int il = l0_; il < l1_; ++il)
                need = std::max(need, (size_t) (c.n_expert - slots[il]) * store_->blob_bytes(il));
            need = vmem_round(std::min(need, small));
            const size_t want = avail > 2 * need ? avail - 2 * need : 0;
            const size_t nb = opt.expert_cache_gib >= 0 ? std::min(want, (size_t) (opt.expert_cache_gib * 1073741824.0)) : want;
            if (nb == budget) break;
            budget = nb;
        }
    }
    cache_.init(model_, *store_, budget, rank, st_, opt.verbose, &budget_, l0_, l1_);
    cache_ready_ = true;
    if (opt.prefill_chunk > 0) {
        prefill_staging_alloc();
        if (opt.verbose)
            fprintf(stderr, "bnk: prompt path: chunks of %d / %d / %d tokens (short turns / prompts / long prompts), "
                            "%.2f / %.2f / %.2f GiB of buffers, 2 x %.2f GiB of expert staging between prompts (more borrowed "
                            "for long ones)\n",
                    pf_small_, pf_base_, pf_big_, prefill_arena_bytes(pf_small_) / 1073741824.0,
                    prefill_arena_bytes(pf_base_) / 1073741824.0, prefill_arena_bytes(pf_big_) / 1073741824.0,
                    pf_.stage_bytes / 1073741824.0);
    }
    reset_here();
    // every slot's snapshot buffers pinned now: pinning one during a request cost its turn ~60 ms per GPU. The
    // drafter's input row is part of a snapshot, so it exists from the start (else the size would change later)
    if (mtp_.loaded())
        for (auto & q : seqs_)
            if (!q->mtp_R) CUDA_CHECK(cudaMalloc(&q->mtp_R, (size_t) c.hc_dim() * 4));
    const size_t ck = state_floats() * 4;
    for (auto & q : seqs_)
        while ((int) q->ckpts.size() < kMaxCheckpoints) {
            q->ckpts.emplace_back();
            q->ckpts.back().host = ck_alloc();
        }
    if (opt.verbose)
        fprintf(stderr, "bnk: layers %d-%d: snapshots %d x %d slots x %.0f MiB pinned\n", l0_, l1_ - 1, kMaxCheckpoints,
                (int) seqs_.size(), ck / 1048576.0);
}

void Engine::alloc_seq(Seq & q) {
    const Config & c = model_.cfg;
    const int W = kMaxWindow, C = c.conv_channels(), HC = c.hc_dim();
    q.conv_buf.resize(c.n_layer);
    q.ssm_state.resize(c.n_layer);
    q.kraw.resize(c.n_layer);
    q.pooled.resize(c.n_layer);
    q.kc.resize(c.n_layer);
    q.vc.resize(c.n_layer);
    q.kv_k = std::vector<ElasticBuf>(c.n_layer);
    q.kv_v = std::vector<ElasticBuf>(c.n_layer);
    q.kv_raw = std::vector<ElasticBuf>(c.n_layer);
    q.kv_pool = std::vector<ElasticBuf>(c.n_layer);
    for (int il = l0_; il < l1_; ++il) {
        if (c.is_attn(il)) {
            // address space for the whole context; VRAM is mapped as the conversation grows (ensure_ctx)
            const size_t kv = (size_t) opt_.max_ctx * c.n_head_kv * kv_row_halves(c.head_dim);
            q.kv_k[il].reserve(kv * sizeof(half), &budget_, "KV cache");
            q.kv_v[il].reserve(kv * sizeof(half), &budget_, "KV cache");
            q.kc[il].view(q.kv_k[il].as<half>(), kv);
            q.vc[il].view(q.kv_v[il].as<half>(), kv);
            if (model_.layers[il].idx_q.valid() && c.compress_ratio[il] > 0 && !getenv("BNK_NO_QSA")) {
                const size_t nraw = (size_t) opt_.max_ctx * c.idx_dim;
                const size_t npool = (size_t) (opt_.max_ctx / c.compress_ratio[il] + 1) * c.idx_dim;
                q.kv_raw[il].reserve(nraw * sizeof(half), &budget_, "indexer keys");
                q.kv_pool[il].reserve(npool * sizeof(float), &budget_, "indexer keys");
                q.kraw[il].view(q.kv_raw[il].as<half>(), nraw);
                q.pooled[il].view(q.kv_pool[il].as<float>(), npool);
            }
        } else {
            q.conv_buf[il].alloc((size_t) (c.ssm_conv - 1 + W) * C);
            q.ssm_state[il].alloc((size_t) c.ssm_vheads * c.ssm_state * c.ssm_state);
        }
    }
    if (c.ple_layer >= 0 && owns(c.ple_layer)) q.ple_hist.alloc((size_t) ((c.ple_conv - 1) * c.ple_ngram + W) * HC);
}

void Engine::use(Seq * q) {
    cur_ = q;
    for (size_t i = 0; i < seqs_.size(); ++i)
        if (seqs_[i].get() == q && mtp_.loaded()) mtp_.select((int) i);
}

int Engine::current_slot() const {
    for (size_t i = 0; i < seqs_.size(); ++i)
        if (seqs_[i].get() == cur_) return (int) i;
    return -1;
}

void Engine::select(int slot) {
    for (Engine * s = this; s; s = s->next_.get()) s->select_here(slot);
}

void Engine::select_here(int slot) {
    DeviceScope ds(dev_);
    if (slot < 0 || slot >= (int) seqs_.size()) throw std::runtime_error("select: no slot " + std::to_string(slot));
    // no join: switching touches nothing on the GPU, and whatever does (forward, prefill, snapshots ...) joins
    use(seqs_[slot].get());
    if (!cur_->used) {   // first use: an empty conversation with the baseline context
        cur_->used = true;
        reset_here();
    }
}

// KV cells [0, cells) backed by VRAM, in steps of kCtxStep cells; the expert cache gives the bytes up.
void Engine::ensure_ctx(int cells) {
    for (Engine * s = this; s; s = s->next_.get()) s->ensure_ctx_here(cells);
}

void Engine::ensure_ctx_here(int cells) {
    DeviceScope ds(dev_);
    cells = std::min(opt_.max_ctx, (cells + kCtxStep - 1) / kCtxStep * kCtxStep);
    if (cells <= cur_->ctx_mapped) return;
    map_ctx(cells);
}

void Engine::map_ctx(int cells) {
    const Config & c = model_.cfg;
    const bool grow = cells > cur_->ctx_mapped;
    if (grow) {
        // one reclaim for everything below (each buffer asking on its own would shrink the cache ~50 times)
        size_t need = 0;
        auto add = [&](const ElasticBuf & b, size_t bytes) {
            if (b.reserved()) need += vmem_round(std::min(bytes, b.reserved())) - std::min(vmem_round(std::min(bytes, b.reserved())), b.mapped());
        };
        for (int il = 0; il < c.n_layer; ++il) {
            const size_t kv = (size_t) cells * c.n_head_kv * kv_row_halves(c.head_dim) * sizeof(half);
            add(cur_->kv_k[il], kv);
            add(cur_->kv_v[il], kv);
            if (cur_->kv_raw[il].reserved()) {
                add(cur_->kv_raw[il], (size_t) cells * c.idx_dim * sizeof(half));
                add(cur_->kv_pool[il], (size_t) (cells / c.compress_ratio[il] + 1) * c.idx_dim * sizeof(float));
            }
        }
        if (mtp_.loaded()) need += mtp_.ctx_bytes_needed(cells);
        if (!budget_.make_room(need) && !evict_idle(need))
            throw std::runtime_error("not enough VRAM for a context of " + std::to_string(cells) + " (" + budget_.describe() + ")");
    }
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
        if (cur_->kv_raw[il].reserved()) {
            const size_t raw = (size_t) cells * c.idx_dim * sizeof(half);
            const size_t pool = (size_t) (cells / c.compress_ratio[il] + 1) * c.idx_dim * sizeof(float);
            if (grow) {
                cur_->kv_raw[il].ensure(raw);
                cur_->kv_pool[il].ensure(pool);
            } else {
                cur_->kv_raw[il].shrink_to(raw);
                cur_->kv_pool[il].shrink_to(pool);
            }
        }
    }
    if (mtp_.loaded()) {
        if (grow) mtp_.ensure_ctx(cells);
        else mtp_.release_ctx(cells);
    }
    cur_->ctx_mapped = cells;
}

// After a prompt: the prompt buffers go back to their baseline, the context keeps what it uses (plus a step),
// and the expert tier takes the free bytes back.
void Engine::rebalance() {
    for (Engine * s = this; s; s = s->next_.get()) s->rebalance_here();
}

void Engine::rebalance_here() {
    DeviceScope ds(dev_);
    if (pf_base_ > 0) {
        prefill_layout(pf_small_);
        prefill_staging_shrink();
    }
    // a window's worth past the position, as ensure_ctx keeps: the first drafts (Generator::start) write the
    // drafter's K/V at pos, pos + 1 ... before any forward maps more (one cell of headroom wrote past the mapping)
    const int keep = std::min(opt_.max_ctx, std::max(kCtxBaseline, (pos() + kMaxWindow + kCtxStep - 1) / kCtxStep * kCtxStep));
    if (keep < cur_->ctx_mapped) map_ctx(keep);
    const size_t slack = 64ull << 20;
    budget_.make_room(0);   // re-syncs with the driver; VRAM taken outside the budget since then leaves the cache
    size_t room = budget_.free() > slack ? budget_.free() - slack : 0;
    if (opt_.expert_cache_gib >= 0) {   // a configured cache size is a cap, also when growing back
        const size_t cap = (size_t) (opt_.expert_cache_gib * 1073741824.0);
        room = std::min(room, cap > cache_.bytes() ? cap - cache_.bytes() : 0);
    }
    if (room > 0) cache_.grow(room, st_);
}

int Engine::checkpoints() const {
    int n = 0;
    for (const auto & k : cur_->ckpts) n += k.pos >= 0;
    return n;
}

void Engine::checkpoint() {
    for (Engine * s = this; s; s = s->next_.get()) s->checkpoint_here();
}

void Engine::checkpoint_here() {
    DeviceScope ds(dev_);
    join_commit_here();
    const int p = pos();
    if (p == 0) return;
    for (auto & k : cur_->ckpts)
        if (k.pos == p) { k.age = ++ckpt_age_; return; }
    Checkpoint * slot = nullptr;
    for (auto & k : cur_->ckpts)
        if (k.pos < 0) { slot = &k; break; }
    if (!slot && (int) cur_->ckpts.size() < kMaxCheckpoints) {
        cur_->ckpts.emplace_back();
        slot = &cur_->ckpts.back();
        slot->host = ck_take();
    }
    if (!slot)  // the least recently used one goes
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

int Engine::rollback(int max_pos) {
    const int r = rollback_here(max_pos);
    for (Engine * s = next_.get(); s; s = s->next_.get())
        if (s->rollback_here(max_pos) != r) throw std::runtime_error("rollback: the pipeline stages disagree");
    return r;
}

int Engine::rollback_here(int max_pos) {
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
    best->age = ++ckpt_age_;
    for (auto & k : cur_->ckpts)  // snapshots past the cut describe a history that is gone
        if (k.pos > best->pos) k.pos = -1;
    return best->pos;
}

void Engine::reset() {
    for (Engine * s = this; s; s = s->next_.get()) s->reset_here();
}

void Engine::reset_here() {
    DeviceScope ds(dev_);
    join_commit_here();
    const Config & c = model_.cfg;
    cur_->history.clear();
    for (auto & k : cur_->ckpts) k.pos = -1;
    cur_->mtp_cell = -1;
    for (int il = l0_; il < l1_; ++il) {
        if (!c.is_attn(il)) {
            CUDA_CHECK(cudaMemsetAsync(cur_->conv_buf[il].p, 0, cur_->conv_buf[il].n * 4, st_));
            CUDA_CHECK(cudaMemsetAsync(cur_->ssm_state[il].p, 0, cur_->ssm_state[il].n * 4, st_));
        }
    }
    if (cur_->ple_hist.p) CUDA_CHECK(cudaMemsetAsync(cur_->ple_hist.p, 0, cur_->ple_hist.n * 4, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    // a new conversation: the context's VRAM back to its baseline (the cache takes it at the next rebalance)
    if (cur_->ctx_mapped > kCtxBaseline) map_ctx(std::min(opt_.max_ctx, kCtxBaseline));
}

void Engine::save_counts() {
    for (Engine * s = this; s; s = s->next_.get()) s->save_counts_here();
}

void Engine::save_counts_here() {
    // the device counters are cumulative for the session; the file gets this session's total added (each stage
    // counts its own layers only and saves after the one before, so every stage adds its part)
    if (opt_.counts_out.empty()) return;
    DeviceScope ds(dev_);
    const Config & c = model_.cfg;
    std::vector<uint32_t> now((size_t) c.n_layer * c.n_expert);
    CUDA_CHECK(cudaMemcpy(now.data(), counts_.p, now.size() * 4, cudaMemcpyDeviceToHost));
    if (saved_counts_.size() != now.size()) saved_counts_.assign(now.size(), 0);
    std::vector<uint32_t> delta(now.size());
    for (size_t i = 0; i < now.size(); ++i) delta[i] = now[i] - saved_counts_[i];
    saved_counts_ = now;
    auto old = load_counts(opt_.counts_out, c.n_layer, c.n_expert);
    if (old.size() == delta.size())
        for (size_t i = 0; i < delta.size(); ++i) delta[i] += old[i];
    bnk::save_counts(opt_.counts_out, delta, c.n_layer, c.n_expert);
}

// xn = norm(res); mixed = mean(xn * sigmoid(up(silu(down(xn)/hc)))); inj = inject(xn)
void Engine::hc_pre(const HcWeights & w, int T, bool want_inject) {
    const Config & c = model_.cfg;
    const int E = c.n_embd, HC = c.hc_dim();
    // the norm quantizes its output for down / inject in the same pass (BNK_FUSE_QUANT=0: separately)
    static const bool fuse = !getenv("BNK_FUSE_QUANT") || atoi(getenv("BNK_FUSE_QUANT")) != 0;
    // a pending hc_combine (the previous block's output into res_) goes into the same pass when fused
    const float * pend = pend_out_;
    pend_out_ = nullptr;
    if (fuse && (!is_float_format(w.down.type) || (want_inject && !is_float_format(w.inject.type)))) {
        hc_norm_q(res_, F(w.norm), xn_, T, c.hc, E, c.rms_eps, act_, st_, pend, inj_);
    } else {
        if (pend) hc_combine(res_, pend, inj_, T, c.hc, E, st_);
        hc_norm(res_, F(w.norm), xn_, T, c.hc, E, c.rms_eps, st_);
        if (!is_float_format(w.down.type) || (want_inject && !is_float_format(w.inject.type)))
            quantize_act(xn_, HC, T, HC, act_, st_);
    }
    // the inject projection: stacked under down (one launch, the rows from hc_rank on into inj_), else beside
    // down -> up -> mix on a side stream (only when up does not quantize into act_, which inject reads)
    const bool stacked = want_inject && w.down_inject.valid();
    const bool side = want_inject && !stacked && is_float_format(w.up.type);
    if (side) gemv_q(w.inject, act_, xn_, HC, T, inj_, c.hc, fork(1));
    if (stacked) gemv_q(w.down_inject, act_, xn_, HC, T, lo_, c.hc_rank, st_, GemvSplit{c.hc_rank, inj_, c.hc});
    else gemv_q(w.down, act_, xn_, HC, T, lo_, c.hc_rank, st_);
    if (want_inject && !side && !stacked) gemv_q(w.inject, act_, xn_, HC, T, inj_, c.hc, st_);
    // silu(lo / hc) inside the quantization pass of w.up's input (one launch instead of two)
    if (!is_float_format(w.up.type) && !use_ref_gemv()) {
        quantize_act_silu(lo_, 1.f / c.hc, c.hc_rank, T, c.hc_rank, act_, st_);
        gemv(w.up, &act_, nullptr, 0, T, gpre_, HC, false, st_);
    } else {
        silu_scale(lo_, (int64_t) T * c.hc_rank, 1.f / c.hc, st_);
        gemv_auto(w.up, lo_, c.hc_rank, T, gpre_, HC, false, act_, st_);
    }
    // mixed and its quantization (mixact_, for the attention / DeltaNet / MoE that follow) in one pass
    if (fuse) {
        hc_mix_q(xn_, gpre_, mixed_, T, c.hc, E, mixact_, st_);
        mix_quantized_ = true;
    } else {
        hc_mix(xn_, gpre_, mixed_, T, c.hc, E, st_);
    }
    if (side) join(1);
}

// host side: the n-gram hash rows of the window, gathered into pinned memory the graph reads
void Engine::ple_gather() {
    const Config & c = model_.cfg;
    if (c.ple_layer < 0 || !owns(c.ple_layer)) return;
    const double t0 = now_ms();
    const int ph = c.ple_heads();
    const TensorRef & tab = *model_.ple_table;
    const size_t rb = tab.row_bytes();
    for (const Win & w : wins_)
    for (int tw = 0; tw < w.T; ++tw) {
        const int p = w.pos0 + tw, t = w.r0 + tw;
        const std::vector<int32_t> & hist = w.q->history;
        int64_t ctx[8];
        ctx[0] = hist[p];
        bool cut = false;
        for (int s = 1; s < c.ple_ngram; ++s) {
            const int q = p - s;
            const int64_t tok = (cut || q < 0) ? -1 : hist[q];
            cut = cut || tok < 0 || tok == c.ple_eos;
            ctx[s] = cut ? c.ple_eos : tok;
        }
        for (int n = 2; n <= c.ple_ngram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * c.ple_mult[0];
            for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * c.ple_mult[j];
            for (int g = 0; g < c.ple_heads_per_ngram; ++g) {
                const int h = (n - 2) * c.ple_heads_per_ngram + g;
                const uint64_t row = mixed % c.ple_vocab[h] + c.ple_offset[h];
                memcpy(h_ple_rows_ + ((size_t) t * ph + h) * rb, tab.data + row * rb, rb);
            }
        }
    }
    head_->times.ple_ms += now_ms() - t0;
}

void Engine::ple(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, HC = c.hc_dim(), ph = c.ple_heads();
    const TensorRef & tab = *model_.ple_table;
    QMat rows{h_ple_rows_, (int) tab.type, (int64_t) T * ph, tab.ne[0], tab.row_bytes()};
    rows.scale = model_.ple_scale;
    dequant_rows(rows, 0, (int64_t) T * ph, ple_emb_, st_);
    const int D = ph * c.ple_dim;
    if (!is_float_format(L.ple_key.type)) quantize_act(ple_emb_, D, T, D, act_, st_);
    dbg(st_, il, "ple_emb", ple_emb_.p, (size_t) T * D);
    gemv_q(L.ple_key, act_, ple_emb_, D, T, ple_key_, HC, st_);
    gemv_q(L.ple_value, act_, ple_emb_, D, T, ple_val_, E, st_);
    dbg(st_, il, "ple_key", ple_key_.p, (size_t) T * HC);
    dbg(st_, il, "ple_val", ple_val_.p, (size_t) T * E);
    const int hist = (c.ple_conv - 1) * c.ple_ngram;
    for (const Win & w : wins_) {
        const size_t o = (size_t) w.r0 * HC;
        ple_gate(ple_key_.p + o, res_.p + o, ple_val_.p + (size_t) w.r0 * E, F(L.ple_norm_key), F(L.ple_norm_query),
                 F(L.ple_norm_conv), ple_gated_.p + o, w.q->ple_hist.p + (size_t) hist * HC, w.T, c.hc, E, c.rms_eps,
                 st_);
        ple_conv_add(res_.p + o, ple_gated_.p + o, w.q->ple_hist, (const half *) L.ple_conv1d.data, w.T, HC,
                     c.ple_conv, c.ple_ngram, st_);
        if (w.commit) shift_rows(w.q->ple_hist, HC, w.T, hist, st_);
    }
    dbg(st_, il, "ple_gated", ple_gated_.p, (size_t) T * HC);
    dbg(st_, il, "ple_res", res_.p, (size_t) T * HC);
}

void Engine::gdn(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, C = c.conv_channels(), K = c.ssm_conv, S = c.ssm_state, nv = c.ssm_vheads;
    quantize_mixed(T);
    float * co = gdn_co_[lane_][il], * gg = gdn_g_[lane_][il], * gb = gdn_b_[lane_][il];
    {   // beta / alpha (needed at the prep) and the gate (at the norm) beside wqkv and its copies
        cudaStream_t sb = fork(0);
        if (L.ssm_ba.valid()) gemv_q(L.ssm_ba, mixact_, mixed_, E, T, gb, nv, sb, GemvSplit{nv, gg, nv});
        else {
            gemv_q(L.ssm_beta, mixact_, mixed_, E, T, gb, nv, sb);
            gemv_q(L.ssm_alpha, mixact_, mixed_, E, T, gg, nv, sb);
        }
        gemv_q(L.wgate, mixact_, mixed_, E, T, z_, nv * S, sb);
    }
    // the new conv inputs go behind each window's own K-1 history rows (one window: straight there)
    const bool one = wins_.size() == 1;
    float * qkv = one ? wins_[0].q->conv_buf[il].p + (size_t) (K - 1) * C : conv_out_.p;
    gemv_q(L.wqkv, mixact_, mixed_, E, T, qkv, C, st_);
    if (!one)
        for (const Win & w : wins_)
            CUDA_CHECK(cudaMemcpyAsync(w.q->conv_buf[il].p + (size_t) (K - 1) * C, conv_out_.p + (size_t) w.r0 * C,
                                       (size_t) w.T * C * 4, cudaMemcpyDeviceToDevice, st_));
    join(0);
    dbg(st_, il, "hc_mixed_attn", mixed_, (size_t) T * E);
    dbg(st_, il, "linear_attn_qkv_mixed", qkv, (size_t) T * C);
    dbg(st_, il, "z", z_, (size_t) T * nv * S);
    for (const Win & w : wins_) {
        float * cw = co + (size_t) w.r0 * C, * gw = gg + (size_t) w.r0 * nv, * bw = gb + (size_t) w.r0 * nv;
        gdn_conv(w.q->conv_buf[il], F(L.ssm_conv1d), cw, w.T, C, K, st_);
        gdn_prep(cw, w.T, C, c.ssm_groups, nv, S, gw, bw, F(L.ssm_dt), F(L.ssm_a), c.rms_eps, st_);
        gdn_recurrence(cw, C, gw, bw, w.q->ssm_state[il], gdn_o_.p + (size_t) w.r0 * nv * S, w.T, c.ssm_groups, nv,
                       S, w.commit ? w.T : 0, st_);
    }
    dbg(st_, il, "conv_output_silu", co, (size_t) T * C);
    dbg(st_, il, "attn_output", gdn_o_, (size_t) T * nv * S);
    // the gated norm inside the quantization pass of ssm_out's input (one launch instead of two)
    if (!is_float_format(L.ssm_out.type) && !use_ref_gemv() && !getenv("BNK_DUMP_DIR") &&
        quantize_gated_norm(gdn_o_, z_, F(L.ssm_norm), T, nv, S, c.rms_eps, act_, st_)) {
        gemv(L.ssm_out, &act_, nullptr, 0, T, out_, E, false, st_);
    } else {
        gated_rmsnorm(gdn_o_, z_, F(L.ssm_norm), gdn_n_, T, nv, S, c.rms_eps, st_);
        dbg(st_, il, "final_output", gdn_n_, (size_t) T * nv * S);
        gemv_auto(L.ssm_out, gdn_n_, nv * S, T, out_, E, false, act_, st_);
    }
    dbg(st_, il, "linear_attn_out", out_, (size_t) T * E);
    for (const Win & w : wins_)
        if (w.commit) shift_rows(w.q->conv_buf[il], C, w.T, K - 1, st_);
}

// Keep the first c rows of the last verify window: replay the delta rule over them and shift the histories.
void Engine::enqueue_commit(int cnt) {
    const Config & c = model_.cfg;
    const int C = c.conv_channels(), K = c.ssm_conv, S = c.ssm_state, nv = c.ssm_vheads;
    const size_t r0 = (size_t) cur_->last_r0;
    for (int il = l0_; il < l1_; ++il) {
        if (c.is_attn(il)) continue;
        const int ln = cur_->last_lane;
        gdn_recurrence(gdn_co_[ln][il].p + r0 * C, C, gdn_g_[ln][il].p + r0 * nv, gdn_b_[ln][il].p + r0 * nv, cur_->ssm_state[il],
                       nullptr, cnt, c.ssm_groups, nv, S, cnt, st_);
    }
    if (n_gdn_) shift_rows_multi(cur_->conv_ptrs, n_gdn_, C, cnt, K - 1, st_);
    if (cur_->ple_hist.p) shift_rows(cur_->ple_hist, c.hc_dim(), cnt, (c.ple_conv - 1) * c.ple_ngram, st_);
}

void Engine::join_commit() {
    for (Engine * s = this; s; s = s->next_.get()) s->join_commit_here();
}

void Engine::join_commit_here() {
    if (!commit_inflight_) return;
    DeviceScope ds(dev_);
    CUDA_CHECK(cudaStreamWaitEvent(st_, ev_committed_, 0));
    commit_inflight_ = false;
}

void Engine::commit(int cnt) {
    for (Engine * s = this; s; s = s->next_.get()) s->commit_here(cnt);
}

void Engine::commit_here(int cnt) {
    DeviceScope ds(dev_);
    if (!cur_->conv_ptrs.p) {   // the GDN conv buffers never move: their pointer table once, before any capture
        const Config & c = model_.cfg;
        std::vector<float *> ptrs;
        for (int il = l0_; il < l1_; ++il)
            if (!c.is_attn(il)) ptrs.push_back(cur_->conv_buf[il].p);
        n_gdn_ = (int) ptrs.size();
        cur_->conv_ptrs.alloc(ptrs.size());
        CUDA_CHECK(cudaMemcpy(cur_->conv_ptrs.p, ptrs.data(), ptrs.size() * sizeof(float *), cudaMemcpyHostToDevice));
    }
    if (cur_->pending_T == 0) throw std::runtime_error("commit: no verify window pending");
    if (cnt < 1 || cnt > cur_->pending_T) throw std::runtime_error("commit: count out of range");
    cur_->history.resize(cur_->history.size() - (cur_->pending_T - cnt));
    cur_->pending_T = 0;
    static const bool same_stream = getenv("BNK_COMMIT_SAME_STREAM") != nullptr;
    if (opt_.use_graphs && !same_stream) {
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
    } else if (opt_.use_graphs) {
        cudaGraphExec_t & cg = cur_->commit_graphs[cur_->last_lane][cur_->last_r0][cnt];
        if (!cg) {
            cudaGraph_t g;
            CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
            enqueue_commit(cnt);
            CUDA_CHECK(cudaStreamEndCapture(st_, &g));
            CUDA_CHECK(cudaGraphInstantiate(&cg, g, 0));
            cudaGraphDestroy(g);
        }
        CUDA_CHECK(cudaGraphLaunch(cg, st_));
    } else {
        enqueue_commit(cnt);
    }
}

void Engine::attn(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, H = c.n_head, Hkv = c.n_head_kv, D = c.head_dim;
    quantize_mixed(T);
    const bool qsa = wins_[0].q->kraw[il].p != nullptr;
    if (qsa) {   // the indexer's projections beside q / k / v
        cudaStream_t sb = fork(0);
        if (L.idx_kq.valid()) gemv_q(L.idx_kq, mixact_, mixed_, E, T, ik_, c.idx_dim, sb, GemvSplit{c.idx_dim, iq_, (int64_t) c.idx_heads * c.idx_dim});
        else {
            gemv_q(L.idx_k, mixact_, mixed_, E, T, ik_, c.idx_dim, sb);
            gemv_q(L.idx_q, mixact_, mixed_, E, T, iq_, c.idx_heads * c.idx_dim, sb);
        }
    }
    gemv_q(L.wq, mixact_, mixed_, E, T, qfull_, H * D * 2, st_);
    if (L.wkv.valid()) gemv_q(L.wkv, mixact_, mixed_, E, T, k_, Hkv * D, st_, GemvSplit{Hkv * D, v_, (int64_t) Hkv * D});
    else {
        gemv_q(L.wk, mixact_, mixed_, E, T, k_, Hkv * D, st_);
        gemv_q(L.wv, mixact_, mixed_, E, T, v_, Hkv * D, st_);
    }
    if (qsa) join(0);
    for (size_t wi = 0; wi < wins_.size(); ++wi) {
        const Win & w = wins_[wi];
        const int * pos0 = &d_par_[wi].pos0;
        const size_t r = (size_t) w.r0;
        float * qw = q_.p + r * H * D, * qf = qfull_.p + r * H * D * 2, * ao = attn_o_.p + r * H * D;
        attn_prep(qf, k_.p + r * Hkv * D, v_.p + r * Hkv * D, F(L.q_norm), F(L.k_norm), qw, w.q->kc[il], w.q->vc[il],
                  w.T, H, Hkv, D, c.n_rot, c.rope_base, pos0, c.rms_eps, st_);
        if (qsa) {
            // QSA: indexer keys and queries, pooled blocks, the per-query block selection, sparse attention
            float * iqw = iq_.p + r * c.idx_heads * c.idx_dim;
            qsa_store_keys(ik_.p + r * c.idx_dim, w.q->kraw[il], w.T, c.idx_dim, pos0, st_);
            qsa_queries(iqw, F(L.idx_q_norm), w.T, qsh_, pos0, st_);
            qsa_pool(w.q->kraw[il], F(L.idx_k_norm), w.q->pooled[il], w.T, qsh_, pos0, st_);
            qsa_select(iqw, w.q->pooled[il], qsa_scores_, max_blocks_, qsa_sel_, qsa_nsel_, w.T, qsh_, pos0, 0, st_);
            qsa_attention(qw, w.q->kc[il], w.q->vc[il], qf, qsa_sel_, qsa_nsel_, ao, w.T, qsh_, pos0,
                          1.f / sqrtf((float) D), attn_scratch_, st_);
        } else {
            attention(qw, w.q->kc[il], w.q->vc[il], qf, ao, w.T, H, Hkv, D, pos0, 1.f / sqrtf((float) D),
                      attn_scratch_, st_);
        }
    }
    gemv_auto(L.wo, attn_o_, H * D, T, out_, E, false, act_, st_);
}

// Measurement only (BNK_PREDICT_STATS, eager mode): how well would layer il+1's router, run on layer il's MoE
// input, predict layer il+1's experts? Counted against the misses (non-resident experts) the CPU would compute.
void Engine::predict_stats(int il, int T) {
    const Config & c = model_.cfg;
    const int k = c.n_expert_used, P = 16;
    static std::vector<int32_t> prev;   // [T][P] predicted for this layer (from the previous one)
    static int prev_T = 0, prev_layer = -1;
    static int64_t miss = 0, hit10 = 0, hit16 = 0, extra10 = 0, extra16 = 0, layers = 0;
    std::vector<int32_t> ids((size_t) T * k);
    CUDA_CHECK(cudaMemcpy(ids.data(), rids_.p, ids.size() * 4, cudaMemcpyDeviceToHost));
    if (prev_layer == il && prev_T == T) {
        for (int Pn : {10, 16}) {
            std::vector<char> pred(c.n_expert, 0);   // the union over the window's tokens
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < Pn; ++j) pred[prev[(size_t) t * P + j]] = 1;
            std::vector<char> needed(c.n_expert, 0);
            for (int i = 0; i < T * k; ++i) {
                const int e = ids[i];
                if (cache_.slot_of(il, e) >= 0) continue;
                if (Pn == 10) ++miss;
                needed[e] = 1;
                (Pn == 10 ? hit10 : hit16) += pred[e];
            }
            for (int e = 0; e < c.n_expert; ++e)   // fetched for nothing: predicted, non-resident, not used
                if (pred[e] && !needed[e] && cache_.slot_of(il, e) < 0) (Pn == 10 ? extra10 : extra16)++;
        }
        ++layers;
    }
    prev_layer = -1;
    if (il + 1 < l1_) {
        if (!pred_logits_.p) {
            pred_logits_.alloc((size_t) kMaxWindow * c.n_expert);
            pred_ids_.alloc((size_t) kMaxWindow * 32);
            pred_w_.alloc((size_t) kMaxWindow * 32);
        }
        gemv_q(model_.layers[il + 1].router, mixact_, mixed_, c.n_embd, T, pred_logits_, c.n_expert, st_);
        route_topk(pred_logits_, T, c.n_expert, P, pred_ids_, pred_w_, 1.f, st_);
        prev.resize((size_t) T * P);
        CUDA_CHECK(cudaMemcpyAsync(prev.data(), pred_ids_.p, prev.size() * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
        prev_T = T;
        prev_layer = il + 1;
    }
    if (layers && layers % 2000 == 0)
        fprintf(stderr, "predict: %lld layer steps, %lld misses; next-layer router on this layer's input covers "
                        "%.1f%% (top-10 per token, %.1f extra experts fetched/layer) or %.1f%% (top-16, %.1f extra)\n",
                (long long) layers, (long long) miss, 100.0 * hit10 / std::max<int64_t>(1, miss),
                (double) extra10 / layers, 100.0 * hit16 / std::max<int64_t>(1, miss), (double) extra16 / layers);
}

void Engine::moe(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, k = c.n_expert_used, FF = c.n_ff_shexp;
    const MoeLayerDesc d = cache_.desc(il);
    quantize_mixed(T);
    // the shared expert (and its gate) needs only mixed / mixact: beside the routed experts (act_ is its own: the
    // routed path does not touch it)
    {
        cudaStream_t sb = fork(0);
        if (L.sh_gu.valid()) gemv_q(L.sh_gu, mixact_, mixed_, E, T, sg_, FF, sb, GemvSplit{FF, su_, FF});
        else {
            gemv_q(L.sh_gate, mixact_, mixed_, E, T, sg_, FF, sb);
            gemv_q(L.sh_up, mixact_, mixed_, E, T, su_, FF, sb);
        }
        // silu(gate) * up inside the quantization pass of sh_down's input; the gate's sigmoid in moe_reduce
        if (!is_float_format(L.sh_down.type) && !use_ref_gemv()) {
            quantize_act_silu_mul(sg_, su_, FF, T, FF, act_, sb);
            gemv(L.sh_down, &act_, nullptr, 0, T, shared_out_, E, false, sb);
        } else {
            silu_mul(sg_, su_, sh_, (int64_t) T * FF, sb);
            gemv_auto(L.sh_down, sh_, FF, T, shared_out_, E, false, act_, sb);
        }
        if (!L.router_sg.valid()) gemv_q(L.shexp_gate_inp, mixact_, mixed_, E, T, sgate_, 1, sb);
    }
    if (L.router_sg.valid()) gemv_q(L.router_sg, mixact_, mixed_, E, T, rlogits_, c.n_expert, st_, GemvSplit{c.n_expert, sgate_, 1});
    else gemv_q(L.router, mixact_, mixed_, E, T, rlogits_, c.n_expert, st_);
    // the experts' choice and the hit list in one launch (BNK_FUSE_PLAN=0: two)
    static const bool fuse_plan = !getenv("BNK_FUSE_PLAN") || atoi(getenv("BNK_FUSE_PLAN")) != 0;
    if (fuse_plan) {
        moe_route_plan(rlogits_, c.n_expert, c.expert_weights_scale, rids_, rw_, T, k, d, moes_, msg(il), mixed_, E,
                       &d_par_->seq, counts_.p + (size_t) il * c.n_expert, st_);
    } else {
        route_topk(rlogits_, T, c.n_expert, k, rids_, rw_, c.expert_weights_scale, st_);
        moe_plan(rids_, rw_, T, k, d, moes_, msg(il), mixed_, E, &d_par_->seq, counts_.p + (size_t) il * c.n_expert,
                 st_);
    }
    if (route_log_)
        CUDA_CHECK(cudaMemcpyAsync(route_dev_.p + (size_t) il * kMaxWindow * k, rids_.p, (size_t) T * k * 4,
                                   cudaMemcpyDeviceToDevice, st_));
    if (!opt_.use_graphs || dump_all) {  // eager mode: answer the CPU part right here
        CUDA_CHECK(cudaStreamSynchronize(st_));
        static const bool pstats = getenv("BNK_PREDICT_STATS") != nullptr;
        if (pstats) predict_stats(il, T);
        MoeMsg * m = msg(il);
        tasks_.clear();
        for (int i = 0; i < m->n_miss; ++i) tasks_.push_back({m->miss_t[i], 0, m->miss_e[i], m->miss_w[i]});
        if (!tasks_.empty()) cpu_->run(il, T, m->x(), tasks_, m->out(E));
        head_->times.misses += m->n_miss;
        head_->times.routed += (int64_t) T * k;
        head_->layer_misses[il] += m->n_miss;
        head_->layer_routed[il] += (int64_t) T * k;
        __atomic_thread_fence(__ATOMIC_RELEASE);
        m->seq_done = m->seq_req;
    }
    moe_hits(d, moes_, mixact_, T, k, E, c.n_ff_exp, st_);
    join(0);
    moe_reduce(moes_, msg(il), shared_out_, sgate_, moe_out_, T, k, E, st_, &d_par_->seq, true);   // (waits for the CPU too)
}

void Engine::head(int T) {
    const Config & c = model_.cfg;
    hc_pre(model_.hc_head, T, false);
    gemv_auto(model_.output, mixed_, c.n_embd, T, logits_, c.n_vocab, false, act_, st_);
}

void Engine::layer_forward(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    if (il == c.ple_layer) {   // the PLE adds into res_: a pending combine first
        if (pend_out_) hc_combine(res_, pend_out_, inj_, T, c.hc, c.n_embd, st_);
        pend_out_ = nullptr;
        ple(il, T);
    }
    hc_pre(L.hc_attn, T, true);
    if (L.attn) attn(il, T); else gdn(il, T);
    // the combines go into the next norm's pass (hc_pre) where one follows: after the attention always, after the
    // MoE unless this is the stage's last layer (the residual leaves the stage) or layers are being dumped
    static const bool defer = (!getenv("BNK_FUSE_QUANT") || atoi(getenv("BNK_FUSE_QUANT")) != 0) && !getenv("BNK_DUMP_DIR");
    const bool dfr = defer && !head_->dump_all;
    if (dfr) pend_out_ = out_;
    else hc_combine(res_, out_, inj_, T, c.hc, c.n_embd, st_);
    dbg(st_, il, "hc_combine", res_, (size_t) T * c.hc_dim());
    hc_pre(L.hc_ffn, T, true);
    dbg(st_, il, "hc_mixed_ffn", mixed_, (size_t) T * c.n_embd);
    dbg(st_, il, "hc_inject_ffn", inj_, (size_t) T * c.hc);
    moe(il, T);
    dbg(st_, il, "ffn_out", moe_out_, (size_t) T * c.n_embd);
    if (dfr && il + 1 < l1_) pend_out_ = moe_out_;
    else hc_combine(res_, moe_out_, inj_, T, c.hc, c.n_embd, st_);
    dbg(st_, il, "l_last", res_, (size_t) T * c.hc_dim());
    if (head_->dump_all) {
        auto & dl = head_->dumped_layers;
        dl.resize(c.n_layer);
        dl[il].resize((size_t) T * c.hc_dim());
        CUDA_CHECK(cudaMemcpyAsync(dl[il].data(), res_.p, dl[il].size() * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
    }
}

// This stage's part of a forward: the token embeddings (first stage) or the previous stage's residual rows, its
// layers, then the head (last stage) or the residual rows out to the next stage.
void Engine::enqueue_forward(int T) {
    const Config & c = model_.cfg;
    const size_t rows = (size_t) T * c.hc_dim() * 4;
    CUDA_CHECK(cudaMemcpyAsync(d_par_, h_par_, wins_.size() * sizeof(WinParams), cudaMemcpyHostToDevice, st_));
    if (first()) {
        dequant_gather(model_.tok_embd, h_tok_, T, x_, st_);
        hc_init(x_, res_, T, c.hc, c.n_embd, st_);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(res_.p, xfer_in_[lane_], rows, cudaMemcpyHostToDevice, st_));
    }
    pend_out_ = nullptr;
    for (int il = l0_; il < l1_; ++il) layer_forward(il, T);
    if (last_) head(T);
    else CUDA_CHECK(cudaMemcpyAsync(xfer_out_[lane_], res_.p, rows, cudaMemcpyDeviceToHost, st_));
}

void Engine::write_route_log(int T, int pos0, bool commit_all, double t0, double t1) {
    const Config & c = model_.cfg;
    const int L = c.n_layer, k = c.n_expert_used;
    std::vector<int32_t> ids((size_t) L * kMaxWindow * k);
    CUDA_CHECK(cudaMemcpy(ids.data(), route_dev_.p, ids.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<uint64_t> ts((size_t) 64 * 3);
    moe_debug_times((uint64_t (*)[3]) ts.data(), L);
    // the conversation: a hash of its first tokens
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < std::min<size_t>(cur_->history.size(), 256); ++i) h = (h ^ (uint32_t) cur_->history[i]) * 16777619u;
    const int32_t hdr[6] = {0x54554f52, T, pos0, commit_all ? 1 : 0, (int32_t) h, L};
    const double tt[2] = {t0, t1};
    fwrite(hdr, 4, 6, route_log_);
    fwrite(tt, 8, 2, route_log_);
    fwrite(ts.data(), 8, (size_t) L * 3, route_log_);
    fwrite(cpu_us_.data(), 4, L, route_log_);
    std::vector<int16_t> out((size_t) L * T * k);
    for (int il = 0; il < L; ++il)
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < k; ++j) {
                const int e = ids[((size_t) il * kMaxWindow + t) * k + j];
                out[((size_t) il * T + t) * k + j] = (int16_t) (cache_.slot_of(il, e) >= 0 ? e : -(e + 1));
            }
    fwrite(out.data(), 2, out.size(), route_log_);
    fflush(route_log_);
    std::fill(cpu_us_.begin(), cpu_us_.end(), 0.f);
}

void Engine::service_cpu(uint32_t seq, int T) {
    const Config & c = model_.cfg;
    const int E = c.n_embd;
    for (int il = l0_; il < l1_; ++il) {
        MoeMsg * m = msg(il);
        const double t0 = now_ms();
        while (__atomic_load_n(&m->seq_req, __ATOMIC_ACQUIRE) != seq) __builtin_ia32_pause();
        const double t1 = now_ms();
        head_->times.wait_ms += t1 - t0;
        const int n = m->n_miss;
        head_->times.misses += n;
        head_->times.routed += (int64_t) T * c.n_expert_used;
        head_->layer_misses[il] += n;
        head_->layer_routed[il] += (int64_t) T * c.n_expert_used;
        if (n > 0) {
            tasks_.clear();
            for (int i = 0; i < n; ++i) tasks_.push_back({m->miss_t[i], 0, m->miss_e[i], m->miss_w[i]});
            cpu_->run(il, T, m->x(), tasks_, m->out(E));
            __atomic_thread_fence(__ATOMIC_RELEASE);
            m->seq_done = seq;
            const double t2 = now_ms();
            head_->times.cpu_experts_ms += t2 - t1;
            if (route_log_) cpu_us_[il] = (float) ((t2 - t1) * 1e3);
            static const bool prof = getenv("BNK_SVC_PROF") != nullptr;
            if (prof) fprintf(stderr, "svc L%02d n=%d wait %.1f us  cpu %.1f us\n", il, n, (t1 - t0) * 1e3, (t2 - t1) * 1e3);
        }
    }
}

void Engine::forward(const int32_t * tokens, int T, bool commit_all) {
    forward_batch({BatchWin{current_slot(), tokens, T, commit_all}});
}

int Engine::row0_of(int slot) const {
    return seqs_.at(slot)->last_r0;
}

void Engine::forward_batch(const std::vector<BatchWin> & batch_in) {
    // windows in slot order, whatever order the caller lists them in: one graph per layout, not per permutation
    std::vector<BatchWin> batch = batch_in;
    std::sort(batch.begin(), batch.end(), [](const BatchWin & a, const BatchWin & b) { return a.slot < b.slot; });
    if (batch.empty() || batch.size() > (size_t) kMaxSlots) throw std::runtime_error("forward: bad batch");
    const double t0 = now_ms();
    // Every stage books the windows and captures its graph if this layout is new; only then are the graphs queued
    // (each stage's after the one before), and the host answers the CPU experts of each stage's layers in turn
    // while the GPUs work through them. Nothing may wait on a launched graph before its CPU part is answered: a
    // capture (which can synchronize its stream, e.g. when the expert cache gives up VRAM) comes first.
    for (Engine * s = this; s; s = s->next_.get()) s->fwd_prepare(batch);
    cudaEvent_t after = nullptr;
    for (Engine * s = this; s; s = s->next_.get()) {
        s->fwd_launch(after);
        after = s->ev_out_;
    }
    for (Engine * s = this; s; s = s->next_.get()) s->fwd_service();
    for (Engine * s = this; s; s = s->next_.get()) s->fwd_end();
    times.total_ms += now_ms() - t0;
    times.calls++;
}

void Engine::fwd_prepare(const std::vector<BatchWin> & batch, int lane) {
    DeviceScope ds(dev_);
    join_commit_here();
    if (lane < 0 || lane >= kLanes || !gdn_co_[lane].size() || (l1_ > l0_ && lane > 0 && first() && last_))
        throw std::runtime_error("forward: no lane " + std::to_string(lane));
    lane_ = lane;
    Seq * keep = cur_;
    int R = 0;
    for (const auto & b : batch) {
        if (b.T < 1) throw std::runtime_error("forward: bad window");
        R += b.T;
    }
    if (R > kMaxWindow) throw std::runtime_error("forward: more than kMaxWindow rows");
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
        ensure_ctx_here(pos() + b.T + kMaxWindow);
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
    ple_gather();
    fwd_graph_ = opt_.use_graphs && !head_->dump_all;
    fwd_exec_ = nullptr;
    if (fwd_graph_) {
        FwdGraph & fg = fwd_graphs_[key];
        if (!fg.exec) {
            while ((int) fwd_graphs_.size() > kMaxFwdGraphs) {   // the least recently used layout goes
                auto lru = fwd_graphs_.end();
                for (auto it = fwd_graphs_.begin(); it != fwd_graphs_.end(); ++it)
                    if (it->second.exec && (lru == fwd_graphs_.end() || it->second.used < lru->second.used)) lru = it;
                if (lru == fwd_graphs_.end()) break;
                // no refund: the driver keeps a destroyed graph's memory and builds the next graph in it (its
                // measured size is then ~0), so what the graphs hold is the high-water mark charged so far
                cudaGraphExecDestroy(lru->second.exec);
                fwd_graphs_.erase(lru);
            }
            // an instantiated graph holds device memory: the expert cache gives it up (charged to the budget,
            // at the measured size), so the reserve for cuBLAS and the other graphs stays intact
            budget_.make_room(kGraphBytesEst);
            size_t f0 = 0, f1 = 0, tot = 0;
            cudaMemGetInfo(&f0, &tot);
            cudaGraph_t g;
            CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
            enqueue_forward(R);
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

void Engine::fwd_launch(cudaEvent_t after) {
    DeviceScope ds(dev_);
    if (after) CUDA_CHECK(cudaStreamWaitEvent(st_, after, 0));   // the previous stage's residual rows
    CUDA_CHECK(cudaEventRecord(ev_t0_, st_));
    if (fwd_graph_) {
        if (route_log_) moe_debug_reset(st_);
        CUDA_CHECK(cudaGraphLaunch(fwd_exec_, st_));
    } else {
        enqueue_forward(last_T);   // eager: the CPU experts are answered inside
    }
    CUDA_CHECK(cudaEventRecord(ev_out_, st_));
    CUDA_CHECK(cudaEventRecord(ev_t1_, st_));
}

void Engine::forward_pipe(const std::vector<BatchWin> & first_in, int first_lane, const std::vector<BatchWin> & second_in,
                          int second_lane) {
    if (stages() != 2) throw std::runtime_error("forward_pipe: needs a pipeline of two GPUs");
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
    Engine * s1 = next_.get();
    const double t0 = now_ms();
    // the second batch's rows came from the first GPU in the previous call (already finished): no event to wait on
    if (!a.empty()) fwd_prepare(a, first_lane);
    if (!b.empty()) s1->fwd_prepare(b, second_lane);
    if (!a.empty()) fwd_launch(nullptr);
    if (!b.empty()) s1->fwd_launch(nullptr);
    // each GPU's CPU experts answered at the same time: the second's on a helper thread
    if (!b.empty()) {
        if (!svc_) svc_ = std::make_unique<Worker>();
        svc_->post([s1]() { s1->fwd_service(); });
    }
    std::exception_ptr err;
    try {
        if (!a.empty()) fwd_service();
    } catch (...) {
        err = std::current_exception();
    }
    if (!b.empty()) svc_->wait();
    if (err) std::rethrow_exception(err);
    if (!a.empty()) fwd_end();
    if (!b.empty()) s1->fwd_end();
    times.total_ms += now_ms() - t0;
    times.calls++;
}

void Engine::fwd_service() {
    if (!fwd_graph_) return;
    DeviceScope ds(dev_);
    service_cpu(seq_, last_T);
    static const bool prof = getenv("BNK_SVC_PROF") != nullptr;
    if (prof) {
        CUDA_CHECK(cudaStreamSynchronize(st_));
        uint64_t ts[64][3];
        moe_debug_times(ts, model_.cfg.n_layer);
        for (int il = l0_; il < l1_; ++il)
            fprintf(stderr, "gpu L%02d plan->wait %.1f us  wait %.1f us\n", il, (ts[il][1] - ts[il][0]) / 1e3,
                    (ts[il][2] - ts[il][1]) / 1e3);
    }
}

void Engine::fwd_end() {
    DeviceScope ds(dev_);
    CUDA_CHECK(cudaStreamSynchronize(st_));
    float ms = 0.f;
    if (cudaEventElapsedTime(&ms, ev_t0_, ev_t1_) == cudaSuccess) {
        gpu_ms_ += ms;
        gpu_forwards_++;
    } else {
        cudaGetLastError();
    }
    if (route_log_ && fwd_graph_) write_route_log(last_T, wins_[0].pos0, wins_[0].commit, fwd_t0_, now_ms());
    for (const Win & w : wins_) w.q->pending_T = w.commit ? 0 : w.T;
    if (opt_.adapt_every > 0 && ++fwd_count_ % opt_.adapt_every == 0) cache_.adapt(counts_, st_, opt_.adapt_swaps);
}

// The MTP layer's K/V for rows just processed: each row pairs with the token after it; the last row waits
// for its successor (the next chunk's first token, or the first generated token).
void Engine::mtp_feed(const float * R_rows, const int32_t * tokens, int n, int pos0) {
    if (!mtp_.loaded()) return;
    DeviceScope ds(dev_);
    const int HC = model_.cfg.hc_dim();
    if (!cur_->mtp_R) CUDA_CHECK(cudaMalloc(&cur_->mtp_R, (size_t) HC * 4));
    if (cur_->mtp_cell >= 0 && cur_->mtp_cell == pos0 - 1) mtp_.run(cur_->mtp_R, tokens, 1, cur_->mtp_cell, false, nullptr);
    for (int r0 = 0; r0 < n - 1; r0 += kMaxWindow) {
        const int rr = std::min(kMaxWindow, n - 1 - r0);
        mtp_.run(R_rows + (size_t) r0 * HC, tokens + r0 + 1, rr, pos0 + r0, false, nullptr);
    }
    CUDA_CHECK(cudaMemcpyAsync(cur_->mtp_R, R_rows + (size_t) (n - 1) * HC, (size_t) HC * 4, cudaMemcpyDeviceToDevice, st_));
    cur_->mtp_cell = pos0 + n - 1;
}

size_t Engine::kv_bytes_mapped() const {
    size_t b = 0;
    for (size_t il = 0; il < cur_->kv_k.size(); ++il)
        b += cur_->kv_k[il].mapped() + cur_->kv_v[il].mapped() + cur_->kv_raw[il].mapped() + cur_->kv_pool[il].mapped();
    return b + (next_ ? next_->kv_bytes_mapped() : 0);
}

void Engine::prefill(const std::vector<int32_t> & tokens) {
    DeviceScope ds(dev_);
    join_commit();
    size_t i = 0;
    ensure_ctx(pos() + (int) tokens.size() + kMaxWindow);
    // the chunk layout for this read, borrowing VRAM from the expert cache meanwhile: a short turn keeps the
    // small layout and stage; longer reads get bigger chunks and a stage that holds a whole layer's outside experts.
    // Every stage reads the same chunks: the largest layout all of them fit.
    const int n = (int) tokens.size();
    if (pf_base_ > 0) {
        int want = (pf_big_ > pf_base_ && n >= 2 * pf_base_) ? pf_big_ : n > pf_small_ ? pf_base_ : pf_small_;
        // several GPUs: the pipeline is full only from the second chunk on, so a read in few big chunks leaves the
        // GPUs taking turns. Smaller chunks for shorter reads, bigger ones (more efficient per token) for long ones
        // (two V100s, cold reads: 16K 1,374 -> 1,616 tok/s with 4,096-token chunks; 8,192 stays best from ~48K)
        static const bool pipe_chunks = !getenv("BNK_PIPE_CHUNKS") || atoi(getenv("BNK_PIPE_CHUNKS")) != 0;
        if (next_ && pipe_chunks && want > pf_base_)
            want = n < 4 * pf_base_ ? pf_base_ : n < 24 * pf_base_ ? std::min(want, 2 * pf_base_) : want;
        // smaller chunks when VRAM is short (long contexts in every slot): slower, but the read goes through;
        // conversations are moved off the GPU only when even the smallest layout does not fit
        auto fits = [&](int N) {
            for (Engine * s = this; s; s = s->next_.get()) {
                DeviceScope d2(s->dev_);
                if (!s->try_layout(N)) return false;
            }
            return true;
        };
        int lay = 0;
        for (int pass = 0; pass < 2 && !lay; ++pass) {
            for (int N : {want, pf_base_, pf_small_})
                if (N <= want && fits(N)) { lay = N; break; }
            if (lay) break;
            bool room = pass == 0;
            for (Engine * s = this; s && room; s = s->next_.get()) {
                DeviceScope d2(s->dev_);
                if (!s->try_layout(pf_small_)) room = s->evict_idle(s->prefill_arena_bytes(pf_small_));
            }
            if (!room) throw std::runtime_error("not enough VRAM for the prompt buffers (" + budget_.describe() + ")");
        }
        if (lay < want && opt_.verbose)
            fprintf(stderr, "bnk: VRAM is short: reading this prompt in chunks of %d tokens (not %d)\n", lay, want);
        for (Engine * s = this; s; s = s->next_.get()) {
            DeviceScope d2(s->dev_);
            s->pf_full_ = n >= opt_.stage_full_min || std::min(lay, n) >= opt_.prefetch_min;
            s->staging_fit();
        }
    }
    Engine * t = tail();
    // several stages: chunk k+1 on the first stage while the worker runs chunk k through the others
    static const bool pipe = !getenv("BNK_PIPE_PROMPT") || atoi(getenv("BNK_PIPE_PROMPT")) != 0;
    if (next_ && pipe && !worker_) worker_ = std::make_unique<Worker>();
    // Chunks of even size: a read of a chunk and a bit would leave the later stages a tiny last chunk to overlap
    // with (2,100 tokens: 1,050 + 1,050 rather than 2,048 + 52). A read that fits one chunk stays whole: with the
    // experts on the tensor cores, a 603-token follow-up split in two measured 617-631 ms against 604 ms whole
    // (each chunk pays for its dense weights' dequantization); BNK_PIPE_PARTS=2 splits it anyway (it was worth
    // 14% with the grouped-GEMM expert path). BNK_PIPE_SPLIT=0 turns the even sizing off, N sets the shortest read.
    static const int split_min = getenv("BNK_PIPE_SPLIT") ? atoi(getenv("BNK_PIPE_SPLIT")) : 256;
    static const int min_parts = getenv("BNK_PIPE_PARTS") ? atoi(getenv("BNK_PIPE_PARTS")) : 1;
    int chunk = pf_max_;
    if (next_ && pipe && split_min > 0 && pf_max_ > 0 && n >= split_min && n >= 2 * opt_.prefill_min) {
        const int parts = std::max(min_parts, (n + pf_max_ - 1) / pf_max_);
        chunk = std::max(opt_.prefill_min, (n + parts - 1) / parts);
    }
    size_t handed = 0;     // tokens given to the later stages so far
    bool posted = false;   // a chunk is with the worker (its progress not reported yet)
    int buf = 0;
    auto drain = [&]() {
        if (!posted) return;
        worker_->wait();
        posted = false;
        if (on_prefill_progress && handed < tokens.size()) on_prefill_progress(handed, tokens.size());
    };
    // an error on this thread waits for the worker first: the caller resets the conversation on every stage next
    auto settle = [&]() {
        if (!posted) return;
        try {
            worker_->wait();
        } catch (...) {
        }
        posted = false;
    };
    while (i < tokens.size()) try {
        const size_t left = tokens.size() - i;
        const int pos0 = pos();
        if (pf_max_ > 0 && (int) left >= opt_.prefill_min && next_ && pipe) {
            const int N = (int) std::min<size_t>(chunk, left);
            const int32_t * tk = tokens.data() + i;
            const double c0 = now_ms();
            prefill_chunk_here(tk, N, buf);
            worker_->wait();   // the previous chunk is through every stage (its progress reported here)
            if (posted && on_prefill_progress) on_prefill_progress(handed, tokens.size());
            const int b = buf;
            worker_->post([this, t, tk, N, pos0, b, c0]() {
                for (Engine * s = next_.get(); s; s = s->next_.get()) s->prefill_chunk_here(tk, N, b);
                t->mtp_feed(t->pf_.res, tk, N, pos0);
                std::lock_guard<std::mutex> lk(stats_mu_);
                pstats.ms += now_ms() - c0;
                pstats.tokens += N;
                pstats.chunks++;
            });
            posted = true;
            buf ^= 1;
            i += N;
            handed = i;
            continue;
        }
        drain();   // a short tail or no pipeline: every stage idle first
        if (pf_max_ > 0 && (int) left >= opt_.prefill_min) {
            const int N = (int) std::min<size_t>(pf_max_, left);
            prefill_chunk(tokens.data() + i, N);
            t->mtp_feed(t->pf_.res, tokens.data() + i, N, pos0);
            i += N;
        } else {
            const int T = (int) std::min<size_t>(kMaxWindow, left);
            forward(tokens.data() + i, T);
            t->mtp_feed(t->res_, tokens.data() + i, T, pos0);
            i += T;
        }
        handed = i;
        if (on_prefill_progress) on_prefill_progress(i, tokens.size());
    } catch (...) {
        settle();
        throw;
    }
    if (posted) {
        worker_->wait();
        if (on_prefill_progress) on_prefill_progress(handed, tokens.size());
    }
    // the first token's logits are the last stage's last rows: other conversations decoded between the chunks may
    // have run through the first stages after this read's last chunk did
    for (Engine * s = this; s; s = s->next_.get()) s->last_T = t->last_T;
    const double r0 = now_ms();
    rebalance();
    last_rebalance_ms = now_ms() - r0;
}

int Engine::argmax(int t) {
    if (next_) return next_->argmax(t);
    DeviceScope ds(dev_);
    argmax_rows(logits_.p + (size_t) t * model_.cfg.n_vocab, 1, model_.cfg.n_vocab, argmax_dev_, st_);
    int32_t r;
    CUDA_CHECK(cudaMemcpyAsync(&r, argmax_dev_.p, 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    return r;
}

void Engine::argmax_all(int T, int32_t * out, int row0) {
    if (next_) return next_->argmax_all(T, out, row0);
    DeviceScope ds(dev_);
    argmax_rows(logits_.p + (size_t) row0 * model_.cfg.n_vocab, T, model_.cfg.n_vocab, argmax_dev_, st_);
    CUDA_CHECK(cudaMemcpyAsync(out, argmax_dev_.p, T * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void Engine::topk_rows_host(int row0, int T, int K, int32_t * ids, float * vals, double * z, float temp) {
    if (next_) return next_->topk_rows_host(row0, T, K, ids, vals, z, temp);
    DeviceScope ds(dev_);
    if (K > kTopkMax || T > kMaxWindow) throw std::runtime_error("topk_rows_host: too many");
    if (!topk_ids_.p) {
        topk_ids_.alloc((size_t) kMaxWindow * kTopkMax);
        topk_vals_.alloc((size_t) kMaxWindow * kTopkMax);
    }
    const int V = model_.cfg.n_vocab;
    topk_rows(logits_.p + (size_t) row0 * V, T, V, K, topk_ids_, topk_vals_, st_);
    CUDA_CHECK(cudaMemcpyAsync(ids, topk_ids_.p, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(vals, topk_vals_.p, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st_));
    if (z) {
        if (!topk_z_.p) topk_z_.alloc(kMaxWindow);
        row_expsum(logits_.p + (size_t) row0 * V, T, V, temp, topk_z_.p, st_);
        CUDA_CHECK(cudaMemcpyAsync(z, topk_z_.p, (size_t) T * sizeof(double), cudaMemcpyDeviceToHost, st_));
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void Engine::logits_rows_host(int T, float * out, int row0) {
    if (next_) return next_->logits_rows_host(T, out, row0);
    DeviceScope ds(dev_);
    CUDA_CHECK(cudaMemcpy(out, logits_.p + (size_t) row0 * model_.cfg.n_vocab, (size_t) T * model_.cfg.n_vocab * 4,
                          cudaMemcpyDeviceToHost));
}

void Engine::set_mtp_pending(const float * R_row_dev, int cell) {
    if (next_) return next_->set_mtp_pending(R_row_dev, cell);
    if (!mtp_.loaded()) return;
    DeviceScope ds(dev_);
    if (!cur_->mtp_R) CUDA_CHECK(cudaMalloc(&cur_->mtp_R, (size_t) model_.cfg.hc_dim() * 4));
    CUDA_CHECK(cudaMemcpyAsync(cur_->mtp_R, R_row_dev, (size_t) model_.cfg.hc_dim() * 4, cudaMemcpyDeviceToDevice, st_));
    cur_->mtp_cell = cell;
}

std::vector<float> Engine::logits_host(int t) {
    if (next_) return next_->logits_host(t);
    DeviceScope ds(dev_);
    std::vector<float> v(model_.cfg.n_vocab);
    CUDA_CHECK(cudaMemcpy(v.data(), logits_.p + (size_t) t * model_.cfg.n_vocab, v.size() * 4,
                          cudaMemcpyDeviceToHost));
    return v;
}

}  // namespace bnk
