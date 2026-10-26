#include "deepseek/ds_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <sys/mman.h>
#include <unistd.h>

#include "ggml-cpu.h"
#include "ggml.h"
#include "kernels/ops.h"

namespace bnk {

static const float * F(const QMat & m) { return (const float *) m.data; }

static void sync_debug(const char * what, int dev, int which = 3) {
    static const int on = env_int("BNK_DS_SYNC_DEBUG", 0);   // 1: after the forward, 2: around the exchange, 3: both
    if (!(on & which)) return;
    const cudaError_t e = cudaDeviceSynchronize(), l = cudaGetLastError();
    if (e != cudaSuccess || l != cudaSuccess)
        throw std::runtime_error(std::string("sync debug: ") + what + " on GPU " + std::to_string(dev) + ": " + cudaGetErrorString(e != cudaSuccess ? e : l));
}


// BNK_DS_TRACE="0,1": in eager mode, the rms of every row of a buffer after each stage of those layers
static bool trace_layer(int il) {
    static const std::string v = getenv("BNK_DS_TRACE") ? getenv("BNK_DS_TRACE") : "";
    if (v.empty()) return false;
    const std::string s = std::to_string(il);
    for (size_t a = 0; a < v.size();) {
        size_t b = v.find(',', a);
        if (b == std::string::npos) b = v.size();
        if (v.substr(a, b - a) == s) return true;
        a = b + 1;
    }
    return false;
}
static void trace(cudaStream_t st, int il, const char * what, const float * dev, int T, int64_t n, int64_t ld = -1) {
    if (!trace_layer(il)) return;
    if (ld < 0) ld = n;
    CUDA_CHECK(cudaStreamSynchronize(st));
    std::vector<float> h((size_t) T * ld);
    CUDA_CHECK(cudaMemcpy(h.data(), dev, h.size() * 4, cudaMemcpyDeviceToHost));
    fprintf(stderr, "trace L%d %-12s T=%d rms:", il, what, T);
    for (int t = 0; t < T; ++t) {
        double s = 0;
        for (int64_t i = 0; i < n; ++i) s += (double) h[(size_t) t * ld + i] * h[(size_t) t * ld + i];
        fprintf(stderr, " %.4g", sqrt(s / n));
    }
    fprintf(stderr, "\n");
}

// ------------------------------------------------------------------------------------------------ stages
std::vector<int> DsEngine::devices() const {
    std::vector<int> d{dev_};
    if (next_) for (int x : next_->devices()) d.push_back(x);
    return d;
}

StageInfo DsEngine::stage_info(int i) const {
    const DsEngine * s = stage(i);
    return StageInfo{s->dev_, s->l0_, s->l1_, s->cache_.resident(), s->cache_.bytes(), s->gpu_ms_, s->gpu_forwards_, s->pf_ms_,
                     s->pf_tokens_};
}

int DsEngine::cache_slots(int il) const {
    return cache_.owns(il) ? cache_.slots(il) : next_ ? next_->cache_slots(il) : 0;
}

size_t DsEngine::kv_bytes_mapped() const {
    size_t b = 0;
    for (const auto & q : seqs_) {
        for (const auto & e : q->ckv_buf) b += e.mapped();
        for (const auto & e : q->ikc_buf) b += e.mapped();
        for (const auto & r : q->ring) b += r.n * sizeof(half);
    }
    return b + (next_ ? next_->kv_bytes_mapped() : 0);
}

DsEngine::~DsEngine() {
    pf_worker_.reset();
    if (first()) for (float * b : pf_xfer_) if (b) cudaFreeHost(b);
    for (auto & kv : fwd_graphs_) if (kv.second.exec) cudaGraphExecDestroy(kv.second.exec);
    for (auto & q : seqs_) {
        for (auto & k : q->ckpts) if (k.host) cudaFreeHost(k.host);
        for (auto & l : q->commit_graphs) for (auto & r : l) for (auto g : r) if (g) cudaGraphExecDestroy(g);
    }
    if (mail_) cudaFreeHost(mail_);
    if (h_par_) cudaFreeHost(h_par_);
    if (d_par_) cudaFree(d_par_);
    if (h_tok_) cudaFreeHost(h_tok_);
    for (auto p : h_eg_rows_) cudaFreeHost(p);
    if (!first()) for (float * b : xfer_in_) if (b) cudaFreeHost(b);
}

void DsEngine::load(const std::string & path, const EngineOptions & opt) {
    ggml_cpu_init();   // the CPU experts' vec_dot kernels read ggml's fp16 table: without this they return zeros
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
        if (d < 0 || d >= count) throw std::runtime_error("no CUDA device " + std::to_string(d));
    // the layer split: --split, else at the compressed-KV source layer nearest the middle (its consumers follow it)
    std::vector<int> bounds{0};
    if (gpus.size() > 1) {
        if (!opt.split.empty()) {
            if (opt.split.size() != gpus.size() - 1) throw std::runtime_error("--split: one layer per stage after the first");
            for (int b : opt.split) bounds.push_back(b);
        } else {
            GgufModel g;
            g.open(path);
            const DsConfig c = load_ds_config(g);
            for (size_t k = 1; k < gpus.size(); ++k) {
                const int target = (int) ((int64_t) c.n_layer * k / gpus.size());
                int best = target;
                for (int il = 1; il < c.n_layer; ++il)
                    if (c.is_kv_src(il) && std::abs(il - target) < std::abs(best - target)) best = il;
                bounds.push_back(best);
            }
        }
    }
    bounds.push_back(-1);
    DsEngine * s = this, * prev = nullptr;
    for (size_t k = 0; k < gpus.size(); ++k) {
        s->head_ = this;
        s->dev_ = gpus[k];
        s->l0_ = bounds[k];
        s->l1_ = bounds[k + 1];
        s->last_ = k + 1 == gpus.size();
        if (opt.verbose && gpus.size() > 1)
            fprintf(stderr, "bnk: stage %zu of %zu: GPU %d, layers %d-%s\n", k + 1, gpus.size(), s->dev_, s->l0_,
                    s->l1_ < 0 ? "end" : std::to_string(s->l1_ - 1).c_str());
        s->load_stage(path, opt);
        if (prev)
            for (int l = 0; l < kLanes; ++l) prev->xfer_out_[l] = s->xfer_in_[l];
        if (k + 1 < gpus.size()) {
            prev = s;
            s->next_ = std::make_unique<DsEngine>();
            s = s->next_.get();
        }
    }
    // Expert placement (docs/DEEPSEEK.md): every stage's cache is filled from the file first (the ranking and its
    // free VRAM decide the residents), then the host arena takes only the experts no cache holds.
    const Config & c = model_.cfg;
    store_->layout(model_.ev);
    Ranking rank = load_ranking(opt.profile, c.n_layer, c.n_expert);
    if (rank.empty() && !opt.counts_out.empty()) rank = load_ranking(opt.counts_out, c.n_layer, c.n_expert);
    if (opt.verbose) fprintf(stderr, "bnk: expert ranking: %s\n", rank.empty() ? "none (uniform)" : "loaded");
    // Before the caches take the free VRAM: the DSpark drafter's weights and buffers (on the last stage's GPU) and the
    // exchange's settings (its staging is held back in init_cache), so the budgets account for them (a 67K-token
    // prompt ran out of VRAM for its compressed KV when the drafter came after the budget)
    if (!opt.mtp.empty() && opt.mtp != "builtin") {   // the DSpark drafter, on the stage that holds the head and the target layers
        DsEngine * last = this;
        while (last->next_) last = last->next_.get();
        DeviceScope dsl(last->dev_);
        last->spark_ = std::make_unique<DsSpark>();
        last->spark_->load(opt.mtp, last->model_, *cpu_, (int) seqs_.size(), opt.cpu_threads, last->st_, opt.verbose);
        const int nt = last->spark_->n_targets();
        last->tgt_.alloc((size_t) kDsRing * nt * c.n_embd);
        last->tgt_index_.assign(c.n_layer, -1);
        for (int k = 0; k < nt; ++k) {
            const int il = last->spark_->target_layers()[k];
            if (!last->owns(il)) throw std::runtime_error("dspark: target layer " + std::to_string(il) + " is not on the stage with the head");
            last->tgt_index_[il] = k;
        }
        last->spark_->bind(last->tgt_.p, kDsRing, last->rope_win_.p, model_.tok_embd);
    }
    // the adaptive exchange: resident experts follow the routing while serving (docs/DEEPSEEK.md); the store keeps
    // spare blobs for the experts on their way out of VRAM
    const int adapt_every = env_int("BNK_DS_ADAPT", opt.adapt_every), adapt_swaps = env_int("BNK_DS_ADAPT_SWAPS", std::min(opt.adapt_swaps, 8));
    const int spare = adapt_every > 0 && adapt_swaps > 0 ? 2 : 0;
    for (DsEngine * st = this; st; st = st->next_.get()) {
        st->adapt_every_ = spare ? adapt_every : 0;
        st->adapt_swaps_ = spare ? adapt_swaps : 0;
    }
    std::vector<char> skip((size_t) c.n_layer * c.n_expert, 0);
    int resident = 0;
    for (DsEngine * st = this; st; st = st->next_.get()) {
        st->init_cache(rank);
        for (int il = st->l0_; il < st->l1_; ++il)
            for (int e : st->cache_.slot_experts(il)) { skip[(size_t) il * c.n_expert + e] = 1; ++resident; }
    }
    store_->build(model_.ev, std::min(opt.cpu_threads, 16), opt.verbose, &skip, spare);
    if (opt.verbose && spare)
        fprintf(stderr, "bnk: expert cache: adaptive exchange every %d forwards, up to %d in flight a stage (BNK_DS_ADAPT=0 off)\n",
                adapt_every, adapt_swaps);
    // pinned one worker a physical core (the container on all 48 CPUs: decode 19-20 -> 21.8 tok/s, the CPU experts
    // 30-33 -> 26.5 ms a token); BNK_CPU_PIN=0 leaves the placement to the scheduler
    cpu_->init(model_.ev, *store_, opt.cpu_threads, true);
    for (DsEngine * st = this; st; st = st->next_.get()) st->pull_init();
    cpu_->set_chunks(8);
    if (env_int("BNK_DS_LOOKUP", 1) != 0) lookup_drafter_ = std::make_unique<LookupOnlyDrafter>();
    if (pf_max_ > 0 && next_) {   // two group transfer buffers for the whole pipeline (a group in flight on each side)
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaHostAlloc((void **) &pf_xfer_[b], (size_t) pf_group_ * (c.hc * model_.ds.n_embd + model_.ds.hc) * 4,
                                     cudaHostAllocPortable));
            for (DsEngine * st = next_.get(); st; st = st->next_.get()) st->pf_xfer_[b] = pf_xfer_[b];
        }
    }
    // the prompt path's prefetch order: each layer's non-resident experts, hottest first (the ranking, then the rest)
    for (DsEngine * st = this; st; st = st->next_.get()) {
        st->pf_rank_.assign(c.n_layer, {});
        std::vector<char> seen((size_t) c.n_layer * c.n_expert, 0);
        for (const auto & le : rank)
            if (st->owns(le.first) && st->cache_.slot_of(le.first, le.second) < 0) {
                st->pf_rank_[le.first].push_back(le.second);
                seen[(size_t) le.first * c.n_expert + le.second] = 1;
            }
        for (int il = st->l0_; il < st->l1_; ++il)
            for (int e = 0; e < c.n_expert; ++e)
                if (!seen[(size_t) il * c.n_expert + e] && st->cache_.slot_of(il, e) < 0) st->pf_rank_[il].push_back(e);
    }
    if (opt.verbose)
        fprintf(stderr, "bnk: experts: %d of %d resident in VRAM (no host copy), %d in RAM\n", resident, c.n_layer * c.n_expert,
                store_->kept());
    CUDA_CHECK(cudaSetDevice(dev_));
}

void DsEngine::load_stage(const std::string & path, const EngineOptions & opt) {
    opt_ = opt;
    // fewer new tokens than this go through decode windows instead of the prompt path (BNK_PREFILL_MIN)
    opt_.prefill_min = env_int("BNK_PREFILL_MIN", opt_.prefill_min);
    const bool head = head_ == this;
    CUDA_CHECK(cudaSetDevice(dev_));
    if (cudaSetDeviceFlags(cudaDeviceMapHost) != cudaSuccess) cudaGetLastError();
    // the forward's stream at the highest priority: the PCIe lane's pull kernels (st_pull_, the lowest) yield to it
    int prio_lo = 0, prio_hi = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
    CUDA_CHECK(cudaStreamCreateWithPriority(&st_, cudaStreamNonBlocking, prio_hi));
    CUDA_CHECK(cudaStreamCreateWithFlags(&st_commit_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_verified_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_committed_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_out_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&ev_t0_));
    CUDA_CHECK(cudaEventCreate(&ev_t1_));
    model_.load(path, opt.ple_gguf, opt.verbose, l0_, l1_, first(), last_);
    if (l1_ < 0) l1_ = model_.ds.n_layer;
    const DsConfig & c = model_.ds;
    // the GPU kernels' SwiGLU clamp is a __device__ symbol: set on every stage's device (it was set on the head's only,
    // so the second GPU's shared experts and resident experts ran unclamped: found by the DSpark drafter's reference)
    moe_set_swiglu_limit(c.swiglu_limit);
    moe_tc_set_swiglu_limit(c.swiglu_limit);
    quantize_set_swiglu_limit(c.swiglu_limit);
    if (head) {
        layer_routed.assign(c.n_layer, 0);
        layer_misses.assign(c.n_layer, 0);
        cpu_experts_set_swiglu_limit(c.swiglu_limit);
        if (kMaxWindow > 8) throw std::runtime_error("kDsRing assumes decode windows of at most 8 rows");
    } else {
        store_ = &head_->own_store_;
        cpu_ = &head_->own_cpu_;
    }
    // rotary tables
    {
        const auto w = ds_rope_inv_freq(c.n_rot, c.rope_base, 1.f, 0, c.yarn_beta_fast, c.yarn_beta_slow);
        const auto cm = ds_rope_inv_freq(c.n_rot, c.comp_rope_base, c.yarn_factor, c.yarn_orig, c.yarn_beta_fast, c.yarn_beta_slow);
        rope_win_.alloc(w.size());
        rope_comp_.alloc(cm.size());
        CUDA_CHECK(cudaMemcpy(rope_win_.p, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(rope_comp_.p, cm.data(), cm.size() * 4, cudaMemcpyHostToDevice));
    }
    const int W = kMaxWindow, E = c.n_embd, HC = c.hc_dim(), H = c.n_head, D = c.head_dim;
    x_.alloc(W * E); res_.alloc(W * HC); mixed_.alloc(W * E); xn_.alloc(W * E); out_.alloc(W * E);
    pre_a_.alloc(W * c.hc); pre_b_.alloc(W * c.hc); attn_pre_.alloc(W * c.hc); post_.alloc(W * c.hc); comb_.alloc(W * c.hc * c.hc);
    mix_scratch_.alloc(kDsMixScratch);   // (its counters must start at zero)
    CUDA_CHECK(cudaMemset(mix_scratch_.p, 0, kDsMixScratch * sizeof(float)));
    qr_.alloc(W * c.q_lora); qrn_.alloc(W * c.q_lora); q_.alloc((size_t) W * H * D); kvr_.alloc(W * D);
    attn_o_.alloc((size_t) W * H * D); oa_.alloc((size_t) W * c.o_groups * c.o_lora);
    int max_ratio = 1;
    for (int il = 0; il < c.n_layer; ++il) max_ratio = std::max(max_ratio, c.ratio[il]);
    lds_ = (int64_t) opt.max_ctx + kMaxWindow;   // compressed rows of the longest context at ratio 1
    n_blocks_max_ = c.cand_bs > 0 ? (int) ((lds_ + c.cand_bs - 1) / c.cand_bs) : 0;
    iq_.alloc((size_t) W * c.idx_heads * c.idx_dim); iw_.alloc(W * c.idx_heads);
    scores_.alloc((size_t) W * lds_);
    idx_.alloc((size_t) W * c.idx_topk);
    if (n_blocks_max_) { cand_scratch_.alloc((size_t) W * n_blocks_max_); cand_.alloc((size_t) W * n_blocks_max_); }
    if (!c.engram_layers.empty()) {
        eg_emb_.alloc((size_t) W * c.eg_cols() * c.eg_dim);
        eg_kv_.alloc((size_t) W * (c.hc + 1) * E);
    }
    rlogits_.alloc(W * c.n_expert); rw_.alloc(W * c.n_used); rids_.alloc(W * c.n_used);
    sg_.alloc(W * c.n_ff); su_.alloc(W * c.n_ff); shared_out_.alloc(W * E); moe_out_.alloc(W * E);
    ones_.alloc(W);
    {
        std::vector<float> one(W, 1.f);
        CUDA_CHECK(cudaMemcpy(ones_.p, one.data(), W * 4, cudaMemcpyHostToDevice));
    }
    if (last_) logits_.alloc((size_t) W * c.n_vocab);
    argmax_dev_.alloc(W);
    const int64_t maxcols = std::max<int64_t>({(int64_t) E, (int64_t) c.q_lora, (int64_t) H * D, (int64_t) c.o_groups * c.o_lora,
                                               (int64_t) c.n_ff, (int64_t) c.eg_cols() * c.eg_dim});
    actq_.alloc(W * maxcols); actd_.alloc(W * maxcols / 32); acts_.alloc(W * maxcols / 16); acth_.alloc(W * maxcols);
    act_.q = actq_; act_.d = actd_; act_.s = acts_; act_.h = acth_;
    actq2_.alloc(W * maxcols); actd2_.alloc(W * maxcols / 32); acts2_.alloc(W * maxcols / 16); acth2_.alloc(W * maxcols);
    act2_.q = actq2_; act2_.d = actd2_; act2_.s = acts2_; act2_.h = acth2_;
    for (int l = 0; l < kLanes; ++l) {
        ckv_raw_[l].resize(c.n_layer);
        sc_raw_[l].resize(c.n_layer);
        for (int il = l0_; il < l1_; ++il)
            if (c.is_kv_src(il)) {
                ckv_raw_[l][il].alloc((size_t) W * D);
                if (c.ratio[il] > 1) sc_raw_[l][il].alloc((size_t) W * D);
            }
        if (first() && last_) break;   // one lane without a pipeline
    }
    // MoE scratch and mailboxes
    const int Fx = c.n_ff;
    hits_buf_.alloc(sizeof(HitList));
    gu_buf_.alloc((size_t) kMaxRouted * W * 2 * Fx);
    hq_buf_.alloc((size_t) kMaxRouted * W * Fx);
    hd_buf_.alloc((size_t) kMaxRouted * W * Fx / 32);
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
    if (!first())
        for (float *& b : xfer_in_) CUDA_CHECK(cudaHostAlloc((void **) &b, xfer_floats() * 4, cudaHostAllocPortable));
    h_eg_rows_.assign(c.engram_layers.size(), nullptr);
    for (size_t e = 0; e < c.engram_layers.size(); ++e)
        if (owns(c.engram_layers[e]))
            CUDA_CHECK(cudaHostAlloc((void **) &h_eg_rows_[e], (size_t) W * c.eg_cols() * model_.engram_tab[e]->row_bytes(),
                                     cudaHostAllocMapped));
    // the prompt path's buffers (before the budget is measured: they are not elastic)
    pf_max_ = std::max(0, std::min(opt.prefill_chunk, opt.prefill_chunk_max));
    if (pf_max_ > 0) {   // BNK_DS_PF_CHUNKS chunks run layer-major as one group (its MoE buffers grow with it)
        const int G = std::max(1, std::min(64, env_int("BNK_DS_PF_CHUNKS", 4)));
        pf_group_ = pf_max_ * G;
        prefill_alloc(pf_max_, G);
    }
    // the VRAM budget: what is free now less the reserve; the context's compressed caches draw on it as a
    // conversation grows, so the full context of every slot is kept out of the expert cache's share (a resident
    // expert has no host copy: the cache never shrinks)
    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    const size_t reserve = (size_t) (opt.vram_reserve_gib * 1073741824.0);
    budget_.set_limit(free_b > reserve ? free_b - reserve : 0);
    budget_.set_reserve(reserve);
    if (pf_max_ > 0) {   // the expert staging arena: elastic, charged to the budget before the cache takes the rest
        const DsConfig & cc = model_.ds;
        size_t worst = 0;
        for (int il = l0_; il < l1_; ++il) {
            const LayerWeights & Lw = model_.ev.layers[il];
            worst = std::max(worst, (size_t) cc.n_expert * ((Lw.gate_bytes + Lw.up_bytes + Lw.down_bytes + 63) / 64 * 64));
        }
        // a whole layer's outside experts would be ~6 GiB per buffer here: 1 GiB each by default (BNK_DS_STAGE_MIB),
        // the prompt path's cost model sends the rest to the CPU
        const size_t small = std::min(worst, vmem_round((size_t) env_int("BNK_DS_STAGE_MIB", 1024) << 20));
        for (int i = 0; i < 2; ++i) {
            pf_.stage_b[i].reserve(worst, &budget_, "expert staging", false);
            pf_.stage_b[i].ensure(small);
            pf_.stage[i] = pf_.stage_b[i].as<uint8_t>();
        }
        pf_.stage_bytes = small;
    }
    seqs_.clear();
    for (int i = 0; i < std::max(1, opt.slots); ++i) {
        seqs_.push_back(std::make_unique<Seq>());
        alloc_seq(*seqs_.back());
    }
    cur_ = seqs_[0].get();
    cur_->used = true;
    ensure_ctx_here(std::min(opt.max_ctx, kCtxBaseline));
    reset_here();
    const size_t ck = state_floats() * 4;
    for (auto & q : seqs_)
        while ((int) q->ckpts.size() < kMaxCheckpoints) {
            q->ckpts.emplace_back();
            CUDA_CHECK(cudaHostAlloc((void **) &q->ckpts.back().host, std::max<size_t>(ck, 64), cudaHostAllocPortable));
        }
    if (opt.verbose) {
        size_t ctx_full = 0;
        for (int il = l0_; il < l1_; ++il)
            if (c.is_kv_src(il)) ctx_full += (size_t) (opt.max_ctx / c.ratio[il] + kMaxWindow) * (D + c.idx_dim) * sizeof(half);
        fprintf(stderr, "bnk: layers %d-%d: context %d: %.1f MiB of compressed KV per slot when full (mapped as used), "
                        "%.1f MiB of window rings, %d CPU expert threads%s\n",
                l0_, l1_ - 1, opt.max_ctx, ctx_full / 1048576.0, (double) (l1_ - l0_) * kDsRing * D * sizeof(half) / 1048576.0,
                opt.cpu_threads, seqs_.size() > 1 ? (", " + std::to_string(seqs_.size()) + " conversation slots").c_str() : "");
    }
}

// This stage's expert cache: everything the budget has left after the context's full extent for every slot and
// a reserve for the graphs. Filled straight from the file (the store is not built yet).
void DsEngine::init_cache(const Ranking & rank) {
    DeviceScope ds(dev_);
    const DsConfig & c = model_.ds;
    size_t ctx_full = 0;
    for (int il = l0_; il < l1_; ++il)
        if (c.is_kv_src(il)) ctx_full += vmem_round((size_t) (opt_.max_ctx / c.ratio[il] + kMaxWindow) * c.head_dim * sizeof(half)) +
                                         vmem_round((size_t) (opt_.max_ctx / c.ratio[il] + kMaxWindow) * c.idx_dim * sizeof(half));
    ctx_full *= seqs_.size();
    size_t mapped = 0;
    for (auto & q : seqs_) {
        for (auto & e : q->ckv_buf) mapped += e.mapped();
        for (auto & e : q->ikc_buf) mapped += e.mapped();
    }
    const size_t graphs = (size_t) kMaxFwdGraphs * kGraphBytesEst + (size_t) seqs_.size() * kMaxWindow * kMaxWindow * (4ull << 20);
    // the PCIe lane's staging slots (pull_init, after the cache): P slots a layer plus P blobs in flight
    size_t lane = 0;
    if (const int P = std::min(8, env_int("BNK_DS_PULL", 0)); P > 0 && l1_ - l0_ >= 2) {
        size_t mb = 0;
        for (int il = l0_; il < l1_; ++il) {
            lane += (size_t) P * ExpertCache::slot_bytes(model_.ev, il);
            mb = std::max(mb, store_->blob_bytes(il));
        }
        lane += (size_t) P * mb;
    }
    // the exchange's swap staging (ExpertCache::adapt_init allocates it outside the budget at the first exchange)
    const size_t xchg = adapt_every_ > 0 ? (size_t) std::max(1, adapt_swaps_) * store_->max_blob_bytes() + (64u << 20) : 0;
    const size_t hold = (ctx_full > mapped ? ctx_full - mapped : 0) + graphs + lane + xchg;
    budget_.sync();   // the limit from what the GPU really has free now (the drafter's weights were allocated outside the budget)
    const size_t avail = budget_.free();
    size_t budget = avail > hold ? avail - hold : 0;
    double cap = opt_.expert_cache_gib;
    if (!opt_.stage_cache_gib.empty()) {
        int my = 0;
        for (const DsEngine * s = head_; s && s != this; s = s->next_.get()) ++my;
        if (my < (int) opt_.stage_cache_gib.size()) cap = opt_.stage_cache_gib[my];
    }
    if (cap >= 0) budget = std::min(budget, (size_t) (cap * 1073741824.0));
    if (opt_.verbose)
        fprintf(stderr, "bnk: layers %d-%d: expert cache budget %.2f GiB (%.2f free, %.2f held for the context, graphs, the exchange's staging and the PCIe lane)\n",
                l0_, l1_ - 1, budget / 1073741824.0, avail / 1073741824.0, hold / 1073741824.0);
    cache_.init(model_.ev, *store_, budget, rank, st_, opt_.verbose, &budget_, l0_, l1_, nullptr);
}

// ------------------------------------------------------------------------------------------------ slots
void DsEngine::alloc_seq(Seq & q) {
    const DsConfig & c = model_.ds;
    q.ring.resize(c.n_layer);
    q.ckv_buf = std::vector<ElasticBuf>(c.n_layer);
    q.ikc_buf = std::vector<ElasticBuf>(c.n_layer);
    q.ckc.resize(c.n_layer);
    q.ikc.resize(c.n_layer);
    q.st_kv.resize(c.n_layer);
    q.st_sc.resize(c.n_layer);
    for (int il = l0_; il < l1_; ++il) {
        q.ring[il].alloc((size_t) kDsRing * c.head_dim);
        if (c.is_kv_src(il)) {
            const size_t rows = (size_t) opt_.max_ctx / c.ratio[il] + kMaxWindow;
            q.ckv_buf[il].reserve(rows * c.head_dim * sizeof(half), &budget_, "compressed KV");
            q.ikc_buf[il].reserve(rows * c.idx_dim * sizeof(half), &budget_, "index keys");
            q.ckc[il].view(q.ckv_buf[il].as<half>(), rows * c.head_dim);
            q.ikc[il].view(q.ikc_buf[il].as<half>(), rows * c.idx_dim);
            if (c.ratio[il] > 1) {
                q.st_kv[il].alloc((size_t) (c.ratio[il] - 1) * c.head_dim);
                q.st_sc[il].alloc((size_t) (c.ratio[il] - 1) * c.head_dim);
            }
        }
    }
}

size_t DsEngine::state_floats() {
    size_t n = 0;
    each_state([&](float *, size_t k) { n += k; });
    return n;
}

void DsEngine::use(Seq * q) { cur_ = q; }

int DsEngine::current_slot() const {
    for (size_t i = 0; i < seqs_.size(); ++i) if (seqs_[i].get() == cur_) return (int) i;
    return 0;
}

void DsEngine::select(int slot) {
    for (DsEngine * s = this; s; s = s->next_.get()) s->select_here(slot);
}

void DsEngine::select_here(int slot) {
    DeviceScope ds(dev_);
    if (slot < 0 || slot >= (int) seqs_.size()) throw std::runtime_error("select: no such slot");
    join_commit_here();
    use(seqs_[slot].get());
    if (spark_) spark_->select(slot);
    if (!cur_->used) { cur_->used = true; reset_here(); }
}

int DsEngine::ctx_bucket(int cells) const {
    return std::min(opt_.max_ctx, std::max(kCtxBaseline, (cells + kCtxStep - 1) / kCtxStep * kCtxStep));
}

void DsEngine::ensure_ctx(int cells) {
    for (DsEngine * s = this; s; s = s->next_.get()) s->ensure_ctx_here(cells);
}

void DsEngine::ensure_ctx_here(int cells) {
    DeviceScope ds(dev_);
    if (cells > opt_.max_ctx) throw std::runtime_error("context full");
    const int want = ctx_bucket(cells);
    if (want > cur_->ctx_mapped) map_ctx(want);
}

void DsEngine::map_ctx(int cells) {
    const DsConfig & c = model_.ds;
    const bool grow = cells > cur_->ctx_mapped;
    for (int il = l0_; il < l1_; ++il) {
        if (!c.is_kv_src(il)) continue;
        const size_t rows = (size_t) cells / c.ratio[il] + kMaxWindow;
        if (grow) {
            cur_->ckv_buf[il].ensure(rows * c.head_dim * sizeof(half));
            cur_->ikc_buf[il].ensure(rows * c.idx_dim * sizeof(half));
        } else {
            cur_->ckv_buf[il].shrink_to(rows * c.head_dim * sizeof(half));
            cur_->ikc_buf[il].shrink_to(rows * c.idx_dim * sizeof(half));
        }
    }
    cur_->ctx_mapped = cells;
}

void DsEngine::reset() {
    for (DsEngine * s = this; s; s = s->next_.get()) s->reset_here();
}

void DsEngine::reset_here() {
    DeviceScope ds(dev_);
    join_commit_here();
    cur_->history.clear();
    cur_->pending_T = 0;
    for (auto & k : cur_->ckpts) k.pos = -1;
    each_state([&](float * d, size_t n) { CUDA_CHECK(cudaMemsetAsync(d, 0, n * 4, st_)); });
    CUDA_CHECK(cudaStreamSynchronize(st_));
    if (cur_->ctx_mapped > kCtxBaseline) map_ctx(std::min(opt_.max_ctx, kCtxBaseline));
    if (spark_) spark_->restart(0);
}

void DsEngine::checkpoint() {
    for (DsEngine * s = this; s; s = s->next_.get()) s->checkpoint_here();
}

void DsEngine::checkpoint_here() {
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
    slot->age = ++ckpt_age_;
}

int DsEngine::rollback(int max_pos) {
    const int r = rollback_here(max_pos);
    for (DsEngine * s = next_.get(); s; s = s->next_.get())
        if (s->rollback_here(max_pos) != r) throw std::runtime_error("rollback: the pipeline stages disagree");
    return r;
}

int DsEngine::rollback_here(int max_pos) {
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
    cur_->pending_T = 0;
    best->age = ++ckpt_age_;
    for (auto & k : cur_->ckpts)
        if (k.pos > best->pos) k.pos = -1;
    if (spark_) spark_->restart(best->pos);
    return best->pos;
}

int DsEngine::reusable(const std::vector<int32_t> & prompt) const {
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

void DsEngine::save_counts() {
    for (DsEngine * s = this; s; s = s->next_.get()) s->save_counts_here();
}

void DsEngine::save_counts_here() {
    if (opt_.counts_out.empty()) return;
    DeviceScope ds(dev_);
    const Config & c = model_.cfg;
    std::vector<uint32_t> now((size_t) c.n_layer * c.n_expert);
    CUDA_CHECK(cudaMemcpy(now.data(), counts_.p, now.size() * 4, cudaMemcpyDeviceToHost));
    if (saved_counts_.size() != now.size()) saved_counts_.assign(now.size(), 0);
    std::vector<uint32_t> delta(now.size());
    for (size_t i = 0; i < now.size(); ++i) delta[i] = now[i] - saved_counts_[i];
    saved_counts_ = now;
    merge_counts(delta, load_counts(opt_.counts_out, c.n_layer, c.n_expert));
    bnk::save_counts(opt_.counts_out, delta, c.n_layer, c.n_expert);
}

// ------------------------------------------------------------------------------------------------ Engram
// the hash ids of the n-grams ending at position p of `hist` (engram.py NgramHashState): rows[col]
void DsEngine::engram_hash(int e, const std::vector<int32_t> & hist, int p, uint64_t * rows) const {
    const DsConfig & c = model_.ds;
    const int64_t pad = c.eg_token_map[c.eg_pad];
    auto cid = [&](int32_t t) -> int64_t { return t < 0 || t >= (int32_t) c.eg_token_map.size() ? -1 : c.eg_token_map[t]; };
    uint64_t toks[16];
    bool blocked = false;
    for (int s = 0; s < c.eg_ngram; ++s) {
        const int q = p - s;
        const int64_t v = q < 0 ? -1 : cid(hist[q]);
        blocked = blocked || v < 0;
        toks[s] = (uint64_t) (blocked ? pad : v);
    }
    uint64_t rolling = toks[0] * c.eg_mult[e][0];
    for (int o = 1; o < c.eg_ngram; ++o) {
        rolling ^= toks[o] * c.eg_mult[e][o];
        for (int h = 0; h < c.eg_heads; ++h) {
            const int col = (o - 1) * c.eg_heads + h;
            rows[col] = rolling % c.eg_primes[e][col] + c.eg_offsets[e][col];
        }
    }
}

// the window's table rows into the mapped host buffers (before the graph runs). The rows sit in the memory-mapped
// table on the NVMe: a row not in the page cache costs a ~100 us fault, so the rows are read on several threads at
// once (48 rows a token: one thread took ~5 ms a token on a cold table).
void DsEngine::engram_gather() {
    const DsConfig & c = model_.ds;
    const double t0 = now_ms();
    const int cols = c.eg_cols();
    struct Job { int e, t, p; const std::vector<int32_t> * hist; };
    std::vector<Job> jobs;
    for (size_t e = 0; e < c.engram_layers.size(); ++e) {
        if (!h_eg_rows_[e]) continue;
        for (const Win & w : wins_)
            for (int tw = 0; tw < w.T; ++tw) jobs.push_back({(int) e, w.r0 + tw, w.pos0 + tw, &w.q->history});
    }
    if (jobs.empty()) return;
    // The tables (103 GB) are memory-mapped from the NVMe and the page cache cannot hold them beside the pinned expert
    // store, so a token whose rows are cold pays a page fault per row, one after another (8 threads: 50 ms a token
    // measured, against 2 ms warm). First ask the kernel for every row's pages at once (MADV_WILLNEED starts the reads
    // and returns), so the copies below find them landing together: one NVMe round trip instead of a chain of them.
    // BNK_DS_EG_WILLNEED=0 turns it off.
    static const bool willneed = env_int("BNK_DS_EG_WILLNEED", 1) != 0;
    if (willneed) {
        const long page = sysconf(_SC_PAGESIZE);
        uint64_t rows[64];
        for (const Job & jb : jobs) {
            const TensorRef & tab = *model_.engram_tab[jb.e];
            const size_t rb = tab.row_bytes();
            engram_hash(jb.e, *jb.hist, jb.p, rows);
            for (int col = 0; col < cols; ++col) {
                const uintptr_t a = (uintptr_t) (tab.data + rows[col] * rb), a0 = a & ~(uintptr_t) (page - 1);
                madvise((void *) a0, (size_t) (a - a0) + rb, MADV_WILLNEED);
            }
        }
    }
    auto run = [&](size_t j0, size_t j1, int c0, int c1) {   // jobs [j0, j1), columns [c0, c1)
        uint64_t rows[64];
        for (size_t j = j0; j < j1; ++j) {
            const Job & jb = jobs[j];
            const TensorRef & tab = *model_.engram_tab[jb.e];
            const size_t rb = tab.row_bytes();
            engram_hash(jb.e, *jb.hist, jb.p, rows);
            for (int col = c0; col < c1; ++col)
                memcpy(h_eg_rows_[jb.e] + ((size_t) jb.t * cols + col) * rb, tab.data + rows[col] * rb, rb);
        }
    };
    // on the CPU pool's threads (spinning, ready): BNK_DS_EG_THREADS_DECODE=1 runs it here, N > 1 the old way (N
    // threads spawned per token, which measured 2 ms or 50 ms a token depending on where the scheduler put them)
    static const int nth = std::max(0, std::min(64, env_int("BNK_DS_EG_THREADS_DECODE", 0)));
    if (nth == 1 || jobs.size() * cols < 16) run(0, jobs.size(), 0, cols);
    else if (nth == 0) {
        const int total = (int) jobs.size() * cols;   // (job, column) units split over the pool
        cpu_->parallel([&](int part, int nparts) {
            const int u0 = (int) ((int64_t) total * part / nparts), u1 = (int) ((int64_t) total * (part + 1) / nparts);
            for (int u = u0; u < u1;) {
                const int j = u / cols, c0 = u % cols, c1 = std::min(cols, c0 + (u1 - u));
                run(j, j + 1, c0, c1);
                u += c1 - c0;
            }
        });
    } else {
        std::vector<std::thread> th;
        for (int i = 0; i < nth; ++i) th.emplace_back(run, 0, jobs.size(), cols * i / nth, cols * (i + 1) / nth);
        for (auto & t : th) t.join();
    }
    head_->times.ple_ms += now_ms() - t0;
}

void DsEngine::engram(int il, int T) {
    const DsConfig & c = model_.ds;
    const DsLayer & L = model_.layers[il];
    const int e = c.engram_index(il), E = c.n_embd;
    const TensorRef & tab = *model_.engram_tab[e];
    const int64_t rows = (int64_t) T * c.eg_cols(), D = (int64_t) c.eg_cols() * c.eg_dim;
    QMat rm{h_eg_rows_[e], (int) tab.type, rows, tab.ne[0], tab.row_bytes()};
    dequant_rows(rm, 0, rows, eg_emb_, st_);
    quantize_act(eg_emb_, D, T, D, act2_, st_);
    gemv(L.eg_wkv, &act2_, nullptr, 0, T, eg_kv_, (int64_t) (c.hc + 1) * E, false, st_);
    ds_engram_gate(res_, eg_kv_, (int64_t) (c.hc + 1) * E, eg_kv_.p + (size_t) c.hc * E, (int64_t) (c.hc + 1) * E, F(L.eg_gate),
                   T, c.hc, E, c.eps, st_);
}

// ------------------------------------------------------------------------------------------------ the forward
void DsEngine::attn(int il, int T) {
    const DsConfig & c = model_.ds;
    const DsLayer & L = model_.layers[il];
    const int E = c.n_embd, H = c.n_head, D = c.head_dim, QL = c.q_lora, R = c.n_rot, IH = c.idx_heads, ID = c.idx_dim;
    const int ratio = c.ratio[il];
    const bool src = c.is_kv_src(il), isrc = c.is_idx_src(il);
    gemv(L.q_a, &act_, nullptr, 0, T, qr_, QL, false, st_);
    rmsnorm_rows(qr_, F(L.q_a_norm), qrn_, T, QL, QL, QL, c.eps, st_);
    quantize_act(qrn_, QL, T, QL, act2_, st_);
    gemv(L.q_b, &act2_, nullptr, 0, T, q_, (int64_t) H * D, false, st_);
    gemv(L.kv, &act_, nullptr, 0, T, kvr_, D, false, st_);
    if (src) {
        gemv(L.comp_kv, nullptr, xn_, E, T, ckv_raw_[lane_][il], D, false, st_);
        if (ratio > 1) gemv(L.comp_gate, nullptr, xn_, E, T, sc_raw_[lane_][il], D, false, st_);
    }
    if (isrc) {
        gemv(L.idx_q_b, nullptr, qrn_, QL, T, iq_, (int64_t) IH * ID, false, st_);
        gemv(L.idx_proj, nullptr, xn_, E, T, iw_, IH, false, st_);
    }
    const float scale = 1.f / sqrtf((float) D);
    trace(st_, il, "qr_norm", qrn_, T, QL);
    trace(st_, il, "q", q_, T, (int64_t) H * D);
    trace(st_, il, "kv_raw", kvr_, T, D);
    for (size_t wi = 0; wi < wins_.size(); ++wi) {
        const Win & w = wins_[wi];
        const int * pos0 = &d_par_[wi].pos0;
        const size_t r = (size_t) w.r0;
        ds_rope_kv_prep(q_.p + r * H * D, H, kvr_.p + r * D, F(L.kv_norm), w.q->ring[il], w.T, D, R, pos0, rope_of(il), c.eps, st_);
        if (src) {
            const float * ckv = ckv_raw_[lane_][il].p + r * D, * sc = ratio > 1 ? sc_raw_[lane_][il].p + r * D : nullptr;
            ds_compress(ckv, sc, w.q->st_kv[il], w.q->st_sc[il], F(L.comp_norm), (const half *) L.idx_k.data, F(L.idx_k_norm),
                        w.q->ckc[il], w.q->ikc[il], w.T, ratio, D, ID, R, pos0, rope_comp_, c.eps, st_);
            if (w.commit) ds_compress_commit(ckv, sc, w.q->st_kv[il], w.q->st_sc[il], w.T, ratio, D, pos0, st_);
        }
        if (ratio) {
            const int ks = c.kv_src[il];
            if (isrc) {
                ds_rope_heads(iq_.p + r * IH * ID, w.T, IH, ID, R, pos0, rope_comp_, st_);
                const int n_max = (int) std::min<int64_t>(lds_, ctx_grid_ / ratio + 1);
                ds_index_scores(iq_.p + r * IH * ID, w.q->ikc[ks], iw_.p + r * IH, scores_.p + r * lds_, w.T, IH, ID, ratio, n_max,
                                lds_, pos0, st_);
                if (il == c.cand_src)
                    ds_cand_blocks(scores_.p + r * lds_, lds_, w.T, ratio, c.cand_bs, c.cand_blocks, (n_max + c.cand_bs - 1) / c.cand_bs,
                                   pos0, cand_scratch_.p + r * n_blocks_max_, cand_.p + r * n_blocks_max_, st_);
                const uint8_t * cand = c.cand_src >= 0 && il > c.cand_src ? cand_.p + r * n_blocks_max_ : nullptr;
                ds_topk_select(scores_.p + r * lds_, lds_, w.T, ratio, c.idx_topk, pos0, cand, c.cand_bs, idx_.p + r * c.idx_topk, st_);
            }
            ds_attention(q_.p + r * H * D, w.q->ring[il], w.q->ckc[ks], idx_.p + r * c.idx_topk, c.idx_topk, F(L.sinks),
                         attn_o_.p + r * H * D, w.T, H, D, c.window, R, pos0, rope_of(il), scale, st_);
        } else {
            ds_attention(q_.p + r * H * D, w.q->ring[il], nullptr, nullptr, 0, F(L.sinks), attn_o_.p + r * H * D, w.T, H, D,
                         c.window, R, pos0, rope_of(il), scale, st_);
        }
    }
    trace(st_, il, "attn_o", attn_o_, T, (int64_t) H * D);
    // the grouped low-rank output projection: group g takes heads [g * H/G, (g+1) * H/G)
    // (one quantization of the whole row, one block-diagonal GEMV: 8 x 2 launches were 16)
    const int G = c.o_groups, GW = H * D / G, OL = c.o_lora;
    quantize_act(attn_o_, (int64_t) H * D, T, (int64_t) H * D, act2_, st_);
    if (L.o_a.layout == 1 && OL % 16 == 0) {
        ActQ8 av = act2_;
        av.cols = GW;
        GemvSplit sp;
        sp.grp_rows = OL;
        sp.grp_cols = GW;
        gemv(L.o_a, &av, nullptr, 0, T, oa_, (int64_t) G * OL, false, st_, sp);
    } else
        for (int g = 0; g < G; ++g) {
            quantize_act(attn_o_.p + (size_t) g * GW, (int64_t) H * D, T, GW, act2_, st_);
            gemv(qmat_rows(L.o_a, (int64_t) g * OL, OL), &act2_, nullptr, 0, T, oa_.p + (size_t) g * OL, (int64_t) G * OL, false, st_);
        }
    quantize_act(oa_, (int64_t) G * OL, T, (int64_t) G * OL, act2_, st_);
    gemv(L.o_b, &act2_, nullptr, 0, T, out_, E, false, st_);
    trace(st_, il, "oa", oa_, T, (int64_t) G * OL);
    trace(st_, il, "attn_out", out_, T, E);
}

void DsEngine::moe(int il, int T) {
    const DsConfig & c = model_.ds;
    const DsLayer & L = model_.layers[il];
    const int E = c.n_embd, k = c.n_used, Fx = c.n_ff;
    const MoeLayerDesc d = cache_.desc(il);
    gemv(L.router, nullptr, xn_, E, T, rlogits_, c.n_expert, false, st_);
    if (T <= 8 && c.n_expert <= 512)
        moe_route_plan_ds(rlogits_, F(L.probs_b), c.n_expert, k, c.route_scale, c.route_norm, rids_, rw_, T, d, moes_, msg(il), xn_, E,
                          &d_par_->seq, counts_.p + (size_t) il * c.n_expert, st_);
    else {
        ds_route(rlogits_, F(L.probs_b), T, c.n_expert, k, c.route_scale, c.route_norm, rids_, rw_, st_);
        moe_plan(rids_, rw_, T, k, d, moes_, msg(il), xn_, E, &d_par_->seq, counts_.p + (size_t) il * c.n_expert, st_);
    }
    if (pull_P_) pull_launch(il);   // (after the plan's mailbox writes: see pull_launch)
    if (!opt_.use_graphs || head_->dump_all) {   // eager: answer the CPU part right here
        CUDA_CHECK(cudaStreamSynchronize(st_));
        MoeMsg * m = msg(il);
        tasks_.clear();
        for (int i = 0; i < m->n_miss; ++i) {
            if (!store_->blob(il, m->miss_e[i])) throw std::runtime_error("expert without a host copy routed to the CPU");
            tasks_.push_back({m->miss_t[i], 0, m->miss_e[i], m->miss_w[i]});
        }
        if (!tasks_.empty()) cpu_->run(il, T, m->x(), tasks_, m->out(E));
        // BNK_DS_TRACE_DUMP=prefix: layer 0's mailbox (the CPU's input rows and its answer) of the first forward
        static const char * td = getenv("BNK_DS_TRACE_DUMP");
        static bool dumped = false;
        if (td && il == 0 && !dumped) {
            dumped = true;
            FILE * fx = fopen((std::string(td) + ".x").c_str(), "wb"), * fo = fopen((std::string(td) + ".out").c_str(), "wb"),
                 * ft = fopen((std::string(td) + ".tasks").c_str(), "w");
            fwrite(m->x(), 4, (size_t) T * E, fx); fwrite(m->out(E), 4, (size_t) T * E, fo);
            for (const auto & tk : tasks_) fprintf(ft, "%d %d %.8g\n", tk.t, tk.expert, tk.w);
            fclose(fx); fclose(fo); fclose(ft);
            std::vector<float> xq_((size_t) T * E);
            CUDA_CHECK(cudaMemcpy(xq_.data(), xn_.p, xq_.size() * 4, cudaMemcpyDeviceToHost));
            double dd = 0; for (size_t i = 0; i < xq_.size(); ++i) dd += (xq_[i] - m->x()[i]) * (xq_[i] - m->x()[i]);
            fprintf(stderr, "trace dump: mailbox x vs xn_ rms diff %.3e; out row0 rms %.4g\n", sqrt(dd / xq_.size()),
                    sqrt([&]{ double s = 0; for (int i = 0; i < E; ++i) s += (double) m->out(E)[i] * m->out(E)[i]; return s / E; }()));
        }
        head_->times.misses += m->n_miss;
        head_->times.routed += (int64_t) T * k;
        head_->layer_misses[il] += m->n_miss;
        head_->layer_routed[il] += (int64_t) T * k;
        __atomic_thread_fence(__ATOMIC_RELEASE);
        m->seq_done = m->seq_req;
    }
    // the shared expert (gate | up, the clamped SwiGLU inside the quantization of down's input)
    if (L.sh_gu.valid()) gemv(L.sh_gu, &act_, nullptr, 0, T, sg_, Fx, false, st_, GemvSplit{Fx, su_, Fx});
    else {
        gemv(L.sh_gate, &act_, nullptr, 0, T, sg_, Fx, false, st_);
        gemv(L.sh_up, &act_, nullptr, 0, T, su_, Fx, false, st_);
    }
    quantize_act_silu_mul(sg_, su_, Fx, T, Fx, act2_, st_);
    gemv(L.sh_down, &act2_, nullptr, 0, T, shared_out_, E, false, st_);
    moe_hits(d, moes_, act_, T, k, E, Fx, st_);   // (a pulled expert sits in a spare slot of the same region)
    moe_reduce(moes_, msg(il), shared_out_, ones_, moe_out_, T, k, E, st_, &d_par_->seq, false);
    if (pull_P_ && il > l0_)
        ds_pull_restore(pull_staged_.p + (size_t) il * pull_P_, pull_P_, const_cast<int32_t *>(d.slot_of), cache_.spare_slot(il, 0),
                        moes_.hits, pull_stats_.p, st_, pull_state_.p + il);
    trace(st_, il, "rlogits", rlogits_, T, c.n_expert);
    trace(st_, il, "rw", rw_, T, k);
    trace(st_, il, "shared_out", shared_out_, T, E);
    trace(st_, il, "moe_out", moe_out_, T, E);
    if (trace_layer(il)) {
        std::vector<int32_t> ids((size_t) T * k);
        CUDA_CHECK(cudaMemcpy(ids.data(), rids_.p, ids.size() * 4, cudaMemcpyDeviceToHost));
        fprintf(stderr, "trace L%d experts:", il);
        for (int t = 0; t < T; ++t) { fprintf(stderr, " ["); for (int j = 0; j < k; ++j) fprintf(stderr, "%d ", ids[t * k + j]); fprintf(stderr, "]"); }
        fprintf(stderr, " misses %d\n", msg(il)->n_miss);
    }
}

void DsEngine::head(int T) {
    const DsConfig & c = model_.ds;
    ds_hc_pre_norm_q(res_, pre_, F(model_.output_norm), mixed_, xn_, T, c.hc, c.n_embd, c.eps, act_, st_);
    gemv(model_.output, &act_, nullptr, 0, T, logits_, c.n_vocab, false, st_);
}

void DsEngine::layer_forward(int il, int T) {
    const DsConfig & c = model_.ds;
    const DsLayer & L = model_.layers[il];
    const int E = c.n_embd;
    if (c.engram_index(il) >= 0) engram(il, T);
    if (!tgt_index_.empty() && tgt_index_[il] >= 0)   // the DSpark drafter reads this layer's attention input, by position
        for (size_t wi = 0; wi < wins_.size(); ++wi)
            ds_hc_mean_ring(res_.p + (size_t) wins_[wi].r0 * c.hc_dim(), tgt_.p + (size_t) tgt_index_[il] * E,
                            (int64_t) spark_->n_targets() * E, wins_[wi].T, c.hc, E, &d_par_[wi].pos0, kDsRing, st_);
    // the attention's mixes (its pre-mix is for the FFN; the attention collapses with the previous sublayer's)
    ds_hc_mixes(res_, F(L.hc_attn_fn), F(L.hc_attn_scale), F(L.hc_attn_base), attn_pre_, post_, comb_, T, c.hc, E, c.eps, c.hc_eps,
                c.sinkhorn, st_, mix_scratch_.p);
    trace(st_, il, "res_in", res_, T, c.hc_dim());
    trace(st_, il, "attn_pre", attn_pre_, T, c.hc);
    trace(st_, il, "post", post_, T, c.hc);
    trace(st_, il, "comb", comb_, T, c.hc * c.hc);
    ds_hc_pre_norm_q(res_, pre_, F(L.attn_norm), mixed_, xn_, T, c.hc, E, c.eps, act_, st_);
    trace(st_, il, "mixed", mixed_, T, E);
    trace(st_, il, "xn", xn_, T, E);
    attn(il, T);
    ds_hc_post(out_, res_, post_, comb_, T, c.hc, E, st_);
    trace(st_, il, "res_attn", res_, T, c.hc_dim());
    ds_hc_mixes(res_, F(L.hc_ffn_fn), F(L.hc_ffn_scale), F(L.hc_ffn_base), pre_next_, post_, comb_, T, c.hc, E, c.eps, c.hc_eps,
                c.sinkhorn, st_, mix_scratch_.p);
    ds_hc_pre_norm_q(res_, attn_pre_, F(L.ffn_norm), mixed_, xn_, T, c.hc, E, c.eps, act_, st_);
    trace(st_, il, "xn_ffn", xn_, T, E);
    if (pull_P_) {
        // the first design waited for this layer's pull here; now the plan takes what has landed and the restore
        // (after the FFN) waits for a straggler, so a late pull costs the CPU an expert, not the GPU a stall
        if (pull_wait_ && il > l0_) CUDA_CHECK(cudaStreamWaitEvent(st_, ev_pull_join_, 0));
        if (il + 1 < l1_) pull_predict(il, T);
    }
    moe(il, T);
    ds_hc_post(moe_out_, res_, post_, comb_, T, c.hc, E, st_);
    trace(st_, il, "res_out", res_, T, c.hc_dim());
    std::swap(pre_, pre_next_);
    if (head_->dump_all) {
        auto & dl = head_->dumped_layers;
        dl.resize(c.n_layer);
        dl[il].resize((size_t) T * c.hc_dim());
        CUDA_CHECK(cudaMemcpyAsync(dl[il].data(), res_.p, dl[il].size() * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
    }
}

void DsEngine::enqueue_forward(int T) {
    const DsConfig & c = model_.ds;
    const size_t HC = c.hc_dim();
    CUDA_CHECK(cudaMemcpyAsync(d_par_, h_par_, wins_.size() * sizeof(WinParams), cudaMemcpyHostToDevice, st_));
    pre_ = pre_a_;
    pre_next_ = pre_b_;
    pull_pending_ = -1;
    if (first()) {
        dequant_gather(model_.tok_embd, h_tok_, T, x_, st_);
        hc_init(x_, res_, T, c.hc, c.n_embd, st_);
        ds_onehot(pre_, T, c.hc, st_);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(res_.p, xfer_in_[lane_], (size_t) T * HC * 4, cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaMemcpyAsync(pre_, xfer_in_[lane_] + (size_t) kMaxWindow * HC, (size_t) T * c.hc * 4, cudaMemcpyHostToDevice, st_));
    }
    for (int il = l0_; il < l1_; ++il) layer_forward(il, T);
    if (pull_P_ && l1_ - l0_ >= 2) CUDA_CHECK(cudaStreamWaitEvent(st_, ev_pull_join_, 0));   // the pull stream rejoins
    if (last_) head(T);
    else {
        CUDA_CHECK(cudaMemcpyAsync(xfer_out_[lane_], res_.p, (size_t) T * HC * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaMemcpyAsync(xfer_out_[lane_] + (size_t) kMaxWindow * HC, pre_, (size_t) T * c.hc * 4, cudaMemcpyDeviceToHost, st_));
    }
}

void DsEngine::service_cpu(uint32_t seq, int T) {
    const DsConfig & c = model_.ds;
    const int E = c.n_embd;
    for (int il = l0_; il < l1_; ++il) {
        MoeMsg * m = msg(il);
        const double t0 = now_ms();
        // the mailbox wait with a watchdog: a GPU that never posts (a fault, a stream waiting on an event never recorded)
        // used to spin here forever; after 20 s the state is reported, after 60 s the forward fails
        for (uint64_t spins = 0; __atomic_load_n(&m->seq_req, __ATOMIC_ACQUIRE) != seq; ++spins) {
            __builtin_ia32_pause();
            if ((spins & 0xfffff) == 0 && now_ms() - t0 > 20000) {
                const cudaError_t q = cudaStreamQuery(st_), last = cudaGetLastError();
                fprintf(stderr, "bnk: layers %d-%d: layer %d's CPU mailbox has not been posted for %.0f s (seq %u, mailbox %u; stream %s, last error %s)\n",
                        l0_, l1_ - 1, il, (now_ms() - t0) / 1000, seq, (unsigned) __atomic_load_n(&m->seq_req, __ATOMIC_ACQUIRE),
                        cudaGetErrorString(q), cudaGetErrorString(last));
                if (now_ms() - t0 > 60000) throw std::runtime_error("the GPU forward stalled: layer " + std::to_string(il) + "'s mailbox never arrived");
            }
        }
        const double t1 = now_ms();
        const int n = m->n_miss;
        {
            std::lock_guard<std::mutex> lk(head_->stats_mu_);
            head_->times.wait_ms += t1 - t0;
            head_->times.misses += n;
            head_->times.routed += (int64_t) T * c.n_used;
            head_->layer_misses[il] += n;
            head_->layer_routed[il] += (int64_t) T * c.n_used;
        }
        if (n > 0) {
            tasks_.clear();
            for (int i = 0; i < n; ++i) {
                if (!store_->blob(il, m->miss_e[i]))
                    throw std::runtime_error("layer " + std::to_string(il) + ": expert " + std::to_string(m->miss_e[i]) +
                                             " has no host copy but was not found in the cache");
                tasks_.push_back({m->miss_t[i], 0, m->miss_e[i], m->miss_w[i]});
            }
            cpu_->run(il, T, m->x(), tasks_, m->out(E));
            __atomic_thread_fence(__ATOMIC_RELEASE);
            m->seq_done = seq;
            std::lock_guard<std::mutex> lk(head_->stats_mu_);
            head_->times.cpu_experts_ms += now_ms() - t1;
        }
    }
}

void DsEngine::forward(const int32_t * tokens, int T, bool commit_all) {
    forward_batch({BatchWin{current_slot(), tokens, T, commit_all}});
}

void DsEngine::forward_batch(const std::vector<BatchWin> & batch_in) {
    std::vector<BatchWin> batch = batch_in;
    std::sort(batch.begin(), batch.end(), [](const BatchWin & a, const BatchWin & b) { return a.slot < b.slot; });
    if (batch.empty() || batch.size() > (size_t) kMaxSlots) throw std::runtime_error("forward: bad batch");
    const double t0 = now_ms();
    for (DsEngine * s = this; s; s = s->next_.get()) s->fwd_prepare(batch);
    cudaEvent_t after = nullptr;
    for (DsEngine * s = this; s; s = s->next_.get()) {
        s->fwd_launch(after);
        after = s->ev_out_;
    }
    for (DsEngine * s = this; s; s = s->next_.get()) s->fwd_service();
    for (DsEngine * s = this; s; s = s->next_.get()) s->fwd_end();
    times.total_ms += now_ms() - t0;
    times.calls++;
}

void DsEngine::forward_pipe(const std::vector<BatchWin> & first_in, int first_lane, const std::vector<BatchWin> & second_in,
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
    DsEngine * s1 = next_.get();
    const double t0 = now_ms();
    if (!a.empty()) fwd_prepare(a, first_lane);
    if (!b.empty()) s1->fwd_prepare(b, second_lane);
    if (!a.empty()) fwd_launch(nullptr);
    if (!b.empty()) s1->fwd_launch(nullptr);
    // both stages' CPU experts at once: the second stage's on a helper thread (the pool takes one caller at a time)
    std::exception_ptr err;
    std::thread th;
    if (!b.empty()) th = std::thread([&]() { try { s1->fwd_service(); } catch (...) { err = std::current_exception(); } });
    if (!a.empty()) fwd_service();
    if (th.joinable()) th.join();
    if (err) std::rethrow_exception(err);
    if (!a.empty()) fwd_end();
    if (!b.empty()) s1->fwd_end();
    times.total_ms += now_ms() - t0;
    times.calls++;
}

void DsEngine::fwd_prepare(const std::vector<BatchWin> & batch, int lane) {
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
    if (R > kMaxWindow) throw std::runtime_error("forward: more than kMaxWindow rows");
    fwd_t0_ = now_ms();
    wins_.clear();
    ++seq_;
    std::vector<int> key{(int) batch.size()};
    int max_cells = 0;
    for (const auto & b : batch) {
        Seq * q = seqs_.at(b.slot).get();
        if (q->pending_T) throw std::runtime_error("forward: the previous verify window was not committed");
        for (const Win & w : wins_)
            if (w.q == q) throw std::runtime_error("forward: a slot twice in one batch");
        use(q);
        if (!q->used) { q->used = true; reset_here(); }
        if (pos() + b.T > opt_.max_ctx) { use(keep); throw std::runtime_error("context full"); }
        ensure_ctx_here(pos() + b.T + kMaxWindow);
        max_cells = std::max(max_cells, q->ctx_mapped);
        const int r0 = wins_.empty() ? 0 : wins_.back().r0 + wins_.back().T;
        wins_.push_back(Win{q, r0, b.T, pos(), b.commit_all});
        q->history.insert(q->history.end(), b.tokens, b.tokens + b.T);
        q->last_r0 = r0;
        q->last_lane = lane;
        for (int j = 0; j < b.T; ++j) h_tok_[r0 + j] = b.tokens[j];
        h_par_[wins_.size() - 1] = WinParams{wins_.back().pos0, b.T, seq_, 0};
        key.insert(key.end(), {b.slot, b.T, b.commit_all ? 1 : 0});
    }
    key.push_back(lane);
    key.push_back(max_cells);
    use(keep);
    last_T = R;
    ctx_grid_ = max_cells;
    engram_gather();
    fwd_graph_ = opt_.use_graphs && !head_->dump_all;
    fwd_exec_ = nullptr;
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
            graph_bytes_ += fg.bytes;
            ++graph_captures_;
        }
        fg.used = ++fwd_graph_age_;
        fwd_exec_ = fg.exec;
    }
}

void DsEngine::fwd_launch(cudaEvent_t after) {
    cache_.wait_exchanges(st_);   // the exchange's copies settle before the forward (docs/DEEPSEEK.md)
    DeviceScope ds(dev_);
    if (after) CUDA_CHECK(cudaStreamWaitEvent(st_, after, 0));
    CUDA_CHECK(cudaEventRecord(ev_t0_, st_));
    if (fwd_graph_) CUDA_CHECK(cudaGraphLaunch(fwd_exec_, st_));
    else enqueue_forward(last_T);
    CUDA_CHECK(cudaEventRecord(ev_out_, st_));
    CUDA_CHECK(cudaEventRecord(ev_t1_, st_));
}

void DsEngine::fwd_service() {
    if (!fwd_graph_) return;
    DeviceScope ds(dev_);
    service_cpu(seq_, last_T);
}

void DsEngine::fwd_end() {
    DeviceScope ds(dev_);
    CUDA_CHECK(cudaStreamSynchronize(st_));
    sync_debug("after the forward", dev_, 1);   // (a device sync is only safe once every stage has been serviced)
    float ms = 0.f;
    if (cudaEventElapsedTime(&ms, ev_t0_, ev_t1_) == cudaSuccess) {
        gpu_ms_ += ms;
        gpu_forwards_++;
    } else {
        cudaGetLastError();
    }
    for (const Win & w : wins_) w.q->pending_T = w.commit ? 0 : w.T;
    // the adaptive exchange, between forwards (the CPU pool and the prompt path are idle here)
    // the adaptive exchange, between forwards (the CPU pool and the prompt path are idle here); counted in window rows,
    // so a drafted forward of several tokens adapts the cache as fast per token as plain decoding
    if (adapt_every_ > 0 && !pull_P_ && (fwd_count_ += last_T) >= adapt_every_) {
        fwd_count_ = 0;
        sync_debug("before adapt_exchange", dev_, 2);
        cache_.adapt_exchange(counts_.p, st_, adapt_swaps_, *store_);
        sync_debug("after adapt_exchange", dev_, 2);
    }
    if (pull_P_) {
        uint32_t v[3];
        CUDA_CHECK(cudaMemcpy(v, pull_stats_.p, sizeof(v), cudaMemcpyDeviceToHost));
        std::lock_guard<std::mutex> lk(head_->stats_mu_);
        head_->times.pulled += v[0] - pull_stats_seen_[0];
        head_->times.pull_used += v[1] - pull_stats_seen_[1];
        head_->times.pull_late += v[2] - pull_stats_seen_[2];
        for (int i = 0; i < 3; ++i) pull_stats_seen_[i] = v[i];
    }
}

// ------------------------------------------------------------------------------------------------ the PCIe lane
void DsEngine::pull_init() {
    DeviceScope ds(dev_);
    const DsConfig & c = model_.ds;
    const int NE = c.n_expert;
    // off by default: measured 17.6 tok/s against 19-20 with it off (docs/DEEPSEEK-NEXT.md: the plan kernel's host
    // mailbox traffic and the pull's reads share the link); BNK_DS_PULL=1 or 2 turns it on
    pull_P_ = env_int("BNK_DS_PULL", 0);
    if (pull_P_ <= 0 || l1_ - l0_ < 2) { pull_P_ = 0; return; }
    if (pull_P_ > 8) pull_P_ = 8;
    if (pull_P_ > 0 && adapt_every_ > 0) {   // the lane's host pointer table would go stale
        if (opt_.verbose) fprintf(stderr, "bnk: the PCIe lane is on: the adaptive exchange is off\n");
        for (DsEngine * st = head_; st; st = st->next_.get()) st->adapt_every_ = 0;
    }
    if (!store_->pinned()) {
        if (opt_.verbose) fprintf(stderr, "bnk: GPU %d: PCIe lane off (the experts' host memory is not page-locked)\n", dev_);
        pull_P_ = 0;
        return;
    }
    for (int il = l0_; il < l1_; ++il)
        if (cache_.n_cold_groups(il)) { pull_P_ = 0; return; }   // (the staging slots take the cold bands' numbers)
    std::vector<uint64_t> ptrs((size_t) c.n_layer * NE, 0);
    size_t slots = 0, max_blob = 0;
    for (int il = l0_; il < l1_; ++il) {
        if (store_->blob_bytes(il) % 16 || cache_.slots(il) + pull_P_ > NE) { pull_P_ = 0; return; }
        slots += (size_t) pull_P_ * cache_.desc(il).blob;
        max_blob = std::max(max_blob, store_->blob_bytes(il));
        for (int e = 0; e < NE; ++e) {
            const uint8_t * b = store_->blob(il, e);
            if (!b) continue;
            void * dp = nullptr;
            if (cudaHostGetDevicePointer(&dp, (void *) b, 0) != cudaSuccess) {
                cudaGetLastError();
                if (opt_.verbose) fprintf(stderr, "bnk: GPU %d: PCIe lane off (the experts' host memory is not mapped)\n", dev_);
                pull_P_ = 0;
                return;
            }
            ptrs[(size_t) il * NE + e] = (uint64_t) dp;
        }
    }
    pull_blob_ptr_.alloc(ptrs.size());
    CUDA_CHECK(cudaMemcpy(pull_blob_ptr_.p, ptrs.data(), ptrs.size() * 8, cudaMemcpyHostToDevice));
    pull_raw_.alloc(max_blob * pull_P_);
    std::vector<uint8_t *> dst((size_t) (l1_ - l0_) * pull_P_);
    for (int il = l0_; il < l1_; ++il) {
        cache_.reserve_spare(il, pull_P_);
        const MoeLayerDesc d = cache_.desc(il);
        for (int p = 0; p < pull_P_; ++p) dst[(size_t) (il - l0_) * pull_P_ + p] = (uint8_t *) d.base + (size_t) cache_.spare_slot(il, p) * d.blob;
    }
    pull_dst_.alloc(dst.size());
    CUDA_CHECK(cudaMemcpy(pull_dst_.p, dst.data(), dst.size() * sizeof(uint8_t *), cudaMemcpyHostToDevice));
    pull_picks_.alloc((size_t) c.n_layer * pull_P_);
    pull_staged_.alloc((size_t) c.n_layer * pull_P_);
    CUDA_CHECK(cudaMemset(pull_picks_.p, 0xff, (size_t) c.n_layer * pull_P_ * 4));
    CUDA_CHECK(cudaMemset(pull_staged_.p, 0xff, (size_t) c.n_layer * pull_P_ * 4));
    pull_logits_.alloc((size_t) kMaxWindow * NE);
    pull_ids_.alloc((size_t) kMaxWindow * c.n_used);
    pull_w_.alloc((size_t) kMaxWindow * c.n_used);
    pull_stats_.alloc(3);
    CUDA_CHECK(cudaMemset(pull_stats_.p, 0, 12));
    pull_state_.alloc(c.n_layer);
    CUDA_CHECK(cudaMemset(pull_state_.p, 0, (size_t) c.n_layer * 4));
    pull_wait_ = env_int("BNK_DS_PULL_WAIT", 0) != 0;
    {   // BNK_DS_PULL_PRIO=1: the pull stream at the lowest priority (its blocks run only beside the forward's kernels;
        // measured late for the window), 0 (default): the forward's own priority
        int prio_lo = 0, prio_hi = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&st_pull_, cudaStreamNonBlocking, env_int("BNK_DS_PULL_PRIO", 0) ? prio_lo : prio_hi));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_pull_fork_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_pull_join_, cudaEventDisableTiming));
    if (opt_.verbose)
        fprintf(stderr, "bnk: GPU %d: PCIe lane: %d expert%s a layer predicted a layer ahead, %.0f MiB of spare slots\n", dev_,
                pull_P_, pull_P_ > 1 ? "s" : "", (slots + max_blob * pull_P_) / 1048576.0);
}

void DsEngine::pull_predict(int il, int T) {
    const DsConfig & c = model_.ds;
    const DsLayer & L1 = model_.layers[il + 1];
    const int E = c.n_embd, k = c.n_used, NE = c.n_expert;
    const MoeLayerDesc d1 = cache_.desc(il + 1);
    // layer il + 1's routing of this layer's FFN input
    gemv(L1.router, nullptr, xn_, E, T, pull_logits_, NE, false, st_);
    ds_route(pull_logits_, F(L1.probs_b), T, NE, k, c.route_scale, c.route_norm, pull_ids_, pull_w_, st_);
    int32_t * picks = pull_picks_.p + (size_t) (il + 1) * pull_P_;
    ds_pull_pick(pull_ids_, pull_w_, T, k, d1.slot_of, picks, pull_P_, st_, pull_state_.p + il + 1);
    pull_pending_ = il;
}

void DsEngine::pull_launch(int il) {
    if (pull_pending_ != il) return;
    pull_pending_ = -1;
    const DsConfig & c = model_.ds;
    const int E = c.n_embd, NE = c.n_expert;
    const MoeLayerDesc d1 = cache_.desc(il + 1);
    int32_t * picks = pull_picks_.p + (size_t) (il + 1) * pull_P_;
    // the pull on the side stream: the blobs over PCIe, repacked into the slots, the slot table patched
    CUDA_CHECK(cudaEventRecord(ev_pull_fork_, st_));
    CUDA_CHECK(cudaStreamWaitEvent(st_pull_, ev_pull_fork_, 0));
    const size_t bb = store_->blob_bytes(il + 1);
    ds_pull_copy(picks, pull_P_, pull_blob_ptr_.p + (size_t) (il + 1) * NE, bb, pull_raw_.p, st_pull_);
    moe_repack_blobs(pull_raw_.p, bb, pull_dst_.p + (size_t) (il + 1 - l0_) * pull_P_, pull_P_, c.n_ff, E, d1.gate_type,
                     ggml_row_size((ggml_type) d1.gate_type, E), d1.down_type, ggml_row_size((ggml_type) d1.down_type, c.n_ff), st_pull_);
    ds_pull_commit(picks, pull_P_, const_cast<int32_t *>(d1.slot_of), cache_.spare_slot(il + 1, 0),
                   pull_staged_.p + (size_t) (il + 1) * pull_P_, st_pull_, pull_state_.p + il + 1);
    CUDA_CHECK(cudaEventRecord(ev_pull_join_, st_pull_));
}

// ------------------------------------------------------------------------------------------------ commit
void DsEngine::enqueue_commit(int cnt) {
    const DsConfig & c = model_.ds;
    const int D = c.head_dim, ln = cur_->last_lane;
    const size_t r0 = (size_t) cur_->last_r0;
    size_t wi = 0;
    while (wi < wins_.size() && wins_[wi].q != cur_) ++wi;
    if (wi == wins_.size()) throw std::runtime_error("commit: the slot was not in the last forward");
    for (int il = l0_; il < l1_; ++il)
        if (c.is_kv_src(il) && c.ratio[il] > 1)
            ds_compress_commit(ckv_raw_[ln][il].p + r0 * D, sc_raw_[ln][il].p + r0 * D, cur_->st_kv[il], cur_->st_sc[il], cnt,
                               c.ratio[il], D, &d_par_[wi].pos0, st_);
}

void DsEngine::join_commit() {
    for (DsEngine * s = this; s; s = s->next_.get()) s->join_commit_here();
}

void DsEngine::join_commit_here() {
    if (!commit_inflight_) return;
    CUDA_CHECK(cudaStreamWaitEvent(st_, ev_committed_, 0));
    commit_inflight_ = false;
}

void DsEngine::commit(int cnt) {
    for (DsEngine * s = this; s; s = s->next_.get()) s->commit_here(cnt);
}

void DsEngine::commit_here(int cnt) {
    DeviceScope ds(dev_);
    if (cur_->pending_T == 0) throw std::runtime_error("commit: no verify window pending");
    if (cnt < 1 || cnt > cur_->pending_T) throw std::runtime_error("commit: count out of range");
    cur_->history.resize(cur_->history.size() - (cur_->pending_T - cnt));
    cur_->pending_T = 0;
    bool any = false;
    for (int il = l0_; il < l1_; ++il) any |= model_.ds.is_kv_src(il) && model_.ds.ratio[il] > 1;
    if (!any) return;
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
        CUDA_CHECK(cudaEventRecord(ev_verified_, st_));
        CUDA_CHECK(cudaStreamWaitEvent(st_commit_, ev_verified_, 0));
        CUDA_CHECK(cudaGraphLaunch(cg, st_commit_));
        CUDA_CHECK(cudaEventRecord(ev_committed_, st_commit_));
        commit_inflight_ = true;
    } else {
        enqueue_commit(cnt);
    }
}

// ------------------------------------------------------------------------------------------------ logits
int DsEngine::argmax(int t) {
    if (next_) return next_->argmax(t);
    DeviceScope ds(dev_);
    argmax_rows(logits_.p + (size_t) t * model_.cfg.n_vocab, 1, model_.cfg.n_vocab, argmax_dev_, st_);
    int32_t r;
    CUDA_CHECK(cudaMemcpyAsync(&r, argmax_dev_.p, 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    return r;
}

void DsEngine::argmax_all(int T, int32_t * out, int row0) {
    if (next_) return next_->argmax_all(T, out, row0);
    DeviceScope ds(dev_);
    argmax_rows(logits_.p + (size_t) row0 * model_.cfg.n_vocab, T, model_.cfg.n_vocab, argmax_dev_, st_);
    CUDA_CHECK(cudaMemcpyAsync(out, argmax_dev_.p, T * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void DsEngine::topk_rows_host(int row0, int T, int K, int32_t * ids, float * vals, double * z, float temp) {
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

void DsEngine::logits_rows_host(int T, float * out, int row0) {
    if (next_) return next_->logits_rows_host(T, out, row0);
    DeviceScope ds(dev_);
    CUDA_CHECK(cudaMemcpy(out, logits_.p + (size_t) row0 * model_.cfg.n_vocab, (size_t) T * model_.cfg.n_vocab * 4,
                          cudaMemcpyDeviceToHost));
}

std::vector<float> DsEngine::logits_host(int t) {
    if (next_) return next_->logits_host(t);
    DeviceScope ds(dev_);
    std::vector<float> v(model_.cfg.n_vocab);
    CUDA_CHECK(cudaMemcpy(v.data(), logits_.p + (size_t) t * model_.cfg.n_vocab, v.size() * 4, cudaMemcpyDeviceToHost));
    return v;
}

}  // namespace bnk
