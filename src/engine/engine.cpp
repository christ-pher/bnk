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
                   cudaStream_t s) {
    if (is_float_format(W.type)) gemv(W, nullptr, x, ldx, T, y, ldy, false, s);
    else gemv(W, &xq, nullptr, 0, T, y, ldy, false, s);
}

Engine::~Engine() {
    for (auto & q : seqs_) {
        for (auto & k : q->ckpts)
            if (k.host) cudaFreeHost(k.host);
        for (auto & gm : q->commit_graphs) for (auto & g : gm) if (g) cudaGraphExecDestroy(g);
    }
    for (auto & kv : fwd_graphs_) cudaGraphExecDestroy(kv.second.exec);
    if (counts_.p) save_counts();
    if (mail_) cudaFreeHost(mail_);
    if (h_par_) cudaFreeHost(h_par_);
    if (h_tok_) cudaFreeHost(h_tok_);
    if (h_ple_rows_) cudaFreeHost(h_ple_rows_);
    if (d_par_) cudaFree(d_par_);
}

void Engine::load(const std::string & path, const EngineOptions & opt) {
    opt_ = opt;
    ggml_cpu_init();
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    CUDA_CHECK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&st_commit_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_verified_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_committed_, cudaEventDisableTiming));
    model_.load(path, opt.verbose, opt.ple_gguf);
    layer_routed.assign(model_.cfg.n_layer, 0);
    layer_misses.assign(model_.cfg.n_layer, 0);
    store_.build(model_, std::min(opt.cpu_threads, 16), opt.verbose);
    cpu_.init(model_, store_, opt.cpu_threads);

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
    shared_out_.alloc(W * E); moe_out_.alloc(W * E); logits_.alloc((size_t) W * c.n_vocab);
    argmax_dev_.alloc(W);
    const int64_t maxcols = std::max<int64_t>(HC, 16384);
    actq_.alloc(W * maxcols); actd_.alloc(W * maxcols / 32);
    acts_.alloc(W * maxcols / 16);
    act_.q = actq_; act_.d = actd_; act_.s = acts_;
    mixq_.alloc(W * E + 64); mixd_.alloc(W * E / 32 + 8); mixs_.alloc(W * E / 16 + 16);
    mixact_.q = mixq_; mixact_.d = mixd_; mixact_.s = mixs_;

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

    if (c.ple_layer >= 0) {
        const int ph = c.ple_heads();
        ple_emb_.alloc(W * ph * c.ple_dim); ple_key_.alloc(W * HC); ple_val_.alloc(W * E);
        ple_gated_.alloc(W * HC);
        CUDA_CHECK(cudaHostAlloc((void **) &h_ple_rows_, (size_t) W * ph * model_.ple_table->row_bytes(),
                                 cudaHostAllocMapped));
    }
    gdn_co_.resize(c.n_layer);
    gdn_g_.resize(c.n_layer);
    gdn_b_.resize(c.n_layer);
    size_t state_bytes = 0, kv_cell_bytes = 0;
    for (int il = 0; il < c.n_layer; ++il) {
        if (c.is_attn(il)) {
            kv_cell_bytes += 2 * (size_t) c.n_head_kv * c.head_dim * sizeof(half);
            if (model_.layers[il].idx_q.valid() && c.compress_ratio[il] > 0 && !getenv("BNK_NO_QSA")) {
                kv_cell_bytes += c.idx_dim * sizeof(half) + c.idx_dim * sizeof(float) / c.compress_ratio[il];
                qsa_on_ = true;
            }
        } else {
            gdn_co_[il].alloc((size_t) W * C);
            gdn_g_[il].alloc((size_t) W * c.ssm_vheads);
            gdn_b_[il].alloc((size_t) W * c.ssm_vheads);
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
        fprintf(stderr, "bnk: context %d: %.0f KiB of KV per token (%.2f GiB when full, mapped as used), "
                        "recurrent state %.2f GiB, %d CPU expert threads%s\n",
                opt.max_ctx, kv_cell_bytes / 1024.0, (double) kv_cell_bytes * opt.max_ctx / 1073741824.0,
                state_bytes / 1073741824.0, cpu_.threads(),
                seqs_.size() > 1 ? (", " + std::to_string(seqs_.size()) + " conversation slots").c_str() : "");

    if (const char * rl = getenv("BNK_ROUTE_LOG")) {
        route_log_ = fopen(rl, "wb");
        if (!route_log_) throw std::runtime_error(std::string("cannot open ") + rl);
        route_dev_.alloc((size_t) c.n_layer * kMaxWindow * c.n_expert_used);
        cpu_us_.assign(c.n_layer, 0.f);
    }
    if (!opt.mtp.empty())
        mtp_.load(opt.mtp, model_, opt.max_ctx, st_, opt.verbose, opt.draft_vocab, &budget_, (int) seqs_.size());
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
    budget_.reclaim = [this](size_t need) { return cache_ready_ ? cache_.shrink(need, st_) : (size_t) 0; };
    ensure_ctx(std::min(opt.max_ctx, kCtxBaseline));
    if (opt.prefill_chunk > 0) prefill_layout(pf_small_);
    Ranking rank = load_ranking(opt.profile, c.n_layer, c.n_expert);
    if (rank.empty() && !opt.counts_out.empty()) rank = load_ranking(opt.counts_out, c.n_layer, c.n_expert);
    if (opt.verbose) fprintf(stderr, "bnk: expert ranking: %s\n", rank.empty() ? "none (uniform)" : "loaded");
    const size_t avail = budget_.free();
    size_t budget = avail;
    if (opt.expert_cache_gib >= 0) budget = std::min(budget, (size_t) (opt.expert_cache_gib * 1073741824.0));
    if (opt.prefill_chunk > 0) {
        // the expert stage kept between prompts (two of them) depends on how many experts each layer leaves
        // outside the cache: iterate to a fixed point
        const size_t small = (size_t) opt.stage_small_mib << 20;
        for (int it = 0; it < 6; ++it) {
            const auto slots = ExpertCache::plan(model_, store_, budget, rank);
            size_t need = 0;
            for (int il = 0; il < c.n_layer; ++il)
                need = std::max(need, (size_t) (c.n_expert - slots[il]) * store_.blob_bytes(il));
            need = vmem_round(std::min(need, small));
            const size_t want = avail > 2 * need ? avail - 2 * need : 0;
            const size_t nb = opt.expert_cache_gib >= 0 ? std::min(want, (size_t) (opt.expert_cache_gib * 1073741824.0)) : want;
            if (nb == budget) break;
            budget = nb;
        }
    }
    cache_.init(model_, store_, budget, rank, st_, opt.verbose, &budget_);
    cache_ready_ = true;
    if (opt.prefill_chunk > 0) {
        prefill_staging_alloc();
        if (opt.verbose)
            fprintf(stderr, "bnk: prompt path: chunks of %d / %d / %d tokens (short turns / prompts / long prompts), "
                            "2 x %.2f GiB of expert staging between prompts (more borrowed for long ones)\n",
                    pf_small_, pf_base_, pf_big_, pf_.stage_bytes / 1073741824.0);
    }
    reset();
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
    for (int il = 0; il < c.n_layer; ++il) {
        if (c.is_attn(il)) {
            // address space for the whole context; VRAM is mapped as the conversation grows (ensure_ctx)
            const size_t kv = (size_t) opt_.max_ctx * c.n_head_kv * c.head_dim;
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
    if (c.ple_layer >= 0) q.ple_hist.alloc((size_t) ((c.ple_conv - 1) * c.ple_ngram + W) * HC);
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
    if (slot < 0 || slot >= (int) seqs_.size()) throw std::runtime_error("select: no slot " + std::to_string(slot));
    // no join: switching touches nothing on the GPU, and whatever does (forward, prefill, snapshots ...) joins
    use(seqs_[slot].get());
    if (!cur_->used) {   // first use: an empty conversation with the baseline context
        cur_->used = true;
        reset();
    }
}

// KV cells [0, cells) backed by VRAM, in steps of kCtxStep cells; the expert cache gives the bytes up.
void Engine::ensure_ctx(int cells) {
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
            const size_t kv = (size_t) cells * c.n_head_kv * c.head_dim * sizeof(half);
            add(cur_->kv_k[il], kv);
            add(cur_->kv_v[il], kv);
            if (cur_->kv_raw[il].reserved()) {
                add(cur_->kv_raw[il], (size_t) cells * c.idx_dim * sizeof(half));
                add(cur_->kv_pool[il], (size_t) (cells / c.compress_ratio[il] + 1) * c.idx_dim * sizeof(float));
            }
        }
        if (mtp_.loaded()) need += mtp_.ctx_bytes_needed(cells);
        if (!budget_.make_room(need)) throw std::runtime_error("not enough VRAM for a context of " + std::to_string(cells));
    }
    for (int il = 0; il < c.n_layer; ++il) {
        if (!cur_->kv_k[il].reserved()) continue;
        const size_t kv = (size_t) cells * c.n_head_kv * c.head_dim * sizeof(half);
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
    if (pf_base_ > 0) {
        prefill_layout(pf_small_);
        prefill_staging_shrink();
    }
    const int keep = std::min(opt_.max_ctx, std::max(kCtxBaseline, (pos() + kCtxStep) / kCtxStep * kCtxStep));
    if (keep < cur_->ctx_mapped) map_ctx(keep);
    const size_t slack = 64ull << 20;
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
    join_commit();
    const int p = pos();
    if (p == 0) return;
    for (auto & k : cur_->ckpts)
        if (k.pos == p) { k.age = ++ckpt_age_; return; }
    size_t floats = 0;
    each_state([&](float *, size_t n) { floats += n; });
    Checkpoint * slot = nullptr;
    for (auto & k : cur_->ckpts)
        if (k.pos < 0) { slot = &k; break; }
    if (!slot && (int) cur_->ckpts.size() < kMaxCheckpoints) {
        cur_->ckpts.emplace_back();
        slot = &cur_->ckpts.back();
        CUDA_CHECK(cudaHostAlloc((void **) &slot->host, floats * 4, 0));
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
    join_commit();
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
    join_commit();
    const Config & c = model_.cfg;
    cur_->history.clear();
    for (auto & k : cur_->ckpts) k.pos = -1;
    cur_->mtp_cell = -1;
    for (int il = 0; il < c.n_layer; ++il) {
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
    // the device counters are cumulative for the session; the file gets this session's total added
    if (opt_.counts_out.empty()) return;
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
    hc_norm(res_, F(w.norm), xn_, T, c.hc, E, c.rms_eps, st_);
    if (!is_float_format(w.down.type) || (want_inject && !is_float_format(w.inject.type)))
        quantize_act(xn_, HC, T, HC, act_, st_);
    gemv_q(w.down, act_, xn_, HC, T, lo_, c.hc_rank, st_);
    if (want_inject) gemv_q(w.inject, act_, xn_, HC, T, inj_, c.hc, st_);
    silu_scale(lo_, (int64_t) T * c.hc_rank, 1.f / c.hc, st_);
    gemv_auto(w.up, lo_, c.hc_rank, T, gpre_, HC, false, act_, st_);
    hc_mix(xn_, gpre_, mixed_, T, c.hc, E, st_);
}

// host side: the n-gram hash rows of the window, gathered into pinned memory the graph reads
void Engine::ple_gather() {
    const Config & c = model_.cfg;
    if (c.ple_layer < 0) return;
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
    times.ple_ms += now_ms() - t0;
}

void Engine::ple(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, HC = c.hc_dim(), ph = c.ple_heads();
    const TensorRef & tab = *model_.ple_table;
    QMat rows{h_ple_rows_, (int) tab.type, (int64_t) T * ph, tab.ne[0], tab.row_bytes()};
    dequant_rows(rows, 0, (int64_t) T * ph, ple_emb_, st_);
    const int D = ph * c.ple_dim;
    if (!is_float_format(L.ple_key.type)) quantize_act(ple_emb_, D, T, D, act_, st_);
    gemv_q(L.ple_key, act_, ple_emb_, D, T, ple_key_, HC, st_);
    gemv_q(L.ple_value, act_, ple_emb_, D, T, ple_val_, E, st_);
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
}

void Engine::gdn(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, C = c.conv_channels(), K = c.ssm_conv, S = c.ssm_state, nv = c.ssm_vheads;
    quantize_act(mixed_, E, T, E, mixact_, st_);
    // the new conv inputs go behind each window's own K-1 history rows (one window: straight there)
    const bool one = wins_.size() == 1;
    float * qkv = one ? wins_[0].q->conv_buf[il].p + (size_t) (K - 1) * C : conv_out_.p;
    gemv_q(L.wqkv, mixact_, mixed_, E, T, qkv, C, st_);
    if (!one)
        for (const Win & w : wins_)
            CUDA_CHECK(cudaMemcpyAsync(w.q->conv_buf[il].p + (size_t) (K - 1) * C, conv_out_.p + (size_t) w.r0 * C,
                                       (size_t) w.T * C * 4, cudaMemcpyDeviceToDevice, st_));
    gemv_q(L.wgate, mixact_, mixed_, E, T, z_, nv * S, st_);
    float * co = gdn_co_[il], * gg = gdn_g_[il], * gb = gdn_b_[il];
    gemv_q(L.ssm_beta, mixact_, mixed_, E, T, gb, nv, st_);
    gemv_q(L.ssm_alpha, mixact_, mixed_, E, T, gg, nv, st_);
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
    gated_rmsnorm(gdn_o_, z_, F(L.ssm_norm), gdn_n_, T, nv, S, c.rms_eps, st_);
    dbg(st_, il, "attn_output", gdn_o_, (size_t) T * nv * S);
    dbg(st_, il, "final_output", gdn_n_, (size_t) T * nv * S);
    gemv_auto(L.ssm_out, gdn_n_, nv * S, T, out_, E, false, act_, st_);
    dbg(st_, il, "linear_attn_out", out_, (size_t) T * E);
    for (const Win & w : wins_)
        if (w.commit) shift_rows(w.q->conv_buf[il], C, w.T, K - 1, st_);
}

// Keep the first c rows of the last verify window: replay the delta rule over them and shift the histories.
void Engine::enqueue_commit(int cnt) {
    const Config & c = model_.cfg;
    const int C = c.conv_channels(), K = c.ssm_conv, S = c.ssm_state, nv = c.ssm_vheads;
    const size_t r0 = (size_t) cur_->last_r0;
    for (int il = 0; il < c.n_layer; ++il) {
        if (c.is_attn(il)) continue;
        gdn_recurrence(gdn_co_[il].p + r0 * C, C, gdn_g_[il].p + r0 * nv, gdn_b_[il].p + r0 * nv, cur_->ssm_state[il],
                       nullptr, cnt, c.ssm_groups, nv, S, cnt, st_);
    }
    shift_rows_multi(cur_->conv_ptrs, n_gdn_, C, cnt, K - 1, st_);
    if (c.ple_layer >= 0) shift_rows(cur_->ple_hist, c.hc_dim(), cnt, (c.ple_conv - 1) * c.ple_ngram, st_);
}

void Engine::join_commit() {
    if (!commit_inflight_) return;
    CUDA_CHECK(cudaStreamWaitEvent(st_, ev_committed_, 0));
    commit_inflight_ = false;
}

void Engine::commit(int cnt) {
    if (!cur_->conv_ptrs.p) {   // the GDN conv buffers never move: their pointer table once, before any capture
        const Config & c = model_.cfg;
        std::vector<float *> ptrs;
        for (int il = 0; il < c.n_layer; ++il)
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
        cudaGraphExec_t & cg = cur_->commit_graphs[cur_->last_r0][cnt];
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
        cudaGraphExec_t & cg = cur_->commit_graphs[cur_->last_r0][cnt];
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
    quantize_act(mixed_, E, T, E, mixact_, st_);
    gemv_q(L.wq, mixact_, mixed_, E, T, qfull_, H * D * 2, st_);
    gemv_q(L.wk, mixact_, mixed_, E, T, k_, Hkv * D, st_);
    gemv_q(L.wv, mixact_, mixed_, E, T, v_, Hkv * D, st_);
    const bool qsa = wins_[0].q->kraw[il].p != nullptr;
    if (qsa) {
        gemv_q(L.idx_k, mixact_, mixed_, E, T, ik_, c.idx_dim, st_);
        gemv_q(L.idx_q, mixact_, mixed_, E, T, iq_, c.idx_heads * c.idx_dim, st_);
    }
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
    if (il + 1 < c.n_layer) {
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
    quantize_act(mixed_, E, T, E, mixact_, st_);
    gemv_q(L.router, mixact_, mixed_, E, T, rlogits_, c.n_expert, st_);
    route_topk(rlogits_, T, c.n_expert, k, rids_, rw_, c.expert_weights_scale, st_);
    if (route_log_)
        CUDA_CHECK(cudaMemcpyAsync(route_dev_.p + (size_t) il * kMaxWindow * k, rids_.p, (size_t) T * k * 4,
                                   cudaMemcpyDeviceToDevice, st_));
    moe_plan(rids_, rw_, T, k, d, moes_, msg(il), mixed_, E, &d_par_->seq, counts_.p + (size_t) il * c.n_expert, st_);
    if (!opt_.use_graphs || dump_all) {  // eager mode: answer the CPU part right here
        CUDA_CHECK(cudaStreamSynchronize(st_));
        static const bool pstats = getenv("BNK_PREDICT_STATS") != nullptr;
        if (pstats) predict_stats(il, T);
        MoeMsg * m = msg(il);
        tasks_.clear();
        for (int i = 0; i < m->n_miss; ++i) tasks_.push_back({m->miss_t[i], 0, m->miss_e[i], m->miss_w[i]});
        if (!tasks_.empty()) cpu_.run(il, T, m->x(), tasks_, m->out(E));
        times.misses += m->n_miss;
        times.routed += (int64_t) T * k;
        layer_misses[il] += m->n_miss;
        layer_routed[il] += (int64_t) T * k;
        __atomic_thread_fence(__ATOMIC_RELEASE);
        m->seq_done = m->seq_req;
    }
    moe_hits(d, moes_, mixact_, T, k, E, c.n_ff_exp, st_);
    gemv_q(L.sh_gate, mixact_, mixed_, E, T, sg_, FF, st_);
    gemv_q(L.sh_up, mixact_, mixed_, E, T, su_, FF, st_);
    silu_mul(sg_, su_, sh_, (int64_t) T * FF, st_);
    gemv_auto(L.sh_down, sh_, FF, T, shared_out_, E, false, act_, st_);
    gemv_q(L.shexp_gate_inp, mixact_, mixed_, E, T, sgate_, 1, st_);
    sigmoid_inplace(sgate_, T, st_);
    moe_wait(moes_, msg(il), &d_par_->seq, st_);
    moe_reduce(moes_, msg(il), shared_out_, sgate_, moe_out_, T, k, E, st_);
}

void Engine::head(int T) {
    const Config & c = model_.cfg;
    hc_pre(model_.hc_head, T, false);
    gemv_auto(model_.output, mixed_, c.n_embd, T, logits_, c.n_vocab, false, act_, st_);
}

void Engine::layer_forward(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    if (il == c.ple_layer) ple(il, T);
    hc_pre(L.hc_attn, T, true);
    if (L.attn) attn(il, T); else gdn(il, T);
    hc_combine(res_, out_, inj_, T, c.hc, c.n_embd, st_);
    dbg(st_, il, "hc_combine", res_, (size_t) T * c.hc_dim());
    hc_pre(L.hc_ffn, T, true);
    dbg(st_, il, "hc_mixed_ffn", mixed_, (size_t) T * c.n_embd);
    dbg(st_, il, "hc_inject_ffn", inj_, (size_t) T * c.hc);
    moe(il, T);
    dbg(st_, il, "ffn_out", moe_out_, (size_t) T * c.n_embd);
    hc_combine(res_, moe_out_, inj_, T, c.hc, c.n_embd, st_);
    dbg(st_, il, "l_last", res_, (size_t) T * c.hc_dim());
    if (dump_all) {
        dumped_layers.resize(c.n_layer);
        dumped_layers[il].resize((size_t) T * c.hc_dim());
        CUDA_CHECK(cudaMemcpyAsync(dumped_layers[il].data(), res_.p, dumped_layers[il].size() * 4,
                                   cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
    }
}

void Engine::enqueue_forward(int T) {
    const Config & c = model_.cfg;
    CUDA_CHECK(cudaMemcpyAsync(d_par_, h_par_, wins_.size() * sizeof(WinParams), cudaMemcpyHostToDevice, st_));
    dequant_gather(model_.tok_embd, h_tok_, T, x_, st_);
    hc_init(x_, res_, T, c.hc, c.n_embd, st_);
    for (int il = 0; il < c.n_layer; ++il) layer_forward(il, T);
    head(T);
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
    for (int il = 0; il < c.n_layer; ++il) {
        MoeMsg * m = msg(il);
        const double t0 = now_ms();
        while (__atomic_load_n(&m->seq_req, __ATOMIC_ACQUIRE) != seq) __builtin_ia32_pause();
        const double t1 = now_ms();
        times.wait_ms += t1 - t0;
        const int n = m->n_miss;
        times.misses += n;
        times.routed += (int64_t) T * c.n_expert_used;
        layer_misses[il] += n;
        layer_routed[il] += (int64_t) T * c.n_expert_used;
        if (n > 0) {
            tasks_.clear();
            for (int i = 0; i < n; ++i) tasks_.push_back({m->miss_t[i], 0, m->miss_e[i], m->miss_w[i]});
            cpu_.run(il, T, m->x(), tasks_, m->out(E));
            __atomic_thread_fence(__ATOMIC_RELEASE);
            m->seq_done = seq;
            const double t2 = now_ms();
            times.cpu_experts_ms += t2 - t1;
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
    join_commit();
    // windows in slot order, whatever order the caller lists them in: one graph per layout, not per permutation
    std::vector<BatchWin> batch = batch_in;
    std::sort(batch.begin(), batch.end(), [](const BatchWin & a, const BatchWin & b) { return a.slot < b.slot; });
    if (batch.empty() || batch.size() > (size_t) kMaxSlots) throw std::runtime_error("forward: bad batch");
    Seq * keep = cur_;
    int R = 0;
    for (const auto & b : batch) {
        if (b.T < 1) throw std::runtime_error("forward: bad window");
        R += b.T;
    }
    if (R > kMaxWindow) throw std::runtime_error("forward: more than kMaxWindow rows");
    const double t0 = now_ms();
    wins_.clear();
    ++seq_;
    std::vector<int> key{(int) batch.size()};
    for (const auto & b : batch) {
        Seq * q = seqs_.at(b.slot).get();
        if (q->pending_T) throw std::runtime_error("forward: the previous verify window was not committed");
        for (const Win & w : wins_)
            if (w.q == q) throw std::runtime_error("forward: a slot twice in one batch");
        use(q);
        if (!q->used) { q->used = true; reset(); }
        if (pos() + b.T > opt_.max_ctx) { use(keep); throw std::runtime_error("context full"); }
        ensure_ctx(pos() + b.T + kMaxWindow);
        const int r0 = wins_.empty() ? 0 : wins_.back().r0 + wins_.back().T;
        wins_.push_back(Win{q, r0, b.T, pos(), b.commit_all});
        q->history.insert(q->history.end(), b.tokens, b.tokens + b.T);
        q->last_r0 = r0;
        memcpy(h_tok_ + r0, b.tokens, b.T * 4);
        h_par_[wins_.size() - 1] = WinParams{wins_.back().pos0, b.T, seq_, 0};
        key.insert(key.end(), {b.slot, b.T, b.commit_all ? 1 : 0});
    }
    use(keep);
    last_T = R;
    ple_gather();
    if (opt_.use_graphs && !dump_all) {
        FwdGraph & fg = fwd_graphs_[key];
        if (!fg.exec) {
            while ((int) fwd_graphs_.size() > kMaxFwdGraphs) {   // the least recently used layout goes
                auto lru = fwd_graphs_.end();
                for (auto it = fwd_graphs_.begin(); it != fwd_graphs_.end(); ++it)
                    if (it->second.exec && (lru == fwd_graphs_.end() || it->second.used < lru->second.used)) lru = it;
                if (lru == fwd_graphs_.end()) break;
                cudaGraphExecDestroy(lru->second.exec);
                budget_.refund(lru->second.bytes);
                graph_bytes_ -= std::min(graph_bytes_, lru->second.bytes);
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
        cudaGraphExec_t ge = fg.exec;
        if (route_log_) moe_debug_reset(st_);
        CUDA_CHECK(cudaGraphLaunch(ge, st_));
        service_cpu(seq_, R);
        static const bool prof = getenv("BNK_SVC_PROF") != nullptr;
        if (prof) {
            CUDA_CHECK(cudaStreamSynchronize(st_));
            uint64_t ts[64][3];
            moe_debug_times(ts, model_.cfg.n_layer);
            for (int il = 0; il < model_.cfg.n_layer; ++il)
                fprintf(stderr, "gpu L%02d plan->wait %.1f us  wait %.1f us\n", il, (ts[il][1] - ts[il][0]) / 1e3,
                        (ts[il][2] - ts[il][1]) / 1e3);
        }
    } else {
        enqueue_forward(R);
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    if (route_log_ && opt_.use_graphs && !dump_all) write_route_log(R, wins_[0].pos0, wins_[0].commit, t0, now_ms());
    for (const Win & w : wins_) w.q->pending_T = w.commit ? 0 : w.T;
    if (opt_.adapt_every > 0 && ++fwd_count_ % opt_.adapt_every == 0) cache_.adapt(counts_, st_, opt_.adapt_swaps);
    times.total_ms += now_ms() - t0;
    times.calls++;
}

// The MTP layer's K/V for rows just processed: each row pairs with the token after it; the last row waits
// for its successor (the next chunk's first token, or the first generated token).
void Engine::mtp_feed(const float * R_rows, const int32_t * tokens, int n, int pos0) {
    if (!mtp_.loaded()) return;
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
    return b;
}

void Engine::prefill(const std::vector<int32_t> & tokens) {
    join_commit();
    size_t i = 0;
    ensure_ctx(pos() + (int) tokens.size() + kMaxWindow);
    // the chunk layout for this read, borrowing VRAM from the expert cache meanwhile: a short turn keeps the
    // small layout and stage; longer reads get bigger chunks and a stage that holds a whole layer's outside experts
    const int n = (int) tokens.size();
    if (pf_base_ > 0) {
        const int lay = (pf_big_ > pf_base_ && n >= 2 * pf_base_) ? pf_big_ : n > pf_small_ ? pf_base_ : pf_small_;
        prefill_layout(lay);
        pf_full_ = n >= opt_.stage_full_min || std::min(lay, n) >= opt_.prefetch_min;
        prefill_staging_ensure(pf_full_);
    }
    while (i < tokens.size()) {
        const size_t left = tokens.size() - i;
        const int pos0 = pos();
        if (pf_max_ > 0 && (int) left >= opt_.prefill_min) {
            const int N = (int) std::min<size_t>(pf_max_, left);
            prefill_chunk(tokens.data() + i, N);
            mtp_feed(pf_.res, tokens.data() + i, N, pos0);
            i += N;
        } else {
            const int T = (int) std::min<size_t>(kMaxWindow, left);
            forward(tokens.data() + i, T);
            mtp_feed(res_, tokens.data() + i, T, pos0);
            i += T;
        }
        if (on_prefill_progress) on_prefill_progress(i, tokens.size());
    }
    const double r0 = now_ms();
    rebalance();
    last_rebalance_ms = now_ms() - r0;
}

int Engine::argmax(int t) {
    argmax_rows(logits_.p + (size_t) t * model_.cfg.n_vocab, 1, model_.cfg.n_vocab, argmax_dev_, st_);
    int32_t r;
    CUDA_CHECK(cudaMemcpyAsync(&r, argmax_dev_.p, 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    return r;
}

void Engine::argmax_all(int T, int32_t * out, int row0) {
    argmax_rows(logits_.p + (size_t) row0 * model_.cfg.n_vocab, T, model_.cfg.n_vocab, argmax_dev_, st_);
    CUDA_CHECK(cudaMemcpyAsync(out, argmax_dev_.p, T * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void Engine::topk_rows_host(int row0, int T, int K, int32_t * ids, float * vals) {
    if (K > kTopkMax || T > kMaxWindow) throw std::runtime_error("topk_rows_host: too many");
    if (!topk_ids_.p) {
        topk_ids_.alloc((size_t) kMaxWindow * kTopkMax);
        topk_vals_.alloc((size_t) kMaxWindow * kTopkMax);
    }
    const int V = model_.cfg.n_vocab;
    topk_rows(logits_.p + (size_t) row0 * V, T, V, K, topk_ids_, topk_vals_, st_);
    CUDA_CHECK(cudaMemcpyAsync(ids, topk_ids_.p, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(vals, topk_vals_.p, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void Engine::logits_rows_host(int T, float * out, int row0) {
    CUDA_CHECK(cudaMemcpy(out, logits_.p + (size_t) row0 * model_.cfg.n_vocab, (size_t) T * model_.cfg.n_vocab * 4,
                          cudaMemcpyDeviceToHost));
}

void Engine::set_mtp_pending(const float * R_row_dev, int cell) {
    if (!mtp_.loaded()) return;
    if (!cur_->mtp_R) CUDA_CHECK(cudaMalloc(&cur_->mtp_R, (size_t) model_.cfg.hc_dim() * 4));
    CUDA_CHECK(cudaMemcpyAsync(cur_->mtp_R, R_row_dev, (size_t) model_.cfg.hc_dim() * 4, cudaMemcpyDeviceToDevice, st_));
    cur_->mtp_cell = cell;
}

std::vector<float> Engine::logits_host(int t) {
    std::vector<float> v(model_.cfg.n_vocab);
    CUDA_CHECK(cudaMemcpy(v.data(), logits_.p + (size_t) t * model_.cfg.n_vocab, v.size() * 4,
                          cudaMemcpyDeviceToHost));
    return v;
}

}  // namespace bnk
