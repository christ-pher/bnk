#include "engine/mtp.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "ggml.h"
#include "kernels/ops.h"
#include "kernels/qsa.h"

namespace bnk {

MtpLayer::~MtpLayer() {
    for (auto & g : graphs_) for (auto & x : g) if (x) cudaGraphExecDestroy(x);
    for (void * p : allocs_) cudaFree(p);
    if (msg_) cudaFreeHost(msg_);
    if (h_io_) cudaFreeHost(h_io_);
    if (h_prob_) cudaFreeHost(h_prob_);
}

namespace {

// The GGUF stores Gemma-style norms (scale = 1 + w): fold the 1 in.
std::vector<float> plus_one(const TensorRef & t) {
    std::vector<float> v(t.ne[0] * t.nrows());
    memcpy(v.data(), t.data, v.size() * 4);
    for (auto & x : v) x += 1.f;
    return v;
}

}  // namespace

void MtpLayer::ensure_ctx(int cells) {
    kv_k_.ensure((size_t) cells * kv_cell_bytes_);
    kv_v_.ensure((size_t) cells * kv_cell_bytes_);
    if (sparse_) {
        kv_raw_.ensure((size_t) cells * qsh_.id * sizeof(half));
        kv_pool_.ensure((size_t) (cells / qsh_.ratio + 1) * qsh_.id * sizeof(float));
    }
}

size_t MtpLayer::ctx_bytes_needed(int cells) const {
    auto more = [](const ElasticBuf & e, size_t bytes) {
        const size_t b = vmem_round(bytes);
        return e.reserved() && b > e.mapped() ? b - e.mapped() : 0;
    };
    size_t n = more(kv_k_, (size_t) cells * kv_cell_bytes_) + more(kv_v_, (size_t) cells * kv_cell_bytes_);
    if (sparse_)
        n += more(kv_raw_, (size_t) cells * qsh_.id * sizeof(half)) +
             more(kv_pool_, (size_t) (cells / qsh_.ratio + 1) * qsh_.id * sizeof(float));
    return n;
}

void MtpLayer::release_ctx(int cells) {
    kv_k_.shrink_to((size_t) cells * kv_cell_bytes_);
    kv_v_.shrink_to((size_t) cells * kv_cell_bytes_);
    if (sparse_) {
        kv_raw_.shrink_to((size_t) cells * qsh_.id * sizeof(half));
        kv_pool_.shrink_to((size_t) (cells / qsh_.ratio + 1) * qsh_.id * sizeof(float));
    }
}

void MtpLayer::load(const std::string & path, const Model & main, int max_ctx, cudaStream_t st, bool verbose,
                    const std::string & draft_vocab, VramBudget * budget) {
    main_ = &main;
    st_ = st;
    max_ctx_ = max_ctx;
    GgufModel g;
    g.open(path);
    const Config & c = main.cfg;
    const int E = c.n_embd, HC = c.hc_dim(), W = kMaxWindow;

    auto dev_f32 = [&](const std::vector<float> & v) {
        float * d;
        CUDA_CHECK(cudaMalloc(&d, v.size() * 4));
        CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        allocs_.push_back(d);
        vram_ += v.size() * 4;
        return d;
    };
    // BF16 matrices become Q8_0 (R layout) unless keep_float
    auto mat = [&](const std::string & name, bool keep_float = false) {
        const TensorRef & t = g.get(name);
        void * d = nullptr;
        QMat m;
        if (t.type == GGML_TYPE_BF16 && !keep_float && t.ne[0] % 32 == 0) {
            const int64_t rows = t.nrows(), cols = t.ne[0];
            std::vector<float> f((size_t) rows * cols);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) t.data, f.data(), rows * cols);
            const size_t rb = ggml_row_size(GGML_TYPE_Q8_0, cols);
            std::vector<uint8_t> q(rb * rows);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, rows, cols, nullptr);
            m = upload_matrix(q.data(), GGML_TYPE_Q8_0, rows, cols, rb, true, &d);
        } else {
            m = upload_matrix(t.data, (int) t.type, t.nrows(), t.ne[0], t.row_bytes(), true, &d);
        }
        allocs_.push_back(d);
        vram_ += m.row_bytes * m.rows;
        return m;
    };
    // rows [r0, r0 + n) of a BF16 matrix, as Q8_0 (R layout)
    auto mat_rows = [&](const std::string & name, int64_t r0, int64_t n) {
        const TensorRef & t = g.get(name);
        if (t.type != GGML_TYPE_BF16) throw std::runtime_error(name + ": expected BF16");
        const int64_t cols = t.ne[0];
        std::vector<float> f((size_t) n * cols);
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) t.data + r0 * cols, f.data(), n * cols);
        const size_t rb = ggml_row_size(GGML_TYPE_Q8_0, cols);
        std::vector<uint8_t> q(rb * n);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, n, cols, nullptr);
        void * d = nullptr;
        QMat m = upload_matrix(q.data(), GGML_TYPE_Q8_0, n, cols, rb, true, &d);
        allocs_.push_back(d);
        vram_ += m.row_bytes * m.rows;
        return m;
    };
    auto norm = [&](const std::string & name) {
        QMat m;
        m.data = dev_f32(plus_one(g.get(name)));
        m.type = 0;
        return m;
    };
    const std::string L = "mtp.layers.0.";
    fc_emb_ = mat("mtp.fc_embedding.weight");
    fc_hid_ = mat("mtp.fc_hidden.weight");
    w_emb_ = dev_f32(plus_one(g.get("mtp.pre_fc_norm_embedding.weight")));
    w_hid_ = dev_f32(plus_one(g.get("mtp.pre_fc_norm_hidden.weight")));
    auto hcw = [&](const std::string & p, bool inject) {
        HcWeights h;
        h.norm = norm(p + "hc_norm.weight");
        h.down = mat(p + "input_mix_weight_down.weight");
        h.up = mat(p + "input_mix_weight_up.weight");
        if (inject) h.inject = mat(p + "block_inject_weight.weight", true);
        return h;
    };
    hc_attn_ = hcw(L + "attn_hyper_connection.", true);
    hc_mlp_ = hcw(L + "mlp_hyper_connection.", true);
    hc_mix_ = hcw("mtp.hyper_connection_mixer.", false);
    wq_ = mat(L + "self_attn.q_proj.weight");
    wk_ = mat(L + "self_attn.k_proj.weight");
    wv_ = mat(L + "self_attn.v_proj.weight");
    wo_ = mat(L + "self_attn.o_proj.weight");
    q_norm_ = dev_f32(plus_one(g.get(L + "self_attn.q_norm.weight")));
    k_norm_ = dev_f32(plus_one(g.get(L + "self_attn.k_norm.weight")));
    // the layer's sparse-attention indexer: one projection whose first ih * id rows are the queries, the rest
    // the key (as llama.cpp's converter splits it); zero-centred norms
    if (g.find(L + "self_attn.indexer.index_qk_proj.weight") && c.idx_heads > 0 && !getenv("BNK_MTP_DENSE")) {
        const int nq = c.idx_heads * c.idx_dim;
        idx_q_ = mat_rows(L + "self_attn.indexer.index_qk_proj.weight", 0, nq);
        idx_k_ = mat_rows(L + "self_attn.indexer.index_qk_proj.weight", nq, c.idx_dim);
        idx_q_norm_ = dev_f32(plus_one(g.get(L + "self_attn.indexer.q_layernorm.weight")));
        idx_k_norm_ = dev_f32(plus_one(g.get(L + "self_attn.indexer.k_layernorm.weight")));
        int r = 0;
        for (int il = 0; il < c.n_layer; ++il) r = std::max(r, c.compress_ratio[il]);
        qsh_ = QsaShape{c.n_head, c.n_head_kv, c.head_dim, c.idx_heads, c.idx_dim, r, r ? c.idx_top_k / r : 0, c.n_rot,
                        c.rope_base, c.rms_eps};
        sparse_ = r > 0;
    }
    router_ = mat(L + "mlp.gate.weight", true);
    sh_gate_ = mat(L + "mlp.shared_expert.gate_proj.weight");
    sh_up_ = mat(L + "mlp.shared_expert.up_proj.weight");
    sh_down_ = mat(L + "mlp.shared_expert.down_proj.weight");
    sh_gate_inp_ = mat(L + "mlp.shared_expert_gate.weight", true);

    // all routed experts resident: blob = [gate rows | up rows | down rows], as the main cache
    const TensorRef & gu = g.get(L + "mlp.experts.gate_up_proj");
    const TensorRef & dn = g.get(L + "mlp.experts.down_proj");
    const int n_exp = (int) gu.ne[2], F2 = (int) gu.ne[1];
    if (F2 != 2 * c.n_ff_exp || gu.ne[0] != E || dn.ne[1] != E || n_exp != c.n_expert)
        throw std::runtime_error("mtp: unexpected expert shapes");
    const size_t gub = gu.nbytes / n_exp, dnb = dn.nbytes / n_exp;
    const size_t blob = (gub + dnb + 63) / 64 * 64;
    uint8_t * ex;
    CUDA_CHECK(cudaMalloc(&ex, blob * n_exp));
    allocs_.push_back(ex);
    vram_ += blob * n_exp;
    {
        std::vector<uint8_t> h(blob * n_exp, 0);
        for (int e = 0; e < n_exp; ++e) {
            memcpy(h.data() + e * blob, gu.data + e * gub, gub);
            memcpy(h.data() + e * blob + gub, dn.data + e * dnb, dnb);
        }
        CUDA_CHECK(cudaMemcpy(ex, h.data(), h.size(), cudaMemcpyHostToDevice));
    }
    std::vector<int32_t> ident(n_exp);
    for (int e = 0; e < n_exp; ++e) ident[e] = e;
    CUDA_CHECK(cudaMalloc(&slot_id_, n_exp * 4));
    allocs_.push_back(slot_id_);
    CUDA_CHECK(cudaMemcpy(slot_id_, ident.data(), n_exp * 4, cudaMemcpyHostToDevice));
    moe_.slot_of = slot_id_;
    moe_.base = ex;
    moe_.blob = blob;
    moe_.gate_bytes = gub / 2;
    moe_.up_bytes = gub / 2;
    moe_.gate_type = gu.type;
    moe_.down_type = dn.type;
    moe_.grow = ggml_row_size(gu.type, E);
    moe_.drow = ggml_row_size(dn.type, c.n_ff_exp);

    // buffers
    const int F = c.n_ff_exp;
    Rin_.alloc(W * HC); emb_.alloc(W * E); en_.alloc(W * E); e2_.alloc(W * E); hn_.alloc(W * HC); h2_.alloc(W * HC);
    R_.alloc(W * HC); xn_.alloc(W * HC); lo_.alloc(W * c.hc_rank); gpre_.alloc(W * HC); mixed_.alloc(W * E);
    inj_.alloc(W * c.hc); inj2_.alloc(W * c.hc);
    qfull_.alloc(W * c.n_head * c.head_dim * 2); k_.alloc(W * c.n_head_kv * c.head_dim);
    v_.alloc(W * c.n_head_kv * c.head_dim); q_.alloc(W * c.n_head * c.head_dim); attn_o_.alloc(c.n_head * c.head_dim);
    attn_scratch_.alloc(attention_scratch_floats(1, c.n_head, c.head_dim, max_ctx));
    bo_.alloc(E); rlog_.alloc(c.n_expert); rw_.alloc(c.n_expert_used); ids_.alloc(c.n_expert_used);
    sg_.alloc(c.n_ff_shexp); su_.alloc(c.n_ff_shexp); sh_.alloc(c.n_ff_shexp); sgate_.alloc(1); shared_.alloc(E);
    y_.alloc(E); sample_.alloc(E); logits_.alloc(c.n_vocab); prob_.alloc(2);
    tok_dev_.alloc(W); pos_dev_.alloc(4); out_dev_.alloc(2);
    actq_.alloc(W * 4 * HC); actd_.alloc(W * 4 * HC / 32);
    act_.q = actq_; act_.d = actd_;
    // K/V cells: address space for the whole context, mapped as it grows (ensure_ctx)
    const size_t kv = (size_t) max_ctx * c.n_head_kv * c.head_dim;
    kv_k_.reserve(kv * sizeof(half), budget, "MTP KV");
    kv_v_.reserve(kv * sizeof(half), budget, "MTP KV");
    kc_.view(kv_k_.as<half>(), kv);
    vc_.view(kv_v_.as<half>(), kv);
    if (sparse_) {
        max_blocks_ = max_ctx / qsh_.ratio + 1;
        kv_raw_.reserve((size_t) max_ctx * c.idx_dim * sizeof(half), budget, "MTP indexer keys");
        kv_pool_.reserve((size_t) max_blocks_ * c.idx_dim * sizeof(float), budget, "MTP indexer keys");
        kraw_.view(kv_raw_.as<half>(), (size_t) max_ctx * c.idx_dim);
        pooled_.view(kv_pool_.as<float>(), (size_t) max_blocks_ * c.idx_dim);
        ik_.alloc((size_t) W * c.idx_dim);
        iq_.alloc((size_t) W * c.idx_heads * c.idx_dim);
        scores_.alloc((size_t) max_blocks_);
        sel_.alloc((size_t) qsh_.top_blocks);
        nsel_.alloc(1);
    }
    hits_buf_.alloc(sizeof(HitList));
    gu_buf_.alloc((size_t) kMaxRouted * W * 2 * F);
    hq_buf_.alloc((size_t) kMaxRouted * W * F);
    hd_buf_.alloc((size_t) kMaxRouted * W * F / 32);
    part_buf_.alloc((size_t) kMaxRouted * W * E);
    moes_.hits = (HitList *) hits_buf_.p;
    moes_.gu = gu_buf_; moes_.hq = hq_buf_; moes_.hd = hd_buf_; moes_.part = part_buf_;
    seq_dev_.alloc(1);
    CUDA_CHECK(cudaHostAlloc((void **) &msg_, MoeMsg::bytes(E), cudaHostAllocMapped));
    memset((void *) msg_, 0, MoeMsg::bytes(E));
    CUDA_CHECK(cudaHostAlloc((void **) &h_io_, 64, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &h_prob_, 64, cudaHostAllocMapped));
    kv_cell_bytes_ = (size_t) c.n_head_kv * c.head_dim * sizeof(half);
    // optional draft vocabulary: the head rows of a token subset
    if (!draft_vocab.empty()) {
        FILE * f = fopen(draft_vocab.c_str(), "rb");
        if (f) {
            std::vector<int32_t> ids;
            int32_t v;
            while (fread(&v, 4, 1, f) == 1)
                if (v >= 0 && v < c.n_vocab) ids.push_back(v);
            fclose(f);
            const QMat & H = main.output;
            if (!ids.empty() && H.row_bytes % 16 == 0) {
                n_dvocab_ = (int) ids.size();
                CUDA_CHECK(cudaMalloc(&dvocab_, ids.size() * 4));
                allocs_.push_back(dvocab_);
                CUDA_CHECK(cudaMemcpy(dvocab_, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
                void * d;
                CUDA_CHECK(cudaMalloc(&d, H.row_bytes * ids.size()));
                allocs_.push_back(d);
                gather_bytes_rows((const uint8_t *) H.data, H.row_bytes, dvocab_, n_dvocab_, (uint8_t *) d, nullptr);
                CUDA_CHECK(cudaDeviceSynchronize());
                dhead_ = H;
                dhead_.data = d;
                dhead_.rows = n_dvocab_;
                vram_ += H.row_bytes * ids.size();
            }
        }
    }
    loaded_ = true;
    if (verbose)
        fprintf(stderr, "bnk: MTP draft layer: %.2f GiB of VRAM, draft head over %s tokens\n", vram_ / 1073741824.0,
                n_dvocab_ ? std::to_string(n_dvocab_).c_str() : "all");
}

void MtpLayer::hc_pre(const HcWeights & w, const float * res, int T, bool inject, float * mixed, float * inj) {
    const Config & c = main_->cfg;
    const int E = c.n_embd, HC = c.hc_dim();
    hc_norm(res, (const float *) w.norm.data, xn_, T, c.hc, E, c.rms_eps, st_);
    gemv_auto(w.down, xn_, HC, T, lo_, c.hc_rank, false, act_, st_);
    if (inject) gemv_auto(w.inject, xn_, HC, T, inj, c.hc, false, act_, st_);
    silu_scale(lo_, (int64_t) T * c.hc_rank, 1.f / c.hc, st_);
    gemv_auto(w.up, lo_, c.hc_rank, T, gpre_, HC, false, act_, st_);
    hc_mix(xn_, gpre_, mixed, T, c.hc, E, st_);
}

int MtpLayer::run(const float * R_rows, const int32_t * next_tokens, int n, int cell0, bool draft, float * prob) {
    if (n < 1 || n > kMaxWindow) throw std::runtime_error("mtp: bad row count");
    const int HC = main_->cfg.hc_dim();
    // no stream sync here: the pinned inputs below are read only by this layer's graphs, and forward() waits for
    // each of those; whatever else is queued (the main model's commit) runs first, in stream order
    CUDA_CHECK(cudaMemcpyAsync(Rin_.p, R_rows, (size_t) n * HC * 4, cudaMemcpyDeviceToDevice, st_));
    memcpy(h_io_ + 8, next_tokens, n * 4);
    return forward(n, cell0, draft, prob);
}

int MtpLayer::step(int32_t tok, int cell, float * prob) {
    const int HC = main_->cfg.hc_dim();
    CUDA_CHECK(cudaMemcpyAsync(Rin_.p, R_.p, (size_t) HC * 4, cudaMemcpyDeviceToDevice, st_));
    h_io_[8] = tok;
    return forward(1, cell, true, prob);
}

int MtpLayer::forward(int n, int cell0, bool draft, float * prob) {
    const double t0 = now_ms();
    if (cell0 + n > max_ctx_) throw std::runtime_error("mtp: context full");
    h_io_[16] = cell0;
    h_io_[17] = cell0 + n - 1;
    cudaGraphExec_t & g = graphs_[n][draft ? 1 : 0];
    if (!g) {
        cudaGraph_t gr;
        CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
        enqueue(n, draft);
        CUDA_CHECK(cudaStreamEndCapture(st_, &gr));
        CUDA_CHECK(cudaGraphInstantiate(&g, gr, 0));
        cudaGraphDestroy(gr);
    }
    CUDA_CHECK(cudaGraphLaunch(g, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    ms += now_ms() - t0;
    if (!draft) return -1;
    if (prob) *prob = h_prob_[0];
    ++calls;
    return h_io_[0];
}

// The layer for n rows (window inputs read from pinned memory, so the graph replays with new values).
void MtpLayer::enqueue(int n, bool draft) {
    const Config & c = main_->cfg;
    const int E = c.n_embd, HC = c.hc_dim(), H = c.n_head, Hkv = c.n_head_kv, D = c.head_dim;
    CUDA_CHECK(cudaMemcpyAsync(pos_dev_.p, h_io_ + 16, 8, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(tok_dev_.p, h_io_ + 8, n * 4, cudaMemcpyHostToDevice, st_));

    // the two input branches
    dequant_gather(main_->tok_embd, tok_dev_, n, emb_, st_);
    rmsnorm_rows(emb_, w_emb_, en_, n, E, E, E, c.rms_eps, st_);
    gemv_auto(fc_emb_, en_, E, n, e2_, E, false, act_, st_);
    rmsnorm_rows(Rin_, w_hid_, hn_, n, HC, HC, HC, c.rms_eps, st_);
    for (int r0 = 0; r0 < n * c.hc; r0 += kMaxWindow) {
        const int rr = std::min(kMaxWindow, n * c.hc - r0);
        gemv_auto(fc_hid_, hn_.p + (size_t) r0 * E, E, rr, h2_.p + (size_t) r0 * E, E, false, act_, st_);
    }
    copy_f32(R_, h2_, (int64_t) n * HC, st_);
    add_bcast_streams(R_, e2_, n, c.hc, E, st_);
    // attention hyper-connection, K/V for every row
    hc_pre(hc_attn_, R_, n, true, mixed_, inj_);
    gemv_auto(wq_, mixed_, E, n, qfull_, H * D * 2, false, act_, st_);
    gemv_auto(wk_, mixed_, E, n, k_, Hkv * D, false, act_, st_);
    gemv_auto(wv_, mixed_, E, n, v_, Hkv * D, false, act_, st_);
    attn_prep(qfull_, k_, v_, q_norm_, k_norm_, q_, kc_, vc_, n, H, Hkv, D, c.n_rot, c.rope_base, pos_dev_.p,
              c.rms_eps, st_);
    if (sparse_) {  // index keys of every row (pooled blocks that end in the window), the draft row's query
        gemv_auto(idx_k_, mixed_, E, n, ik_, c.idx_dim, false, act_, st_);
        qsa_store_keys(ik_, kraw_, n, c.idx_dim, pos_dev_.p, st_);
        qsa_pool(kraw_, idx_k_norm_, pooled_, n, qsh_, pos_dev_.p, st_);
    }
    if (!draft) return;
    // the rest of the layer on the last row
    const int L = n - 1;
    float * RL = R_.p + (size_t) L * HC;
    if (sparse_) {
        // the draft row attends to the blocks its indexer selects, plus its tail (dense while the context is short)
        const float * mL = mixed_.p + (size_t) L * E;
        gemv_auto(idx_q_, mL, E, 1, iq_, c.idx_heads * c.idx_dim, false, act_, st_);
        qsa_queries(iq_, idx_q_norm_, 1, qsh_, pos_dev_.p + 1, st_);
        qsa_select(iq_, pooled_, scores_, max_blocks_, sel_, nsel_, 1, qsh_, pos_dev_.p + 1, 0, st_);
        qsa_attention(q_.p + (size_t) L * H * D, kc_, vc_, qfull_.p + (size_t) L * H * D * 2, sel_, nsel_, attn_o_, 1,
                      qsh_, pos_dev_.p + 1, 1.f / sqrtf((float) D), attn_scratch_, st_);
    } else {
        attention(q_.p + (size_t) L * H * D, kc_, vc_, qfull_.p + (size_t) L * H * D * 2, attn_o_, 1, H, Hkv, D,
                  pos_dev_.p + 1, 1.f / sqrtf((float) D), attn_scratch_, st_);
    }
    gemv_auto(wo_, attn_o_, H * D, 1, bo_, E, false, act_, st_);
    hc_combine(RL, bo_, inj_.p + (size_t) L * c.hc, 1, c.hc, E, st_);
    hc_pre(hc_mlp_, RL, 1, true, mixed_, inj2_);
    // MoE: every expert resident
    gemv_auto(router_, mixed_, E, 1, rlog_, c.n_expert, false, act_, st_);
    route_topk(rlog_, 1, c.n_expert, c.n_expert_used, ids_, rw_, c.expert_weights_scale, st_);
    moe_plan(ids_, rw_, 1, c.n_expert_used, moe_, moes_, msg_, mixed_, E, seq_dev_, nullptr, st_);
    quantize_act(mixed_, E, 1, E, act_, st_);
    moe_hits(moe_, moes_, act_, 1, c.n_expert_used, E, c.n_ff_exp, st_);
    gemv_auto(sh_gate_, mixed_, E, 1, sg_, c.n_ff_shexp, false, act_, st_);
    gemv_auto(sh_up_, mixed_, E, 1, su_, c.n_ff_shexp, false, act_, st_);
    silu_mul(sg_, su_, sh_, c.n_ff_shexp, st_);
    gemv_auto(sh_down_, sh_, c.n_ff_shexp, 1, shared_, E, false, act_, st_);
    gemv_auto(sh_gate_inp_, mixed_, E, 1, sgate_, 1, false, act_, st_);
    sigmoid_inplace(sgate_, 1, st_);
    moe_reduce(moes_, msg_, shared_, sgate_, y_, 1, c.n_expert_used, E, st_);
    hc_combine(RL, y_, inj2_, 1, c.hc, E, st_);
    // keep the residual for the next step, then the final mixer and the (draft) head
    if (L > 0) copy_f32(R_, RL, HC, st_);
    hc_pre(hc_mix_, R_, 1, false, sample_, nullptr);
    const QMat & head = n_dvocab_ ? dhead_ : main_->output;
    gemv_auto(head, sample_, E, 1, logits_, (int) head.rows, false, act_, st_);
    argmax_prob(logits_, (int) head.rows, out_dev_, prob_, st_);
    if (n_dvocab_) map_id(out_dev_, dvocab_, st_);
    CUDA_CHECK(cudaMemcpyAsync(h_io_, out_dev_.p, 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(h_prob_, prob_.p, 4, cudaMemcpyDeviceToHost, st_));
}

}  // namespace bnk
