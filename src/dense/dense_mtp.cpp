#include "dense/dense_mtp.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "dense/dense_attn.h"
#include "dense/dense_ops.h"
#include "ggml.h"
#include "kernels/ops.h"
#include "kernels/rfmt.h"

namespace bnk {

DenseMtp::~DenseMtp() {
    for (auto & sl : slots_)
        for (auto & g : sl.graphs) for (auto & x : g) if (x) cudaGraphExecDestroy(x);
    for (auto & kv : mgraphs_) if (kv.second.exec) cudaGraphExecDestroy(kv.second.exec);
    for (void * p : allocs_) cudaFree(p);
    if (h_multi_) cudaFreeHost(h_multi_);
    if (h_mprob_) cudaFreeHost(h_mprob_);
    if (h_io_) cudaFreeHost(h_io_);
    if (h_prob_) cudaFreeHost(h_prob_);
}

static const float * F(const QMat & m) { return (const float *) m.data; }

void DenseMtp::ensure_ctx(int cells) {
    cur_->kv_k.ensure((size_t) cells * kv_cell_bytes_);
    cur_->kv_v.ensure((size_t) cells * kv_cell_bytes_);
}

size_t DenseMtp::ctx_bytes_needed(int cells) const {
    auto more = [](const ElasticBuf & e, size_t bytes) {
        const size_t b = vmem_round(bytes);
        return e.reserved() && b > e.mapped() ? b - e.mapped() : 0;
    };
    return more(cur_->kv_k, (size_t) cells * kv_cell_bytes_) + more(cur_->kv_v, (size_t) cells * kv_cell_bytes_);
}

void DenseMtp::release_ctx(int cells) {
    cur_->kv_k.shrink_to((size_t) cells * kv_cell_bytes_);
    cur_->kv_v.shrink_to((size_t) cells * kv_cell_bytes_);
}

void DenseMtp::load(const DenseModel & main, int max_ctx, cudaStream_t st, bool verbose, const std::string & draft_vocab,
                    VramBudget * budget, int slots) {
    if (!main.mtp.present) throw std::runtime_error("mtp: the model has no MTP block");
    main_ = &main;
    st_ = st;
    max_ctx_ = max_ctx;
    CUDA_CHECK(cudaGetDevice(&dev_));
    const Config & c = main.cfg;
    const int E = c.n_embd, W = kMaxWindow, FF = c.n_ff;
    Rin_.alloc(W * E); emb_.alloc(W * E); cat_.alloc((size_t) W * 2 * E); R_.alloc(W * E); xn_.alloc(W * E);
    qfull_.alloc(W * c.n_head * c.head_dim * 2); k_.alloc(W * c.n_head_kv * c.head_dim);
    v_.alloc(W * c.n_head_kv * c.head_dim); q_.alloc(W * c.n_head * c.head_dim); attn_o_.alloc(c.n_head * c.head_dim);
    attn_scratch_.alloc(std::max(attention_scratch_floats(1, c.n_head, c.head_dim, max_ctx),
                                 attention_decode_tc_scratch_floats(1, c.n_head, c.head_dim)));
    bo_.alloc(E); sg_.alloc(FF); su_.alloc(FF); y_.alloc(E); sample_.alloc(E); logits_.alloc(c.n_vocab); prob_.alloc(2);
    tok_dev_.alloc(W); pos_dev_.alloc(4); out_dev_.alloc(2);
    const int64_t maxcols = std::max<int64_t>(2 * E, FF);
    actq_.alloc(W * maxcols); actd_.alloc(W * maxcols / 32); acts_.alloc(W * maxcols / 16); acth_.alloc(W * maxcols);
    act_.q = actq_; act_.d = actd_; act_.s = acts_; act_.h = acth_;
    // K/V cells per conversation slot: address space for the whole context, mapped as it grows (ensure_ctx)
    kv_cell_bytes_ = (size_t) c.n_head_kv * kv_row_halves(c.head_dim) * sizeof(half);
    const size_t kv = (size_t) max_ctx * c.n_head_kv * kv_row_halves(c.head_dim);
    slots_ = std::vector<Slot>(std::max(1, slots));
    for (Slot & sl : slots_) {
        sl.kv_k.reserve(kv * sizeof(half), budget, "MTP KV");
        sl.kv_v.reserve(kv * sizeof(half), budget, "MTP KV");
        sl.kc.view(sl.kv_k.as<half>(), kv);
        sl.vc.view(sl.kv_v.as<half>(), kv);
    }
    cur_ = &slots_[0];
    CUDA_CHECK(cudaHostAlloc((void **) &h_io_, 64, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &h_prob_, 64, cudaHostAllocMapped));
    // optional draft vocabulary: the head rows of a token subset, re-quantized to Q4_K (BNK_DRAFT_HEAD=TYPE another
    // type, =keep the model's own). The 248K-row head is the biggest read of a draft step: the subset halves it.
    if (!draft_vocab.empty()) {
        FILE * f = fopen(draft_vocab.c_str(), "rb");
        if (f) {
            std::vector<int32_t> ids;
            int32_t v;
            while (fread(&v, 4, 1, f) == 1)
                if (v >= 0 && v < c.n_vocab) ids.push_back(v);
            fclose(f);
            const QMat & H = main.output;
            const char * dh = getenv("BNK_DRAFT_HEAD") ? getenv("BNK_DRAFT_HEAD") : "q4_K";
            const TensorRef * ht = main.gguf.find("output.weight");
            int dt = -1;
            for (int t = 0; t < GGML_TYPE_COUNT; ++t)
                if (ggml_get_type_traits((ggml_type) t)->type_name && !strcmp(ggml_type_name((ggml_type) t), dh)) dt = t;
            if (dt >= 0 && (!r_supported(dt) || (ht && (int) ht->type == dt) || (ht && ht->ne[0] % 256))) dt = -1;
            if (!ids.empty() && ht && dt >= 0 && ggml_get_type_traits((ggml_type) ht->type)->to_float) {
                n_dvocab_ = (int) ids.size();
                CUDA_CHECK(cudaMalloc(&dvocab_, ids.size() * 4));
                allocs_.push_back(dvocab_);
                CUDA_CHECK(cudaMemcpy(dvocab_, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
                const int64_t cols = ht->ne[0], n = (int64_t) ids.size();
                const size_t rb = ggml_row_size((ggml_type) dt, cols);
                std::vector<uint8_t> q(rb * n);
                const auto * tr = ggml_get_type_traits((ggml_type) ht->type);
                const int nth = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
                std::vector<std::thread> th;
                for (int w = 0; w < nth; ++w)
                    th.emplace_back([&, w]() {
                        std::vector<float> fl((size_t) cols * 64);
                        for (int64_t r0 = (int64_t) w * 64; r0 < n; r0 += (int64_t) nth * 64) {
                            const int64_t m = std::min<int64_t>(64, n - r0);
                            for (int64_t i = 0; i < m; ++i)
                                tr->to_float(ht->data + (size_t) ids[r0 + i] * ht->row_bytes(), fl.data() + i * cols, cols);
                            ggml_quantize_chunk((ggml_type) dt, fl.data(), q.data() + (size_t) r0 * rb, 0, m, cols, nullptr);
                        }
                    });
                for (auto & t : th) t.join();
                void * d;
                dhead_ = upload_matrix(q.data(), dt, n, cols, rb, true, &d);
                allocs_.push_back(d);
                vram_ += dhead_.row_bytes * n;
            } else if (!ids.empty() && H.row_bytes % 16 == 0) {
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
        fprintf(stderr, "bnk: MTP draft block: %.2f GiB of VRAM beside the model's, draft head over %s tokens\n",
                vram_ / 1073741824.0, n_dvocab_ ? std::to_string(n_dvocab_).c_str() : "all");
}

int DenseMtp::run(const float * R_rows, const int32_t * next_tokens, int n, int cell0, bool draft, float * prob) {
    DeviceScope ds(dev_);
    if (n < 1 || n > kMaxWindow) throw std::runtime_error("mtp: bad row count");
    const int E = main_->cfg.n_embd;
    CUDA_CHECK(cudaMemcpyAsync(Rin_.p, R_rows, (size_t) n * E * 4, cudaMemcpyDeviceToDevice, st_));
    memcpy(h_io_ + 8, next_tokens, n * 4);
    return forward(n, cell0, draft, prob);
}

int DenseMtp::step(int32_t tok, int cell, float * prob) {
    DeviceScope ds(dev_);
    const int E = main_->cfg.n_embd;
    CUDA_CHECK(cudaMemcpyAsync(Rin_.p, R_.p, (size_t) E * 4, cudaMemcpyDeviceToDevice, st_));
    h_io_[8] = tok;
    return forward(1, cell, true, prob);
}

int DenseMtp::forward(int n, int cell0, bool draft, float * prob) {
    const double t0 = now_ms();
    if (cell0 + n > max_ctx_) throw std::runtime_error("mtp: context full");
    h_io_[16] = cell0;
    h_io_[17] = cell0 + n - 1;
    cudaGraphExec_t & g = cur_->graphs[n][draft ? 1 : 0];
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

// R_ [N][E] = eh_proj(cat(rms(embed(tok)) * enorm, rms(Rin) * hnorm)) for N rows (tokens in tok_dev_)
void DenseMtp::inputs(int N) {
    const Config & c = main_->cfg;
    const DenseMtpWeights & m = main_->mtp;
    const int E = c.n_embd;
    dequant_gather(main_->tok_embd, tok_dev_, N, emb_, st_);
    rmsnorm_rows(emb_, F(m.enorm), cat_, N, E, E, 2 * E, c.rms_eps, st_);
    rmsnorm_rows(Rin_, F(m.hnorm), cat_.p + E, N, E, E, 2 * E, c.rms_eps, st_);
    quantize_act(cat_, 2 * E, N, 2 * E, act_, st_);
    gemv(m.eh_proj, &act_, nullptr, 0, N, R_, E, false, st_);
}

// The block for n rows (window inputs read from pinned memory, so the graph replays with new values).
void DenseMtp::enqueue(int n, bool draft) {
    const Config & c = main_->cfg;
    const DenseLayer & L = main_->mtp.layer;
    const int E = c.n_embd, H = c.n_head, Hkv = c.n_head_kv, D = c.head_dim, FF = c.n_ff;
    CUDA_CHECK(cudaMemcpyAsync(pos_dev_.p, h_io_ + 16, 8, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(tok_dev_.p, h_io_ + 8, n * 4, cudaMemcpyHostToDevice, st_));
    inputs(n);
    // K/V for every row
    dense_norm_q(R_, nullptr, F(L.attn_norm), xn_, n, E, c.rms_eps, act_, st_);
    gemv(L.wq, &act_, nullptr, 0, n, qfull_, H * D * 2, false, st_);
    if (L.wkv.valid()) gemv(L.wkv, &act_, nullptr, 0, n, k_, Hkv * D, false, st_, GemvSplit{Hkv * D, v_, (int64_t) Hkv * D});
    else {
        gemv(L.wk, &act_, nullptr, 0, n, k_, Hkv * D, false, st_);
        gemv(L.wv, &act_, nullptr, 0, n, v_, Hkv * D, false, st_);
    }
    attn_prep(qfull_, k_, v_, F(L.q_norm), F(L.k_norm), q_, cur_->kc, cur_->vc, n, H, Hkv, D, c.n_rot, c.rope_base,
              pos_dev_.p, c.rms_eps, st_);
    if (!draft) return;
    // the rest of the block on the last row
    const int Lr = n - 1;
    float * RL = R_.p + (size_t) Lr * E;
    static const bool tc = !getenv("BNK_ATTN_TC") || atoi(getenv("BNK_ATTN_TC")) != 0;
    if (!(tc && attention_decode_tc(q_.p + (size_t) Lr * H * D, cur_->kc, cur_->vc, qfull_.p + (size_t) Lr * H * D * 2, attn_o_,
                                    1, H, Hkv, D, pos_dev_.p + 1, 1.f / sqrtf((float) D), attn_scratch_, st_)))
        attention(q_.p + (size_t) Lr * H * D, cur_->kc, cur_->vc, qfull_.p + (size_t) Lr * H * D * 2, attn_o_, 1, H, Hkv, D,
                  pos_dev_.p + 1, 1.f / sqrtf((float) D), attn_scratch_, st_);
    quantize_act(attn_o_, H * D, 1, H * D, act_, st_);
    gemv(L.wo, &act_, nullptr, 0, 1, bo_, E, false, st_);
    dense_norm_q(RL, bo_, F(L.post_norm), xn_, 1, E, c.rms_eps, act_, st_);
    if (L.gu.valid()) gemv(L.gu, &act_, nullptr, 0, 1, sg_, FF, false, st_, GemvSplit{FF, su_, FF});
    else {
        gemv(L.gate, &act_, nullptr, 0, 1, sg_, FF, false, st_);
        gemv(L.up, &act_, nullptr, 0, 1, su_, FF, false, st_);
    }
    quantize_act_silu_mul(sg_, su_, FF, 1, FF, act_, st_);
    gemv(L.down, &act_, nullptr, 0, 1, y_, E, false, st_);
    add_rows(RL, y_, E, st_);
    // keep the residual for the next step (row 0), then the final norm and the (draft) head
    if (Lr > 0) copy_f32(R_, RL, E, st_);
    rmsnorm_rows(R_, F(main_->mtp.head_norm), sample_, 1, E, E, E, c.rms_eps, st_);
    quantize_act(sample_, E, 1, E, act_, st_);
    const QMat & head = n_dvocab_ ? dhead_ : main_->output;
    gemv(head, &act_, nullptr, 0, 1, logits_, (int) head.rows, false, st_);
    argmax_prob(logits_, (int) head.rows, out_dev_, prob_, st_);
    if (n_dvocab_) map_id(out_dev_, dvocab_, st_);
    CUDA_CHECK(cudaMemcpyAsync(h_io_, out_dev_.p, 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(h_prob_, prob_.p, 4, cudaMemcpyDeviceToHost, st_));
}

// ---------------------------------------------------------------------------------------------- batched drafting
void DenseMtp::ensure_multi() {
    if (h_multi_) return;
    const Config & c = main_->cfg;
    const int E = c.n_embd, W = kMaxWindow, FF = c.n_ff;
    const int V = n_dvocab_ ? n_dvocab_ : c.n_vocab;
    Dres_.alloc(W * E); dattn_.alloc(W * c.n_head * c.head_dim); dbo_.alloc(W * E); dxn_.alloc(W * E);
    dsg_.alloc(W * FF); dsu_.alloc(W * FF); dy_.alloc(W * E); dsample_.alloc(W * E); dlogits_.alloc((size_t) W * V);
    dprob_.alloc(W); dout_.alloc(W); mpos_.alloc(2 * W);
    CUDA_CHECK(cudaHostAlloc((void **) &h_multi_, 32 * 4, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &h_mprob_, 8 * 4, cudaHostAllocMapped));
}

void DenseMtp::run_multi(const std::vector<RunItem> & items, int32_t * tok_out, float * prob_out) {
    DeviceScope ds(dev_);
    ensure_multi();
    const int E = main_->cfg.n_embd;
    std::vector<MItem> mi;
    int r = 0;
    for (const auto & it : items) {
        CUDA_CHECK(cudaMemcpyAsync(Rin_.p + (size_t) r * E, it.R_rows, (size_t) it.n * E * 4, cudaMemcpyDeviceToDevice, st_));
        memcpy(h_multi_ + r, it.tokens, it.n * 4);
        mi.push_back({it.slot, it.n, it.cell0});
        r += it.n;
    }
    if (r > kMaxWindow) throw std::runtime_error("mtp: more than kMaxWindow rows");
    forward_multi(mi, tok_out, prob_out);
}

void DenseMtp::step_multi(const std::vector<StepItem> & items, int32_t * tok_out, float * prob_out) {
    DeviceScope ds(dev_);
    ensure_multi();
    const int E = main_->cfg.n_embd;
    std::vector<MItem> mi;
    for (size_t i = 0; i < items.size(); ++i) {
        const StepItem & it = items[i];
        CUDA_CHECK(cudaMemcpyAsync(Rin_.p + i * E, Dres_.p + (size_t) it.prev * E, (size_t) E * 4, cudaMemcpyDeviceToDevice, st_));
        h_multi_[i] = it.tok;
        mi.push_back({multi_slots_.at(it.prev), 1, it.cell});
    }
    forward_multi(mi, tok_out, prob_out);
}

void DenseMtp::forward_multi(const std::vector<MItem> & items, int32_t * tok_out, float * prob_out) {
    const double t0 = now_ms();
    const int S = (int) items.size();
    if (S < 1 || S > kMaxWindow) throw std::runtime_error("mtp: bad batch");
    std::vector<int> key;
    for (int i = 0; i < S; ++i) {
        const MItem & it = items[i];
        if (it.cell0 + it.n > max_ctx_) throw std::runtime_error("mtp: context full");
        h_multi_[8 + 2 * i] = it.cell0;
        h_multi_[8 + 2 * i + 1] = it.cell0 + it.n - 1;
        key.push_back(it.slot);
        key.push_back(it.n);
    }
    MGraph & g = mgraphs_[key];
    if (!g.exec) {
        while (mgraphs_.size() > 48) {
            auto lru = mgraphs_.end();
            for (auto it = mgraphs_.begin(); it != mgraphs_.end(); ++it)
                if (it->second.exec && (lru == mgraphs_.end() || it->second.used < lru->second.used)) lru = it;
            if (lru == mgraphs_.end()) break;
            cudaGraphExecDestroy(lru->second.exec);
            mgraphs_.erase(lru);
        }
        cudaGraph_t gr;
        CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
        enqueue_multi(items);
        CUDA_CHECK(cudaStreamEndCapture(st_, &gr));
        CUDA_CHECK(cudaGraphInstantiate(&g.exec, gr, 0));
        cudaGraphDestroy(gr);
    }
    g.used = ++mgraph_age_;
    CUDA_CHECK(cudaGraphLaunch(g.exec, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    multi_slots_.clear();
    for (const auto & it : items) multi_slots_.push_back(it.slot);
    for (int i = 0; i < S; ++i) {
        tok_out[i] = h_multi_[24 + i];
        prob_out[i] = h_mprob_[i];
    }
    ms += now_ms() - t0;
    calls += S;
}

void DenseMtp::enqueue_multi(const std::vector<MItem> & items) {
    const Config & c = main_->cfg;
    const DenseLayer & L = main_->mtp.layer;
    const int E = c.n_embd, H = c.n_head, Hkv = c.n_head_kv, D = c.head_dim, FF = c.n_ff;
    const int S = (int) items.size();
    int N = 0;
    std::vector<int> r0(S), last(S);
    for (int i = 0; i < S; ++i) {
        r0[i] = N;
        N += items[i].n;
        last[i] = N - 1;
    }
    CUDA_CHECK(cudaMemcpyAsync(mpos_.p, h_multi_ + 8, 2 * S * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(tok_dev_.p, h_multi_, N * 4, cudaMemcpyHostToDevice, st_));
    inputs(N);
    // K/V for every row, into each item's own slot
    dense_norm_q(R_, nullptr, F(L.attn_norm), xn_, N, E, c.rms_eps, act_, st_);
    gemv(L.wq, &act_, nullptr, 0, N, qfull_, H * D * 2, false, st_);
    if (L.wkv.valid()) gemv(L.wkv, &act_, nullptr, 0, N, k_, Hkv * D, false, st_, GemvSplit{Hkv * D, v_, (int64_t) Hkv * D});
    else {
        gemv(L.wk, &act_, nullptr, 0, N, k_, Hkv * D, false, st_);
        gemv(L.wv, &act_, nullptr, 0, N, v_, Hkv * D, false, st_);
    }
    for (int i = 0; i < S; ++i) {
        Slot & sl = slots_.at(items[i].slot);
        const size_t r = (size_t) r0[i];
        attn_prep(qfull_.p + r * H * D * 2, k_.p + r * Hkv * D, v_.p + r * Hkv * D, F(L.q_norm), F(L.k_norm),
                  q_.p + r * H * D, sl.kc, sl.vc, items[i].n, H, Hkv, D, c.n_rot, c.rope_base, mpos_.p + 2 * i,
                  c.rms_eps, st_);
    }
    // the draft rows (each item's last): residual gathered, attention each on its own slot
    for (int i = 0; i < S; ++i) {
        copy_f32(Dres_.p + (size_t) i * E, R_.p + (size_t) last[i] * E, E, st_);
        Slot & sl = slots_.at(items[i].slot);
        const size_t Lr = (size_t) last[i];
        static const bool tc = !getenv("BNK_ATTN_TC") || atoi(getenv("BNK_ATTN_TC")) != 0;
        if (!(tc && attention_decode_tc(q_.p + Lr * H * D, sl.kc, sl.vc, qfull_.p + Lr * H * D * 2, dattn_.p + (size_t) i * H * D,
                                        1, H, Hkv, D, mpos_.p + 2 * i + 1, 1.f / sqrtf((float) D), attn_scratch_, st_)))
            attention(q_.p + Lr * H * D, sl.kc, sl.vc, qfull_.p + Lr * H * D * 2, dattn_.p + (size_t) i * H * D, 1, H, Hkv, D,
                      mpos_.p + 2 * i + 1, 1.f / sqrtf((float) D), attn_scratch_, st_);
    }
    // the rest of the block on the draft rows together
    quantize_act(dattn_, H * D, S, H * D, act_, st_);
    gemv(L.wo, &act_, nullptr, 0, S, dbo_, E, false, st_);
    dense_norm_q(Dres_, dbo_, F(L.post_norm), dxn_, S, E, c.rms_eps, act_, st_);
    if (L.gu.valid()) gemv(L.gu, &act_, nullptr, 0, S, dsg_, FF, false, st_, GemvSplit{FF, dsu_, FF});
    else {
        gemv(L.gate, &act_, nullptr, 0, S, dsg_, FF, false, st_);
        gemv(L.up, &act_, nullptr, 0, S, dsu_, FF, false, st_);
    }
    quantize_act_silu_mul(dsg_, dsu_, FF, S, FF, act_, st_);
    gemv(L.down, &act_, nullptr, 0, S, dy_, E, false, st_);
    add_rows(Dres_, dy_, (int64_t) S * E, st_);
    rmsnorm_rows(Dres_, F(main_->mtp.head_norm), dsample_, S, E, E, E, c.rms_eps, st_);
    quantize_act(dsample_, E, S, E, act_, st_);
    const QMat & head = n_dvocab_ ? dhead_ : main_->output;
    gemv(head, &act_, nullptr, 0, S, dlogits_, (int) head.rows, false, st_);
    argmax_prob_rows(dlogits_, (int) head.rows, S, dout_, dprob_, st_);
    if (n_dvocab_) map_ids(dout_, dvocab_, S, st_);
    CUDA_CHECK(cudaMemcpyAsync(h_multi_ + 24, dout_.p, S * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(h_mprob_, dprob_.p, S * 4, cudaMemcpyDeviceToHost, st_));
}

}  // namespace bnk
