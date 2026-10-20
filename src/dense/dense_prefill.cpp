// Batched prompt processing for the dense engine: fp16 tensor-core GEMMs (cuBLAS, weights dequantized on the
// fly) for every projection, the DeltaNet recurrence and the attention over whole chunks.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "dense/dense_attn.h"
#include "dense/dense_engine.h"
#include "dense/dense_ops.h"
#include "kernels/ops.h"

namespace bnk {

static const float * F(const QMat & m) { return (const float *) m.data; }

void DenseEngine::prefill_alloc(int N) {
    const Config & c = model_.lc;
    const int E = c.n_embd, C = c.conv_channels(), VI = c.ssm_vheads * c.ssm_state, FF = c.n_ff;
    auto & p = pf_;
    p.x.alloc((size_t) N * E); p.res.alloc((size_t) N * E); p.xn.alloc((size_t) N * E); p.out.alloc((size_t) N * E);
    p.conv.alloc((size_t) (N + c.ssm_conv - 1) * C); p.co.alloc((size_t) N * C); p.z.alloc((size_t) N * VI);
    p.g.alloc((size_t) N * c.ssm_vheads); p.b.alloc((size_t) N * c.ssm_vheads); p.o.alloc((size_t) N * VI);
    p.n.alloc((size_t) N * VI);
    p.qfull.alloc((size_t) N * c.n_head * c.head_dim * 2); p.k.alloc((size_t) N * c.n_head_kv * c.head_dim);
    p.v.alloc((size_t) N * c.n_head_kv * c.head_dim); p.q.alloc((size_t) N * c.n_head * c.head_dim);
    p.ao.alloc((size_t) N * c.n_head * c.head_dim);
    p.sg.alloc((size_t) N * FF); p.su.alloc((size_t) N * FF);
    p.tok.alloc(N);
    // GEMM workspace: the largest matrix dequantized at once (gate / up / down: FF x E), and N x max K inputs
    size_t maxw = (size_t) FF * E;
    for (int il = l0_; il < l1_; ++il) {
        const DenseLayer & L = model_.layers[il];
        for (const QMat * m : {&L.wq, &L.wk, &L.wv, &L.wo, &L.wqkv, &L.wgate, &L.ssm_out, &L.gate, &L.up, &L.down})
            if (m->valid()) maxw = std::max(maxw, (size_t) m->rows * m->cols);
    }
    gemm_.init(st_, maxw, (size_t) N * std::max({(int64_t) E, (int64_t) VI, (int64_t) c.n_head * c.head_dim, (int64_t) FF}));
}

void DenseEngine::pf_gdn(int il, int N) {
    const Config & c = model_.lc;
    const DenseLayer & L = model_.layers[il];
    const int E = c.n_embd, C = c.conv_channels(), K = c.ssm_conv, S = c.ssm_state, nv = c.ssm_vheads;
    auto & p = pf_;
    CUDA_CHECK(cudaMemcpyAsync(p.conv.p, cur_->conv_buf[il].p, (size_t) (K - 1) * C * 4, cudaMemcpyDeviceToDevice, st_));
    f32_to_f16(p.xn, gemm_.xbuf(), (int64_t) N * E, st_);
    gemm_.run_h(L.wqkv, gemm_.xbuf(), E, N, p.conv.p + (size_t) (K - 1) * C, C);
    gemm_.run_h(L.wgate, gemm_.xbuf(), E, N, p.z, nv * S);
    gemm_.run_h(L.ssm_beta, gemm_.xbuf(), E, N, p.b, nv);
    gemm_.run_h(L.ssm_alpha, gemm_.xbuf(), E, N, p.g, nv);
    gdn_conv(p.conv, F(L.ssm_conv1d), p.co, N, C, K, st_);
    gdn_prep(p.co, N, C, c.ssm_groups, nv, S, p.g, p.b, F(L.ssm_dt), F(L.ssm_a), c.rms_eps, st_);
    gdn_recurrence(p.co, C, p.g, p.b, cur_->ssm_state[il], p.o, N, c.ssm_groups, nv, S, N, st_);
    gated_rmsnorm_silu(p.o, p.z, F(L.ssm_norm), p.n, N, nv, S, c.rms_eps, st_);
    gemm_.run(L.ssm_out, p.n, nv * S, N, p.out, E);
    CUDA_CHECK(cudaMemcpyAsync(cur_->conv_buf[il].p, p.conv.p + (size_t) N * C, (size_t) (K - 1) * C * 4,
                               cudaMemcpyDeviceToDevice, st_));
}

void DenseEngine::pf_attn(int il, int N) {
    const Config & c = model_.lc;
    const DenseLayer & L = model_.layers[il];
    const int E = c.n_embd, H = c.n_head, Hkv = c.n_head_kv, D = c.head_dim;
    auto & p = pf_;
    const int pos0 = pos() - N;
    f32_to_f16(p.xn, gemm_.xbuf(), (int64_t) N * E, st_);
    gemm_.run_h(L.wq, gemm_.xbuf(), E, N, p.qfull, H * D * 2);
    gemm_.run_h(L.wk, gemm_.xbuf(), E, N, p.k, Hkv * D);
    gemm_.run_h(L.wv, gemm_.xbuf(), E, N, p.v, Hkv * D);
    attn_prep(p.qfull, p.k, p.v, F(L.q_norm), F(L.k_norm), p.q, cur_->kc[il], cur_->vc[il], N, H, Hkv, D, c.n_rot, c.rope_base,
              &d_par_->pos0, c.rms_eps, st_, mrope_of(cur_));
    // the tensor-core kernel (head dim 256); BNK_PF_ATTN_TC=0: the one-warp-per-row kernel
    static const bool tc = !getenv("BNK_PF_ATTN_TC") || atoi(getenv("BNK_PF_ATTN_TC")) != 0;
    if (!(tc &&
          attention_prefill_tc(p.q, cur_->kc[il], cur_->vc[il], p.qfull, p.ao, N, H, Hkv, D, pos0, 1.f / sqrtf((float) D), st_)))
        attention_prefill(p.q, cur_->kc[il], cur_->vc[il], p.qfull, p.ao, N, H, Hkv, D, pos0, 1.f / sqrtf((float) D), st_);
    gemm_.run(L.wo, p.ao, H * D, N, p.out, E);
}

void DenseEngine::pf_ffn(int il, int N) {
    const Config & c = model_.lc;
    const DenseLayer & L = model_.layers[il];
    const int E = c.n_embd, FF = c.n_ff;
    auto & p = pf_;
    f32_to_f16(p.xn, gemm_.xbuf(), (int64_t) N * E, st_);
    gemm_.run_h(L.gate, gemm_.xbuf(), E, N, p.sg, FF);
    gemm_.run_h(L.up, gemm_.xbuf(), E, N, p.su, FF);
    silu_mul(p.sg, p.su, p.sg, (int64_t) N * FF, st_);
    gemm_.run(L.down, p.sg, FF, N, p.out, E);
}

void DenseEngine::prefill_chunk(const int32_t * tokens, int N) {
    const double t0 = now_ms();
    for (DenseEngine * s = this; s; s = s->next_.get()) s->prefill_chunk_here(tokens, N);
    pstats.ms += now_ms() - t0;
    pstats.tokens += N;
    pstats.chunks++;
}

// A chunk's inputs on this stage: the history, the parameters, the embeddings (first stage) or the previous
// stage's residual rows.
void DenseEngine::pf_in(const int32_t * tokens, int N, int buf) {
    join_commit_here();
    if (N < 1 || N > pf_max_) throw std::runtime_error("prefill_chunk: bad size");
    if (cur_->pending_T) throw std::runtime_error("prefill_chunk: a verify window is pending");
    if (pos() + N > opt_.max_ctx) throw std::runtime_error("context full");
    const int E = model_.cfg.n_embd;
    auto & p = pf_;
    const int pos0 = pos();
    cur_->history.insert(cur_->history.end(), tokens, tokens + N);
    *h_par_ = WinParams{pos0, N, ++seq_, 0};
    CUDA_CHECK(cudaMemcpyAsync(d_par_, h_par_, sizeof(WinParams), cudaMemcpyHostToDevice, st_));
    if (first()) {
        const bool img = std::any_of(tokens, tokens + N, is_image_id);
        std::vector<int32_t> mapped;
        if (img) {
            mapped.assign(tokens, tokens + N);
            for (auto & t : mapped) t = vocab_id(t);
        }
        CUDA_CHECK(cudaMemcpyAsync(p.tok.p, img ? mapped.data() : tokens, (size_t) N * 4, cudaMemcpyHostToDevice, st_));
        dequant_gather(model_.tok_embd, p.tok, N, p.res, st_);
        if (img) {   // the encoder's rows over the image cells' (every tensor-parallel rank embeds)
            if (!head_->images_) throw std::runtime_error("the prompt has images but no vision encoder is loaded (--mmproj)");
            head_->images_->put_rows(cur_->history, pos0, N, p.res.p, st_);
        }
    } else {
        CUDA_CHECK(cudaMemcpyAsync(p.res.p, pf_xfer_in_[buf], (size_t) N * E * 4, cudaMemcpyHostToDevice, st_));
    }
}

// res -> norm -> attention / DeltaNet -> p.out
void DenseEngine::pf_layer_a(int il, int N) {
    const Config & c = model_.cfg;
    const DenseLayer & L = model_.layers[il];
    rmsnorm_rows(pf_.res, F(L.attn_norm), pf_.xn, N, c.n_embd, c.n_embd, c.n_embd, c.rms_eps, st_);
    if (L.attn) pf_attn(il, N); else pf_gdn(il, N);
}

// res -> norm -> FFN -> p.out  (the residual already holds the block's output)
void DenseEngine::pf_layer_b(int il, int N) {
    const Config & c = model_.cfg;
    const DenseLayer & L = model_.layers[il];
    rmsnorm_rows(pf_.res, F(L.post_norm), pf_.xn, N, c.n_embd, c.n_embd, c.n_embd, c.rms_eps, st_);
    pf_ffn(il, N);
}

// the head on the last token (through the decode buffers, row 0), or the rows out to the next stage
void DenseEngine::pf_out(int N, int buf) {
    const int E = model_.cfg.n_embd;
    auto & p = pf_;
    if (last_) {
        CUDA_CHECK(cudaMemcpyAsync(res_.p, p.res.p + (size_t) (N - 1) * E, (size_t) E * 4, cudaMemcpyDeviceToDevice, st_));
        pend_ = nullptr;
        head(1);
    } else if (!tp()) {
        CUDA_CHECK(cudaMemcpyAsync(pf_xfer_out_[buf], p.res.p, (size_t) N * E * 4, cudaMemcpyDeviceToHost, st_));
    }
    last_T = 1;
}

// This stage's part of a prompt chunk: embeddings (first stage) or the previous stage's residual rows, its layers,
// then the head on the last token (last stage) or the rows out to the next stage.
void DenseEngine::prefill_chunk_here(const int32_t * tokens, int N, int buf) {
    DeviceScope ds(dev_);
    const double t0 = now_ms();
    const int E = model_.cfg.n_embd, pos0 = pos();
    auto & p = pf_;
    pf_in(tokens, N, buf);
    // BNK_PF_PROF=1: GPU time per block kind over the chunk (attention, DeltaNet, FFN), synchronously measured
    static const bool prof = getenv("BNK_PF_PROF") != nullptr;
    double t_attn = 0, t_gdn = 0, t_ffn = 0;
    auto lap = [&](double & acc, auto && f) {
        if (!prof) { f(); return; }
        CUDA_CHECK(cudaStreamSynchronize(st_));
        const double a = now_ms();
        f();
        CUDA_CHECK(cudaStreamSynchronize(st_));
        acc += now_ms() - a;
    };
    for (int il = l0_; il < l1_; ++il) {
        lap(model_.layers[il].attn ? t_attn : t_gdn, [&]() { pf_layer_a(il, N); });
        add_rows(p.res, p.out, (int64_t) N * E, st_);
        lap(t_ffn, [&]() { pf_layer_b(il, N); });
        add_rows(p.res, p.out, (int64_t) N * E, st_);
        pf_tap(il, N);
    }
    if (prof)
        fprintf(stderr, "pf GPU %d: chunk of %d at %d: attention %.1f ms, DeltaNet %.1f ms, FFN %.1f ms\n", dev_, N, pos0,
                t_attn, t_gdn, t_ffn);
    pf_out(N, buf);
    CUDA_CHECK(cudaStreamSynchronize(st_));
    pf_ms_ += now_ms() - t0;
    pf_tokens_ += N;
}

// Tensor parallel: both ranks run the chunk in lockstep, their partial sums exchanged (as fp16) after each block.
void DenseEngine::tp_prefill_chunk(const int32_t * tokens, int N) {
    const double t0 = now_ms();
    DenseEngine * rk[2] = {this, peer_};
    const Config & c = model_.cfg;
    const int E = c.n_embd;
    for (DenseEngine * r : rk) {
        DeviceScope ds(r->dev_);
        r->pf_in(tokens, N, 0);
    }
    for (int il = 0; il < c.n_layer; ++il) {
        for (DenseEngine * r : rk) {
            DeviceScope ds(r->dev_);
            r->pf_layer_a(il, N);
            r->tp_pf_send(r->pf_.out, N, 0, il);
        }
        for (DenseEngine * r : rk) {
            DeviceScope ds(r->dev_);
            r->tp_pf_recv(r->pf_.res, N, 0, il);
        }
        for (DenseEngine * r : rk) {
            DeviceScope ds(r->dev_);
            r->pf_layer_b(il, N);
            r->tp_pf_send(r->pf_.out, N, 1, il);
        }
        for (DenseEngine * r : rk) {
            DeviceScope ds(r->dev_);
            r->tp_pf_recv(r->pf_.res, N, 1, il);
            r->pf_tap(il, N);
        }
    }
    for (DenseEngine * r : rk) {
        DeviceScope ds(r->dev_);
        r->pf_out(N, 0);
        CUDA_CHECK(cudaStreamSynchronize(r->st_));
        r->pf_tokens_ += N;
    }
    pf_ms_ += now_ms() - t0;
    (void) E;
}

// Processes a prompt (batched chunks for long stretches, decode windows for short tails) and commits it; the
// logits of its last token are in logits row 0. Several stages: chunk k+1 on the first stage while a worker runs
// chunk k through the others.
void DenseEngine::sync_positions(const std::vector<int32_t> & tokens) {
    if (!mrope_) return;
    const auto & h = cur_->history;
    const bool img = std::any_of(h.begin(), h.end(), is_image_id) || std::any_of(tokens.begin(), tokens.end(), is_image_id);
    if (!img && cur_->rpos_plain) return;
    std::vector<int32_t> pos;
    if (img) {
        std::vector<int32_t> seq(h);
        seq.insert(seq.end(), tokens.begin(), tokens.end());
        images_->positions(seq, opt_.max_ctx, pos);
    } else {
        pos.resize((size_t) opt_.max_ctx * 3);
        for (size_t k = 0; k < pos.size(); ++k) pos[k] = (int32_t) (k / 3);
    }
    for (DenseEngine * s = this; s; s = s->next_.get()) {
        DeviceScope d2(s->dev_);
        CUDA_CHECK(cudaMemcpy(s->cur_->rpos.p, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));
        s->cur_->rpos_plain = !img;
    }
}

void DenseEngine::prefill(const std::vector<int32_t> & tokens) {
    DeviceScope ds(dev_);
    join_commit();
    sync_positions(tokens);
    // image cells go through prompt chunks however few they are (their embeddings are put in there)
    size_t img_end = 0;
    for (size_t k = tokens.size(); k > 0 && !img_end; --k)
        if (is_image_id(tokens[k - 1])) img_end = k;
    if (img_end && (opt_.prefill_chunk <= 0 || !images_)) throw std::runtime_error("images need the prompt path and a vision encoder (--mmproj)");
    size_t i = 0;
    ensure_ctx(pos() + (int) tokens.size() + kMaxWindowDense);
    const int n = (int) tokens.size();
    DenseEngine * t = tail();
    if (tp()) {   // both ranks together, chunk by chunk
        while (i < tokens.size()) {
            const size_t left = tokens.size() - i;
            const int pos0 = pos();
            if (pf_max_ > 0 && ((int) left >= opt_.prefill_min || i < img_end)) {
                const int N = (int) std::min<size_t>(pf_max_, left);
                const double c0 = now_ms();
                tp_prefill_chunk(tokens.data() + i, N);
                mtp_feed(pf_.res, tokens.data() + i, N, pos0);
                pstats.ms += now_ms() - c0;
                pstats.tokens += N;
                pstats.chunks++;
                i += N;
            } else {
                const int T = (int) std::min<size_t>(kMaxWindowDense, left);
                forward(tokens.data() + i, T);
                mtp_feed(res_, tokens.data() + i, T, pos0);
                i += T;
            }
            if (on_prefill_progress) on_prefill_progress(i, tokens.size());
        }
        last_rebalance_ms = 0;
        return;
    }
    static const bool pipe = !getenv("BNK_PIPE_PROMPT") || atoi(getenv("BNK_PIPE_PROMPT")) != 0;
    if (next_ && pipe && !worker_) worker_ = std::make_unique<Worker>();
    // Chunks of even size on a pipeline, enough of them that the stages overlap (a read in one chunk leaves the
    // GPUs taking turns: 3,000 tokens 715 tok/s whole, 1,100+ in four): at least one per stage, chunks of at least
    // BNK_PIPE_CHUNK_MIN (512) tokens as the GEMMs lose efficiency below that; a read shorter than BNK_PIPE_SPLIT
    // (256) stays whole.
    static const int split_min = getenv("BNK_PIPE_SPLIT") ? atoi(getenv("BNK_PIPE_SPLIT")) : 256;
    static const int chunk_min = getenv("BNK_PIPE_CHUNK_MIN") ? atoi(getenv("BNK_PIPE_CHUNK_MIN")) : 512;
    static const int min_parts = getenv("BNK_PIPE_PARTS") ? atoi(getenv("BNK_PIPE_PARTS")) : 0;
    int chunk = pf_max_;
    if (next_ && pipe && split_min > 0 && pf_max_ > 0 && n >= split_min && n >= 2 * opt_.prefill_min) {
        int parts = std::max((n + pf_max_ - 1) / pf_max_, std::max(1, n / std::max(chunk_min, opt_.prefill_min)));
        parts = std::max(parts, std::min(stages(), n / opt_.prefill_min));
        if (min_parts > 0) parts = std::max(parts, min_parts);
        chunk = std::max(opt_.prefill_min, (n + parts - 1) / parts);
    }
    size_t handed = 0;
    bool posted = false;
    int buf = 0;
    auto drain = [&]() {
        if (!posted) return;
        worker_->wait();
        posted = false;
        if (on_prefill_progress && handed < tokens.size()) on_prefill_progress(handed, tokens.size());
    };
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
        const bool chunked = (int) left >= opt_.prefill_min || i < img_end;
        if (pf_max_ > 0 && chunked && next_ && pipe) {
            const int N = (int) std::min<size_t>(chunk, left);
            const int32_t * tk = tokens.data() + i;
            const double c0 = now_ms();
            prefill_chunk_here(tk, N, buf);
            worker_->wait();
            if (posted && on_prefill_progress) on_prefill_progress(handed, tokens.size());
            const int b = buf;
            worker_->post([this, t, tk, N, pos0, b, c0]() {
                for (DenseEngine * s = next_.get(); s; s = s->next_.get()) s->prefill_chunk_here(tk, N, b);
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
        drain();
        if (pf_max_ > 0 && chunked) {
            const int N = (int) std::min<size_t>(pf_max_, left);
            prefill_chunk(tokens.data() + i, N);
            t->mtp_feed(t->pf_.res, tokens.data() + i, N, pos0);
            i += N;
        } else {
            const int T = (int) std::min<size_t>(kMaxWindowDense, left);
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
    for (DenseEngine * s = this; s; s = s->next_.get()) s->last_T = t->last_T;
    last_rebalance_ms = 0;
}

}  // namespace bnk
