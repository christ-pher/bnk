// The glm5next (GLM-5.3-Flash) layer, on the engine's forward driver: mHC hyper-connections around a KDA
// linear-attention or MLA + DSA attention block and a dense or MoE MLP block. Decode windows (<= kMaxWindow rows,
// GEMVs) and prompt chunks (N rows, GEMMs) share the kernels; docs/GLM.md has the maths.
#include "engine/engine.h"

#include <cmath>
#include <cstring>

#include "kernels/gemm.h"
#include "kernels/glm_ops.h"
#include "kernels/ops.h"

namespace bnk {

static const float * F(const QMat & m) { return (const float *) m.data; }

static void gemv_q(const QMat & W, const ActQ8 & xq, const float * x, int64_t ldx, int T, float * y, int64_t ldy,
                   cudaStream_t s, GemvSplit sp = {}) {
    if (is_float_format(W.type)) gemv(W, nullptr, x, ldx, T, y, ldy, false, s, sp);
    else gemv(W, &xq, nullptr, 0, T, y, ldy, false, s, sp);
}

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

void Engine::glm_alloc(int W) {
    const Config & c = model_.cfg;
    const int E = c.n_embd, HC = c.hc_dim(), nw = (2 + c.hc) * c.hc;
    g_hcw_.alloc((size_t) W * nw);
    g_small_.alloc((size_t) W * nw);
    g_gates_.alloc((size_t) W * (2 * c.ssm_state + c.ssm_vheads));
    g_qr_.alloc((size_t) W * (c.q_lora + c.kv_lora));
    g_qb_.alloc((size_t) W * (c.mla_q_dim() + c.idx_heads * c.idx_dim));
    g_qabs_.alloc((size_t) W * c.n_head * c.kv_lora);
    g_olat_.alloc((size_t) W * c.n_head * c.kv_lora);
    g_idxw_.alloc((size_t) W * (2 * c.idx_dim + c.idx_heads));   // the indexer projection: key | weights | gate
    g_ffn_gu_.alloc((size_t) W * c.n_ff);
    g_ffn_h_.alloc((size_t) W * c.n_ff);
    g_ones_.alloc(W);
    std::vector<float> ones(W, 1.f);
    CUDA_CHECK(cudaMemcpy(g_ones_.p, ones.data(), W * 4, cudaMemcpyHostToDevice));
    (void) E; (void) HC;
}

// w = (pre, post, comb) of every row from the residual streams; mixed_ = the collapsed block input
void Engine::glm_mhc_pre(const QMat & fn, const QMat & base, const QMat & scale, float * w, int T) {
    const Config & c = model_.cfg;
    const int E = c.n_embd, HC = c.hc_dim(), nw = (2 + c.hc) * c.hc;
    mhc_norm(res_, xn_, T, c.hc, E, c.rms_eps, st_);
    gemv_q(fn, act_, xn_, HC, T, g_small_, nw, st_);
    mhc_weights(g_small_, F(base), F(scale), w, T, c.hc, c.hc_eps, c.hc_sinkhorn, st_);
    mhc_collapse(res_, w, mixed_, T, c.hc, E, st_);
}

void Engine::glm_kda(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, C = c.conv_channels(), K = c.ssm_conv, S = c.ssm_state, H = c.ssm_vheads, QKV = c.kda_qkv();
    const int GA = 2 * S + H;   // f_a | g_a | b outputs per row
    float * co = gdn_co_[lane_][il], * gg = gdn_g_[lane_][il], * gb = gdn_b_[lane_][il];
    {   // the gates beside the q/k/v projection
        cudaStream_t sb = fork(0);
        gemv_q(L.kda_fgb_a, mixact_, xn_, E, T, g_gates_, GA, sb);
        gemv(L.kda_f_b, nullptr, g_gates_, GA, T, gg, QKV, false, sb);
        gemv(L.kda_g_b, nullptr, g_gates_.p + S, GA, T, z_, QKV, false, sb);
        kda_gate(gg, F(L.ssm_dt), F(L.ssm_a), c.kda_lower, T, H, S, sb);
        CUDA_CHECK(cudaMemcpy2DAsync(gb, H * 4, g_gates_.p + 2 * S, GA * 4, H * 4, T, cudaMemcpyDeviceToDevice, sb));
        sigmoid_inplace(gb, (int64_t) T * H, sb);
    }
    const bool one = wins_.size() == 1;
    float * qkv = one ? wins_[0].q->conv_buf[il].p + (size_t) (K - 1) * C : conv_out_.p;
    gemv_q(L.wqkv, mixact_, xn_, E, T, qkv, C, st_);
    if (!one)
        for (const Win & w : wins_)
            CUDA_CHECK(cudaMemcpyAsync(w.q->conv_buf[il].p + (size_t) (K - 1) * C, conv_out_.p + (size_t) w.r0 * C,
                                       (size_t) w.T * C * 4, cudaMemcpyDeviceToDevice, st_));
    join(0);
    dbg(st_, il, "kda_qkv", qkv, (size_t) T * C);
    dbg(st_, il, "kda_g", gg, (size_t) T * QKV);
    dbg(st_, il, "kda_beta", gb, (size_t) T * H);
    for (const Win & w : wins_) {
        float * cw = co + (size_t) w.r0 * C;
        gdn_conv(w.q->conv_buf[il], F(L.ssm_conv1d), cw, w.T, C, K, st_);
        kda_l2norm(cw, w.T, C, H, S, 1e-6f, st_);
        kda_recurrence(cw, C, gg + (size_t) w.r0 * QKV, gb + (size_t) w.r0 * H, w.q->ssm_state[il],
                       gdn_o_.p + (size_t) w.r0 * QKV, w.T, H, S, w.commit ? w.T : 0, st_);
    }
    dbg(st_, il, "kda_conv", co, (size_t) T * C);
    dbg(st_, il, "kda_o", gdn_o_, (size_t) T * QKV);
    if (!is_float_format(L.ssm_out.type) && !use_ref_gemv() && !getenv("BNK_DUMP_DIR") &&
        quantize_gated_norm(gdn_o_, z_, F(L.ssm_norm), T, H, S, c.rms_eps, act_, st_)) {
        gemv(L.ssm_out, &act_, nullptr, 0, T, out_, E, false, st_);
    } else {
        gated_rmsnorm(gdn_o_, z_, F(L.ssm_norm), gdn_n_, T, H, S, c.rms_eps, st_);
        dbg(st_, il, "kda_normed", gdn_n_, (size_t) T * QKV);
        gemv_auto(L.ssm_out, gdn_n_, QKV, T, out_, E, false, act_, st_);
    }
    dbg(st_, il, "attn_out", out_, (size_t) T * E);
    for (const Win & w : wins_)
        if (w.commit) shift_rows(w.q->conv_buf[il], C, w.T, K - 1, st_);
}

void Engine::glm_mla(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, H = c.n_head, Dk = c.qk_dim, Dv = c.v_dim, Dl = c.kv_lora, QR = c.q_lora + c.kv_lora;
    const int QB = c.mla_q_dim() + c.idx_heads * c.idx_dim, IX = 2 * c.idx_dim + c.idx_heads;
    const bool sparse = wins_[0].q->kraw[il].p != nullptr;
    {   // the indexer's projection beside the attention's
        cudaStream_t sb = fork(0);
        gemv_q(L.idx_x, mixact_, xn_, E, T, g_idxw_, IX, sb);
    }
    gemv_q(L.attn_qkv_a, mixact_, xn_, E, T, g_qr_, QR, st_);
    rmsnorm_rows(g_qr_, F(L.attn_q_a_norm), g_qr_, T, c.q_lora, QR, QR, c.rms_eps, st_);
    rmsnorm_rows(g_qr_.p + c.q_lora, F(L.attn_kv_a_norm), g_qr_.p + c.q_lora, T, Dl, QR, QR, c.rms_eps, st_);
    quantize_act(g_qr_, QR, T, c.q_lora, act_, st_);
    gemv_q(L.attn_q_b_idx, act_, g_qr_, QR, T, g_qb_, QB, st_);
    mla_absorb_q(g_qb_, QB, (const half *) L.attn_kv_b.data, g_qabs_, T, H, Dk, Dv, Dl, 1.f / sqrtf((float) Dk), st_);
    join(0);
    dbg(st_, il, "mla_qr", g_qr_, (size_t) T * QR);
    dbg(st_, il, "mla_qb", g_qb_, (size_t) T * QB);
    dbg(st_, il, "mla_ix", g_idxw_, (size_t) T * IX);
    for (size_t wi = 0; wi < wins_.size(); ++wi) {
        const Win & w = wins_[wi];
        const int * pos0 = &d_par_[wi].pos0;
        const size_t r = (size_t) w.r0;
        mla_store_latent(g_qr_.p + r * QR + c.q_lora, QR, w.q->kc[il], w.T, Dl, pos0, st_);
        const int32_t * sel = nullptr, * nsel = nullptr;
        if (sparse) {
            dsa_store(g_idxw_.p + r * IX, IX, F(L.idx_k_norm_w), F(L.idx_k_norm_b), w.q->kraw[il], w.T, c.idx_dim,
                      c.idx_heads, pos0, 1e-6f, st_);
            dsa_pool(w.q->kraw[il], F(L.idx_ape), w.q->pooled[il], w.T, c.idx_dim, c.idx_kpool, pos0, st_);
            dsa_score(g_qb_.p + r * QB + c.mla_q_dim(), QB, g_idxw_.p + r * IX + c.idx_dim, IX, w.q->pooled[il],
                      qsa_scores_, max_blocks_, w.T, c.idx_heads, c.idx_dim, c.idx_kpool, qsh_.dense_cells(), pos0, 0, st_);
            dsa_topk(qsa_scores_, max_blocks_, qsa_sel_, qsa_nsel_, w.T, qsh_.top_blocks, c.idx_kpool, qsh_.dense_cells(),
                     pos0, 0, st_);
            sel = qsa_sel_;
            nsel = qsa_nsel_;
        }
        mla_attention(g_qabs_.p + r * H * Dl, w.q->kc[il], sel, nsel, g_olat_.p + r * H * Dl, w.T, H, Dl, c.idx_kpool,
                      qsh_.top_blocks, pos0, attn_scratch_, st_);
    }
    dbg(st_, il, "mla_olat", g_olat_, (size_t) T * H * Dl);
    mla_out(g_olat_, (const half *) L.attn_kv_b.data, attn_o_, T, H, Dk, Dv, Dl, st_);
    dbg(st_, il, "mla_o", attn_o_, (size_t) T * H * Dv);
    gemv_auto(L.wo, attn_o_, H * Dv, T, out_, E, false, act_, st_);
    dbg(st_, il, "attn_out", out_, (size_t) T * E);
}

void Engine::glm_ffn(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, FF = c.n_ff;
    gemv_q(L.ffn_gate_up, mixact_, xn_, E, T, g_ffn_gu_, FF, st_, GemvSplit{FF, g_ffn_h_, FF});
    swiglu_clamp(g_ffn_gu_, g_ffn_h_, g_ffn_h_, (int64_t) T * FF, c.swiglu_limit, st_);
    gemv_auto(L.ffn_down, g_ffn_h_, FF, T, moe_out_, E, false, act_, st_);
}

void Engine::glm_moe(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, k = c.n_expert_used, FF = c.n_ff_shexp;
    const MoeLayerDesc d = cache_.desc(il);
    {   // the shared expert beside the routed ones
        cudaStream_t sb = fork(0);
        gemv_q(L.sh_gu, mixact_, xn_, E, T, sg_, FF, sb, GemvSplit{FF, su_, FF});
        swiglu_clamp(sg_, su_, sh_, (int64_t) T * FF, c.swiglu_limit, sb);
        gemv_auto(L.sh_down, sh_, FF, T, shared_out_, E, false, act_, sb);
    }
    gemv_q(L.router, mixact_, xn_, E, T, rlogits_, c.n_expert, st_);
    route_sigmoid_topk(rlogits_, F(L.router_bias), T, c.n_expert, k, rids_, rw_, c.expert_weights_scale, c.router_norm, st_);
    moe_plan(rids_, rw_, T, k, d, moes_, msg(il), xn_, E, &d_par_->seq, counts_.p + (size_t) il * c.n_expert, st_);
    if (route_log_)
        CUDA_CHECK(cudaMemcpyAsync(route_dev_.p + (size_t) il * kMaxWindow * k, rids_.p, (size_t) T * k * 4,
                                   cudaMemcpyDeviceToDevice, st_));
    if (!opt_.use_graphs || dump_all) {  // eager mode: answer the CPU part right here
        CUDA_CHECK(cudaStreamSynchronize(st_));
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
    moe_reduce(moes_, msg(il), shared_out_, g_ones_, moe_out_, T, k, E, st_, &d_par_->seq, false);
}

void Engine::glm_layer_forward(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd;
    glm_mhc_pre(L.hc_attn_fn, L.hc_attn_base, L.hc_attn_scale, g_hcw_, T);
    dbg(st_, il, "hc_attn_w", g_hcw_, (size_t) T * (2 + c.hc) * c.hc);
    dbg(st_, il, "hc_attn_in", mixed_, (size_t) T * E);
    rmsnorm_rows(mixed_, F(L.attn_norm), xn_, T, E, E, E, c.rms_eps, st_);
    quantize_act(xn_, E, T, E, mixact_, st_);
    dbg(st_, il, "attn_in", xn_, (size_t) T * E);
    if (L.attn) glm_mla(il, T); else glm_kda(il, T);
    mhc_combine(res_, out_, g_hcw_, T, c.hc, E, st_);
    dbg(st_, il, "hc_combine", res_, (size_t) T * c.hc_dim());
    glm_mhc_pre(L.hc_ffn_fn, L.hc_ffn_base, L.hc_ffn_scale, g_hcw_, T);
    rmsnorm_rows(mixed_, F(L.ffn_norm), xn_, T, E, E, E, c.rms_eps, st_);
    quantize_act(xn_, E, T, E, mixact_, st_);
    dbg(st_, il, "ffn_in", xn_, (size_t) T * E);
    if (L.moe) glm_moe(il, T); else glm_ffn(il, T);
    dbg(st_, il, "ffn_out", moe_out_, (size_t) T * E);
    mhc_combine(res_, moe_out_, g_hcw_, T, c.hc, E, st_);
    dbg(st_, il, "l_last", res_, (size_t) T * c.hc_dim());
    if (head_->dump_all) {
        auto & dl = head_->dumped_layers;
        dl.resize(c.n_layer);
        dl[il].resize((size_t) T * c.hc_dim());
        CUDA_CHECK(cudaMemcpyAsync(dl[il].data(), res_.p, dl[il].size() * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
    }
}

void Engine::glm_head(int T) {
    const Config & c = model_.cfg;
    const int E = c.n_embd;
    hc_mean(res_, x_, T, c.hc, E, st_);
    rmsnorm_rows(x_, F(model_.hc_head.norm), xn_, T, E, E, E, c.rms_eps, st_);
    gemv_auto(model_.output, xn_, E, T, logits_, c.n_vocab, false, act_, st_);
}

// ------------------------------------------------------------------------------------------- prompt path
void Engine::glm_pf_carve(int N, const std::function<void(DevBuf<float> &, size_t)> & carve) {
    const Config & c = model_.cfg;
    const int nw = (2 + c.hc) * c.hc, P = std::min(N, kGlmPiece);
    auto & g = gpf_;
    carve(g.hcw, (size_t) N * nw);
    carve(g.small, (size_t) N * nw);
    carve(g.qr, (size_t) N * (c.q_lora + c.kv_lora));
    carve(g.qn, (size_t) N * c.q_lora);
    carve(g.qb, (size_t) N * (c.mla_q_dim() + c.idx_heads * c.idx_dim));
    carve(g.ix, (size_t) N * (2 * c.idx_dim + c.idx_heads));
    carve(g.qabs, (size_t) P * c.n_head * c.kv_lora);
    carve(g.olat, (size_t) P * c.n_head * c.kv_lora);
    carve(g.gates, (size_t) N * (2 * c.ssm_state + c.ssm_vheads));
    carve(g.g, (size_t) N * c.kda_qkv());
    carve(g.gu, (size_t) P * c.n_ff);
    carve(g.u, (size_t) P * c.n_ff);
    carve(g.h, (size_t) P * c.n_ff);
}

void Engine::glm_pf_mhc_pre(const QMat & fn, const QMat & base, const QMat & scale, float * w, int N) {
    const Config & c = model_.cfg;
    const int E = c.n_embd, HC = c.hc_dim(), nw = (2 + c.hc) * c.hc;
    auto & p = pf_;
    mhc_norm(p.res, p.xn, N, c.hc, E, c.rms_eps, st_);
    gemm_.run(fn, p.xn, HC, N, gpf_.small, nw);
    mhc_weights(gpf_.small, F(base), F(scale), w, N, c.hc, c.hc_eps, c.hc_sinkhorn, st_);
    mhc_collapse(p.res, w, p.mixed, N, c.hc, E, st_);
}

void Engine::glm_pf_kda(int il, int N) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, C = c.conv_channels(), K = c.ssm_conv, S = c.ssm_state, H = c.ssm_vheads, QKV = c.kda_qkv();
    const int GA = 2 * S + H;
    auto & p = pf_;
    auto & g = gpf_;
    CUDA_CHECK(cudaMemcpyAsync(p.conv.p, cur_->conv_buf[il].p, (size_t) (K - 1) * C * 4, cudaMemcpyDeviceToDevice, st_));
    f32_to_f16(p.xn, gemm_.xbuf(), (int64_t) N * E, st_);
    gemm_.run_h(L.wqkv, gemm_.xbuf(), E, N, p.conv.p + (size_t) (K - 1) * C, C);
    gemm_.run_h(L.kda_fgb_a, gemm_.xbuf(), E, N, g.gates, GA);
    // f_b / g_b read strided inputs: the fp16 copies made row by row would be N launches, so convert the whole
    // [N][GA] block once and run the GEMMs on it with their strides
    f32_to_f16(g.gates, gemm_.xbuf(), (int64_t) N * GA, st_);
    gemm_.run_h(L.kda_f_b, gemm_.xbuf(), GA, N, g.g, QKV);
    gemm_.run_h(L.kda_g_b, gemm_.xbuf() + S, GA, N, p.z, QKV);
    kda_gate(g.g, F(L.ssm_dt), F(L.ssm_a), c.kda_lower, N, H, S, st_);
    CUDA_CHECK(cudaMemcpy2DAsync(p.b.p, H * 4, g.gates.p + 2 * S, GA * 4, H * 4, N, cudaMemcpyDeviceToDevice, st_));
    sigmoid_inplace(p.b, (int64_t) N * H, st_);
    gdn_conv(p.conv, F(L.ssm_conv1d), p.co, N, C, K, st_);
    kda_l2norm(p.co, N, C, H, S, 1e-6f, st_);
    kda_recurrence(p.co, C, g.g, p.b, cur_->ssm_state[il], p.o, N, H, S, N, st_);
    dbg(st_, il, "kda_conv", p.co, (size_t) N * C);
    dbg(st_, il, "kda_g", g.g, (size_t) N * QKV);
    dbg(st_, il, "kda_o", p.o, (size_t) N * QKV);
    gated_rmsnorm(p.o, p.z, F(L.ssm_norm), p.n, N, H, S, c.rms_eps, st_);
    dbg(st_, il, "kda_normed", p.n, (size_t) N * QKV);
    gemm_.run(L.ssm_out, p.n, QKV, N, p.out, E);
    CUDA_CHECK(cudaMemcpyAsync(cur_->conv_buf[il].p, p.conv.p + (size_t) N * C, (size_t) (K - 1) * C * 4,
                               cudaMemcpyDeviceToDevice, st_));
}

void Engine::glm_pf_mla(int il, int N) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, H = c.n_head, Dk = c.qk_dim, Dv = c.v_dim, Dl = c.kv_lora, QR = c.q_lora + c.kv_lora;
    const int QB = c.mla_q_dim() + c.idx_heads * c.idx_dim, IX = 2 * c.idx_dim + c.idx_heads, r = c.idx_kpool;
    auto & p = pf_;
    auto & g = gpf_;
    const int pos0 = pos() - N;
    const bool sparse = cur_->kraw[il].p != nullptr;
    f32_to_f16(p.xn, gemm_.xbuf(), (int64_t) N * E, st_);
    gemm_.run_h(L.attn_qkv_a, gemm_.xbuf(), E, N, g.qr, QR);
    if (sparse) gemm_.run_h(L.idx_x, gemm_.xbuf(), E, N, g.ix, IX);
    rmsnorm_rows(g.qr, F(L.attn_q_a_norm), g.qn, N, c.q_lora, QR, c.q_lora, c.rms_eps, st_);
    rmsnorm_rows(g.qr.p + c.q_lora, F(L.attn_kv_a_norm), g.qr.p + c.q_lora, N, Dl, QR, QR, c.rms_eps, st_);
    mla_store_latent(g.qr.p + c.q_lora, QR, cur_->kc[il], N, Dl, &d_par_->pos0, st_);
    gemm_.run(L.attn_q_b_idx, g.qn, c.q_lora, N, g.qb, QB);
    dbg(st_, il, "mla_qb", g.qb, (size_t) N * QB);
    dbg(st_, il, "mla_ix", g.ix, (size_t) N * IX);
    const int32_t * sel = nullptr, * nsel = nullptr;
    if (sparse) {
        dsa_store(g.ix, IX, F(L.idx_k_norm_w), F(L.idx_k_norm_b), cur_->kraw[il], N, c.idx_dim, c.idx_heads, &d_par_->pos0,
                  1e-6f, st_);
        dsa_pool(cur_->kraw[il], F(L.idx_ape), cur_->pooled[il], N, c.idx_dim, r, &d_par_->pos0, st_);
        // selections for every row: 64 rows at a time bound the score scratch (rows that are dense select nothing)
        for (int t0 = 0; t0 < N; t0 += 64) {
            const int t1 = std::min(N, t0 + 64);
            dsa_score(g.qb.p + (size_t) t0 * QB + c.mla_q_dim(), QB, g.ix.p + (size_t) t0 * IX + c.idx_dim, IX,
                      cur_->pooled[il], p.scores, max_blocks_, t1 - t0, c.idx_heads, c.idx_dim, r, qsh_.dense_cells(),
                      &d_par_->pos0, t0, st_);
            dsa_topk(p.scores, max_blocks_, p.sel.p + (size_t) t0 * qsh_.top_blocks, p.nsel.p + t0, t1 - t0, qsh_.top_blocks,
                     r, qsh_.dense_cells(), &d_par_->pos0, t0, st_);
        }
        sel = p.sel;
        nsel = p.nsel;
    }
    // the attention in pieces of rows (qabs / olat are sized for a piece)
    for (int t0 = 0; t0 < N; t0 += kGlmPiece) {
        const int n = std::min(kGlmPiece, N - t0);
        mla_absorb_q(g.qb.p + (size_t) t0 * QB, QB, (const half *) L.attn_kv_b.data, g.qabs, n, H, Dk, Dv, Dl,
                     1.f / sqrtf((float) Dk), st_);
        // rows [t0, t0 + n) of the chunk: the kernel indexes sel / nsel by the chunk row and positions by pos0 + row
        mla_attention_prefill(g.qabs.p - (size_t) t0 * H * Dl, cur_->kc[il], sel, nsel, g.olat.p - (size_t) t0 * H * Dl,
                              t0, t0 + n, H, Dl, r, qsh_.top_blocks, pos0, st_);
        mla_out(g.olat, (const half *) L.attn_kv_b.data, p.ao.p + (size_t) t0 * H * Dv, n, H, Dk, Dv, Dl, st_);
    }
    dbg(st_, il, "mla_o", p.ao, (size_t) N * H * Dv);
    gemm_.run(L.wo, p.ao, H * Dv, N, p.out, E);
}

void Engine::glm_pf_ffn(int il, int N) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, FF = c.n_ff;
    auto & p = pf_;
    auto & g = gpf_;
    // piece by piece (the gate / up / h buffers hold kGlmPiece rows); each GEMM converts its own input, since
    // Gemm::run's fp16 buffer is one (the down product's conversion would overwrite the next piece's input)
    for (int t0 = 0; t0 < N; t0 += kGlmPiece) {
        const int n = std::min(kGlmPiece, N - t0);
        f32_to_f16(p.xn.p + (size_t) t0 * E, gemm_.xbuf(), (int64_t) n * E, st_);
        gemm_.run_h(qmat_rows(L.ffn_gate_up, 0, FF), gemm_.xbuf(), E, n, g.gu, FF);
        gemm_.run_h(qmat_rows(L.ffn_gate_up, FF, FF), gemm_.xbuf(), E, n, g.u, FF);
        swiglu_clamp(g.gu, g.u, g.h, (int64_t) n * FF, c.swiglu_limit, st_);
        gemm_.run(L.ffn_down, g.h, FF, n, p.moe.p + (size_t) t0 * E, E);
    }
}

// pf_moe reads its input from p.mixed (the GEMMs, the tensor-core path's fp16 copy, the CPU experts' host copy)
void Engine::glm_pf_moe(int il, int N) {
    copy_f32(pf_.mixed, pf_.xn, (int64_t) N * model_.cfg.n_embd, st_);
    pf_moe(il, N);
}

void Engine::glm_pf_layer(int il, int N) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd;
    auto & p = pf_;
    // BNK_PF_PROF=1: wall time of each part of the layer (synchronizing after each)
    static const bool prof = env_int("BNK_PF_PROF", 0) != 0;
    double t0 = 0, t1 = 0, t2 = 0;
    auto tick = [&]() { if (!prof) return 0.0; CUDA_CHECK(cudaStreamSynchronize(st_)); return now_ms(); };
    t0 = tick();
    glm_pf_mhc_pre(L.hc_attn_fn, L.hc_attn_base, L.hc_attn_scale, gpf_.hcw, N);
    dbg(st_, il, "hc_attn_w", gpf_.hcw, (size_t) N * (2 + c.hc) * c.hc);
    dbg(st_, il, "hc_attn_in", p.mixed, (size_t) N * E);
    rmsnorm_rows(p.mixed, F(L.attn_norm), p.xn, N, E, E, E, c.rms_eps, st_);
    dbg(st_, il, "attn_in", p.xn, (size_t) N * E);
    if (L.attn) glm_pf_mla(il, N); else glm_pf_kda(il, N);
    dbg(st_, il, "attn_out", p.out, (size_t) N * E);
    mhc_combine(p.res, p.out, gpf_.hcw, N, c.hc, E, st_);
    dbg(st_, il, "hc_combine", p.res, (size_t) N * c.hc_dim());
    t1 = tick();
    glm_pf_mhc_pre(L.hc_ffn_fn, L.hc_ffn_base, L.hc_ffn_scale, gpf_.hcw, N);
    rmsnorm_rows(p.mixed, F(L.ffn_norm), p.xn, N, E, E, E, c.rms_eps, st_);
    dbg(st_, il, "ffn_in", p.xn, (size_t) N * E);
    if (L.moe) glm_pf_moe(il, N); else glm_pf_ffn(il, N);
    dbg(st_, il, "ffn_out", p.moe, (size_t) N * E);
    mhc_combine(p.res, p.moe, gpf_.hcw, N, c.hc, E, st_);
    dbg(st_, il, "l_last", p.res, (size_t) N * c.hc_dim());
    t2 = tick();
    if (prof) fprintf(stderr, "pf L%02d N=%d %s %.1f ms  %s %.1f ms\n", il, N, L.attn ? "mla" : "kda", t1 - t0,
                      L.moe ? "moe" : "ffn", t2 - t1);
}

}  // namespace bnk
