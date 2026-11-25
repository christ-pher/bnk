#include "inkling/ink_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "inkling/ink_ops.h"
#include "kernels/ops.h"

namespace bnk {

namespace {
constexpr int kInkCkpts = 4;   // snapshots kept (each holds the window rings: ~250 MB of host RAM)
}

InkEngine::~InkEngine() {
    for (auto & row : graphs_)
        for (auto & g : row)
            if (g) cudaGraphExecDestroy(g);
    if (h_tok_) cudaFreeHost(h_tok_);
    if (h_pos_) cudaFreeHost(h_pos_);
    if (h_tau_) cudaFreeHost(h_tau_);
    if (st_) cudaStreamDestroy(st_);
}

StageInfo InkEngine::stage_info(int) const {
    StageInfo s;
    s.device = dev_;
    s.layer_begin = 0;
    s.layer_end = model_.cfg.n_layer;
    return s;
}

void InkEngine::load(const std::string & path, const EngineOptions & opt) {
    opt_ = opt;
    if (!opt.gpus.empty() && opt.gpus[0] >= 0) dev_ = opt.gpus[0];
    CUDA_CHECK(cudaSetDevice(dev_));
    DeviceScope ds(dev_);
    model_.load(path, opt.verbose);
    const InkConfig & c = model_.cfg;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    chunk_ = std::max(kInkWin, std::min(env_int("BNK_INK_CHUNK", 512), 2048));
    ring_ = c.n_swa + chunk_;
    conv_ch_ = 2 * c.kv_dim() + 2 * c.E;
    const int W = chunk_, E = c.E, F = c.n_ff, R = W * c.n_used;

    visits_.resize(c.n_visits());
    for (int v = 0; v < c.n_visits(); ++v) {
        const int ip = v < c.n_layer ? v : c.loop_start + (v - c.n_layer) % c.n_core();
        Visit & V = visits_[v];
        V.window = c.swa[ip] ? c.n_swa : 0;
        V.ring = c.swa[ip] ? ring_ : opt.max_ctx;
        V.K.alloc((size_t) V.ring * c.kv_dim());
        V.V.alloc((size_t) V.ring * c.kv_dim());
        V.conv.alloc((size_t) 3 * conv_ch_);
        V.win_in.alloc((size_t) kInkWin * conv_ch_);
        kv_bytes_ += (size_t) 2 * V.ring * c.kv_dim() * sizeof(half);
    }
    x_.alloc((size_t) W * E); ffn_inp_.alloc((size_t) W * E); tmp_.alloc((size_t) W * E); o_.alloc((size_t) W * E);
    moe_.alloc((size_t) W * E);
    mixed_.alloc((size_t) 4 * W * E); msg_.alloc((size_t) 4 * W * E);
    for (auto & b : ins_) b.alloc((size_t) W * E);
    q_.alloc((size_t) W * c.q_dim()); attn_.alloc((size_t) W * c.q_dim()); gate_.alloc((size_t) W * c.q_dim());
    r_.alloc((size_t) W * c.n_head * c.n_rel); sink_.alloc((size_t) W * c.n_head);
    dyn_k_.alloc((size_t) W * c.n_head_kv * c.n_conv); dyn_v_.alloc((size_t) W * c.n_head_kv * c.n_conv);
    conv_in_.alloc((size_t) W * std::max(E, c.q_dim())); kc_.alloc((size_t) W * c.kv_dim()); vc_.alloc((size_t) W * c.kv_dim());
    ctrl_in_.alloc((size_t) W * 2 * E); ctrl_g_.alloc((size_t) W * c.gate_rank);
    ctrl_gain_.alloc((size_t) W * E); ctrl_forget_.alloc((size_t) W * E);
    tau_.alloc(W);
    mix_sc_.alloc(ink_mix_scratch_floats(W));
    rlog_.alloc((size_t) W * c.n_expert);
    gx_.alloc((size_t) R * E); gu_.alloc((size_t) R * 2 * F); gh_.alloc((size_t) R * F); gd_.alloc((size_t) R * E);
    blocks_buf_.alloc((size_t) kMaxBlocks * W * E);
    logits_.alloc((size_t) kInkWin * c.n_vocab);
    tok_dev_.alloc(W); ids_dev_.alloc(R); rows_dev_.alloc(R); inv_dev_.alloc(R); argmax_dev_.alloc(kInkWin);
    w_dev_.alloc(R); wg_dev_.alloc(R);
    h_ids_.resize(R); h_w_.resize(R);
    CUDA_CHECK(cudaHostAlloc((void **) &h_tok_, W * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **) &h_pos_, 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **) &h_tau_, W * 4, cudaHostAllocDefault));
    d_pos_.alloc(1);
    plan_buf_.alloc(sizeof(InkMoePlan));
    mh_.alloc((size_t) kInkMaxPairs * F); my_.alloc((size_t) kInkMaxPairs * E); mgu_.alloc((size_t) kInkMaxPairs * 2 * F);
    mq_.alloc((size_t) kInkMaxPairs * E); mqd_.alloc((size_t) kInkMaxPairs * E / 32);
    graphs_on_ = opt.use_graphs && env_int("BNK_INK_GRAPHS", 1) != 0;
    const int64_t maxcols = 2 * E;
    actq_.alloc((size_t) kInkWin * maxcols); actd_.alloc((size_t) kInkWin * maxcols / 32);
    acts_.alloc((size_t) kInkWin * maxcols / 16); acth_.alloc((size_t) kInkWin * maxcols);
    act_.q = actq_; act_.d = actd_; act_.s = acts_; act_.h = acth_;
    gemm_.init(st_, (size_t) std::max<int64_t>((int64_t) c.q_dim() * E, (int64_t) 2 * F * E), (size_t) W * 2 * E);
    ckpts_.resize(kInkCkpts);
    reset();
    if (opt.verbose) {
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        fprintf(stderr, "bnk: inkling: context %d, K/V %.2f GiB (window rings of %d cells), prompt chunks of %d, %.1f GiB "
                        "free on GPU %d\n", opt.max_ctx, kv_bytes_ / double(1 << 30), ring_, chunk_, fr / double(1 << 30), dev_);
    }
}

// ------------------------------------------------------------------------------------------------ state
void InkEngine::reset() {
    DeviceScope ds(dev_);
    history_.clear();
    pending_T_ = 0;
    for (auto & k : ckpts_) k.pos = -1;
    for (auto & V : visits_) CUDA_CHECK(cudaMemsetAsync(V.conv.p, 0, V.conv.n * 4, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void InkEngine::checkpoint() {
    DeviceScope ds(dev_);
    const int p = pos();
    if (p == 0) return;
    for (auto & k : ckpts_)
        if (k.pos == p) { k.age = ++ckpt_age_; return; }
    Ckpt * slot = nullptr;
    for (auto & k : ckpts_)
        if (k.pos < 0) { slot = &k; break; }
    if (!slot) slot = &*std::min_element(ckpts_.begin(), ckpts_.end(), [](const Ckpt & a, const Ckpt & b) { return a.age < b.age; });
    size_t nconv = 0, nring = 0;
    for (auto & V : visits_) {
        nconv += V.conv.n;
        if (V.window) nring += V.K.n + V.V.n;
    }
    slot->conv.resize(nconv);
    slot->ring.resize(nring);
    size_t oc = 0, orr = 0;
    for (auto & V : visits_) {
        CUDA_CHECK(cudaMemcpyAsync(slot->conv.data() + oc, V.conv.p, V.conv.n * 4, cudaMemcpyDeviceToHost, st_));
        oc += V.conv.n;
        if (V.window) {
            CUDA_CHECK(cudaMemcpyAsync(slot->ring.data() + orr, V.K.p, V.K.n * 2, cudaMemcpyDeviceToHost, st_));
            orr += V.K.n;
            CUDA_CHECK(cudaMemcpyAsync(slot->ring.data() + orr, V.V.p, V.V.n * 2, cudaMemcpyDeviceToHost, st_));
            orr += V.V.n;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    slot->pos = p;
    slot->age = ++ckpt_age_;
}

int InkEngine::rollback(int max_pos) {
    DeviceScope ds(dev_);
    Ckpt * best = nullptr;
    for (auto & k : ckpts_)
        if (k.pos >= 0 && k.pos <= max_pos && k.pos <= pos() && (!best || k.pos > best->pos)) best = &k;
    if (!best) return -1;
    size_t oc = 0, orr = 0;
    for (auto & V : visits_) {
        CUDA_CHECK(cudaMemcpyAsync(V.conv.p, best->conv.data() + oc, V.conv.n * 4, cudaMemcpyHostToDevice, st_));
        oc += V.conv.n;
        if (V.window) {
            CUDA_CHECK(cudaMemcpyAsync(V.K.p, best->ring.data() + orr, V.K.n * 2, cudaMemcpyHostToDevice, st_));
            orr += V.K.n;
            CUDA_CHECK(cudaMemcpyAsync(V.V.p, best->ring.data() + orr, V.V.n * 2, cudaMemcpyHostToDevice, st_));
            orr += V.V.n;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    history_.resize(best->pos);
    pending_T_ = 0;
    best->age = ++ckpt_age_;
    for (auto & k : ckpts_)
        if (k.pos > best->pos) k.pos = -1;
    return best->pos;
}

int InkEngine::reusable(const std::vector<int32_t> & prompt) const {
    const auto & h = history_;
    size_t common = 0;
    while (common < h.size() && common < prompt.size() && h[common] == prompt[common]) ++common;
    if (!h.empty() && common == h.size() && h.size() < prompt.size()) return (int) h.size();
    if (common == 0) return 0;
    const int lim = (int) std::min(common, prompt.size() - 1);
    int best = 0;
    for (const auto & k : ckpts_)
        if (k.pos >= 0 && k.pos <= lim && k.pos <= (int) h.size()) best = std::max(best, k.pos);
    return best;
}

// ------------------------------------------------------------------------------------------------ pieces
void InkEngine::mm(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy, bool acc) {
    if (T <= 0) return;
    static const bool bf16act = env_int("BNK_INK_BF16ACT", 0) != 0;   // tests: activations rounded as the CPU reference does
    if (bf16act && W.type == 30) {
        if (!rb_.p) rb_.alloc((size_t) chunk_ * 2 * model_.cfg.E);
        ink_round_bf16(x, ldx, T, (int) W.cols, rb_, st_);
        x = rb_;
        ldx = W.cols;
    }
    if (T <= kMaxGemvRows) gemv_auto(W, x, ldx, T, y, ldy, acc, act_, st_);
    else gemm_.run(W, x, ldx, T, y, ldy, acc);
}

void InkEngine::mix(const std::vector<const float *> & blocks, const float * cur, const float * query, int nq, int T,
                    float * out) {
    InkStates st;
    for (const float * b : blocks) st.p[st.n++] = b;
    st.p[st.n++] = cur;
    ink_mix(st, query, nq, T, model_.cfg.E, out, mix_sc_, st_);
}

void InkEngine::control(float * cur, const float * msg, const InkLoop & loop, int j, int T) {
    const InkConfig & c = model_.cfg;
    const int E = c.E;
    ink_concat_rms2(cur, msg, ctrl_in_, T, E, 1e-6f, st_);
    mm(loop.down[j], ctrl_in_, 2 * E, T, ctrl_g_, c.gate_rank);
    mm(loop.gain[j], ctrl_g_, c.gate_rank, T, ctrl_gain_, E);
    mm(loop.forget[j], ctrl_g_, c.gate_rank, T, ctrl_forget_, E);
    ink_ctrl_combine(cur, cur, msg, ctrl_gain_, ctrl_forget_, (int64_t) T * E, st_);
}

// the conv of channels [off, off + C) of visit v: x rows (stride ldx) -> y [T][C]; keep: the state advances now,
// else the rows wait in the visit's window inputs for the commit
void InkEngine::conv(int v, int off, int C, const float * x, int64_t ldx, float * y, int T, const float * w,
                     const float * dyn, int H, bool keep) {
    Visit & V = visits_[v];
    if (!keep)
        CUDA_CHECK(cudaMemcpy2DAsync(V.win_in.p + off, (size_t) conv_ch_ * 4, x, (size_t) ldx * 4, (size_t) C * 4, T,
                                     cudaMemcpyDeviceToDevice, st_));
    const float scale = 1.f / sqrtf((dyn ? 2.f : 1.f) * model_.cfg.n_conv + 1.f);
    ink_conv(x, ldx, y, T, C, V.conv.p + off, conv_ch_, w, dyn, model_.cfg.n_head_kv * model_.cfg.n_conv, H, scale,
             keep ? T : 0, st_);
}

void InkEngine::moe(const InkLayer & L, const float * x, int T, float * out) {
    const InkConfig & c = model_.cfg;
    const int E = c.E, F = c.n_ff, k = c.n_used, n = T * k;
    mm(L.router, x, E, T, rlog_, c.n_expert);
    ink_route(rlog_, L.exp_bias, T, c.n_expert, k, c.w_scale, ids_dev_, w_dev_, st_);
    static const bool host_moe = env_int("BNK_INK_HOST_MOE", 0) != 0;   // tests: the grouped GEMV / GEMM path
    if (T <= kInkWin && !host_moe) {
        InkMoeArgs a;
        a.ids = ids_dev_; a.w = w_dev_; a.T = T; a.k = k; a.n_exp = c.n_expert; a.E = E; a.F = F; a.x = x;
        a.gate_up = (const uint8_t *) L.gate_up.data; a.gu_type = L.gate_up.type; a.gu_row_bytes = L.gate_up.row_bytes;
        a.gu_scale = L.d_gu_scale;
        a.down = (const uint8_t *) L.down.data; a.down_type = L.down.type; a.down_row_bytes = L.down.row_bytes;
        a.down_scale = L.d_down_scale;
        a.beta = c.situ_beta; a.lb = c.situ_lin;
        a.plan = (InkMoePlan *) plan_buf_.p; a.gu = mgu_; a.h = mh_; a.y = my_; a.out = out; a.q = mq_; a.qd = mqd_;
        ink_moe_window(a, st_);
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(h_ids_.data(), ids_dev_.p, (size_t) n * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(h_w_.data(), w_dev_.p, (size_t) n * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    // the routed pairs grouped by expert: rows (tokens), weights, and each pair's place in the grouping
    std::vector<int> cnt(c.n_expert + 1, 0);
    for (int i = 0; i < n; ++i) cnt[h_ids_[i] + 1]++;
    for (int e = 0; e < c.n_expert; ++e) cnt[e + 1] += cnt[e];
    std::vector<int32_t> rows(n), inv(n);
    std::vector<float> wg(n);
    std::vector<int> fill(cnt.begin(), cnt.end() - 1);
    for (int i = 0; i < n; ++i) {
        const int p = fill[h_ids_[i]]++;
        rows[p] = i / k;
        wg[p] = h_w_[i];
        inv[i] = p;
    }
    CUDA_CHECK(cudaMemcpyAsync(rows_dev_.p, rows.data(), (size_t) n * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(inv_dev_.p, inv.data(), (size_t) n * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(wg_dev_.p, wg.data(), (size_t) n * 4, cudaMemcpyHostToDevice, st_));
    ink_gather(x, E, rows_dev_, n, E, gx_, st_);
    for (int e = 0; e < c.n_expert; ++e) {
        const int s = cnt[e], m = cnt[e + 1] - cnt[e];
        if (m == 0) continue;
        mm(InkModel::expert(L.gate_up, e, c.n_expert), gx_.p + (size_t) s * E, E, m, gu_.p + (size_t) s * 2 * F, 2 * F);
        ink_situ(gu_.p + (size_t) s * 2 * F, 2 * F, gh_.p + (size_t) s * F, F, m, F, c.situ_beta, c.situ_lin,
                 L.gu_scale[e], wg_dev_.p + s, L.down_scale[e], st_);
        mm(InkModel::expert(L.down, e, c.n_expert), gh_.p + (size_t) s * F, F, m, gd_.p + (size_t) s * E, E);
    }
    ink_moe_sum(out, gd_, inv_dev_, T, k, E, st_);
}

float * InkEngine::fresh_block() {
    if (blk_next_ >= kMaxBlocks) throw std::runtime_error("inkling: block buffers exhausted");
    return blocks_buf_.p + (size_t) blk_next_++ * chunk_ * model_.cfg.E;
}

// One layer visit (llama.cpp's llama_model_inkling2::graph::build_layer).
void InkEngine::layer(int v, int ip, int T, int pos0, std::vector<const float *> & blocks, const float * msgs,
                      const InkLoop * loop, bool keep) {
    const InkConfig & c = model_.cfg;
    const InkLayer & L = model_.layers[ip];
    const int E = c.E, H = c.n_head, KVH = c.n_head_kv, D = c.head_dim, KVD = c.kv_dim(), QD = c.q_dim();
    const size_t TE = (size_t) T * E;
    const bool swa = L.swa;

    // the attention's inputs
    const float * in[3] = {x_, x_, x_};
    if (L.attn_res_query) {
        mix(blocks, x_, L.attn_res_query, L.attn_res_nq, T, mixed_);
        for (int j = 0; j < 3; ++j) in[j] = mixed_.p + (swa ? 0 : j) * TE;
    }
    float * a[3] = {ins_[0].p, ins_[1].p, ins_[2].p};
    if (msgs) {
        for (int j = 0; j < 3; ++j) {
            CUDA_CHECK(cudaMemcpyAsync(a[j], in[j], TE * 4, cudaMemcpyDeviceToDevice, st_));
            control(a[j], msgs + j * TE, *loop, j, T);
            in[j] = a[j];
        }
    }
    if (swa && msgs) {
        for (int j = 0; j < 3; ++j) ink_rms(in[j], E, L.attn_norm, a[j], E, T, E, c.rms_eps, st_);
    } else if (swa) {
        ink_rms(in[0], E, L.attn_norm, a[0], E, T, E, c.rms_eps, st_);
        a[1] = a[2] = a[0];
    } else {
        ink_rms(in[0], E, L.q_norm, a[0], E, T, E, c.rms_eps, st_);
        ink_rms(in[1], E, L.k_norm, a[1], E, T, E, c.rms_eps, st_);
        ink_rms(in[2], E, L.v_norm, a[2], E, T, E, c.rms_eps, st_);
    }

    // q, k, v, the relative-position queries, the sinks
    mm(L.wq, a[0], E, T, q_, QD);
    mm(L.wk, a[1], E, T, conv_in_, KVD);
    if (!swa) mm(L.k_conv_proj, a[1], E, T, dyn_k_, KVH * c.n_conv);
    conv(v, 0, KVD, conv_in_, KVD, kc_, T, L.k_conv, swa ? nullptr : dyn_k_.p, KVH, keep);
    mm(L.wv, a[2], E, T, conv_in_, KVD);
    if (!swa) mm(L.v_conv_proj, a[2], E, T, dyn_v_, KVH * c.n_conv);
    conv(v, KVD, KVD, conv_in_, KVD, vc_, T, L.v_conv, swa ? nullptr : dyn_v_.p, KVH, keep);
    mm(L.wrel, a[0], E, T, r_, H * c.n_rel);
    mm(L.wsink, a[0], E, T, sink_, H);
    const float * tau = swa ? nullptr : tau_.p;
    ink_clip_heads(q_, T, H, D, L.q_clip, c.qk_clip, tau, st_);
    ink_clip_heads(kc_, T, KVH, D, nullptr, c.qk_clip, nullptr, st_);
    ink_rms(vc_, D, nullptr, vc_, D, T * KVH, D, c.rms_eps, st_);
    ink_clip_heads(r_, T, H, c.n_rel, L.rel_clip, c.rel_clip, tau, st_);

    Visit & V = visits_[v];
    ink_kv_store(kc_, vc_, V.K, V.V, V.ring, d_pos_, T, KVD, st_);
    ink_attention(q_, r_, sink_, V.K, V.V, V.ring, L.rel_bias, c.extent(ip), c.n_rel, V.window, d_pos_, T, H, KVH, D,
                  1.f / sqrtf((float) D), attn_, nullptr, 0, st_);
    mm(L.wgate, a[0], E, T, gate_, QD);
    ink_mul_sigmoid(attn_, gate_, (int64_t) T * QD, st_);
    mm(L.wo, attn_, QD, T, conv_in_, E);
    conv(v, 2 * KVD, E, conv_in_, E, tmp_, T, L.attn_post_conv, nullptr, 1, keep);

    // the first layer of each block starts a new residual stream; its input joins the blocks
    if (ip % c.res_block == 0) {
        // a fresh buffer per push (the bank of the previous pass keeps pointing at its own)
        float * b = fresh_block();
        CUDA_CHECK(cudaMemcpyAsync(b, x_, TE * 4, cudaMemcpyDeviceToDevice, st_));
        blocks.push_back(b);
        CUDA_CHECK(cudaMemcpyAsync(x_, tmp_, TE * 4, cudaMemcpyDeviceToDevice, st_));
    } else {
        ink_add(x_, x_, tmp_, (int64_t) TE, st_);
    }
    CUDA_CHECK(cudaMemcpyAsync(ffn_inp_, x_, TE * 4, cudaMemcpyDeviceToDevice, st_));

    // MoE
    mix(blocks, ffn_inp_, L.ffn_res_query, 1, T, mixed_);
    if (msgs) control(mixed_, msgs + 3 * TE, *loop, 3, T);
    ink_rms(mixed_, E, L.ffn_norm, o_, E, T, E, c.rms_eps, st_);
    moe(L, o_, T, moe_);
    conv(v, 2 * KVD + E, E, moe_, E, tmp_, T, L.ffn_post_conv, nullptr, 1, keep);
    ink_add(x_, ffn_inp_, tmp_, (int64_t) TE, st_);
}

// The whole forward over T rows at positions pos()..pos()+T-1 (llama_model_inkling2_loop::graph).
void InkEngine::run(const int32_t * tokens, int T, bool keep) {
    const InkConfig & c = model_.cfg;
    const int E = c.E;
    const size_t TE = (size_t) T * E;
    const int pos0 = pos();
    if (T < 1 || T > chunk_) throw std::runtime_error("inkling: window out of range");
    if (pos0 + T > opt_.max_ctx) throw std::runtime_error("inkling: context full");
    memcpy(h_tok_, tokens, (size_t) T * 4);
    *h_pos_ = pos0;
    for (int t = 0; t < T; ++t)
        h_tau_[t] = 1.f + c.log_alpha * logf(std::max(1.f, (float) (pos0 + t + 1) / c.log_floor));
    if (T <= kInkWin && graphs_on_ && !logits_hook) {
        cudaGraphExec_t & g = graphs_[T][keep ? 1 : 0];
        if (!g) {
            cudaGraph_t gr;
            CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
            enqueue(T, keep);
            CUDA_CHECK(cudaStreamEndCapture(st_, &gr));
            CUDA_CHECK(cudaGraphInstantiate(&g, gr, 0));
            cudaGraphDestroy(gr);
            ++captures_;
        }
        CUDA_CHECK(cudaGraphLaunch(g, st_));
    } else {
        enqueue(T, keep);
    }
    logits_rows_ = T <= kInkWin ? T : logits_hook ? T - (T - 1) / kInkWin * kInkWin : 1;
    if (logits_hook) {
        CUDA_CHECK(cudaStreamSynchronize(st_));
        if (T <= kInkWin) logits_hook(pos0, T, logits_);
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    last_T = logits_rows_;
    history_.insert(history_.end(), tokens, tokens + T);
}

// The forward's GPU work for T rows whose inputs are staged (h_tok_, h_pos_, h_tau_): no host round trip for
// windows of up to kInkWin rows (captured as a graph); prompt chunks group their experts on the host.
void InkEngine::enqueue(int T, bool keep) {
    const InkConfig & c = model_.cfg;
    const int E = c.E;
    const size_t TE = (size_t) T * E;
    const int pos0 = *h_pos_;
    CUDA_CHECK(cudaMemcpyAsync(tok_dev_.p, h_tok_, (size_t) T * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(d_pos_.p, h_pos_, 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(tau_.p, h_tau_, (size_t) T * 4, cudaMemcpyHostToDevice, st_));
    dequant_gather(model_.tok_embd, tok_dev_, T, x_, st_);
    ink_rms(x_, E, model_.tok_norm, x_, E, T, E, c.rms_eps, st_);

    blk_next_ = 0;
    auto fresh = [&]() { return fresh_block(); };
    std::vector<const float *> blocks;
    int v = 0;
    for (int il = 0; il < c.loop_start; ++il) layer(v++, il, T, pos0, blocks, nullptr, nullptr, keep);
    float * prelude = fresh();
    CUDA_CHECK(cudaMemcpyAsync(prelude, x_, TE * 4, cudaMemcpyDeviceToDevice, st_));
    const std::vector<const float *> prelude_blocks = blocks;
    for (int pass = 0; pass < c.loop_count; ++pass) {
        std::vector<const float *> bank;
        float * bank_out = nullptr;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = fresh();
            CUDA_CHECK(cudaMemcpyAsync(bank_out, x_, TE * 4, cudaMemcpyDeviceToDevice, st_));
            CUDA_CHECK(cudaMemcpyAsync(x_, prelude, TE * 4, cudaMemcpyDeviceToDevice, st_));
            blocks = prelude_blocks;
        }
        for (int ip = c.loop_start; ip < c.loop_end; ++ip) {
            const int slot = pass == 0 ? ip : c.n_layer + (pass - 1) * c.n_core() + (ip - c.loop_start);
            if (pass == 0) {
                layer(slot, ip, T, pos0, blocks, nullptr, nullptr, keep);
                continue;
            }
            const InkLoop & lp = model_.loops[ip - c.loop_start];
            mix(bank, bank_out, lp.msg_query, 4, T, msg_);
            layer(slot, ip, T, pos0, blocks, msg_, &lp, keep);
        }
    }
    for (int il = c.loop_end; il < c.n_layer; ++il) layer(il, il, T, pos0, blocks, nullptr, nullptr, keep);

    // the output: mixed over the blocks, normed, the head (unpadded rows), soft-capped
    mix(blocks, x_, model_.output_res_query, 1, T, mixed_);
    ink_rms(mixed_, E, model_.output_norm, o_, E, T, E, c.rms_eps, st_);
    const QMat head = qmat_rows(model_.output, 0, c.n_vocab);
    auto head_rows = [&](int r0, int n) {
        mm(head, o_.p + (size_t) r0 * E, E, n, logits_, c.n_vocab);
        if (c.softcap > 0) ink_softcap(logits_, (int64_t) n * c.n_vocab, c.softcap, st_);
    };
    if (T <= kInkWin) {
        head_rows(0, T);   // a decode / verify window: every row
    } else if (logits_hook) {   // tests: every row of a prompt chunk (the last piece stays in logits_)
        for (int r0 = 0; r0 < T; r0 += kInkWin) {
            const int n = std::min(kInkWin, T - r0);
            head_rows(r0, n);
            CUDA_CHECK(cudaStreamSynchronize(st_));
            logits_hook(pos0 + r0, n, logits_);
        }
    } else {
        head_rows(T - 1, 1);   // a prompt chunk: its last row
    }
}

void InkEngine::prefill(const std::vector<int32_t> & tokens) {
    DeviceScope ds(dev_);
    if (pending_T_) throw std::runtime_error("inkling: prefill with a verify window pending");
    const double t0 = now_ms();
    for (size_t i = 0; i < tokens.size(); i += chunk_) {
        const int n = (int) std::min<size_t>(chunk_, tokens.size() - i);
        run(tokens.data() + i, n, true);
        if (on_prefill_progress) on_prefill_progress(i + n, tokens.size());
    }
    times.total_ms += now_ms() - t0;
}

void InkEngine::forward(const int32_t * tokens, int T, bool commit_all) {
    forward_batch({BatchWin{0, tokens, T, commit_all}});
}

void InkEngine::forward_batch(const std::vector<BatchWin> & wins) {
    DeviceScope ds(dev_);
    if (wins.size() != 1 || wins[0].slot != 0) throw std::runtime_error("inkling: one conversation slot");
    if (pending_T_) throw std::runtime_error("inkling: forward with a verify window pending");
    const BatchWin & w = wins[0];
    if (w.T > kInkWin) throw std::runtime_error("inkling: window too long");
    const double t0 = now_ms();
    run(w.tokens, w.T, w.commit_all);
    if (!w.commit_all) pending_T_ = w.T;
    times.total_ms += now_ms() - t0;
    times.calls++;
}

void InkEngine::commit(int cnt) {
    DeviceScope ds(dev_);
    if (pending_T_ == 0) throw std::runtime_error("commit: no verify window pending");
    if (cnt < 1 || cnt > pending_T_) throw std::runtime_error("commit: count out of range");
    for (auto & V : visits_) ink_conv_advance(V.win_in, V.conv, cnt, conv_ch_, st_);
    CUDA_CHECK(cudaStreamSynchronize(st_));
    history_.resize(history_.size() - (pending_T_ - cnt));
    pending_T_ = 0;
}

// ------------------------------------------------------------------------------------------------ logits
int InkEngine::argmax(int t) {
    DeviceScope ds(dev_);
    const int V = model_.cfg.n_vocab;
    argmax_rows(logits_.p + (size_t) t * V, 1, V, argmax_dev_, st_);
    int32_t r;
    CUDA_CHECK(cudaMemcpyAsync(&r, argmax_dev_.p, 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    return r;
}

void InkEngine::argmax_all(int T, int32_t * out, int row0) {
    DeviceScope ds(dev_);
    const int V = model_.cfg.n_vocab;
    argmax_rows(logits_.p + (size_t) row0 * V, T, V, argmax_dev_, st_);
    CUDA_CHECK(cudaMemcpyAsync(out, argmax_dev_.p, (size_t) T * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void InkEngine::topk_rows_host(int row0, int T, int K, int32_t * ids, float * vals, double * z, float temp) {
    DeviceScope ds(dev_);
    if (K > kTopkMax || T > kInkWin) throw std::runtime_error("topk_rows_host: too many");
    if (!topk_ids_.p) {
        topk_ids_.alloc((size_t) kInkWin * kTopkMax);
        topk_vals_.alloc((size_t) kInkWin * kTopkMax);
    }
    const int V = model_.cfg.n_vocab;
    topk_rows(logits_.p + (size_t) row0 * V, T, V, K, topk_ids_, topk_vals_, st_);
    CUDA_CHECK(cudaMemcpyAsync(ids, topk_ids_.p, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(vals, topk_vals_.p, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st_));
    if (z) {
        if (!topk_z_.p) topk_z_.alloc(kInkWin);
        row_expsum(logits_.p + (size_t) row0 * V, T, V, temp, topk_z_.p, st_);
        CUDA_CHECK(cudaMemcpyAsync(z, topk_z_.p, (size_t) T * sizeof(double), cudaMemcpyDeviceToHost, st_));
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void InkEngine::logits_rows_host(int T, float * out, int row0) {
    DeviceScope ds(dev_);
    CUDA_CHECK(cudaMemcpy(out, logits_.p + (size_t) row0 * model_.cfg.n_vocab, (size_t) T * model_.cfg.n_vocab * 4,
                          cudaMemcpyDeviceToHost));
}

std::vector<float> InkEngine::logits_host(int t) {
    DeviceScope ds(dev_);
    std::vector<float> v(model_.cfg.n_vocab);
    CUDA_CHECK(cudaMemcpy(v.data(), logits_.p + (size_t) t * model_.cfg.n_vocab, v.size() * 4, cudaMemcpyDeviceToHost));
    return v;
}

}  // namespace bnk
