#include "engine/engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "ggml.h"
#include "ggml-cpu.h"
#include "kernels/ops.h"

namespace bnk {

static const float * F(const QMat & m) { return (const float *) m.data; }

// Debug dumps of layer-0 intermediates (BNK_DUMP_DIR): appended per window, raw fp32.
static void dbg(cudaStream_t st, int il, const char * name, const float * dev, size_t n, bool host = false) {
    static const char * dir = getenv("BNK_DUMP_DIR");
    static int layer = env_int("BNK_DUMP_LAYER", 0);
    if (!dir || il != layer) return;
    std::vector<float> h(n);
    if (host) memcpy(h.data(), dev, n * 4);
    else {
        CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaMemcpy(h.data(), dev, n * 4, cudaMemcpyDeviceToHost));
    }
    FILE * f = fopen((std::string(dir) + "/" + name).c_str(), "ab");
    fwrite(h.data(), 4, n, f);
    fclose(f);
}

void Engine::load(const std::string & path, const EngineOptions & opt) {
    opt_ = opt;
    ggml_cpu_init();
    CUDA_CHECK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    model_.load(path, opt.verbose);
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
    moe_out_.alloc(W * E); logits_.alloc((size_t) W * c.n_vocab);
    tok_dev_.alloc(W); argmax_dev_.alloc(W);
    const int64_t maxcols = std::max<int64_t>(HC, 16384);
    actq_.alloc(W * maxcols); actd_.alloc(W * maxcols / 32);
    act_.q = actq_; act_.d = actd_;

    if (c.ple_layer >= 0) {
        const int ph = c.ple_heads();
        ple_emb_.alloc(W * ph * c.ple_dim); ple_key_.alloc(W * HC); ple_val_.alloc(W * E);
        ple_gated_.alloc(W * HC);
        ple_hist_.alloc((size_t) ((c.ple_conv - 1) * c.ple_ngram + W) * HC);
        ple_rows_dev_.alloc((size_t) W * ph * model_.ple_table->row_bytes());
        CUDA_CHECK(cudaMallocHost(&h_ple_rows_, (size_t) W * ph * model_.ple_table->row_bytes()));
    }
    conv_buf_.resize(c.n_layer);
    ssm_state_.resize(c.n_layer);
    kc_.resize(c.n_layer);
    vc_.resize(c.n_layer);
    size_t state_bytes = 0;
    for (int il = 0; il < c.n_layer; ++il) {
        if (c.is_attn(il)) {
            kc_[il].alloc((size_t) opt.max_ctx * c.n_head_kv * c.head_dim);
            vc_[il].alloc((size_t) opt.max_ctx * c.n_head_kv * c.head_dim);
            state_bytes += 2 * kc_[il].n * sizeof(half);
        } else {
            conv_buf_[il].alloc((size_t) (c.ssm_conv - 1 + W) * C);
            ssm_state_[il].alloc((size_t) c.ssm_vheads * c.ssm_state * c.ssm_state);
            state_bytes += (conv_buf_[il].n + ssm_state_[il].n) * 4;
        }
    }
    CUDA_CHECK(cudaMallocHost(&h_x_, W * E * 4));
    CUDA_CHECK(cudaMallocHost(&h_out_, W * E * 4));
    CUDA_CHECK(cudaMallocHost(&h_w_, W * c.n_expert_used * 4));
    CUDA_CHECK(cudaMallocHost(&h_ids_, W * c.n_expert_used * 4));
    if (opt.verbose)
        fprintf(stderr, "bnk: context %d, KV + recurrent state %.2f GiB, %d CPU expert threads\n", opt.max_ctx,
                state_bytes / 1073741824.0, cpu_.threads());
    reset();
}

void Engine::reset() {
    const Config & c = model_.cfg;
    history_.clear();
    for (int il = 0; il < c.n_layer; ++il) {
        if (!c.is_attn(il)) {
            CUDA_CHECK(cudaMemsetAsync(conv_buf_[il].p, 0, conv_buf_[il].n * 4, st_));
            CUDA_CHECK(cudaMemsetAsync(ssm_state_[il].p, 0, ssm_state_[il].n * 4, st_));
        }
    }
    if (ple_hist_.p) CUDA_CHECK(cudaMemsetAsync(ple_hist_.p, 0, ple_hist_.n * 4, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

// xn = norm(res); mixed = mean(xn * sigmoid(up(silu(down(xn)/hc)))); inj = inject(xn)
void Engine::hc_pre(const HcWeights & w, int T, bool want_inject) {
    const Config & c = model_.cfg;
    const int E = c.n_embd, HC = c.hc_dim();
    hc_norm(res_, F(w.norm), xn_, T, c.hc, E, c.rms_eps, st_);
    gemv_auto(w.down, xn_, HC, T, lo_, c.hc_rank, false, act_, st_);
    silu_scale(lo_, (int64_t) T * c.hc_rank, 1.f / c.hc, st_);
    gemv_auto(w.up, lo_, c.hc_rank, T, gpre_, HC, false, act_, st_);
    hc_mix(xn_, gpre_, mixed_, T, c.hc, E, st_);
    if (want_inject) gemv_auto(w.inject, xn_, HC, T, inj_, c.hc, false, act_, st_);
}

void Engine::ple(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, HC = c.hc_dim(), ph = c.ple_heads();
    const TensorRef & tab = *model_.ple_table;
    const size_t rb = tab.row_bytes();
    const double t0 = now_ms();
    // n-gram hashes over the token history (window tokens are already appended)
    for (int t = 0; t < T; ++t) {
        const int p = pos0_ + t;
        int64_t ctx[8];
        ctx[0] = history_[p];
        bool cut = false;
        for (int s = 1; s < c.ple_ngram; ++s) {
            const int q = p - s;
            const int64_t tok = (cut || q < 0) ? -1 : history_[q];
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
    CUDA_CHECK(cudaMemcpyAsync(ple_rows_dev_.p, h_ple_rows_, (size_t) T * ph * rb, cudaMemcpyHostToDevice, st_));
    QMat rows{ple_rows_dev_.p, (int) tab.type, (int64_t) T * ph, tab.ne[0], rb};
    dequant_rows(rows, 0, (int64_t) T * ph, ple_emb_, st_);
    const int D = ph * c.ple_dim;
    gemv_auto(L.ple_key, ple_emb_, D, T, ple_key_, HC, false, act_, st_);
    gemv_auto(L.ple_value, ple_emb_, D, T, ple_val_, E, false, act_, st_);
    const int hist = (c.ple_conv - 1) * c.ple_ngram;
    ple_gate(ple_key_, res_, ple_val_, F(L.ple_norm_key), F(L.ple_norm_query), F(L.ple_norm_conv), ple_gated_,
             ple_hist_.p + (size_t) hist * HC, T, c.hc, E, c.rms_eps, st_);
    ple_conv_add(res_, ple_gated_, ple_hist_, (const half *) L.ple_conv1d.data, T, HC, c.ple_conv, c.ple_ngram, st_);
    // keep the last `hist` rows as the history
    CUDA_CHECK(cudaMemcpyAsync(ple_hist_.p, ple_hist_.p + (size_t) T * HC, (size_t) hist * HC * 4,
                               cudaMemcpyDeviceToDevice, st_));
}

void Engine::gdn(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, C = c.conv_channels(), K = c.ssm_conv, S = c.ssm_state, nv = c.ssm_vheads;
    float * cb = conv_buf_[il];
    gemv_auto(L.wqkv, mixed_, E, T, cb + (size_t) (K - 1) * C, C, false, act_, st_);
    gemv_auto(L.wgate, mixed_, E, T, z_, nv * S, false, act_, st_);
    gemv_auto(L.ssm_beta, mixed_, E, T, beta_, nv, false, act_, st_);
    gemv_auto(L.ssm_alpha, mixed_, E, T, alpha_, nv, false, act_, st_);
    dbg(st_, il, "hc_mixed_attn", mixed_, (size_t) T * E);
    dbg(st_, il, "linear_attn_qkv_mixed", cb + (size_t) (K - 1) * C, (size_t) T * C);
    dbg(st_, il, "z", z_, (size_t) T * nv * S);
    gdn_conv(cb, F(L.ssm_conv1d), conv_out_, T, C, K, st_);
    dbg(st_, il, "conv_output_silu", conv_out_, (size_t) T * C);
    gdn_prep(conv_out_, T, C, c.ssm_groups, nv, S, alpha_, beta_, F(L.ssm_dt), F(L.ssm_a), c.rms_eps, st_);
    gdn_recurrence(conv_out_, C, alpha_, beta_, ssm_state_[il], gdn_o_, T, c.ssm_groups, nv, S, T, st_);
    gated_rmsnorm(gdn_o_, z_, F(L.ssm_norm), gdn_n_, T, nv, S, c.rms_eps, st_);
    dbg(st_, il, "attn_output", gdn_o_, (size_t) T * nv * S);
    dbg(st_, il, "final_output", gdn_n_, (size_t) T * nv * S);
    gemv_auto(L.ssm_out, gdn_n_, nv * S, T, out_, E, false, act_, st_);
    dbg(st_, il, "linear_attn_out", out_, (size_t) T * E);
    // conv history: the last K-1 inputs
    CUDA_CHECK(cudaMemcpyAsync(cb, cb + (size_t) T * C, (size_t) (K - 1) * C * 4, cudaMemcpyDeviceToDevice, st_));
}

void Engine::attn(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, H = c.n_head, Hkv = c.n_head_kv, D = c.head_dim;
    if (!qsa_warned_ && c.compress_ratio[il] > 0 && pos0_ + T > c.idx_top_k + c.compress_ratio[il] - 1) {
        fprintf(stderr, "bnk: warning: context beyond %d tokens needs QSA sparse attention (not yet implemented)\n",
                c.idx_top_k + c.compress_ratio[il] - 1);
        qsa_warned_ = true;
    }
    gemv_auto(L.wq, mixed_, E, T, qfull_, H * D * 2, false, act_, st_);
    gemv_auto(L.wk, mixed_, E, T, k_, Hkv * D, false, act_, st_);
    gemv_auto(L.wv, mixed_, E, T, v_, Hkv * D, false, act_, st_);
    attn_prep(qfull_, k_, v_, F(L.q_norm), F(L.k_norm), q_, kc_[il], vc_[il], T, H, Hkv, D, c.n_rot, c.rope_base,
              pos0_, c.rms_eps, st_);
    attention(q_, kc_[il], vc_[il], qfull_, attn_o_, T, H, Hkv, D, pos0_, 1.f / sqrtf((float) D), attn_scratch_, st_);
    gemv_auto(L.wo, attn_o_, H * D, T, out_, E, false, act_, st_);
}

void Engine::moe(int il, int T) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, k = c.n_expert_used, FF = c.n_ff_shexp;
    gemv_auto(L.router, mixed_, E, T, rlogits_, c.n_expert, false, act_, st_);
    route_topk(rlogits_, T, c.n_expert, k, rids_, rw_, c.expert_weights_scale, st_);
    CUDA_CHECK(cudaMemcpyAsync(h_ids_, rids_.p, T * k * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(h_w_, rw_.p, T * k * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(h_x_, mixed_.p, T * E * 4, cudaMemcpyDeviceToHost, st_));
    // shared expert on the GPU meanwhile
    gemv_auto(L.sh_gate, mixed_, E, T, sg_, FF, false, act_, st_);
    gemv_auto(L.sh_up, mixed_, E, T, su_, FF, false, act_, st_);
    silu_mul(sg_, su_, sh_, (int64_t) T * FF, st_);
    gemv_auto(L.sh_down, sh_, FF, T, out_, E, false, act_, st_);
    gemv_auto(L.shexp_gate_inp, mixed_, E, T, sgate_, 1, false, act_, st_);
    sigmoid_inplace(sgate_, T, st_);
    CUDA_CHECK(cudaStreamSynchronize(st_));
    const double t0 = now_ms();
    tasks_.clear();
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < k; ++j) tasks_.push_back({t, j, h_ids_[t * k + j], h_w_[t * k + j]});
    cpu_.run(il, T, h_x_, tasks_, h_out_);
    dbg(st_, il, "ffn_moe_out", h_out_, (size_t) T * E, true);
    dbg(st_, il, "hc_mixed_ffn", h_x_, (size_t) T * E, true);
    times.cpu_experts_ms += now_ms() - t0;
    CUDA_CHECK(cudaMemcpyAsync(moe_out_.p, h_out_, T * E * 4, cudaMemcpyHostToDevice, st_));
    add_scaled(moe_out_, out_, sgate_, T, E, st_);
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
    dbg(st_, il, "layer_in", res_, (size_t) T * c.hc_dim());
    hc_pre(L.hc_attn, T, true);
    if (L.attn) attn(il, T); else gdn(il, T);
    hc_combine(res_, out_, inj_, T, c.hc, c.n_embd, st_);
    dbg(st_, il, "hc_combine", res_, (size_t) T * c.hc_dim());
    hc_pre(L.hc_ffn, T, true);
    moe(il, T);
    dbg(st_, il, "ffn_out", moe_out_, (size_t) T * c.n_embd);
    dbg(st_, il, "hc_inject_ffn", inj_, (size_t) T * c.hc);
    hc_combine(res_, moe_out_, inj_, T, c.hc, c.n_embd, st_);
    dbg(st_, il, "l_last", res_, (size_t) T * c.hc_dim());
    if (dump_all) {
        dumped_layers.resize(c.n_layer);
        dumped_layers[il].resize((size_t) T * c.hc_dim());
        CUDA_CHECK(cudaMemcpyAsync(dumped_layers[il].data(), res_.p, dumped_layers[il].size() * 4,
                                   cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
    }
    if (il == dump_layer) {
        dumped.resize((size_t) T * c.hc_dim());
        CUDA_CHECK(cudaMemcpyAsync(dumped.data(), res_.p, dumped.size() * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
    }
}

void Engine::forward(const int32_t * tokens, int T, bool want_logits) {
    const Config & c = model_.cfg;
    if (T < 1 || T > kMaxWindow) throw std::runtime_error("forward: bad window");
    if (pos() + T > opt_.max_ctx) throw std::runtime_error("context full");
    const double t0 = now_ms();
    last_T = T;
    pos0_ = pos();
    history_.insert(history_.end(), tokens, tokens + T);
    CUDA_CHECK(cudaMemcpyAsync(tok_dev_.p, tokens, T * 4, cudaMemcpyHostToDevice, st_));
    dequant_gather(model_.tok_embd, tok_dev_, T, x_, st_);
    hc_init(x_, res_, T, c.hc, c.n_embd, st_);
    for (int il = 0; il < c.n_layer; ++il) layer_forward(il, T);
    if (want_logits) head(T);
    CUDA_CHECK(cudaStreamSynchronize(st_));
    times.total_ms += now_ms() - t0;
    times.calls++;
}

void Engine::prefill(const std::vector<int32_t> & tokens) {
    for (size_t i = 0; i < tokens.size(); i += kMaxWindow) {
        const int T = (int) std::min<size_t>(kMaxWindow, tokens.size() - i);
        forward(tokens.data() + i, T, i + T == tokens.size());
    }
}

int Engine::argmax(int t) {
    argmax_rows(logits_.p + (size_t) t * model_.cfg.n_vocab, 1, model_.cfg.n_vocab, argmax_dev_, st_);
    int32_t r;
    CUDA_CHECK(cudaMemcpyAsync(&r, argmax_dev_.p, 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    return r;
}

std::vector<float> Engine::logits_host(int t) {
    std::vector<float> v(model_.cfg.n_vocab);
    CUDA_CHECK(cudaMemcpy(v.data(), logits_.p + (size_t) t * model_.cfg.n_vocab, v.size() * 4,
                          cudaMemcpyDeviceToHost));
    return v;
}

}  // namespace bnk
