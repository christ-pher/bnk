#include "dense/dense_dflash.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "dense/dense_ops.h"
#include "kernels/ops.h"

namespace bnk {

static const float * F(const QMat & m) { return (const float *) m.data; }

DenseDFlash::~DenseDFlash() {
    if (!loaded_) return;
    DeviceScope ds(dev_);
    for (cudaGraphExec_t g : graphs_)
        if (g) cudaGraphExecDestroy(g);
    if (h_par_) cudaFreeHost(h_par_);
    if (h_cpos_) cudaFreeHost(h_cpos_);
    if (h_out_) cudaFreeHost(h_out_);
    if (h_prob_) cudaFreeHost(h_prob_);
    if (h_tok_) cudaFreeHost(h_tok_);
    if (h_q_) cudaFreeHost(h_q_);
    if (h_cand_) cudaFreeHost(h_cand_);
}

int DenseDFlash::tap_index(int il) const {
    for (int i = 0; i < (int) taps_.size(); ++i)
        if (taps_[i] == il) return i;
    return -1;
}

size_t DenseDFlash::vram_bytes() const {
    size_t b = w_.vram_bytes;
    for (const auto & k : kc_) b += 2 * k.n * sizeof(half);
    return b;
}

void DenseDFlash::load(const std::string & path, const DenseModel & main, int slots, cudaStream_t st, bool verbose) {
    CUDA_CHECK(cudaGetDevice(&dev_));
    st_ = st;
    main_ = &main;
    if (!main.output.valid() || !main.tok_embd.valid()) throw std::runtime_error("dflash: needs the target's head and token table on its GPU");
    w_.gguf.open(path);
    const GgufModel & g = w_.gguf;
    E = (int) g.get_u64("dflash2.hidden_size");
    FF = (int) g.get_u64("dflash2.intermediate_size");
    H = (int) g.get_u64("dflash2.num_attention_heads");
    HKV = (int) g.get_u64("dflash2.num_key_value_heads");
    D = (int) g.get_u64("dflash2.head_dim");
    eps_ = (float) g.get_f64("dflash2.rms_norm_eps", 1e-6);
    theta_ = (float) g.get_f64("dflash2.rope_theta", 1e7);
    win_ = (int) g.get_u64("dflash2.sliding_window");
    mask_ = (int) g.get_u64("dflash2.mask_token_id");
    gsize_ = (int) g.get_u64("dflash2.conv_group_size");
    rank_ = (int) g.get_u64("dflash2.selector_rank");
    topk_ = (int) g.get_u64("dflash2.selector_top_k");
    nvocab_ = (int) g.get_u64("dflash2.vocab_size");
    for (int64_t t : g.get_int_arr("dflash2.target_layer_ids")) taps_.push_back((int) t);
    if (E != main.cfg.n_embd || nvocab_ != (int) main.output.rows || (int) taps_.size() != kTaps ||
        (int) g.get_u64("dflash2.block_size") != kBlock || g.get_u64("dflash2.conv_kernel_size") != 2 || D != 128 ||
        topk_ > 32 || 2 * win_ > kRing)
        throw std::runtime_error("dflash: " + path + " does not fit this model or this implementation");
    for (int il : taps_)
        if (il < 0 || il >= main.cfg.n_layer) throw std::runtime_error("dflash: tap layer out of range");

    for (int i = 0; i < kTaps; ++i) fc_[i] = w_.upload("fc." + std::to_string(i) + ".weight");
    hidden_norm_ = w_.upload("hidden_norm.weight");
    norm_ = w_.upload("norm.weight");
    sel_proj_ = w_.upload("candidate_selector.hidden_projection.weight");
    sel_pre_ = (half *) w_.upload("candidate_selector.predecessor_codebook").data;
    sel_succ_ = (half *) w_.upload("candidate_selector.successor_codebook").data;
    const int NL = (int) g.get_u64("dflash2.num_hidden_layers");
    layers_.resize(NL);
    for (int l = 0; l < NL; ++l) {
        Layer & L = layers_[l];
        const std::string b = "layers." + std::to_string(l) + ".";
        L.in_norm = w_.upload(b + "input_layernorm.weight");
        L.post_norm = w_.upload(b + "post_attention_layernorm.weight");
        L.wq = w_.upload(b + "self_attn.q_proj.weight");
        L.wk = w_.upload(b + "self_attn.k_proj.weight");
        L.wv = w_.upload(b + "self_attn.v_proj.weight");
        L.wo = w_.upload(b + "self_attn.o_proj.weight");
        L.q_norm = w_.upload(b + "self_attn.q_norm.weight");
        L.k_norm = w_.upload(b + "self_attn.k_norm.weight");
        L.gate = w_.upload(b + "mlp.gate_proj.weight");
        L.up = w_.upload(b + "mlp.up_proj.weight");
        L.down = w_.upload(b + "mlp.down_proj.weight");
        L.aconv_proj = w_.upload(b + "attention_conv.kernel_projection.weight");
        L.mconv_proj = w_.upload(b + "mlp_conv.kernel_projection.weight");
        L.aconv_base = (float *) w_.upload(b + "attention_conv.base_kernel").data;
        L.mconv_base = (float *) w_.upload(b + "mlp_conv.base_kernel").data;
    }

    const int W = kMaxWindowDense, B = kBlock, maxk = win_ + B;
    x_.alloc(B * E); xn_.alloc(B * E); xc_.alloc(B * E); a_.alloc(B * E);
    dyn_.alloc(B * 4 * (E / gsize_));
    q_.alloc(B * H * D); o_.alloc(B * H * D); k_.alloc(W * HKV * D); v_.alloc(W * HKV * D);
    g_.alloc(B * FF); u_.alloc(B * FF); h_.alloc(B * FF);
    scores_.alloc((size_t) H * B * maxk);
    logits_.alloc((size_t) (B - 1) * nvocab_);
    hp_.alloc((B - 1) * rank_);
    cids_.alloc((B - 1) * topk_); cvals_.alloc((B - 1) * topk_);
    out_.alloc(B); prob_.alloc(B); qout_.alloc((B - 1) * topk_);
    CUDA_CHECK(cudaHostAlloc((void **) &h_q_, (B - 1) * topk_ * 4, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &h_cand_, (B - 1) * topk_ * 4, cudaHostAllocMapped));
    cn_.alloc(W * E);
    const int64_t maxcols = std::max<int64_t>({(int64_t) E, (int64_t) FF, (int64_t) H * D});
    actq_.alloc(W * maxcols); actd_.alloc(W * maxcols / 32); acts_.alloc(W * maxcols / 16); acth_.alloc(W * maxcols);
    act_.q = actq_; act_.d = actd_; act_.s = acts_; act_.h = acth_;
    par_.alloc(1);
    cpos_.alloc(1);
    CUDA_CHECK(cudaHostAlloc((void **) &h_par_, sizeof(DFlashParams), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &h_cpos_, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &h_out_, B * 4, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &h_prob_, B * 4, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &h_tok_, B * 4, cudaHostAllocMapped));
    for (int i = 0; i < B; ++i) h_tok_[i] = mask_;

    kc_.resize(slots);
    vc_.resize(slots);
    for (int s = 0; s < slots; ++s) {
        kc_[s].alloc((size_t) NL * kRing * HKV * D);
        vc_[s].alloc((size_t) NL * kRing * HKV * D);
    }
    fed_.assign(slots, 0);
    dists_.assign(slots, {});
    valid_lo_.assign(slots, 0);
    graphs_.assign(slots, nullptr);
    multi_drafts_.assign(slots, {});
    multi_probs_.assign(slots, {});
    multi_next_.assign(slots, 0);
    loaded_ = true;
    if (verbose)
        fprintf(stderr, "bnk: DFlash2 drafter %s: %d layers, taps %d/%d/%d/%d/%d, block %d, %.2f GiB (K/V %.0f MiB per slot)\n",
                path.c_str(), NL, taps_[0], taps_[1], taps_[2], taps_[3], taps_[4], kBlock, vram_bytes() / 1073741824.0,
                2.0 * kc_[0].n * sizeof(half) / 1048576.0);
}

// context rows (at most kMaxWindowDense): hidden_norm, then each layer's K and V (normed, roped) into the ring
void DenseDFlash::ctx_rows(const float * acc_rows, int n, int pos0) {
    *h_cpos_ = pos0;
    CUDA_CHECK(cudaMemcpyAsync(cpos_.p, h_cpos_, sizeof(int), cudaMemcpyHostToDevice, st_));
    rmsnorm_rows(acc_rows, F(hidden_norm_), cn_, n, E, E, E, eps_, st_);
    const size_t ls = (size_t) kRing * HKV * D;
    for (int l = 0; l < (int) layers_.size(); ++l) {
        const Layer & L = layers_[l];
        gemv_auto(L.wk, cn_, E, n, k_, HKV * D, false, act_, st_);
        gemv_auto(L.wv, cn_, E, n, v_, HKV * D, false, act_, st_);
        dflash_headnorm_rope(k_, n, HKV, D, F(L.k_norm), cpos_, theta_, eps_, st_);
        dflash_kv_store(k_, v_, kc_[cur_].p + l * ls, vc_[cur_].p + l * ls, n, HKV * D, kRing, cpos_, st_);
    }
    // the copy of the position must not be overwritten before the kernels above read it
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void DenseDFlash::feed(const float * acc_rows, int n, int pos0) {
    DeviceScope ds(dev_);
    // rows below fed_ are already in (the first draft after a prompt passes its last row again)
    const int skip = std::max(0, std::min(n, fed_[cur_] - pos0));
    if (pos0 + skip > fed_[cur_]) valid_lo_[cur_] = pos0 + skip;   // a gap: nothing below it is this history's
    for (int r0 = skip; r0 < n; r0 += kMaxWindowDense) {
        const int rr = std::min(kMaxWindowDense, n - r0);
        ctx_rows(acc_rows + (size_t) r0 * E, rr, pos0 + r0);
    }
    fed_[cur_] = std::max(fed_[cur_], pos0 + n);
}

// The block pass (graphed per slot: the ring is the slot's): embed, 5 layers, norm, head, top-k, selector.
void DenseDFlash::enqueue_block() {
    const int B = kBlock, maxk = win_ + B, G = E / gsize_;
    const size_t ls = (size_t) kRing * HKV * D;
    const int * spos = (const int *) par_.p;   // DFlashParams::s first
    dequant_gather(main_->tok_embd, h_tok_, B, x_, st_);
    for (int l = 0; l < (int) layers_.size(); ++l) {
        const Layer & L = layers_[l];
        half * kr = kc_[cur_].p + l * ls, * vr = vc_[cur_].p + l * ls;
        rmsnorm_rows(x_, F(L.in_norm), xn_, B, E, E, E, eps_, st_);
        gemv_auto(L.aconv_proj, xn_, E, B, dyn_, 4 * G, false, act_, st_);
        dflash_conv(xn_, dyn_, L.aconv_base, xc_, B, E, gsize_, 0, st_);
        gemv_auto(L.wq, xc_, E, B, q_, H * D, false, act_, st_);
        gemv_auto(L.wk, xc_, E, B, k_, HKV * D, false, act_, st_);
        gemv_auto(L.wv, xc_, E, B, v_, HKV * D, false, act_, st_);
        dflash_headnorm_rope(q_, B, H, D, F(L.q_norm), spos, theta_, eps_, st_);
        dflash_headnorm_rope(k_, B, HKV, D, F(L.k_norm), spos, theta_, eps_, st_);
        dflash_kv_store(k_, v_, kr, vr, B, HKV * D, kRing, spos, st_);
        dflash_attention(q_, kr, vr, scores_, o_, par_, B, H, HKV, D, kRing, win_, maxk, st_);
        gemv_auto(L.wo, o_, H * D, B, a_, E, false, act_, st_);
        dflash_conv(a_, dyn_, L.aconv_base, xc_, B, E, gsize_, 1, st_);
        add_rows(x_, xc_, (int64_t) B * E, st_);
        rmsnorm_rows(x_, F(L.post_norm), xn_, B, E, E, E, eps_, st_);
        gemv_auto(L.mconv_proj, xn_, E, B, dyn_, 4 * G, false, act_, st_);
        dflash_conv(xn_, dyn_, L.mconv_base, xc_, B, E, gsize_, 0, st_);
        gemv_auto(L.gate, xc_, E, B, g_, FF, false, act_, st_);
        gemv_auto(L.up, xc_, E, B, u_, FF, false, act_, st_);
        silu_mul(g_, u_, h_, (int64_t) B * FF, st_);
        gemv_auto(L.down, h_, FF, B, a_, E, false, act_, st_);
        dflash_conv(a_, dyn_, L.mconv_base, xc_, B, E, gsize_, 1, st_);
        add_rows(x_, xc_, (int64_t) B * E, st_);
    }
    // the drafted rows (all but the anchor's): norm, the target's head, the selector
    rmsnorm_rows(x_ + E, F(norm_), xn_, B - 1, E, E, E, eps_, st_);
    gemv_auto(main_->output, xn_, E, B - 1, logits_, nvocab_, false, act_, st_);
    topk_rows(logits_, B - 1, nvocab_, topk_, cids_, cvals_, st_);
    gemv_auto(sel_proj_, xn_, E, B - 1, hp_, rank_, false, act_, st_);
    dflash_select(cids_, cvals_, hp_, sel_pre_, sel_succ_, par_, B - 1, topk_, rank_, out_, prob_, qout_, st_);
    CUDA_CHECK(cudaMemcpyAsync(h_out_, out_.p, (B - 1) * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(h_prob_, prob_.p, (B - 1) * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(h_q_, qout_.p, (B - 1) * topk_ * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(h_cand_, cids_.p, (B - 1) * topk_ * 4, cudaMemcpyDeviceToHost, st_));
}

void DenseDFlash::draft_block(int anchor, int s) {
    h_tok_[0] = anchor;
    h_par_->s = s;
    h_par_->lo = std::max(valid_lo_[cur_], s - (win_ - 1));
    h_par_->nkeys = s + kBlock - h_par_->lo;
    h_par_->anchor = anchor;
    h_par_->temp = temp_;
    for (int i = 0; i < kBlock; ++i) {   // splitmix64 uniforms in [0, 1)
        uint64_t z = (rng_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        h_par_->u[i] = (float) ((z ^ (z >> 31)) >> 40) / 16777216.f;
    }
    CUDA_CHECK(cudaMemcpyAsync(par_.p, h_par_, sizeof(DFlashParams), cudaMemcpyHostToDevice, st_));
    static const bool graphs = !getenv("BNK_DFLASH_GRAPH") || atoi(getenv("BNK_DFLASH_GRAPH")) != 0;
    cudaGraphExec_t & ge = graphs_[cur_];
    if (graphs && !ge) {
        cudaGraph_t gr;
        CUDA_CHECK(cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal));
        enqueue_block();
        CUDA_CHECK(cudaStreamEndCapture(st_, &gr));
        CUDA_CHECK(cudaGraphInstantiate(&ge, gr, 0));
        cudaGraphDestroy(gr);
    }
    if (graphs) CUDA_CHECK(cudaGraphLaunch(ge, st_));
    else enqueue_block();
    CUDA_CHECK(cudaStreamSynchronize(st_));
    ndrafts_ = kBlock - 1;
    for (int i = 0; i < ndrafts_; ++i) {
        drafts_[i] = h_out_[i];
        probs_[i] = h_prob_[i];
    }
    auto & ds = dists_[cur_];
    ds.assign(temp_ > 0.f ? ndrafts_ : 0, {});
    for (int i = 0; i < (int) ds.size(); ++i)
        for (int j = 0; j < topk_; ++j)
            if (h_q_[i * topk_ + j] > 0.f) ds[i].emplace_back(h_cand_[i * topk_ + j], h_q_[i * topk_ + j]);
    next_ = 0;
}

// rows R_rows .. are the target's verified rows at positions cell0..: their accumulator rows become context; the
// block is anchored at the last next token (position cell0 + n)
int DenseDFlash::run(const float * R_rows, const int32_t * next_tokens, int n, int cell0, bool draft, float * prob) {
    DeviceScope ds(dev_);
    const double t0 = now_ms();
    const int64_t row0 = (R_rows - res_) / E;
    if (row0 < 0 || row0 + n > kMaxWindowDense) throw std::runtime_error("dflash: rows outside the decode window");
    feed(acc_ + row0 * E, n, cell0);
    int d = -1;
    if (draft) {
        draft_block(next_tokens[n - 1], cell0 + n);
        d = drafts_[0];
        if (prob) *prob = probs_[0];
        next_ = 1;
    }
    ms += now_ms() - t0;
    calls++;
    return d;
}

// the block's next draft (the selector's path already chose it given the previous one)
int DenseDFlash::step(int32_t, int, float * prob) {
    if (next_ >= ndrafts_) {
        if (prob) *prob = 0.f;
        return -1;
    }
    if (prob) *prob = probs_[next_];
    return drafts_[next_++];
}

void DenseDFlash::run_multi(const std::vector<RunItem> & items, int32_t * tok_out, float * prob_out) {
    const int keep = cur_;
    multi_drafts_.resize(items.size());
    multi_probs_.resize(items.size());
    multi_next_.assign(items.size(), 1);
    mmap_.resize(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        const RunItem & it = items[i];
        select(it.slot);
        tok_out[i] = run(it.R_rows, it.tokens, it.n, it.cell0, true, &prob_out[i]);
        multi_drafts_[i].assign(drafts_, drafts_ + ndrafts_);
        multi_probs_[i].assign(probs_, probs_ + ndrafts_);
        mmap_[i] = (int) i;
    }
    select(keep);
}

void DenseDFlash::step_multi(const std::vector<StepItem> & items, int32_t * tok_out, float * prob_out) {
    std::vector<int> map(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        const int p = map[i] = mmap_.at(items[i].prev);   // the item's run_multi index
        int & nx = multi_next_[p];
        if (nx < (int) multi_drafts_[p].size()) {
            tok_out[i] = multi_drafts_[p][nx];
            prob_out[i] = multi_probs_[p][nx];
            ++nx;
        } else {
            tok_out[i] = -1;
            prob_out[i] = 0.f;
        }
    }
    mmap_ = map;
}

}  // namespace bnk
