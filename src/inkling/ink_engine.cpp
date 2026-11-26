#include "inkling/ink_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "inkling/ink_attn.h"
#include "inkling/ink_ops.h"
#include "kernels/gemm.h"
#include "kernels/ops.h"

namespace bnk {

namespace {
constexpr int kInkCkpts = 4;   // snapshots kept (each holds the window rings: ~250 MB of host RAM per rank)
}

InkEngine::~InkEngine() {
    rank1_.reset();
    DeviceScope ds(dev_);
    for (auto & sl : graphs_)
        for (auto & row : sl)
            for (auto & g : row)
                if (g) cudaGraphExecDestroy(g);
    for (auto & evs : ev_)
        for (cudaEvent_t e : evs) cudaEventDestroy(e);
    for (cudaEvent_t e : {ev_head_, ev_fork_, ev_join_})
        if (e) cudaEventDestroy(e);
    if (tp_rank_ == 0) {   // the pinned inputs belong to rank 0 (rank 1 reads them)
        if (h_tok_) cudaFreeHost(h_tok_);
        if (h_pos_) cudaFreeHost(h_pos_);
        if (h_tau_) cudaFreeHost(h_tau_);
    }
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
    // the GPUs: --gpus (-1 or empty: every GPU, up to two); tensor parallel over two (BNK_INK_TP=0: one)
    std::vector<int> gpus;
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (opt.gpus.empty() || opt.gpus[0] < 0) {
        for (int d = 0; d < std::min(ndev, 2); ++d) gpus.push_back(d);
    } else {
        gpus = opt.gpus;
    }
    if (env_int("BNK_INK_TP", 1) == 0 || gpus.size() < 2) gpus.resize(1);
    if (gpus.size() > 2) gpus.resize(2);
    const int n = (int) gpus.size();
    load_rank(path, opt, gpus[0], n, 0);
    if (n == 2) {
        rank1_ = std::make_unique<InkEngine>();
        rank1_->load_rank(path, opt, gpus[1], n, 1);
        peer_ = rank1_.get();
        rank1_->peer_ = this;
        int ab = 0, ba = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&ab, dev_, rank1_->dev_));
        CUDA_CHECK(cudaDeviceCanAccessPeer(&ba, rank1_->dev_, dev_));
        if (!ab || !ba) throw std::runtime_error("inkling: tensor parallel needs peer access between the GPUs (docs/DUAL-GPU.md)");
        for (InkEngine * r : ranks()) {
            DeviceScope ds(r->dev_);
            const cudaError_t e = cudaDeviceEnablePeerAccess(r->peer_->dev_, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            else CUDA_CHECK(e);
            r->p2p_ = true;
        }
        // rank 1 reads rank 0's pinned inputs
        rank1_->h_tok_ = h_tok_;
        rank1_->h_pos_ = h_pos_;
        rank1_->h_tau_ = h_tau_;
    }
    CUDA_CHECK(cudaSetDevice(dev_));   // rank 0's GPU current again (rank 1's load set its own)
    if (!opt.mmproj.empty()) {
        DeviceScope ds(dev_);
        images_ = std::make_unique<ImageStore>();
        images_->cache_limit = (size_t) (opt.vision_cache_gib * 1073741824.0);
        images_->open(opt.mmproj, dev_, nullptr, opt.verbose);
        if (images_->cfg().kind != Projector::Inkling2V || images_->cfg().out_dim != model_.cfg.E)
            throw std::runtime_error("vision: " + opt.mmproj + " is not Inkling 2's encoder (inkling2v, out " +
                                     std::to_string(model_.cfg.E) + ")");
    }
    if (!opt.mtp.empty() && env_int("BNK_INK_SPARK", 1) != 0) {
        DeviceScope ds(dev_);
        spark_ = std::make_unique<InkSpark>();
        std::vector<InkSpark::HeadPart> heads;   // each rank's vocabulary rows of the head, where they are
        for (InkEngine * r : ranks()) heads.push_back({r->dev_, r->st_, r->model_.output, r->model_.cfg.vocab0});
        spark_->load(opt.mtp, model_.tok_embd, model_.cfg.n_vocab, heads, opt.max_ctx, chunk_, opt.verbose, n_slots_);
        std::fill(std::begin(tap_of_visit_), std::end(tap_of_visit_), -1);
        const auto & tv = spark_->target_visits();
        for (int k = 0; k < (int) tv.size(); ++k)
            if (tv[k] < model_.cfg.n_visits()) tap_of_visit_[tv[k]] = k;
            else if (tv[k] != model_.cfg.n_visits() || k != InkSpark::kTaps - 1)
                throw std::runtime_error("inkling: the drafter's last tap must be the normed output");
        feat_.alloc((size_t) chunk_ * InkSpark::kTaps * model_.cfg.E);
        if (n_slots_ > 1) bfeat_.alloc((size_t) kInkWin * InkSpark::kTaps * model_.cfg.E);
        feat_out_ = feat_.p;
    }
    if (opt.verbose) {
        for (InkEngine * r : ranks()) {
            DeviceScope ds(r->dev_);
            size_t fr = 0, tot = 0;
            cudaMemGetInfo(&fr, &tot);
            fprintf(stderr, "bnk: inkling: GPU %d: context %d, K/V %.2f GiB (window rings of %d cells), prompt chunks of %d, "
                            "%.1f GiB free\n", r->dev_, opt.max_ctx, r->kv_bytes_ / double(1 << 30), r->ring_, r->chunk_,
                    fr / double(1 << 30));
        }
        if (n == 2) fprintf(stderr, "bnk: inkling: tensor parallel over GPUs %d and %d (peer copies)\n", gpus[0], gpus[1]);
    }
}

void InkEngine::alloc_visits(std::vector<Visit> & vs) {
    const InkConfig & c = model_.cfg;
    vs.resize(c.n_visits());   // (from empty: built in place)
    for (int v = 0; v < c.n_visits(); ++v) {
        const int ip = v < c.n_layer ? v : c.loop_start + (v - c.n_layer) % c.n_core();
        Visit & V = vs[v];
        V.window = c.swa[ip] ? c.n_swa : 0;
        V.ring = c.swa[ip] ? ring_ : opt_.max_ctx;
        V.K.alloc((size_t) V.ring * c.kv_dim());
        V.V.alloc((size_t) V.ring * c.kv_dim());
        V.conv.alloc((size_t) 3 * conv_ch_);
        V.win_in.alloc((size_t) kInkWin * conv_ch_);
        kv_bytes_ += (size_t) 2 * V.ring * c.kv_dim() * sizeof(half);
    }
}

void InkEngine::select(int slot) {
    if (slot == cur_) return;
    if (slot < 0 || slot >= n_slots_) throw std::runtime_error("inkling: slot " + std::to_string(slot) + " out of range");
    for (InkEngine * r : ranks()) r->select_here(slot);
    if (spark_) spark_->select(slot);
}

// park the current conversation in its entry, bring slot's in (vector swaps: the buffers change hands; the streams
// are in order, so work already queued on the old buffers is unaffected)
void InkEngine::select_here(int slot) {
    for (int pass = 0; pass < 2; ++pass) {
        Parked & pk = parked_[pass == 0 ? cur_ : slot];
        visits_.swap(pk.visits);
        history_.swap(pk.history);
        std::swap(pending_T_, pk.pending_T);
        ckpts_.swap(pk.ckpts);
        std::swap(ckpt_age_, pk.ckpt_age);
    }
    cur_ = slot;
}

void InkEngine::load_rank(const std::string & path, const EngineOptions & opt, int dev, int tp_n, int tp_rank) {
    opt_ = opt;
    dev_ = dev;
    tp_n_ = tp_n;
    tp_rank_ = tp_rank;
    CUDA_CHECK(cudaSetDevice(dev_));
    DeviceScope ds(dev_);
    model_.load(path, opt.verbose, tp_n, tp_rank);
    const InkConfig & c = model_.cfg;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    chunk_ = std::max(kInkWin, std::min(env_int("BNK_INK_CHUNK", 512), 2048));
    ring_ = c.n_swa + chunk_;
    conv_ch_ = 2 * c.kv_dim() + 2 * c.E;
    const int W = chunk_, E = c.E, F = c.n_ff, R = W * c.n_used;

    n_slots_ = std::max(1, opt.slots);
    if (n_slots_ > kInkMaxSlots) throw std::runtime_error("inkling: at most " + std::to_string(kInkMaxSlots) + " slots");
    alloc_visits(visits_);
    parked_.resize(n_slots_);
    for (int sl = 1; sl < n_slots_; ++sl) {
        alloc_visits(parked_[sl].visits);
        parked_[sl].ckpts.resize(kInkCkpts);
    }
    x_.alloc((size_t) W * E); ffn_inp_.alloc((size_t) W * E); tmp_.alloc((size_t) W * E); o_.alloc((size_t) W * E);
    moe_.alloc((size_t) W * E);
    mixed_.alloc((size_t) 4 * W * E); msg_.alloc((size_t) 4 * W * E);
    for (auto & b : ins_) b.alloc((size_t) W * E);
    attn_.alloc((size_t) W * c.q_dim());
    conv_in_.alloc((size_t) W * std::max(E, c.q_dim())); kc_.alloc((size_t) W * c.kv_dim()); vc_.alloc((size_t) W * c.kv_dim());
    ctrl_in_.alloc((size_t) kInkCtrlJobs * W * 2 * E); ctrl_g_.alloc((size_t) kInkCtrlJobs * W * c.gate_rank);
    ctrl_gf_.alloc((size_t) kInkCtrlJobs * W * 2 * E);
    conv_scratch_.alloc((size_t) kInkConvJobs * W * std::max(E, c.kv_dim()));
    int max_proj = 0;
    for (const auto & L : model_.layers) max_proj = std::max(max_proj, (int) L.proj.rows);
    proj_.alloc((size_t) W * max_proj);
    tau_.alloc(W);
    mix_sc_.alloc(ink_mix_scratch_floats(W));
    // prompt chunks: up to 8 splits of W rows; decode windows: up to kInkAttnSplits of kInkWin rows (fewer row-splits)
    static_assert(kInkWin * kInkAttnSplits <= 512 * 8, "attention scratch");
    attn_scratch_.alloc(ink_attn_scratch_floats(W, c.n_head, std::max(c.n_swa, c.rel_extent), std::max(8, kInkWin * kInkAttnSplits / W + 1)));
    rlog_.alloc((size_t) W * c.n_expert);
    gx_.alloc((size_t) R * E); gu_.alloc((size_t) R * 2 * F); gh_.alloc((size_t) R * F); gd_.alloc((size_t) R * E);
    blocks_buf_.alloc((size_t) kMaxBlocks * W * E);
    logits_.alloc((size_t) kInkWin * (tp_rank == 0 ? c.n_vocab : c.vocab_n));
    tok_dev_.alloc(W); ids_dev_.alloc(R); rows_dev_.alloc(R); inv_dev_.alloc(R); argmax_dev_.alloc(kInkWin);
    w_dev_.alloc(R); wg_dev_.alloc(R);
    h_ids_.resize(R); h_w_.resize(R);
    if (tp_rank == 0) {
        CUDA_CHECK(cudaHostAlloc((void **) &h_tok_, W * 4, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc((void **) &h_pos_, 4, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc((void **) &h_tau_, W * 4, cudaHostAllocPortable));
    }
    d_pos_.alloc(1);
    plan_buf_.alloc(sizeof(InkMoePlan));
    mh_.alloc((size_t) kInkMaxPairs * F); my_.alloc((size_t) kInkMaxPairs * E); mgu_.alloc((size_t) kInkMaxPairs * 2 * F);
    mq_.alloc((size_t) kInkMaxPairs * E); mqd_.alloc((size_t) kInkMaxPairs * E / 32);
    xh_.alloc((size_t) W * E); hh_.alloc((size_t) R * F); dh_.alloc((size_t) R * E);
    tc_items_gu_.alloc(c.n_expert); tc_items_dn_.alloc(c.n_expert); tc_blocks_.alloc(R / kTcTile + c.n_expert);
    moe_tc_set_situ(c.situ_beta, c.situ_lin);
    if (tp_n > 1) {
        for (auto & pt : inbox_)
            for (auto & b : pt) b.alloc((size_t) W * E);
        for (auto & evs : ev_) {
            evs.resize(c.n_visits());
            for (cudaEvent_t & e : evs) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_head_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming));
        if (tp_rank == 1) head_part_.alloc((size_t) kInkWin * c.vocab_n);
    }
    graphs_on_ = opt.use_graphs && env_int("BNK_INK_GRAPHS", 1) != 0;
    const int64_t maxcols = 2 * E;
    actq_.alloc((size_t) kInkWin * maxcols); actd_.alloc((size_t) kInkWin * maxcols / 32);
    acts_.alloc((size_t) kInkWin * maxcols / 16); acth_.alloc((size_t) kInkWin * maxcols);
    act_.q = actq_; act_.d = actd_; act_.s = acts_; act_.h = acth_;
    gemm_.init(st_, (size_t) std::max<int64_t>({(int64_t) max_proj * E, (int64_t) c.q_dim() * E, (int64_t) 2 * F * E}),
               (size_t) W * 2 * E);
    ckpts_.resize(kInkCkpts);
    lg_ = logits_.p;
    feat_out_ = feat_.p;
    reset_here();
}

// ------------------------------------------------------------------------------------------------ state
void InkEngine::reset() {
    history_.clear();
    pending_T_ = 0;
    for (auto & k : ckpts_) k.pos = -1;
    for (InkEngine * r : ranks()) r->reset_here();
    if (spark_) spark_->restart(0);
}

void InkEngine::reset_here() {
    DeviceScope ds(dev_);
    for (auto & V : visits_) CUDA_CHECK(cudaMemsetAsync(V.conv.p, 0, V.conv.n * 4, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

// a snapshot: every rank's conv states and window rings into its slot `slot`
void InkEngine::save_ckpt(int slot, int p) {
    DeviceScope ds(dev_);
    Ckpt & k = ckpts_[slot];
    size_t nconv = 0, nring = 0;
    for (auto & V : visits_) {
        nconv += V.conv.n;
        if (V.window) nring += V.K.n + V.V.n;
    }
    k.conv.resize(nconv);
    k.ring.resize(nring);
    size_t oc = 0, orr = 0;
    for (auto & V : visits_) {
        CUDA_CHECK(cudaMemcpyAsync(k.conv.data() + oc, V.conv.p, V.conv.n * 4, cudaMemcpyDeviceToHost, st_));
        oc += V.conv.n;
        if (V.window) {
            CUDA_CHECK(cudaMemcpyAsync(k.ring.data() + orr, V.K.p, V.K.n * 2, cudaMemcpyDeviceToHost, st_));
            orr += V.K.n;
            CUDA_CHECK(cudaMemcpyAsync(k.ring.data() + orr, V.V.p, V.V.n * 2, cudaMemcpyDeviceToHost, st_));
            orr += V.V.n;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    k.pos = p;
}

void InkEngine::load_ckpt(int slot) {
    DeviceScope ds(dev_);
    const Ckpt & k = ckpts_[slot];
    size_t oc = 0, orr = 0;
    for (auto & V : visits_) {
        CUDA_CHECK(cudaMemcpyAsync(V.conv.p, k.conv.data() + oc, V.conv.n * 4, cudaMemcpyHostToDevice, st_));
        oc += V.conv.n;
        if (V.window) {
            CUDA_CHECK(cudaMemcpyAsync(V.K.p, k.ring.data() + orr, V.K.n * 2, cudaMemcpyHostToDevice, st_));
            orr += V.K.n;
            CUDA_CHECK(cudaMemcpyAsync(V.V.p, k.ring.data() + orr, V.V.n * 2, cudaMemcpyHostToDevice, st_));
            orr += V.V.n;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void InkEngine::checkpoint() {
    const int p = pos();
    if (p == 0) return;
    for (auto & k : ckpts_)
        if (k.pos == p) { k.age = ++ckpt_age_; return; }
    int slot = -1;
    for (int i = 0; i < (int) ckpts_.size(); ++i)
        if (ckpts_[i].pos < 0) { slot = i; break; }
    if (slot < 0)
        slot = (int) (std::min_element(ckpts_.begin(), ckpts_.end(), [](const Ckpt & a, const Ckpt & b) { return a.age < b.age; }) - ckpts_.begin());
    for (InkEngine * r : ranks()) r->save_ckpt(slot, p);
    if (spark_) spark_->save(ckpts_[slot].spark);
    ckpts_[slot].age = ++ckpt_age_;
}

int InkEngine::rollback(int max_pos) {
    int best = -1;
    for (int i = 0; i < (int) ckpts_.size(); ++i) {
        const auto & k = ckpts_[i];
        if (k.pos >= 0 && k.pos <= max_pos && k.pos <= pos() && (best < 0 || k.pos > ckpts_[best].pos)) best = i;
    }
    if (best < 0) return -1;
    for (InkEngine * r : ranks()) r->load_ckpt(best);
    const int p = ckpts_[best].pos;
    if (spark_) {
        spark_->load_state(ckpts_[best].spark);
        if (spark_->fed() != p) spark_->restart(p);
    }
    history_.resize(p);
    pending_T_ = 0;
    ckpts_[best].age = ++ckpt_age_;
    for (auto & k : ckpts_)
        if (k.pos > p) k.pos = -1;
    for (InkEngine * r : ranks())
        for (auto & k : r->ckpts_)
            if (k.pos > p) k.pos = -1;
    return p;
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

// ------------------------------------------------------------------------------------------------ exchange
// The all-reduce of a [T][E] partial with the peer, in two phases so that each rank records its event before the
// other waits for it (a capture may not wait on an event recorded later): send pushes this rank's partial into the
// peer's inbox for (point, visit parity) and marks it; recv waits for the peer's mark and adds the inbox. Both
// ranks add the same two fp32 values, so their replicated state stays bit for bit identical.
void InkEngine::send(int point, int v, const float * buf, int T) {
    if (tp_n_ == 1) return;
    DeviceScope ds(dev_);
    CUDA_CHECK(cudaMemcpyAsync(peer_->inbox_[point][v & 1].p, buf, (size_t) T * model_.cfg.E * 4, cudaMemcpyDeviceToDevice, st_));
    CUDA_CHECK(cudaEventRecord(ev_[point][v], st_));
}

void InkEngine::recv(int point, int v, float *, int) {
    if (tp_n_ == 1) return;
    DeviceScope ds(dev_);
    CUDA_CHECK(cudaStreamWaitEvent(st_, peer_->ev_[point][v], 0));
    peer_in_[point] = inbox_[point][v & 1];   // added by the next conv as it reads its input
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
                    float * out, const float * const * norms) {
    InkStates st;
    for (const float * b : blocks) st.p[st.n++] = b;
    st.p[st.n++] = cur;
    static const bool one = env_int("BNK_INK_MIX1", 1) != 0;   // decode windows: the one-block mix (+ norms)
    if (T <= kInkWin && one) {
        ink_mix_norm(st, query, nq, T, model_.cfg.E, norms, model_.cfg.rms_eps, out, st_);
        return;
    }
    ink_mix(st, query, nq, T, model_.cfg.E, out, mix_sc_, st_);
    if (norms) {
        const size_t TE = (size_t) T * model_.cfg.E;
        for (int j = 0; j < nq; ++j)
            if (norms[j]) ink_rms(out + j * TE, model_.cfg.E, norms[j], out + j * TE, model_.cfg.E, T, model_.cfg.E, model_.cfg.rms_eps, st_);
    }
}

void InkEngine::control(float * cur, const float * msg, const InkLoop & loop, int j, int T) {
    const InkConfig & c = model_.cfg;
    const int E = c.E;
    ink_concat_rms2(cur, msg, ctrl_in_, T, E, 1e-6f, st_);
    mm(loop.down[j], ctrl_in_, 2 * E, T, ctrl_g_, c.gate_rank);
    mm(loop.gf[j], ctrl_g_, c.gate_rank, T, ctrl_gf_, 2 * E);   // gain | forget
    ink_ctrl_combine(cur, cur, msg, ctrl_gf_, T, E, st_);
}

// the conv of channels [off, off + C) of visit v: x rows (stride ldx) -> y [T][C]; keep: the state advances now,
// else the rows wait in the visit's window inputs for the commit
void InkEngine::conv(int v, int off, int C, const float * x, int64_t ldx, float * y, int T, const float * w,
                     const float * dyn, int64_t ld_dyn, int H, bool keep) {
    Visit & V = visits_[v];
    if (!keep)
        CUDA_CHECK(cudaMemcpy2DAsync(V.win_in.p + off, (size_t) conv_ch_ * 4, x, (size_t) ldx * 4, (size_t) C * 4, T,
                                     cudaMemcpyDeviceToDevice, st_));
    const float scale = 1.f / sqrtf((dyn ? 2.f : 1.f) * model_.cfg.n_conv + 1.f);
    ink_conv(x, ldx, y, T, C, V.conv.p + off, conv_ch_, w, dyn, (int) ld_dyn, H, scale, keep ? T : 0, st_);
}

void InkEngine::moe(const InkLayer & L, const float * x, int T, float * out) {
    const InkConfig & c = model_.cfg;
    const int E = c.E, F = c.n_ff, k = c.n_used, n = T * k;
    mm(L.router, x, E, T, rlog_, c.n_expert);
    static const bool host_moe = env_int("BNK_INK_HOST_MOE", 0) != 0;   // tests: the grouped GEMV / GEMM path
    if (T <= kInkWin && !host_moe) {
        InkMoeArgs a;
        a.logits = rlog_; a.bias = L.exp_bias; a.w_scale = c.w_scale;   // routed with the plan
        a.T = T; a.k = k; a.n_exp = c.n_expert; a.E = E; a.F = F; a.x = x;
        a.gate_up = (const uint8_t *) L.gate_up.data; a.gu_type = L.gate_up.type; a.gu_row_bytes = L.gate_up.row_bytes;
        a.gu_scale = L.d_gu_scale;
        a.down = (const uint8_t *) L.down.data; a.down_type = L.down.type; a.down_row_bytes = L.down.row_bytes;
        a.down_scale = L.d_down_scale;
        a.beta = c.situ_beta; a.lb = c.situ_lin;
        a.plan = (InkMoePlan *) plan_buf_.p; a.gu = mgu_; a.h = mh_; a.y = my_; a.out = out; a.q = mq_; a.qd = mqd_;
        ink_moe_window(a, st_);
        return;
    }
    ink_route(rlog_, L.exp_bias, T, c.n_expert, k, c.w_scale, ids_dev_, w_dev_, st_);
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
    static const bool tc_off = env_int("BNK_INK_MOE_TC", 1) == 0;   // tests: the grouped GEMV / GEMM path
    if (!tc_off && L.gate_up.type == 40 && L.down.type == 40) {   // NVFP4
        // the tensor-core experts: one launch for every (expert, 64-token tile, row tile) of gate | up, one of down
        std::vector<TcItem> igu, idn;
        std::vector<TcBlock> bl;
        for (int e = 0; e < c.n_expert; ++e) {
            const int m = cnt[e + 1] - cnt[e];
            if (!m) continue;
            const int item = (int) igu.size();
            igu.push_back(TcItem{(const uint8_t *) L.gate_up.data + (size_t) e * 2 * F * L.gate_up.row_bytes, cnt[e], m, e});
            idn.push_back(TcItem{(const uint8_t *) L.down.data + (size_t) e * E * L.down.row_bytes, cnt[e], m, e});
            for (int t0 = 0; t0 < m; t0 += kTcTile) bl.push_back(TcBlock{item, t0});
        }
        CUDA_CHECK(cudaMemcpyAsync(tc_items_gu_.p, igu.data(), igu.size() * sizeof(TcItem), cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaMemcpyAsync(tc_items_dn_.p, idn.data(), idn.size() * sizeof(TcItem), cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaMemcpyAsync(tc_blocks_.p, bl.data(), bl.size() * sizeof(TcBlock), cudaMemcpyHostToDevice, st_));
        MoeLayerDesc dg{}, dd{};
        dg.gate_bytes = (size_t) F * L.gate_up.row_bytes;   // the up rows follow the gate rows
        dg.grow = L.gate_up.row_bytes;
        dg.gate_type = L.gate_up.type;
        dg.scales.gate = dg.scales.up = L.d_gu_scale;
        dd.drow = L.down.row_bytes;
        dd.down_type = L.down.type;
        dd.scales.down = L.d_down_scale;
        f32_to_f16(x, xh_, (int64_t) T * E, st_);
        moe_tc_gu(tc_items_gu_, tc_blocks_, (int) bl.size(), dg, xh_, E, rows_dev_, hh_, E, F, st_);
        moe_tc_down(tc_items_dn_, tc_blocks_, (int) bl.size(), dd, hh_, dh_, E, F, st_);
        moe_gather_sum(dh_, inv_dev_, w_dev_, T, k, E, nullptr, nullptr, nullptr, out, st_);
        CUDA_CHECK(cudaStreamSynchronize(st_));   // the host vectors above go out of scope
        return;
    }
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

// ------------------------------------------------------------------------------------------------ the visit
// A layer visit (llama.cpp's llama_model_inkling2::graph::build_layer) in three phases around the two all-reduces.
// visit_a: the attention's inputs (depth mix, loop controls, norms), this rank's heads, the out projection's
// partial into conv_in_.
void InkEngine::visit_a(int v, int ip, int T, int pos0, const InkLoop * loop, bool keep) {
    DeviceScope ds(dev_);
    const InkConfig & c = model_.cfg;
    const InkLayer & L = model_.layers[ip];
    const int E = c.E, H = c.n_head, KVH = c.n_head_kv, D = c.head_dim, KVD = c.kv_dim(), QD = c.q_dim();
    const size_t TE = (size_t) T * E;
    const bool swa = L.swa;
    if (loop) mix(bank_, bank_out_, loop->msg_query, 4, T, msg_);

    // the attention's inputs, normed: the mix and its norms in one pass; with loop messages, the mix, then the q / k
    // / v controls (batched) with the norms folded into their combine; layer 0 has no mix
    float * a[3] = {ins_[0].p, ins_[1].p, ins_[2].p};
    const float * nw[3] = {swa ? L.attn_norm : L.q_norm, swa ? L.attn_norm : L.k_norm, swa ? L.attn_norm : L.v_norm};
    if (L.attn_res_query && !loop) {
        mix(blocks_, x_, L.attn_res_query, L.attn_res_nq, T, mixed_, nw);
        for (int j = 0; j < 3; ++j) a[j] = mixed_.p + (swa ? 0 : j) * TE;
    } else if (L.attn_res_query) {
        mix(blocks_, x_, L.attn_res_query, L.attn_res_nq, T, mixed_);
        const float * cur[3], * msg[3];
        for (int j = 0; j < 3; ++j) {
            cur[j] = mixed_.p + (swa ? 0 : j) * TE;
            msg[j] = msg_.p + j * TE;
        }
        controls(loop, 0, 3, cur, msg, a, nw, T);
    } else {   // layer 0: the embedding, normed
        ink_rms(x_, E, L.attn_norm, a[0], E, T, E, c.rms_eps, st_);
        a[1] = a[2] = a[0];
    }

    // the projections stacked by input: one GEMV when the three inputs are the same (window layers without loop
    // messages), else q | gate | rel | sink, k [| taps], v [| taps]; consumers read them with the stack's stride
    const int S = (int) L.proj.rows;
    float * P = proj_.p;
    if (a[0] == a[1] && a[1] == a[2]) {
        mm(L.proj, a[0], E, T, P, S);
    } else {
        mm(qmat_rows(L.proj, 0, L.off_k), a[0], E, T, P, S);
        mm(qmat_rows(L.proj, L.off_k, L.off_v - L.off_k), a[1], E, T, P + L.off_k, S);
        mm(qmat_rows(L.proj, L.off_v, S - L.off_v), a[2], E, T, P + L.off_v, S);
    }
    // k and v through their convolutions (one launch), then the clips and v's norm (one launch)
    Visit & V = visits_[v];
    {
        const float sc = 1.f / sqrtf((swa ? 1.f : 2.f) * c.n_conv + 1.f);
        InkConvJob j[2];
        for (int i = 0; i < 2; ++i) {
            j[i].x = P + (i ? L.off_v : L.off_k);
            j[i].ldx = S;
            j[i].y = i ? vc_.p : kc_.p;
            j[i].C = KVD;
            j[i].state = V.conv.p + i * KVD;
            j[i].w = i ? L.v_conv : L.k_conv;
            j[i].dyn = swa ? nullptr : P + (i ? L.off_vp : L.off_kp);
            j[i].ld_dyn = S;
            j[i].H = KVH;
            j[i].scale = sc;
            j[i].win = keep ? nullptr : V.win_in.p + i * KVD;
        }
        ink_conv_jobs(j, 2, T, conv_ch_, conv_ch_, keep ? T : 0, conv_scratch_, st_);
    }
    float * q = P, * R = P + L.off_rel;
    const float * sink = P + L.off_sink, * gate = P + L.off_gate;
    ink_post_proj(q, R, S, L.q_clip, L.rel_clip, c.qk_clip, c.rel_clip, swa ? nullptr : tau_.p, kc_, vc_, T, H, KVH, D,
                  c.n_rel, c.rms_eps, st_);

    ink_kv_store(kc_, vc_, V.K, V.V, V.ring, d_pos_, T, KVD, st_);
    {
        // key splits: decode windows a fixed number (the graph's grid), prompt chunks by their keys
        int ns;
        if (T <= kInkWin) {
            ns = V.window ? env_int("BNK_INK_SPLITS_SWA", 8) : env_int("BNK_INK_SPLITS", 32);
        } else {
            const int keys = V.window ? std::min(pos0 + T, V.window + T) : pos0 + T;
            ns = std::max(1, std::min(8, keys / 4096));   // the scratch holds 8 splits of a chunk
        }
        ns = std::max(1, std::min(ns, kInkAttnSplits));
        ink_attention_tc(q, R, sink, V.K, V.V, V.ring, L.rel_bias, c.extent(ip), c.n_rel, V.window, d_pos_, pos0, T,
                         H, KVH, D, 1.f / sqrtf((float) D), ns, attn_, attn_scratch_, st_, S, S, S);
    }
    ink_mul_sigmoid_rows(attn_, gate, S, T, QD, st_);
    mm(L.wo, attn_, QD, T, conv_in_, E);   // this rank's heads' share of the output (all-reduced next)
}

// visit_b: the out projection's sum (this rank's partial + the peer's, added as the conv reads them) through its
// conv into the residual (or a new block), the FFN's input, this rank's share of the experts into moe_
void InkEngine::visit_b(int v, int ip, int T, const InkLoop * loop, bool keep) {
    DeviceScope ds(dev_);
    const InkConfig & c = model_.cfg;
    const InkLayer & L = model_.layers[ip];
    const int E = c.E, KVD = c.kv_dim();
    const size_t TE = (size_t) T * E;
    const bool start = ip % c.res_block == 0;
    if (start) {   // the first layer of a block: its input joins the blocks (a fresh buffer per push), a new stream starts
        float * b = fresh_block();
        CUDA_CHECK(cudaMemcpyAsync(b, x_, TE * 4, cudaMemcpyDeviceToDevice, st_));
        blocks_.push_back(b);
    }
    Visit & V = visits_[v];
    InkConvJob j;
    j.x = conv_in_; j.x2 = peer_in_[0]; j.ldx = E;
    j.y = x_; j.y2 = ffn_inp_; j.res = start ? nullptr : x_.p;
    j.C = E; j.state = V.conv.p + 2 * KVD; j.w = L.attn_post_conv; j.scale = 1.f / sqrtf(c.n_conv + 1.f);
    j.win = keep ? nullptr : V.win_in.p + 2 * KVD;
    ink_conv_jobs(&j, 1, T, conv_ch_, conv_ch_, keep ? T : 0, conv_scratch_, st_);
    if (loop) {
        mix(blocks_, ffn_inp_, L.ffn_res_query, 1, T, mixed_);
        const float * cur[1] = {mixed_.p}, * msg[1] = {msg_.p + 3 * TE};
        float * out[1] = {o_.p};
        const float * nrm[1] = {L.ffn_norm};
        controls(loop, 3, 1, cur, msg, out, nrm, T);
    } else {
        const float * nw[1] = {L.ffn_norm};
        mix(blocks_, ffn_inp_, L.ffn_res_query, 1, T, o_, nw);
    }
    moe(L, o_, T, moe_);
}

// visit_c: the MoE's sum (this rank's + the peer's) through its conv, plus the FFN's input: the residual
void InkEngine::visit_c(int v, int T, bool keep) {
    DeviceScope ds(dev_);
    const InkConfig & c = model_.cfg;
    const int E = c.E, KVD = c.kv_dim();
    const int ip = v < c.n_layer ? v : c.loop_start + (v - c.n_layer) % c.n_core();
    Visit & V = visits_[v];
    InkConvJob j;
    j.x = moe_; j.x2 = peer_in_[1]; j.ldx = E;
    j.y = x_; j.res = ffn_inp_;
    j.C = E; j.state = V.conv.p + 2 * KVD + E; j.w = model_.layers[ip].ffn_post_conv; j.scale = 1.f / sqrtf(c.n_conv + 1.f);
    j.win = keep ? nullptr : V.win_in.p + 2 * KVD + E;
    ink_conv_jobs(&j, 1, T, conv_ch_, conv_ch_, keep ? T : 0, conv_scratch_, st_);
}

// Loop controls j0 .. j0 + n - 1 of a visit, batched: out[i] = control(cur[i], msg[i]), RMS-normed by norm[i]
void InkEngine::controls(const InkLoop * loop, int j0, int n, const float * const * cur, const float * const * msg,
                         float * const * out, const float * const * norm, int T) {
    const InkConfig & c = model_.cfg;
    const int E = c.E, rk = c.gate_rank;
    InkCtrlJob jb[kInkCtrlJobs];
    for (int i = 0; i < n; ++i) {
        jb[i].cur = cur[i];
        jb[i].msg = msg[i];
        jb[i].in2 = ctrl_in_.p + (size_t) i * chunk_ * 2 * E;
        jb[i].g = ctrl_g_.p + (size_t) i * chunk_ * rk;
        jb[i].gf = ctrl_gf_.p + (size_t) i * chunk_ * 2 * E;
        jb[i].out = out[i];
        jb[i].norm = norm[i];
        jb[i].down = loop->down[j0 + i].data;
        jb[i].gfw = loop->gf[j0 + i].data;
    }
    ink_ctrl_concat(jb, n, T, E, st_);
    if (T <= kInkWin) {
        ink_ctrl_down(jb, n, T, E, rk, st_);
        ink_ctrl_up(jb, n, T, E, rk, st_);
    } else {
        for (int i = 0; i < n; ++i) {
            mm(loop->down[j0 + i], jb[i].in2, 2 * E, T, jb[i].g, rk);
            mm(loop->gf[j0 + i], jb[i].g, rk, T, jb[i].gf, 2 * E);
        }
    }
    ink_ctrl_combine_norm(jb, n, T, E, c.rms_eps, st_);
}

// the window's inputs (staged in rank 0's pinned buffers) and the embedding
void InkEngine::fwd_in(int T) {
    DeviceScope ds(dev_);
    const InkConfig & c = model_.cfg;
    const int E = c.E;
    CUDA_CHECK(cudaMemcpyAsync(tok_dev_.p, h_tok_, (size_t) T * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(d_pos_.p, h_pos_, 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(tau_.p, h_tau_, (size_t) T * 4, cudaMemcpyHostToDevice, st_));
    dequant_gather(model_.tok_embd, tok_dev_, T, x_, st_);
    ink_rms(x_, E, model_.tok_norm, x_, E, T, E, c.rms_eps, st_);
    InkEngine * r0 = tp_rank_ == 0 ? this : peer_;   // the store and the sequence live on rank 0
    if (r0->window_images_)   // image cells: the encoder's rows, not normed (llama.cpp: only token rows are)
        r0->images_->put_rows(r0->seq_, *h_pos_, T, x_, st_);
    blk_next_ = 0;
    blocks_.clear();
    bank_.clear();
    bank_out_ = nullptr;
}

// the start of a pass through the core: pass 0 keeps the prelude's output; the later ones mix the previous pass's
// bank into their messages and start again from the prelude
void InkEngine::pass_begin(int pass, int T) {
    DeviceScope ds(dev_);
    const size_t TE = (size_t) T * model_.cfg.E;
    if (pass == 0) {
        prelude_ = fresh_block();
        CUDA_CHECK(cudaMemcpyAsync(prelude_, x_, TE * 4, cudaMemcpyDeviceToDevice, st_));
        prelude_blocks_ = blocks_;
        return;
    }
    bank_.assign(blocks_.begin() + prelude_blocks_.size() + 1, blocks_.end());
    bank_out_ = fresh_block();
    CUDA_CHECK(cudaMemcpyAsync(bank_out_, x_, TE * 4, cudaMemcpyDeviceToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(x_, prelude_, TE * 4, cudaMemcpyDeviceToDevice, st_));
    blocks_ = prelude_blocks_;
}

// the output: mixed over the blocks and normed (head_prep), then rows [r0, r0 + n) through this rank's vocabulary
// rows, soft-capped, into logits rows 0.. (rank 1's go into rank 0's logits after its own)
void InkEngine::head_prep(int T) {
    DeviceScope ds(dev_);
    const InkConfig & c = model_.cfg;
    const float * nw[1] = {model_.output_norm};
    mix(blocks_, x_, model_.output_res_query, 1, T, o_, nw);
}

void InkEngine::head_rows(int r0, int n) {
    DeviceScope ds(dev_);
    const InkConfig & c = model_.cfg;
    const int E = c.E, V = c.n_vocab;
    float * dst = tp_rank_ == 0 ? logits_.p + c.vocab0 : head_part_.p;
    const int ld = tp_rank_ == 0 ? V : c.vocab_n;
    mm(model_.output, o_.p + (size_t) r0 * E, E, n, dst, ld);
    if (c.softcap > 0) {
        if (ld == c.vocab_n) ink_softcap(dst, (int64_t) n * ld, c.softcap, st_);
        else
            for (int i = 0; i < n; ++i) ink_softcap(dst + (size_t) i * ld, c.vocab_n, c.softcap, st_);
    }
    if (tp_rank_ == 1) {
        CUDA_CHECK(cudaMemcpy2DAsync(peer_->logits_.p + c.vocab0, (size_t) V * 4, dst, (size_t) ld * 4,
                                     (size_t) c.vocab_n * 4, n, cudaMemcpyDeviceToDevice, st_));
        CUDA_CHECK(cudaEventRecord(ev_head_, st_));
    }
}

// The whole forward over T rows at positions pos()..pos()+T-1 (llama_model_inkling2_loop::graph).
void InkEngine::run(const int32_t * tokens, int T, bool keep) {
    const InkConfig & c = model_.cfg;
    const int pos0 = pos();
    if (T < 1 || T > chunk_) throw std::runtime_error("inkling: window out of range");
    if (pos0 + T > opt_.max_ctx) throw std::runtime_error("inkling: context full");
    window_images_ = false;
    for (int t = 0; t < T; ++t) {
        const bool img = is_image_id(tokens[t]);
        window_images_ |= img;
        h_tok_[t] = img ? 0 : tokens[t];   // an image cell's row is the encoder's (placed after the embedding)
    }
    if (window_images_) {
        if (!images()) throw std::runtime_error("inkling: an image in the prompt, but no vision encoder (mmproj)");
        seq_.assign(history_.begin(), history_.end());
        seq_.insert(seq_.end(), tokens, tokens + T);
    }
    *h_pos_ = pos0;
    lg_ = logits_.p;
    feat_out_ = feat_.p;
    for (int t = 0; t < T; ++t)
        h_tau_[t] = 1.f + c.log_alpha * logf(std::max(1.f, (float) (pos0 + t + 1) / c.log_floor));
    DeviceScope ds(dev_);
    if (T <= kInkWin && graphs_on_ && !logits_hook && !window_images_) {
        cudaGraphExec_t & g = graphs_[cur_][T][keep ? 1 : 0];
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
    logits_rows_ = T <= kInkWin ? T : logits_hook ? hook_last_.second : 1;
    CUDA_CHECK(cudaStreamSynchronize(st_));
    if (logits_hook) logits_hook(T <= kInkWin ? pos0 : hook_last_.first, logits_rows_, logits_);
    last_T = logits_rows_;
    history_.insert(history_.end(), tokens, tokens + T);
}

// The forward's GPU work for T rows whose inputs are staged (h_tok_, h_pos_, h_tau_), on every rank: each phase on
// each rank, then the exchange. Decode windows have no host round trip (one graph over both GPUs: rank 1's stream
// forks from rank 0's and joins it at the end); prompt chunks group their experts on the host.
void InkEngine::enqueue(int T, bool keep) {
    const InkConfig & c = model_.cfg;
    const int pos0 = *h_pos_;
    auto rk = ranks();
    if (rank1_) {   // rank 1 starts after rank 0's previous work (and joins a capture)
        CUDA_CHECK(cudaEventRecord(ev_fork_, st_));
        DeviceScope d1(rank1_->dev_);
        CUDA_CHECK(cudaStreamWaitEvent(rank1_->st_, ev_fork_, 0));
    }
    for (InkEngine * r : rk) r->fwd_in(T);
    // core layers in passes > 0 take loop controls: each rank its own copy of them (rank 0's read over PCIe made
    // rank 1's controls 4-10x slower)
    int order = 0;   // the visit's place in the token's order (the drafter's taps are counted so)
    auto visit = [&](int v, int ip, bool looped) {
        if (spark_ && tap_of_visit_[order] >= 0) {   // rank 0's stream at the visit's input
            DeviceScope d0(dev_);
            const int E = c.E;
            CUDA_CHECK(cudaMemcpy2DAsync(feat_.p + (size_t) tap_of_visit_[order] * E, (size_t) InkSpark::kTaps * E * 4, x_.p,
                                         (size_t) E * 4, (size_t) E * 4, T, cudaMemcpyDeviceToDevice, st_));
        }
        ++order;
        for (InkEngine * r : rk) {
            const InkLoop * loop = looped ? &r->model_.loops[ip - c.loop_start] : nullptr;
            r->visit_a(v, ip, T, pos0, loop, keep);
            r->send(0, v, r->conv_in_, T);
        }
        for (InkEngine * r : rk) {
            const InkLoop * loop = looped ? &r->model_.loops[ip - c.loop_start] : nullptr;
            r->recv(0, v, r->conv_in_, T);
            r->visit_b(v, ip, T, loop, keep);
            r->send(1, v, r->moe_, T);
        }
        for (InkEngine * r : rk) {
            r->recv(1, v, r->moe_, T);
            r->visit_c(v, T, keep);
        }
    };
    for (int il = 0; il < c.loop_start; ++il) visit(il, il, false);
    for (int pass = 0; pass < c.loop_count; ++pass) {
        for (InkEngine * r : rk) r->pass_begin(pass, T);
        for (int ip = c.loop_start; ip < c.loop_end; ++ip) {
            const int slot = pass == 0 ? ip : c.n_layer + (pass - 1) * c.n_core() + (ip - c.loop_start);
            visit(slot, ip, pass > 0);
        }
    }
    for (int il = c.loop_end; il < c.n_layer; ++il) visit(il, il, false);
    for (InkEngine * r : rk) r->head_prep(T);
    if (spark_) {   // the normed output: the last tap
        DeviceScope d0(dev_);
        const int E = c.E;
        CUDA_CHECK(cudaMemcpy2DAsync(feat_.p + (size_t) (InkSpark::kTaps - 1) * E, (size_t) InkSpark::kTaps * E * 4, o_.p,
                                     (size_t) E * 4, (size_t) E * 4, T, cudaMemcpyDeviceToDevice, st_));
    }
    auto head = [&](int r0, int n) {
        for (InkEngine * r : rk) r->head_rows(r0, n);
        if (rank1_) {   // rank 0 waits for rank 1's logits rows (and joins its stream back)
            DeviceScope d0(dev_);
            CUDA_CHECK(cudaStreamWaitEvent(st_, rank1_->ev_head_, 0));
        }
    };
    if (T <= kInkWin) {
        head(0, T);   // a decode / verify window: every row
    } else if (logits_hook) {   // tests: every row of a prompt chunk, 16 at a time (the last piece stays in logits_)
        for (int r0 = 0; r0 < T; r0 += kInkWin) {
            const int n = std::min(kInkWin, T - r0);
            head(r0, n);
            DeviceScope d0(dev_);
            CUDA_CHECK(cudaStreamSynchronize(st_));
            if (r0 + n < T) logits_hook(pos0 + r0, n, logits_);   // the last piece: after run()
            hook_last_ = {pos0 + r0, n};
        }
    } else {
        head(T - 1, 1);   // a prompt chunk: its last row
    }
}

void InkEngine::prefill(const std::vector<int32_t> & tokens) {
    if (pending_T_) throw std::runtime_error("inkling: prefill with a verify window pending");
    if (images()) {   // encode the prompt's new images first
        DeviceScope d0(dev_);
        images_->encode_all(tokens.data(), (int) tokens.size());
    }
    const double t0 = now_ms();
    for (size_t i = 0; i < tokens.size(); i += chunk_) {
        const int n = (int) std::min<size_t>(chunk_, tokens.size() - i);
        const int p0 = pos();
        run(tokens.data() + i, n, true);
        if (spark_) {
            DeviceScope d0(dev_);
            if (spark_->fed() != p0) spark_->restart(std::min(spark_->fed(), p0));
            if (spark_->fed() == p0) spark_->feed(feat_, n, p0);
        }
        if (on_prefill_progress) on_prefill_progress(i + n, tokens.size());
    }
    times.total_ms += now_ms() - t0;
}

void InkEngine::forward(const int32_t * tokens, int T, bool commit_all) {
    forward_batch({BatchWin{0, tokens, T, commit_all}});
}

// the slots' windows one after the other; with more than one, each window's logits and features rows are copied
// side by side (row0_of) for the readers
void InkEngine::forward_batch(const std::vector<BatchWin> & wins) {
    DeviceScope ds(dev_);
    if (wins.empty()) return;
    int total = 0;
    for (const auto & w : wins) total += w.T;
    if (total > kInkWin) throw std::runtime_error("inkling: windows too long");
    const bool many = wins.size() > 1;
    if (many && !blogits_.p) blogits_.alloc((size_t) kInkWin * model_.cfg.n_vocab);
    const double t0 = now_ms();
    int r0 = 0;
    for (const auto & w : wins) {
        select(w.slot);
        if (pending_T_) throw std::runtime_error("inkling: forward with a verify window pending");
        run(w.tokens, w.T, w.commit_all);
        if (!w.commit_all) pending_T_ = w.T;
        row0_[w.slot] = r0;
        if (many) {
            const size_t V = model_.cfg.n_vocab;
            CUDA_CHECK(cudaMemcpyAsync(blogits_.p + r0 * V, logits_.p, (size_t) w.T * V * 4, cudaMemcpyDeviceToDevice, st_));
            if (spark_) {
                const size_t F = (size_t) InkSpark::kTaps * model_.cfg.E;
                CUDA_CHECK(cudaMemcpyAsync(bfeat_.p + r0 * F, feat_.p, (size_t) w.T * F * 4, cudaMemcpyDeviceToDevice, st_));
            }
        }
        r0 += w.T;
    }
    if (many) {
        CUDA_CHECK(cudaStreamSynchronize(st_));
        lg_ = blogits_.p;
        feat_out_ = bfeat_.p;
    } else {
        row0_[wins[0].slot] = 0;
    }
    times.total_ms += now_ms() - t0;
    times.calls++;
}

void InkEngine::commit(int cnt) {
    if (pending_T_ == 0) throw std::runtime_error("commit: no verify window pending");
    if (cnt < 1 || cnt > pending_T_) throw std::runtime_error("commit: count out of range");
    for (InkEngine * r : ranks()) r->commit_here(cnt);
    history_.resize(history_.size() - (pending_T_ - cnt));
    pending_T_ = 0;
}

void InkEngine::commit_here(int cnt) {
    DeviceScope ds(dev_);
    for (auto & V : visits_) ink_conv_advance(V.win_in, V.conv, cnt, conv_ch_, st_);
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

// ------------------------------------------------------------------------------------------------ logits
int InkEngine::argmax(int t) {
    DeviceScope ds(dev_);
    const int V = model_.cfg.n_vocab;
    argmax_rows(lg_ + (size_t) t * V, 1, V, argmax_dev_, st_);
    int32_t r;
    CUDA_CHECK(cudaMemcpyAsync(&r, argmax_dev_.p, 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    return r;
}

void InkEngine::argmax_all(int T, int32_t * out, int row0) {
    DeviceScope ds(dev_);
    const int V = model_.cfg.n_vocab;
    argmax_rows(lg_ + (size_t) row0 * V, T, V, argmax_dev_, st_);
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
    topk_rows(lg_ + (size_t) row0 * V, T, V, K, topk_ids_, topk_vals_, st_);
    CUDA_CHECK(cudaMemcpyAsync(ids, topk_ids_.p, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(vals, topk_vals_.p, (size_t) T * K * 4, cudaMemcpyDeviceToHost, st_));
    if (z) {
        if (!topk_z_.p) topk_z_.alloc(kInkWin);
        row_expsum(lg_ + (size_t) row0 * V, T, V, temp, topk_z_.p, st_);
        CUDA_CHECK(cudaMemcpyAsync(z, topk_z_.p, (size_t) T * sizeof(double), cudaMemcpyDeviceToHost, st_));
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

void InkEngine::logits_rows_host(int T, float * out, int row0) {
    DeviceScope ds(dev_);
    CUDA_CHECK(cudaMemcpy(out, lg_ + (size_t) row0 * model_.cfg.n_vocab, (size_t) T * model_.cfg.n_vocab * 4,
                          cudaMemcpyDeviceToHost));
}

std::vector<float> InkEngine::logits_host(int t) {
    DeviceScope ds(dev_);
    std::vector<float> v(model_.cfg.n_vocab);
    CUDA_CHECK(cudaMemcpy(v.data(), lg_ + (size_t) t * model_.cfg.n_vocab, v.size() * 4, cudaMemcpyDeviceToHost));
    return v;
}

}  // namespace bnk
