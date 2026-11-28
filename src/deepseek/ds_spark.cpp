#include "deepseek/ds_spark.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <algorithm>
#include <cmath>
#include <random>

#include "core/util.h"
#include "deepseek/ds_ops.h"
#include "ggml.h"
#include "kernels/ops.h"

namespace bnk {

DsSpark::~DsSpark() {
    if (vcp_ev_) cudaEventDestroy(vcp_ev_);
    if (vdone_ev_) cudaEventDestroy(vdone_ev_);
    if (vcp_st_) cudaStreamDestroy(vcp_st_);
}

QMat DsSpark::upload(const std::string & name, bool required) {
    const TensorRef * t = gguf_.find(name);
    if (!t) {
        if (required) throw std::runtime_error("dspark: missing tensor " + name);
        return QMat{};
    }
    const size_t before = main_->vram_dense_bytes;
    QMat m = main_->upload_tensor(*t, name);
    vram_bytes_ += main_->vram_dense_bytes - before;
    return m;
}

void DsSpark::load(const std::string & path, DsModel & m, CpuExpertPool & pool, int slots, int threads, cudaStream_t st,
                   bool verbose) {
    main_ = &m;
    pool_ = &pool;
    st_ = st;
    CUDA_CHECK(cudaGetDevice(&dev_));
    gguf_.open(path);
    if (gguf_.get_str("general.architecture") != "dflash" || !gguf_.has_key("dflash.hyper_connection.single_pass"))
        throw std::runtime_error(path + " is not a DeepSeek-V4.1 DSpark draft (llama.cpp's DFLASH layout, tools/dspark_gguf.py)");
    const std::string a = "dflash.";
    // the drafter's hyper-parameters: the target's, with its own layers and expert counts; every stage window-only
    c_ = m.ds;
    c_.n_layer = (int) gguf_.get_u64(a + "block_count");
    c_.n_expert = (int) gguf_.get_u64(a + "expert_count");
    c_.n_used = (int) gguf_.get_u64(a + "expert_used_count");
    c_.n_ff = (int) gguf_.get_u64(a + "expert_feed_forward_length", m.ds.n_ff);
    c_.route_scale = (float) gguf_.get_f64(a + "expert_weights_scale", m.ds.route_scale);
    c_.route_norm = gguf_.get_bool(a + "expert_weights_norm", m.ds.route_norm);
    c_.window = (int) gguf_.get_u64(a + "attention.sliding_window", m.ds.window);
    if (c_.window == m.ds.window + 1) c_.window = m.ds.window;   // llama.cpp adds one for its mask row
    c_.ratio.assign(c_.n_layer, 0);
    c_.kv_src.assign(c_.n_layer, -1);
    c_.topk_src.assign(c_.n_layer, -1);
    c_.engram_layers.clear();
    block_ = (int) gguf_.get_u64(a + "block_size");
    noise_id_ = (int32_t) gguf_.get_u64("tokenizer.ggml.mask_token_id");
    for (int64_t v : gguf_.get_int_arr(a + "target_layers")) targets_.push_back((int) v);
    if (block_ < 2 || block_ > kMaxBlock || targets_.empty() || c_.n_layer < 1 || c_.n_expert < 1)
        throw std::runtime_error("dspark: unexpected block size, target layers or stage count");
    for (int il : targets_)
        if (il < 0 || il >= m.ds.n_layer) throw std::runtime_error("dspark: a target layer outside the model");
    if (gguf_.get_u64(a + "embedding_length") != (uint64_t) m.ds.n_embd || gguf_.get_u64(a + "attention.head_count") != (uint64_t) m.ds.n_head)
        throw std::runtime_error("dspark: the draft's dimensions do not match the target's");
    if (const char * pm = getenv("BNK_DS_SPARK_PMIN")) p_min_ = (float) atof(pm);
    if (const char * v = getenv("BNK_DS_SPARK_ROUND_MS")) round_ms_ = (float) atof(v);
    if (const char * v = getenv("BNK_DS_SPARK_ROW_MS")) row_ms_ = (float) atof(v);
    if (const char * dp = getenv("BNK_DS_SPARK_DUMP")) { dump_prefix_ = dp; dumps_left_ = env_int("BNK_DS_SPARK_DUMP_N", 3); }

    const double t_load0 = now_ms();
    if (verbose) fprintf(stderr, "bnk: DSpark drafter: loading %s (%d stages, blocks of %d)\n", path.c_str(), c_.n_layer, block_);
    main_proj_ = upload("fc.weight");
    main_norm_ = upload("enc.output_norm.weight");
    out_norm_ = upload("output_norm.weight");
    markov_w1_ = upload("markov_w1.weight");
    markov_w2_ = upload("markov_w2.weight");
    conf_proj_ = upload("conf_proj.weight", false);
    if (main_proj_.cols != (int64_t) targets_.size() * m.ds.n_embd || main_proj_.rows != m.ds.n_embd)
        throw std::runtime_error("dspark: fc shape does not match the target layers");
    if (markov_w1_.rows != m.ds.n_vocab || markov_w2_.rows != m.ds.n_vocab || markov_w1_.cols != markov_w2_.cols)
        throw std::runtime_error("dspark: markov head shapes");

    // the stages, named as a model's layers (llama.cpp's names, the ones DsModel::load reads)
    stages_.resize(c_.n_layer);
    ev_.cfg.n_layer = c_.n_layer; ev_.cfg.n_embd = c_.n_embd; ev_.cfg.n_expert = c_.n_expert; ev_.cfg.n_ff_exp = c_.n_ff;
    ev_.cfg.n_vocab = c_.n_vocab;
    ev_.layers.assign(c_.n_layer, LayerWeights{});
    for (int il = 0; il < c_.n_layer; ++il) {
        DsLayer & L = stages_[il];
        const std::string b = "blk." + std::to_string(il) + ".";
        L.attn_norm = upload(b + "attn_norm.weight");
        L.ffn_norm = upload(b + "ffn_norm.weight");
        L.sinks = upload(b + "attn_sinks.weight");
        L.q_a = upload(b + "attn_q_a.weight");
        L.q_a_norm = upload(b + "attn_q_a_norm.weight");
        L.q_b = upload(b + "attn_q_b.weight");
        L.kv = upload(b + "attn_kv.weight");
        L.kv_norm = upload(b + "attn_kv_a_norm.weight");
        L.o_a = upload(b + "attn_output_a.weight");
        L.o_b = upload(b + "attn_output_b.weight");
        if (L.o_a.rows != (int64_t) c_.o_groups * c_.o_lora || L.o_a.cols != (int64_t) c_.n_head * c_.head_dim / c_.o_groups ||
            L.o_b.cols != (int64_t) c_.o_groups * c_.o_lora)
            throw std::runtime_error("dspark: attention output projection shapes in stage " + std::to_string(il));
        L.hc_attn_fn = upload(b + "hc_attn_fn.weight");
        L.hc_attn_base = upload(b + "hc_attn_base.weight");
        L.hc_attn_scale = upload(b + "hc_attn_scale.weight");
        L.hc_ffn_fn = upload(b + "hc_ffn_fn.weight");
        L.hc_ffn_base = upload(b + "hc_ffn_base.weight");
        L.hc_ffn_scale = upload(b + "hc_ffn_scale.weight");
        if (L.hc_attn_fn.type != GGML_TYPE_F32 || L.hc_attn_fn.rows != c_.mix_dim() || L.hc_attn_fn.cols != c_.hc_dim())
            throw std::runtime_error("dspark: hyper-connection mixer shape or type in stage " + std::to_string(il));
        L.router = upload(b + "ffn_gate_inp.weight");
        L.probs_b = upload(b + "exp_probs_b.bias");
        if (L.router.type != GGML_TYPE_F32 || L.probs_b.type != GGML_TYPE_F32 || L.router.rows != c_.n_expert)
            throw std::runtime_error("dspark: router / bias in stage " + std::to_string(il));
        L.sh_gate = upload(b + "ffn_gate_shexp.weight");
        L.sh_up = upload(b + "ffn_up_shexp.weight");
        L.sh_down = upload(b + "ffn_down_shexp.weight");
        // the routed experts: sources in the mapping, for the host store
        const TensorRef & g = gguf_.get(b + "ffn_gate_exps.weight");
        const TensorRef & u = gguf_.get(b + "ffn_up_exps.weight");
        const TensorRef & d = gguf_.get(b + "ffn_down_exps.weight");
        const int n = c_.n_expert;
        if (g.ne[2] != n || u.ne[2] != n || d.ne[2] != n || g.ne[0] != c_.n_embd || g.ne[1] != c_.n_ff || d.ne[0] != c_.n_ff ||
            d.ne[1] != c_.n_embd)
            throw std::runtime_error("dspark: expert tensor shapes in stage " + std::to_string(il));
        LayerWeights & W = ev_.layers[il];
        W.gate_type = g.type; W.up_type = u.type; W.down_type = d.type;
        W.gate_bytes = g.nbytes / n; W.up_bytes = u.nbytes / n; W.down_bytes = d.nbytes / n;
        W.gate_src = g.data; W.up_src = u.data; W.down_src = d.data;
    }
    if (verbose) fprintf(stderr, "bnk: DSpark drafter: dense weights uploaded (%.2f GiB) in %.1f s; copying the experts into RAM\n", vram_bytes_ / 1073741824.0, (now_ms() - t_load0) / 1000);
    store_.build(ev_, std::min(threads, 16), verbose);
    // the per-slot rings and the blocks
    slots_.resize(std::max(1, slots));
    for (auto & s : slots_) {
        s.ring.resize(c_.n_layer);
        for (auto & r : s.ring) {
            r.alloc((size_t) kDsRing * c_.head_dim);
            ds_zero_half(r.p, (size_t) kDsRing * c_.head_dim, st);
        }
    }
    blocks_.resize(slots_.size());
    // work buffers
    {
        const int T = block_, W = kMaxWindow, E = c_.n_embd, H = c_.n_head, D = c_.head_dim;
        const int64_t HD = (int64_t) H * D, GO = (int64_t) c_.o_groups * c_.o_lora, NT = (int64_t) targets_.size() * E;
        x_.alloc((size_t) T * E); res_blk_.alloc((size_t) T * c_.hc_dim()); mixed_.alloc((size_t) T * E); xn_.alloc((size_t) T * E);
        out_.alloc((size_t) T * E); pre_a_.alloc((size_t) T * c_.hc); pre_b_.alloc((size_t) T * c_.hc); attn_pre_.alloc((size_t) T * c_.hc);
        post_.alloc((size_t) T * c_.hc); comb_.alloc((size_t) T * c_.hc * c_.hc); mix_scratch_.alloc(kDsMixScratch);   // the mixer's scratch and counters (ds_hc_mixes: the counters must start at zero)
        CUDA_CHECK(cudaMemset(mix_scratch_.p, 0, kDsMixScratch * sizeof(float)));
        qr_.alloc((size_t) T * c_.q_lora); qrn_.alloc((size_t) T * c_.q_lora); q_.alloc((size_t) T * HD); kvr_.alloc((size_t) W * D);
        attn_o_.alloc((size_t) T * HD); oa_.alloc((size_t) T * GO);
        rlogits_.alloc((size_t) T * c_.n_expert); rids_.alloc((size_t) T * c_.n_used); rw_.alloc((size_t) T * c_.n_used);
        sg_.alloc((size_t) T * c_.n_ff); su_.alloc((size_t) T * c_.n_ff); shared_out_.alloc((size_t) T * E); cpu_out_.alloc((size_t) T * E);
        moe_out_.alloc((size_t) T * E); logits_.alloc((size_t) T * c_.n_vocab); bias_.alloc((size_t) c_.n_vocab);
        w1_row_.alloc((size_t) markov_w1_.cols); main_x_.alloc((size_t) W * E); main_xn_.alloc((size_t) W * E);
        ids_.alloc(T); pos_.alloc(1);
        mrow_.alloc((size_t) c_.n_vocab); mk_vals_.alloc(kDraftK); mk_ids_.alloc(kDraftK);
        h_mk_vals_.resize(kDraftK); h_mk_ids_.resize(kDraftK);
        const int64_t maxcols = std::max<int64_t>({(int64_t) E, (int64_t) c_.q_lora, HD, GO, (int64_t) c_.n_ff, NT});
        actq_.alloc((size_t) W * maxcols); actd_.alloc((size_t) W * maxcols / 32); acth_.alloc((size_t) W * maxcols);
        act_.q = actq_; act_.d = actd_; act_.h = acth_;
        actq2_.alloc((size_t) W * maxcols); actd2_.alloc((size_t) W * maxcols / 32); acth2_.alloc((size_t) W * maxcols);
        act2_.q = actq2_; act2_.d = actd2_; act2_.h = acth2_;
        h_x_.resize((size_t) T * E); h_out_.resize((size_t) T * E); h_logits_.resize((size_t) T * c_.n_vocab);
        h_bias_.resize(c_.n_vocab); h_mixed_.resize((size_t) T * E); h_w1_.resize(markov_w1_.cols); h_ids_.resize(T);
        if (conf_proj_.valid()) {
            conf_w_.resize((size_t) conf_proj_.cols);
            CUDA_CHECK(cudaMemcpy(conf_w_.data(), conf_proj_.data, conf_w_.size() * 4, cudaMemcpyDeviceToHost));
            if (conf_proj_.type != GGML_TYPE_F32 || conf_proj_.cols != E + markov_w1_.cols)
                throw std::runtime_error("dspark: confidence projection shape or type");
        }
    }
    vram_init(verbose);
    CUDA_CHECK(cudaStreamSynchronize(st));
    if (verbose)
        fprintf(stderr, "bnk: DSpark drafter: %d stages, %d experts (%d used) a stage, blocks of %d, target layers %d-%d; dense %.2f GiB in VRAM, experts %.2f GiB in RAM (%s)\n",
                c_.n_layer, c_.n_expert, c_.n_used, block_, targets_.front(), targets_.back(), vram_bytes_ / 1073741824.0,
                store_.total_bytes() / 1073741824.0, ggml_type_str((ggml_type) ev_.layers[0].gate_type).c_str());
    loaded_ = true;
}

void DsSpark::restart(int pos) {
    DeviceScope dsc(dev_);
    Slot & s = slots_[cur_];
    // positions below pos stay valid when the ring still holds them; past a deep rewind the slots hold later positions
    if (pos < s.fed_lo || s.fed_hi - pos > nslots_ - c_.window) s.fed_lo = pos;
    s.fed_hi = std::min(s.fed_hi, pos);
    blocks_[cur_] = Block{};
}

static const float * F(const QMat & m) { return (const float *) m.data; }

void DsSpark::dump_host(const char * what, const void * host, size_t bytes) {
    if (dumps_left_ <= 0) return;
    const std::string fn = dump_prefix_ + "." + std::to_string(dump_n_) + "." + what;
    FILE * f = fopen(fn.c_str(), "wb");
    if (!f) throw std::runtime_error("dspark: cannot write " + fn);
    fwrite(host, 1, bytes, f);
    fclose(f);
}

void DsSpark::dump(const char * what, const void * dev, size_t bytes) {
    if (dumps_left_ <= 0) return;
    std::vector<uint8_t> h(bytes);
    CUDA_CHECK(cudaStreamSynchronize(st_));
    CUDA_CHECK(cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost));
    dump_host(what, h.data(), bytes);
}

// the capture ring's rows at positions pos0..pos0+n-1 (the target layers' attention inputs) into every stage's ring
void DsSpark::feed(int pos0, int n) {
    const int E = c_.n_embd, D = c_.head_dim, R = c_.n_rot;
    const int64_t NT = (int64_t) targets_.size() * E;
    if (n <= 0) return;
    Slot & sl = slots_[cur_];
    if (pos0 > sl.fed_hi || pos0 < sl.fed_lo) sl.fed_lo = pos0;   // a gap or a rewind: the span restarts here
    for (int done = 0; done < n;) {
        const int p = pos0 + done, slot = p % nslots_;
        const int m = std::min(n - done, nslots_ - slot);   // up to the ring's wrap
        if (m > kMaxWindow) throw std::runtime_error("dspark: feed chunk larger than a window");
        const float * rows = targets_dev_ + (size_t) slot * NT;
        if (dumps_left_ > 0) {
            static int feed_n = 0;
            const int32_t hdr[2] = {p, m};
            dump_host(("feed" + std::to_string(feed_n) + ".hdr").c_str(), hdr, sizeof hdr);
            dump(("feed" + std::to_string(feed_n) + ".rows").c_str(), rows, (size_t) m * NT * 4);
            ++feed_n;
        }
        CUDA_CHECK(cudaMemcpyAsync(pos_.p, &p, 4, cudaMemcpyHostToDevice, st_));
        quantize_act(rows, NT, m, NT, act_, st_);
        gemv(main_proj_, &act_, nullptr, 0, m, main_x_, E, false, st_);
        rmsnorm_rows(main_x_, F(main_norm_), main_xn_, m, E, E, E, c_.eps, st_);
        quantize_act(main_xn_, E, m, E, act_, st_);
        for (int il = 0; il < c_.n_layer; ++il) {
            gemv(stages_[il].kv, &act_, nullptr, 0, m, kvr_, D, false, st_);
            ds_kv_prep(kvr_, F(stages_[il].kv_norm), sl.ring[il].p, m, D, R, pos_.p, rope_, c_.eps, st_);
        }
        done += m;
    }
    sl.fed_hi = pos0 + n;
}

void DsSpark::feed_range(int p_from, int p_to) {
    DeviceScope dsc(dev_);
    p_from = std::max(p_from, p_to - nslots_);
    for (int p = p_from; p < p_to; p += kMaxWindow) feed(p, std::min(kMaxWindow, p_to - p));
    CUDA_CHECK(cudaStreamSynchronize(st_));
}

// one stage over the block rows: the hyper-connection mixes, the window attention over the ring and the block, the
// MoE on the CPU pool plus the shared expert (DsEngine::layer_forward without the Engram, the compressor and the graphs)
void DsSpark::stage_forward(int il, int T) {
    const DsLayer & L = stages_[il];
    const int E = c_.n_embd, H = c_.n_head, D = c_.head_dim, QL = c_.q_lora, R = c_.n_rot, k = c_.n_used, Fx = c_.n_ff;
    Slot & sl = slots_[cur_];
    ds_hc_mixes(res_blk_, F(L.hc_attn_fn), F(L.hc_attn_scale), F(L.hc_attn_base), attn_pre_, post_, comb_, T, c_.hc, E, c_.eps, c_.hc_eps,
                c_.sinkhorn, st_, mix_scratch_.p);
    ds_hc_pre_norm_q(res_blk_, pre_, F(L.attn_norm), mixed_, xn_, T, c_.hc, E, c_.eps, act_, st_);
    // attention: q through the low-rank pair, one latent per row into the ring at the block's positions
    gemv(L.q_a, &act_, nullptr, 0, T, qr_, QL, false, st_);
    rmsnorm_rows(qr_, F(L.q_a_norm), qrn_, T, QL, QL, QL, c_.eps, st_);
    quantize_act(qrn_, QL, T, QL, act2_, st_);
    gemv(L.q_b, &act2_, nullptr, 0, T, q_, (int64_t) H * D, false, st_);
    gemv(L.kv, &act_, nullptr, 0, T, kvr_, D, false, st_);
    ds_rope_kv_prep(q_, H, kvr_, F(L.kv_norm), sl.ring[il].p, T, D, R, pos_.p, rope_, c_.eps, st_);
    ds_attention_block(q_, sl.ring[il].p, F(L.sinks), attn_o_, T, H, D, c_.window, R, pos_.p, rope_, 1.f / sqrtf((float) D), st_, sl.fed_lo);
    const int G = c_.o_groups, GW = H * D / G, OL = c_.o_lora;
    for (int g = 0; g < G; ++g) {
        quantize_act(attn_o_.p + (size_t) g * GW, (int64_t) H * D, T, GW, act2_, st_);
        gemv(qmat_rows(L.o_a, (int64_t) g * OL, OL), &act2_, nullptr, 0, T, oa_.p + (size_t) g * OL, (int64_t) G * OL, false, st_);
    }
    quantize_act(oa_, (int64_t) G * OL, T, (int64_t) G * OL, act2_, st_);
    gemv(L.o_b, &act2_, nullptr, 0, T, out_, E, false, st_);
    dump(("attn_out" + std::to_string(il)).c_str(), out_.p, (size_t) T * E * 4);
    ds_hc_post(out_, res_blk_, post_, comb_, T, c_.hc, E, st_);
    dump(("res_attn" + std::to_string(il)).c_str(), res_blk_.p, (size_t) T * c_.hc_dim() * 4);
    // the FFN: routing on the GPU, the routed experts on the CPU pool, the shared expert on the GPU
    ds_hc_mixes(res_blk_, F(L.hc_ffn_fn), F(L.hc_ffn_scale), F(L.hc_ffn_base), pre_next_, post_, comb_, T, c_.hc, E, c_.eps, c_.hc_eps,
                c_.sinkhorn, st_, mix_scratch_.p);
    ds_hc_pre_norm_q(res_blk_, attn_pre_, F(L.ffn_norm), mixed_, xn_, T, c_.hc, E, c_.eps, act_, st_);
    gemv(L.router, nullptr, xn_, E, T, rlogits_, c_.n_expert, false, st_);
    ds_route(rlogits_, F(L.probs_b), T, c_.n_expert, k, c_.route_scale, c_.route_norm, rids_, rw_, st_);
    std::vector<int32_t> ids((size_t) T * k);
    std::vector<float> w((size_t) T * k);
    CUDA_CHECK(cudaMemcpyAsync(ids.data(), rids_.p, ids.size() * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(w.data(), rw_.p, w.size() * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(h_x_.data(), xn_.p, (size_t) T * E * 4, cudaMemcpyDeviceToHost, st_));
    static const bool prof = getenv("BNK_DS_SPARK_PROF") != nullptr;
    const double tp0 = prof ? now_ms() : 0;
    CUDA_CHECK(cudaStreamSynchronize(st_));
    const double tp1 = prof ? now_ms() : 0;
    tasks_.clear();
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < k; ++j)
            if (ids[(size_t) t * k + j] >= 0) tasks_.push_back({t, j, ids[(size_t) t * k + j], w[(size_t) t * k + j]});
    // the resident ones on this GPU (queued ahead of the CPU's share, so the two overlap), the rest on the pool
    cpu_tasks_.clear();
    vlist_.clear();
    const bool vram = !vk_.empty() && vk_[il] > 0;
    if (vram) {
        for (const auto & tk : tasks_) {
            vhits_[il][tk.expert] += 1.0;
            if (vslot_of_[il][tk.expert] < 0) { cpu_tasks_.push_back(tk); continue; }
            if (std::find(vlist_.begin(), vlist_.end(), tk.expert) == vlist_.end()) vlist_.push_back(tk.expert);
        }
    }
    if (!vlist_.empty()) {
        const LayerWeights & W = ev_.layers[il];
        const int F = c_.n_ff;
        const size_t blob = store_.blob_bytes(il);
        h_vw_.assign((size_t) vlist_.size() * T, 0.f);
        for (const auto & tk : tasks_) {
            const auto it = std::find(vlist_.begin(), vlist_.end(), tk.expert);
            if (it != vlist_.end()) h_vw_[(size_t) (it - vlist_.begin()) * T + tk.t] += tk.w;
        }
        CUDA_CHECK(cudaMemcpyAsync(vw_.p, h_vw_.data(), h_vw_.size() * 4, cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaMemsetAsync(vout_.p, 0, (size_t) T * E * 4, st_));
        for (size_t j = 0; j < vlist_.size(); ++j) {
            const uint8_t * b = vslab_[il].p + (size_t) vslot_of_[il][vlist_[j]] * blob;
            QMat gu, dn;
            gu.data = b; gu.type = W.gate_type; gu.rows = 2 * F; gu.cols = E; gu.row_bytes = W.gate_bytes / F;
            dn.data = b + W.gate_bytes + W.up_bytes; dn.type = W.down_type; dn.rows = E; dn.cols = F; dn.row_bytes = W.down_bytes / E;
            gemv(gu, &act_, nullptr, 0, T, vgu_, 2 * F, false, st_);   // act_: xn_ quantized (ds_hc_pre_norm_q)
            quantize_act_silu_mul(vgu_, vgu_.p + F, 2 * F, T, F, act2_, st_);
            gemv(dn, &act2_, nullptr, 0, T, vdn_, E, false, st_);
            ds_add_scaled_rows(vdn_, vw_.p + j * T, vout_, T, E, st_);
        }
    }
    const std::vector<ExpertTask> & on_cpu = vram ? cpu_tasks_ : tasks_;
    if (!on_cpu.empty()) pool_->run(ev_, store_, il, T, h_x_.data(), on_cpu, h_out_.data());
    else std::fill(h_out_.begin(), h_out_.begin() + (size_t) T * E, 0.f);
    if (prof) {
        static double gpu_ms = 0, cpu_ms = 0; static long n = 0; static std::vector<std::vector<long>> hits;
        if (hits.empty()) hits.assign(c_.n_layer, std::vector<long>(c_.n_expert, 0));
        gpu_ms += tp1 - tp0; cpu_ms += now_ms() - tp1;
        for (const auto & tk : tasks_) hits[il][tk.expert]++;
        if (il == c_.n_layer - 1 && ++n % 200 == 0) {
            fprintf(stderr, "PROF dspark %ld blocks: wait for GPU before MoE %.2f ms/block, CPU experts %.2f ms/block\n", n, gpu_ms / n, cpu_ms / n);
            for (int l = 0; l < c_.n_layer; ++l) {
                std::vector<long> h = hits[l]; std::sort(h.rbegin(), h.rend()); long tot = 0, a = 0; for (long v : h) tot += v;
                std::string line; for (int k2 = 0; k2 < c_.n_expert; ++k2) { a += h[k2]; if (k2 == 7 || k2 == 15 || k2 == 31 || k2 == 63) line += " top" + std::to_string(k2 + 1) + " " + std::to_string(tot ? 100 * a / tot : 0) + "%"; }
                fprintf(stderr, "PROF dspark stage %d expert share:%s\n", l, line.c_str());
            }
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(cpu_out_.p, h_out_.data(), (size_t) T * E * 4, cudaMemcpyHostToDevice, st_));
    if (L.sh_gu.valid()) gemv(L.sh_gu, &act_, nullptr, 0, T, sg_, Fx, false, st_, GemvSplit{Fx, su_, Fx});
    else {
        gemv(L.sh_gate, &act_, nullptr, 0, T, sg_, Fx, false, st_);
        gemv(L.sh_up, &act_, nullptr, 0, T, su_, Fx, false, st_);
    }
    quantize_act_silu_mul(sg_, su_, Fx, T, Fx, act2_, st_);
    gemv(L.sh_down, &act2_, nullptr, 0, T, shared_out_, E, false, st_);
    // moe_out = shared + routed
    ds_add(shared_out_, cpu_out_, moe_out_, (size_t) T * E, st_);
    if (!vlist_.empty()) ds_add(moe_out_, vout_, moe_out_, (size_t) T * E, st_);
    if (dumps_left_ > 0) {
        dump(("ffn_in" + std::to_string(il)).c_str(), xn_.p, (size_t) T * E * 4);
        dump_host(("routed" + std::to_string(il)).c_str(), h_out_.data(), (size_t) T * E * 4);
        dump(("shared" + std::to_string(il)).c_str(), shared_out_.p, (size_t) T * E * 4);
        std::vector<int32_t> ti; std::vector<float> tw;
        for (const auto & tk : tasks_) { ti.push_back(tk.t); ti.push_back(tk.expert); tw.push_back(tk.w); }
        dump_host(("tasks" + std::to_string(il) + ".ids").c_str(), ti.data(), ti.size() * 4);
        dump_host(("tasks" + std::to_string(il) + ".w").c_str(), tw.data(), tw.size() * 4);
    }
    ds_hc_post(moe_out_, res_blk_, post_, comb_, T, c_.hc, E, st_);
    std::swap(pre_, pre_next_);
}

// the block after `anchor` at position `pos`: embed anchor + noise, the stages, the head, the Markov chain and the
// confidence head on the host
void DsSpark::draft_block(int32_t anchor, int pos) {
    // BNK_DS_SPARK_ROWS=n: draft blocks of n rows instead of the model's block size (the schedule keeps ~2.5 drafts a
    // round, and each row costs its routed experts on the CPU; drafts need not be exact, only their distribution known)
    static const int rows_env = env_int("BNK_DS_SPARK_ROWS", 0);
    const int T = rows_env >= 2 && rows_env < block_ ? rows_env : block_, E = c_.n_embd, V = c_.n_vocab, RK = (int) markov_w1_.cols;
    const double t_draft0 = now_ms();
    Block & b = blocks_[cur_];
    b = Block{};
    h_ids_[0] = anchor;
    for (int i = 1; i < T; ++i) h_ids_[i] = noise_id_;
    CUDA_CHECK(cudaMemcpyAsync(ids_.p, h_ids_.data(), T * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(pos_.p, &pos, 4, cudaMemcpyHostToDevice, st_));
    dequant_gather(tok_embd_, ids_.p, T, x_, st_);
    hc_init(x_, res_blk_, T, c_.hc, E, st_);
    pre_ = pre_a_;
    pre_next_ = pre_b_;
    ds_onehot(pre_, T, c_.hc, st_);
    {
        const int32_t hdr[2] = {anchor, pos};
        dump_host("block.hdr", hdr, sizeof hdr);
        dump("x", x_.p, (size_t) T * E * 4);
    }
    for (int il = 0; il < c_.n_layer; ++il) {
        stage_forward(il, T);
        dump(("res" + std::to_string(il)).c_str(), res_blk_.p, (size_t) T * c_.hc_dim() * 4);
    }
    vram_step();
    // the head: collapse with the last FFN's pre-mix (no head mixer), norm, the target's output matrix
    ds_hc_pre_norm_q(res_blk_, pre_, F(out_norm_), mixed_, xn_, T, c_.hc, E, c_.eps, act_, st_);
    gemv(main_->output, &act_, nullptr, 0, T, logits_, V, false, st_);
    dump("mixed", mixed_.p, (size_t) T * E * 4);
    dump("logits", logits_.p, (size_t) T * V * 4);
    if (dumps_left_ > 0) { --dumps_left_; ++dump_n_; }
    // BNK_DS_SPARK_HOST_DRAFT=1: the Markov step on the host over the whole vocabulary (as before 2026-11-25); else on
    // the GPU: each position's logits + bias, its top kDraftK, and the draft drawn from those (renormalized), so only
    // kDraftK values a position cross PCIe instead of the 129K-float row (and the host's 129K expf and partial sort)
    static const bool host_draft = getenv("BNK_DS_SPARK_HOST_DRAFT") != nullptr;
    static const bool trace = getenv("BNK_DS_SPARK_TRACE") != nullptr;
    if (host_draft || trace)
        CUDA_CHECK(cudaMemcpyAsync(h_logits_.data(), logits_.p, (size_t) T * V * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(h_mixed_.data(), mixed_.p, (size_t) T * E * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    // the Markov chain: position i's logits plus W2 W1[token_{i-1}], sampled (or argmax) at the request's temperature;
    // the confidence of position i from its collapsed state and W1[token_{i-1}]
    static thread_local std::mt19937_64 rng{0x5eed};
    int32_t prev = anchor;
    float survival = 1.f, expected = 1.f;   // tokens the round yields in expectation: the bonus token plus the kept drafts
    std::vector<float> row((size_t) V);
    std::vector<int> order;
    for (int i = 0; i < T; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(ids_.p, &prev, 4, cudaMemcpyHostToDevice, st_));
        dequant_gather(markov_w1_, ids_.p, 1, w1_row_, st_);
        quantize_act(w1_row_, RK, 1, RK, act2_, st_);
        gemv(markov_w2_, &act2_, nullptr, 0, 1, bias_, V, false, st_);
        if (!host_draft) {
            ds_add(logits_.p + (size_t) i * V, bias_.p, mrow_.p, (size_t) V, st_);
            topk_rows(mrow_.p, 1, V, kDraftK, mk_ids_.p, mk_vals_.p, st_);
            CUDA_CHECK(cudaMemcpyAsync(h_mk_ids_.data(), mk_ids_.p, kDraftK * 4, cudaMemcpyDeviceToHost, st_));
            CUDA_CHECK(cudaMemcpyAsync(h_mk_vals_.data(), mk_vals_.p, kDraftK * 4, cudaMemcpyDeviceToHost, st_));
        }
        if (host_draft || trace) CUDA_CHECK(cudaMemcpyAsync(h_bias_.data(), bias_.p, (size_t) V * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaMemcpyAsync(h_w1_.data(), w1_row_.p, (size_t) RK * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
        const float * lg = h_logits_.data() + (size_t) i * V;
        float mx = -INFINITY;
        if (host_draft || trace)
            for (int v = 0; v < V; ++v) { row[v] = lg[v] + h_bias_[v]; mx = std::max(mx, row[v]); }
        float conf = 1.f;
        if (!conf_w_.empty()) {
            double a = 0;
            const float * h = h_mixed_.data() + (size_t) i * E;
            for (int e = 0; e < E; ++e) a += (double) conf_w_[e] * h[e];
            for (int r = 0; r < RK; ++r) a += (double) conf_w_[E + r] * h_w1_[r];
            conf = 1.f / (1.f + expf((float) -a));
        }
        int32_t tok;
        std::vector<std::pair<int32_t, float>> dist;
        if (!host_draft) {   // from the GPU's head: the draft distribution is the head renormalized (exactly known)
            const int kk = kDraftK;   // (topk_rows leaves its K unordered)
            int best = 0;
            for (int j = 1; j < kk; ++j)
                if (h_mk_vals_[j] > h_mk_vals_[best]) best = j;
            if (temperature_ <= 0.f) {
                tok = h_mk_ids_[best];
            } else {
                const float inv = 1.f / temperature_, top = h_mk_vals_[best];
                double z = 0;
                std::vector<double> e(kk);
                for (int j = 0; j < kk; ++j) { e[j] = std::exp((double) (h_mk_vals_[j] - top) * inv); z += e[j]; }
                const double u = std::uniform_real_distribution<double>(0, 1)(rng) * z;
                double acc = 0;
                int pick = kk - 1;
                for (int j = 0; j < kk; ++j) { acc += e[j]; if (acc >= u) { pick = j; break; } }
                tok = h_mk_ids_[pick];
                dist.reserve(kk);
                for (int j = 0; j < kk; ++j) dist.push_back({h_mk_ids_[j], (float) (e[j] / z)});
            }
        } else if (temperature_ <= 0.f) {
            tok = (int32_t) (std::max_element(row.begin(), row.end()) - row.begin());
        } else {
            const float inv = 1.f / temperature_;
            double z = 0;
            for (int v = 0; v < V; ++v) { row[v] = expf((row[v] - mx) * inv); z += row[v]; }
            const double u = std::uniform_real_distribution<double>(0, 1)(rng) * z;
            double acc = 0;
            tok = V - 1;
            for (int v = 0; v < V; ++v) { acc += row[v]; if (acc >= u) { tok = v; break; } }
            // the distribution the draft came from (its top candidates and the draft itself), for rejection sampling
            order.resize(V);
            for (int v = 0; v < V; ++v) order[v] = v;
            const int K = 64;
            std::partial_sort(order.begin(), order.begin() + K, order.end(), [&](int a, int c) { return row[a] > row[c]; });
            bool has = false;
            for (int j = 0; j < K; ++j) { dist.push_back({order[j], (float) (row[order[j]] / z)}); has |= order[j] == tok; }
            if (!has) dist.push_back({tok, (float) (row[tok] / z)});
        }
        if (trace) {
            int nan = 0, base_arg = 0;
            float base_mx = -INFINITY, bias_mx = 0.f;
            for (int v = 0; v < V; ++v) {
                if (std::isnan(lg[v]) || std::isnan(h_bias_[v])) ++nan;
                if (lg[v] > base_mx) { base_mx = lg[v]; base_arg = v; }
                bias_mx = std::max(bias_mx, std::fabs(h_bias_[v]));
            }
            double hn = 0;
            for (int e = 0; e < E; ++e) hn += (double) h_mixed_[(size_t) i * E + e] * h_mixed_[(size_t) i * E + e];
            fprintf(stderr, "dspark pos %d row %d prev %d: tok %d (base argmax %d, max logit %.2f, |bias| max %.2f, nan %d), conf %.3f, |h| %.2f\n",
                    pos, i, prev, tok, base_arg, base_mx, bias_mx, nan, conf, sqrt(hn / E));
        }
        survival *= conf;
        // the schedule: the draft's expected tokens per millisecond of its verify row against the round's average so far
        const float avg = expected / (round_ms_ + draft_ms_ + row_ms_ * i);
        if (i > 0 && (survival < p_min_ || survival < row_ms_ * avg)) break;
        expected += survival;
        b.toks.push_back(tok);
        b.conf.push_back(conf);
        b.dist.push_back(std::move(dist));
        prev = tok;
    }
    draft_ms_ = 0.9f * draft_ms_ + 0.1f * (float) (now_ms() - t_draft0);
}

void DsSpark::vram_init(bool verbose) {
    // default 32 a stage (1.7 GiB of the last GPU's expert cache): prose +12%, code / math unchanged (docs/DSPARK.md)
    const char * env = getenv("BNK_DS_SPARK_VRAM");
    if (!env) env = "32";
    std::vector<int> k;
    if (env) {
        for (const char * p = env; *p;) {
            k.push_back(std::max(0, atoi(p)));
            while (*p && *p != ',') ++p;
            if (*p == ',') ++p;
        }
    }
    if (k.empty()) return;
    while ((int) k.size() < c_.n_layer) k.push_back(k.back());
    vk_.assign(c_.n_layer, 0);
    vslab_.resize(c_.n_layer);
    vslot_of_.assign(c_.n_layer, std::vector<int>(c_.n_expert, -1));
    vexp_in_.resize(c_.n_layer);
    vhits_.assign(c_.n_layer, std::vector<double>(c_.n_expert, 0.0));
    for (int il = 0; il < c_.n_layer; ++il) {
        const LayerWeights & W = ev_.layers[il];
        // gate | up as one 2F-row matrix: the same format and row size
        if (W.gate_type != W.up_type || W.gate_bytes != W.up_bytes || W.gate_type == GGML_TYPE_F32 || W.gate_type == GGML_TYPE_F16 ||
            W.gate_type == GGML_TYPE_BF16)
            continue;
        vk_[il] = std::min(k[il], c_.n_expert);
        if (!vk_[il]) continue;
        vslab_[il].alloc((size_t) vk_[il] * store_.blob_bytes(il));
        vexp_in_[il].assign(vk_[il], -1);
        vram_experts_bytes_ += (size_t) vk_[il] * store_.blob_bytes(il);
    }
    if (!vram_experts_bytes_) { vk_.clear(); return; }
    const int T = block_, E = c_.n_embd, F = c_.n_ff;
    vgu_.alloc((size_t) T * 2 * F); vdn_.alloc((size_t) T * E); vout_.alloc((size_t) T * E);
    vw_.alloc((size_t) c_.n_used * T * T);
    CUDA_CHECK(cudaStreamCreateWithFlags(&vcp_st_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&vcp_ev_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&vdone_ev_, cudaEventDisableTiming));
    vram_bytes_ += vram_experts_bytes_;
    if (verbose) {
        std::string ks;
        for (int il = 0; il < c_.n_layer; ++il) ks += (il ? "," : "") + std::to_string(vk_[il]);
        fprintf(stderr, "bnk: DSpark drafter: %s hot experts a stage resident in VRAM (%.2f GiB, filled as the routing shows them)\n",
                ks.c_str(), vram_experts_bytes_ / 1073741824.0);
    }
}

// One copy at a time on its own stream: an expert's blob from the (pinned) store into a slot, after the blocks
// already queued are done with whatever the slot held. A slot leaves the routing the moment its copy is decided
// and joins it once the copy has landed.
void DsSpark::vram_step() {
    if (vk_.empty()) return;
    ++vblocks_;
    if (vblocks_ % 4096 == 0)
        for (auto & h : vhits_) for (double & v : h) v *= 0.5;
    if (vpend_stage_ >= 0) {
        if (cudaEventQuery(vcp_ev_) == cudaErrorNotReady) return;
        vexp_in_[vpend_stage_][vpend_slot_] = vpend_exp_;
        vslot_of_[vpend_stage_][vpend_exp_] = vpend_slot_;
        vpend_stage_ = -1;
    }
    // the stage whose hottest non-resident expert most outweighs what it would replace
    int best_il = -1, best_slot = -1, best_e = -1;
    double best_gain = 0.0;
    for (int il = 0; il < c_.n_layer; ++il) {
        if (!vk_[il]) continue;
        int hot = -1;
        for (int e = 0; e < c_.n_expert; ++e)
            if (vslot_of_[il][e] < 0 && vhits_[il][e] > 0 && (hot < 0 || vhits_[il][e] > vhits_[il][hot])) hot = e;
        if (hot < 0) continue;
        int slot = -1;
        double cold = 0.0;
        for (int sl = 0; sl < vk_[il]; ++sl) {
            const int e = vexp_in_[il][sl];
            const double h = e < 0 ? -1.0 : vhits_[il][e];
            if (slot < 0 || h < cold) { slot = sl; cold = h; }
        }
        // an empty slot takes any expert; an occupied one only for a clearly hotter one (no churn between equals)
        const double gain = cold < 0 ? vhits_[il][hot] + 1e9 : vhits_[il][hot] - 1.25 * cold - 2.0;
        if (gain > best_gain) { best_gain = gain; best_il = il; best_slot = slot; best_e = hot; }
    }
    if (best_il < 0) return;
    const uint8_t * src = store_.blob(best_il, best_e);
    if (!src) return;
    const int old = vexp_in_[best_il][best_slot];
    if (old >= 0) vslot_of_[best_il][old] = -1;
    vexp_in_[best_il][best_slot] = -1;
    const size_t blob = store_.blob_bytes(best_il);
    CUDA_CHECK(cudaEventRecord(vdone_ev_, st_));   // the blocks queued so far may still read the old expert
    CUDA_CHECK(cudaStreamWaitEvent(vcp_st_, vdone_ev_, 0));
    CUDA_CHECK(cudaMemcpyAsync(vslab_[best_il].p + (size_t) best_slot * blob, src, blob, cudaMemcpyHostToDevice, vcp_st_));
    CUDA_CHECK(cudaEventRecord(vcp_ev_, vcp_st_));
    vpend_stage_ = best_il; vpend_slot_ = best_slot; vpend_exp_ = best_e;
}

int DsSpark::run(const float * R_rows, const int32_t * next_tokens, int n, int cell0, bool draft, float * prob) {
    if (prob) *prob = 0.f;
    if (!targets_dev_) throw std::runtime_error("dspark: the engine has not bound the target rows");
    DeviceScope dsc(dev_);
    (void) R_rows;   // the rows come from the capture ring by position
    feed(cell0, n);
    if (!draft) return -1;
    draft_block(next_tokens[n - 1], cell0 + n);
    Block & b = blocks_[cur_];
    if (b.toks.empty()) return -1;
    b.next = 1;
    if (prob) *prob = b.conf.empty() ? 1.f : b.conf[0];
    return b.toks[0];
}

int DsSpark::step(int32_t, int, float * prob) {
    Block & b = blocks_[cur_];
    if (prob) *prob = 0.f;
    if (b.next >= (int) b.toks.size()) return -1;
    if (prob) *prob = b.conf.empty() ? 1.f : b.conf[b.next];
    return b.toks[b.next++];
}

void DsSpark::run_multi(const std::vector<RunItem> & items, int32_t * tok_out, float * prob_out) {
    for (size_t i = 0; i < items.size(); ++i) {
        const RunItem & it = items[i];
        select(it.slot);
        tok_out[i] = run(it.R_rows, it.tokens, it.n, it.cell0, true, &prob_out[i]);
    }
    item_slot_.clear();
    for (const auto & it : items) item_slot_.push_back(it.slot);
}

// items name their conversation by index into the previous call (StepItem::prev), not by slot: step each one in its own
// slot's block (stepping cur_, the last run_multi item's slot, gave a two-slot round one conversation's drafts for both)
void DsSpark::step_multi(const std::vector<StepItem> & items, int32_t * tok_out, float * prob_out) {
    std::vector<int> slots;
    slots.reserve(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        const int prev = items[i].prev;
        const int slot = prev >= 0 && prev < (int) item_slot_.size() ? item_slot_[prev] : cur_;
        select(slot);
        tok_out[i] = step(items[i].tok, items[i].cell, &prob_out[i]);
        slots.push_back(slot);
    }
    item_slot_ = std::move(slots);
}

const std::vector<std::pair<int32_t, float>> * DsSpark::draft_dist(int slot, int i) const {
    const Block & b = blocks_[slot];
    if (i < 0 || i >= (int) b.dist.size() || b.dist[i].empty()) return nullptr;
    return &b.dist[i];
}

}  // namespace bnk
