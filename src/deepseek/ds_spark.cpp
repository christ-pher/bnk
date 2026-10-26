#include "deepseek/ds_spark.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "deepseek/ds_ops.h"
#include "ggml.h"

namespace bnk {

DsSpark::~DsSpark() = default;

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
    CUDA_CHECK(cudaStreamSynchronize(st));
    if (verbose)
        fprintf(stderr, "bnk: DSpark drafter: %d stages, %d experts (%d used) a stage, blocks of %d, target layers %d-%d; dense %.2f GiB in VRAM, experts %.2f GiB in RAM (%s)\n",
                c_.n_layer, c_.n_expert, c_.n_used, block_, targets_.front(), targets_.back(), vram_bytes_ / 1073741824.0,
                store_.total_bytes() / 1073741824.0, ggml_type_str((ggml_type) ev_.layers[0].gate_type).c_str());
    loaded_ = true;
}

void DsSpark::restart(int pos) {
    Slot & s = slots_[cur_];
    if (s.fed - pos > kDsRing - c_.window) s.valid_lo = pos;
    else s.valid_lo = std::min(s.valid_lo, pos);
    s.fed = std::min(s.fed, pos);
    blocks_[cur_] = Block{};
}

// ---- the forward: not written yet (docs/DSPARK.md, the port's steps 2-4)
void DsSpark::feed(const float *, int, int) { throw std::runtime_error("dspark: feed() is not implemented yet"); }
void DsSpark::draft_block(int32_t, int) { throw std::runtime_error("dspark: draft_block() is not implemented yet"); }

int DsSpark::run(const float *, const int32_t * next_tokens, int n, int cell0, bool draft, float * prob) {
    if (prob) *prob = 0.f;
    if (!targets_dev_) throw std::runtime_error("dspark: the engine has not bound the target rows");
    feed(targets_dev_, n, cell0);
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
}

void DsSpark::step_multi(const std::vector<StepItem> & items, int32_t * tok_out, float * prob_out) {
    for (size_t i = 0; i < items.size(); ++i) tok_out[i] = step(items[i].tok, items[i].cell, &prob_out[i]);
}

const std::vector<std::pair<int32_t, float>> * DsSpark::draft_dist(int slot, int i) const {
    const Block & b = blocks_[slot];
    if (i < 0 || i >= (int) b.dist.size() || b.dist[i].empty()) return nullptr;
    return &b.dist[i];
}

}  // namespace bnk
