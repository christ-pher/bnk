#include "core/model.h"

#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>

#include "core/util.h"

namespace bnk {

Config load_config(const GgufModel & g) {
    const std::string arch = g.get_str("general.architecture");
    if (arch != "qwen4exp") throw std::runtime_error("unsupported architecture '" + arch + "' (expected qwen4exp)");
    auto k = [&](const char * s) { return arch + "." + s; };
    Config c;
    c.n_layer = (int) g.get_u64(k("block_count"));
    c.n_embd = (int) g.get_u64(k("embedding_length"));
    c.n_head = (int) g.get_u64(k("attention.head_count"));
    c.n_head_kv = (int) g.get_u64(k("attention.head_count_kv"));
    c.head_dim = (int) g.get_u64(k("attention.key_length"));
    c.n_rot = (int) g.get_u64(k("rope.dimension_count"), c.head_dim);
    c.rope_base = (float) g.get_f64(k("rope.freq_base"), 1e7);
    c.rms_eps = (float) g.get_f64(k("attention.layer_norm_rms_epsilon"), 1e-6);
    c.n_expert = (int) g.get_u64(k("expert_count"));
    c.n_expert_used = (int) g.get_u64(k("expert_used_count"));
    c.n_ff_exp = (int) g.get_u64(k("expert_feed_forward_length"));
    c.n_ff_shexp = (int) g.get_u64(k("expert_shared_feed_forward_length"));
    c.expert_weights_scale = (float) g.get_f64(k("expert_weights_scale"), 0.0);
    c.ssm_conv = (int) g.get_u64(k("ssm.conv_kernel"));
    c.ssm_state = (int) g.get_u64(k("ssm.state_size"));
    c.ssm_groups = (int) g.get_u64(k("ssm.group_count"));
    c.ssm_vheads = (int) g.get_u64(k("ssm.time_step_rank"));
    c.ssm_inner = (int) g.get_u64(k("ssm.inner_size"));
    c.full_attn_interval = (int) g.get_u64(k("full_attention_interval"), 4);
    c.hc = (int) g.get_u64(k("hyper_connection.count"));
    c.hc_rank = (int) g.get_u64(k("hyper_connection.low_rank"));
    c.idx_heads = (int) g.get_u64(k("attention.indexer.head_count"));
    c.idx_dim = (int) g.get_u64(k("attention.indexer.key_length"));
    c.idx_top_k = (int) g.get_u64(k("attention.indexer.top_k"));
    for (auto v : g.get_int_arr(k("attention.compress_ratios"))) c.compress_ratio.push_back((int) v);
    c.compress_ratio.resize(c.n_layer, 0);

    auto ple_layers = g.get_int_arr(k("ple.layers"));
    if (!ple_layers.empty()) {
        if (ple_layers.size() != 1) throw std::runtime_error("only one PLE layer is supported");
        c.ple_layer = (int) ple_layers[0];
        c.ple_ngram = (int) g.get_u64(k("ple.ngram_size"));
        c.ple_heads_per_ngram = (int) g.get_u64(k("ple.heads_per_ngram"));
        c.ple_conv = (int) g.get_u64(k("ple.conv_kernel"));
        c.ple_eos = (int) g.get_u64(k("ple.eos_token_id"));
        c.ple_image = (int) g.get_u64(k("ple.image_token_id"), 0);
        c.ple_dim = (int) g.get_u64(k("embedding_length_per_layer_input"));
        for (auto v : g.get_int_arr(k("ple.layer_multipliers"))) c.ple_mult.push_back((uint64_t) v);
        for (auto v : g.get_int_arr(k("ple.head_offsets"))) c.ple_offset.push_back((uint64_t) v);
        for (auto v : g.get_int_arr(k("ple.head_vocab_sizes"))) c.ple_vocab.push_back((uint64_t) v);
        if ((int) c.ple_mult.size() < c.ple_ngram || (int) c.ple_offset.size() < c.ple_heads() ||
            (int) c.ple_vocab.size() < c.ple_heads())
            throw std::runtime_error("PLE metadata arrays are too short");
    }
    c.eos_token = (int) g.get_u64("tokenizer.ggml.eos_token_id");
    c.pad_token = (int) g.get_u64("tokenizer.ggml.padding_token_id", c.eos_token);
    const TensorRef & emb = g.get("token_embd.weight");
    c.n_vocab = (int) emb.ne[1];
    if (c.n_embd != emb.ne[0]) throw std::runtime_error("embedding width mismatch");
    return c;
}

QMat Model::upload(const std::string & name, bool required) {
    const TensorRef * t = gguf.find(name);
    if (!t) {
        if (required) throw std::runtime_error("missing tensor " + name);
        return QMat{};
    }
    void * d = nullptr;
    // token embeddings stay in ggml layout (gathered by row); every other matrix is repacked when possible
    const bool repack = repack_ && name != "token_embd.weight";
    QMat m = upload_matrix(t->data, (int) t->type, t->nrows(), t->ne[0], t->row_bytes(), repack, &d);
    device_allocs_.push_back(d);
    vram_dense_bytes += m.row_bytes * m.rows;
    return m;
}

Model::~Model() {
    for (void * p : device_allocs_) cudaFree(p);
}

void Model::load(const std::string & path, bool verbose) {
    gguf.open(path);
    cfg = load_config(gguf);
    const Config & c = cfg;
    if (verbose)
        fprintf(stderr, "bnk: %s: %d layers, %d experts (%d used), hc %d, %zu shard(s)\n",
                gguf.get_str("general.name").c_str(), c.n_layer, c.n_expert, c.n_expert_used, c.hc,
                gguf.shard_paths().size());

    auto hcw = [&](const std::string & p, bool inject) {
        HcWeights h;
        h.norm = upload(p + "_norm.weight");
        h.down = upload(p + "_down.weight");
        h.up = upload(p + "_up.weight");
        if (inject) h.inject = upload(p + "_inject.weight");
        return h;
    };

    layers.resize(c.n_layer);
    for (int il = 0; il < c.n_layer; ++il) {
        LayerWeights & L = layers[il];
        const std::string b = "blk." + std::to_string(il) + ".";
        L.hc_attn = hcw(b + "hc_attn", true);
        L.hc_ffn = hcw(b + "hc_ffn", true);
        L.attn = c.is_attn(il);
        if (L.attn) {
            L.wq = upload(b + "attn_q.weight");
            L.wk = upload(b + "attn_k.weight");
            L.wv = upload(b + "attn_v.weight");
            L.wo = upload(b + "attn_output.weight");
            L.q_norm = upload(b + "attn_q_norm.weight");
            L.k_norm = upload(b + "attn_k_norm.weight");
            L.idx_q = upload(b + "indexer.q_proj.weight", false);
            L.idx_k = upload(b + "indexer.k_proj.weight", false);
            L.idx_q_norm = upload(b + "indexer.q_norm.weight", false);
            L.idx_k_norm = upload(b + "indexer.k_norm.weight", false);
        } else {
            L.wqkv = upload(b + "attn_qkv.weight");
            L.wgate = upload(b + "attn_gate.weight");
            L.ssm_beta = upload(b + "ssm_beta.weight");
            L.ssm_alpha = upload(b + "ssm_alpha.weight");
            L.ssm_out = upload(b + "ssm_out.weight");
            L.ssm_conv1d = upload(b + "ssm_conv1d.weight");
            L.ssm_dt = upload(b + "ssm_dt.bias");
            L.ssm_a = upload(b + "ssm_a");
            L.ssm_norm = upload(b + "ssm_norm.weight");
        }
        if (il == c.ple_layer) {
            L.ple_key = upload(b + "ple_key.weight");
            L.ple_value = upload(b + "ple_value.weight");
            L.ple_norm_key = upload(b + "ple_norm_key.weight");
            L.ple_norm_query = upload(b + "ple_norm_query.weight");
            L.ple_norm_conv = upload(b + "ple_norm_conv.weight");
            L.ple_conv1d = upload(b + "ple_conv1d.weight");
        }
        L.router = upload(b + "ffn_gate_inp.weight");
        L.shexp_gate_inp = upload(b + "ffn_gate_inp_shexp.weight");
        L.sh_gate = upload(b + "ffn_gate_shexp.weight");
        L.sh_up = upload(b + "ffn_up_shexp.weight");
        L.sh_down = upload(b + "ffn_down_shexp.weight");

        const TensorRef & g = gguf.get(b + "ffn_gate_exps.weight");
        const TensorRef & u = gguf.get(b + "ffn_up_exps.weight");
        const TensorRef & d = gguf.get(b + "ffn_down_exps.weight");
        if (g.ne[2] != c.n_expert || u.ne[2] != c.n_expert || d.ne[2] != c.n_expert)
            throw std::runtime_error("expert tensor shape mismatch in layer " + std::to_string(il));
        L.gate_type = g.type; L.up_type = u.type; L.down_type = d.type;
        L.gate_bytes = g.nbytes / c.n_expert;
        L.up_bytes = u.nbytes / c.n_expert;
        L.down_bytes = d.nbytes / c.n_expert;
        L.gate_src = g.data; L.up_src = u.data; L.down_src = d.data;
    }
    hc_head = hcw("output_hc", false);
    output = upload("output.weight");
    tok_embd = upload("token_embd.weight");
    if (c.ple_layer >= 0) ple_table = &gguf.get("per_layer_token_embd.weight");
    if (verbose) fprintf(stderr, "bnk: dense weights in VRAM: %.2f GiB\n", vram_dense_bytes / 1073741824.0);
}

}  // namespace bnk
