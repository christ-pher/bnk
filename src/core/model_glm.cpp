// glm5next (GLM-5.3-Flash): hyper-parameters from the GGUF written by tools/convert_glm_fp4.py and the placement
// of its weights. The architecture (docs/GLM.md): 45 layers, each an attention block (34 KDA linear-attention
// layers, 11 MLA layers with a DSA indexer; NoPE) and an MLP block (3 leading dense layers, then 288 routed
// experts top-8 with one shared expert), both wrapped in mHC hyper-connections over 4 residual streams.
#include "core/model.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "core/util.h"
#include "ggml.h"

namespace bnk {

Config load_config_glm(const GgufModel & g) {
    auto k = [&](const char * s) { return std::string("glm5next.") + s; };
    Config c;
    c.arch = Arch::GLM5NEXT;
    c.n_layer = (int) g.get_u64(k("block_count"));
    c.n_embd = (int) g.get_u64(k("embedding_length"));
    c.n_ff = (int) g.get_u64(k("feed_forward_length"));
    c.n_expert = (int) g.get_u64(k("expert_count"));
    c.n_expert_used = (int) g.get_u64(k("expert_used_count"));
    c.n_ff_exp = (int) g.get_u64(k("expert_feed_forward_length"));
    c.n_ff_shexp = (int) g.get_u64(k("expert_shared_feed_forward_length"));
    c.expert_weights_scale = (float) g.get_f64(k("expert_weights_scale"), 1.0);
    c.router_norm = g.get_bool(k("expert_weights_norm"), true);
    c.router_sigmoid = g.get_str(k("expert_gating_func"), "sigmoid") == "sigmoid";
    c.swiglu_limit = (float) g.get_f64(k("swiglu_limit"), 0.0);
    c.rms_eps = (float) g.get_f64(k("attention.layer_norm_rms_epsilon"), 1e-5);
    for (auto v : g.get_int_arr(k("layer_types"))) c.layer_type.push_back((int) v);
    for (auto v : g.get_int_arr(k("mlp_layer_types"))) c.mlp_type.push_back((int) v);
    if ((int) c.layer_type.size() != c.n_layer || (int) c.mlp_type.size() != c.n_layer)
        throw std::runtime_error("glm5next: layer_types / mlp_layer_types do not match block_count");
    // hyper-connections
    c.hc = (int) g.get_u64(k("hc.mult"));
    c.hc_rank = 0;
    c.hc_eps = (float) g.get_f64(k("hc.eps"), 1e-6);
    c.hc_sinkhorn = (int) g.get_u64(k("hc.sinkhorn_iters"), 20);
    // KDA, in DeltaNet's terms: nk = nv = heads, S = head dim, conv channels q | k | v
    c.ssm_vheads = (int) g.get_u64(k("kda.head_count"));
    c.ssm_groups = c.ssm_vheads;
    c.ssm_state = (int) g.get_u64(k("kda.head_dim"));
    c.ssm_inner = c.ssm_vheads * c.ssm_state;
    c.ssm_conv = (int) g.get_u64(k("kda.conv_kernel"));
    c.kda_lower = (float) g.get_f64(k("kda.gate_lower_bound"), -5.0);
    // MLA + DSA: the cache row is the latent (one "KV head" of kv_lora halves)
    c.n_head = (int) g.get_u64(k("attention.head_count"));
    c.q_lora = (int) g.get_u64(k("attention.q_lora_rank"));
    c.kv_lora = (int) g.get_u64(k("attention.kv_lora_rank"));
    c.qk_dim = (int) g.get_u64(k("attention.key_length"));
    c.v_dim = (int) g.get_u64(k("attention.value_length"));
    c.n_rot = (int) g.get_u64(k("attention.rope_length"), 0);
    c.n_head_kv = 1;
    c.head_dim = c.kv_lora;
    c.idx_heads = (int) g.get_u64(k("attention.indexer.head_count"));
    c.idx_dim = (int) g.get_u64(k("attention.indexer.key_length"));
    c.idx_top_k = (int) g.get_u64(k("attention.indexer.top_k"));
    c.idx_kpool = (int) g.get_u64(k("attention.indexer.kpool"), 4);
    c.idx_tail = g.get_bool(k("attention.indexer.select_tail"), true);
    c.compress_ratio.assign(c.n_layer, 0);
    for (int il = 0; il < c.n_layer; ++il)
        if (c.layer_type[il] == 1) c.compress_ratio[il] = c.idx_kpool;
    c.full_attn_interval = 0;
    if (c.n_rot != 0) throw std::runtime_error("glm5next: rotary dims are not supported (NoPE expected)");
    c.eos_token = (int) g.get_u64("tokenizer.ggml.eos_token_id");
    c.pad_token = (int) g.get_u64("tokenizer.ggml.padding_token_id", c.eos_token);
    for (auto v : g.get_int_arr(k("eos_token_ids"))) c.eos_tokens.push_back((int) v);
    if (c.eos_tokens.empty()) c.eos_tokens.push_back(c.eos_token);
    const TensorRef & emb = g.get("token_embd.weight");
    c.n_vocab = (int) emb.ne[1];
    if (c.n_embd != emb.ne[0]) throw std::runtime_error("embedding width mismatch");
    return c;
}

void Model::load_glm(bool verbose, bool embd, bool head) {
    const Config & c = cfg;
    if (verbose && l0 == 0)
        fprintf(stderr, "bnk: %s: glm5next, %d layers (%d KDA, %d MLA), %d experts (%d used), hc %d, %zu shard(s)\n",
                gguf.get_str("general.name").c_str(), c.n_layer,
                (int) std::count(c.layer_type.begin(), c.layer_type.end(), 0),
                (int) std::count(c.layer_type.begin(), c.layer_type.end(), 1), c.n_expert, c.n_expert_used, c.hc,
                gguf.shard_paths().size());
    layers.resize(c.n_layer);
    for (int il = 0; il < c.n_layer; ++il) {
        LayerWeights & L = layers[il];
        const std::string b = "blk." + std::to_string(il) + ".";
        L.attn = c.is_attn(il);
        L.moe = c.is_moe(il);
        if (L.moe) expert_src(il);
        if (il < l0 || il >= l1) continue;   // another stage's layer
        L.hc_attn_fn = upload(b + "hc_attn_fn.weight");
        L.hc_attn_base = upload(b + "hc_attn_base");
        L.hc_attn_scale = upload(b + "hc_attn_scale");
        L.hc_ffn_fn = upload(b + "hc_ffn_fn.weight");
        L.hc_ffn_base = upload(b + "hc_ffn_base");
        L.hc_ffn_scale = upload(b + "hc_ffn_scale");
        L.attn_norm = upload(b + "attn_norm.weight");
        L.ffn_norm = upload(b + "ffn_norm.weight");
        if (L.attn) {
            L.attn_qkv_a = upload(b + "attn_qkv_a.weight");
            L.attn_q_a_norm = upload(b + "attn_q_a_norm.weight");
            L.attn_kv_a_norm = upload(b + "attn_kv_a_norm.weight");
            L.attn_q_b_idx = upload(b + "attn_q_b_idx.weight");
            L.attn_kv_b = upload(b + "attn_kv_b.weight");
            L.wo = upload(b + "attn_output.weight");
            L.idx_x = upload(b + "idx_x.weight");
            L.idx_k_norm_w = upload(b + "idx_k_norm.weight");
            L.idx_k_norm_b = upload(b + "idx_k_norm.bias");
            L.idx_ape = upload(b + "idx_ape");
            if (L.attn_kv_b.type != GGML_TYPE_F16)
                throw std::runtime_error(b + "attn_kv_b.weight: expected F16 after loading, got " + ggml_type_name((ggml_type) L.attn_kv_b.type));
        } else {
            L.wqkv = upload(b + "kda_qkv.weight");
            L.ssm_conv1d = upload(b + "kda_conv.weight");
            L.ssm_out = upload(b + "kda_out.weight");
            L.kda_fgb_a = upload(b + "kda_fgb_a.weight");
            L.kda_f_b = upload(b + "kda_f_b.weight");
            L.kda_g_b = upload(b + "kda_g_b.weight");
            L.ssm_dt = upload(b + "kda_dt_bias");
            L.ssm_a = upload(b + "kda_a_log");
            L.ssm_norm = upload(b + "kda_o_norm.weight");
        }
        if (L.moe) {
            L.router = upload(b + "ffn_gate_inp.weight");
            L.router_bias = upload(b + "ffn_gate_inp.bias");
            L.sh_gu = upload(b + "ffn_gate_up_shexp.weight");
            L.sh_gate = qmat_rows(L.sh_gu, 0, c.n_ff_shexp);
            L.sh_up = qmat_rows(L.sh_gu, c.n_ff_shexp, c.n_ff_shexp);
            L.sh_down = upload(b + "ffn_down_shexp.weight");
        } else {
            L.ffn_gate_up = upload(b + "ffn_gate_up.weight");
            L.ffn_down = upload(b + "ffn_down.weight");
        }
    }
    if (head) {
        hc_head.norm = upload("output_norm.weight");
        output = upload("output.weight");
    }
    if (embd) tok_embd = upload("token_embd.weight");
    if (verbose) {
        int absent = 0;
        for (int il = 0; il < c.n_layer; ++il) absent += layers[il].moe && !layers[il].experts_present;
        if (absent) fprintf(stderr, "bnk: WARNING: %d MoE layers have no converted experts in this file (partial conversion)\n", absent);
        if (l0 == 0 && l1 == c.n_layer) fprintf(stderr, "bnk: dense weights in VRAM: %.2f GiB\n", vram_dense_bytes / 1073741824.0);
        else fprintf(stderr, "bnk: layers %d-%d: dense weights in VRAM: %.2f GiB\n", l0, l1 - 1, vram_dense_bytes / 1073741824.0);
    }
}

}  // namespace bnk
