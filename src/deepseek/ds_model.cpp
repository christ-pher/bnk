#include "deepseek/ds_model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>
#include <sys/mman.h>

#include "core/util.h"
#include "ggml.h"

namespace bnk {

DsConfig load_ds_config(const GgufModel & g) {
    const std::string arch = g.get_str("general.architecture");
    if (arch != "deepseek41") throw std::runtime_error("unsupported architecture '" + arch + "' (expected deepseek41)");
    auto k = [&](const char * s) { return arch + "." + s; };
    auto need = [&](const char * s) {
        if (!g.has_key(k(s))) throw std::runtime_error("missing GGUF key " + k(s));
        return g.get_u64(k(s));
    };
    DsConfig c;
    c.n_layer = (int) need("block_count");
    c.n_embd = (int) need("embedding_length");
    c.n_head = (int) need("attention.head_count");
    if (g.get_u64(k("attention.head_count_kv"), 1) != 1) throw std::runtime_error("deepseek41: expected one KV head");
    c.head_dim = (int) need("attention.key_length");
    c.n_rot = (int) need("rope.dimension_count");
    c.q_lora = (int) need("attention.q_lora_rank");
    c.o_groups = (int) need("attention.output_group_count");
    c.o_lora = (int) need("attention.output_lora_rank");
    c.window = (int) need("attention.sliding_window");
    c.idx_heads = (int) need("attention.indexer.head_count");
    c.idx_dim = (int) need("attention.indexer.key_length");
    c.idx_topk = (int) need("attention.indexer.top_k");
    c.rope_base = (float) g.get_f64(k("rope.freq_base"), 1e4);
    c.comp_rope_base = (float) g.get_f64(k("attention.compress_rope_freq_base"), c.rope_base);
    c.yarn_factor = (float) g.get_f64(k("rope.scaling.factor"), 1.0);
    c.yarn_orig = (int) g.get_u64(k("rope.scaling.original_context_length"), 0);
    c.yarn_beta_fast = (float) g.get_f64(k("rope.scaling.yarn_beta_fast"), 32.0);
    c.yarn_beta_slow = (float) g.get_f64(k("rope.scaling.yarn_beta_slow"), 1.0);
    if (g.get_str(k("rope.scaling.type"), "yarn") != "yarn") c.yarn_orig = 0;
    c.eps = (float) g.get_f64(k("attention.layer_norm_rms_epsilon"), 1e-20);
    c.hc = (int) need("hyper_connection.count");
    c.sinkhorn = (int) g.get_u64(k("hyper_connection.sinkhorn_iterations"), 20);
    c.hc_eps = (float) g.get_f64(k("hyper_connection.epsilon"), 1e-6);
    c.n_expert = (int) need("expert_count");
    c.n_used = (int) need("expert_used_count");
    c.n_ff = (int) need("expert_feed_forward_length");
    c.route_scale = (float) g.get_f64(k("expert_weights_scale"), 1.0);
    c.route_norm = g.get_bool(k("expert_weights_norm"), true);
    if (g.get_str(k("expert_scoring_func"), "sqrtsoftplus") != "sqrtsoftplus" && g.get_u64(k("expert_gating_func"), 4) != 4)
        throw std::runtime_error("deepseek41: expected sqrtsoftplus routing");
    {
        const auto cl = g.get_f32_arr(k("swiglu_clamp_exp"));
        c.swiglu_limit = cl.empty() ? 0.f : cl[0];
        for (float v : cl) if (v != c.swiglu_limit) throw std::runtime_error("deepseek41: per-layer swiglu clamps differ");
    }
    c.ratio.assign(c.n_layer, 0);
    {
        const auto r = g.get_int_arr(k("attention.compress_ratios"));
        for (int il = 0; il < c.n_layer && il < (int) r.size(); ++il) c.ratio[il] = (int) r[il];
    }
    const auto kvs = g.get_int_arr(k("kv_source_layer_ids")), ids = g.get_int_arr(k("index_source_layer_ids"));
    c.kv_src.assign(c.n_layer, -1);
    c.topk_src.assign(c.n_layer, -1);
    int last_kv = -1, last_idx = -1;
    for (int il = 0; il < c.n_layer; ++il) {
        for (auto v : kvs) if ((int) v == il) last_kv = il;
        for (auto v : ids) if ((int) v == il) last_idx = il;
        if (c.ratio[il] == 0) continue;
        if (last_kv < 0 || last_idx < 0 || c.ratio[last_kv] != c.ratio[il] || c.ratio[last_idx] != c.ratio[il] || c.ratio[il] > 2)
            throw std::runtime_error("deepseek41: layer " + std::to_string(il) + " has no matching kv / index source");
        c.kv_src[il] = last_kv;
        c.topk_src[il] = last_idx;
    }
    c.cand_src = (int) (int64_t) g.get_u64(k("attention.candidate_source_layer_id"), (uint64_t) -1);
    if (c.cand_src >= c.n_layer) c.cand_src = -1;
    if (c.cand_src >= 0) {
        c.cand_blocks = (int) need("attention.candidate_topk_blocks");
        c.cand_bs = (int) need("attention.candidate_block_size");
        if (c.topk_src[c.cand_src] != c.cand_src) throw std::runtime_error("deepseek41: the candidate source is not an index source");
    }
    // Engram
    for (auto v : g.get_int_arr(k("engram.layer_ids"))) c.engram_layers.push_back((int) v);
    if (!c.engram_layers.empty()) {
        c.eg_ngram = (int) need("engram_max_ngram_size");
        c.eg_heads = (int) need("engram_n_heads");
        c.eg_dim = (int) need("engram_head_dim");
        c.eg_pad = (int) g.get_u64(k("engram_pad_token_id"), 2);
        for (auto v : g.get_int_arr(k("engram.num_embeddings"))) c.eg_rows.push_back((int64_t) v);
        const size_t n_e = c.engram_layers.size(), cols = (size_t) c.eg_cols();
        if (c.eg_rows.size() != n_e) throw std::runtime_error("deepseek41: engram.num_embeddings does not match the layers");
        const TensorRef & tm = g.get("engram_token_map.weight");
        const TensorRef & pr = g.get("engram_primes.weight");
        const TensorRef & of = g.get("engram_offsets.weight");
        const TensorRef & mu = g.get("engram_multipliers.weight");
        if (tm.type != GGML_TYPE_I32 || pr.type != GGML_TYPE_I64 || of.type != GGML_TYPE_I64 || mu.type != GGML_TYPE_I64)
            throw std::runtime_error("deepseek41: unexpected Engram metadata tensor types");
        if ((size_t) pr.ne[0] != cols || (size_t) pr.ne[1] != n_e || (size_t) of.ne[0] != cols || (size_t) of.ne[1] != n_e ||
            mu.ne[0] != c.eg_ngram || (size_t) mu.ne[1] != n_e)
            throw std::runtime_error("deepseek41: Engram metadata tensor shapes do not match the config");
        c.eg_token_map.assign((const int32_t *) tm.data, (const int32_t *) tm.data + tm.ne[0]);
        c.eg_primes.resize(n_e); c.eg_offsets.resize(n_e); c.eg_mult.resize(n_e);
        for (size_t e = 0; e < n_e; ++e) {
            const int64_t * p = (const int64_t *) pr.data + e * cols, * o = (const int64_t *) of.data + e * cols,
                          * m = (const int64_t *) mu.data + e * c.eg_ngram;
            c.eg_primes[e].assign(p, p + cols);
            c.eg_offsets[e].assign(o, o + cols);
            c.eg_mult[e].assign(m, m + c.eg_ngram);
            for (size_t j = 0; j < cols; ++j)
                if (c.eg_primes[e][j] < 2 || (int64_t) (c.eg_offsets[e][j] + c.eg_primes[e][j]) > c.eg_rows[e])
                    throw std::runtime_error("deepseek41: an Engram bucket range lies outside its table");
        }
    }
    const TensorRef & emb = g.get("token_embd.weight");
    c.n_vocab = (int) emb.ne[1];
    if (c.n_embd != emb.ne[0]) throw std::runtime_error("embedding width mismatch");
    return c;
}

QMat DsModel::upload_f32(const std::vector<float> & v, int64_t rows, int64_t cols) {
    void * d = nullptr;
    QMat m = upload_matrix(v.data(), GGML_TYPE_F32, rows, cols, (size_t) cols * 4, false, &d);
    device_allocs_.push_back(d);
    vram_dense_bytes += m.row_bytes * m.rows;
    return m;
}

QMat DsModel::upload_tensor(const TensorRef & t, const std::string & name) {
    void * d = nullptr;
    // the token table is gathered by row: page-locked host memory read over PCIe (as the Qwen engines do)
    static const bool embd_host = !getenv("BNK_EMBD_HOST") || atoi(getenv("BNK_EMBD_HOST")) != 0;
    if (embd_host && name == "token_embd.weight") {
        const size_t bytes = (size_t) t.nrows() * t.row_bytes();
        void * h = nullptr;
        if (cudaHostAlloc(&h, bytes, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess)
            throw std::runtime_error("cudaHostAlloc failed (token table)");
        memcpy(h, t.data, bytes);
        void * dp = nullptr;
        if (cudaHostGetDevicePointer(&dp, h, 0) != cudaSuccess) throw std::runtime_error("no device pointer for the token table");
        host_allocs_.push_back(h);
        QMat m;
        m.type = (int) t.type; m.rows = t.nrows(); m.cols = t.ne[0]; m.row_bytes = t.row_bytes(); m.data = dp;
        return m;
    }
    if (t.type == GGML_TYPE_BF16) {
        const int64_t n = t.nrows() * t.ne[0];
        std::vector<float> f((size_t) n);
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) t.data, f.data(), n);
        // the router picks the experts (near ties): fp32; the small vectors too; the rest fp16 when in range
        const bool f32 = name.find("ffn_gate_inp") != std::string::npos || t.nrows() < 8;
        float mx = 0.f;
        for (float v : f) mx = std::max(mx, std::fabs(v));
        if (f32 || mx >= 65504.f) return upload_f32(f, t.nrows(), t.ne[0]);
        std::vector<ggml_fp16_t> h((size_t) n);
        ggml_fp32_to_fp16_row(f.data(), h.data(), n);
        QMat m = upload_matrix(h.data(), GGML_TYPE_F16, t.nrows(), t.ne[0], (size_t) t.ne[0] * 2, false, &d);
        device_allocs_.push_back(d);
        vram_dense_bytes += m.row_bytes * m.rows;
        return m;
    }
    QMat m = upload_matrix(t.data, (int) t.type, t.nrows(), t.ne[0], t.row_bytes(), true, &d);
    device_allocs_.push_back(d);
    vram_dense_bytes += m.row_bytes * m.rows;
    return m;
}

QMat DsModel::upload(const std::string & name, bool required) {
    const TensorRef * t = gguf.find(name);
    if (!t) {
        if (required) throw std::runtime_error("missing tensor " + name);
        return QMat{};
    }
    return upload_tensor(*t, name);
}

bool DsModel::stack(const std::string & na, const std::string & nb, QMat & a, QMat & b, QMat & ab) {
    const TensorRef * ta = gguf.find(na), * tb = gguf.find(nb);
    if (!ta || !tb || ta->type != tb->type || ta->ne[0] != tb->ne[0] || ta->row_bytes() != tb->row_bytes()) return false;
    if (ta->type == GGML_TYPE_BF16 || ta->type == GGML_TYPE_F32 || ta->type == GGML_TYPE_F16) return false;
    std::vector<uint8_t> buf((size_t) (ta->nrows() + tb->nrows()) * ta->row_bytes());
    memcpy(buf.data(), ta->data, (size_t) ta->nrows() * ta->row_bytes());
    memcpy(buf.data() + (size_t) ta->nrows() * ta->row_bytes(), tb->data, (size_t) tb->nrows() * tb->row_bytes());
    for (QMat * m : {&a, &b})
        for (size_t i = 0; i < device_allocs_.size(); ++i)
            if (device_allocs_[i] == m->data) {
                vram_dense_bytes -= m->row_bytes * m->rows;
                cudaFree(device_allocs_[i]);
                device_allocs_.erase(device_allocs_.begin() + (long) i);
                break;
            }
    void * d = nullptr;
    ab = upload_matrix(buf.data(), (int) ta->type, ta->nrows() + tb->nrows(), ta->ne[0], ta->row_bytes(), true, &d);
    device_allocs_.push_back(d);
    vram_dense_bytes += ab.row_bytes * ab.rows;
    a = qmat_rows(ab, 0, ta->nrows());
    b = qmat_rows(ab, ta->nrows(), tb->nrows());
    return true;
}

void DsModel::expert_src(int il) {
    const std::string b = "blk." + std::to_string(il) + ".";
    const TensorRef & g = gguf.get(b + "ffn_gate_exps.weight");
    const TensorRef & u = gguf.get(b + "ffn_up_exps.weight");
    const TensorRef & d = gguf.get(b + "ffn_down_exps.weight");
    const int n = ds.n_expert;
    if (g.ne[2] != n || u.ne[2] != n || d.ne[2] != n || g.ne[0] != ds.n_embd || g.ne[1] != ds.n_ff || d.ne[0] != ds.n_ff ||
        d.ne[1] != ds.n_embd)
        throw std::runtime_error("expert tensor shape mismatch in layer " + std::to_string(il));
    LayerWeights & L = ev.layers[il];
    L.gate_type = g.type; L.up_type = u.type; L.down_type = d.type;
    L.gate_bytes = g.nbytes / n; L.up_bytes = u.nbytes / n; L.down_bytes = d.nbytes / n;
    L.gate_src = g.data; L.up_src = u.data; L.down_src = d.data;
}

DsModel::~DsModel() {
    for (void * p : device_allocs_) cudaFree(p);
    for (void * p : host_allocs_) cudaFreeHost(p);
}

void DsModel::load(const std::string & path, const std::string & engram_path, bool verbose, int first, int end, bool embd,
                   bool head) {
    gguf.open(path);
    ds = load_ds_config(gguf);
    const DsConfig & c = ds;
    l0 = first;
    l1 = end < 0 ? c.n_layer : end;
    // the shared Config: what the generator, the server and the expert machinery read
    cfg = Config{};
    cfg.n_layer = c.n_layer; cfg.n_embd = c.n_embd; cfg.n_vocab = c.n_vocab;
    cfg.n_head = c.n_head; cfg.n_head_kv = 1; cfg.head_dim = c.head_dim; cfg.n_rot = c.n_rot;
    cfg.rope_base = c.rope_base; cfg.rms_eps = c.eps;
    cfg.n_expert = c.n_expert; cfg.n_expert_used = c.n_used; cfg.n_ff_exp = c.n_ff; cfg.n_ff_shexp = c.n_ff;
    cfg.expert_weights_scale = c.route_scale;
    cfg.hc = c.hc;
    cfg.full_attn_interval = 1;   // every layer is an attention layer
    cfg.compress_ratio = c.ratio;
    cfg.eos_token = (int) gguf.get_u64("tokenizer.ggml.eos_token_id", 1);
    cfg.pad_token = (int) gguf.get_u64("tokenizer.ggml.padding_token_id", cfg.eos_token);
    ev.cfg = cfg;
    ev.layers.assign(c.n_layer, LayerWeights{});
    if (verbose && l0 == 0)
        fprintf(stderr, "bnk: %s: %d layers, %d experts (%d used), hc %d, %zu shard(s), swiglu clamp %.0f\n",
                gguf.get_str("general.name").c_str(), c.n_layer, c.n_expert, c.n_used, c.hc, gguf.shard_paths().size(),
                c.swiglu_limit);

    layers.resize(c.n_layer);
    for (int il = 0; il < c.n_layer; ++il) {
        expert_src(il);
        if (il < l0 || il >= l1) continue;
        DsLayer & L = layers[il];
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
        if (L.o_a.rows != (int64_t) c.o_groups * c.o_lora || L.o_a.cols != (int64_t) c.n_head * c.head_dim / c.o_groups ||
            L.o_b.cols != (int64_t) c.o_groups * c.o_lora)
            throw std::runtime_error("attention output projection shapes do not match the config in layer " + std::to_string(il));
        L.hc_attn_fn = upload(b + "hc_attn_fn.weight");
        L.hc_attn_base = upload(b + "hc_attn_base.weight");
        L.hc_attn_scale = upload(b + "hc_attn_scale.weight");
        L.hc_ffn_fn = upload(b + "hc_ffn_fn.weight");
        L.hc_ffn_base = upload(b + "hc_ffn_base.weight");
        L.hc_ffn_scale = upload(b + "hc_ffn_scale.weight");
        if (L.hc_attn_fn.type != GGML_TYPE_F32 || L.hc_attn_fn.rows != c.mix_dim() || L.hc_attn_fn.cols != c.hc_dim())
            throw std::runtime_error("hyper-connection mixer: unexpected shape or type in layer " + std::to_string(il));
        if (c.is_kv_src(il)) {
            L.comp_kv = upload(b + "attn_compressor_kv.weight");
            L.comp_norm = upload(b + "attn_compressor_norm.weight");
            if (c.ratio[il] > 1) L.comp_gate = upload(b + "attn_compressor_gate.weight");
            L.idx_k = upload(b + "indexer.attn_k.weight");
            L.idx_k_norm = upload(b + "indexer.k_norm.weight");
        }
        if (c.is_idx_src(il)) {
            L.idx_q_b = upload(b + "indexer.attn_q_b.weight");
            L.idx_proj = upload(b + "indexer.proj.weight");
        }
        if (c.engram_index(il) >= 0) {
            L.eg_wkv = upload(b + "engram_wkv.weight");
            if (L.eg_wkv.cols != (int64_t) c.eg_cols() * c.eg_dim || L.eg_wkv.rows != (int64_t) c.n_embd * (c.hc + 1))
                throw std::runtime_error("engram_wkv: unexpected shape in layer " + std::to_string(il));
            // the gate weight is only ever used as q_weight * k_weight (bf16 x bf16 is exact in fp32)
            const TensorRef & q = gguf.get(b + "engram_q.weight");
            const TensorRef & kk = gguf.get(b + "engram_k.weight");
            if (q.type != GGML_TYPE_BF16 || kk.type != GGML_TYPE_BF16 || q.ne[0] != c.n_embd || q.ne[1] != c.hc ||
                kk.ne[0] != c.n_embd || kk.ne[1] != c.hc)
                throw std::runtime_error("engram_q / engram_k: unexpected shape or type in layer " + std::to_string(il));
            const int64_t n = (int64_t) c.hc * c.n_embd;
            std::vector<float> fq((size_t) n), fk((size_t) n);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) q.data, fq.data(), n);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) kk.data, fk.data(), n);
            for (int64_t i = 0; i < n; ++i) fq[(size_t) i] *= fk[(size_t) i];
            L.eg_gate = upload_f32(fq, c.hc, c.n_embd);
        }
        L.router = upload(b + "ffn_gate_inp.weight");
        L.probs_b = upload(b + "exp_probs_b.bias");
        if (L.router.type != GGML_TYPE_F32 || L.probs_b.type != GGML_TYPE_F32)
            throw std::runtime_error("router / bias: expected fp32 in layer " + std::to_string(il));
        L.sh_gate = upload(b + "ffn_gate_shexp.weight");
        L.sh_up = upload(b + "ffn_up_shexp.weight");
        L.sh_down = upload(b + "ffn_down_shexp.weight");
        static const bool stacks = !getenv("BNK_STACK") || atoi(getenv("BNK_STACK")) != 0;
        if (stacks) stack(b + "ffn_gate_shexp.weight", b + "ffn_up_shexp.weight", L.sh_gate, L.sh_up, L.sh_gu);
    }
    if (head) {
        output_norm = upload("output_norm.weight");
        output = upload("output.weight");
    }
    if (embd) tok_embd = upload("token_embd.weight");

    // the Engram tables: a second GGUF, mapped; rows are read at hashed positions
    bool owns_engram = false;
    for (int il : c.engram_layers) owns_engram |= il >= l0 && il < l1;
    if (owns_engram) {
        std::string ep = engram_path;
        if (ep.empty()) {
            const std::string f = gguf.get_str("deepseek41.engram.file", "");
            if (f.empty()) throw std::runtime_error("the model names no Engram file; pass --engram FILE");
            const std::string dir = path.substr(0, path.find_last_of('/') + 1);
            ep = dir + f;
        }
        engram_gguf.open(ep);
        if (engram_gguf.get_str("general.architecture") != "deepseek41-engram")
            throw std::runtime_error(ep + " is not a deepseek41 Engram file");
        engram_tab.assign(c.engram_layers.size(), nullptr);
        double gib = 0;
        for (size_t e = 0; e < c.engram_layers.size(); ++e) {
            const int il = c.engram_layers[e];
            const TensorRef & t = engram_gguf.get("blk." + std::to_string(il) + ".engram_embd.weight");
            if (t.ne[0] != c.eg_dim || t.ne[1] < c.eg_rows[e])
                throw std::runtime_error("the Engram table of layer " + std::to_string(il) + " does not match the model's metadata");
            engram_tab[e] = &t;
            engram_gguf.advise(t, MADV_RANDOM);
            gib += t.nbytes / 1073741824.0;
        }
        if (verbose) fprintf(stderr, "bnk: Engram tables: %zu layers, %.1f GiB (%s), memory-mapped from %s\n",
                             c.engram_layers.size(), gib, ggml_type_name(engram_tab[0]->type), ep.c_str());
    }
    if (verbose) {
        if (l0 == 0 && l1 == c.n_layer) fprintf(stderr, "bnk: dense weights in VRAM: %.2f GiB\n", vram_dense_bytes / 1073741824.0);
        else fprintf(stderr, "bnk: layers %d-%d: dense weights in VRAM: %.2f GiB\n", l0, l1 - 1, vram_dense_bytes / 1073741824.0);
    }
}

}  // namespace bnk
