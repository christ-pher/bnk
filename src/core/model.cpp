#include <cmath>
#include "core/model.h"

#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>

#include "core/util.h"
#include "ggml.h"
#include <vector>
#include <cstdlib>

namespace bnk {

Config load_config(const GgufModel & g) {
    const std::string arch = g.get_str("general.architecture");
    if (arch != "qwen4exp") throw std::runtime_error("unsupported architecture '" + arch + "' (expected qwen4exp)");
    auto k = [&](const char * s) { return arch + "." + s; };
    Config c;
    // block_count includes the MTP ("nextn") prediction blocks some files carry after the trunk (e.g.
    // CYBER-FROST-3.8: 49 = 48 + 1); the trunk is what runs here, the draft layer comes from --mtp
    const int nextn = (int) g.get_u64(k("nextn_predict_layers"), 0);
    c.n_layer = (int) g.get_u64(k("block_count")) - nextn;
    if (c.n_layer <= 0) throw std::runtime_error("block_count does not exceed nextn_predict_layers");
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
    // Some conversions (CYBER-FROST-3.8 PS-GUFF) write compress_ratios as all zeros although the attention layers
    // carry their QSA indexers; the checkpoint's config says indexer_compress_ratio 4, like the base model. Zeros
    // would mean dense attention (llama.cpp's reading), which matches the trained model only within the indexer's
    // budget, so restore the architecture's ratio. BNK_NO_QSA=1 still runs dense attention.
    bool any_ratio = false, indexers = false;
    for (int il = 0; il < c.n_layer; ++il) {
        any_ratio |= c.compress_ratio[il] > 0;
        indexers |= c.is_attn(il) && g.find("blk." + std::to_string(il) + ".indexer.k_proj.weight") != nullptr;
    }
    if (!any_ratio && indexers) {
        constexpr int kQsaRatio = 4;
        for (int il = 0; il < c.n_layer; ++il) if (c.is_attn(il)) c.compress_ratio[il] = kQsaRatio;
        fprintf(stderr, "bnk: compress_ratios are all 0 but the attention layers have QSA indexers: using ratio %d\n",
                kQsaRatio);
    }

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
    return upload_data(name, t->data, (int) t->type, t->nrows(), t->ne[0], t->row_bytes());
}

QMat Model::upload_data(const std::string & name, const uint8_t * data, int type_, int64_t nrows, int64_t ne0,
                        size_t row_bytes) {
    struct { const uint8_t * data; int type; int64_t nrows_, ne0_; size_t rb; int64_t nrows() const { return nrows_; }
             int64_t ne[1]; size_t row_bytes() const { return rb; } } tt{data, type_, nrows, ne0, row_bytes, {ne0}};
    const auto * t = &tt;
    void * d = nullptr;
    // token embeddings stay in ggml layout (gathered by row); every other matrix is repacked when possible
    const bool repack = repack_ && name != "token_embd.weight";
    // BF16 hyper-connection / PLE projections become Q8_0: half the bytes and the dp4a path (the router and
    // the small vectors keep their precision)
    static const std::string which = getenv("BNK_HCQ8") ? getenv("BNK_HCQ8") : "down,inject,ple";
    auto sel = [&](const char * part, const char * key) {
        return which.find(key) != std::string::npos && name.find(part) != std::string::npos;
    };
    const bool to_q8 = hc_q8_ && t->type == GGML_TYPE_BF16 && t->ne[0] % 32 == 0 && t->nrows() >= 4 &&
                       (sel("_down", "down") || sel("_up.", "up") || sel("_inject", "inject") ||
                        sel("ple_key", "ple") || sel("ple_value", "ple"));
    if (to_q8) {
        const int64_t rows = t->nrows(), cols = t->ne[0];
        std::vector<float> f((size_t) rows * cols);
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) t->data, f.data(), rows * cols);
        const size_t rb = ggml_row_size(GGML_TYPE_Q8_0, cols);
        std::vector<uint8_t> q(rb * rows);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, rows, cols, nullptr);
        QMat m = upload_matrix(q.data(), GGML_TYPE_Q8_0, rows, cols, rb, repack, &d);
        device_allocs_.push_back(d);
        vram_dense_bytes += m.row_bytes * m.rows;
        return m;
    }
    // the other BF16 matrices become F16 (exact within fp16's normal range: 3 more mantissa bits), which the
    // tensor cores read directly when several tokens go through them; kept BF16 if a value is out of range
    static const bool keep_bf16 = getenv("BNK_KEEP_BF16") != nullptr;
    // (not the router: its fp32 path picks the experts, and stays exact)
    if (!keep_bf16 && t->type == GGML_TYPE_BF16 && t->nrows() >= 2 && t->ne[0] % 32 == 0 &&
        (name.find("ffn_gate_inp") == std::string::npos || getenv("BNK_ROUTER_F16"))) {
        const int64_t n = t->nrows() * t->ne[0];
        std::vector<float> f((size_t) n);
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) t->data, f.data(), n);
        float mx = 0.f;
        for (float v : f) mx = std::max(mx, std::fabs(v));
        if (mx < 65504.f) {
            std::vector<ggml_fp16_t> h((size_t) n);
            ggml_fp32_to_fp16_row(f.data(), h.data(), n);
            QMat m = upload_matrix(h.data(), GGML_TYPE_F16, t->nrows(), t->ne[0], (size_t) t->ne[0] * 2, repack, &d);
            device_allocs_.push_back(d);
            vram_dense_bytes += m.row_bytes * m.rows;
            return m;
        }
    }
    QMat m = upload_matrix(t->data, (int) t->type, t->nrows(), t->ne[0], t->row_bytes(), repack, &d);
    device_allocs_.push_back(d);
    vram_dense_bytes += m.row_bytes * m.rows;
    return m;
}

// The two tensors stacked as one matrix (their host rows concatenated) through the same conversions as upload():
// a and b become views of the result's rows. False (a and b stay separate) when the formats or widths differ.
bool Model::stack(const std::string & na, const std::string & nb, QMat & a, QMat & b, QMat & ab) {
    const TensorRef * ta = gguf.find(na), * tb = gguf.find(nb);
    if (!ta || !tb || ta->type != tb->type || ta->ne[0] != tb->ne[0] || ta->row_bytes() != tb->row_bytes()) return false;
    if (!a.valid() || !b.valid()) return false;
    // the conversions in upload() depend on the name: both must convert the same way
    auto kind = [&](const std::string & n) {
        return (int) (n.find("_down") != std::string::npos || n.find("_inject") != std::string::npos ||
                      n.find("ple_") != std::string::npos) * 2 + (int) (n.find("ffn_gate_inp") != std::string::npos);
    };
    if (ta->type == GGML_TYPE_BF16 && kind(na) != kind(nb)) return false;
    std::vector<uint8_t> buf((size_t) (ta->nrows() + tb->nrows()) * ta->row_bytes());
    memcpy(buf.data(), ta->data, (size_t) ta->nrows() * ta->row_bytes());
    memcpy(buf.data() + (size_t) ta->nrows() * ta->row_bytes(), tb->data, (size_t) tb->nrows() * tb->row_bytes());
    // the old separate uploads are released: the stacked one replaces them
    for (QMat * m : {&a, &b})
        for (size_t i = 0; i < device_allocs_.size(); ++i)
            if (device_allocs_[i] == m->data) {
                vram_dense_bytes -= m->row_bytes * m->rows;
                cudaFree(device_allocs_[i]);
                device_allocs_.erase(device_allocs_.begin() + (long) i);
                break;
            }
    ab = upload_data(na, buf.data(), (int) ta->type, ta->nrows() + tb->nrows(), ta->ne[0], ta->row_bytes());
    if (ab.layout != a.layout || ab.type != a.type) {   // (conversions differ after all: keep them apart)
        a = upload(na);
        b = upload(nb);
        ab = QMat{};
        return false;
    }
    a = qmat_rows(ab, 0, ta->nrows());
    b = qmat_rows(ab, ta->nrows(), tb->nrows());
    return true;
}

Model::~Model() {
    for (void * p : device_allocs_) cudaFree(p);
}

void Model::load(const std::string & path, bool verbose, const std::string & ple_path, int first, int end, bool embd,
                 bool head) {
    gguf.open(path);
    cfg = load_config(gguf);
    const Config & c = cfg;
    l0 = first;
    l1 = end < 0 ? c.n_layer : end;
    if (verbose && l0 == 0)
        fprintf(stderr, "bnk: %s: %d layers, %d experts (%d used), hc %d, %zu shard(s)\n",
                gguf.get_str("general.name").c_str(), c.n_layer, c.n_expert, c.n_expert_used, c.hc,
                gguf.shard_paths().size());

    static const bool stacks = !getenv("BNK_STACK") || atoi(getenv("BNK_STACK")) != 0;
    auto hcw = [&](const std::string & p, bool inject) {
        HcWeights h;
        h.norm = upload(p + "_norm.weight");
        h.down = upload(p + "_down.weight");
        h.up = upload(p + "_up.weight");
        if (inject) {
            h.inject = upload(p + "_inject.weight");
            if (stacks) stack(p + "_down.weight", p + "_inject.weight", h.down, h.inject, h.down_inject);
        }
        return h;
    };

    layers.resize(c.n_layer);
    for (int il = 0; il < c.n_layer; ++il) {
        LayerWeights & L = layers[il];
        const std::string b = "blk." + std::to_string(il) + ".";
        L.attn = c.is_attn(il);
        if (il < l0 || il >= l1) {   // another stage's layer: only where its experts are (the expert store needs them)
            expert_src(il);
            continue;
        }
        L.hc_attn = hcw(b + "hc_attn", true);
        L.hc_ffn = hcw(b + "hc_ffn", true);
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
            if (stacks) {
                stack(b + "attn_k.weight", b + "attn_v.weight", L.wk, L.wv, L.wkv);
                if (L.idx_q.valid()) stack(b + "indexer.k_proj.weight", b + "indexer.q_proj.weight", L.idx_k, L.idx_q, L.idx_kq);
            }
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
            if (stacks) stack(b + "ssm_beta.weight", b + "ssm_alpha.weight", L.ssm_beta, L.ssm_alpha, L.ssm_ba);
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
        if (stacks) {
            stack(b + "ffn_gate_inp.weight", b + "ffn_gate_inp_shexp.weight", L.router, L.shexp_gate_inp, L.router_sg);
            stack(b + "ffn_gate_shexp.weight", b + "ffn_up_shexp.weight", L.sh_gate, L.sh_up, L.sh_gu);
        }
        expert_src(il);
    }
    if (head) {
        hc_head = hcw("output_hc", false);
        output = upload("output.weight");
    }
    if (embd) tok_embd = upload("token_embd.weight");
    if (c.ple_layer >= l0 && c.ple_layer < l1) {
        // the PLE table may live in another GGUF (--ple-gguf): a fine-tune that left it untouched can share the
        // base quant's copy instead of storing another 28.8 GB. The model's own table wins when it has one.
        ple_table = gguf.find("per_layer_token_embd.weight");
        if (!ple_table) {
            if (ple_path.empty())
                throw std::runtime_error("per_layer_token_embd.weight is not in the model; pass --ple-gguf FILE");
            ple_gguf.open(ple_path);
            ple_table = &ple_gguf.get("per_layer_token_embd.weight");
            const uint64_t rows = c.ple_offset[c.ple_heads() - 1] + c.ple_vocab[c.ple_heads() - 1];
            if (ple_table->ne[0] != c.ple_dim || (uint64_t) ple_table->ne[1] < rows)
                throw std::runtime_error("the PLE table in " + ple_path + " does not match the model's PLE metadata");
            if (verbose) fprintf(stderr, "bnk: PLE table (%.2f GiB, %s) from %s\n", ple_table->nbytes / 1073741824.0,
                                 ggml_type_name(ple_table->type), ple_path.c_str());
        }
    }
    if (verbose) {
        if (l0 == 0 && l1 == c.n_layer) fprintf(stderr, "bnk: dense weights in VRAM: %.2f GiB\n", vram_dense_bytes / 1073741824.0);
        else fprintf(stderr, "bnk: layers %d-%d: dense weights in VRAM: %.2f GiB\n", l0, l1 - 1, vram_dense_bytes / 1073741824.0);
    }
}

void Model::expert_src(int il) {
    const Config & c = cfg;
    LayerWeights & L = layers[il];
    const std::string b = "blk." + std::to_string(il) + ".";
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
    // per-expert global scales (NVFP4): absent means 1
    const TensorRef * sg = gguf.find(b + "ffn_gate_exps.scale"), * su = gguf.find(b + "ffn_up_exps.scale"),
                    * sd = gguf.find(b + "ffn_down_exps.scale");
    if (sg || su || sd) {
        auto read = [&](const TensorRef * t, std::vector<float> & v) {
            v.assign(c.n_expert, 1.f);
            if (!t) return;
            if (t->type != GGML_TYPE_F32 || t->ne[0] != c.n_expert)
                throw std::runtime_error("expert scale tensor " + t->name + ": expected F32[" + std::to_string(c.n_expert) + "]");
            memcpy(v.data(), t->data, (size_t) c.n_expert * 4);
        };
        std::vector<float> vu, vd, wm(c.n_expert);
        read(sg, L.exp_sgate); read(su, vu); read(sd, vd);
        for (int e = 0; e < c.n_expert; ++e) wm[e] = vu[e] * vd[e];
        auto up = [&](const std::vector<float> & v) {
            void * p = nullptr;
            if (cudaMalloc(&p, v.size() * 4) != cudaSuccess) throw std::runtime_error("cudaMalloc failed (expert scales)");
            cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
            device_allocs_.push_back(p);
            return (const float *) p;
        };
        L.exp_sgate_dev = up(L.exp_sgate);
        L.exp_wmul_dev = up(wm);
    }
}

}  // namespace bnk
