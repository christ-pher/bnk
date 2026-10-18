#include "dense/dense_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "core/util.h"
#include "ggml.h"

namespace bnk {

Config load_dense_config(const GgufModel & g) {
    const std::string arch = g.get_str("general.architecture");
    if (arch != "qwen35") throw std::runtime_error("unsupported architecture '" + arch + "' (expected qwen35)");
    auto k = [&](const char * s) { return arch + "." + s; };
    Config c;
    // block_count counts the MTP ("nextn") block after the trunk (Qwen3.8-27B: 65 = 64 + 1)
    const int nextn = (int) g.get_u64(k("nextn_predict_layers"), 0);
    c.n_layer = (int) g.get_u64(k("block_count")) - nextn;
    if (c.n_layer <= 0) throw std::runtime_error("block_count does not exceed nextn_predict_layers");
    c.n_embd = (int) g.get_u64(k("embedding_length"));
    c.n_ff = (int) g.get_u64(k("feed_forward_length"));
    c.n_head = (int) g.get_u64(k("attention.head_count"));
    c.n_head_kv = (int) g.get_u64(k("attention.head_count_kv"));
    c.head_dim = (int) g.get_u64(k("attention.key_length"));
    c.n_rot = (int) g.get_u64(k("rope.dimension_count"), c.head_dim);
    c.rope_base = (float) g.get_f64(k("rope.freq_base"), 1e7);
    c.rms_eps = (float) g.get_f64(k("attention.layer_norm_rms_epsilon"), 1e-6);
    c.ssm_conv = (int) g.get_u64(k("ssm.conv_kernel"));
    c.ssm_state = (int) g.get_u64(k("ssm.state_size"));
    c.ssm_groups = (int) g.get_u64(k("ssm.group_count"));
    c.ssm_vheads = (int) g.get_u64(k("ssm.time_step_rank"));
    c.ssm_inner = (int) g.get_u64(k("ssm.inner_size"));
    c.full_attn_interval = (int) g.get_u64(k("full_attention_interval"), 4);
    // one residual stream: the engines' residual rows are hc_dim() = n_embd floats
    c.hc = 1;
    c.hc_rank = 0;
    c.compress_ratio.assign(c.n_layer, 0);
    c.eos_token = (int) g.get_u64("tokenizer.ggml.eos_token_id");
    c.pad_token = (int) g.get_u64("tokenizer.ggml.padding_token_id", c.eos_token);
    const TensorRef & emb = g.get("token_embd.weight");
    c.n_vocab = (int) emb.ne[1];
    if (c.n_embd != emb.ne[0]) throw std::runtime_error("embedding width mismatch");
    if (c.ssm_vheads * c.ssm_state != c.ssm_inner) throw std::runtime_error("ssm.inner_size != value heads x state");
    if (c.n_ff <= 0) throw std::runtime_error("feed_forward_length missing");
    return c;
}

DenseModel::~DenseModel() {
    for (void * p : device_allocs_) cudaFree(p);
    for (void * p : host_allocs_) cudaFreeHost(p);
}

QMat DenseModel::upload_tensor(const TensorRef & t, bool repack) {
    // the token table is gathered by row (a few KB per token): page-locked host memory over PCIe keeps its
    // 1.3 GiB of VRAM for the context (BNK_EMBD_HOST=0 puts it on the GPU)
    static const bool embd_host = !getenv("BNK_EMBD_HOST") || atoi(getenv("BNK_EMBD_HOST")) != 0;
    if (embd_host && t.name == "token_embd.weight") {
        const size_t bytes = (size_t) t.nrows() * t.row_bytes();
        void * h = nullptr;
        if (cudaHostAlloc(&h, bytes, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess)
            throw std::runtime_error("cudaHostAlloc failed (token table)");
        memcpy(h, t.data, bytes);
        void * dp = nullptr;
        if (cudaHostGetDevicePointer(&dp, h, 0) != cudaSuccess) throw std::runtime_error("no device pointer for the token table");
        host_allocs_.push_back(h);
        QMat m;
        m.type = (int) t.type;
        m.rows = t.nrows();
        m.cols = t.ne[0];
        m.row_bytes = t.row_bytes();
        m.data = dp;
        return m;
    }
    // BF16 matrices (F16 is read directly by the kernels, BF16 is not) become F16 when they fit its range
    if (t.type == GGML_TYPE_BF16 && t.nrows() >= 2 && t.ne[0] % 32 == 0) {
        const int64_t n = t.nrows() * t.ne[0];
        std::vector<float> f((size_t) n);
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) t.data, f.data(), n);
        float mx = 0.f;
        for (float v : f) mx = std::max(mx, std::fabs(v));
        if (mx < 65504.f) {
            std::vector<ggml_fp16_t> h((size_t) n);
            ggml_fp32_to_fp16_row(f.data(), h.data(), n);
            void * d = nullptr;
            QMat m = upload_matrix(h.data(), GGML_TYPE_F16, t.nrows(), t.ne[0], (size_t) t.ne[0] * 2, repack, &d);
            device_allocs_.push_back(d);
            vram_bytes += m.row_bytes * m.rows;
            return m;
        }
    }
    void * d = nullptr;
    QMat m = upload_matrix(t.data, (int) t.type, t.nrows(), t.ne[0], t.row_bytes(), repack, &d);
    device_allocs_.push_back(d);
    vram_bytes += m.row_bytes * m.rows;
    return m;
}

QMat DenseModel::upload(const std::string & name, bool required) {
    const TensorRef * t = gguf.find(name);
    if (!t) {
        if (required) throw std::runtime_error("missing tensor " + name);
        return QMat{};
    }
    return upload_tensor(*t, name != "token_embd.weight");
}

bool DenseModel::stack(const std::string & na, const std::string & nb, QMat & a, QMat & b, QMat & ab) {
    static const bool stacks = !getenv("BNK_STACK") || atoi(getenv("BNK_STACK")) != 0;
    if (!stacks) return false;
    const TensorRef * ta = gguf.find(na), * tb = gguf.find(nb);
    if (!ta || !tb || ta->type != tb->type || ta->ne[0] != tb->ne[0] || ta->row_bytes() != tb->row_bytes()) return false;
    if (!a.valid() || !b.valid()) return false;
    std::vector<uint8_t> buf((size_t) (ta->nrows() + tb->nrows()) * ta->row_bytes());
    memcpy(buf.data(), ta->data, (size_t) ta->nrows() * ta->row_bytes());
    memcpy(buf.data() + (size_t) ta->nrows() * ta->row_bytes(), tb->data, (size_t) tb->nrows() * tb->row_bytes());
    for (QMat * m : {&a, &b})   // the separate uploads go: the stacked one replaces them
        for (size_t i = 0; i < device_allocs_.size(); ++i)
            if (device_allocs_[i] == m->data) {
                vram_bytes -= m->row_bytes * m->rows;
                cudaFree(device_allocs_[i]);
                device_allocs_.erase(device_allocs_.begin() + (long) i);
                break;
            }
    TensorRef tt = *ta;
    tt.name = na + "|" + nb;
    tt.data = buf.data();
    tt.ne[1] = ta->nrows() + tb->nrows();
    tt.ne[2] = tt.ne[3] = 1;
    tt.nbytes = buf.size();
    ab = upload_tensor(tt, true);
    if (ab.layout != a.layout || ab.type != a.type) {   // (converted differently after all: keep them apart)
        a = upload(na);
        b = upload(nb);
        ab = QMat{};
        return false;
    }
    a = qmat_rows(ab, 0, ta->nrows());
    b = qmat_rows(ab, ta->nrows(), tb->nrows());
    return true;
}

void DenseModel::load_layer(DenseLayer & L, const std::string & b, bool attn) {
    L.attn = attn;
    L.attn_norm = upload(b + "attn_norm.weight");
    L.post_norm = upload(b + "post_attention_norm.weight");
    if (attn) {
        L.wq = upload(b + "attn_q.weight");
        L.wk = upload(b + "attn_k.weight");
        L.wv = upload(b + "attn_v.weight");
        L.wo = upload(b + "attn_output.weight");
        L.q_norm = upload(b + "attn_q_norm.weight");
        L.k_norm = upload(b + "attn_k_norm.weight");
        stack(b + "attn_k.weight", b + "attn_v.weight", L.wk, L.wv, L.wkv);
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
        stack(b + "ssm_beta.weight", b + "ssm_alpha.weight", L.ssm_beta, L.ssm_alpha, L.ssm_ba);
    }
    L.gate = upload(b + "ffn_gate.weight");
    L.up = upload(b + "ffn_up.weight");
    L.down = upload(b + "ffn_down.weight");
    stack(b + "ffn_gate.weight", b + "ffn_up.weight", L.gate, L.up, L.gu);
}

void DenseModel::load(const std::string & path, bool verbose, int first, int end, bool embd, bool head, bool want_mtp) {
    gguf.open(path);
    cfg = load_dense_config(gguf);
    const Config & c = cfg;
    l0 = first;
    l1 = end < 0 ? c.n_layer : end;
    if (verbose && l0 == 0) {
        int na = 0;
        for (int il = 0; il < c.n_layer; ++il) na += c.is_attn(il);
        fprintf(stderr, "bnk: %s: %d layers (%d DeltaNet + %d attention), width %d, FFN %d, vocab %d, %zu shard(s)\n",
                gguf.get_str("general.name").c_str(), c.n_layer, c.n_layer - na, na, c.n_embd, c.n_ff, c.n_vocab,
                gguf.shard_paths().size());
    }
    layers.resize(c.n_layer);
    for (int il = l0; il < l1; ++il) load_layer(layers[il], "blk." + std::to_string(il) + ".", c.is_attn(il));
    if (head) {
        output_norm = upload("output_norm.weight");
        output = upload("output.weight");
    }
    if (embd) tok_embd = upload("token_embd.weight");
    if (want_mtp) {
        const std::string b = "blk." + std::to_string(c.n_layer) + ".";
        if (gguf.find(b + "nextn.eh_proj.weight")) {
            load_layer(mtp.layer, b, true);
            mtp.eh_proj = upload(b + "nextn.eh_proj.weight");
            mtp.enorm = upload(b + "nextn.enorm.weight");
            mtp.hnorm = upload(b + "nextn.hnorm.weight");
            mtp.head_norm = upload(b + "nextn.shared_head_norm.weight");
            if (mtp.eh_proj.cols != 2 * c.n_embd || mtp.eh_proj.rows != c.n_embd)
                throw std::runtime_error("nextn.eh_proj: unexpected shape");
            mtp.present = true;
        } else if (verbose) {
            fprintf(stderr, "bnk: the model file has no MTP block: no speculative decoding\n");
        }
    }
    if (verbose) {
        if (l0 == 0 && l1 == c.n_layer) fprintf(stderr, "bnk: weights in VRAM: %.2f GiB\n", vram_bytes / 1073741824.0);
        else fprintf(stderr, "bnk: layers %d-%d: weights in VRAM: %.2f GiB\n", l0, l1 - 1, vram_bytes / 1073741824.0);
    }
}

}  // namespace bnk
