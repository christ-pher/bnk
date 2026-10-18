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

// the bytes of row ranges of t, concatenated
static std::vector<uint8_t> gather_rows(const TensorRef & t, const DenseModel::Ranges & rows, int64_t * nrows) {
    const size_t rb = t.row_bytes();
    int64_t n = 0;
    for (auto & r : rows) n += r.second - r.first;
    std::vector<uint8_t> buf((size_t) n * rb);
    size_t off = 0;
    for (auto & r : rows) {
        if (r.first < 0 || r.second > t.nrows() || r.second < r.first) throw std::runtime_error(t.name + ": row slice out of range");
        memcpy(buf.data() + off, t.data + (size_t) r.first * rb, (size_t) (r.second - r.first) * rb);
        off += (size_t) (r.second - r.first) * rb;
    }
    *nrows = n;
    return buf;
}

QMat DenseModel::upload_rows(const std::string & name, const Ranges & rows) {
    const TensorRef & t = gguf.get(name);
    int64_t n = 0;
    std::vector<uint8_t> buf = gather_rows(t, rows, &n);
    TensorRef tt = t;
    tt.data = buf.data();
    tt.ne[1] = n;
    tt.ne[2] = tt.ne[3] = 1;
    tt.nbytes = buf.size();
    return upload_tensor(tt, name != "token_embd.weight");
}

// column ranges of every row (whole quant blocks): the row's bytes for those columns, concatenated
QMat DenseModel::upload_cols(const std::string & name, const Ranges & cols) {
    const TensorRef & t = gguf.get(name);
    const int64_t blck = ggml_blck_size(t.type);
    const size_t tsz = ggml_type_size(t.type);
    int64_t ncols = 0;
    for (auto & c : cols) {
        if (c.first % blck || c.second % blck || c.first < 0 || c.second > t.ne[0] || c.second < c.first)
            throw std::runtime_error(name + ": column slice not in whole " + std::to_string(blck) + "-wide blocks");
        ncols += c.second - c.first;
    }
    const size_t rb = t.row_bytes(), nrb = (size_t) (ncols / blck) * tsz;
    const int64_t nrows = t.nrows();
    std::vector<uint8_t> buf((size_t) nrows * nrb);
    for (int64_t r = 0; r < nrows; ++r) {
        size_t off = 0;
        for (auto & c : cols) {
            const size_t bytes = (size_t) ((c.second - c.first) / blck) * tsz;
            memcpy(buf.data() + r * nrb + off, t.data + r * rb + (size_t) (c.first / blck) * tsz, bytes);
            off += bytes;
        }
    }
    TensorRef tt = t;
    tt.data = buf.data();
    tt.ne[0] = ncols;
    tt.ne[1] = nrows;
    tt.ne[2] = tt.ne[3] = 1;
    tt.nbytes = buf.size();
    return upload_tensor(tt, true);
}

// row ranges of two matrices of the same format and width stacked (a | b); a and b become row views
QMat DenseModel::upload_rows2(const std::string & na, const std::string & nb, const Ranges & ra, const Ranges & rb,
                              QMat & a, QMat & b) {
    const TensorRef & ta = gguf.get(na), & tb = gguf.get(nb);
    if (ta.type != tb.type || ta.ne[0] != tb.ne[0]) {
        a = upload_rows(na, ra);
        b = upload_rows(nb, rb);
        return QMat{};
    }
    int64_t n1 = 0, n2 = 0;
    std::vector<uint8_t> b1 = gather_rows(ta, ra, &n1), b2 = gather_rows(tb, rb, &n2);
    b1.insert(b1.end(), b2.begin(), b2.end());
    TensorRef tt = ta;
    tt.name = na + "|" + nb;
    tt.data = b1.data();
    tt.ne[1] = n1 + n2;
    tt.ne[2] = tt.ne[3] = 1;
    tt.nbytes = b1.size();
    QMat ab = upload_tensor(tt, true);
    a = qmat_rows(ab, 0, n1);
    b = qmat_rows(ab, n1, n2);
    return ab;
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
    if (tp_n > 1) {
        // this rank's slice: output rows of the projections into the block, input columns of the projections out
        const Config & c = cfg;
        const int r = tp_rank, D = c.head_dim, S = c.ssm_state;
        const int Hl = c.n_head / tp_n, Hkvl = c.n_head_kv / tp_n, nkl = c.ssm_groups / tp_n, nvl = c.ssm_vheads / tp_n;
        const int FFl = c.n_ff / tp_n, per_k = c.ssm_vheads / c.ssm_groups;
        L.gu = upload_rows2(b + "ffn_gate.weight", b + "ffn_up.weight", {{(int64_t) r * FFl, (int64_t) (r + 1) * FFl}},
                            {{(int64_t) r * FFl, (int64_t) (r + 1) * FFl}}, L.gate, L.up);
        L.down = upload_cols(b + "ffn_down.weight", {{(int64_t) r * FFl, (int64_t) (r + 1) * FFl}});
        if (attn) {
            L.wq = upload_rows(b + "attn_q.weight", {{(int64_t) r * Hl * 2 * D, (int64_t) (r + 1) * Hl * 2 * D}});
            L.wkv = upload_rows2(b + "attn_k.weight", b + "attn_v.weight", {{(int64_t) r * Hkvl * D, (int64_t) (r + 1) * Hkvl * D}},
                                 {{(int64_t) r * Hkvl * D, (int64_t) (r + 1) * Hkvl * D}}, L.wk, L.wv);
            L.wo = upload_cols(b + "attn_output.weight", {{(int64_t) r * Hl * D, (int64_t) (r + 1) * Hl * D}});
            L.q_norm = upload(b + "attn_q_norm.weight");
            L.k_norm = upload(b + "attn_k_norm.weight");
        } else {
            // key groups [r nkl, (r+1) nkl); value heads h with h % nk in that range: in the tiled order, per_k runs
            // of nkl heads (h = j nk + r nkl + i), so local head j nkl + i has local group i
            Ranges vh, vrows, vcols;
            for (int j = 0; j < per_k; ++j) {
                const int64_t h0 = (int64_t) j * c.ssm_groups + (int64_t) r * nkl;
                vh.push_back({h0, h0 + nkl});
                vrows.push_back({h0 * S, (h0 + nkl) * S});
            }
            Ranges qkv{{(int64_t) r * nkl * S, (int64_t) (r + 1) * nkl * S},
                       {(int64_t) c.ssm_groups * S + (int64_t) r * nkl * S, (int64_t) c.ssm_groups * S + (int64_t) (r + 1) * nkl * S}};
            Ranges conv_rows = qkv;
            for (auto & v : vrows) {
                qkv.push_back({(int64_t) 2 * c.ssm_groups * S + v.first, (int64_t) 2 * c.ssm_groups * S + v.second});
                conv_rows.push_back({(int64_t) 2 * c.ssm_groups * S + v.first, (int64_t) 2 * c.ssm_groups * S + v.second});
            }
            L.wqkv = upload_rows(b + "attn_qkv.weight", qkv);
            L.wgate = upload_rows(b + "attn_gate.weight", vrows);
            L.ssm_ba = upload_rows2(b + "ssm_beta.weight", b + "ssm_alpha.weight", vh, vh, L.ssm_beta, L.ssm_alpha);
            L.ssm_out = upload_cols(b + "ssm_out.weight", vrows);
            L.ssm_conv1d = upload_rows(b + "ssm_conv1d.weight", conv_rows);
            // dt and a are vectors over the value heads: their "rows" are elements (ne[0] = nv): slice the columns
            L.ssm_dt = upload_cols(b + "ssm_dt.bias", vh);
            L.ssm_a = upload_cols(b + "ssm_a", vh);
            L.ssm_norm = upload(b + "ssm_norm.weight");
        }
        (void) nvl;
        return;
    }
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

void DenseModel::load(const std::string & path, bool verbose, int first, int end, bool embd, bool head, bool want_mtp,
                      int tpn, int tpr) {
    gguf.open(path);
    cfg = load_dense_config(gguf);
    const Config & c = cfg;
    tp_n = tpn;
    tp_rank = tpr;
    lc = cfg;
    if (tp_n > 1) {
        if (c.n_head % tp_n || c.n_head_kv % tp_n || c.ssm_groups % tp_n || c.n_ff % tp_n ||
            (c.n_head / tp_n) % (c.n_head_kv / tp_n))
            throw std::runtime_error("tensor parallel: the heads, KV heads, DeltaNet groups and FFN must divide by " + std::to_string(tp_n));
        lc.n_head = c.n_head / tp_n;
        lc.n_head_kv = c.n_head_kv / tp_n;
        lc.ssm_groups = c.ssm_groups / tp_n;
        lc.ssm_vheads = c.ssm_vheads / tp_n;
        lc.ssm_inner = c.ssm_inner / tp_n;
        lc.n_ff = c.n_ff / tp_n;
    }
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
            const int save_tp = tp_n;
            tp_n = 1;   // the drafter runs whole on one GPU
            load_layer(mtp.layer, b, true);
            tp_n = save_tp;
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
        if (tp_n > 1) fprintf(stderr, "bnk: slice %d of %d: weights in VRAM: %.2f GiB\n", tp_rank + 1, tp_n, vram_bytes / 1073741824.0);
        else if (l0 == 0 && l1 == c.n_layer) fprintf(stderr, "bnk: weights in VRAM: %.2f GiB\n", vram_bytes / 1073741824.0);
        else fprintf(stderr, "bnk: layers %d-%d: weights in VRAM: %.2f GiB\n", l0, l1 - 1, vram_bytes / 1073741824.0);
    }
}

}  // namespace bnk
