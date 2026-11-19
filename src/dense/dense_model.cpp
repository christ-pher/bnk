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
    {
        const auto sec = g.get_int_arr(k("rope.dimension_sections"));
        for (int i = 0; i < 3 && i < (int) sec.size(); ++i) c.rope_sec[i] = (int) sec[i];
    }
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

// the formats the GEMV kernels read (kernels/quant.cuh BNK_DISPATCH_*, plus the float types); anything else is
// re-quantized to Q8_0 at load (a few small tensors of a rate-constrained mix, e.g. Q2_K / IQ1_M)
static bool gemv_format(int t) {
    switch (t) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS: case GGML_TYPE_Q2_0: case GGML_TYPE_NVFP4: case GGML_TYPE_NVFP4_16:
            return true;
        default:
            return false;
    }
}

// a matrix's global scale: the side tensor `<name>.scale` (F32[1]) an NVFP4 checkpoint's converter writes beside
// `<name>.weight` (ModelOpt's weight_scale_2); 1 when absent. The GEMVs and the prompt path's dequantization apply it.
float DenseModel::global_scale(const std::string & wname) const {
    const std::string suf = ".weight";
    if (wname.size() <= suf.size() || wname.compare(wname.size() - suf.size(), suf.size(), suf)) return 1.f;
    const TensorRef * s = gguf.find(wname.substr(0, wname.size() - suf.size()) + ".scale");
    if (!s || s->type != GGML_TYPE_F32 || s->nbytes < 4) return 1.f;
    float v;
    memcpy(&v, s->data, 4);
    return v;
}

QMat DenseModel::upload_tensor(const TensorRef & t, bool repack) {
    QMat m = upload_tensor_raw(t, repack);
    m.scale = global_scale(t.name);
    return m;
}

QMat DenseModel::upload_tensor_raw(const TensorRef & t, bool repack) {
    // BF16 matrices become Q8_0 too (what the Q8_0 release ships them as; the F16 path below is for files that
    // carry F16 matrices), as do the formats the kernels do not read
    if ((!gemv_format((int) t.type) || t.type == GGML_TYPE_BF16) && t.ne[0] % 32 == 0 && t.nrows() >= 2 && t.name != "token_embd.weight") {
        const auto * tt = ggml_get_type_traits(t.type);
        if (!tt || !tt->to_float) throw std::runtime_error(std::string("unsupported weight format ") + ggml_type_name(t.type) + " in " + t.name);
        const int64_t rows = t.nrows(), cols = t.ne[0];
        std::vector<float> f((size_t) rows * cols);
        tt->to_float(t.data, f.data(), rows * cols);
        const size_t rb = ggml_row_size(GGML_TYPE_Q8_0, cols);
        std::vector<uint8_t> q(rb * rows);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, rows, cols, nullptr);
        static bool said = false;
        if (!said) {
            fprintf(stderr, "bnk: %s and other formats the kernels do not read become Q8_0 at load (%s ...)\n", ggml_type_name(t.type), t.name.c_str());
            said = true;
        }
        void * d = nullptr;
        QMat m = upload_matrix(q.data(), GGML_TYPE_Q8_0, rows, cols, rb, repack, &d);
        device_allocs_.push_back(d);
        vram_bytes += m.row_bytes * m.rows;
        return m;
    }
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
    if (ta.type != tb.type || ta.ne[0] != tb.ne[0] || global_scale(na) != global_scale(nb)) {
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
    ab.scale = global_scale(na);
    a = qmat_rows(ab, 0, n1);
    b = qmat_rows(ab, n1, n2);
    return ab;
}

bool DenseModel::stack(const std::string & na, const std::string & nb, QMat & a, QMat & b, QMat & ab) {
    static const bool stacks = !getenv("BNK_STACK") || atoi(getenv("BNK_STACK")) != 0;
    if (!stacks) return false;
    const TensorRef * ta = gguf.find(na), * tb = gguf.find(nb);
    if (!ta || !tb || ta->type != tb->type || ta->ne[0] != tb->ne[0] || ta->row_bytes() != tb->row_bytes()) return false;
    if (global_scale(na) != global_scale(nb)) return false;   // one stacked matrix has one global scale
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
    ab.scale = global_scale(na);
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

// ------------------------------------------------------------------------------------------------ LoRA adapters
static std::vector<float> tensor_floats(const TensorRef & t) {
    std::vector<float> f((size_t) t.nrows() * t.ne[0]);
    if (t.type == GGML_TYPE_F32) {
        memcpy(f.data(), t.data, f.size() * 4);
        return f;
    }
    const auto * tt = ggml_get_type_traits(t.type);
    if (!tt || !tt->to_float) throw std::runtime_error("lora: unsupported format " + std::string(ggml_type_name(t.type)) + " in " + t.name);
    tt->to_float(t.data, f.data(), (int64_t) f.size());
    return f;
}

void DenseModel::release(const QMat & m) {
    for (size_t i = 0; i < device_allocs_.size(); ++i)
        if (device_allocs_[i] == m.data) {
            vram_bytes -= std::min(vram_bytes, m.row_bytes * (size_t) m.rows);
            cudaFree(device_allocs_[i]);
            device_allocs_.erase(device_allocs_.begin() + (long) i);
            return;
        }
}

// rows [r0, r1) x columns [c0, c1) of t in its own format (columns in whole blocks)
static std::vector<uint8_t> slice_bytes(const TensorRef & t, int64_t r0, int64_t r1, int64_t c0, int64_t c1) {
    const int64_t blck = ggml_blck_size(t.type);
    const size_t tsz = ggml_type_size(t.type), rb = t.row_bytes(), nrb = (size_t) ((c1 - c0) / blck) * tsz;
    std::vector<uint8_t> buf((size_t) (r1 - r0) * nrb);
    for (int64_t r = r0; r < r1; ++r) memcpy(buf.data() + (size_t) (r - r0) * nrb, t.data + (size_t) r * rb + (size_t) (c0 / blck) * tsz, nrb);
    return buf;
}

int DenseModel::load_lora(const std::string & path, float user_scale, bool verbose) {
    GgufModel g;
    g.open(path);
    if (g.get_str("general.type") != "adapter" || g.get_str("adapter.type") != "lora")
        throw std::runtime_error(path + ": not a LoRA adapter GGUF (general.type adapter, adapter.type lora)");
    const std::string arch = g.get_str("general.architecture"), march = gguf.get_str("general.architecture");
    if (arch != march) throw std::runtime_error("lora: adapter for '" + arch + "', model is '" + march + "'");
    const double alpha = g.get_f64("adapter.lora.alpha", 0);
    const Config & c = cfg;
    const int r = tp_rank, D = c.head_dim, Hl = c.n_head / tp_n, Hkvl = c.n_head_kv / tp_n, FFl = c.n_ff / tp_n;
    const size_t vram0 = vram_bytes;
    int used = 0, rank_seen = 0;

    // a site's projections (all reading the same input): B rows [r0, r1) and A columns [c0, c1) of this rank (the
    // base matrix's own slice: its rows when row-split, its columns when column-split); `views` the base QMats of
    // the first parts that A's rows are stacked under (q; gate, up; o; down), `joint` gate | up's view
    static const bool stack_ok = !getenv("BNK_LORA_STACK") || atoi(getenv("BNK_LORA_STACK")) != 0;
    int stacked = 0;
    struct Part { std::string w; int64_t r0, r1, c0, c1; };
    auto build = [&](DenseLora & L, const std::vector<Part> & parts, std::vector<QMat *> views, QMat * joint) {
        std::vector<float> A;
        int64_t acols = -1;
        for (size_t i = 0; i < parts.size(); ++i) {
            const Part & p = parts[i];
            const TensorRef * ta = g.find(p.w + ".lora_a"), * tb = g.find(p.w + ".lora_b");
            if (!ta && !tb) continue;
            if (!ta || !tb) throw std::runtime_error("lora: " + p.w + " has only one of lora_a / lora_b");
            const TensorRef & w = gguf.get(p.w);
            const int64_t rr = ta->ne[1];
            if (ta->ne[0] != w.ne[0] || tb->ne[1] != w.ne[1] || tb->ne[0] != rr)
                throw std::runtime_error("lora: " + p.w + ": A/B shapes do not match the weight");
            if (rr % 8) throw std::runtime_error("lora: rank " + std::to_string(rr) + " is not a multiple of 8");
            rank_seen = (int) rr;
            const int64_t nc = p.c1 - p.c0;
            if (nc % 32) throw std::runtime_error("lora: " + p.w + ": column slice not a multiple of 32");
            if (acols >= 0 && acols != nc) throw std::runtime_error("lora: stacked A matrices of different widths");
            acols = nc;
            const std::vector<float> fa = tensor_floats(*ta);   // [rr][in]
            for (int64_t k = 0; k < rr; ++k)
                A.insert(A.end(), fa.begin() + (size_t) (k * ta->ne[0] + p.c0), fa.begin() + (size_t) (k * ta->ne[0] + p.c1));
            const std::vector<float> fb = tensor_floats(*tb);   // [out][rr]
            const float scale = (float) (alpha > 0 ? alpha / (double) rr : 1.0) * user_scale;
            const int64_t nr = p.r1 - p.r0;
            std::vector<float> bs((size_t) (nr * rr));
            for (int64_t k = 0; k < nr * rr; ++k) bs[(size_t) k] = fb[(size_t) (p.r0 * rr + k)] * scale;
            std::vector<ggml_fp16_t> bh(bs.size());
            ggml_fp32_to_fp16_row(bs.data(), bh.data(), (int64_t) bs.size());
            void * d = nullptr;
            L.b[i] = upload_matrix(bh.data(), GGML_TYPE_F16, nr, rr, (size_t) rr * 2, false, &d);
            device_allocs_.push_back(d);
            vram_bytes += (size_t) nr * rr * 2;
            L.off[i] = L.width;
            L.width += (int) rr;
            used += 2;
        }
        if (!L.width) return;
        const size_t rb = ggml_row_size(GGML_TYPE_Q8_0, acols);
        std::vector<uint8_t> q(rb * (size_t) L.width);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, A.data(), q.data(), 0, L.width, acols, nullptr);
        lora_width = std::max(lora_width, L.width);
        bool can = stack_ok;
        for (size_t i = 0; i < views.size(); ++i) {
            const TensorRef & w = gguf.get(parts[i].w);
            can = can && w.type == GGML_TYPE_Q8_0 && global_scale(parts[i].w) == 1.f && views[i]->valid() &&
                  views[i]->layout == 1 && parts[i].c1 - parts[i].c0 == acols;
        }
        if (can) {
            std::vector<uint8_t> buf;
            std::vector<int64_t> nrows;
            for (size_t i = 0; i < views.size(); ++i) {
                const Part & p = parts[i];
                std::vector<uint8_t> s = slice_bytes(gguf.get(p.w), p.r0, p.r1, p.c0, p.c1);
                buf.insert(buf.end(), s.begin(), s.end());
                nrows.push_back(p.r1 - p.r0);
            }
            int64_t base = 0;
            for (int64_t n : nrows) base += n;
            buf.insert(buf.end(), q.begin(), q.end());
            void * d = nullptr;
            QMat big = upload_matrix(buf.data(), GGML_TYPE_Q8_0, base + L.width, acols, rb, true, &d);
            if (big.layout == 1) {
                if (joint && joint->valid()) release(*joint);
                for (QMat * v : views) release(*v);
                device_allocs_.push_back(d);
                vram_bytes += big.row_bytes * (size_t) big.rows;
                int64_t o = 0;
                for (size_t i = 0; i < views.size(); ++i) {
                    *views[i] = qmat_rows(big, o, nrows[i]);
                    o += nrows[i];
                }
                if (joint) *joint = qmat_rows(big, 0, base);
                L.host = big;
                L.host_rows = (int) base;
                L.a = qmat_rows(big, base, L.width);
                ++stacked;
                return;
            }
            cudaFree(d);
        }
        void * d = nullptr;
        L.a = upload_matrix(q.data(), GGML_TYPE_Q8_0, L.width, acols, rb, true, &d);
        device_allocs_.push_back(d);
        vram_bytes += rb * (size_t) L.width;
    };

    int sites = 0;
    for (int il = l0; il < l1; ++il) {
        DenseLayer & L = layers[il];
        const std::string b = "blk." + std::to_string(il) + ".";
        const int64_t E = c.n_embd;
        if (L.attn) {
            const int64_t q = (int64_t) Hl * 2 * D, kv = (int64_t) Hkvl * D, o = (int64_t) Hl * D;
            build(L.lora_qkv, {{b + "attn_q.weight", r * q, (r + 1) * q, 0, E},
                               {b + "attn_k.weight", r * kv, (r + 1) * kv, 0, E},
                               {b + "attn_v.weight", r * kv, (r + 1) * kv, 0, E}}, {&L.wq}, nullptr);
            build(L.lora_o, {{b + "attn_output.weight", 0, E, r * o, (r + 1) * o}}, {&L.wo}, nullptr);
        }
        build(L.lora_gu, {{b + "ffn_gate.weight", (int64_t) r * FFl, (int64_t) (r + 1) * FFl, 0, E},
                          {b + "ffn_up.weight", (int64_t) r * FFl, (int64_t) (r + 1) * FFl, 0, E}}, {&L.gate, &L.up}, &L.gu);
        build(L.lora_down, {{b + "ffn_down.weight", 0, E, (int64_t) r * FFl, (int64_t) (r + 1) * FFl}}, {&L.down}, nullptr);
        sites += L.lora_qkv.valid() + L.lora_o.valid() + L.lora_gu.valid() + L.lora_down.valid();
    }
    // everything the adapter holds for these layers must have been applied (a DeltaNet projection, say, is not)
    int expected = 0, mtp_skipped = 0;
    for (const TensorRef & t : g.tensors()) {
        int il = -1;
        if (sscanf(t.name.c_str(), "blk.%d.", &il) != 1) throw std::runtime_error("lora: tensor " + t.name + " is not applied by this engine");
        if (il >= c.n_layer) { ++mtp_skipped; continue; }
        if (il >= l0 && il < l1) ++expected;
    }
    if (expected != used) throw std::runtime_error("lora: " + std::to_string(expected - used) + " tensor(s) of " + path +
                                                   " target projections this engine does not adapt");
    if (verbose)
        fprintf(stderr, "bnk: lora %s: rank %d, alpha %g, scale %g: %d projection(s) at %d site(s) (%d stacked under their base "
                "matrix) in layers %d-%d, %.1f MiB%s%s\n",
                path.c_str(), rank_seen, alpha, (alpha > 0 && rank_seen ? alpha / rank_seen : 1.0) * user_scale, used / 2, sites,
                stacked, l0, l1 - 1, ((double) vram_bytes - (double) vram0) / 1048576.0, tp_n > 1 ? " (this slice)" : "",
                mtp_skipped ? " (the MTP block's tensors left out)" : "");
    return lora_width;
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
