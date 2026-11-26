#include "inkling/ink_model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "ggml.h"

namespace bnk {

namespace {
std::string blk(int il, const char * name) { return "blk." + std::to_string(il) + "." + name; }
}

QMat InkModel::upload(const std::string & name, bool required) {
    const TensorRef * t = gguf.find(name);
    if (!t) {
        if (required) throw std::runtime_error("inkling: missing tensor " + name);
        return {};
    }
    void * d = nullptr;
    if (cudaMalloc(&d, t->nbytes) != cudaSuccess) throw std::runtime_error("inkling: cudaMalloc failed for " + name);
    if (cudaMemcpy(d, t->data, t->nbytes, cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("inkling: upload failed for " + name);
    allocs_.push_back(d);
    vram_bytes += t->nbytes;
    QMat m;
    m.data = d;
    m.type = (int) t->type;
    m.cols = t->ne[0];
    m.rows = t->nrows();
    m.row_bytes = t->row_bytes();
    return m;
}

const float * InkModel::upload_f32(const std::string & name, size_t expect, bool required) {
    const TensorRef * t = gguf.find(name);
    if (!t) {
        if (required) throw std::runtime_error("inkling: missing tensor " + name);
        return nullptr;
    }
    if (t->type != GGML_TYPE_F32) throw std::runtime_error("inkling: " + name + " is not F32");
    if (expect && (size_t) (t->nbytes / 4) != expect)
        throw std::runtime_error("inkling: " + name + " has " + std::to_string(t->nbytes / 4) + " values, expected " +
                                 std::to_string(expect));
    return (const float *) upload(name).data;
}

const float * InkModel::upload_vec(const std::vector<float> & v) {
    void * d = nullptr;
    if (cudaMalloc(&d, v.size() * 4) != cudaSuccess || cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("inkling: upload failed");
    allocs_.push_back(d);
    return (const float *) d;
}

QMat InkModel::upload_bytes(const std::string & name, const std::vector<uint8_t> & host, int64_t rows, int64_t cols,
                            int type) {
    void * d = nullptr;
    if (cudaMalloc(&d, host.size()) != cudaSuccess) throw std::runtime_error("inkling: cudaMalloc failed for " + name);
    if (cudaMemcpy(d, host.data(), host.size(), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("inkling: upload failed for " + name);
    allocs_.push_back(d);
    vram_bytes += host.size();
    QMat m;
    m.data = d;
    m.type = type;
    m.rows = rows;
    m.cols = cols;
    m.row_bytes = ggml_row_size((ggml_type) type, cols);
    return m;
}

QMat InkModel::upload_rows(const std::string & name, const Ranges & rows) {
    const TensorRef & t = gguf.get(name);
    const size_t rb = t.row_bytes();
    std::vector<uint8_t> host;
    int64_t n = 0;
    for (auto [r0, r1] : rows) {
        host.insert(host.end(), t.data + (size_t) r0 * rb, t.data + (size_t) r1 * rb);
        n += r1 - r0;
    }
    return upload_bytes(name, host, n, t.ne[0], (int) t.type);
}

QMat InkModel::upload_stack(const std::vector<std::pair<std::string, Ranges>> & parts) {
    std::vector<uint8_t> host;
    int64_t n = 0, cols = -1;
    int type = -1;
    for (const auto & [name, rows] : parts) {
        const TensorRef & t = gguf.get(name);
        if (cols >= 0 && (t.ne[0] != cols || (int) t.type != type)) throw std::runtime_error("inkling: cannot stack " + name);
        cols = t.ne[0];
        type = (int) t.type;
        const size_t rb = t.row_bytes();
        for (auto [r0, r1] : rows) {
            host.insert(host.end(), t.data + (size_t) r0 * rb, t.data + (size_t) r1 * rb);
            n += r1 - r0;
        }
    }
    return upload_bytes(parts[0].first, host, n, cols, type);
}

QMat InkModel::upload_cols(const std::string & name, const Ranges & cols) {
    const TensorRef & t = gguf.get(name);
    const size_t rb = t.row_bytes();
    const int64_t bs = ggml_blck_size(t.type);
    const size_t ts = ggml_type_size(t.type);
    int64_t nc = 0;
    for (auto [c0, c1] : cols) {
        if (c0 % bs || c1 % bs) throw std::runtime_error("inkling: a column slice inside a quant block: " + name);
        nc += c1 - c0;
    }
    std::vector<uint8_t> host;
    host.reserve((size_t) t.nrows() * nc / bs * ts);
    for (int64_t r = 0; r < t.nrows(); ++r)
        for (auto [c0, c1] : cols)
            host.insert(host.end(), t.data + r * rb + c0 / bs * ts, t.data + r * rb + c1 / bs * ts);
    return upload_bytes(name, host, t.nrows(), nc, (int) t.type);
}

const float * InkModel::upload_f32_range(const std::string & name, int64_t off, int64_t n) {
    const TensorRef & t = gguf.get(name);
    if (t.type != GGML_TYPE_F32 || (int64_t) (t.nbytes / 4) < off + n) throw std::runtime_error("inkling: " + name);
    std::vector<float> v((const float *) t.data + off, (const float *) t.data + off + n);
    return upload_vec(v);
}

InkModel::~InkModel() {
    for (void * p : allocs_) cudaFree(p);
}

void InkModel::load(const std::string & path, bool verbose, int tp_n, int tp_rank) {
    gguf.open(path);
    const std::string arch = gguf.get_str("general.architecture");
    if (arch != "inkling2_loop") throw std::runtime_error("inkling: unsupported architecture '" + arch + "'");
    auto k = [&](const char * s) { return arch + "." + s; };
    InkConfig & c = cfg;
    c.n_layer = (int) gguf.get_u64(k("block_count"));
    c.E = (int) gguf.get_u64(k("embedding_length"));
    c.n_head = (int) gguf.get_u64(k("attention.head_count"));
    c.n_head_kv = (int) gguf.get_u64(k("attention.head_count_kv"));
    c.head_dim = (int) gguf.get_u64(k("attention.key_length"));
    c.n_expert = (int) gguf.get_u64(k("expert_count"));
    c.n_used = (int) gguf.get_u64(k("expert_used_count"));
    c.n_ff = (int) gguf.get_u64(k("expert_feed_forward_length"));
    c.n_vocab_pad = (int) gguf.get_u64(k("vocab_size"));
    c.n_vocab = (int) gguf.get_u64(k("vocab_size_unpadded"));
    c.rms_eps = (float) gguf.get_f64(k("attention.layer_norm_rms_epsilon"), 1e-6);
    c.qk_clip = (float) gguf.get_f64(k("attention.qk_clip"));
    c.rel_clip = (float) gguf.get_f64(k("attention.relative_clip"));
    c.n_rel = (int) gguf.get_u64(k("attention.relative_length"));
    c.rel_extent = (int) gguf.get_u64(k("attention.relative_extent"));
    c.n_swa = (int) gguf.get_u64(k("attention.sliding_window"));
    c.n_conv = (int) gguf.get_u64(k("shortconv.l_cache"));
    c.res_block = (int) gguf.get_u64(k("attn_res.block_size"));
    c.log_floor = (float) gguf.get_f64(k("attention.log_scale_floor"));
    c.log_alpha = (float) gguf.get_f64(k("attention.log_scale_alpha"));
    c.situ_beta = (float) gguf.get_f64(k("activation.situ_beta"));
    c.situ_lin = (float) gguf.get_f64(k("activation.situ_linear_beta"));
    c.w_scale = (float) gguf.get_f64(k("expert_weights_scale"), 1.0);
    c.softcap = (float) gguf.get_f64(k("final_logit_softcapping"), 0.0);
    c.loop_start = (int) gguf.get_u64(k("loop.start"));
    c.loop_end = (int) gguf.get_u64(k("loop.end"));
    c.loop_count = (int) gguf.get_u64(k("loop.count"));
    c.gate_rank = (int) gguf.get_u64(k("loop.gate_rank"));
    c.eos = (int) gguf.get_u64("tokenizer.ggml.eos_token_id");
    c.pad = (int) gguf.get_u64("tokenizer.ggml.padding_token_id");
    if (gguf.get_bool(k("loop.varying"), false)) throw std::runtime_error("inkling: a varying loop is not supported");
    for (int64_t v : gguf.get_int_arr(k("attention.sliding_window_pattern"))) c.swa.push_back(v != 0);
    if ((int) c.swa.size() != c.n_layer || c.head_dim != 128 || c.n_head % c.n_head_kv || c.n_conv != 4 ||
        c.n_rel != 16 || c.loop_start <= 0 || c.loop_end > c.n_layer || c.loop_start % c.res_block ||
        c.loop_end % c.res_block || c.loop_count < 2 || (int64_t) c.n_ff * 2 * c.n_expert != gguf.get("blk.0.ffn_gate_up_exps.weight").nrows())
        throw std::runtime_error("inkling: unexpected dimensions (this port expects Inkling-2-Mini's)");

    base.n_layer = c.n_layer;
    base.n_embd = c.E;
    base.n_vocab = c.n_vocab;
    base.n_head = c.n_head;
    base.n_head_kv = c.n_head_kv;
    base.head_dim = c.head_dim;
    base.rms_eps = c.rms_eps;
    base.n_expert = c.n_expert;
    base.n_expert_used = c.n_used;
    base.n_ff_exp = c.n_ff;
    base.expert_weights_scale = c.w_scale;
    base.eos_token = c.eos;
    base.pad_token = c.pad;

    if (tp_n < 1 || c.n_head_kv % tp_n || c.n_ff % (64 * tp_n)) throw std::runtime_error("inkling: tensor parallel over this many GPUs");
    c.tp_n = tp_n;
    c.tp_rank = tp_rank;
    const int H_all = c.n_head, KVH_all = c.n_head_kv, F_all = c.n_ff;
    c.n_head /= tp_n;
    c.n_head_kv /= tp_n;
    c.n_ff /= tp_n;
    const int E = c.E, H = c.n_head, KV = c.kv_dim(), C = c.n_conv, D = c.head_dim, F = c.n_ff;
    const int64_t h0 = (int64_t) tp_rank * H, kv0 = (int64_t) tp_rank * KV, f0 = (int64_t) tp_rank * F;
    (void) H_all; (void) KVH_all;
    const bool tp = tp_n > 1;
    tok_embd = upload("token_embd.weight");
    tok_norm = upload_f32("token_embd_norm.weight", E);
    c.vocab_n = (c.n_vocab + tp_n - 1) / tp_n;
    c.vocab0 = tp_rank * c.vocab_n;
    c.vocab_n = std::min(c.vocab_n, c.n_vocab - c.vocab0);
    output = upload_rows("output.weight", {{c.vocab0, c.vocab0 + c.vocab_n}});
    output_norm = upload_f32("output_norm.weight", E);
    output_res_query = upload_f32("output_res_query.weight", E);
    if (tok_embd.type != GGML_TYPE_BF16 || output.type != GGML_TYPE_BF16 || tok_embd.rows != c.n_vocab_pad)
        throw std::runtime_error("inkling: expected BF16 token and output tables");

    layers.resize(c.n_layer);
    for (int il = 0; il < c.n_layer; ++il) {
        InkLayer & L = layers[il];
        L.swa = c.swa[il];
        // the attention: this rank's query heads (q, gate, rel, sink rows; out columns) and KV heads (k, v rows),
        // stacked by input
        {
            const int64_t QD = (int64_t) H * D, nr = (int64_t) H * c.n_rel, kp0 = kv0 / D * C, nkp = (int64_t) c.n_head_kv * C;
            std::vector<std::pair<std::string, Ranges>> parts = {
                {blk(il, "attn_q.weight"), {{h0 * D, h0 * D + QD}}},
                {blk(il, "attn_gate.weight"), {{h0 * D, h0 * D + QD}}},
                {blk(il, "attn_rel.weight"), {{h0 * c.n_rel, h0 * c.n_rel + nr}}},
                {blk(il, "attn_sink_proj.weight"), {{h0, h0 + H}}},
                {blk(il, "attn_k.weight"), {{kv0, kv0 + KV}}}};
            L.off_gate = (int) QD;
            L.off_rel = (int) (2 * QD);
            L.off_sink = (int) (2 * QD + nr);
            L.off_k = (int) (2 * QD + nr + H);
            int off = L.off_k + KV;
            if (!c.swa[il]) {
                parts.push_back({blk(il, "attn_k_conv_proj.weight"), {{kp0, kp0 + nkp}}});
                L.off_kp = off;
                off += (int) nkp;
            }
            parts.push_back({blk(il, "attn_v.weight"), {{kv0, kv0 + KV}}});
            L.off_v = off;
            off += KV;
            if (!c.swa[il]) {
                parts.push_back({blk(il, "attn_v_conv_proj.weight"), {{kp0, kp0 + nkp}}});
                L.off_vp = off;
                off += (int) nkp;
            }
            L.proj = upload_stack(parts);
            if (L.proj.rows != off) throw std::runtime_error("inkling: stacked projections");
            L.wq = qmat_rows(L.proj, 0, QD);
            L.wgate = qmat_rows(L.proj, L.off_gate, QD);
            L.wrel = qmat_rows(L.proj, L.off_rel, nr);
            L.wsink = qmat_rows(L.proj, L.off_sink, H);
            L.wk = qmat_rows(L.proj, L.off_k, KV);
            L.wv = qmat_rows(L.proj, L.off_v, KV);
            if (!c.swa[il]) {
                L.k_conv_proj = qmat_rows(L.proj, L.off_kp, nkp);
                L.v_conv_proj = qmat_rows(L.proj, L.off_vp, nkp);
            }
        }
        L.wo = tp ? upload_cols(blk(il, "attn_output.weight"), {{h0 * D, (h0 + H) * D}}) : upload(blk(il, "attn_output.weight"));
        L.q_clip = upload_f32_range(blk(il, "attn_q_clip.weight"), h0 * D, (int64_t) H * D);
        L.rel_clip = upload_f32_range(blk(il, "attn_rel_clip.weight"), h0 * c.n_rel, (int64_t) H * c.n_rel);
        L.rel_bias = upload_f32(blk(il, "attn_rel_bias.weight"), (size_t) c.extent(il) * c.n_rel);
        L.k_conv = upload_f32_range(blk(il, "attn_k_conv.weight"), kv0 * C, (int64_t) KV * C);
        L.v_conv = upload_f32_range(blk(il, "attn_v_conv.weight"), kv0 * C, (int64_t) KV * C);
        L.attn_post_conv = upload_f32(blk(il, "attn_post_conv.weight"), (size_t) E * C);
        L.ffn_post_conv = upload_f32(blk(il, "ffn_post_conv.weight"), (size_t) E * C);
        if (L.swa) {
            L.attn_norm = upload_f32(blk(il, "attn_norm.weight"), E);
        } else {
            L.q_norm = upload_f32(blk(il, "attn_q_inp_norm.weight"), E);
            L.k_norm = upload_f32(blk(il, "attn_k_inp_norm.weight"), E);
            L.v_norm = upload_f32(blk(il, "attn_v_inp_norm.weight"), E);
        }
        if (il > 0) {
            L.attn_res_nq = L.swa ? 1 : 3;
            L.attn_res_query = upload_f32(blk(il, "attn_res_query.weight"), (size_t) E * L.attn_res_nq);
        }
        L.ffn_res_query = upload_f32(blk(il, "ffn_res_query.weight"), E);
        L.ffn_norm = upload_f32(blk(il, "ffn_norm.weight"), E);
        L.router = upload(blk(il, "ffn_gate_inp.weight"));
        if (L.router.type != GGML_TYPE_F32 || L.router.rows != c.n_expert) throw std::runtime_error("inkling: router");
        L.exp_bias = upload_f32(blk(il, "exp_probs_b.bias"), c.n_expert);
        // the experts: this rank's share of every expert's FFN width (its gate rows, the same up rows; down columns)
        if (tp) {
            Ranges gu;
            for (int64_t e = 0; e < c.n_expert; ++e) {
                gu.push_back({e * 2 * F_all + f0, e * 2 * F_all + f0 + F});
                gu.push_back({e * 2 * F_all + F_all + f0, e * 2 * F_all + F_all + f0 + F});
            }
            L.gate_up = upload_rows(blk(il, "ffn_gate_up_exps.weight"), gu);
            L.down = upload_cols(blk(il, "ffn_down_exps.weight"), {{f0, f0 + F}});
        } else {
            L.gate_up = upload(blk(il, "ffn_gate_up_exps.weight"));
            L.down = upload(blk(il, "ffn_down_exps.weight"));
        }
        L.gu_scale.assign(c.n_expert, 1.f);
        L.down_scale.assign(c.n_expert, 1.f);
        for (auto [nm, dst] : {std::pair{"ffn_gate_up_exps.scale", &L.gu_scale}, {"ffn_down_exps.scale", &L.down_scale}}) {
            if (const TensorRef * t = gguf.find(blk(il, nm))) {
                if (t->type != GGML_TYPE_F32 || t->nbytes != (size_t) c.n_expert * 4) throw std::runtime_error("inkling: expert scales");
                memcpy(dst->data(), t->data, t->nbytes);
            } else if (L.gate_up.type == GGML_TYPE_NVFP4) {
                throw std::runtime_error("inkling: NVFP4 experts without their scales in layer " + std::to_string(il));
            }
        }
        if (L.gate_up.type == GGML_TYPE_NVFP4) L.d_gu_scale = upload_vec(L.gu_scale);
        if (L.down.type == GGML_TYPE_NVFP4) L.d_down_scale = upload_vec(L.down_scale);
    }
    loops.resize(c.n_core());
    static const char * ctl[4] = {"q", "k", "v", "ffn"};
    for (int il = c.loop_start; il < c.loop_end; ++il) {
        InkLoop & P = loops[il - c.loop_start];
        P.msg_query = upload_f32(blk(il, "loop_msg_query.weight"), (size_t) E * 4);
        for (int j = 0; j < 4; ++j) {
            const std::string p = std::string("loop_") + ctl[j] + "_ctrl_";
            P.down[j] = upload(blk(il, (p + "down.weight").c_str()));
            P.gf[j] = upload_stack({{blk(il, (p + "gain.weight").c_str()), {{0, E}}}, {blk(il, (p + "forget.weight").c_str()), {{0, E}}}});
            P.gain[j] = qmat_rows(P.gf[j], 0, E);
            P.forget[j] = qmat_rows(P.gf[j], E, E);
            if (P.down[j].cols != 2 * E || P.down[j].rows != c.gate_rank || P.gain[j].cols != c.gate_rank)
                throw std::runtime_error("inkling: loop control shapes");
        }
    }
    if (verbose)
        fprintf(stderr, "bnk: inkling: %d layers (core %d-%d x %d = %d visits), %d experts (%d used, %s), %.2f GiB in VRAM%s\n",
                c.n_layer, c.loop_start, c.loop_end - 1, c.loop_count, c.n_visits(), c.n_expert, c.n_used,
                ggml_type_name((ggml_type) layers[0].gate_up.type), vram_bytes / double(1 << 30),
                tp ? (" (rank " + std::to_string(tp_rank) + " of " + std::to_string(tp_n) + ": " + std::to_string(H) + " query heads, " +
                      std::to_string(c.n_head_kv) + " KV heads, FFN " + std::to_string(F) + ")").c_str() : "");
}

}  // namespace bnk
