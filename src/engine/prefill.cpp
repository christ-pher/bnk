// Batched prompt processing: fp16 tensor-core GEMMs for the dense projections, grouped GEMMs per expert, and the
// non-resident experts of the next layer streaming over PCIe while the current layer runs.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "engine/engine.h"
#include "ggml.h"
#include "kernels/ops.h"

namespace bnk {

static const float * F(const QMat & m) { return (const float *) m.data; }

// Every chunk-sized device buffer of the prompt path, carved from one elastic arena for chunks of N tokens: the
// arena maps only what N needs, so a long prompt can borrow VRAM from the expert cache for bigger chunks.
void Engine::prefill_layout(int N) {
    if (N == pf_max_) return;
    const size_t total = prefill_carve(N, false);
    CUDA_CHECK(cudaMemsetAsync(pf_arena_.ptr(), 0, total, st_));
    pf_.max_items = 2 * (N * model_.cfg.n_expert_used + model_.cfg.n_expert);
    pf_max_ = N;
}

size_t Engine::prefill_arena_bytes(int N) { return prefill_carve(N, true); }

// The arena's layout for chunks of N tokens: its size (dry), or map it and point every buffer into it.
size_t Engine::prefill_carve(int N, bool dry_only) {
    const Config & c = model_.cfg;
    const int E = c.n_embd, HC = c.hc_dim(), C = c.conv_channels(), VI = c.ssm_vheads * c.ssm_state;
    const int F = c.n_ff_exp, k = c.n_expert_used;
    auto & p = pf_;
    size_t off = 0;
    bool dry = true;
    auto carve = [&](auto & buf, size_t n) {
        using T = std::remove_reference_t<decltype(*buf.p)>;
        off = (off + 255) / 256 * 256;
        if (!dry) buf.view((T *) (pf_arena_.as<uint8_t>() + off), n);
        off += n * sizeof(T);
    };
    auto all = [&]() {
        carve(p.x, (size_t) N * E); carve(p.res, (size_t) N * HC); carve(p.xn, (size_t) N * HC);
        carve(p.lo, (size_t) N * c.hc_rank); carve(p.gpre, (size_t) N * HC); carve(p.mixed, (size_t) N * E);
        carve(p.inj, (size_t) N * c.hc); carve(p.out, (size_t) N * E);
        carve(p.conv, (size_t) (N + c.ssm_conv - 1) * C); carve(p.co, (size_t) N * C); carve(p.z, (size_t) N * VI);
        carve(p.g, (size_t) N * c.ssm_vheads); carve(p.b, (size_t) N * c.ssm_vheads); carve(p.o, (size_t) N * VI);
        carve(p.n, (size_t) N * VI);
        carve(p.qfull, (size_t) N * c.n_head * c.head_dim * 2); carve(p.k, (size_t) N * c.n_head_kv * c.head_dim);
        carve(p.v, (size_t) N * c.n_head_kv * c.head_dim); carve(p.q, (size_t) N * c.n_head * c.head_dim);
        carve(p.ao, (size_t) N * c.n_head * c.head_dim);
        carve(p.rlog, (size_t) N * c.n_expert); carve(p.w, (size_t) N * k); carve(p.ids, (size_t) N * k);
        carve(p.perm, (size_t) N * k); carve(p.inv, (size_t) N * k); carve(p.tok, N);
        carve(p.sg, (size_t) N * c.n_ff_shexp); carve(p.su, (size_t) N * c.n_ff_shexp); carve(p.sh, (size_t) N * c.n_ff_shexp);
        carve(p.sgate, N); carve(p.shared, (size_t) N * E); carve(p.moe, (size_t) N * E);
        carve(p.xg, (size_t) N * k * E); carve(p.gu, (size_t) N * k * 2 * F); carve(p.hh, (size_t) N * k * F);
        carve(p.dd, (size_t) (N * k + 1) * E);   // + a zero row for pairs computed elsewhere
        carve(p.cpu, (size_t) N * E);
        if (qsa_on_) { carve(p.ik, (size_t) N * c.idx_dim); carve(p.iq, (size_t) N * c.idx_heads * c.idx_dim); }
        carve(p.xq, (size_t) N * E); carve(p.xd, (size_t) N * E / 32);
        carve(p.hq, (size_t) N * k * F); carve(p.hd, (size_t) N * k * F / 32); carve(p.gu32, (size_t) N * k * 2 * F);
        carve(p.items, (size_t) (2 * (N * k + c.n_expert)) * sizeof(PfItem));
        if (c.ple_layer >= 0) {
            const int ph = c.ple_heads();
            carve(p.ple_emb, (size_t) N * ph * c.ple_dim); carve(p.ple_key, (size_t) N * HC);
            carve(p.ple_val, (size_t) N * E); carve(p.ple_gated, (size_t) N * HC);
            carve(p.ple_hist, (size_t) ((c.ple_conv - 1) * c.ple_ngram + N) * HC);
        }
    };
    all();
    const size_t total = off;
    if (dry_only) return total;
    if (total > pf_arena_.mapped()) pf_arena_.ensure(total);
    else pf_arena_.shrink_to(total);
    dry = false;
    off = 0;
    all();
    return total;
}

// Host-side and fixed parts of the prompt path, for chunks of up to N tokens.
void Engine::prefill_alloc(int N) {
    const Config & c = model_.cfg;
    const int E = c.n_embd, HC = c.hc_dim(), F = c.n_ff_exp, k = c.n_expert_used;
    auto & p = pf_;
    pf_cap_ = N;
    if (qsa_on_) { p.scores.alloc((size_t) 64 * max_blocks_); p.sel.alloc((size_t) 64 * qsh_.top_blocks); p.nsel.alloc(64); }
    p.max_items = 2 * (N * k + c.n_expert);
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_items, (size_t) p.max_items * sizeof(PfItem), 0));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_x, (size_t) N * E * 4, 0));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_cpu, (size_t) N * E * 4, 0));
    // fp16 copies of a group of experts' gate/up/down
    p.wexp_experts = 32;
    p.wexp.alloc((size_t) p.wexp_experts * 3 * F * E);
    if (c.ple_layer >= 0)
        CUDA_CHECK(cudaHostAlloc((void **) &p.ple_rows, (size_t) N * c.ple_heads() * model_.ple_table->row_bytes(),
                                 cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_ids, (size_t) N * k * 4, 0));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_perm, (size_t) N * k * 4, 0));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_inv, (size_t) N * k * 4, 0));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_w, (size_t) N * k * 4, 0));
    CUDA_CHECK(cudaStreamCreateWithFlags(&p.copy, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&p.xready, cudaEventDisableTiming | cudaEventBlockingSync));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&p.copied[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&p.used[i], cudaEventDisableTiming));
    }
    // GEMM workspace: the largest dense matrix except the head (the head runs as a GEMV), and N x max K inputs
    size_t maxw = 0;
    auto upd = [&](const QMat & m) { if (m.valid()) maxw = std::max(maxw, (size_t) m.rows * m.cols); };
    for (const auto & L : model_.layers) {
        for (const QMat * m : {&L.hc_attn.down, &L.hc_attn.up, &L.wq, &L.wk, &L.wv, &L.wo, &L.wqkv, &L.wgate,
                               &L.ssm_out, &L.ple_key, &L.ple_value, &L.router, &L.sh_gate, &L.sh_up, &L.sh_down})
            upd(*m);
    }
    gemm_.init(st_, maxw, (size_t) N * std::max(HC, c.n_head * c.head_dim * 2));
}

// Sized after the cache: the largest number of non-resident experts in any layer, twice (double buffer).
// Two stages for the next layers' non-resident experts, sized for the cache as it is now (a shrinking cache leaves
// more experts outside, so this may take a few rounds to settle).
void Engine::prefill_staging_ensure() {
    const Config & c = model_.cfg;
    for (int round = 0; round < 4; ++round) {
        size_t need = 0;
        for (int il = 0; il < c.n_layer; ++il)
            need = std::max(need, (size_t) (c.n_expert - cache_.resident(il)) * store_.blob_bytes(il));
        if (need <= pf_.stage_bytes && pf_stage_b_[0].mapped() >= need) break;
        for (int i = 0; i < 2; ++i) pf_stage_b_[i].ensure(need);
        pf_.stage_bytes = need;
    }
}

void Engine::prefill_staging_alloc() {
    const Config & c = model_.cfg;
    size_t worst = 0;
    for (int il = 0; il < c.n_layer; ++il) worst = std::max(worst, (size_t) c.n_expert * store_.blob_bytes(il));
    for (int i = 0; i < 2; ++i) {
        pf_stage_b_[i].reserve(worst, &budget_, "expert staging");
        pf_.stage[i] = pf_stage_b_[i].as<uint8_t>();
    }
    pf_.stage_bytes = 0;
    prefill_staging_ensure();
    pf_.stage_idx.assign(c.n_layer, std::vector<int>(c.n_expert, -1));
}

// Copies layer il's non-resident experts into stage[il & 1] on the copy stream: every one (prefetch), or the
// listed ones.
void Engine::pf_stage_layer(int il) {
    const Config & c = model_.cfg;
    auto & p = pf_;
    std::vector<int> & idx = p.stage_idx[il];
    std::fill(idx.begin(), idx.end(), -1);
    std::vector<int> list;
    for (int e = 0; e < c.n_expert; ++e)
        if (cache_.slot_of(il, e) < 0) list.push_back(e);
    pf_stage_list(il, list);
}

void Engine::pf_stage_list(int il, const std::vector<int> & experts) {
    auto & p = pf_;
    const int b = il & 1;
    CUDA_CHECK(cudaStreamWaitEvent(p.copy, p.used[b], 0));
    const size_t blob = store_.blob_bytes(il);
    std::vector<int> & idx = p.stage_idx[il];
    int n = 0;
    for (int e : experts) {
        idx[e] = n;
        CUDA_CHECK(cudaMemcpyAsync(p.stage[b] + (size_t) n * blob, store_.blob(il, e), blob, cudaMemcpyHostToDevice, p.copy));
        ++n;
    }
    CUDA_CHECK(cudaEventRecord(p.copied[b], p.copy));
}

void Engine::pf_hc_pre(const HcWeights & w, int N, bool inject) {
    const Config & c = model_.cfg;
    const int E = c.n_embd, HC = c.hc_dim();
    auto & p = pf_;
    hc_norm(p.res, F(w.norm), p.xn, N, c.hc, E, c.rms_eps, st_);
    f32_to_f16(p.xn, gemm_.xbuf(), (int64_t) N * HC, st_);
    gemm_.run_h(w.down, gemm_.xbuf(), HC, N, p.lo, c.hc_rank);
    if (inject) gemm_.run_h(w.inject, gemm_.xbuf(), HC, N, p.inj, c.hc);
    silu_scale(p.lo, (int64_t) N * c.hc_rank, 1.f / c.hc, st_);
    gemm_.run(w.up, p.lo, c.hc_rank, N, p.gpre, HC);
    hc_mix(p.xn, p.gpre, p.mixed, N, c.hc, E, st_);
}

void Engine::pf_ple(int il, int N) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, HC = c.hc_dim(), ph = c.ple_heads();
    auto & p = pf_;
    const TensorRef & tab = *model_.ple_table;
    const size_t rb = tab.row_bytes();
    // host: the n-gram rows of the chunk
    const double t0 = now_ms();
    const int pos0 = pos() - N;
    for (int t = 0; t < N; ++t) {
        const int ps = pos0 + t;
        int64_t ctx[8];
        ctx[0] = history_[ps];
        bool cut = false;
        for (int s = 1; s < c.ple_ngram; ++s) {
            const int q = ps - s;
            const int64_t tok = (cut || q < 0) ? -1 : history_[q];
            cut = cut || tok < 0 || tok == c.ple_eos;
            ctx[s] = cut ? c.ple_eos : tok;
        }
        for (int n = 2; n <= c.ple_ngram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * c.ple_mult[0];
            for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * c.ple_mult[j];
            for (int g = 0; g < c.ple_heads_per_ngram; ++g) {
                const int h = (n - 2) * c.ple_heads_per_ngram + g;
                const uint64_t row = mixed % c.ple_vocab[h] + c.ple_offset[h];
                memcpy(p.ple_rows + ((size_t) t * ph + h) * rb, tab.data + row * rb, rb);
            }
        }
    }
    times.ple_ms += now_ms() - t0;
    QMat rows{p.ple_rows, (int) tab.type, (int64_t) N * ph, tab.ne[0], rb};
    dequant_rows(rows, 0, (int64_t) N * ph, p.ple_emb, st_);
    const int D = ph * c.ple_dim;
    gemm_.run(L.ple_key, p.ple_emb, D, N, p.ple_key, HC);
    gemm_.run(L.ple_value, p.ple_emb, D, N, p.ple_val, E);
    const int hist = (c.ple_conv - 1) * c.ple_ngram;
    CUDA_CHECK(cudaMemcpyAsync(p.ple_hist.p, ple_hist_.p, (size_t) hist * HC * 4, cudaMemcpyDeviceToDevice, st_));
    ple_gate(p.ple_key, p.res, p.ple_val, F(L.ple_norm_key), F(L.ple_norm_query), F(L.ple_norm_conv), p.ple_gated,
             p.ple_hist.p + (size_t) hist * HC, N, c.hc, E, c.rms_eps, st_);
    ple_conv_add(p.res, p.ple_gated, p.ple_hist, (const half *) L.ple_conv1d.data, N, HC, c.ple_conv, c.ple_ngram, st_);
    CUDA_CHECK(cudaMemcpyAsync(ple_hist_.p, p.ple_hist.p + (size_t) N * HC, (size_t) hist * HC * 4,
                               cudaMemcpyDeviceToDevice, st_));
}

void Engine::pf_gdn(int il, int N) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, C = c.conv_channels(), K = c.ssm_conv, S = c.ssm_state, nv = c.ssm_vheads;
    auto & p = pf_;
    CUDA_CHECK(cudaMemcpyAsync(p.conv.p, conv_buf_[il].p, (size_t) (K - 1) * C * 4, cudaMemcpyDeviceToDevice, st_));
    f32_to_f16(p.mixed, gemm_.xbuf(), (int64_t) N * E, st_);
    gemm_.run_h(L.wqkv, gemm_.xbuf(), E, N, p.conv.p + (size_t) (K - 1) * C, C);
    gemm_.run_h(L.wgate, gemm_.xbuf(), E, N, p.z, nv * S);
    gemm_.run_h(L.ssm_beta, gemm_.xbuf(), E, N, p.b, nv);
    gemm_.run_h(L.ssm_alpha, gemm_.xbuf(), E, N, p.g, nv);
    gdn_conv(p.conv, F(L.ssm_conv1d), p.co, N, C, K, st_);
    gdn_prep(p.co, N, C, c.ssm_groups, nv, S, p.g, p.b, F(L.ssm_dt), F(L.ssm_a), c.rms_eps, st_);
    gdn_recurrence(p.co, C, p.g, p.b, ssm_state_[il], p.o, N, c.ssm_groups, nv, S, N, st_);
    gated_rmsnorm(p.o, p.z, F(L.ssm_norm), p.n, N, nv, S, c.rms_eps, st_);
    gemm_.run(L.ssm_out, p.n, nv * S, N, p.out, E);
    CUDA_CHECK(cudaMemcpyAsync(conv_buf_[il].p, p.conv.p + (size_t) N * C, (size_t) (K - 1) * C * 4,
                               cudaMemcpyDeviceToDevice, st_));
}

void Engine::pf_attn(int il, int N) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, H = c.n_head, Hkv = c.n_head_kv, D = c.head_dim;
    auto & p = pf_;
    const int pos0 = pos() - N;
    auto attention_prefill_rows = [&](int t0, int t1) {
        CUDA_CHECK(cudaMemsetAsync(p.nsel.p, 0, 64 * 4, st_));
        qsa_attention_prefill(p.q, kc_[il], vc_[il], p.qfull, p.sel, p.nsel, p.ao, t0, t1, qsh_, pos0,
                              1.f / sqrtf((float) D), st_);
    };
    f32_to_f16(p.mixed, gemm_.xbuf(), (int64_t) N * E, st_);
    gemm_.run_h(L.wq, gemm_.xbuf(), E, N, p.qfull, H * D * 2);
    gemm_.run_h(L.wk, gemm_.xbuf(), E, N, p.k, Hkv * D);
    gemm_.run_h(L.wv, gemm_.xbuf(), E, N, p.v, Hkv * D);
    attn_prep(p.qfull, p.k, p.v, F(L.q_norm), F(L.k_norm), p.q, kc_[il], vc_[il], N, H, Hkv, D, c.n_rot, c.rope_base,
              &d_par_->pos0, c.rms_eps, st_);
    if (kraw_[il].p) {
        gemm_.run_h(L.idx_k, gemm_.xbuf(), E, N, p.ik, c.idx_dim);
        gemm_.run_h(L.idx_q, gemm_.xbuf(), E, N, p.iq, c.idx_heads * c.idx_dim);
        qsa_store_keys(p.ik, kraw_[il], N, c.idx_dim, &d_par_->pos0, st_);
        qsa_queries(p.iq, F(L.idx_q_norm), N, qsh_, &d_par_->pos0, st_);
        qsa_pool(kraw_[il], F(L.idx_k_norm), pooled_[il], N, qsh_, &d_par_->pos0, st_);
        // rows past the dense limit select blocks; 64 rows at a time bound the score scratch
        for (int t0 = 0; t0 < N; t0 += 64) {
            const int t1 = std::min(N, t0 + 64);
            if (pos0 + t1 <= qsh_.dense_cells()) {
                attention_prefill_rows(t0, t1);
                continue;
            }
            qsa_select(p.iq.p + (size_t) t0 * c.idx_heads * c.idx_dim, pooled_[il], p.scores, max_blocks_, p.sel,
                       p.nsel, t1 - t0, qsh_, &d_par_->pos0, t0, st_);
            qsa_attention_prefill(p.q, kc_[il], vc_[il], p.qfull, p.sel, p.nsel, p.ao, t0, t1, qsh_, pos0,
                                  1.f / sqrtf((float) D), st_);
            static const char * dsel = getenv("BNK_DUMP_SEL");
            if (dsel && il == env_int("BNK_DUMP_SEL_LAYER", 3)) {
                std::vector<int32_t> h((size_t) (t1 - t0) * qsh_.top_blocks), hn(t1 - t0);
                CUDA_CHECK(cudaStreamSynchronize(st_));
                CUDA_CHECK(cudaMemcpy(h.data(), p.sel.p, h.size() * 4, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hn.data(), p.nsel.p, hn.size() * 4, cudaMemcpyDeviceToHost));
                FILE * f = fopen(dsel, "ab");
                for (int t = 0; t < t1 - t0; ++t) {
                    const int32_t row = pos0 + t0 + t;
                    fwrite(&row, 4, 1, f);
                    fwrite(&hn[t], 4, 1, f);
                    fwrite(h.data() + (size_t) t * qsh_.top_blocks, 4, qsh_.top_blocks, f);
                }
                fclose(f);
            }
        }
    } else {
        attention_prefill(p.q, kc_[il], vc_[il], p.qfull, p.ao, N, H, Hkv, D, pos0, 1.f / sqrtf((float) D), st_);
    }
    gemm_.run(L.wo, p.ao, H * D, N, p.out, E);
}

void Engine::pf_moe(int il, int N) {
    const Config & c = model_.cfg;
    const LayerWeights & L = model_.layers[il];
    const int E = c.n_embd, k = c.n_expert_used, F = c.n_ff_exp, FF = c.n_ff_shexp;
    auto & p = pf_;
    // routing
    f32_to_f16(p.mixed, gemm_.xbuf(), (int64_t) N * E, st_);
    gemm_.run_h(L.router, gemm_.xbuf(), E, N, p.rlog, c.n_expert);
    route_topk(p.rlog, N, c.n_expert, k, p.ids, p.w, c.expert_weights_scale, st_);
    // the shared expert meanwhile
    gemm_.run_h(L.sh_gate, gemm_.xbuf(), E, N, p.sg, FF);
    gemm_.run_h(L.sh_up, gemm_.xbuf(), E, N, p.su, FF);
    gemm_.run_h(L.shexp_gate_inp, gemm_.xbuf(), E, N, p.sgate, 1);
    silu_mul(p.sg, p.su, p.sh, (int64_t) N * FF, st_);
    gemm_.run(L.sh_down, p.sh, FF, N, p.shared, E);
    sigmoid_inplace(p.sgate, N, st_);
    CUDA_CHECK(cudaMemcpyAsync(p.h_ids, p.ids.p, (size_t) N * k * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));

    // group the (token, slot) pairs by expert
    const int P = N * k;
    std::vector<int> cnt(c.n_expert, 0), off(c.n_expert + 1, 0);
    for (int i = 0; i < P; ++i) cnt[p.h_ids[i]]++;
    for (int e = 0; e < c.n_expert; ++e) off[e + 1] = off[e] + cnt[e];
    for (int e = 0; e < c.n_expert; ++e) times.routed += cnt[e];
    for (int e = 0; e < c.n_expert; ++e) layer_routed[il] += cnt[e];

    // where each used expert runs: VRAM cache, PCIe stage, or the CPU pool
    const size_t blob = store_.blob_bytes(il);
    std::vector<char> on_cpu(c.n_expert, 0);
    if (!p.prefetch) {
        std::fill(p.stage_idx[il].begin(), p.stage_idx[il].end(), -1);
        std::vector<int> nr;
        for (int e = 0; e < c.n_expert; ++e)
            if (cnt[e] && cache_.slot_of(il, e) < 0) nr.push_back(e);
        std::sort(nr.begin(), nr.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
        // the most-used go over PCIe while that is cheaper than leaving them to the CPU
        const double t_pcie = opt_.pcie_us_per_mb * blob / 1e6;
        const double t_tok = opt_.cpu_us_per_expert_token * blob / (2.2e6);
        double cpu_total = 0;
        for (int e : nr) cpu_total += cnt[e] * t_tok;
        size_t n_pcie = 0;
        double pcie_total = 0;
        while (n_pcie < nr.size() && n_pcie * blob < p.stage_bytes) {
            const double cpu_after = cpu_total - cnt[nr[n_pcie]] * t_tok;
            if (std::max(pcie_total + t_pcie, cpu_after) >= std::max(pcie_total, cpu_total)) break;
            pcie_total += t_pcie;
            cpu_total = cpu_after;
            ++n_pcie;
        }
        std::vector<int> staged(nr.begin(), nr.begin() + n_pcie);
        for (size_t i = n_pcie; i < nr.size(); ++i) on_cpu[nr[i]] = 1;
        pf_stage_list(il, staged);
    }
    std::vector<int> fill(off.begin(), off.end() - 1);
    std::vector<ExpertTask> cpu_tasks;
    for (int i = 0; i < P; ++i) {
        const int e = p.h_ids[i];
        const int pos = fill[e]++;
        p.h_perm[pos] = i / k;
        p.h_inv[i] = on_cpu[e] ? P : pos;       // CPU pairs read the zero row
        if (on_cpu[e]) cpu_tasks.push_back({i / k, i % k, e, 0.f});
    }
    CUDA_CHECK(cudaMemcpyAsync(p.perm.p, p.h_perm, (size_t) P * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(p.inv.p, p.h_inv, (size_t) P * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemsetAsync(p.dd.p + (size_t) P * E, 0, (size_t) E * 2, st_));
    gather_rows_f16(p.mixed, E, p.perm, P, p.xg, st_);
    if (!cpu_tasks.empty()) {
        CUDA_CHECK(cudaMemcpyAsync(p.h_x, p.mixed.p, (size_t) N * E * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaMemcpyAsync(p.h_w, p.w.p, (size_t) P * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaEventRecord(p.xready, st_));
    }

    // the experts on the GPU: fp16 copies of a group, then two grouped GEMMs (resident ones first).
    // Resident experts are R-layout cache slots; staged ones the host store's ggml blobs.
    const MoeLayerDesc d = cache_.desc(il);
    MoeLayerDesc ds = d;   // the stage's view
    {
        const LayerWeights & Lw = model_.layers[il];
        ds.rlay = 0;
        ds.blob = blob;
        ds.gate_bytes = Lw.gate_bytes;
        ds.up_bytes = Lw.up_bytes;
        ds.grow = ggml_row_size((ggml_type) Lw.gate_type, E);
        ds.drow = ggml_row_size((ggml_type) Lw.down_type, F);
    }
    const size_t gu_elems = (size_t) 2 * F * E, dn_elems = (size_t) E * F;
    std::vector<int> group;
    auto flush = [&]() {
        if (group.empty()) return;
        std::vector<const half *> wg, xg, wd, hx;
        std::vector<half *> yg, yd;
        std::vector<int> n;
        for (size_t i = 0; i < group.size(); ++i) {
            const int e = group[i];
            half * wgu = p.wexp.p + i * (gu_elems + dn_elems);
            half * wdn = wgu + gu_elems;
            const int sl = cache_.slot_of(il, e);
            const MoeLayerDesc & dv = sl >= 0 ? d : ds;
            const uint8_t * src = sl >= 0 ? d.base + (size_t) sl * d.blob : p.stage[il & 1] + (size_t) p.stage_idx[il][e] * blob;
            QMat mg{src, dv.gate_type, 2 * F, E, dv.grow};
            QMat md{src + dv.gate_bytes + dv.up_bytes, dv.down_type, E, F, dv.drow};
            if (dv.rlay) {
                mg.layout = md.layout = 1;
                for (int q = 0; q < 5; ++q) { mg.r_off[q] = dv.go[q]; md.r_off[q] = dv.dof[q]; }
            }
            dequant_rows_f16(mg, 0, 2 * F, wgu, st_);
            dequant_rows_f16(md, 0, E, wdn, st_);
            wg.push_back(wgu);
            wd.push_back(wdn);
            xg.push_back(p.xg.p + (size_t) off[e] * E);
            yg.push_back(p.gu.p + (size_t) off[e] * 2 * F);
            hx.push_back(p.hh.p + (size_t) off[e] * F);
            yd.push_back(p.dd.p + (size_t) off[e] * E);
            n.push_back(cnt[e]);
        }
        gemm_.grouped_h(wg, xg, yg, n, 2 * F, E);
        for (size_t i = 0; i < group.size(); ++i)
            swiglu_f16(yg[i], (half *) hx[i], n[i], F, st_);
        gemm_.grouped_h(wd, hx, yd, n, E, F);
        group.clear();
    };
    auto src_of = [&](int e) {
        const int sl = cache_.slot_of(il, e);
        return sl >= 0 ? d.base + (size_t) sl * d.blob : p.stage[il & 1] + (size_t) p.stage_idx[il][e] * blob;
    };
    for (int pass = 0; pass < 2; ++pass) {   // 0: resident, 1: staged (after their copy)
        if (pass == 1) CUDA_CHECK(cudaStreamWaitEvent(st_, p.copied[il & 1], 0));
        for (int e = 0; e < c.n_expert; ++e) {
            if (!cnt[e] || on_cpu[e] || cnt[e] < opt_.gemm_min_tokens) continue;
            if ((cache_.slot_of(il, e) >= 0) != (pass == 0)) continue;
            group.push_back(e);
            if ((int) group.size() == p.wexp_experts) flush();
        }
        flush();
    }
    // experts with few tokens: dp4a straight on their quantized weights, in items of <= 8 tokens; resident
    // (R slots) and staged (ggml) items go in two launches
    int n_items[2] = {0, 0};
    const int half_cap = p.max_items / 2;
    for (int e = 0; e < c.n_expert; ++e) {
        if (!cnt[e] || on_cpu[e] || cnt[e] >= opt_.gemm_min_tokens) continue;
        const int kind = cache_.slot_of(il, e) >= 0 ? 0 : 1;
        for (int j0 = 0; j0 < cnt[e]; j0 += kMaxWindow) {
            PfItem & it = p.h_items[kind * half_cap + n_items[kind]++];
            it.blob = src_of(e);
            it.n = std::min(kMaxWindow, cnt[e] - j0);
            for (int j = 0; j < it.n; ++j) {
                it.pair[j] = off[e] + j0 + j;
                it.tok[j] = p.h_perm[off[e] + j0 + j];
            }
        }
    }
    if (n_items[0] || n_items[1]) {
        ActQ8 xq;
        xq.q = p.xq; xq.d = p.xd;
        quantize_act(p.mixed, E, N, E, xq, st_);
        CUDA_CHECK(cudaMemcpyAsync(p.items.p, p.h_items, (size_t) p.max_items * sizeof(PfItem), cudaMemcpyHostToDevice, st_));
        const PfItem * dev = (const PfItem *) p.items.p;
        moe_list(dev, n_items[0], d, xq, E, F, p.gu32, p.hq, p.hd, p.dd, st_);
        moe_list(dev + half_cap, n_items[1], ds, xq, E, F, p.gu32, p.hq, p.hd, p.dd, st_);
    }
    CUDA_CHECK(cudaEventRecord(p.used[il & 1], st_));
    // the CPU part runs here while the GPU works through the queue
    const float * extra = nullptr;
    if (!cpu_tasks.empty()) {
        CUDA_CHECK(cudaEventSynchronize(p.xready));   // h_x / h_w are here; the expert GEMMs keep running
        for (auto & t : cpu_tasks) t.w = p.h_w[t.t * k + t.slot];
        const double tc = now_ms();
        cpu_.run(il, N, p.h_x, cpu_tasks, p.h_cpu);
        times.cpu_experts_ms += now_ms() - tc;
        times.misses += (int64_t) cpu_tasks.size();
        layer_misses[il] += (int64_t) cpu_tasks.size();
        CUDA_CHECK(cudaMemcpyAsync(p.cpu.p, p.h_cpu, (size_t) N * E * 4, cudaMemcpyHostToDevice, st_));
        extra = p.cpu.p;
    }
    // per token: shared * gate + the CPU part + its GPU experts (in routing order)
    moe_gather_sum(p.dd, p.inv, p.w, N, k, E, p.shared, p.sgate, extra, p.moe, st_);
}

void Engine::prefill_chunk(const int32_t * tokens, int N) {
    if (N < 1 || N > pf_max_) throw std::runtime_error("prefill_chunk: bad size");
    if (pending_T_) throw std::runtime_error("prefill_chunk: a verify window is pending");
    if (pos() + N > opt_.max_ctx) throw std::runtime_error("context full");
    const Config & c = model_.cfg;
    const int E = c.n_embd, HC = c.hc_dim();
    const double t0 = now_ms();
    auto & p = pf_;
    const int pos0 = pos();
    history_.insert(history_.end(), tokens, tokens + N);
    *h_par_ = WinParams{pos0, N, ++seq_, 0};
    CUDA_CHECK(cudaMemcpyAsync(d_par_, h_par_, sizeof(WinParams), cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(p.tok.p, tokens, (size_t) N * 4, cudaMemcpyHostToDevice, st_));
    dequant_gather(model_.tok_embd, p.tok, N, p.x, st_);
    hc_init(p.x, p.res, N, c.hc, E, st_);
    p.prefetch = N >= opt_.prefetch_min;
    if (p.prefetch) pf_stage_layer(0);
    for (int il = 0; il < c.n_layer; ++il) {
        const LayerWeights & L = model_.layers[il];
        if (p.prefetch && il + 1 < c.n_layer) pf_stage_layer(il + 1);
        if (il == c.ple_layer) pf_ple(il, N);
        pf_hc_pre(L.hc_attn, N, true);
        if (L.attn) pf_attn(il, N); else pf_gdn(il, N);
        hc_combine(p.res, p.out, p.inj, N, c.hc, E, st_);
        pf_hc_pre(L.hc_ffn, N, true);
        pf_moe(il, N);
        hc_combine(p.res, p.moe, p.inj, N, c.hc, E, st_);
    }
    // the head on the last token, through the decode buffers (row 0)
    CUDA_CHECK(cudaMemcpyAsync(res_.p, p.res.p + (size_t) (N - 1) * HC, (size_t) HC * 4, cudaMemcpyDeviceToDevice, st_));
    head(1);
    CUDA_CHECK(cudaStreamSynchronize(st_));
    last_T = 1;
    pstats.ms += now_ms() - t0;
    pstats.tokens += N;
    pstats.chunks++;
}

}  // namespace bnk
