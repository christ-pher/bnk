// Batched prompt processing for the DeepSeek engine: a chunk of N tokens through fp16 tensor-core GEMMs (weights
// dequantized on the fly) for the dense projections, the engine's own kernels over the whole chunk for the
// hyper-connections, attention, compressor, indexer and Engram, and the routed experts on the tensor cores:
// resident experts from the VRAM cache, the most-used non-resident ones streamed from the host store into a staging
// arena, the rest on the CPU pool (the Qwen engine's cost model, docs/DEEPSEEK.md).
#include <algorithm>
#include <cmath>
#include <cstring>

#include <cublas_v2.h>

#include "deepseek/ds_engine.h"
#include "kernels/gemm.h"
#include "kernels/ops.h"

namespace bnk {

static const float * F(const QMat & m) { return (const float *) m.data; }

void DsEngine::prefill_alloc(int N) {
    const DsConfig & c = model_.ds;
    const int E = c.n_embd, HC = c.hc_dim(), H = c.n_head, D = c.head_dim, QL = c.q_lora, G = c.o_groups, OL = c.o_lora;
    const int IH = c.idx_heads, ID = c.idx_dim, Fx = c.n_ff, k = c.n_used, NE = c.n_expert, P = N * k;
    auto & p = pf_;
    p.res.alloc((size_t) N * HC); p.mixed.alloc((size_t) N * E); p.xn.alloc((size_t) N * E); p.out.alloc((size_t) N * E);
    p.pre_a.alloc((size_t) N * c.hc); p.pre_b.alloc((size_t) N * c.hc); p.attn_pre.alloc((size_t) N * c.hc);
    p.post.alloc((size_t) N * c.hc); p.comb.alloc((size_t) N * c.hc * c.hc);
    p.qr.alloc((size_t) N * QL); p.qrn.alloc((size_t) N * QL); p.q.alloc((size_t) N * H * D); p.kvr.alloc((size_t) N * D);
    p.attn_o.alloc((size_t) N * H * D); p.oa.alloc((size_t) N * G * OL); p.ckv.alloc((size_t) N * D); p.sc.alloc((size_t) N * D);
    p.kwin.alloc((size_t) (N + c.window) * D);
    p.iq.alloc((size_t) N * IH * ID); p.iw.alloc((size_t) N * IH);
    // the indexer's scores for a piece of the chunk at a time: at most ~128 MiB
    p.score_rows = std::max(1, std::min(N, (int) ((128u << 20) / (lds_ * 4))));
    p.scores.alloc((size_t) p.score_rows * lds_);
    if (n_blocks_max_) { p.cand_scratch.alloc((size_t) p.score_rows * n_blocks_max_); p.cand.alloc((size_t) N * n_blocks_max_); }
    p.idx.alloc((size_t) N * c.idx_topk);
    p.pos.alloc((size_t) (N + p.score_rows - 1) / p.score_rows + 1);
    p.tok.alloc(N); p.ids.alloc(P); p.perm.alloc(P); p.inv.alloc(P);
    int n_eg = 0;
    for (int il : c.engram_layers) n_eg += owns(il);
    if (n_eg) {
        const int64_t cols = (int64_t) c.eg_cols() * c.eg_dim;
        p.eg_emb.alloc((size_t) N * cols);
        p.eg_kv.alloc((size_t) N * (c.hc + 1) * E);
        p.eg_rows_per_layer = (size_t) N * c.eg_cols() * model_.engram_tab[0]->row_bytes();
        CUDA_CHECK(cudaHostAlloc((void **) &p.eg_rows, p.eg_rows_per_layer * c.engram_layers.size(), cudaHostAllocMapped));
    }
    p.rlog.alloc((size_t) N * NE); p.w.alloc(P);
    p.sg.alloc((size_t) N * 2 * Fx); p.sh.alloc((size_t) N * Fx); p.shared.alloc((size_t) N * E); p.moe.alloc((size_t) N * E);
    p.cpu.alloc((size_t) N * E);
    p.xg.alloc((size_t) N * E); p.hh.alloc((size_t) P * Fx); p.dd.alloc((size_t) (P + 1) * E);
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_ids, (size_t) P * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_perm, (size_t) P * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_inv, (size_t) P * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_w, (size_t) P * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_x, (size_t) N * E * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_cpu, (size_t) N * E * 4, cudaHostAllocDefault));
    const int max_blocks = P / kTcTile + NE + 1;
    p.items_bytes = (size_t) 2 * NE * sizeof(TcItem) + (size_t) 2 * max_blocks * sizeof(TcBlock);
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_items, p.items_bytes, cudaHostAllocDefault));
    p.items.alloc(p.items_bytes);
    p.stage_idx.assign(NE, -1);
    CUDA_CHECK(cudaStreamCreateWithFlags(&p.copy, cudaStreamNonBlocking));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&p.copied[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&p.used[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&p.xready, cudaEventDisableTiming));
    // GEMM workspace: the largest matrix dequantized at once, and N x the widest input
    size_t maxw = 0;
    for (int il = l0_; il < l1_; ++il) {
        const DsLayer & L = model_.layers[il];
        for (const QMat * m : {&L.q_a, &L.q_b, &L.kv, &L.o_b, &L.sh_gu, &L.sh_gate, &L.sh_up, &L.sh_down, &L.eg_wkv, &L.comp_kv,
                               &L.comp_gate, &L.idx_q_b, &L.idx_proj})
            if (m->valid()) maxw = std::max(maxw, (size_t) m->rows * m->cols);
        maxw = std::max(maxw, (size_t) OL * (H * D / G));
    }
    const int64_t maxk = std::max<int64_t>({(int64_t) E, (int64_t) QL, (int64_t) H * D / G, (int64_t) G * OL, (int64_t) Fx,
                                            (int64_t) c.eg_cols() * c.eg_dim});
    gemm_.init(st_, maxw, (size_t) N * maxk);
}

// the chunk's Engram table rows for every Engram layer this stage owns (host, before the chunk runs)
void DsEngine::pf_engram_gather(int N) {
    const DsConfig & c = model_.ds;
    auto & p = pf_;
    if (!p.eg_rows) return;
    const double t0 = now_ms();
    const int cols = c.eg_cols(), pos0 = pos() - N;
    uint64_t rows[64];
    for (size_t e = 0; e < c.engram_layers.size(); ++e) {
        if (!owns(c.engram_layers[e])) continue;
        const TensorRef & tab = *model_.engram_tab[e];
        const size_t rb = tab.row_bytes();
        uint8_t * dst = p.eg_rows + e * p.eg_rows_per_layer;
        for (int t = 0; t < N; ++t) {
            engram_hash((int) e, cur_->history, pos0 + t, rows);
            for (int col = 0; col < cols; ++col) memcpy(dst + ((size_t) t * cols + col) * rb, tab.data + rows[col] * rb, rb);
        }
    }
    std::lock_guard<std::mutex> lk(head_->stats_mu_);
    head_->times.ple_ms += now_ms() - t0;
}

void DsEngine::pf_engram(int il, int N) {
    const DsConfig & c = model_.ds;
    const DsLayer & L = model_.layers[il];
    const int e = c.engram_index(il), E = c.n_embd;
    auto & p = pf_;
    const TensorRef & tab = *model_.engram_tab[e];
    const int64_t rows = (int64_t) N * c.eg_cols(), D = (int64_t) c.eg_cols() * c.eg_dim;
    QMat rm{p.eg_rows + e * p.eg_rows_per_layer, (int) tab.type, rows, tab.ne[0], tab.row_bytes()};
    dequant_rows(rm, 0, rows, p.eg_emb, st_);
    gemm_.run(L.eg_wkv, p.eg_emb, D, N, p.eg_kv, (int64_t) (c.hc + 1) * E);
    ds_engram_gate(p.res, p.eg_kv, (int64_t) (c.hc + 1) * E, p.eg_kv.p + (size_t) c.hc * E, (int64_t) (c.hc + 1) * E, F(L.eg_gate), N,
                   c.hc, E, c.eps, st_);
}

void DsEngine::pf_attn(int il, int N) {
    const DsConfig & c = model_.ds;
    const DsLayer & L = model_.layers[il];
    const int E = c.n_embd, H = c.n_head, D = c.head_dim, QL = c.q_lora, R = c.n_rot, IH = c.idx_heads, ID = c.idx_dim;
    const int ratio = c.ratio[il];
    const bool src = c.is_kv_src(il), isrc = c.is_idx_src(il);
    auto & p = pf_;
    const int pos0 = pos() - N, origin = std::max(0, pos0 - c.window);
    const int * d_pos0 = &d_par_->pos0;
    // every projection of the layer input first (Gemm::run converts its own input into the same fp16 workspace)
    f32_to_f16(p.xn, gemm_.xbuf(), (int64_t) N * E, st_);
    gemm_.run_h(L.q_a, gemm_.xbuf(), E, N, p.qr, QL);
    gemm_.run_h(L.kv, gemm_.xbuf(), E, N, p.kvr, D);
    if (src) {
        gemm_.run_h(L.comp_kv, gemm_.xbuf(), E, N, p.ckv, D);
        if (ratio > 1) gemm_.run_h(L.comp_gate, gemm_.xbuf(), E, N, p.sc, D);
    }
    if (isrc) gemm_.run_h(L.idx_proj, gemm_.xbuf(), E, N, p.iw, IH);
    // then those of the normalized Q latent
    rmsnorm_rows(p.qr, F(L.q_a_norm), p.qrn, N, QL, QL, QL, c.eps, st_);
    gemm_.run(L.q_b, p.qrn, QL, N, p.q, (int64_t) H * D);
    if (isrc) gemm_.run(L.idx_q_b, p.qrn, QL, N, p.iq, (int64_t) IH * ID);
    ds_rope_heads(p.q, N, H, D, R, d_pos0, rope_of(il), st_);
    // the window keys of the chunk in a linear buffer (the ring holds 136 slots): the positions before the chunk
    // from the ring first, then the chunk's own rows
    if (pos0 > origin) {
        // ring slot -> linear row, one small copy per position (at most 128)
        for (int q = origin; q < pos0; ++q)
            CUDA_CHECK(cudaMemcpyAsync(p.kwin.p + (size_t) (q - origin) * D, cur_->ring[il].p + (size_t) (q % kDsRing) * D, (size_t) D * 2,
                                       cudaMemcpyDeviceToDevice, st_));
    }
    ds_kv_prep(p.kvr, F(L.kv_norm), p.kwin, N, D, R, d_pos0, rope_of(il), c.eps, st_, origin);
    if (src) {
        ds_compress(p.ckv, ratio > 1 ? p.sc.p : nullptr, cur_->st_kv[il], cur_->st_sc[il], F(L.comp_norm), (const half *) L.idx_k.data,
                    F(L.idx_k_norm), cur_->ckc[il], cur_->ikc[il], N, ratio, D, ID, R, d_pos0, rope_comp_, c.eps, st_);
        ds_compress_commit(p.ckv, ratio > 1 ? p.sc.p : nullptr, cur_->st_kv[il], cur_->st_sc[il], N, ratio, D, d_pos0, st_);
    }
    const float scale = 1.f / sqrtf((float) D);
    if (ratio) {
        const int ks = c.kv_src[il];
        if (isrc) {
            ds_rope_heads(p.iq, N, IH, ID, R, d_pos0, rope_comp_, st_);
            const int n_max = (int) std::min<int64_t>(lds_, ctx_grid_ / ratio + 1);
            for (int r0 = 0, j = 0; r0 < N; r0 += p.score_rows, ++j) {
                const int n = std::min(p.score_rows, N - r0);
                const int * pj = p.pos.p + j;
                ds_index_scores(p.iq.p + (size_t) r0 * IH * ID, cur_->ikc[ks], p.iw.p + (size_t) r0 * IH, p.scores, n, IH, ID, ratio, n_max,
                                lds_, pj, st_);
                if (il == c.cand_src)
                    ds_cand_blocks(p.scores, lds_, n, ratio, c.cand_bs, c.cand_blocks, (n_max + c.cand_bs - 1) / c.cand_bs, pj,
                                   p.cand_scratch, p.cand.p + (size_t) r0 * n_blocks_max_, st_);
                const uint8_t * cand = c.cand_src >= 0 && il > c.cand_src ? p.cand.p + (size_t) r0 * n_blocks_max_ : nullptr;
                ds_topk_select(p.scores, lds_, n, ratio, c.idx_topk, pj, cand, c.cand_bs, p.idx.p + (size_t) r0 * c.idx_topk, st_);
            }
        }
        ds_attention(p.q, p.kwin, cur_->ckc[ks], p.idx, c.idx_topk, F(L.sinks), p.attn_o, N, H, D, c.window, R, d_pos0, rope_of(il),
                     scale, st_, origin);
    } else {
        ds_attention(p.q, p.kwin, nullptr, nullptr, 0, F(L.sinks), p.attn_o, N, H, D, c.window, R, d_pos0, rope_of(il), scale, st_,
                     origin);
    }
    // the ring takes the chunk's last positions for the decode windows that follow
    ds_ring_fill(p.kwin, origin, cur_->ring[il], std::max(pos0, pos0 + N - kDsRing), pos0 + N, D, st_);
    const int G = c.o_groups, GW = H * D / G, OL = c.o_lora;
    for (int g = 0; g < G; ++g)
        gemm_.run(qmat_rows(L.o_a, (int64_t) g * OL, OL), p.attn_o.p + (size_t) g * GW, (int64_t) H * D, N, p.oa.p + (size_t) g * OL,
                  (int64_t) G * OL);
    gemm_.run(L.o_b, p.oa, (int64_t) G * OL, N, p.out, E);
}

void DsEngine::pf_moe(int il, int N) {
    const DsConfig & c = model_.ds;
    const DsLayer & L = model_.layers[il];
    const int E = c.n_embd, k = c.n_used, Fx = c.n_ff, NE = c.n_expert, P = N * k;
    auto & p = pf_;
    // routing in fp32 (near ties pick the experts)
    {
        const float one = 1.f, zero = 0.f;
        if (cublasSgemm(gemm_.handle(), CUBLAS_OP_T, CUBLAS_OP_N, NE, N, E, &one, F(L.router), E, p.xn, E, &zero, p.rlog, NE) !=
            CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("prefill: router GEMM failed");
    }
    ds_route(p.rlog, F(L.probs_b), N, NE, k, c.route_scale, c.route_norm, p.ids, p.w, st_);
    // the shared expert meanwhile
    f32_to_f16(p.xn, gemm_.xbuf(), (int64_t) N * E, st_);
    if (L.sh_gu.valid()) gemm_.run_h(L.sh_gu, gemm_.xbuf(), E, N, p.sg, 2 * Fx);
    else {
        gemm_.run_h(L.sh_gate, gemm_.xbuf(), E, N, p.sg, 2 * Fx);
        gemm_.run_h(L.sh_up, gemm_.xbuf(), E, N, p.sg.p + Fx, 2 * Fx);
    }
    ds_silu_mul_clamp(p.sg, p.sh, N, Fx, c.swiglu_limit, st_);
    gemm_.run(L.sh_down, p.sh, Fx, N, p.shared, E);
    CUDA_CHECK(cudaMemcpyAsync(p.h_ids, p.ids.p, (size_t) P * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));

    // the (token, slot) pairs grouped by expert
    std::vector<int> cnt(NE, 0), off(NE + 1, 0);
    for (int i = 0; i < P; ++i) cnt[p.h_ids[i]]++;
    for (int e = 0; e < NE; ++e) off[e + 1] = off[e] + cnt[e];
    {
        std::lock_guard<std::mutex> lk(head_->stats_mu_);
        head_->times.routed += P;
        head_->layer_routed[il] += P;
    }
    // where each used non-resident expert runs: the PCIe stage (most-used first, while that beats the CPU) or the CPU
    const size_t blob = store_->blob_bytes(il);
    std::vector<char> on_cpu(NE, 0);
    std::fill(p.stage_idx.begin(), p.stage_idx.end(), -1);
    {
        std::vector<int> nr;
        for (int e = 0; e < NE; ++e)
            if (cnt[e] && cache_.slot_of(il, e) < 0) nr.push_back(e);
        std::sort(nr.begin(), nr.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
        static const double pcie_us = env_double("BNK_PCIE_US_PER_MB", opt_.pcie_us_per_mb);
        static const double cpu_us = env_double("BNK_CPU_US_PER_EXPERT_TOKEN", opt_.cpu_us_per_expert_token);
        const double t_pcie = pcie_us * blob / 1e6;
        const double t_tok = cpu_us * blob / 2.2e6;   // (the constant was measured on 2.2 MB experts)
        double cpu_total = 0;
        for (int e : nr) cpu_total += cnt[e] * t_tok;
        size_t n_pcie = 0;
        double pcie_total = 0;
        while (n_pcie < nr.size() && (n_pcie + 1) * blob <= p.stage_bytes) {
            const double cpu_after = cpu_total - cnt[nr[n_pcie]] * t_tok;
            if (std::max(pcie_total + t_pcie, cpu_after) >= std::max(pcie_total, cpu_total)) break;
            pcie_total += t_pcie;
            cpu_total = cpu_after;
            ++n_pcie;
        }
        for (size_t i = n_pcie; i < nr.size(); ++i) on_cpu[nr[i]] = 1;
        const int b = il & 1;
        CUDA_CHECK(cudaStreamWaitEvent(p.copy, p.used[b], 0));
        for (size_t i = 0; i < n_pcie; ++i) {
            const int e = nr[i];
            p.stage_idx[e] = (int) i;
            CUDA_CHECK(cudaMemcpyAsync(p.stage[b] + i * blob, store_->blob(il, e), blob, cudaMemcpyHostToDevice, p.copy));
        }
        CUDA_CHECK(cudaEventRecord(p.copied[b], p.copy));
    }
    std::vector<int> fill(off.begin(), off.end() - 1);
    std::vector<ExpertTask> cpu_tasks;
    for (int i = 0; i < P; ++i) {
        const int e = p.h_ids[i];
        const int at = fill[e]++;
        p.h_perm[at] = i / k;
        p.h_inv[i] = on_cpu[e] ? P : at;   // CPU pairs read the zero row
        if (on_cpu[e]) cpu_tasks.push_back({i / k, i % k, e, 0.f});
    }
    CUDA_CHECK(cudaMemcpyAsync(p.perm.p, p.h_perm, (size_t) P * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(p.inv.p, p.h_inv, (size_t) P * 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemsetAsync(p.dd.p + (size_t) P * E, 0, (size_t) E * 2, st_));
    f32_to_f16(p.xn, p.xg, (int64_t) N * E, st_);
    if (!cpu_tasks.empty()) {
        CUDA_CHECK(cudaMemcpyAsync(p.h_x, p.xn.p, (size_t) N * E * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaMemcpyAsync(p.h_w, p.w.p, (size_t) P * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaEventRecord(p.xready, st_));
    }
    // the experts on the GPU: resident ones are R-layout cache slots, staged ones the host store's ggml blobs
    const MoeLayerDesc d = cache_.desc(il);
    MoeLayerDesc ds = d;
    {
        const LayerWeights & Lw = model_.ev.layers[il];
        ds.rlay = 0;
        ds.blob = blob;
        ds.gate_bytes = Lw.gate_bytes;
        ds.up_bytes = Lw.up_bytes;
        ds.grow = ggml_row_size((ggml_type) Lw.gate_type, E);
        ds.drow = ggml_row_size((ggml_type) Lw.down_type, Fx);
    }
    const int max_blocks = P / kTcTile + NE + 1;
    TcItem * h_it[2] = {(TcItem *) p.h_items, (TcItem *) p.h_items + NE};
    TcBlock * h_bl[2] = {(TcBlock *) ((TcItem *) p.h_items + 2 * NE), (TcBlock *) ((TcItem *) p.h_items + 2 * NE) + max_blocks};
    int n_it[2] = {0, 0}, n_bl[2] = {0, 0};
    for (int e = 0; e < NE; ++e) {
        if (!cnt[e] || on_cpu[e]) continue;
        const int sl = cache_.slot_of(il, e);
        const int kind = sl >= 0 ? 0 : 1;
        const uint8_t * src = sl >= 0 ? d.base + (size_t) sl * d.blob : p.stage[il & 1] + (size_t) p.stage_idx[e] * blob;
        h_it[kind][n_it[kind]] = TcItem{src, off[e], cnt[e], e};
        for (int t0 = 0; t0 < cnt[e]; t0 += kTcTile) h_bl[kind][n_bl[kind]++] = TcBlock{n_it[kind], t0};
        ++n_it[kind];
    }
    CUDA_CHECK(cudaMemcpyAsync(p.items.p, p.h_items, p.items_bytes, cudaMemcpyHostToDevice, st_));
    const TcItem * d_it[2] = {(const TcItem *) p.items.p, (const TcItem *) p.items.p + NE};
    const TcBlock * d_bl[2] = {(const TcBlock *) ((const TcItem *) p.items.p + 2 * NE),
                               (const TcBlock *) ((const TcItem *) p.items.p + 2 * NE) + max_blocks};
    for (int kind = 0; kind < 2; ++kind) {
        if (!n_bl[kind]) continue;
        if (kind == 1) CUDA_CHECK(cudaStreamWaitEvent(st_, p.copied[il & 1], 0));
        moe_tc_gu(d_it[kind], d_bl[kind], n_bl[kind], kind == 0 ? d : ds, p.xg, E, p.perm, p.hh, E, Fx, st_);
    }
    for (int kind = 0; kind < 2; ++kind)
        if (n_bl[kind]) moe_tc_down(d_it[kind], d_bl[kind], n_bl[kind], kind == 0 ? d : ds, p.hh, p.dd, E, Fx, st_);
    CUDA_CHECK(cudaEventRecord(p.used[il & 1], st_));
    // the CPU part while the GPU works through the queue
    const float * extra = nullptr;
    if (!cpu_tasks.empty()) {
        CUDA_CHECK(cudaEventSynchronize(p.xready));
        for (auto & t : cpu_tasks) t.w = p.h_w[t.t * k + t.slot];
        for (const auto & t : cpu_tasks)
            if (!store_->blob(il, t.expert)) throw std::runtime_error("prefill: an expert without a host copy was sent to the CPU");
        const double tc = now_ms();
        cpu_->run(il, N, p.h_x, cpu_tasks, p.h_cpu);
        {
            std::lock_guard<std::mutex> lk(head_->stats_mu_);
            head_->times.cpu_experts_ms += now_ms() - tc;
            head_->times.misses += (int64_t) cpu_tasks.size();
            head_->layer_misses[il] += (int64_t) cpu_tasks.size();
        }
        CUDA_CHECK(cudaMemcpyAsync(p.cpu.p, p.h_cpu, (size_t) N * E * 4, cudaMemcpyHostToDevice, st_));
        extra = p.cpu.p;
    }
    // per token: the shared expert + the CPU part + its GPU experts
    moe_gather_sum(p.dd, p.inv, p.w, N, k, E, p.shared, nullptr, extra, p.moe, st_);
}

void DsEngine::pf_layer(int il, int N) {
    const DsConfig & c = model_.ds;
    const DsLayer & L = model_.layers[il];
    const int E = c.n_embd;
    auto & p = pf_;
    if (c.engram_index(il) >= 0) pf_engram(il, N);
    ds_hc_mixes(p.res, F(L.hc_attn_fn), F(L.hc_attn_scale), F(L.hc_attn_base), p.attn_pre, p.post, p.comb, N, c.hc, E, c.eps, c.hc_eps,
                c.sinkhorn, st_);
    ds_hc_pre(p.res, pre_, p.mixed, N, c.hc, E, st_);
    rmsnorm_rows(p.mixed, F(L.attn_norm), p.xn, N, E, E, E, c.eps, st_);
    pf_attn(il, N);
    ds_hc_post(p.out, p.res, p.post, p.comb, N, c.hc, E, st_);
    ds_hc_mixes(p.res, F(L.hc_ffn_fn), F(L.hc_ffn_scale), F(L.hc_ffn_base), pre_next_, p.post, p.comb, N, c.hc, E, c.eps, c.hc_eps,
                c.sinkhorn, st_);
    ds_hc_pre(p.res, p.attn_pre, p.mixed, N, c.hc, E, st_);
    rmsnorm_rows(p.mixed, F(L.ffn_norm), p.xn, N, E, E, E, c.eps, st_);
    pf_moe(il, N);
    ds_hc_post(p.moe, p.res, p.post, p.comb, N, c.hc, E, st_);
    std::swap(pre_, pre_next_);
}

// the logits of the chunk's last token (row 0 of the decode logits, last_T = 1)
void DsEngine::pf_head(int N) {
    const DsConfig & c = model_.ds;
    auto & p = pf_;
    ds_hc_pre(p.res.p + (size_t) (N - 1) * c.hc_dim(), pre_ + (size_t) (N - 1) * c.hc, mixed_, 1, c.hc, c.n_embd, st_);
    rmsnorm_rows(mixed_, F(model_.output_norm), xn_, 1, c.n_embd, c.n_embd, c.n_embd, c.eps, st_);
    quantize_act(xn_, c.n_embd, 1, c.n_embd, act_, st_);
    gemv(model_.output, &act_, nullptr, 0, 1, logits_, c.n_vocab, false, st_);
}

void DsEngine::prefill_chunk(const int32_t * tokens, int N) {
    const double t0 = now_ms();
    for (DsEngine * s = this; s; s = s->next_.get()) s->prefill_chunk_here(tokens, N);
    last_T = 1;
    std::lock_guard<std::mutex> lk(stats_mu_);
    times.total_ms += now_ms() - t0;
}

void DsEngine::prefill_chunk_here(const int32_t * tokens, int N) {
    DeviceScope dsc(dev_);
    join_commit_here();
    const DsConfig & c = model_.ds;
    const int E = c.n_embd, HC = c.hc_dim();
    auto & p = pf_;
    if (N < 1 || N > pf_max_) throw std::runtime_error("prefill_chunk: bad size");
    if (cur_->pending_T) throw std::runtime_error("prefill_chunk: a verify window is pending");
    if (pos() + N > opt_.max_ctx) throw std::runtime_error("context full");
    const int pos0 = pos();
    ensure_ctx_here(pos0 + N + kMaxWindow);
    ctx_grid_ = cur_->ctx_mapped;
    cur_->history.insert(cur_->history.end(), tokens, tokens + N);
    *h_par_ = WinParams{pos0, N, ++seq_, 0};
    CUDA_CHECK(cudaMemcpyAsync(d_par_, h_par_, sizeof(WinParams), cudaMemcpyHostToDevice, st_));
    {   // the first position of every indexer sub-chunk
        std::vector<int32_t> ps;
        for (int r0 = 0; r0 < N; r0 += p.score_rows) ps.push_back(pos0 + r0);
        CUDA_CHECK(cudaMemcpyAsync(p.pos.p, ps.data(), ps.size() * 4, cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));   // (pageable source)
    }
    pf_engram_gather(N);
    pre_ = p.pre_a;
    pre_next_ = p.pre_b;
    if (first()) {
        CUDA_CHECK(cudaMemcpyAsync(p.tok.p, tokens, (size_t) N * 4, cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
        dequant_gather(model_.tok_embd, p.tok, N, p.mixed, st_);
        hc_init(p.mixed, p.res, N, c.hc, E, st_);
        ds_onehot(pre_, N, c.hc, st_);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(p.res.p, pf_xfer_, (size_t) N * HC * 4, cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaMemcpyAsync(pre_, pf_xfer_ + (size_t) pf_max_ * HC, (size_t) N * c.hc * 4, cudaMemcpyHostToDevice, st_));
    }
    const double t0 = now_ms();
    for (int il = l0_; il < l1_; ++il) pf_layer(il, N);
    if (last_) pf_head(N);
    else {
        CUDA_CHECK(cudaMemcpyAsync(pf_xfer_, p.res.p, (size_t) N * HC * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaMemcpyAsync(pf_xfer_ + (size_t) pf_max_ * HC, pre_, (size_t) N * c.hc * 4, cudaMemcpyDeviceToHost, st_));
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    cur_->pending_T = 0;
    std::lock_guard<std::mutex> lk(head_->stats_mu_);
    gpu_ms_ += now_ms() - t0;
}

}  // namespace bnk
