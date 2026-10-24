// Batched prompt processing for the DeepSeek engine (docs/DEEPSEEK.md). A group of up to pf_group_ tokens runs
// layer-major: each layer first takes the group's chunks one after another through the fp16 tensor-core GEMMs
// (weights dequantized on the fly), the engine's own attention / compressor / indexer / Engram kernels and the shared
// expert, then the layer's routed experts over the whole group at once on the tensor cores: resident experts from the
// VRAM cache, the others streamed from the host store into two staging buffers in waves (the copy engine kept busy,
// the next layer's first wave prefetched by rank during this layer's attention), the rest on the CPU pool. The split
// between PCIe and the CPU follows a cost model whose constants are corrected from the measured copy and CPU times.
// Every non-resident expert crosses PCIe once per layer per group instead of once per chunk, which is what long
// prompts are bound by.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

#include <cublas_v2.h>

#include "deepseek/ds_engine.h"
#include "kernels/gemm.h"
#include "kernels/ops.h"

namespace bnk {

static const float * F(const QMat & m) { return (const float *) m.data; }

// BNK_DS_PF_PROF=1: wall time per phase of the prompt path (synchronously measured, so a little slower)
static const bool g_pf_prof = getenv("BNK_DS_PF_PROF") != nullptr;
struct PfProf {
    double engram_host = 0, setup = 0, attn = 0, route = 0, enqueue = 0, cpu = 0, gather = 0, head = 0, total = 0;
    double copy_mb = 0;
    int64_t cpu_pairs = 0, gpu_pairs = 0, staged = 0, resident = 0;
};
static thread_local PfProf g_prof;
static inline double prof_lap(cudaStream_t st, double & t) {
    if (!g_pf_prof) return 0;
    CUDA_CHECK(cudaStreamSynchronize(st));
    const double now = now_ms(), d = now - t;
    t = now;
    return d;
}

// ------------------------------------------------------------------------------------------------ worker
DsEngine::Worker::Worker() {
    th = std::thread([this]() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv.wait(lk, [&]() { return quit || job; });
            if (!job) return;
            auto f = std::move(job);
            job = nullptr;
            lk.unlock();
            try {
                f();
            } catch (...) {
                lk.lock();
                err = std::current_exception();
                lk.unlock();
            }
            lk.lock();
            busy = false;
            cv.notify_all();
        }
    });
}

DsEngine::Worker::~Worker() {
    {
        std::lock_guard<std::mutex> lk(mu);
        quit = true;
    }
    cv.notify_all();
    th.join();
}

void DsEngine::Worker::post(std::function<void()> f) {
    wait();
    std::lock_guard<std::mutex> lk(mu);
    job = std::move(f);
    busy = true;
    cv.notify_all();
}

void DsEngine::Worker::wait() {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [&]() { return !busy; });
    if (err) {
        auto e = err;
        err = nullptr;
        std::rethrow_exception(e);
    }
}

// ------------------------------------------------------------------------------------------------ buffers
void DsEngine::prefill_alloc(int N, int G) {
    const DsConfig & c = model_.ds;
    const int E = c.n_embd, HC = c.hc_dim(), H = c.n_head, D = c.head_dim, QL = c.q_lora, OG = c.o_groups, OL = c.o_lora;
    const int IH = c.idx_heads, ID = c.idx_dim, Fx = c.n_ff, k = c.n_used, NE = c.n_expert;
    const int NG = N * G, P = NG * k;
    auto & p = pf_;
    p.n_chunks_max = G;
    // group-sized
    p.res.alloc((size_t) NG * HC); p.xn.alloc((size_t) NG * E);
    p.pre_a.alloc((size_t) NG * c.hc); p.pre_b.alloc((size_t) NG * c.hc); p.post.alloc((size_t) NG * c.hc);
    p.comb.alloc((size_t) NG * c.hc * c.hc);
    p.shared.alloc((size_t) NG * E); p.moe.alloc((size_t) NG * E); p.cpu.alloc((size_t) NG * E);
    p.rlog.alloc((size_t) NG * NE); p.w.alloc(P); p.w2.alloc(P); p.ids.alloc(P); p.ids2.alloc(P); p.perm.alloc(P + 4); p.inv.alloc(P + 4);
    p.xg.alloc((size_t) NG * E); p.hh.alloc((size_t) P * Fx); p.dd.alloc((size_t) (P + 1) * E);
    // chunk-sized
    p.mixed.alloc((size_t) N * E); p.xnc.alloc((size_t) N * E); p.out.alloc((size_t) N * E);
    p.attn_pre.alloc((size_t) N * c.hc); p.postc.alloc((size_t) N * c.hc); p.combc.alloc((size_t) N * c.hc * c.hc);
    p.qr.alloc((size_t) N * QL); p.qrn.alloc((size_t) N * QL); p.q.alloc((size_t) N * H * D); p.kvr.alloc((size_t) N * D);
    p.attn_o.alloc((size_t) N * H * D); p.oa.alloc((size_t) N * OG * OL); p.ckv.alloc((size_t) N * D); p.sc.alloc((size_t) N * D);
    p.kwin.alloc((size_t) (N + c.window) * D);
    p.iq.alloc((size_t) N * IH * ID); p.iw.alloc((size_t) N * IH);
    // the indexer's scores for a piece of the chunk at a time: at most ~128 MiB
    p.score_rows = std::max(1, std::min(N, (int) ((128u << 20) / (lds_ * 4))));
    p.n_sub = (N + p.score_rows - 1) / p.score_rows;
    p.scores.alloc((size_t) p.score_rows * lds_);
    if (n_blocks_max_) { p.cand_scratch.alloc((size_t) p.score_rows * n_blocks_max_); p.cand.alloc((size_t) N * n_blocks_max_); }
    p.idx.alloc((size_t) N * c.idx_topk);
    p.pos.alloc((size_t) G * p.n_sub + 1);
    p.tok.alloc(N);
    p.sg.alloc((size_t) N * 2 * Fx); p.sh.alloc((size_t) N * Fx);
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_par, (size_t) G * sizeof(WinParams), cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc((void **) &p.d_par, (size_t) G * sizeof(WinParams)));
    int n_eg = 0;
    for (int il : c.engram_layers) n_eg += owns(il);
    if (n_eg) {
        const int64_t cols = (int64_t) c.eg_cols() * c.eg_dim;
        p.eg_emb.alloc((size_t) N * cols);
        p.eg_kv.alloc((size_t) N * (c.hc + 1) * E);
        p.eg_rows_per_layer = (size_t) NG * c.eg_cols() * model_.engram_tab[0]->row_bytes();
        CUDA_CHECK(cudaHostAlloc((void **) &p.eg_rows, p.eg_rows_per_layer * c.engram_layers.size(), cudaHostAllocMapped));
    }
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_ids, (size_t) P * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_ids2, (size_t) P * 4, cudaHostAllocDefault));
    // the host side of the expert tables and the CPU answer: mapped, read by kernels (not copy-engine transfers: those
    // would queue behind the expert prefetch and stall the compute stream, see ds_copy_from_host)
    const size_t P16 = ((size_t) P * 4 + 15) / 16 * 16;
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_perm, P16, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_inv, P16, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_w, (size_t) P * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_x, (size_t) NG * E * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_cpu, (size_t) NG * E * 4, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_cpu_map, (size_t) NG * 4, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_cpu_tok, (size_t) NG * 4, cudaHostAllocMapped));
    // the CPU's rows go out compact: a copy of every row (168 MB at 8,192 tokens) took 52 ms on the compute stream
    // per layer beside the expert stream, and everything behind it waited. A quarter of the group's rows is room enough
    // (the CPU takes a few hundred tokens' experts a layer); more falls back to the whole copy.
    p.xc_rows = std::max(1, NG / 4);
    p.xc.alloc((size_t) p.xc_rows * E);
    p.max_blocks = P / kTcTile + NE + 1;
    p.items_bytes = ((size_t) NE * sizeof(TcItem) + (size_t) p.max_blocks * sizeof(TcBlock) + 15) / 16 * 16;
    CUDA_CHECK(cudaHostAlloc((void **) &p.h_items, p.items_bytes, cudaHostAllocMapped));
    p.items.alloc(p.items_bytes);
    CUDA_CHECK(cudaHostGetDevicePointer((void **) &p.d_h_perm, p.h_perm, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **) &p.d_h_inv, p.h_inv, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **) &p.d_h_cpu, p.h_cpu, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **) &p.d_h_cpu_map, p.h_cpu_map, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **) &p.d_h_cpu_tok, p.h_cpu_tok, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((void **) &p.d_h_items, p.h_items, 0));
    p.stage_idx.assign(NE, -1);
    p.stage_buf.assign(NE, 0);
    CUDA_CHECK(cudaStreamCreateWithFlags(&p.copy, cudaStreamNonBlocking));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&p.copied[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&p.used[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&p.xready, cudaEventDisableTiming));
    for (int i = 0; i < kPfWaves; ++i) {
        CUDA_CHECK(cudaEventCreate(&p.wt0[i]));
        CUDA_CHECK(cudaEventCreate(&p.wt1[i]));
    }
    // GEMM workspace: the largest matrix dequantized at once, and a chunk x the widest input
    size_t maxw = 0;
    for (int il = l0_; il < l1_; ++il) {
        const DsLayer & L = model_.layers[il];
        for (const QMat * m : {&L.q_a, &L.q_b, &L.kv, &L.o_b, &L.sh_gu, &L.sh_gate, &L.sh_up, &L.sh_down, &L.eg_wkv, &L.comp_kv,
                               &L.comp_gate, &L.idx_q_b, &L.idx_proj})
            if (m->valid()) maxw = std::max(maxw, (size_t) m->rows * m->cols);
        maxw = std::max(maxw, (size_t) OL * (H * D / OG));
    }
    const int64_t maxk = std::max<int64_t>({(int64_t) E, (int64_t) QL, (int64_t) H * D / OG, (int64_t) OG * OL, (int64_t) Fx,
                                            (int64_t) c.eg_cols() * c.eg_dim});
    gemm_.init(st_, maxw, (size_t) N * maxk);
    pf_pcie_us_mb_ = env_double("BNK_PCIE_US_PER_MB", opt_.pcie_us_per_mb);
}

// the group's Engram table rows for one Engram layer (host; the rows are read from the memory-mapped table on
// several threads so that the page faults overlap). Runs on a thread of its own from the group's start, joined before
// the layer: a cold table costs ~0.4 s per 2,048 tokens, which the earlier layers' GPU work hides.
void DsEngine::pf_engram_gather(int N, int e_only) {
    const DsConfig & c = model_.ds;
    auto & p = pf_;
    if (!p.eg_rows) return;
    const double t0 = now_ms();
    const int cols = c.eg_cols(), pos0 = pos() - N;
    for (size_t e = 0; e < c.engram_layers.size(); ++e) {
        if (!owns(c.engram_layers[e]) || (e_only >= 0 && (int) e != e_only)) continue;
        const TensorRef & tab = *model_.engram_tab[e];
        const size_t rb = tab.row_bytes();
        uint8_t * dst = p.eg_rows + e * p.eg_rows_per_layer;
        auto part = [&](int t0_, int t1_) {
            uint64_t rows[64];
            for (int t = t0_; t < t1_; ++t) {
                engram_hash((int) e, cur_->history, pos0 + t, rows);
                for (int col = 0; col < cols; ++col) memcpy(dst + ((size_t) t * cols + col) * rb, tab.data + rows[col] * rb, rb);
            }
        };
        static const int max_th = env_int("BNK_DS_EG_THREADS", 32);
        const int nth = std::max(1, std::min(max_th, N / 32));
        if (nth == 1) part(0, N);
        else {
            std::vector<std::thread> th;
            for (int i = 0; i < nth; ++i) th.emplace_back(part, (int) ((int64_t) N * i / nth), (int) ((int64_t) N * (i + 1) / nth));
            for (auto & t : th) t.join();
        }
    }
    std::lock_guard<std::mutex> lk(head_->stats_mu_);
    head_->times.ple_ms += now_ms() - t0;
    pf_eg_ms_ += now_ms() - t0;   // (overlapped with the GPU work of the layers before)
}

// ------------------------------------------------------------------------------------------------ a chunk's layer
void DsEngine::pf_engram(int il, int c_, int N) {
    const DsConfig & c = model_.ds;
    const DsLayer & L = model_.layers[il];
    const int e = c.engram_index(il), E = c.n_embd;
    auto & p = pf_;
    const TensorRef & tab = *model_.engram_tab[e];
    const size_t o = (size_t) c_ * pf_max_;
    const int64_t rows = (int64_t) N * c.eg_cols(), D = (int64_t) c.eg_cols() * c.eg_dim;
    QMat rm{p.eg_rows + e * p.eg_rows_per_layer + o * c.eg_cols() * tab.row_bytes(), (int) tab.type, rows, tab.ne[0], tab.row_bytes()};
    dequant_rows(rm, 0, rows, p.eg_emb, st_);
    gemm_.run(L.eg_wkv, p.eg_emb, D, N, p.eg_kv, (int64_t) (c.hc + 1) * E);
    ds_engram_gate(p.res.p + o * c.hc_dim(), p.eg_kv, (int64_t) (c.hc + 1) * E, p.eg_kv.p + (size_t) c.hc * E, (int64_t) (c.hc + 1) * E,
                   F(L.eg_gate), N, c.hc, E, c.eps, st_);
}

// attention of chunk c_ (N rows at position pos0) from p.xnc into p.out
void DsEngine::pf_attn(int il, int c_, int N, int pos0) {
    const DsConfig & c = model_.ds;
    const DsLayer & L = model_.layers[il];
    const int E = c.n_embd, H = c.n_head, D = c.head_dim, QL = c.q_lora, R = c.n_rot, IH = c.idx_heads, ID = c.idx_dim;
    const int ratio = c.ratio[il];
    const bool src = c.is_kv_src(il), isrc = c.is_idx_src(il);
    auto & p = pf_;
    const int origin = std::max(0, pos0 - c.window);
    const int * d_pos0 = &p.d_par[c_].pos0;
    // every projection of the layer input first (Gemm::run converts its own input into the same fp16 workspace)
    f32_to_f16(p.xnc, gemm_.xbuf(), (int64_t) N * E, st_);
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
    if (pos0 > origin) ds_ring_read(cur_->ring[il], origin, p.kwin, origin, pos0, D, st_);
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
                const int * pj = p.pos.p + c_ * p.n_sub + j;
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
    // the ring takes the chunk's last positions for the chunks and decode windows that follow
    ds_ring_fill(p.kwin, origin, cur_->ring[il], std::max(pos0, pos0 + N - kDsRing), pos0 + N, D, st_);
    const int OG = c.o_groups, GW = H * D / OG, OL = c.o_lora;
    for (int g = 0; g < OG; ++g)
        gemm_.run(qmat_rows(L.o_a, (int64_t) g * OL, OL), p.attn_o.p + (size_t) g * GW, (int64_t) H * D, N, p.oa.p + (size_t) g * OL,
                  (int64_t) OG * OL);
    gemm_.run(L.o_b, p.oa, (int64_t) OG * OL, N, p.out, E);
}

// the shared expert of chunk c_ from its rows of p.xn into its rows of p.shared
void DsEngine::pf_shared(int il, int c_, int N) {
    const DsConfig & c = model_.ds;
    const DsLayer & L = model_.layers[il];
    const int E = c.n_embd, Fx = c.n_ff;
    auto & p = pf_;
    const size_t o = (size_t) c_ * pf_max_;
    f32_to_f16(p.xn.p + o * E, gemm_.xbuf(), (int64_t) N * E, st_);
    if (L.sh_gu.valid()) gemm_.run_h(L.sh_gu, gemm_.xbuf(), E, N, p.sg, 2 * Fx);
    else {
        gemm_.run_h(L.sh_gate, gemm_.xbuf(), E, N, p.sg, 2 * Fx);
        gemm_.run_h(L.sh_up, gemm_.xbuf(), E, N, p.sg.p + Fx, 2 * Fx);
    }
    ds_silu_mul_clamp(p.sg, p.sh, N, Fx, c.swiglu_limit, st_);
    gemm_.run(L.sh_down, p.sh, Fx, N, p.shared.p + o * E, E);
}

// ------------------------------------------------------------------------------------------------ the experts
// the cost model's constants from the last layer's measurements (after the router sync: every timed copy is done)
void DsEngine::pf_update_costs() {
    auto & p = pf_;
    if (p.n_timed > 0 && p.timed_mb > 0) {
        double ms = 0;
        bool ok = true;
        for (int i = 0; i < p.n_timed && ok; ++i) {
            float f = 0;
            if (cudaEventElapsedTime(&f, p.wt0[i], p.wt1[i]) == cudaSuccess) ms += f;
            else { cudaGetLastError(); ok = false; }
        }
        if (ok && ms > 0) {
            const double us_mb = ms * 1000.0 / p.timed_mb;
            pf_pcie_us_mb_ = pf_pcie_us_mb_ <= 0 ? us_mb : 0.7 * pf_pcie_us_mb_ + 0.3 * us_mb;
        }
    }
    p.n_timed = 0;
    p.timed_mb = 0;
    if (p.cpu_pred_ms > 0 && p.cpu_meas_ms > 0) {
        const double r = p.cpu_meas_ms / p.cpu_pred_ms;
        std::lock_guard<std::mutex> lk(head_->stats_mu_);
        head_->pf_cpu_scale_ = 0.7 * head_->pf_cpu_scale_ + 0.3 * r * head_->pf_cpu_scale_;
        head_->pf_cpu_scale_ = std::min(100.0, std::max(0.01, head_->pf_cpu_scale_));
    }
    p.cpu_pred_ms = p.cpu_meas_ms = 0;
}

// The first waves of layer il's non-resident experts into up to nbuf staging buffers, in the given order, during the
// layer before (the copy engine would otherwise idle through the attention phase). The order is the next layer's
// routing guessed from this layer's FFN input (its router applied one layer early: what the tokens would pick there,
// most-used first; nearly all of it is right for a long chunk), or the ranking for a group's first layer.
void DsEngine::pf_prefetch(int il, const std::vector<int> & r, int nbuf) {
    auto & p = pf_;
    p.pre_layer = -1;
    p.pre_list.clear();
    p.pre_bufs.clear();
    p.pre_pos.clear();
    if (il < l0_ || il >= l1_ || r.empty() || nbuf <= 0) return;
    const size_t blob = store_->blob_bytes(il);
    const size_t cap = blob ? p.stage_bytes / blob : 0;
    size_t at = 0;
    for (int w = 0; w < nbuf && at < r.size(); ++w) {
        const size_t n = std::min(cap, r.size() - at);
        if (!n) break;
        const int b = p.toggle;
        p.toggle ^= 1;
        CUDA_CHECK(cudaStreamWaitEvent(p.copy, p.used[b], 0));
        for (size_t i = 0; i < n; ++i)
            CUDA_CHECK(cudaMemcpyAsync(p.stage[b] + i * blob, store_->blob(il, r[at + i]), blob, cudaMemcpyHostToDevice, p.copy));
        CUDA_CHECK(cudaEventRecord(p.copied[b], p.copy));
        for (size_t i = 0; i < n; ++i) {
            p.pre_list.push_back(r[at + i]);
            p.pre_bufs.push_back((char) b);
            p.pre_pos.push_back((int) i);
        }
        at += n;
        g_prof.copy_mb += n * blob / 1048576.0;
    }
    if (p.pre_list.empty()) return;
    p.pre_layer = il;
}

void DsEngine::pf_moe(int il, int N) {
    const DsConfig & c = model_.ds;
    const DsLayer & L = model_.layers[il];
    const int E = c.n_embd, k = c.n_used, Fx = c.n_ff, NE = c.n_expert, P = N * k;
    auto & p = pf_;
    double tp = now_ms();
    // routing in fp32 (near ties pick the experts)
    {
        const float one = 1.f, zero = 0.f;
        if (cublasSgemm(gemm_.handle(), CUBLAS_OP_T, CUBLAS_OP_N, NE, N, E, &one, F(L.router), E, p.xn, E, &zero, p.rlog, NE) !=
            CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("prefill: router GEMM failed");
    }
    ds_route(p.rlog, F(L.probs_b), N, NE, k, c.route_scale, c.route_norm, p.ids, p.w, st_);
    CUDA_CHECK(cudaMemcpyAsync(p.h_ids, p.ids.p, (size_t) P * 4, cudaMemcpyDeviceToHost, st_));
    static const int guess_bufs = std::max(0, std::min(2, env_int("BNK_DS_PF_GUESS", 2)));
    const bool guess = guess_bufs > 0 && il + 1 < l1_;
    if (guess) {   // the next layer's router on this layer's FFN input: its experts, to prefetch during its attention
        const DsLayer & L2 = model_.layers[il + 1];
        const float one = 1.f, zero = 0.f;
        if (cublasSgemm(gemm_.handle(), CUBLAS_OP_T, CUBLAS_OP_N, NE, N, E, &one, F(L2.router), E, p.xn, E, &zero, p.rlog, NE) !=
            CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("prefill: router GEMM failed");
        ds_route(p.rlog, F(L2.probs_b), N, NE, k, c.route_scale, c.route_norm, p.ids2, p.w2, st_);
        CUDA_CHECK(cudaMemcpyAsync(p.h_ids2, p.ids2.p, (size_t) P * 4, cudaMemcpyDeviceToHost, st_));
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    g_prof.route += now_ms() - tp;
    tp = now_ms();
    pf_update_costs();

    // the (token, slot) pairs grouped by expert
    std::vector<int> cnt(NE, 0), off(NE + 1, 0);
    for (int i = 0; i < P; ++i) cnt[p.h_ids[i]]++;
    for (int e = 0; e < NE; ++e) off[e + 1] = off[e] + cnt[e];
    {
        std::lock_guard<std::mutex> lk(head_->stats_mu_);
        head_->times.routed += P;
        head_->layer_routed[il] += P;
    }
    // where each used non-resident expert runs: the prefetched wave (free), PCIe waves (most-used first, while that
    // beats the CPU), or the CPU
    const size_t blob = store_->blob_bytes(il);
    const size_t cap = blob ? p.stage_bytes / blob : 0;
    std::vector<char> on_cpu(NE, 0), pre(NE, 0);
    std::fill(p.stage_idx.begin(), p.stage_idx.end(), -1);
    if (p.pre_layer == il)
        for (size_t i = 0; i < p.pre_list.size(); ++i) {
            const int e = p.pre_list[i];
            pre[e] = 1;
            p.stage_idx[e] = p.pre_pos[i];
            p.stage_buf[e] = p.pre_bufs[i];
        }
    std::vector<int> nr;   // candidates for PCIe / CPU, most-used first
    for (int e = 0; e < NE; ++e)
        if (cnt[e] && cache_.slot_of(il, e) < 0 && !pre[e]) nr.push_back(e);
    std::sort(nr.begin(), nr.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
    double cpu_scale;
    {
        std::lock_guard<std::mutex> lk(head_->stats_mu_);
        cpu_scale = head_->pf_cpu_scale_;
    }
    static const double cpu_us = env_double("BNK_CPU_US_PER_EXPERT_TOKEN", opt_.cpu_us_per_expert_token);
    const double t_pcie = pf_pcie_us_mb_ * blob / 1e6 / 1000.0;              // ms per expert over PCIe
    const double t_tok = cpu_scale * cpu_us * blob / 2.2e6 / 1000.0;          // ms per (expert, token) on the CPU (the
                                                                              // constant was measured on 2.2 MB experts)
    auto t_cpu = [&](int e) { return t_tok * (cnt[e] + 1); };                 // one decode of the rows plus a dot per token
    double cpu_total = 0;
    for (int e : nr) cpu_total += t_cpu(e);
    size_t n_pcie = 0;
    double pcie_total = 0;
    const size_t max_pcie = std::min(nr.size(), cap * (size_t) (kPfWaves - 1));
    while (n_pcie < max_pcie) {
        const double cpu_after = cpu_total - t_cpu(nr[n_pcie]);
        if (std::max(pcie_total + t_pcie, cpu_after) >= std::max(pcie_total, cpu_total)) break;
        pcie_total += t_pcie;
        cpu_total = cpu_after;
        ++n_pcie;
    }
    for (size_t i = n_pcie; i < nr.size(); ++i) on_cpu[nr[i]] = 1;
    // the waves: cap experts each, alternating buffers
    std::vector<std::pair<size_t, size_t>> waves;   // [from, to) into nr
    std::vector<int> wave_buf;
    for (size_t i = 0; i < n_pcie; i += cap) {
        waves.push_back({i, std::min(n_pcie, i + cap)});
        wave_buf.push_back(p.toggle);
        p.toggle ^= 1;
        for (size_t j = waves.back().first; j < waves.back().second; ++j) {
            p.stage_idx[nr[j]] = (int) (j - waves.back().first);
            p.stage_buf[nr[j]] = (char) wave_buf.back();
        }
    }
    // the permutation (pairs grouped by expert) and the CPU tasks
    std::vector<int> fill(off.begin(), off.end() - 1);
    std::vector<ExpertTask> cpu_tasks;
    for (int i = 0; i < P; ++i) {
        const int e = p.h_ids[i];
        const int at = fill[e]++;
        p.h_perm[at] = i / k;
        p.h_inv[i] = on_cpu[e] ? P : at;   // CPU pairs read the zero row
        if (on_cpu[e]) cpu_tasks.push_back({i / k, i % k, e, 0.f});
    }
    ds_copy_from_host(p.d_h_perm, p.perm.p, ((size_t) P * 4 + 15) / 16 * 16, st_);
    ds_copy_from_host(p.d_h_inv, p.inv.p, ((size_t) P * 4 + 15) / 16 * 16, st_);
    ds_zero_half(p.dd.p + (size_t) P * E, (size_t) E, st_);
    f32_to_f16(p.xn, p.xg, (int64_t) N * E, st_);
    // the CPU tokens' rows (h_cpu_map: a token's row among them, -1 none), compact when they fit xc
    int M = 0;
    bool compact = false;
    if (!cpu_tasks.empty()) {
        for (int t = 0; t < N; ++t) p.h_cpu_map[t] = -1;
        for (const auto & t : cpu_tasks)
            if (p.h_cpu_map[t.t] < 0) { p.h_cpu_map[t.t] = M; p.h_cpu_tok[M++] = t.t; }
        compact = M <= p.xc_rows;
        if (compact) {
            ds_rows_gather(p.xn.p, p.d_h_cpu_tok, p.xc.p, M, E, st_);
            CUDA_CHECK(cudaMemcpyAsync(p.h_x, p.xc.p, (size_t) M * E * 4, cudaMemcpyDeviceToHost, st_));
        } else {
            for (int t = 0; t < N; ++t) p.h_cpu_map[t] = p.h_cpu_map[t] >= 0 ? t : -1;
            CUDA_CHECK(cudaMemcpyAsync(p.h_x, p.xn.p, (size_t) N * E * 4, cudaMemcpyDeviceToHost, st_));
        }
        CUDA_CHECK(cudaMemcpyAsync(p.h_w, p.w.p, (size_t) P * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaEventRecord(p.xready, st_));
    }
    // the GPU items: resident experts (R-layout cache slots) first, then the prefetched wave, then each PCIe wave;
    // the blocks of each set contiguous
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
    TcItem * h_it = (TcItem *) p.h_items;
    TcBlock * h_bl = (TcBlock *) (h_it + NE);
    int n_it = 0, n_bl = 0;
    struct Set { int bl0, nbl; int buf; };   // buf: -1 resident, else the staging buffer to wait for
    std::vector<Set> sets;
    auto add_set = [&](auto && take, int buf) {
        Set s{n_bl, 0, buf};
        for (int e = 0; e < NE; ++e) {
            if (!cnt[e] || on_cpu[e] || !take(e)) continue;
            const int sl = cache_.slot_of(il, e);
            const uint8_t * src = sl >= 0 ? d.base + (size_t) sl * d.blob : p.stage[(int) p.stage_buf[e]] + (size_t) p.stage_idx[e] * blob;
            h_it[n_it] = TcItem{src, off[e], cnt[e], e};
            for (int t0 = 0; t0 < cnt[e]; t0 += kTcTile) h_bl[n_bl++] = TcBlock{n_it, t0};
            ++n_it;
            if (sl >= 0) g_prof.resident++; else g_prof.staged++;
        }
        s.nbl = n_bl - s.bl0;
        sets.push_back(s);
    };
    add_set([&](int e) { return cache_.slot_of(il, e) >= 0; }, -1);
    if (p.pre_layer == il)   // the prefetched sets, one per buffer, in the order they were copied
        for (int b : {p.pre_bufs.front(), (char) (p.pre_bufs.front() ^ 1)})
            if (std::find(p.pre_bufs.begin(), p.pre_bufs.end(), (char) b) != p.pre_bufs.end())
                add_set([&](int e) { return pre[e] && p.stage_buf[e] == (char) b; }, b);
    for (size_t w = 0; w < waves.size(); ++w) {
        std::vector<char> in(NE, 0);
        for (size_t j = waves[w].first; j < waves[w].second; ++j) in[nr[j]] = 1;
        add_set([&](int e) { return in[e] != 0; }, wave_buf[w]);
    }
    if (n_bl > p.max_blocks) throw std::runtime_error("prefill: too many expert blocks");
    ds_copy_from_host(p.d_h_items, p.items.p, p.items_bytes, st_);
    const TcItem * d_it = (const TcItem *) p.items.p;
    const TcBlock * d_bl = (const TcBlock *) (d_it + NE);
    // resident experts: no wait; the prefetched wave: its copies are in flight since the layer before
    const size_t n_fixed = sets.size() - waves.size();
    for (size_t si = 0; si < n_fixed; ++si) {
        const Set & s = sets[si];
        if (!s.nbl) continue;
        if (s.buf >= 0) CUDA_CHECK(cudaStreamWaitEvent(st_, p.copied[s.buf], 0));
        moe_tc_gu(d_it, d_bl + s.bl0, s.nbl, s.buf < 0 ? d : ds, p.xg, E, p.perm, p.hh, E, Fx, st_);
        moe_tc_down(d_it, d_bl + s.bl0, s.nbl, s.buf < 0 ? d : ds, p.hh, p.dd, E, Fx, st_);
        if (s.buf >= 0) CUDA_CHECK(cudaEventRecord(p.used[s.buf], st_));
    }
    // the PCIe waves: each one's copies after its buffer's last reader, its compute after its copies
    for (size_t w = 0; w < waves.size(); ++w) {
        const int b = wave_buf[w];
        CUDA_CHECK(cudaStreamWaitEvent(p.copy, p.used[b], 0));
        if (p.n_timed < kPfWaves) CUDA_CHECK(cudaEventRecord(p.wt0[p.n_timed], p.copy));
        for (size_t j = waves[w].first; j < waves[w].second; ++j) {
            const int e = nr[j];
            CUDA_CHECK(cudaMemcpyAsync(p.stage[b] + (size_t) p.stage_idx[e] * blob, store_->blob(il, e), blob, cudaMemcpyHostToDevice, p.copy));
        }
        if (p.n_timed < kPfWaves) {
            CUDA_CHECK(cudaEventRecord(p.wt1[p.n_timed], p.copy));
            p.timed_mb += (waves[w].second - waves[w].first) * blob / 1048576.0;
            ++p.n_timed;
        }
        g_prof.copy_mb += (waves[w].second - waves[w].first) * blob / 1048576.0;
        CUDA_CHECK(cudaEventRecord(p.copied[b], p.copy));
        const Set & s = sets[n_fixed + w];
        CUDA_CHECK(cudaStreamWaitEvent(st_, p.copied[b], 0));
        if (s.nbl) {
            moe_tc_gu(d_it, d_bl + s.bl0, s.nbl, ds, p.xg, E, p.perm, p.hh, E, Fx, st_);
            moe_tc_down(d_it, d_bl + s.bl0, s.nbl, ds, p.hh, p.dd, E, Fx, st_);
        }
        CUDA_CHECK(cudaEventRecord(p.used[b], st_));
    }
    // the next layer's first waves meanwhile: the experts its guessed routing names, most-used first
    if (guess) {
        std::vector<int> cnt2(NE, 0);
        for (int i = 0; i < P; ++i) cnt2[p.h_ids2[i]]++;
        std::vector<int> order;
        for (int e = 0; e < NE; ++e)
            if (cnt2[e] && cache_.slot_of(il + 1, e) < 0) order.push_back(e);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return cnt2[a] > cnt2[b]; });
        pf_prefetch(il + 1, order, guess_bufs);
    } else {
        pf_prefetch(il + 1, {}, 0);
    }
    g_prof.enqueue += now_ms() - tp;
    g_prof.gpu_pairs += P - (int64_t) cpu_tasks.size();
    g_prof.cpu_pairs += (int64_t) cpu_tasks.size();
    // the CPU part while the GPU works through the queue
    const float * extra = nullptr;
    if (!cpu_tasks.empty()) {
        CUDA_CHECK(cudaEventSynchronize(p.xready));
        for (auto & t : cpu_tasks) t.w = p.h_w[t.t * k + t.slot];
        for (const auto & t : cpu_tasks)
            if (!store_->blob(il, t.expert)) throw std::runtime_error("prefill: an expert without a host copy was sent to the CPU");
        const double tc = now_ms();
        if (compact) {
            std::vector<ExpertTask> ct = cpu_tasks;   // the tasks on the compact rows
            for (auto & t : ct) t.t = p.h_cpu_map[t.t];
            cpu_->run(il, M, p.h_x, ct, p.h_cpu);
        } else
            cpu_->run(il, N, p.h_x, cpu_tasks, p.h_cpu);
        const double ms = now_ms() - tc;
        p.cpu_meas_ms = ms;
        p.cpu_pred_ms = cpu_total;
        g_prof.cpu += ms;
        {
            std::lock_guard<std::mutex> lk(head_->stats_mu_);
            head_->times.cpu_experts_ms += ms;
            head_->times.misses += (int64_t) cpu_tasks.size();
            head_->layer_misses[il] += (int64_t) cpu_tasks.size();
        }
        ds_rows_from_host(p.d_h_cpu, p.d_h_cpu_map, p.cpu.p, N, E, st_);
        extra = p.cpu.p;
    }
    // per token: the shared expert + the CPU part + its GPU experts
    moe_gather_sum(p.dd, p.inv, p.w, N, k, E, p.shared, nullptr, extra, p.moe, st_);
}

// the logits of the group's last token (row 0 of the decode logits, last_T = 1)
void DsEngine::pf_head(int N) {
    const DsConfig & c = model_.ds;
    auto & p = pf_;
    ds_hc_pre(p.res.p + (size_t) (N - 1) * c.hc_dim(), pre_ + (size_t) (N - 1) * c.hc, mixed_, 1, c.hc, c.n_embd, st_);
    rmsnorm_rows(mixed_, F(model_.output_norm), xn_, 1, c.n_embd, c.n_embd, c.n_embd, c.eps, st_);
    quantize_act(xn_, c.n_embd, 1, c.n_embd, act_, st_);
    gemv(model_.output, &act_, nullptr, 0, 1, logits_, c.n_vocab, false, st_);
}

// ------------------------------------------------------------------------------------------------ a group
void DsEngine::prefill_group(const int32_t * tokens, int N) {
    const double t0 = now_ms();
    for (DsEngine * s = this; s; s = s->next_.get()) s->prefill_group_here(tokens, N, 0);
    last_T = 1;
    std::lock_guard<std::mutex> lk(stats_mu_);
    times.total_ms += now_ms() - t0;
}

void DsEngine::prefill_group_here(const int32_t * tokens, int N, int buf) {
    DeviceScope dsc(dev_);
    join_commit_here();
    const DsConfig & c = model_.ds;
    const int E = c.n_embd, HC = c.hc_dim();
    auto & p = pf_;
    if (N < 1 || N > pf_group_) throw std::runtime_error("prefill_group: bad size");
    if (cur_->pending_T) throw std::runtime_error("prefill_group: a verify window is pending");
    if (pos() + N > opt_.max_ctx) throw std::runtime_error("context full");
    const double t0 = now_ms();
    g_prof = PfProf{};
    { std::lock_guard<std::mutex> lk(head_->stats_mu_); pf_eg_ms_ = 0; }
    const int pos0 = pos();
    ensure_ctx_here(pos0 + N + kMaxWindow);
    ctx_grid_ = cur_->ctx_mapped;
    cur_->history.insert(cur_->history.end(), tokens, tokens + N);
    // the chunks: pf_max_ rows each, the remainder last
    const int nc = (N + pf_max_ - 1) / pf_max_;
    std::vector<int> csz(nc), coff(nc);
    for (int i = 0; i < nc; ++i) { coff[i] = i * pf_max_; csz[i] = std::min(pf_max_, N - coff[i]); }
    {
        std::vector<int32_t> ps((size_t) nc * p.n_sub, 0);
        for (int i = 0; i < nc; ++i) {
            p.h_par[i] = WinParams{pos0 + coff[i], csz[i], ++seq_, 0};
            for (int j = 0; j < p.n_sub; ++j) ps[(size_t) i * p.n_sub + j] = pos0 + coff[i] + j * p.score_rows;
        }
        CUDA_CHECK(cudaMemcpyAsync(p.d_par, p.h_par, (size_t) nc * sizeof(WinParams), cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaMemcpyAsync(p.pos.p, ps.data(), ps.size() * 4, cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));   // (pageable source)
    }
    // the Engram rows of each of this stage's Engram layers, gathered on a thread each while the earlier layers run
    std::vector<std::thread> eg_th(c.engram_layers.size());
    for (size_t e = 0; e < c.engram_layers.size(); ++e)
        if (owns(c.engram_layers[e]) && p.eg_rows) eg_th[e] = std::thread([this, N, e]() { pf_engram_gather(N, (int) e); });
    auto eg_join = [&](int il) {
        const int e = c.engram_index(il);
        if (e >= 0 && eg_th[e].joinable()) eg_th[e].join();
    };
    pre_ = p.pre_a;
    pre_next_ = p.pre_b;
    if (first()) {
        CUDA_CHECK(cudaMemcpyAsync(p.tok.p, tokens, (size_t) std::min(N, pf_max_) * 4, cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
        for (int i = 0; i < nc; ++i) {   // the token table rows, a chunk at a time through the chunk buffers
            if (i) {
                CUDA_CHECK(cudaMemcpyAsync(p.tok.p, tokens + coff[i], (size_t) csz[i] * 4, cudaMemcpyHostToDevice, st_));
                CUDA_CHECK(cudaStreamSynchronize(st_));
            }
            dequant_gather(model_.tok_embd, p.tok, csz[i], p.mixed, st_);
            hc_init(p.mixed, p.res.p + (size_t) coff[i] * HC, csz[i], c.hc, E, st_);
        }
        ds_onehot(pre_, N, c.hc, st_);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(p.res.p, pf_xfer_[buf], (size_t) N * HC * 4, cudaMemcpyHostToDevice, st_));
        CUDA_CHECK(cudaMemcpyAsync(pre_, pf_xfer_[buf] + (size_t) pf_group_ * HC, (size_t) N * c.hc * 4, cudaMemcpyHostToDevice, st_));
    }
    double tl = now_ms();
    if (g_pf_prof) g_prof.setup += tl - t0 - g_prof.engram_host + prof_lap(st_, tl);
    p.pre_layer = -1;
    // the first layer's first wave by rank, while its attention runs (BNK_DS_PF_PREFETCH buffers; the later layers'
    // prefetches follow their guessed routing)
    static const int rank_bufs = std::max(0, std::min(2, env_int("BNK_DS_PF_PREFETCH", 1)));
    if (!pf_rank_.empty()) pf_prefetch(l0_, pf_rank_[l0_], rank_bufs);
    for (int il = l0_; il < l1_; ++il) {
        const DsLayer & L = model_.layers[il];
        eg_join(il);
        for (int i = 0; i < nc; ++i) {
            const int n = csz[i];
            float * res_c = p.res.p + (size_t) coff[i] * HC, * pre_c = pre_ + (size_t) coff[i] * c.hc;
            if (c.engram_index(il) >= 0) pf_engram(il, i, n);
            ds_hc_mixes(res_c, F(L.hc_attn_fn), F(L.hc_attn_scale), F(L.hc_attn_base), p.attn_pre, p.postc, p.combc, n, c.hc, E, c.eps,
                        c.hc_eps, c.sinkhorn, st_);
            ds_hc_pre(res_c, pre_c, p.mixed, n, c.hc, E, st_);
            rmsnorm_rows(p.mixed, F(L.attn_norm), p.xnc, n, E, E, E, c.eps, st_);
            pf_attn(il, i, n, pos0 + coff[i]);
            ds_hc_post(p.out, res_c, p.postc, p.combc, n, c.hc, E, st_);
            ds_hc_mixes(res_c, F(L.hc_ffn_fn), F(L.hc_ffn_scale), F(L.hc_ffn_base), pre_next_ + (size_t) coff[i] * c.hc,
                        p.post.p + (size_t) coff[i] * c.hc, p.comb.p + (size_t) coff[i] * c.hc * c.hc, n, c.hc, E, c.eps, c.hc_eps,
                        c.sinkhorn, st_);
            ds_hc_pre(res_c, p.attn_pre, p.mixed, n, c.hc, E, st_);
            rmsnorm_rows(p.mixed, F(L.ffn_norm), p.xn.p + (size_t) coff[i] * E, n, E, E, E, c.eps, st_);
            pf_shared(il, i, n);
        }
        g_prof.attn += prof_lap(st_, tl);
        pf_moe(il, N);
        ds_hc_post(p.moe, p.res, p.post, p.comb, N, c.hc, E, st_);
        std::swap(pre_, pre_next_);
        g_prof.gather += prof_lap(st_, tl);
    }
    for (auto & t : eg_th) if (t.joinable()) t.join();
    if (last_) pf_head(N);
    else {
        CUDA_CHECK(cudaMemcpyAsync(pf_xfer_[buf], p.res.p, (size_t) N * HC * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaMemcpyAsync(pf_xfer_[buf] + (size_t) pf_group_ * HC, pre_, (size_t) N * c.hc * 4, cudaMemcpyDeviceToHost, st_));
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    g_prof.head += prof_lap(st_, tl);
    cur_->pending_T = 0;
    const double ms = now_ms() - t0;
    if (g_pf_prof)
        fprintf(stderr, "bnk: prompt group %d tokens, %d chunks, layers %d-%d: %.0f ms (%.0f tok/s); engram host (overlapped) %.0f, setup %.0f, "
                        "attention+dense %.0f, route %.0f, enqueue %.0f, cpu %.0f, gather/wait %.0f, head/out %.0f ms; %.0f MiB over PCIe "
                        "(%.1f us/MB), experts %lld resident %lld staged, pairs %lld gpu %lld cpu, cpu scale %.2f\n",
                N, nc, l0_, l1_ - 1, ms, N / (ms / 1000), (double) pf_eg_ms_, g_prof.setup, g_prof.attn, g_prof.route, g_prof.enqueue,
                g_prof.cpu, g_prof.gather, g_prof.head, g_prof.copy_mb, pf_pcie_us_mb_, (long long) g_prof.resident, (long long) g_prof.staged,
                (long long) g_prof.gpu_pairs, (long long) g_prof.cpu_pairs, head_->pf_cpu_scale_);
    std::lock_guard<std::mutex> lk(head_->stats_mu_);
    pf_ms_ += ms;
    pf_tokens_ += N;
}

// ------------------------------------------------------------------------------------------------ the prompt
void DsEngine::prefill(const std::vector<int32_t> & tokens) {
    const size_t n = tokens.size();
    if (n == 0) return;
    if (on_prefill_progress) on_prefill_progress(0, n);
    size_t i = 0;
    if (pf_max_ > 0) {
        // several stages: group k+1 on the first stage while the worker runs group k through the others
        static const bool pipe = !getenv("BNK_PIPE_PROMPT") || atoi(getenv("BNK_PIPE_PROMPT")) != 0;
        const bool pipeline = next_ && pipe;
        if (pipeline && !pf_worker_) pf_worker_ = std::make_unique<Worker>();
        // groups of even size (a read of a group and a bit would leave a tiny last group to overlap with)
        const size_t long_part = n >= (size_t) std::max(1, opt_.prefill_min) ? n : 0;
        size_t group = pf_group_;
        if (pipeline && long_part > (size_t) pf_group_) {
            const size_t parts = (long_part + pf_group_ - 1) / pf_group_;
            group = (long_part + parts - 1) / parts;
            group = std::min<size_t>(pf_group_, (group + 63) / 64 * 64);
        }
        bool posted = false;
        size_t handed = 0;
        int buf = 0;
        auto settle = [&]() {
            if (!posted) return;
            try { pf_worker_->wait(); } catch (...) {}
            posted = false;
        };
        try {
            while (n - i >= (size_t) std::max(1, opt_.prefill_min)) {
                const int N = (int) std::min(group, n - i);
                const int32_t * tk = tokens.data() + i;
                if (pipeline) {
                    const double c0 = now_ms();
                    prefill_group_here(tk, N, buf);
                    pf_worker_->wait();   // the previous group is through every stage
                    if (posted && on_prefill_progress) on_prefill_progress(handed, n);
                    const int b = buf;
                    pf_worker_->post([this, tk, N, b, c0]() {
                        for (DsEngine * s = next_.get(); s; s = s->next_.get()) s->prefill_group_here(tk, N, b);
                        std::lock_guard<std::mutex> lk(stats_mu_);
                        times.total_ms += now_ms() - c0;
                    });
                    posted = true;
                    buf ^= 1;
                } else {
                    prefill_group(tk, N);
                }
                i += N;
                handed = i;
                if (!pipeline && on_prefill_progress) on_prefill_progress(i, n);
            }
            if (posted) {
                pf_worker_->wait();
                posted = false;
                if (on_prefill_progress) on_prefill_progress(handed, n);
            }
        } catch (...) {
            settle();
            throw;
        }
        last_T = 1;
    }
    for (; i < n; i += kMaxWindow) {
        forward(tokens.data() + i, (int) std::min<size_t>(kMaxWindow, n - i), true);
        if (on_prefill_progress) on_prefill_progress(std::min(n, i + kMaxWindow), n);
    }
}

}  // namespace bnk
