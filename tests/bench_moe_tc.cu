// Timing of the prompt path's tensor-core expert GEMM (moe_tc_gu + moe_tc_down) on a real layer's experts in the
// cache's R layout, routed as a prompt group routes them: N tokens x k slots over NE experts, uniform at random.
//   build/bench_moe_tc MODEL.gguf [layer=3] [tokens=8192] [experts=256] [k=6] [reps=10]
// Prints the time of each kernel and its rate in TFLOPS (2 flops a weight a token).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/gguf_model.h"
#include "core/util.h"
#include "ggml.h"
#include "kernels/moe.h"
#include "kernels/rfmt.h"

using namespace bnk;

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench_moe_tc MODEL.gguf [layer] [tokens] [experts] [k] [reps]\n"); return 1; }
    GgufModel g;
    g.open(argv[1]);
    const int il = argc > 2 ? atoi(argv[2]) : 3, N = argc > 3 ? atoi(argv[3]) : 8192, NE = argc > 4 ? atoi(argv[4]) : 256;
    const int k = argc > 5 ? atoi(argv[5]) : 6, reps = argc > 6 ? atoi(argv[6]) : 10;
    const std::string b = "blk." + std::to_string(il) + ".";
    const TensorRef & gt = g.get(b + "ffn_gate_exps.weight");
    const TensorRef & ut = g.get(b + "ffn_up_exps.weight");
    const TensorRef & dt = g.get(b + "ffn_down_exps.weight");
    const int E = (int) gt.ne[0], F = (int) gt.ne[1];
    const size_t g_rb = ggml_row_size(gt.type, E), d_rb = ggml_row_size(dt.type, F);
    const size_t gsz = g_rb * F, dsz = d_rb * E, src_blob = 2 * gsz + dsz;
    const RLayout gl = r_layout((int) gt.type, E), dl = r_layout((int) dt.type, F);
    const size_t blob = ((size_t) 2 * F * gl.row_bytes + (size_t) E * dl.row_bytes + 63) / 64 * 64;
    uint8_t * src, * base;
    CUDA_CHECK(cudaMalloc(&src, src_blob));
    CUDA_CHECK(cudaMalloc(&base, blob * NE));
    uint8_t ** ddst;
    CUDA_CHECK(cudaMalloc(&ddst, sizeof(void *)));
    std::vector<uint8_t> h(src_blob);
    for (int e = 0; e < NE; ++e) {   // one expert at a time through the source buffer
        memcpy(h.data(), gt.data + (size_t) e * gsz, gsz);
        memcpy(h.data() + gsz, ut.data + (size_t) e * gsz, gsz);
        memcpy(h.data() + 2 * gsz, dt.data + (size_t) e * dsz, dsz);
        CUDA_CHECK(cudaMemcpy(src, h.data(), src_blob, cudaMemcpyHostToDevice));
        uint8_t * dst = base + e * blob;
        CUDA_CHECK(cudaMemcpy(ddst, &dst, sizeof(void *), cudaMemcpyHostToDevice));
        moe_repack_blobs(src, src_blob, ddst, 1, F, E, (int) gt.type, g_rb, (int) dt.type, d_rb, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    MoeLayerDesc d;
    d.base = base;
    d.blob = blob;
    d.gate_bytes = d.up_bytes = (size_t) F * gl.row_bytes;
    d.gate_type = (int) gt.type;
    d.down_type = (int) dt.type;
    d.grow = gl.row_bytes;
    d.drow = dl.row_bytes;
    d.rlay = 1;
    const uint32_t g5[5] = {gl.off_a, gl.off_b, gl.off_c, gl.off_d, gl.off_e}, d5[5] = {dl.off_a, dl.off_b, dl.off_c, dl.off_d, dl.off_e};
    for (int i = 0; i < 5; ++i) { d.go[i] = g5[i]; d.dof[i] = d5[i]; }
    // routing: k distinct experts a token, uniform
    std::mt19937 rng(7);
    const int P = N * k;
    std::vector<int> ids(P);
    for (int t = 0; t < N; ++t)
        for (int s = 0; s < k; ++s) {
            int e;
            bool dup;
            do {
                e = (int) (rng() % NE);
                dup = false;
                for (int q = 0; q < s; ++q) dup = dup || ids[t * k + q] == e;
            } while (dup);
            ids[t * k + s] = e;
        }
    std::vector<int> cnt(NE, 0), off(NE + 1, 0);
    for (int i = 0; i < P; ++i) cnt[ids[i]]++;
    for (int e = 0; e < NE; ++e) off[e + 1] = off[e] + cnt[e];
    std::vector<int> fill(off.begin(), off.end() - 1), perm(P);
    for (int i = 0; i < P; ++i) perm[fill[ids[i]]++] = i / k;
    std::vector<TcItem> items;
    std::vector<TcBlock> blocks;
    for (int e = 0; e < NE; ++e) {
        if (!cnt[e]) continue;
        items.push_back(TcItem{base + e * blob, off[e], cnt[e], e});
        for (int t0 = 0; t0 < cnt[e]; t0 += kTcTile) blocks.push_back(TcBlock{(int) items.size() - 1, t0});
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<half> hx((size_t) N * E);
    for (auto & v : hx) v = __float2half(nd(rng));
    half * dx, * dh, * dD; int32_t * dperm; TcItem * dit; TcBlock * dbl;
    CUDA_CHECK(cudaMalloc(&dx, hx.size() * 2));
    CUDA_CHECK(cudaMalloc(&dperm, P * 4));
    CUDA_CHECK(cudaMalloc(&dh, (size_t) P * F * 2));
    CUDA_CHECK(cudaMalloc(&dD, (size_t) P * E * 2));
    CUDA_CHECK(cudaMalloc(&dit, items.size() * sizeof(TcItem)));
    CUDA_CHECK(cudaMalloc(&dbl, blocks.size() * sizeof(TcBlock)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dperm, perm.data(), P * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dit, items.data(), items.size() * sizeof(TcItem), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbl, blocks.data(), blocks.size() * sizeof(TcBlock), cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1, e2;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1)); CUDA_CHECK(cudaEventCreate(&e2));
    double tg = 0, td = 0;
    for (int r = 0; r <= reps; ++r) {   // rep 0 warms up
        CUDA_CHECK(cudaEventRecord(e0));
        moe_tc_gu(dit, dbl, (int) blocks.size(), d, dx, E, dperm, dh, E, F, 0);
        CUDA_CHECK(cudaEventRecord(e1));
        moe_tc_down(dit, dbl, (int) blocks.size(), d, dh, dD, E, F, 0);
        CUDA_CHECK(cudaEventRecord(e2));
        CUDA_CHECK(cudaEventSynchronize(e2));
        float a, c;
        CUDA_CHECK(cudaEventElapsedTime(&a, e0, e1));
        CUDA_CHECK(cudaEventElapsedTime(&c, e1, e2));
        if (r) { tg += a; td += c; }
    }
    tg /= reps; td /= reps;
    const double fl_g = 2.0 * P * 2.0 * F * E, fl_d = 2.0 * P * (double) F * E;
    printf("layer %d (%s gate/up, %s down, E %d F %d): %d tokens x %d over %d experts, %zu tiles\n", il, ggml_type_name(gt.type),
           ggml_type_name(dt.type), E, F, N, k, NE, blocks.size());
    printf("gate/up %.2f ms (%.1f TFLOPS), down %.2f ms (%.1f TFLOPS), total %.2f ms (%.1f TFLOPS)\n", tg, fl_g / tg / 1e9, td,
           fl_d / td / 1e9, tg + td, (fl_g + fl_d) / (tg + td) / 1e9);
    return 0;
}
