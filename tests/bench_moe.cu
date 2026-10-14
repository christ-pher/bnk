// The routed experts of one decode window on the GPU (moe_hits: gate/up, activation, down), on a real layer's
// experts in the cache's R layout: time and bandwidth.
//   build/bench_moe MODEL.gguf [layer=3] [T=5] [k=10]
// T tokens each pick k of the layer's experts at random (as a verify window does), so most experts get one token.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "core/gguf_model.h"
#include "core/util.h"
#include "ggml.h"
#include "kernels/moe.h"
#include "kernels/rfmt.h"

using namespace bnk;

int main(int argc, char ** argv) {
    GgufModel g;
    g.open(argv[1]);
    const int il = argc > 2 ? atoi(argv[2]) : 3, T = argc > 3 ? atoi(argv[3]) : 5, k = argc > 4 ? atoi(argv[4]) : 10;
    const std::string b = "blk." + std::to_string(il) + ".";
    const TensorRef & gt = g.get(b + "ffn_gate_exps.weight");
    const TensorRef & ut = g.get(b + "ffn_up_exps.weight");
    const TensorRef & dt = g.get(b + "ffn_down_exps.weight");
    const int E = (int) gt.ne[0], F = (int) gt.ne[1], NE = 128;   // the first 128 experts are enough
    const size_t g_rb = ggml_row_size(gt.type, E), d_rb = ggml_row_size(dt.type, F);
    const size_t gsz = g_rb * F, dsz = d_rb * E, src_blob = 2 * gsz + dsz;
    std::vector<uint8_t> h(src_blob * NE);
    for (int e = 0; e < NE; ++e) {
        memcpy(h.data() + e * src_blob, gt.data + (size_t) e * gsz, gsz);
        memcpy(h.data() + e * src_blob + gsz, ut.data + (size_t) e * gsz, gsz);
        memcpy(h.data() + e * src_blob + 2 * gsz, dt.data + (size_t) e * dsz, dsz);
    }
    uint8_t * src;
    CUDA_CHECK(cudaMalloc(&src, h.size()));
    CUDA_CHECK(cudaMemcpy(src, h.data(), h.size(), cudaMemcpyHostToDevice));
    const RLayout gl = r_layout((int) gt.type, E), dl = r_layout((int) dt.type, F);
    const size_t blob = ((size_t) 2 * F * gl.row_bytes + (size_t) E * dl.row_bytes + 63) / 64 * 64;
    uint8_t * base;
    CUDA_CHECK(cudaMalloc(&base, blob * NE));
    std::vector<uint8_t *> dst(NE);
    for (int e = 0; e < NE; ++e) dst[e] = base + e * blob;
    uint8_t ** ddst;
    CUDA_CHECK(cudaMalloc(&ddst, NE * sizeof(void *)));
    CUDA_CHECK(cudaMemcpy(ddst, dst.data(), NE * sizeof(void *), cudaMemcpyHostToDevice));
    moe_repack_blobs(src, src_blob, ddst, NE, F, E, (int) gt.type, g_rb, (int) dt.type, d_rb, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

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
    for (int i = 0; i < 5; ++i) d.go[i] = g5[i], d.dof[i] = d5[i];

    // the window's hit list: each token picks k distinct experts
    HitList hl{};
    std::mt19937 rng(5);
    std::vector<int> ent_of(NE, -1);
    for (int t = 0; t < T; ++t) {
        std::set<int> picked;
        while ((int) picked.size() < k) picked.insert((int) (rng() % NE));
        for (int e : picked) {
            if (ent_of[e] < 0) {
                ent_of[e] = hl.n++;
                hl.e[ent_of[e]].expert = e;
                hl.e[ent_of[e]].slot = e;
            }
            HitEntry & he = hl.e[ent_of[e]];
            he.mask |= 1u << t;
            he.w[t] = 1.f / k;
        }
    }
    MoeScratch s;
    CUDA_CHECK(cudaMalloc(&s.hits, sizeof(HitList)));
    CUDA_CHECK(cudaMemcpy(s.hits, &hl, sizeof(HitList), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&s.gu, (size_t) kMaxRouted * kMaxWindow * 2 * F * 4));
    CUDA_CHECK(cudaMalloc(&s.hq, (size_t) kMaxRouted * kMaxWindow * F));
    CUDA_CHECK(cudaMalloc(&s.hd, (size_t) kMaxRouted * kMaxWindow * F / 32 * 4));
    CUDA_CHECK(cudaMalloc(&s.part, (size_t) kMaxRouted * kMaxWindow * E * 4));

    std::vector<float> hx((size_t) T * E);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : hx) v = nd(rng);
    float * dx;
    CUDA_CHECK(cudaMalloc(&dx, hx.size() * 4));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    ActQ8 a;
    CUDA_CHECK(cudaMalloc(&a.q, (size_t) kMaxWindow * E));
    CUDA_CHECK(cudaMalloc(&a.d, (size_t) kMaxWindow * E / 32 * 4));
    CUDA_CHECK(cudaMalloc(&a.s, (size_t) kMaxWindow * E / 16 * 2));
    CUDA_CHECK(cudaMalloc(&a.h, (size_t) kMaxWindow * E * 2));
    quantize_act(dx, E, T, E, a, 0);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int w = 0; w < 5; ++w) moe_hits(d, s, a, T, k, E, F, 0);
    const int reps = 100;
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) moe_hits(d, s, a, T, k, E, F, 0);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    const double us = ms * 1000 / reps;
    const double gub = 2.0 * F * gl.row_bytes, dnb = (double) E * dl.row_bytes;
    printf("layer %d (%s gate/up, %s down), T %d, k %d: %d experts hit, %.1f us, %.0f GB/s (%.2f MB of weights)\n", il,
           ggml_type_name(gt.type), ggml_type_name(dt.type), T, k, hl.n, us, hl.n * (gub + dnb) / (us * 1e3),
           hl.n * (gub + dnb) / 1e6);
    return 0;
}
