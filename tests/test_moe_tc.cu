// The prompt path's tensor-core expert GEMM (moe_tc_gu / moe_tc_down) against a host reference, on a real layer's
// experts in the cache's R layout: random fp16 activations routed to NE experts with uneven token counts (partial
// 64-token tiles, several tiles for the busiest), the pairs sorted by expert as pf_moe lays them out.
//   build/test_moe_tc MODEL.gguf [layer=3] [tokens=300] [experts=24]
// The reference rounds the weights, the input and h to fp16 as the kernel does and accumulates in fp64, so the
// remaining difference is the accumulation order; the fp32-weight reference shows the fp16 rounding itself.
#include <cmath>
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

static float h2f_host(half h) { return __half2float(h); }

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_moe_tc MODEL.gguf [layer] [tokens] [experts]\n"); return 1; }
    GgufModel g;
    g.open(argv[1]);
    const int il = argc > 2 ? atoi(argv[2]) : 3, N = argc > 3 ? atoi(argv[3]) : 300, NE = argc > 4 ? atoi(argv[4]) : 24;
    const std::string b = "blk." + std::to_string(il) + ".";
    const TensorRef & gt = g.get(b + "ffn_gate_exps.weight");
    const TensorRef & ut = g.get(b + "ffn_up_exps.weight");
    const TensorRef & dt = g.get(b + "ffn_down_exps.weight");
    const int E = (int) gt.ne[0], F = (int) gt.ne[1];
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
    for (int i = 0; i < 5; ++i) { d.go[i] = g5[i]; d.dof[i] = d5[i]; }

    // the experts' fp32 weights (ggml's own dequantization) for the reference
    std::vector<std::vector<float>> wg(NE), wu(NE), wd(NE);
    for (int e = 0; e < NE; ++e) {
        wg[e].resize((size_t) F * E); wu[e].resize((size_t) F * E); wd[e].resize((size_t) E * F);
        const auto * tr = ggml_get_type_traits(gt.type);
        const auto * trd = ggml_get_type_traits(dt.type);
        tr->to_float(gt.data + (size_t) e * gsz, wg[e].data(), (int64_t) F * E);
        tr->to_float(ut.data + (size_t) e * gsz, wu[e].data(), (int64_t) F * E);
        trd->to_float(dt.data + (size_t) e * dsz, wd[e].data(), (int64_t) E * F);
    }

    // routing: token t picks expert (t * 7 + t / 3) % NE, with the first expert getting a burst (several tiles)
    std::mt19937 rng(7);
    std::vector<int> expert_of(N);
    for (int t = 0; t < N; ++t) expert_of[t] = (t % 5 == 0 || t < 70) ? 0 : (int) (rng() % NE);
    std::vector<int> cnt(NE, 0), off(NE + 1, 0);
    for (int t = 0; t < N; ++t) cnt[expert_of[t]]++;
    for (int e = 0; e < NE; ++e) off[e + 1] = off[e] + cnt[e];
    std::vector<int> fill(off.begin(), off.end() - 1), perm(N);
    for (int t = 0; t < N; ++t) perm[fill[expert_of[t]]++] = t;   // pair p -> token
    std::vector<TcItem> items;
    std::vector<TcBlock> blocks;
    for (int e = 0; e < NE; ++e) {
        if (!cnt[e]) continue;
        items.push_back(TcItem{base + e * blob, off[e], cnt[e]});
        for (int t0 = 0; t0 < cnt[e]; t0 += kTcTile) blocks.push_back(TcBlock{(int) items.size() - 1, t0});
    }
    // random input, fp16 (as pf_moe converts mixed)
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<half> hx((size_t) N * E);
    std::vector<float> fx((size_t) N * E);
    for (size_t i = 0; i < hx.size(); ++i) { hx[i] = __float2half(nd(rng)); fx[i] = h2f_host(hx[i]); }
    half * dx; int32_t * dperm; half * dh; half * dD; TcItem * dit; TcBlock * dbl;
    CUDA_CHECK(cudaMalloc(&dx, hx.size() * 2));
    CUDA_CHECK(cudaMalloc(&dperm, N * 4));
    CUDA_CHECK(cudaMalloc(&dh, (size_t) N * F * 2));
    CUDA_CHECK(cudaMalloc(&dD, (size_t) N * E * 2));
    CUDA_CHECK(cudaMalloc(&dit, items.size() * sizeof(TcItem)));
    CUDA_CHECK(cudaMalloc(&dbl, blocks.size() * sizeof(TcBlock)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dperm, perm.data(), N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dit, items.data(), items.size() * sizeof(TcItem), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbl, blocks.data(), blocks.size() * sizeof(TcBlock), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dh, 0, (size_t) N * F * 2));
    CUDA_CHECK(cudaMemset(dD, 0, (size_t) N * E * 2));
    moe_tc_gu(dit, dbl, (int) blocks.size(), d, dx, E, dperm, dh, E, F, 0);
    moe_tc_down(dit, dbl, (int) blocks.size(), d, dh, dD, E, F, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> outh((size_t) N * F), outD((size_t) N * E);
    CUDA_CHECK(cudaMemcpy(outh.data(), dh, outh.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outD.data(), dD, outD.size() * 2, cudaMemcpyDeviceToHost));

    // references per pair: fp16-rounded weights (as the kernel feeds the tensor cores) and fp32 weights
    double eh16 = 0, eh32 = 0, nh = 0, ed16 = 0, ed32 = 0, ndn = 0, mh16 = 0, md16 = 0;
    int bad = 0;
    std::vector<float> hr16(F), hr32(F), x16(E);
    for (int p = 0; p < N; ++p) {
        int e = 0;
        while (off[e + 1] <= p) ++e;
        const int t = perm[p];
        for (int i = 0; i < E; ++i) x16[i] = fx[(size_t) t * E + i];
        for (int f = 0; f < F; ++f) {
            double g16 = 0, u16 = 0, g32 = 0, u32 = 0;
            for (int i = 0; i < E; ++i) {
                const float gw = wg[e][(size_t) f * E + i], uw = wu[e][(size_t) f * E + i];
                g32 += (double) gw * x16[i];
                u32 += (double) uw * x16[i];
                g16 += (double) h2f_host(__float2half(gw)) * x16[i];
                u16 += (double) h2f_host(__float2half(uw)) * x16[i];
            }
            hr16[f] = (float) (g16 / (1 + exp(-g16)) * u16);
            hr32[f] = (float) (g32 / (1 + exp(-g32)) * u32);
            const float got = h2f_host(outh[(size_t) p * F + f]);
            eh16 += (got - hr16[f]) * (double) (got - hr16[f]);
            eh32 += (got - hr32[f]) * (double) (got - hr32[f]);
            nh += (double) hr32[f] * hr32[f];
            mh16 = fmax(mh16, fabs(got - hr16[f]) / (fabs(hr16[f]) + 1e-2));
            if (!std::isfinite(got)) ++bad;
        }
        // the down reference reads h as the kernel wrote it (fp16), so it checks the down product alone
        for (int i = 0; i < E; ++i) {
            double y16 = 0, y32 = 0;
            for (int f = 0; f < F; ++f) {
                const float hv = h2f_host(outh[(size_t) p * F + f]), dw = wd[e][(size_t) i * F + f];
                y32 += (double) dw * hv;
                y16 += (double) h2f_host(__float2half(dw)) * hv;
            }
            const float got = h2f_host(outD[(size_t) p * E + i]);
            ed16 += (got - y16) * (got - y16);
            ed32 += (got - y32) * (got - y32);
            ndn += y32 * y32;
            md16 = fmax(md16, fabs(got - y16) / (fabs(y16) + 1e-2));
            if (!std::isfinite(got)) ++bad;
        }
    }
    printf("layer %d (%s gate/up, %s down), %d tokens over %d experts, %zu tiles (expert 0: %d tokens)\n", il,
           ggml_type_name(gt.type), ggml_type_name(dt.type), N, NE, blocks.size(), cnt[0]);
    printf("h = silu(gate x) up:  rel rms error vs fp16-weight reference %.2e, vs fp32-weight reference %.2e, max rel %.2e\n",
           sqrt(eh16 / nh), sqrt(eh32 / nh), mh16);
    printf("D = down h:           rel rms error vs fp16-weight reference %.2e, vs fp32-weight reference %.2e, max rel %.2e\n",
           sqrt(ed16 / ndn), sqrt(ed32 / ndn), md16);
    printf("%s\n", bad ? "FAIL: non-finite outputs" : (sqrt(eh16 / nh) < 2e-3 && sqrt(ed16 / ndn) < 2e-3 ? "PASS" : "FAIL: error too large"));
    return bad || !(sqrt(eh16 / nh) < 2e-3 && sqrt(ed16 / ndn) < 2e-3);
}
