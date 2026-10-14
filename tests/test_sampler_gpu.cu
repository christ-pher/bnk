// Sampling without top-k from the GPU's candidates: topk_rows (1,024 per row) and row_expsum (the normalizer), then
// Sampler::distribution_head, against Sampler::distribution over the whole row on the host. Where the head path
// answers, the same tokens in the same order with the same probabilities; where min-p / top-p keep more than 1,024
// it must say so (false), never answer differently.
//   build/test_sampler_gpu
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/util.h"
#include "engine/sampler.h"
#include "kernels/ops.h"

using namespace bnk;

int main() {
    const int V = 248320, T = 8, K = 1024;
    std::mt19937 rng(11);
    std::vector<float> x((size_t) T * V);
    float * dx;
    int32_t * did;
    float * dv;
    double * dz;
    CUDA_CHECK(cudaMalloc(&dx, x.size() * 4));
    CUDA_CHECK(cudaMalloc(&did, (size_t) T * K * 4));
    CUDA_CHECK(cudaMalloc(&dv, (size_t) T * K * 4));
    CUDA_CHECK(cudaMalloc(&dz, (size_t) T * 8));
    Sampler smp;
    std::vector<std::pair<int32_t, float>> want, got;
    const std::vector<int32_t> recent;
    int cases = 0, answered = 0, bad = 0;
    for (int rep = 0; rep < 6; ++rep) {
        for (int t = 0; t < T; ++t) {   // flat to peaked rows, one with ties
            std::normal_distribution<float> nd(0.f, 0.5f + t + rep);
            float * row = x.data() + (size_t) t * V;
            for (int i = 0; i < V; ++i) row[i] = nd(rng);
            if (t % 2) row[rng() % V] += 8.f + 3 * t;
            if (t == 6)
                for (int i = 0; i < V; ++i) row[i] = std::round(row[i] * 2.f) / 2.f;
        }
        CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        topk_rows(dx, T, V, K, did, dv, 0);
        for (float temp : {0.6f, 1.0f, 1.5f}) {
            row_expsum(dx, T, V, temp, dz, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<int32_t> ids((size_t) T * K);
            std::vector<float> vals((size_t) T * K);
            std::vector<double> z(T);
            CUDA_CHECK(cudaMemcpy(ids.data(), did, ids.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(vals.data(), dv, vals.size() * 4, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(z.data(), dz, T * 8, cudaMemcpyDeviceToHost));
            for (int t = 0; t < T; ++t)
                for (float top_p : {1.0f, 0.95f, 0.8f, 0.99f})
                    for (float min_p : {0.0f, 0.05f, 0.01f}) {
                        SamplingParams sp;
                        sp.temperature = temp;
                        sp.top_p = top_p;
                        sp.min_p = min_p;
                        smp.distribution(x.data() + (size_t) t * V, V, sp, recent, want);
                        ++cases;
                        const bool ok = smp.distribution_head(ids.data() + (size_t) t * K, vals.data() + (size_t) t * K,
                                                              K, z[t], sp, got);
                        if (!ok) {
                            if (want.size() < (size_t) K / 2) {   // it fell back where the cut was well inside
                                if (++bad <= 10)
                                    printf("FALLBACK rep %d row %d temp %.1f top_p %.2f min_p %.2f: host keeps %zu\n", rep,
                                           t, temp, top_p, min_p, want.size());
                            }
                            continue;
                        }
                        ++answered;
                        if (top_p == 1.0f && min_p == 0.0f) {   // nothing cut: the host's are in no particular order
                            std::sort(want.begin(), want.end());
                            std::sort(got.begin(), got.end());
                        }
                        bool same = want.size() == got.size();
                        float err = 0;
                        for (size_t i = 0; same && i < want.size(); ++i) {
                            same = want[i].first == got[i].first;
                            err = std::max(err, std::fabs(want[i].second - got[i].second));
                        }
                        if (!same || err > 1e-5f) {
                            if (++bad <= 10)
                                printf("MISMATCH rep %d row %d temp %.1f top_p %.2f min_p %.2f: %zu vs %zu tokens, err %g\n",
                                       rep, t, temp, top_p, min_p, want.size(), got.size(), err);
                        }
                    }
        }
    }
    printf("%d cases, %d answered from the GPU's candidates, %d bad: %s\n", cases, answered, bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
