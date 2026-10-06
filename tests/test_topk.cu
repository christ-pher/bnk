// topk_rows (GPU) against a host reference: random rows with negatives, heavy ties at the boundary, and the K
// a request may ask for. Also times it against the host's full-row selection.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "core/util.h"
#include "kernels/ops.h"

using namespace bnk;

int main() {
    const int V = 248320, T = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 4.f);
    std::vector<float> x((size_t) T * V);
    for (auto & v : x) v = nd(rng);
    for (int i = 0; i < 3000; ++i) x[(size_t) 1 * V + rng() % V] = 9.5f;     // row 1: ties across the boundary
    for (int i = 0; i < V; ++i) x[(size_t) 2 * V + i] = -1.f - (i % 7);      // row 2: all negative, 7 values
    x[(size_t) 3 * V + 5] = 40.f;                                            // row 3: one dominant logit
    float * dx;
    int32_t * did;
    float * dv;
    CUDA_CHECK(cudaMalloc(&dx, x.size() * 4));
    CUDA_CHECK(cudaMalloc(&did, (size_t) T * 1024 * 4));
    CUDA_CHECK(cudaMalloc(&dv, (size_t) T * 1024 * 4));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    int bad = 0;
    for (int K : {1, 5, 20, 40, 100, 256, 1024}) {
        topk_rows(dx, T, V, K, did, dv, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int32_t> ids((size_t) T * K);
        std::vector<float> vals((size_t) T * K);
        CUDA_CHECK(cudaMemcpy(ids.data(), did, ids.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vals.data(), dv, vals.size() * 4, cudaMemcpyDeviceToHost));
        for (int t = 0; t < T; ++t) {
            const float * row = x.data() + (size_t) t * V;
            // reference: sort by (value desc, index asc), take K
            std::vector<int32_t> idx(V);
            for (int i = 0; i < V; ++i) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                              [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
            std::vector<int32_t> want(idx.begin(), idx.begin() + K), got(ids.begin() + (size_t) t * K, ids.begin() + (size_t) (t + 1) * K);
            std::sort(want.begin(), want.end());
            std::sort(got.begin(), got.end());
            bool ok = want == got;
            for (int j = 0; j < K && ok; ++j) ok = vals[(size_t) t * K + j] == row[ids[(size_t) t * K + j]];
            if (!ok) {
                ++bad;
                printf("MISMATCH K=%d row %d\n", K, t);
            }
        }
    }
    // speed: 4 rows, K = 20 (a typical verify window)
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 100; ++i) topk_rows(dx, 4, V, 20, did, dv, 0);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    printf("topk_rows: 4 rows x %d, K=20: %.3f ms per call\n", V, ms / 100);
    printf(bad ? "FAIL (%d)\n" : "PASS\n", bad);
    return bad != 0;
}
