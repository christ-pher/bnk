// Host sampling cost per row at the model's vocabulary size (CPU only): what each verify row pays today.
//   build/bench_sampler [n_vocab=248320] [top_k=20]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "engine/sampler.h"

using namespace bnk;

int main(int argc, char ** argv) {
    const int V = argc > 1 ? atoi(argv[1]) : 248320;
    SamplingParams sp;
    sp.temperature = 1.0f;
    sp.top_k = argc > 2 ? atoi(argv[2]) : 20;
    sp.top_p = 0.95f;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 3.f);
    std::vector<float> logits(V);
    for (auto & x : logits) x = nd(rng);
    logits[1234] = 25.f;   // a confident model: one clear favourite
    Sampler s;
    std::vector<std::pair<int32_t, float>> dist;
    std::vector<int32_t> recent;
    for (int i = 0; i < 20; ++i) s.distribution(logits.data(), V, sp, recent, dist);   // warm up
    const int reps = 200;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) s.distribution(logits.data(), V, sp, recent, dist);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    printf("vocab %d, top_k %d: %.3f ms per row (%zu candidates)\n", V, sp.top_k, ms, dist.size());
    return 0;
}
