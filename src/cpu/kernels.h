// bnk's own AVX2 dot products for formats ggml handles slowly on this CPU.
#pragma once

#include <cstddef>
#include <cstdint>

namespace bnk {

// Activations for Q2_0 rows, per 64 values: int8 codes in Q2_0's bit-plane order, one scale per 32.
// Layout q[64]: [plane0: half0 (elements 4i, i<8), half1 (32+4i)] [plane1 ...] [plane2 ...] [plane3 ...].
struct A8P64 {
    int8_t q[64];
    float d[2];
    float dsum;  // d0*sum(half0) + d1*sum(half1)
};

void quantize_a8p64(const float * x, int n, A8P64 * out);
// dot of one Q2_0 row (n/64 blocks of 18 bytes) with quantized activations
float dot_q2_0(const uint8_t * row, const A8P64 * a, int n);

}  // namespace bnk
