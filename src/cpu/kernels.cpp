#include "cpu/kernels.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

#include "ggml.h"

namespace bnk {

void quantize_a8p64(const float * x, int n, A8P64 * out) {
    for (int b = 0; b < n / 64; ++b) {
        const float * xb = x + b * 64;
        A8P64 & o = out[b];
        o.dsum = 0.f;
        for (int h = 0; h < 2; ++h) {
            float amax = 0.f;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xb[32 * h + j]));
            const float d = amax / 127.f;
            const float id = d > 0.f ? 1.f / d : 0.f;
            int32_t sum = 0;
            for (int k = 0; k < 4; ++k)
                for (int i = 0; i < 8; ++i) {
                    const int8_t q = (int8_t) lrintf(xb[32 * h + 4 * i + k] * id);
                    o.q[16 * k + 8 * h + i] = q;
                    sum += q;
                }
            o.d[h] = d;
            o.dsum += d * (float) sum;
        }
    }
}

// Q2_0 block: fp16 d, 16 bytes of 2-bit codes (element j: byte j/4, bits 2*(j%4)); value = (code - 1) * d.
// After the plane shifts, int32 lane pairs (0,1) (2,3) (4,5) (6,7) hold halves 0,1,0,1.
float dot_q2_0(const uint8_t * row, const A8P64 * a, int n) {
    const int nb = n / 64;
    const __m256i shift01 = _mm256_setr_epi64x(0, 0, 2, 2);
    const __m256i shift23 = _mm256_setr_epi64x(4, 4, 6, 6);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    float tail = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t * blk = row + b * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float dw = ggml_fp16_to_fp32(dh);
        __m128i q16;
        memcpy(&q16, blk + 2, 16);
        const __m256i qq = _mm256_broadcastsi128_si256(q16);
        const __m256i u01 = _mm256_and_si256(_mm256_srlv_epi64(qq, shift01), m3);
        const __m256i u23 = _mm256_and_si256(_mm256_srlv_epi64(qq, shift23), m3);
        const __m256i a01 = _mm256_loadu_si256((const __m256i *) a[b].q);
        const __m256i a23 = _mm256_loadu_si256((const __m256i *) (a[b].q + 32));
        const __m256i p = _mm256_add_epi16(_mm256_maddubs_epi16(u01, a01), _mm256_maddubs_epi16(u23, a23));
        const __m256i s32 = _mm256_madd_epi16(p, ones16);
        const float s0 = dw * a[b].d[0], s1 = dw * a[b].d[1];
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s32), _mm256_setr_ps(s0, s0, s1, s1, s0, s0, s1, s1), acc);
        tail -= dw * a[b].dsum;
    }
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s) + tail;
}

}  // namespace bnk
