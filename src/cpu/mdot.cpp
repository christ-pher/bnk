// Multi-token AVX2 dot products: one decode of a weight row serves up to 4 activation rows.
// Adapted from ggml's AVX2 vec_dot kernels (ggml-cpu/arch/x86/quants.c, MIT): the weight decode (grid and sign
// lookups) is hoisted out of the per-token loop, which is what a verify window's shared experts need.
#include "cpu/mdot.h"

#include <immintrin.h>

#include <cstdlib>
#include <cstring>
#include <mutex>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-impl.h"

namespace bnk {

namespace {

inline float hsum8(__m256 x) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
inline float fp16(uint16_t h) { return _cvtsh_ss(h); }

// ggml's keven_signs_q2xs: for a 7-bit sign index, 8 int8 multipliers (+1/-1); the 8th sign makes the count even
struct EvenSigns {
    int8_t v[128][8];
    EvenSigns() {
        for (int i = 0; i < 128; ++i) {
            const int bits = i | ((__builtin_popcount(i) & 1) << 7);
            for (int j = 0; j < 8; ++j) v[i][j] = (bits >> j) & 1 ? -1 : 1;
        }
    }
};
const EvenSigns keven_signs;

const uint8_t k_mask1[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
const uint8_t k_mask2[32] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
                             1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};

// signed activations: (q8 ^ s) - s negates where the sign mask s is all ones
inline __m256i apply_sign(__m256i q8, __m256i s) { return _mm256_sub_epi8(_mm256_xor_si256(s, q8), s); }
inline __m256i sign_mask16(const uint16_t * signs, __m256i mask1, __m256i mask2) {
    __m256i a = _mm256_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
    a = _mm256_and_si256(_mm256_shuffle_epi8(a, mask1), mask2);
    return _mm256_cmpeq_epi8(a, mask2);
}

// ---------------------------------------------------------------------------------------------- IQ3_XXS
template <int NT>
void iq3_xxs_rows(const block_iq3_xxs * x, int nb, const block_q8_K * const * y, float * s) {
    const uint64_t * signs64 = (const uint64_t *) keven_signs.v;
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    uint32_t aux32[2];
    for (int i = 0; i < nb; ++i) {
        const float dw = fp16(x[i].d);
        const uint8_t * q3 = x[i].qs;
        const uint8_t * gas = x[i].qs + QK_K / 4;
        __m256i sumi[NT];
        for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q2_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            const __m256i q2_2 = _mm256_set_epi32(iq3xxs_grid[q3[15]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[12]],
                                                  iq3xxs_grid[q3[11]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[9]], iq3xxs_grid[q3[8]]);
            q3 += 16;
            memcpy(aux32, gas, 8);
            gas += 8;
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                   signs64[(aux32[0] >> 7) & 127], signs64[(aux32[0] >> 0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
            const __m256i sc1 = _mm256_set1_epi16(2 * (aux32[0] >> 28) + 1);
            const __m256i sc2 = _mm256_set1_epi16(2 * (aux32[1] >> 28) + 1);
            for (int t = 0; t < NT; ++t) {
                const int8_t * q8 = y[t][i].qs + ib32 * 32;
                const __m256i a1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) q8), s2_1);
                const __m256i a2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) (q8 + 32)), s2_2);
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, a1), sc1);
                const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, a2), sc2);
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p1, p2));
            }
        }
        for (int t = 0; t < NT; ++t)
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
    }
    for (int t = 0; t < NT; ++t) s[t] = 0.25f * hsum8(acc[t]);
}

// ---------------------------------------------------------------------------------------------- IQ3_S
template <int NT>
void iq3_s_rows(const block_iq3_s * x, int nb, const block_q8_K * const * y, float * s) {
    const __m256i mask1 = _mm256_loadu_si256((const __m256i *) k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i *) k_mask2);
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask = _mm256_set1_epi32(256);
    union { __m256i vec[2]; uint32_t index[16]; } idx;
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float dw = fp16(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint16_t * signs = (const uint16_t *) x[i].signs;
        __m256i sumi[NT];
        for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *) qs));
            qs += 16;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32 + 0]);
            idx.vec[1] = _mm256_set1_epi32(qh[ib32 + 1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
            const __m256i q2_1 = _mm256_set_epi32(iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]],
                                                  iq3s_grid[idx.index[4]], iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]],
                                                  iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]);
            const __m256i q2_2 = _mm256_set_epi32(iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]],
                                                  iq3s_grid[idx.index[12]], iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]],
                                                  iq3s_grid[idx.index[9]], iq3s_grid[idx.index[8]]);
            const __m256i s2_1 = sign_mask16(signs, mask1, mask2);
            const __m256i s2_2 = sign_mask16(signs + 2, mask1, mask2);
            signs += 4;
            const uint16_t ls1 = x[i].scales[ib32 / 2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32 / 2] >> 4;
            const __m256i sc1 = _mm256_set1_epi16(2 * ls1 + 1), sc2 = _mm256_set1_epi16(2 * ls2 + 1);
            for (int t = 0; t < NT; ++t) {
                const int8_t * q8 = y[t][i].qs + ib32 * 32;
                const __m256i a1 = apply_sign(_mm256_loadu_si256((const __m256i *) q8), s2_1);
                const __m256i a2 = apply_sign(_mm256_loadu_si256((const __m256i *) (q8 + 32)), s2_2);
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, a1), sc1);
                const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, a2), sc2);
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p1, p2));
            }
        }
        for (int t = 0; t < NT; ++t)
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
    }
    for (int t = 0; t < NT; ++t) s[t] = hsum8(acc[t]);
}

// ---------------------------------------------------------------------------------------------- IQ2_S
template <int NT>
void iq2_s_rows(const block_iq2_s * x, int nb, const block_q8_K * const * y, float * s) {
    const __m256i mask1 = _mm256_loadu_si256((const __m256i *) k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i *) k_mask2);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float dw = fp16(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint16_t * signs = (const uint16_t *) (x[i].qs + QK_K / 8);
        __m256i sumi[NT];
        for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32 + 0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32 + 0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32 + 0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32 + 0] << 8) & 0x300)]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32 + 1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32 + 1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32 + 1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32 + 1] << 8) & 0x300)]);
            qs += 8;
            const __m256i s2_1 = sign_mask16(signs, mask1, mask2);
            const __m256i s2_2 = sign_mask16(signs + 2, mask1, mask2);
            signs += 4;
            // per-16 scales: 2*(scale nibble)+1 (the 0.25 * (0.5 + s) of the reference, scaled by 8)
            const int s0 = x[i].scales[ib32] & 0xf, s1 = x[i].scales[ib32] >> 4;
            const int s2 = x[i].scales[ib32 + 1] & 0xf, s3 = x[i].scales[ib32 + 1] >> 4;
            const __m256i sc1 = _mm256_setr_epi16(2 * s0 + 1, 2 * s0 + 1, 2 * s0 + 1, 2 * s0 + 1, 2 * s0 + 1, 2 * s0 + 1, 2 * s0 + 1, 2 * s0 + 1,
                                                  2 * s1 + 1, 2 * s1 + 1, 2 * s1 + 1, 2 * s1 + 1, 2 * s1 + 1, 2 * s1 + 1, 2 * s1 + 1, 2 * s1 + 1);
            const __m256i sc2 = _mm256_setr_epi16(2 * s2 + 1, 2 * s2 + 1, 2 * s2 + 1, 2 * s2 + 1, 2 * s2 + 1, 2 * s2 + 1, 2 * s2 + 1, 2 * s2 + 1,
                                                  2 * s3 + 1, 2 * s3 + 1, 2 * s3 + 1, 2 * s3 + 1, 2 * s3 + 1, 2 * s3 + 1, 2 * s3 + 1, 2 * s3 + 1);
            for (int t = 0; t < NT; ++t) {
                const int8_t * q8 = y[t][i].qs + ib32 * 32;
                const __m256i a1 = apply_sign(_mm256_loadu_si256((const __m256i *) q8), s2_1);
                const __m256i a2 = apply_sign(_mm256_loadu_si256((const __m256i *) (q8 + 32)), s2_2);
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, a1), sc1);
                const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, a2), sc2);
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p1, p2));
            }
        }
        for (int t = 0; t < NT; ++t)
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
    }
    for (int t = 0; t < NT; ++t) s[t] = 0.125f * hsum8(acc[t]);
}

// ---------------------------------------------------------------------------------------------- IQ2_XXS
// Mirrors ggml_vec_dot_iq2_xxs_q8_K (AVX2): grid from iq2xxs_grid, signs from keven_signs applied to the
// activations, one 4-bit scale (2*ls+1) per 32-block; the per-32 weight decode is hoisted out of the token loop.
template <int NT>
void iq2_xxs_rows(const block_iq2_xxs * x, int nb, const block_q8_K * const * y, float * s) {
    const uint64_t * signs64 = (const uint64_t *) keven_signs.v;
    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *) aux32;
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float dw = fp16(x[i].d);
        const uint16_t * q2 = x[i].qs;
        __m256i sumi[NT];
        for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            memcpy(aux32, q2, 4 * sizeof(uint32_t));
            q2 += 8;
            const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]],
                                                   iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]],
                                                   iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
            const __m256i sc1 = _mm256_set1_epi16(2 * (aux32[1] >> 28) + 1);
            const __m256i sc2 = _mm256_set1_epi16(2 * (aux32[3] >> 28) + 1);
            for (int t = 0; t < NT; ++t) {
                const int8_t * q8 = y[t][i].qs + ib32 * 32;
                const __m256i a1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) q8), s2_1);
                const __m256i a2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) (q8 + 32)), s2_2);
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, a1), sc1);
                const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, a2), sc2);
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p1, p2));
            }
        }
        for (int t = 0; t < NT; ++t)
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
    }
    for (int t = 0; t < NT; ++t) s[t] = 0.125f * hsum8(acc[t]);
}

// The same rows through one table: the grid entry with its signs already applied, for every (grid index, sign index)
// pair at g << 7 | i (256 KB, built on first use), so a 32-value chunk costs 4 table loads instead of 8 and no sign
// pass per token: with q8 + 128 as the unsigned operand of vpmaddubsw, the product is (q8 + 128) . signed_grid, and
// 128 . signed_grid (the chunk's pair sums, s1 / s2) is taken off - the same integers as ggml's |grid| . signed q8, so
// the same bits. After Project Maya's ds4_iq_cpu.cpp (MIT). Measured slower on an EPYC 7402 (Zen 2, 512 KB L2):
// 0.44 vs 0.34 ms per expert at one token (bench_ds_cpu, layer 3), the table and the weight stream sharing L2, so it
// is off unless BNK_CPU_XXS_CT=1 (a CPU with a bigger L2 may differ).
struct SignedGrid {
    alignas(64) uint64_t v[256 * 128];
    SignedGrid() {
        for (int g = 0; g < 256; ++g)
            for (int i = 0; i < 128; ++i) {
                const int8_t * sg = keven_signs.v[i];
                uint64_t w = 0;
                for (int j = 0; j < 8; ++j) {
                    const int gb = (int) ((iq2xxs_grid[g] >> (8 * j)) & 255u);
                    w |= (uint64_t) (uint8_t) (int8_t) (sg[j] < 0 ? -gb : gb) << (8 * j);
                }
                v[(size_t) g << 7 | (size_t) i] = w;
            }
    }
};
const SignedGrid & signed_grid() {
    static const SignedGrid t;
    return t;
}
inline __m256i bq(const void * p) { return _mm256_castpd_si256(_mm256_broadcast_sd((const double *) p)); }
inline __m256i tab4(const uint64_t * t, unsigned a, unsigned b, unsigned c, unsigned d) {   // qwords t[a] t[b] t[c] t[d]
    return _mm256_blend_epi32(_mm256_blend_epi32(bq(t + a), bq(t + b), 0x0C), _mm256_blend_epi32(bq(t + c), bq(t + d), 0xC0), 0xF0);
}
template <int NT>
void iq2_xxs_rows_ct(const block_iq2_xxs * x, int nb, const block_q8_K * const * y, float * s) {
    const uint64_t * ct = signed_grid().v;
    const __m256i k80 = _mm256_set1_epi8((char) 0x80);
    uint32_t aux32[4];
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float dw = fp16(x[i].d);
        const uint16_t * q2 = x[i].qs;
        __m256i sumi[NT];
        for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            memcpy(aux32, q2, 4 * sizeof(uint32_t));
            q2 += 8;
            const uint32_t a0 = aux32[0], a1 = aux32[1], b0 = aux32[2], b1 = aux32[3];
            const __m256i g1 = tab4(ct, (a0 & 255u) << 7 | (a1 & 127u), ((a0 >> 8) & 255u) << 7 | ((a1 >> 7) & 127u),
                                    ((a0 >> 16) & 255u) << 7 | ((a1 >> 14) & 127u), (a0 >> 24) << 7 | ((a1 >> 21) & 127u));
            const __m256i g2 = tab4(ct, (b0 & 255u) << 7 | (b1 & 127u), ((b0 >> 8) & 255u) << 7 | ((b1 >> 7) & 127u),
                                    ((b0 >> 16) & 255u) << 7 | ((b1 >> 14) & 127u), (b0 >> 24) << 7 | ((b1 >> 21) & 127u));
            const __m256i s1 = _mm256_maddubs_epi16(k80, g1), s2 = _mm256_maddubs_epi16(k80, g2);
            const __m256i sc1 = _mm256_set1_epi16((short) (2 * (a1 >> 28) + 1)), sc2 = _mm256_set1_epi16((short) (2 * (b1 >> 28) + 1));
            for (int t = 0; t < NT; ++t) {
                const int8_t * q8 = y[t][i].qs + ib32 * 32;
                const __m256i u1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *) q8), k80);
                const __m256i u2 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *) (q8 + 32)), k80);
                const __m256i d1 = _mm256_sub_epi16(_mm256_maddubs_epi16(u1, g1), s1);
                const __m256i d2 = _mm256_sub_epi16(_mm256_maddubs_epi16(u2, g2), s2);
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(_mm256_madd_epi16(d1, sc1), _mm256_madd_epi16(d2, sc2)));
            }
        }
        for (int t = 0; t < NT; ++t)
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
    }
    for (int t = 0; t < NT; ++t) s[t] = 0.125f * hsum8(acc[t]);
}
bool xxs_ct_on() {
    static const bool on = getenv("BNK_CPU_XXS_CT") && atoi(getenv("BNK_CPU_XXS_CT")) != 0;
    return on;
}

// ---------------------------------------------------------------------------------------------- IQ2_XS
// Mirrors dequantize/vec_dot of iq2_xs: 9-bit grid index and 7-bit sign index packed per uint16, two 4-bit
// per-16 scales per 32-block. Processed one 32-block at a time (grid+signs hoisted out of the token loop).
template <int NT>
void iq2_xs_rows(const block_iq2_xs * x, int nb, const block_q8_K * const * y, float * s) {
    const uint64_t * signs64 = (const uint64_t *) keven_signs.v;
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float dw = fp16(x[i].d);
        const uint16_t * q2 = x[i].qs;
        __m256i sumi[NT];
        for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const __m256i q2v = _mm256_set_epi64x(iq2xs_grid[q2[3] & 511], iq2xs_grid[q2[2] & 511],
                                                  iq2xs_grid[q2[1] & 511], iq2xs_grid[q2[0] & 511]);
            const __m256i s2 = _mm256_set_epi64x(signs64[q2[3] >> 9], signs64[q2[2] >> 9],
                                                 signs64[q2[1] >> 9], signs64[q2[0] >> 9]);
            q2 += 4;
            const int slo = x[i].scales[ib32] & 0xf, shi = x[i].scales[ib32] >> 4;
            const __m256i sc = _mm256_setr_epi16(2 * slo + 1, 2 * slo + 1, 2 * slo + 1, 2 * slo + 1,
                                                 2 * slo + 1, 2 * slo + 1, 2 * slo + 1, 2 * slo + 1,
                                                 2 * shi + 1, 2 * shi + 1, 2 * shi + 1, 2 * shi + 1,
                                                 2 * shi + 1, 2 * shi + 1, 2 * shi + 1, 2 * shi + 1);
            for (int t = 0; t < NT; ++t) {
                const int8_t * q8 = y[t][i].qs + ib32 * 32;
                const __m256i a = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) q8), s2);
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(q2v, a), sc);
                sumi[t] = _mm256_add_epi32(sumi[t], p);
            }
        }
        for (int t = 0; t < NT; ++t)
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
    }
    for (int t = 0; t < NT; ++t) s[t] = 0.125f * hsum8(acc[t]);
}

// ---------------------------------------------------------------------------------------------- IQ4_XS
template <int NT>
void iq4_xs_rows(const block_iq4_xs * x, int nb, const block_q8_K * const * y, float * s) {
    const __m128i values128 = _mm_loadu_si128((const __m128i *) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        const float dw = fp16(x[ibl].d);
        __m256i sumi[NT];
        for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const __m128i b1 = _mm_loadu_si128((const __m128i *) qs);
            const __m128i b2 = _mm_loadu_si128((const __m128i *) (qs + 16));
            qs += 32;
            const __m256i w1 = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(b1, 4), m4b)),
                                                _mm_shuffle_epi8(values128, _mm_and_si128(b1, m4b)));
            const __m256i w2 = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(b2, 4), m4b)),
                                                _mm_shuffle_epi8(values128, _mm_and_si128(b2, m4b)));
            const __m256i aw1 = _mm256_sign_epi8(w1, w1), aw2 = _mm256_sign_epi8(w2, w2);
            const int16_t ls1 = ((x[ibl].scales_l[ib / 2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((x[ibl].scales_l[ib / 2] >> 4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            const __m256i sc1 = _mm256_set1_epi16(ls1), sc2 = _mm256_set1_epi16(ls2);
            for (int t = 0; t < NT; ++t) {
                const int8_t * q8 = y[t][ibl].qs + ib * 32;
                const __m256i a1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) q8), w1);
                const __m256i a2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) (q8 + 32)), w2);
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw1, a1), sc1);
                const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw2, a2), sc2);
                sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p1, p2));
            }
        }
        for (int t = 0; t < NT; ++t)
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * y[t][ibl].d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
    }
    for (int t = 0; t < NT; ++t) s[t] = hsum8(acc[t]);
}

// ---------------------------------------------------------------------------------------------- Q5_K x Q8_K
// Same integer and float operations as ggml_vec_dot_q5_K_q8_K (AVX2), per token, so results are bit-identical.
template <int NT>
void q5_k_rows(const block_q5_K * x, int nb, const block_q8_K * const * y, float * s) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    __m256 acc[NT];
    float summs[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); summs[t] = 0.f; }
    uint32_t utmp[4];
    for (int i = 0; i < nb; ++i) {
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0f) | (((utmp[1] >> 6) & 0x03030303) << 4);
        const uint32_t uaux = utmp[1] & 0x3f3f3f3f;
        utmp[1] = (utmp[2] & 0x0f0f0f0f) | (((utmp[0] >> 6) & 0x03030303) << 4);
        utmp[2] = uaux;
        utmp[0] &= 0x3f3f3f3f;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const uint8_t * sc8 = (const uint8_t *) utmp;  // the 8 sub-block scales
        // decode the 256 weights (8 x 32 unsigned 5-bit values) once
        const __m256i hbits = _mm256_loadu_si256((const __m256i *) x[i].qh);
        __m256i q5[8];
        for (int j = 0; j < 4; ++j) {
            const __m256i q5bits = _mm256_loadu_si256((const __m256i *) (x[i].qs + 32 * j));
            const __m256i h0 = _mm256_and_si256(hbits, _mm256_set1_epi8((char) (1 << (2 * j))));
            const __m256i h1 = _mm256_and_si256(hbits, _mm256_set1_epi8((char) (1 << (2 * j + 1))));
            q5[2 * j] = _mm256_add_epi8(_mm256_and_si256(q5bits, m4), _mm256_slli_epi16(_mm256_srli_epi16(h0, 2 * j), 4));
            q5[2 * j + 1] = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4),
                                            _mm256_slli_epi16(_mm256_srli_epi16(h1, 2 * j + 1), 4));
        }
        const float dx = fp16(x[i].data.data.d), mx = fp16(x[i].data.data.dmin);
        for (int t = 0; t < NT; ++t) {
            const block_q8_K & b = y[t][i];
            const float d = b.d * dx;
            const float dmin = -b.d * mx;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i *) b.bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs[t] += dmin * _mm_extract_epi32(hsum, 0);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < 4; ++j) {
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i *) (b.qs + 64 * j));
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) (b.qs + 64 * j + 32));
                const __m256i p0 = _mm256_madd_epi16(_mm256_set1_epi16(sc8[2 * j]), _mm256_maddubs_epi16(q5[2 * j], q8_0));
                const __m256i p1 = _mm256_madd_epi16(_mm256_set1_epi16(sc8[2 * j + 1]), _mm256_maddubs_epi16(q5[2 * j + 1], q8_1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) s[t] = hsum8(acc[t]) + summs[t];
}

// ---------------------------------------------------------------------------------------------- IQ4_NL x Q8_0
template <int NT>
void iq4_nl_rows(const block_iq4_nl * x, int nb, const block_q8_0 * const * y, float * s) {
    // two accumulators by block parity, as ggml's kernel, so the result is bit-identical to vec_dot (nb is even)
    const __m128i values128 = _mm_loadu_si128((const __m128i *) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[2][NT];
    for (int t = 0; t < NT; ++t) acc[0][t] = acc[1][t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m128i b = _mm_loadu_si128((const __m128i *) x[ib].qs);
        const __m256i w = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(b, 4), m4b)),
                                           _mm_shuffle_epi8(values128, _mm_and_si128(b, m4b)));
        const __m256i aw = _mm256_sign_epi8(w, w);
        const float dw = fp16(x[ib].d);
        for (int t = 0; t < NT; ++t) {
            const __m256i a = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) y[t][ib].qs), w);
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, a), ones);
            acc[ib & 1][t] = _mm256_fmadd_ps(_mm256_set1_ps(fp16(y[t][ib].d) * dw), _mm256_cvtepi32_ps(p), acc[ib & 1][t]);
        }
    }
    for (int t = 0; t < NT; ++t) s[t] = hsum8(_mm256_add_ps(acc[0][t], acc[1][t]));
}

// ---------------------------------------------------------------------------------------------- NVFP4 x Q8_0
// A 64-weight block holds four 16-weight sub-blocks (UE4M3 scale each; byte i of a sub-block's 8 code bytes is
// element i in the low nibble, i + 8 in the high one) and pairs with two Q8_0 blocks. The codes go through the
// doubled-e2m1 table (kvalues_fp4) and the halved scales (ggml's convention), so the products are the weights'.
static float ue4m3h_tab[128];
static const float * ue4m3h_table() {
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 128; ++i) ue4m3h_tab[i] = ggml_ue4m3_to_fp32((uint8_t) i);
        init = true;
    }
    return ue4m3h_tab;
}

template <int NT>
void nvfp4_rows(const block_nvfp4 * x, int nb, const block_q8_0 * const * y, float * s) {
    const float * uh = ue4m3h_table();
    const __m128i lut = _mm_loadu_si128((const __m128i *) kvalues_fp4);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
#pragma GCC unroll 2
        for (int h = 0; h < 2; ++h) {   // the block's two 32-weight halves = y blocks 2 ib + h
            const __m128i q = _mm_loadu_si128((const __m128i *) (x[ib].qs + 16 * h));
            const __m128i vlo = _mm_shuffle_epi8(lut, _mm_and_si128(q, m4b));                      // A0-7 | B0-7
            const __m128i vhi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4b));  // A8-15 | B8-15
            const __m256i w = _mm256_set_m128i(_mm_unpackhi_epi64(vlo, vhi), _mm_unpacklo_epi64(vlo, vhi));
            const __m256i aw = _mm256_sign_epi8(w, w);
            const float dA = uh[x[ib].d[2 * h] & 0x7f], dB = uh[x[ib].d[2 * h + 1] & 0x7f];
            for (int t = 0; t < NT; ++t) {
                const block_q8_0 & yb = y[t][2 * ib + h];
                const __m256i a = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) yb.qs), w);
                const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, a), ones);   // 0-3: A, 4-7: B
                const float yd = fp16(yb.d);
                const __m256 sc = _mm256_set_m128(_mm_set1_ps(dB * yd), _mm_set1_ps(dA * yd));
                acc[t] = _mm256_fmadd_ps(sc, _mm256_cvtepi32_ps(p), acc[t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) s[t] = hsum8(acc[t]);
}

template <typename BX, typename BY, void (*K1)(const BX *, int, const BY * const *, float *),
          void (*K2)(const BX *, int, const BY * const *, float *), void (*K3)(const BX *, int, const BY * const *, float *),
          void (*K4)(const BX *, int, const BY * const *, float *)>
void dispatch(const void * row, int nb, const void * const * y, int T, float * s) {
    const BX * x = (const BX *) row;
    const BY * const * yy = (const BY * const *) y;
    for (int t0 = 0; t0 < T; t0 += 4) {
        const int n = T - t0 < 4 ? T - t0 : 4;
        switch (n) {
            case 1: K1(x, nb, yy + t0, s + t0); break;
            case 2: K2(x, nb, yy + t0, s + t0); break;
            case 3: K3(x, nb, yy + t0, s + t0); break;
            default: K4(x, nb, yy + t0, s + t0);
        }
    }
}

}  // namespace

int mdot_min_tokens(int type) {
    // BNK_CPU_MDOT1=1: every supported format from one token; IQ2_XXS's table rows (xxs_ct_on) are the one-token kernel
    // of choice anyway (ggml's dot builds grid and signs separately)
    static const int mdot1 = getenv("BNK_CPU_MDOT1") ? atoi(getenv("BNK_CPU_MDOT1")) : 0;
    if (mdot1) return 1;
    if (type == GGML_TYPE_IQ2_XXS && xxs_ct_on()) return 1;
    return 2;
}

bool mdot_supported(int type) {
    return type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S ||
           type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS ||
           type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q5_K || type == GGML_TYPE_NVFP4;
}

void mdot(int type, int n, const void * row, const void * const * y, int T, float * s) {
    switch (type) {
        case GGML_TYPE_IQ3_XXS:
            dispatch<block_iq3_xxs, block_q8_K, iq3_xxs_rows<1>, iq3_xxs_rows<2>, iq3_xxs_rows<3>, iq3_xxs_rows<4>>(row, n / QK_K, y, T, s);
            break;
        case GGML_TYPE_IQ3_S:
            dispatch<block_iq3_s, block_q8_K, iq3_s_rows<1>, iq3_s_rows<2>, iq3_s_rows<3>, iq3_s_rows<4>>(row, n / QK_K, y, T, s);
            break;
        case GGML_TYPE_IQ2_S:
            dispatch<block_iq2_s, block_q8_K, iq2_s_rows<1>, iq2_s_rows<2>, iq2_s_rows<3>, iq2_s_rows<4>>(row, n / QK_K, y, T, s);
            break;
        case GGML_TYPE_IQ2_XXS:
            if (xxs_ct_on())
                dispatch<block_iq2_xxs, block_q8_K, iq2_xxs_rows_ct<1>, iq2_xxs_rows_ct<2>, iq2_xxs_rows_ct<3>, iq2_xxs_rows_ct<4>>(row, n / QK_K, y, T, s);
            else
                dispatch<block_iq2_xxs, block_q8_K, iq2_xxs_rows<1>, iq2_xxs_rows<2>, iq2_xxs_rows<3>, iq2_xxs_rows<4>>(row, n / QK_K, y, T, s);
            break;
        case GGML_TYPE_IQ2_XS:
            dispatch<block_iq2_xs, block_q8_K, iq2_xs_rows<1>, iq2_xs_rows<2>, iq2_xs_rows<3>, iq2_xs_rows<4>>(row, n / QK_K, y, T, s);
            break;
        case GGML_TYPE_IQ4_XS:
            dispatch<block_iq4_xs, block_q8_K, iq4_xs_rows<1>, iq4_xs_rows<2>, iq4_xs_rows<3>, iq4_xs_rows<4>>(row, n / QK_K, y, T, s);
            break;
        case GGML_TYPE_IQ4_NL:
            dispatch<block_iq4_nl, block_q8_0, iq4_nl_rows<1>, iq4_nl_rows<2>, iq4_nl_rows<3>, iq4_nl_rows<4>>(row, n / QK4_NL, y, T, s);
            break;
        case GGML_TYPE_Q5_K:
            dispatch<block_q5_K, block_q8_K, q5_k_rows<1>, q5_k_rows<2>, q5_k_rows<3>, q5_k_rows<4>>(row, n / QK_K, y, T, s);
            break;
        case GGML_TYPE_NVFP4:
            dispatch<block_nvfp4, block_q8_0, nvfp4_rows<1>, nvfp4_rows<2>, nvfp4_rows<3>, nvfp4_rows<4>>(row, n / QK_NVFP4, y, T, s);
            break;
        default: break;
    }
}

}  // namespace bnk
