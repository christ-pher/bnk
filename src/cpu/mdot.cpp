// Multi-token AVX2 dot products: one decode of a weight row serves up to 4 activation rows.
// Adapted from ggml's AVX2 vec_dot kernels (ggml-cpu/arch/x86/quants.c, MIT): the weight decode (grid and sign
// lookups) is hoisted out of the per-token loop, which is what a verify window's shared experts need.
#include "cpu/mdot.h"

#include <immintrin.h>

#include <cstring>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml.h"

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

// ---------------------------------------------------------------------------------------------- IQ4_NL x Q8_0
template <int NT>
void iq4_nl_rows(const block_iq4_nl * x, int nb, const block_q8_0 * const * y, float * s) {
    const __m128i values128 = _mm_loadu_si128((const __m128i *) kvalues_iq4nl);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m128i b = _mm_loadu_si128((const __m128i *) x[ib].qs);
        const __m256i w = _mm256_set_m128i(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(b, 4), m4b)),
                                           _mm_shuffle_epi8(values128, _mm_and_si128(b, m4b)));
        const __m256i aw = _mm256_sign_epi8(w, w);
        const float dw = fp16(x[ib].d);
        for (int t = 0; t < NT; ++t) {
            const __m256i a = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) y[t][ib].qs), w);
            const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, a), ones);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * fp16(y[t][ib].d)), _mm256_cvtepi32_ps(p), acc[t]);
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

bool mdot_supported(int type) {
    return type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S ||
           type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ4_NL;
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
        case GGML_TYPE_IQ4_XS:
            dispatch<block_iq4_xs, block_q8_K, iq4_xs_rows<1>, iq4_xs_rows<2>, iq4_xs_rows<3>, iq4_xs_rows<4>>(row, n / QK_K, y, T, s);
            break;
        case GGML_TYPE_IQ4_NL:
            dispatch<block_iq4_nl, block_q8_0, iq4_nl_rows<1>, iq4_nl_rows<2>, iq4_nl_rows<3>, iq4_nl_rows<4>>(row, n / QK4_NL, y, T, s);
            break;
        default: break;
    }
}

}  // namespace bnk
