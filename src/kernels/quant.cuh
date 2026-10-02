// Per-format weight unpacking for the dp4a GEMV family.
//
// Every supported ggml format is presented to the kernels through one interface: a 32-weight
// sub-block becomes eight packed int8x4 words plus per-16 scales (and, for formats with a min,
// per-16 mins), so that   w_real[i] = d[i/16] * w_int8[i] - m[i/16].
// The bit layouts follow ggml's reference dequantizers (ggml-quants.c / ggml-cuda/dequantize.cuh);
// tests/test_dequant.cu checks every format against ggml's own CPU to_float.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

namespace bnk {

// Format ids are ggml_type values; kept as ints so device code needs no ggml.h.
enum : int {
    QT_F32 = 0, QT_F16 = 1, QT_Q4_0 = 2, QT_Q8_0 = 8, QT_Q2_K = 10, QT_Q3_K = 11, QT_Q4_K = 12,
    QT_Q5_K = 13, QT_Q6_K = 14, QT_IQ2_XXS = 16, QT_IQ2_XS = 17, QT_IQ3_XXS = 18, QT_IQ4_NL = 20,
    QT_IQ3_S = 21, QT_IQ2_S = 22, QT_IQ4_XS = 23, QT_BF16 = 30, QT_Q2_0 = 42,
};

// Loads that only assume 2-byte alignment (most ggml block sizes are not multiples of 4).
__device__ __forceinline__ uint32_t ld32a2(const uint8_t * p) {
    const uint16_t * q = (const uint16_t *) p;
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}
__device__ __forceinline__ uint16_t ld16(const uint8_t * p) { return *(const uint16_t *) p; }
__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// Negate byte k of g when bit k of s4 is set.
__device__ __forceinline__ int apply_signs4(uint32_t g, uint32_t s4) {
    const uint32_t m = ((s4 * 0x00204081u) & 0x01010101u) * 0xFFu;
    return (int) __vsub4(g ^ m, m);
}

// ggml's 16-entry int8 table lookup for 8 packed nibbles: .x the low nibbles, .y the high ones.
__device__ __forceinline__ int2 lut16(uint32_t q4, const int8_t * table) {
    const uint32_t * t32 = (const uint32_t *) table;
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = __byte_perm(t32[0], t32[1], q4 >> sh);
        const uint32_t hi = __byte_perm(t32[2], t32[3], q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2((int) __byte_perm(tmp[0], tmp[1], 0x6420), (int) __byte_perm(tmp[0], tmp[1], 0x7531));
}

struct Unpacked {
    int w[8];
    float d0, d1;  // scales of elements 0-15 / 16-31
    float m0, m1;  // mins (only formats with HAS_MIN)
};

template <int T> struct QTraits;

// ---------------------------------------------------------------------------------------- k-quants
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t * q, int & d, int & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

template <> struct QTraits<QT_Q4_K> {
    static constexpr int QK = 256, BYTES = 144;
    static constexpr bool HAS_MIN = true;
    __device__ static void unpack(const uint8_t * b, int s, Unpacked & u) {
        const float dall = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, mn;
        scale_min_k4(s, b + 4, sc, mn);
        u.d0 = u.d1 = dall * sc;
        u.m0 = u.m1 = dmin * mn;
        const uint8_t * q = b + 16 + 32 * (s / 2);
        const int sh = 4 * (s & 1);
#pragma unroll
        for (int i = 0; i < 8; ++i) u.w[i] = (int) ((ld32a2(q + 4 * i) >> sh) & 0x0F0F0F0F);
    }
};

template <> struct QTraits<QT_Q5_K> {
    static constexpr int QK = 256, BYTES = 176;
    static constexpr bool HAS_MIN = true;
    __device__ static void unpack(const uint8_t * b, int s, Unpacked & u) {
        const float dall = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
        int sc, mn;
        scale_min_k4(s, b + 4, sc, mn);
        u.d0 = u.d1 = dall * sc;
        u.m0 = u.m1 = dmin * mn;
        const uint8_t * qh = b + 16;
        const uint8_t * q = b + 48 + 32 * (s / 2);
        const int sh = 4 * (s & 1);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t lo = (ld32a2(q + 4 * i) >> sh) & 0x0F0F0F0F;
            const uint32_t hi = (ld32a2(qh + 4 * i) >> s) & 0x01010101;
            u.w[i] = (int) (lo | (hi << 4));
        }
    }
};

template <> struct QTraits<QT_Q6_K> {
    static constexpr int QK = 256, BYTES = 210;
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * b, int s, Unpacked & u) {
        const int ip = s / 4, q = s % 4;
        const uint8_t * ql = b + 64 * ip + 32 * (q & 1);
        const uint8_t * qh = b + 128 + 32 * ip;
        const int8_t * sc = (const int8_t *) (b + 192) + 8 * ip + 2 * q;
        const float d = h2f(ld16(b + 208));
        u.d0 = d * sc[0];
        u.d1 = d * sc[1];
        u.m0 = u.m1 = 0.f;
        const int lsh = (q >= 2) ? 4 : 0, hsh = 2 * q;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t lo = (ld32a2(ql + 4 * i) >> lsh) & 0x0F0F0F0F;
            const uint32_t hi = (ld32a2(qh + 4 * i) >> hsh) & 0x03030303;
            u.w[i] = (int) __vsubss4(lo | (hi << 4), 0x20202020);
        }
    }
};

// ---------------------------------------------------------------------------------------- i-quants
template <> struct QTraits<QT_IQ2_S> {
    static constexpr int QK = 256, BYTES = 82;
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * b, int ib, Unpacked & u) {
        const float d = h2f(ld16(b));
        const uint8_t * qs = b + 2;
        const uint8_t qh = b[66 + ib];
        const uint8_t sc = b[74 + ib];
        u.d0 = d * (0.5f + (sc & 0xf)) * 0.25f;
        u.d1 = d * (0.5f + (sc >> 4)) * 0.25f;
        u.m0 = u.m1 = 0.f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const int idx = qs[4 * ib + il] | ((qh << (8 - 2 * il)) & 0x300);
            const uint2 g = *(const uint2 *) (iq2s_grid + idx);
            const uint32_t sg = qs[32 + 4 * ib + il];
            u.w[2 * il + 0] = apply_signs4(g.x, sg & 0xF);
            u.w[2 * il + 1] = apply_signs4(g.y, sg >> 4);
        }
    }
};

template <> struct QTraits<QT_IQ3_XXS> {
    static constexpr int QK = 256, BYTES = 98;
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * b, int ib, Unpacked & u) {
        const float d = h2f(ld16(b));
        const uint8_t * q3 = b + 2 + 8 * ib;
        const uint32_t aux = ld32a2(b + 2 + 64 + 4 * ib);
        u.d0 = u.d1 = d * (0.5f + (aux >> 28)) * 0.5f;
        u.m0 = u.m1 = 0.f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t sg = ksigns_iq2xs[(aux >> (7 * il)) & 127];
            u.w[2 * il + 0] = apply_signs4(iq3xxs_grid[q3[2 * il + 0]], sg & 0xF);
            u.w[2 * il + 1] = apply_signs4(iq3xxs_grid[q3[2 * il + 1]], sg >> 4);
        }
    }
};

template <> struct QTraits<QT_IQ3_S> {
    static constexpr int QK = 256, BYTES = 110;
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * b, int ib, Unpacked & u) {
        const float d = h2f(ld16(b));
        const uint8_t * qs = b + 2 + 8 * ib;
        const int qh = b[66 + ib];
        const uint8_t * signs = b + 74 + 4 * ib;
        u.d0 = u.d1 = d * (1 + 2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 0xf));
        u.m0 = u.m1 = 0.f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t g1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 256)];
            const uint32_t g2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
            const uint32_t sg = signs[il];
            u.w[2 * il + 0] = apply_signs4(g1, sg & 0xF);
            u.w[2 * il + 1] = apply_signs4(g2, sg >> 4);
        }
    }
};

template <> struct QTraits<QT_IQ4_NL> {
    static constexpr int QK = 32, BYTES = 18;
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * b, int, Unpacked & u) {
        u.d0 = u.d1 = h2f(ld16(b));
        u.m0 = u.m1 = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int2 v = lut16(ld32a2(b + 2 + 4 * i), kvalues_iq4nl);
            u.w[i] = v.x;
            u.w[4 + i] = v.y;
        }
    }
};

template <> struct QTraits<QT_IQ4_XS> {
    static constexpr int QK = 256, BYTES = 136;
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * b, int ib, Unpacked & u) {
        const float d = h2f(ld16(b));
        const int sh = ld16(b + 2);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        u.d0 = u.d1 = d * (ls - 32);
        u.m0 = u.m1 = 0.f;
        const uint8_t * q4 = b + 8 + 16 * ib;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int2 v = lut16(ld32a2(q4 + 4 * i), kvalues_iq4nl);
            u.w[i] = v.x;
            u.w[4 + i] = v.y;
        }
    }
};

// ---------------------------------------------------------------------------------------- simple
template <> struct QTraits<QT_Q8_0> {
    static constexpr int QK = 32, BYTES = 34;
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * b, int, Unpacked & u) {
        u.d0 = u.d1 = h2f(ld16(b));
        u.m0 = u.m1 = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) u.w[i] = (int) ld32a2(b + 2 + 4 * i);
    }
};

template <> struct QTraits<QT_Q2_0> {
    static constexpr int QK = 64, BYTES = 18;
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * b, int s, Unpacked & u) {
        u.d0 = u.d1 = h2f(ld16(b));
        u.m0 = u.m1 = 0.f;
        const uint8_t * q = b + 2 + 8 * s;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t x = q[i];
            const uint32_t spread = (x | (x << 6) | (x << 12) | (x << 18)) & 0x03030303u;
            u.w[i] = (int) __vsub4(spread, 0x01010101u);
        }
    }
};

template <int T>
__device__ __forceinline__ void unpack_sub(const uint8_t * row, int sb, Unpacked & u) {
    using Q = QTraits<T>;
    constexpr int SUBS = Q::QK / 32;
    const uint8_t * blk = row + (size_t) (sb / SUBS) * Q::BYTES;
    Q::unpack(blk, sb % SUBS, u);
}

// The formats the dp4a family handles; float formats (F32/F16/BF16) take the float GEMV path.
__host__ __device__ constexpr bool is_dp4a_format(int t) {
    return t == QT_Q4_K || t == QT_Q5_K || t == QT_Q6_K || t == QT_IQ2_S || t == QT_IQ3_XXS ||
           t == QT_IQ3_S || t == QT_IQ4_NL || t == QT_IQ4_XS || t == QT_Q8_0 || t == QT_Q2_0;
}
__host__ __device__ constexpr bool is_float_format(int t) { return t == QT_F32 || t == QT_F16 || t == QT_BF16; }

// Dispatch a functor templated on the format: f.template operator()<FMT>().
#define BNK_DISPATCH_DP4A(fmt, F)                                    \
    switch (fmt) {                                                   \
        case QT_Q4_K: F.template operator()<QT_Q4_K>(); break;       \
        case QT_Q5_K: F.template operator()<QT_Q5_K>(); break;       \
        case QT_Q6_K: F.template operator()<QT_Q6_K>(); break;       \
        case QT_IQ2_S: F.template operator()<QT_IQ2_S>(); break;     \
        case QT_IQ3_XXS: F.template operator()<QT_IQ3_XXS>(); break; \
        case QT_IQ3_S: F.template operator()<QT_IQ3_S>(); break;     \
        case QT_IQ4_NL: F.template operator()<QT_IQ4_NL>(); break;   \
        case QT_IQ4_XS: F.template operator()<QT_IQ4_XS>(); break;   \
        case QT_Q8_0: F.template operator()<QT_Q8_0>(); break;       \
        case QT_Q2_0: F.template operator()<QT_Q2_0>(); break;       \
        default: bnk_unsupported_format(fmt);                        \
    }

void bnk_unsupported_format(int fmt);

}  // namespace bnk
