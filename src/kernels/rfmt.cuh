// Device unpacking of R-layout rows (see rfmt.h) into the dp4a form of quant.cuh.
#pragma once

#include "kernels/quant.cuh"
#include "kernels/rfmt.h"

namespace bnk {

struct ROff {
    uint32_t a, b, c, d, e;
};
inline ROff roff(const RLayout & L) { return ROff{L.off_a, L.off_b, L.off_c, L.off_d, L.off_e}; }

// 4 bits of h (bit k -> element k) into bit 4 of bytes 0..3
__device__ __forceinline__ uint32_t spread1_hi(uint32_t h4) { return ((h4 * 0x00204081u) & 0x01010101u) << 4; }
// 4 2-bit fields of a byte into bytes 0..3
__device__ __forceinline__ uint32_t spread2(uint32_t b) { return (b | (b << 6) | (b << 12) | (b << 18)) & 0x03030303u; }

template <int FMT> struct RT;

template <> struct RT<QT_Q4_K> {
    static constexpr bool HAS_MIN = true;
    __device__ static void unpack(const uint8_t * row, const ROff & o, int sb, Unpacked & u) {
        const uint4 q = *(const uint4 *) (row + o.a + 16 * sb);
        const uint16_t sm = *(const uint16_t *) (row + o.b + 2 * sb);
        const uint32_t dm = *(const uint32_t *) (row + o.c + 4 * (sb >> 3));
        const float d = h2f((uint16_t) dm), dmin = h2f((uint16_t) (dm >> 16));
        u.d0 = u.d1 = d * (float) (sm & 0xff);
        u.m0 = u.m1 = dmin * (float) (sm >> 8);
        const uint32_t x[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u.w[i] = (int) (x[i] & 0x0F0F0F0F);
            u.w[4 + i] = (int) ((x[i] >> 4) & 0x0F0F0F0F);
        }
    }
};

template <> struct RT<QT_Q5_K> {
    static constexpr bool HAS_MIN = true;
    __device__ static void unpack(const uint8_t * row, const ROff & o, int sb, Unpacked & u) {
        const uint4 q = *(const uint4 *) (row + o.a + 16 * sb);
        const uint32_t hb = *(const uint32_t *) (row + o.b + 4 * sb);
        const uint16_t sm = *(const uint16_t *) (row + o.c + 2 * sb);
        const uint32_t dm = *(const uint32_t *) (row + o.d + 4 * (sb >> 3));
        const float d = h2f((uint16_t) dm), dmin = h2f((uint16_t) (dm >> 16));
        u.d0 = u.d1 = d * (float) (sm & 0xff);
        u.m0 = u.m1 = dmin * (float) (sm >> 8);
        const uint32_t x[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u.w[i] = (int) ((x[i] & 0x0F0F0F0F) | spread1_hi((hb >> (4 * i)) & 0xF));
            u.w[4 + i] = (int) (((x[i] >> 4) & 0x0F0F0F0F) | spread1_hi((hb >> (16 + 4 * i)) & 0xF));
        }
    }
};

template <> struct RT<QT_Q6_K> {
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * row, const ROff & o, int sb, Unpacked & u) {
        const uint4 q = *(const uint4 *) (row + o.a + 16 * sb);
        const uint2 h = *(const uint2 *) (row + o.b + 8 * sb);
        const int8_t * sc = (const int8_t *) (row + o.c + 2 * sb);
        const float d = h2f(*(const uint16_t *) (row + o.d + 2 * (sb >> 3)));
        u.d0 = d * sc[0];
        u.d1 = d * sc[1];
        u.m0 = u.m1 = 0.f;
        const uint32_t x[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t hl = spread2((h.x >> (8 * i)) & 0xff) << 4;
            const uint32_t hh = spread2((h.y >> (8 * i)) & 0xff) << 4;
            u.w[i] = (int) __vsubss4((x[i] & 0x0F0F0F0F) | hl, 0x20202020);
            u.w[4 + i] = (int) __vsubss4(((x[i] >> 4) & 0x0F0F0F0F) | hh, 0x20202020);
        }
    }
};

template <> struct RT<QT_Q8_0> {
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * row, const ROff & o, int sb, Unpacked & u) {
        const uint4 a = *(const uint4 *) (row + o.a + 32 * sb);
        const uint4 b = *(const uint4 *) (row + o.a + 32 * sb + 16);
        u.d0 = u.d1 = h2f(*(const uint16_t *) (row + o.b + 2 * sb));
        u.m0 = u.m1 = 0.f;
        u.w[0] = a.x; u.w[1] = a.y; u.w[2] = a.z; u.w[3] = a.w;
        u.w[4] = b.x; u.w[5] = b.y; u.w[6] = b.z; u.w[7] = b.w;
    }
};

template <> struct RT<QT_IQ4_NL> {
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * row, const ROff & o, int sb, Unpacked & u) {
        const uint4 q = *(const uint4 *) (row + o.a + 16 * sb);
        u.d0 = u.d1 = h2f(*(const uint16_t *) (row + o.b + 2 * sb));
        u.m0 = u.m1 = 0.f;
        const uint32_t x[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int2 v = lut16(x[i], kvalues_iq4nl);
            u.w[i] = v.x;
            u.w[4 + i] = v.y;
        }
    }
};

template <> struct RT<QT_IQ4_XS> {
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * row, const ROff & o, int sb, Unpacked & u) {
        const uint4 q = *(const uint4 *) (row + o.a + 16 * sb);
        const int ls = row[o.b + sb];
        u.d0 = u.d1 = h2f(*(const uint16_t *) (row + o.c + 2 * (sb >> 3))) * (float) (ls - 32);
        u.m0 = u.m1 = 0.f;
        const uint32_t x[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int2 v = lut16(x[i], kvalues_iq4nl);
            u.w[i] = v.x;
            u.w[4 + i] = v.y;
        }
    }
};

template <> struct RT<QT_IQ2_S> {
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * row, const ROff & o, int sb, Unpacked & u) {
        const uint32_t g = *(const uint32_t *) (row + o.a + 4 * sb);
        const uint32_t sg = *(const uint32_t *) (row + o.b + 4 * sb);
        const uint32_t qh = row[o.c + sb];
        const uint32_t sc = row[o.d + sb];
        const float d = h2f(*(const uint16_t *) (row + o.e + 2 * (sb >> 3)));
        u.d0 = d * (0.5f + (sc & 0xf)) * 0.25f;
        u.d1 = d * (0.5f + (sc >> 4)) * 0.25f;
        u.m0 = u.m1 = 0.f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const int idx = ((g >> (8 * il)) & 0xff) | ((qh << (8 - 2 * il)) & 0x300);
            const uint2 gr = *(const uint2 *) (iq2s_grid + idx);
            const uint32_t s8 = (sg >> (8 * il)) & 0xff;
            u.w[2 * il + 0] = apply_signs4(gr.x, s8 & 0xF);
            u.w[2 * il + 1] = apply_signs4(gr.y, s8 >> 4);
        }
    }
};

template <> struct RT<QT_IQ3_XXS> {
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * row, const ROff & o, int sb, Unpacked & u) {
        const uint2 g = *(const uint2 *) (row + o.a + 8 * sb);
        const uint32_t aux = *(const uint32_t *) (row + o.b + 4 * sb);
        const float d = h2f(*(const uint16_t *) (row + o.c + 2 * (sb >> 3)));
        u.d0 = u.d1 = d * (0.5f + (aux >> 28)) * 0.5f;
        u.m0 = u.m1 = 0.f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t gw = il < 2 ? g.x : g.y;
            const int sh = 16 * (il & 1);
            const uint32_t sg = ksigns_iq2xs[(aux >> (7 * il)) & 127];
            u.w[2 * il + 0] = apply_signs4(iq3xxs_grid[(gw >> sh) & 0xff], sg & 0xF);
            u.w[2 * il + 1] = apply_signs4(iq3xxs_grid[(gw >> (sh + 8)) & 0xff], sg >> 4);
        }
    }
};

template <> struct RT<QT_IQ3_S> {
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * row, const ROff & o, int sb, Unpacked & u) {
        const uint2 q = *(const uint2 *) (row + o.a + 8 * sb);
        const uint32_t sg = *(const uint32_t *) (row + o.b + 4 * sb);
        const int qh = row[o.c + sb];
        const int sc = row[o.d + sb];
        u.d0 = u.d1 = h2f(*(const uint16_t *) (row + o.e + 2 * (sb >> 3))) * (float) (1 + 2 * sc);
        u.m0 = u.m1 = 0.f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t qw = il < 2 ? q.x : q.y;
            const int sh = 16 * (il & 1);
            const uint32_t g1 = iq3s_grid[((qw >> sh) & 0xff) | ((qh << (8 - 2 * il)) & 256)];
            const uint32_t g2 = iq3s_grid[((qw >> (sh + 8)) & 0xff) | ((qh << (7 - 2 * il)) & 256)];
            const uint32_t s8 = (sg >> (8 * il)) & 0xff;
            u.w[2 * il + 0] = apply_signs4(g1, s8 & 0xF);
            u.w[2 * il + 1] = apply_signs4(g2, s8 >> 4);
        }
    }
};

template <> struct RT<QT_Q2_0> {
    static constexpr bool HAS_MIN = false;
    __device__ static void unpack(const uint8_t * row, const ROff & o, int sb, Unpacked & u) {
        const uint2 q = *(const uint2 *) (row + o.a + 8 * sb);
        u.d0 = u.d1 = h2f(*(const uint16_t *) (row + o.b + 2 * (sb >> 1)));
        u.m0 = u.m1 = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t b = ((i < 4 ? q.x : q.y) >> (8 * (i & 3))) & 0xff;
            u.w[i] = (int) __vsub4(spread2(b), 0x01010101u);
        }
    }
};

#define BNK_DISPATCH_R(fmt, F)                                       \
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

}  // namespace bnk
