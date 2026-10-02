// The R repack of one row, shared by the host loader and the device cache fill (rfmt.h describes the layout).
#pragma once

#include <cstdint>
#include <cstring>

#include "kernels/rfmt.h"

#ifdef __CUDACC__
#define BNK_HD __host__ __device__
#else
#define BNK_HD
#endif

namespace bnk {

enum : int { F_Q8_0 = 8, F_Q4_K = 12, F_Q5_K = 13, F_Q6_K = 14, F_IQ3_XXS = 18, F_IQ4_NL = 20, F_IQ3_S = 21,
             F_IQ2_S = 22, F_IQ4_XS = 23, F_Q2_0 = 42 };

BNK_HD static inline void rfmt_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// 32 4-bit codes (element order) into a nibble slot: byte j = code[j] | code[j+16] << 4
BNK_HD static inline void rfmt_put_nibbles(const uint8_t * code, uint8_t * slot) {
    for (int j = 0; j < 16; ++j) slot[j] = (uint8_t) ((code[j] & 15) | ((code[j + 16] & 15) << 4));
}

BNK_HD static inline void rfmt_repack_row(const RLayout & L, const uint8_t * s, uint8_t * d) {
    memset(d, 0, L.row_bytes);
    const int n = L.nsb;
    uint8_t * A = d + L.off_a, * B = d + L.off_b, * C = d + L.off_c, * D = d + L.off_d, * E = d + L.off_e;
    switch (L.fmt) {
        case F_Q4_K:
            for (int sb = 0; sb < n; ++sb) {
                const uint8_t * blk = s + (sb / 8) * 144;
                const int k = sb % 8;
                uint8_t code[32];
                for (int e = 0; e < 32; ++e) code[e] = (blk[16 + 32 * (k / 2) + e] >> (4 * (k & 1))) & 15;
                rfmt_put_nibbles(code, A + 16 * sb);
                rfmt_scale_min_k4(k, blk + 4, B[2 * sb], B[2 * sb + 1]);
                if (k == 0) memcpy(C + 4 * (sb / 8), blk, 4);
            }
            break;
        case F_Q5_K:
            for (int sb = 0; sb < n; ++sb) {
                const uint8_t * blk = s + (sb / 8) * 176;
                const int k = sb % 8;
                uint8_t code[32];
                uint32_t hb = 0;
                for (int e = 0; e < 32; ++e) {
                    code[e] = (blk[48 + 32 * (k / 2) + e] >> (4 * (k & 1))) & 15;
                    hb |= (uint32_t) ((blk[16 + e] >> k) & 1) << e;
                }
                rfmt_put_nibbles(code, A + 16 * sb);
                memcpy(B + 4 * sb, &hb, 4);
                rfmt_scale_min_k4(k, blk + 4, C[2 * sb], C[2 * sb + 1]);
                if (k == 0) memcpy(D + 4 * (sb / 8), blk, 4);
            }
            break;
        case F_Q6_K:
            for (int sb = 0; sb < n; ++sb) {
                const uint8_t * blk = s + (sb / 8) * 210;
                const int k = sb % 8, ip = k / 4, q = k % 4;
                uint8_t code[32];
                uint64_t hb = 0;
                for (int e = 0; e < 32; ++e) {
                    code[e] = (blk[64 * ip + 32 * (q & 1) + e] >> (q >= 2 ? 4 : 0)) & 15;
                    hb |= (uint64_t) ((blk[128 + 32 * ip + e] >> (2 * q)) & 3) << (2 * e);
                }
                rfmt_put_nibbles(code, A + 16 * sb);
                memcpy(B + 8 * sb, &hb, 8);
                C[2 * sb] = blk[192 + 8 * ip + 2 * q];
                C[2 * sb + 1] = blk[192 + 8 * ip + 2 * q + 1];
                if (k == 0) memcpy(D + 2 * (sb / 8), blk + 208, 2);
            }
            break;
        case F_Q8_0:
            for (int sb = 0; sb < n; ++sb) {
                memcpy(A + 32 * sb, s + 34 * sb + 2, 32);
                memcpy(B + 2 * sb, s + 34 * sb, 2);
            }
            break;
        case F_IQ4_NL:
            for (int sb = 0; sb < n; ++sb) {
                memcpy(A + 16 * sb, s + 18 * sb + 2, 16);
                memcpy(B + 2 * sb, s + 18 * sb, 2);
            }
            break;
        case F_IQ4_XS:
            for (int sb = 0; sb < n; ++sb) {
                const uint8_t * blk = s + (sb / 8) * 136;
                const int ib = sb % 8;
                uint16_t sh;
                memcpy(&sh, blk + 2, 2);
                B[sb] = (uint8_t) (((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4));
                memcpy(A + 16 * sb, blk + 8 + 16 * ib, 16);
                if (ib == 0) memcpy(C + 2 * (sb / 8), blk, 2);
            }
            break;
        case F_IQ2_S:
            for (int sb = 0; sb < n; ++sb) {
                const uint8_t * blk = s + (sb / 8) * 82;
                const int ib = sb % 8;
                memcpy(A + 4 * sb, blk + 2 + 4 * ib, 4);
                memcpy(B + 4 * sb, blk + 2 + 32 + 4 * ib, 4);
                C[sb] = blk[66 + ib];
                D[sb] = blk[74 + ib];
                if (ib == 0) memcpy(E + 2 * (sb / 8), blk, 2);
            }
            break;
        case F_IQ3_XXS:
            for (int sb = 0; sb < n; ++sb) {
                const uint8_t * blk = s + (sb / 8) * 98;
                const int ib = sb % 8;
                memcpy(A + 8 * sb, blk + 2 + 8 * ib, 8);
                memcpy(B + 4 * sb, blk + 2 + 64 + 4 * ib, 4);
                if (ib == 0) memcpy(C + 2 * (sb / 8), blk, 2);
            }
            break;
        case F_IQ3_S:
            for (int sb = 0; sb < n; ++sb) {
                const uint8_t * blk = s + (sb / 8) * 110;
                const int ib = sb % 8;
                memcpy(A + 8 * sb, blk + 2 + 8 * ib, 8);
                memcpy(B + 4 * sb, blk + 74 + 4 * ib, 4);
                C[sb] = blk[66 + ib];
                D[sb] = (blk[106 + ib / 2] >> (4 * (ib % 2))) & 0xf;
                if (ib == 0) memcpy(E + 2 * (sb / 8), blk, 2);
            }
            break;
        case F_Q2_0:
            for (int sb = 0; sb < n; ++sb) {
                const uint8_t * blk = s + (sb / 2) * 18;
                memcpy(A + 8 * sb, blk + 2 + 8 * (sb % 2), 8);
                if (sb % 2 == 0) memcpy(B + 2 * (sb / 2), blk, 2);
            }
            break;
    }
}

// One sub-block of rfmt_repack_row (the per-sub-block fields are disjoint; row padding is left as is).
BNK_HD static inline void rfmt_repack_sb(const RLayout & L, const uint8_t * s, uint8_t * d, int sb) {
    uint8_t * A = d + L.off_a, * B = d + L.off_b, * C = d + L.off_c, * D = d + L.off_d, * E = d + L.off_e;
    switch (L.fmt) {
        case F_Q4_K:
            {
                const uint8_t * blk = s + (sb / 8) * 144;
                const int k = sb % 8;
                uint8_t code[32];
                for (int e = 0; e < 32; ++e) code[e] = (blk[16 + 32 * (k / 2) + e] >> (4 * (k & 1))) & 15;
                rfmt_put_nibbles(code, A + 16 * sb);
                rfmt_scale_min_k4(k, blk + 4, B[2 * sb], B[2 * sb + 1]);
                if (k == 0) memcpy(C + 4 * (sb / 8), blk, 4);
            }
            break;
        case F_Q5_K:
            {
                const uint8_t * blk = s + (sb / 8) * 176;
                const int k = sb % 8;
                uint8_t code[32];
                uint32_t hb = 0;
                for (int e = 0; e < 32; ++e) {
                    code[e] = (blk[48 + 32 * (k / 2) + e] >> (4 * (k & 1))) & 15;
                    hb |= (uint32_t) ((blk[16 + e] >> k) & 1) << e;
                }
                rfmt_put_nibbles(code, A + 16 * sb);
                memcpy(B + 4 * sb, &hb, 4);
                rfmt_scale_min_k4(k, blk + 4, C[2 * sb], C[2 * sb + 1]);
                if (k == 0) memcpy(D + 4 * (sb / 8), blk, 4);
            }
            break;
        case F_Q6_K:
            {
                const uint8_t * blk = s + (sb / 8) * 210;
                const int k = sb % 8, ip = k / 4, q = k % 4;
                uint8_t code[32];
                uint64_t hb = 0;
                for (int e = 0; e < 32; ++e) {
                    code[e] = (blk[64 * ip + 32 * (q & 1) + e] >> (q >= 2 ? 4 : 0)) & 15;
                    hb |= (uint64_t) ((blk[128 + 32 * ip + e] >> (2 * q)) & 3) << (2 * e);
                }
                rfmt_put_nibbles(code, A + 16 * sb);
                memcpy(B + 8 * sb, &hb, 8);
                C[2 * sb] = blk[192 + 8 * ip + 2 * q];
                C[2 * sb + 1] = blk[192 + 8 * ip + 2 * q + 1];
                if (k == 0) memcpy(D + 2 * (sb / 8), blk + 208, 2);
            }
            break;
        case F_Q8_0:
            {
                memcpy(A + 32 * sb, s + 34 * sb + 2, 32);
                memcpy(B + 2 * sb, s + 34 * sb, 2);
            }
            break;
        case F_IQ4_NL:
            {
                memcpy(A + 16 * sb, s + 18 * sb + 2, 16);
                memcpy(B + 2 * sb, s + 18 * sb, 2);
            }
            break;
        case F_IQ4_XS:
            {
                const uint8_t * blk = s + (sb / 8) * 136;
                const int ib = sb % 8;
                uint16_t sh;
                memcpy(&sh, blk + 2, 2);
                B[sb] = (uint8_t) (((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4));
                memcpy(A + 16 * sb, blk + 8 + 16 * ib, 16);
                if (ib == 0) memcpy(C + 2 * (sb / 8), blk, 2);
            }
            break;
        case F_IQ2_S:
            {
                const uint8_t * blk = s + (sb / 8) * 82;
                const int ib = sb % 8;
                memcpy(A + 4 * sb, blk + 2 + 4 * ib, 4);
                memcpy(B + 4 * sb, blk + 2 + 32 + 4 * ib, 4);
                C[sb] = blk[66 + ib];
                D[sb] = blk[74 + ib];
                if (ib == 0) memcpy(E + 2 * (sb / 8), blk, 2);
            }
            break;
        case F_IQ3_XXS:
            {
                const uint8_t * blk = s + (sb / 8) * 98;
                const int ib = sb % 8;
                memcpy(A + 8 * sb, blk + 2 + 8 * ib, 8);
                memcpy(B + 4 * sb, blk + 2 + 64 + 4 * ib, 4);
                if (ib == 0) memcpy(C + 2 * (sb / 8), blk, 2);
            }
            break;
        case F_IQ3_S:
            {
                const uint8_t * blk = s + (sb / 8) * 110;
                const int ib = sb % 8;
                memcpy(A + 8 * sb, blk + 2 + 8 * ib, 8);
                memcpy(B + 4 * sb, blk + 74 + 4 * ib, 4);
                C[sb] = blk[66 + ib];
                D[sb] = (blk[106 + ib / 2] >> (4 * (ib % 2))) & 0xf;
                if (ib == 0) memcpy(E + 2 * (sb / 8), blk, 2);
            }
            break;
        case F_Q2_0:
            {
                const uint8_t * blk = s + (sb / 2) * 18;
                memcpy(A + 8 * sb, blk + 2 + 8 * (sb % 2), 8);
                if (sb % 2 == 0) memcpy(B + 2 * (sb / 2), blk, 2);
            }
            break;
    }
}


}  // namespace bnk
