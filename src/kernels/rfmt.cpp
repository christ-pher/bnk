#include "kernels/rfmt.h"

#include <cstring>
#include <stdexcept>

namespace bnk {

enum : int { F_Q8_0 = 8, F_Q4_K = 12, F_Q5_K = 13, F_Q6_K = 14, F_IQ3_XXS = 18, F_IQ4_NL = 20, F_IQ3_S = 21,
             F_IQ2_S = 22, F_IQ4_XS = 23, F_Q2_0 = 42 };

bool r_supported(int fmt) {
    switch (fmt) {
        case F_Q8_0: case F_Q4_K: case F_Q5_K: case F_Q6_K: case F_IQ3_XXS: case F_IQ4_NL: case F_IQ3_S:
        case F_IQ2_S: case F_IQ4_XS: case F_Q2_0: return true;
        default: return false;
    }
}

static size_t al16(size_t x) { return (x + 15) & ~(size_t) 15; }

RLayout r_layout(int fmt, int64_t cols) {
    RLayout L;
    L.fmt = fmt;
    L.nsb = (int) (cols / 32);
    const size_t n = L.nsb, nb = cols / 256;
    // field sizes in order a, b, c, d, e
    size_t sz[5] = {0, 0, 0, 0, 0};
    switch (fmt) {
        case F_Q4_K: sz[0] = 16 * n; sz[1] = 2 * n; sz[2] = 4 * nb; break;
        case F_Q5_K: sz[0] = 16 * n; sz[1] = 4 * n; sz[2] = 2 * n; sz[3] = 4 * nb; break;
        case F_Q6_K: sz[0] = 16 * n; sz[1] = 8 * n; sz[2] = 2 * n; sz[3] = 2 * nb; break;
        case F_Q8_0: sz[0] = 32 * n; sz[1] = 2 * n; break;
        case F_IQ4_NL: sz[0] = 16 * n; sz[1] = 2 * n; break;
        case F_IQ4_XS: sz[0] = 16 * n; sz[1] = n; sz[2] = 2 * nb; break;
        case F_IQ2_S: sz[0] = 4 * n; sz[1] = 4 * n; sz[2] = n; sz[3] = n; sz[4] = 2 * nb; break;
        case F_IQ3_XXS: sz[0] = 8 * n; sz[1] = 4 * n; sz[2] = 2 * nb; break;
        case F_IQ3_S: sz[0] = 8 * n; sz[1] = 4 * n; sz[2] = n; sz[3] = n; sz[4] = 2 * nb; break;
        case F_Q2_0: sz[0] = 8 * n; sz[1] = 2 * (cols / 64); break;
        default: throw std::runtime_error("R layout: unsupported format");
    }
    uint32_t * offs[5] = {&L.off_a, &L.off_b, &L.off_c, &L.off_d, &L.off_e};
    size_t o = 0;
    for (int i = 0; i < 5; ++i) {
        *offs[i] = (uint32_t) o;
        o = al16(o + sz[i]);
    }
    L.row_bytes = o;
    return L;
}

static void scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// 32 4-bit codes (element order) into a nibble slot: byte j = code[j] | code[j+16] << 4
static void put_nibbles(const uint8_t * code, uint8_t * slot) {
    for (int j = 0; j < 16; ++j) slot[j] = (uint8_t) ((code[j] & 15) | ((code[j + 16] & 15) << 4));
}

static void repack_row(const RLayout & L, const uint8_t * s, uint8_t * d) {
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
                put_nibbles(code, A + 16 * sb);
                scale_min_k4(k, blk + 4, B[2 * sb], B[2 * sb + 1]);
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
                put_nibbles(code, A + 16 * sb);
                memcpy(B + 4 * sb, &hb, 4);
                scale_min_k4(k, blk + 4, C[2 * sb], C[2 * sb + 1]);
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
                put_nibbles(code, A + 16 * sb);
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

void r_repack(const RLayout & L, const uint8_t * src, size_t src_rb, int64_t nrows, uint8_t * dst) {
    for (int64_t r = 0; r < nrows; ++r) repack_row(L, src + r * src_rb, dst + r * L.row_bytes);
}

}  // namespace bnk
