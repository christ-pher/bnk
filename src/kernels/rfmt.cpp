#include "kernels/rfmt.h"
#include "kernels/rfmt_impl.h"

#include <cstring>
#include <stdexcept>

namespace bnk {


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

void r_repack(const RLayout & L, const uint8_t * src, size_t src_rb, int64_t nrows, uint8_t * dst) {
    for (int64_t r = 0; r < nrows; ++r) rfmt_repack_row(L, src + r * src_rb, dst + r * L.row_bytes);
}

}  // namespace bnk
