// "R" layouts: a lossless, GPU-friendly re-arrangement of ggml quant rows.
//
// Each row becomes structure-of-arrays over its 32-weight sub-blocks: the 4-bit (or 8-bit) codes of
// sub-block s sit in one 16-byte (32-byte) slot, so a lane loads them with a single vector load, and
// the small fields (high bits, signs, scales, super-block scales) follow in their own arrays. Nibble
// slots hold element j in the low nibble of byte j and element j+16 in the high nibble. Rows are padded
// to 16 bytes. Dequantization stays bit-exact (tests/test_dequant.cu).
#pragma once

#include <cstddef>
#include <cstdint>

namespace bnk {

struct RLayout {
    int fmt = 0;           // ggml type of the source
    int nsb = 0;           // 32-weight sub-blocks per row
    size_t row_bytes = 0;  // padded
    // field offsets inside a row (bytes); unused fields are 0
    uint32_t off_a = 0, off_b = 0, off_c = 0, off_d = 0, off_e = 0;
};

bool r_supported(int fmt);
RLayout r_layout(int fmt, int64_t cols);
// Converts `nrows` ggml rows (src row stride src_rb) into R rows at dst (stride L.row_bytes).
void r_repack(const RLayout & L, const uint8_t * src, size_t src_rb, int64_t nrows, uint8_t * dst);

}  // namespace bnk
