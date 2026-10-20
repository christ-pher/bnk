// Multimodal rotary positions (interleaved M-RoPE, Qwen3-VL): every cell has a time, row and column position;
// rotary pair p takes the time position when p % 3 == 0, the row when 1, the column when 2 (each while p is
// under 3x its section's size, the time position past that). Text cells have all three equal; an image's cells
// share its first position as time and add their row / column in the merged grid (docs/VISION.md).
#pragma once

#include <cstdint>

namespace bnk {

struct MRope {
    const int32_t * pos = nullptr;   // device, [cells][3] (time, row, column); nullptr: the cell index itself
    int sec[3] = {0, 0, 0};          // rotary pairs per section (rope.dimension_sections)
};

#ifdef __CUDACC__
__device__ __forceinline__ float mrope_pos(const MRope & m, int cell, int pair) {
    if (!m.pos) return (float) cell;
    const int s = pair % 3;
    const int which = pair < 3 * m.sec[s] ? s : 0;
    return (float) m.pos[(int64_t) cell * 3 + which];
}
#endif

}  // namespace bnk
