// Inkling 2's attention on the tensor cores (ink_attn.cu): prompt chunks and decode windows, split over the keys.
#pragma once

#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace bnk {

// scratch floats for T rows, H heads, a bias extent and up to max_splits key splits
size_t ink_attn_scratch_floats(int T, int H, int extent, int max_splits);
// out [T][H][128] = attention of q [T][H][128] over the ring caches (pos0_dev: device memory, else pos0), with the
// relative bias R [T][H][n_rel] . rel_bias [extent][n_rel] by distance minus sink [T][H], the window (0: none) and
// the sink's zero logit; nsplit key splits (combined in a second pass when > 1)
void ink_attention_tc(const float * q, const float * R, const float * sink, const half * K, const half * V, int ring,
                      const float * rel_bias, int extent, int n_rel, int window, const int * pos0_dev, int pos0,
                      int T, int H, int KVH, int D, float scale, int nsplit, float * out, float * scratch,
                      cudaStream_t s);

}  // namespace bnk
