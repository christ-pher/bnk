// Kernels of the dense engine's residual stream.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "kernels/gemv.h"

namespace bnk {

// res[t] += pend[t] (when pend is given), then xn[t] = rmsnorm(res[t]) * w and its quantization into q (rows of E
// floats; one pass)
void dense_norm_q(float * res, const float * pend, const float * w, float * xn, int T, int E, float eps, ActQ8 & q,
                  cudaStream_t s);
// y[i] += x[i]
void add_rows(float * y, const float * x, int64_t n, cudaStream_t s);

}  // namespace bnk
