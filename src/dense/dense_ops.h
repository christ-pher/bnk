// Kernels of the dense engine's residual stream.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/gemv.h"

namespace bnk {

// res[t] += pend[t] (when pend is given), then xn[t] = rmsnorm(res[t]) * w and its quantization into q (rows of E
// floats; one pass)
void dense_norm_q(float * res, const float * pend, const float * w, float * xn, int T, int E, float eps, ActQ8 & q,
                  cudaStream_t s);
// y[i] += x[i]
void add_rows(float * y, const float * x, int64_t n, cudaStream_t s);
// y[i] += (float) a[i] + (float) b[i]  (two fp16 partial sums: both ranks of a tensor-parallel pair add the same
// rounded values, so their residuals stay identical)
void add_rows_f16x2(float * y, const half * a, const half * b, int64_t n, cudaStream_t s);

}  // namespace bnk
