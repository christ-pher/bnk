// Prompt attention on the tensor cores (the dense engine's chunks).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace bnk {

// Rows t < N at positions pos0 + t over the fp16 KV cache rows [0, pos0 + t] (causal), then the output gate:
// out[t][h][:] = softmax(q k^T * scale) v * sigmoid(gate). q [N][H][D], qfull_gate [N][H][2D] (the gate in the
// second half of each head), kcache / vcache [pos][Hkv][D] fp16. False when unsupported (D != 256, 8-bit KV):
// run attention_prefill then.
bool attention_prefill_tc(const float * q, const half * kcache, const half * vcache, const float * qfull_gate, float * out,
                          int N, int H, int Hkv, int D, int pos0, float scale, cudaStream_t s);

}  // namespace bnk
