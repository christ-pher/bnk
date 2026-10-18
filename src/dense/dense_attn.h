// Attention on the tensor cores over the KV cache (the dense engine's prompt chunks and decode windows).
#pragma once

#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace bnk {

// Rows t < N at positions pos0 + t over the KV cache rows [0, pos0 + t] (causal), then the output gate:
// out[t][h][:] = softmax(q k^T * scale) v * sigmoid(gate). q [N][H][D], qfull_gate [N][H][2D] (the gate in the
// second half of each head), kcache / vcache [pos][Hkv][D] (fp16, or 8-bit cells when kv_q8()). False when
// unsupported (D != 256, an unsupported GQA ratio): run attention_prefill then.
bool attention_prefill_tc(const float * q, const half * kcache, const half * vcache, const float * qfull_gate, float * out,
                          int N, int H, int Hkv, int D, int pos0, float scale, cudaStream_t s);

// The same for a decode window of T <= 8 rows, pos0 in device memory (graph-friendly): the keys are split over
// decode_splits() blocks per KV head (BNK_DEC_SPLITS, default 64: 85 / 84 / 81 tok/s at 64K for 64 / 128 / 256; at most kDecodeSplitsMax), combined through
// `scratch` (attention_decode_tc_scratch_floats floats, sized for the maximum).
constexpr int kDecodeSplitsMax = 256;
int decode_splits();
size_t attention_decode_tc_scratch_floats(int T, int H, int D);
bool attention_decode_tc(const float * q, const half * kcache, const half * vcache, const float * qfull_gate, float * out,
                         int T, int H, int Hkv, int D, const int * pos0, float scale, float * scratch, cudaStream_t s);

}  // namespace bnk
