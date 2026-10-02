// QSA: the sparse attention of qwen4exp's full-attention layers.
//
// Every compress_ratio (r = 4) consecutive cells form a block; a block's key is the mean of its cells' raw
// indexer keys, RMS-normalized and rotated at the block's first position. A query at position q scores every
// complete block before its own (tail) block with sum_h relu(q_h . K_b) over the indexer heads, keeps the
// top_k / r best blocks, and attends to their cells plus its tail (positions tail_start..q, always visible).
// With q + 1 <= top_k + r - 1 cells the selection covers everything: the attention is dense.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace bnk {

struct QsaShape {
    int H, Hkv, D;          // attention heads, KV heads, head dim
    int ih, id;             // indexer heads, indexer dim
    int ratio, top_blocks;  // r, top_k / r
    int n_rot;
    float rope_base, eps;
    int dense_cells() const { return top_blocks * ratio + ratio - 1; }  // with this many cells or fewer: dense
};

// raw indexer keys k [T][id] into kraw rows pos0+t (fp16)
void qsa_store_keys(const float * k, half * kraw, int T, int id, const int * pos0, cudaStream_t s);
// pooled block keys for the blocks that end inside [pos0, pos0 + T): pooled[b][id]
void qsa_pool(const half * kraw, const float * k_norm, float * pooled, int T, const QsaShape & sh, const int * pos0,
              cudaStream_t s);
// indexer queries in place: q [T][ih][id] normalized and rotated at pos0 + t
void qsa_queries(float * q, const float * q_norm, int T, const QsaShape & sh, const int * pos0, cudaStream_t s);
// for queries t < T that need it (q + 1 > dense_cells): block scores and the top block list sel[t][top_blocks]
// (scratch: scores [T][max_blocks]); n_sel[t] = number of selected blocks (0 = dense row)
void qsa_select(const float * q, const float * pooled, float * scores, int max_blocks, int32_t * sel, int32_t * n_sel,
                int T, const QsaShape & sh, const int * pos0, int t_off, cudaStream_t s);
// attention over the selected blocks + tail (or every cell for dense rows), times sigmoid(gate); split-K
// decode kernel (scratch as attention()).
void qsa_attention(const float * q, const half * kc, const half * vc, const float * qfull_gate, const int32_t * sel,
                   const int32_t * n_sel, float * out, int T, const QsaShape & sh, const int * pos0, float scale,
                   float * scratch, cudaStream_t s);
// one warp per (t, head) prompt variant (no scratch); rows t in [t0, t1)
void qsa_attention_prefill(const float * q, const half * kc, const half * vc, const float * qfull_gate,
                           const int32_t * sel, const int32_t * n_sel, float * out, int t0, int t1,
                           const QsaShape & sh, int pos0, float scale, cudaStream_t s);

}  // namespace bnk
