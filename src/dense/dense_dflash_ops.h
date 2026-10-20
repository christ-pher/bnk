// The DFlash2 drafter's kernels (dense_dflash_ops.cu).
#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace bnk {

// block start, first key position, keys, anchor token; temperature (0: greedy drafts) and a uniform per draft
struct DFlashParams { int s, lo, nkeys, anchor; float temp, u[8]; };

void dflash_conv(const float * in, const float * dyn, const float * base, float * out, int T, int E, int gsize, int side,
                 cudaStream_t s);
void dflash_headnorm_rope(float * x, int rows, int nh, int D, const float * w, const int * pos0_dev, float theta,
                          float eps, cudaStream_t s);
void dflash_kv_store(const float * k, const float * v, half * kring, half * vring, int rows, int n, int ring,
                     const int * pos0_dev, cudaStream_t s);
// S: scratch [H][R][maxk]
void dflash_attention(const float * q, const half * kring, const half * vring, float * S, float * out,
                      const DFlashParams * par_dev, int R, int H, int HKV, int D, int ring, int win, int maxk,
                      cudaStream_t s);
// per row: the top K of the 2K candidates (ids, vals) and (rids + off, rvals), into ids / vals (K <= 16)
void dflash_merge_topk(int32_t * ids, float * vals, const int32_t * rids, const float * rvals, int off, int rows, int K,
                       cudaStream_t s);
// qout [n][K]: each position's distribution over its candidates (sampled drafts)
void dflash_select(const int32_t * cand, const float * unary, const float * hp, const half * pre, const half * succ,
                   const DFlashParams * par_dev, int n, int K, int rank, int32_t * out, float * prob, float * qout,
                   cudaStream_t s);

}  // namespace bnk
