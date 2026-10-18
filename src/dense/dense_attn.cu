// Prompt attention on the tensor cores (sm_70 mma.m8n8k4): flash attention over the fp16 KV cache for the dense
// engine's chunks. The one-warp-per-row kernel (kernels/ops.cu attention_prefill) runs on the CUDA cores at ~2
// TFLOPS; this one streams 32-key tiles through the tensor cores with the online softmax in registers.
//
// Fragment layout (as gemv_r.cu / moe_tc.cu): a warp is 4 quad pairs (qp = (lane >> 2) & 3), each computing an
// 8 x 8 tile; thread j = (lane & 3) | ((lane >> 4) << 2) supplies 4 consecutive k values of row j (A) and of
// column j (B). Thread holds c[i] for row (lane & 1) + (i & 2) + ((lane >> 4) << 2) and column
// (i & 4) + (lane & 2) + (i & 1) of its quad pair's tile.
//
// Block: 8 query rows x the query heads of one KV head (GQA group: 6 warps for 24 / 4), so a K/V tile staged in
// shared memory once serves every warp. Per 32-key tile: S = Q K^T as 4 sub-tiles of 8 keys (the quad pairs split
// the head dim and add up), the causal mask, the row max / sum update and the rescale of O in registers, P as
// fp16 A fragments (through shared memory), O += P V with the quad pairs splitting the 256 output columns (8
// column tiles each; V transposed in shared memory so the 4 keys of a B fragment are one 8-byte load). Then
// O / l, the output gate sigmoid, out [T][H][D] fp32.
#include "dense/dense_attn.h"

#include <cfloat>
#include <cstdio>

#include "kernels/mma.cuh"

namespace bnk {

namespace {

constexpr int kD = 256, kRows = 8, kKeys = 32;
constexpr int kKS = kD + 8;      // K tile row stride (halves): rows 4 banks apart, 16-byte loads of 8 rows conflict-free
constexpr int kVS = kKeys + 4;   // V^T row stride (halves): 8-byte aligned rows, few conflicts on the 8-byte loads
constexpr int kMaxGroup = 8;     // query heads per KV head (warps per block)

__device__ __forceinline__ float sigmoid_(float x) { return 1.f / (1.f + __expf(-x)); }

template <int G>
__global__ void __launch_bounds__(32 * G) attn_tc_k(const float * __restrict__ q, const half * __restrict__ kc,
                                                    const half * __restrict__ vc, const float * __restrict__ qfull,
                                                    float * __restrict__ out, int N, int H, int Hkv, int pos0, float scale) {
    __shared__ __align__(16) half ks[kKeys * kKS];       // the K tile [key][d]
    __shared__ __align__(16) half vt[kD * kVS];          // the V tile transposed [d][key]
    __shared__ __align__(16) half ps[G][kRows * kKeys];  // each warp's P tile [row][key]
    constexpr int kThreads = 32 * G;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int qp = (lane >> 2) & 3, j = (lane & 3) | ((lane >> 4) << 2);
    const int hk = blockIdx.y, h = hk * G + warp;
    const int r0 = blockIdx.x * kRows;                   // the block's first query row
    const int my_row = r0 + j;                           // the row this thread's A fragments stand for
    const int n_kv = min(pos0 + N, pos0 + r0 + kRows);   // keys the block's rows may see (causal), exclusive
    const int64_t kv_stride = (int64_t) Hkv * kD;

    // Q fragments: row my_row's dims of this quad pair's two 32-wide sub-blocks (qp, qp + 4), scaled, fp16; the
    // scale carries log2(e) so the softmax runs on exp2
    const float scale2 = scale * 1.4426950408889634f;
    uint32_t qa[2][16];
    {
        const int rr = min(my_row, N - 1);
        const float * qr = q + ((int64_t) rr * H + h) * kD;
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int d0 = (qp + 4 * b) * 32;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float4 f = *(const float4 *) (qr + d0 + 4 * i);
                qa[b][2 * i] = h2u(__floats2half2_rn(sat_h(f.x * scale2), sat_h(f.y * scale2)));
                qa[b][2 * i + 1] = h2u(__floats2half2_rn(sat_h(f.z * scale2), sat_h(f.w * scale2)));
            }
        }
    }
    const int crow0 = (lane & 1) + ((lane >> 4) << 2), crow1 = crow0 + 2;   // this thread's C rows (i & 2 == 0 / 2)
    const int qpos0 = pos0 + r0 + crow0, qpos1 = qpos0 + 2;
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.f, l1 = 0.f;
    float o[8][8];                                       // column tiles qp, qp + 4, ..., qp + 28 of the 32
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int i = 0; i < 8; ++i) o[n][i] = 0.f;

    for (int k0 = 0; k0 < n_kv; k0 += kKeys) {
        // ---- the K tile and the transposed V tile to shared memory (keys past n_kv read as the last valid key:
        // masked below)
        __syncthreads();   // the previous tile's reads are done
        // K: consecutive threads along a key's row (16-byte stores, conflict-free); V: consecutive threads along
        // the keys, so each transposed store of a warp is 32 consecutive halves of one d row (conflict-free)
        for (int idx = threadIdx.x; idx < kKeys * (kD / 8); idx += kThreads) {
            const int key = idx / (kD / 8), c8 = idx % (kD / 8);
            const int kk = min(k0 + key, n_kv - 1);
            *(uint4 *) (ks + key * kKS + c8 * 8) = __ldg((const uint4 *) (kc + ((int64_t) kk * Hkv + hk) * kD + c8 * 8));
        }
        for (int idx = threadIdx.x; idx < kKeys * (kD / 8); idx += kThreads) {
            const int key = idx % kKeys, c8 = idx / kKeys;
            const int kk = min(k0 + key, n_kv - 1);
            const uint4 v = __ldg((const uint4 *) (vc + ((int64_t) kk * Hkv + hk) * kD + c8 * 8));
            const half * vh = (const half *) &v;
#pragma unroll
            for (int i = 0; i < 8; ++i) vt[(c8 * 8 + i) * kVS + key] = vh[i];
        }
        __syncthreads();
        // ---- S = Q K^T for the 4 sub-tiles of 8 keys: thread j's B fragments are key (8 st + j)'s dims
        float s[4][8];
#pragma unroll
        for (int st = 0; st < 4; ++st)
#pragma unroll
            for (int i = 0; i < 8; ++i) s[st][i] = 0.f;
#pragma unroll
        for (int st2 = 0; st2 < 4; st2 += 2) {   // two sub-tiles' chains interleaved
            const half * kr0 = ks + (st2 * 8 + j) * kKS, * kr1 = kr0 + 8 * kKS;
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                uint32_t kb0[16], kb1[16];
                const uint4 * kp0 = (const uint4 *) (kr0 + (qp + 4 * b) * 32), * kp1 = (const uint4 *) (kr1 + (qp + 4 * b) * 32);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ((uint4 *) kb0)[i] = kp0[i];
                    ((uint4 *) kb1)[i] = kp1[i];
                }
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    mma884(s[st2], qa[b][2 * kk], qa[b][2 * kk + 1], kb0[2 * kk], kb0[2 * kk + 1]);
                    mma884(s[st2 + 1], qa[b][2 * kk], qa[b][2 * kk + 1], kb1[2 * kk], kb1[2 * kk + 1]);
                }
            }
        }
        // the quad pairs' quarters of the head dim added: every quad pair then holds the tile
#pragma unroll
        for (int st = 0; st < 4; ++st)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                s[st][i] += __shfl_xor_sync(0xffffffff, s[st][i], 4);
                s[st][i] += __shfl_xor_sync(0xffffffff, s[st][i], 8);
            }
        // ---- causal mask, row max over the tile (this thread's 4 columns per row and the other thread's 4)
        float tm0 = -FLT_MAX, tm1 = -FLT_MAX;
        if (k0 + kKeys > pos0 + r0) {   // the tile reaches the block's diagonal: mask (block-uniform branch)
#pragma unroll
            for (int st = 0; st < 4; ++st)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int key = k0 + st * 8 + (i & 4) + (lane & 2) + (i & 1);
                    const int qpos = (i & 2) ? qpos1 : qpos0;
                    if (key > qpos || key >= n_kv) s[st][i] = -FLT_MAX;
                }
        }
#pragma unroll
        for (int st = 0; st < 4; ++st)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i & 2) tm1 = fmaxf(tm1, s[st][i]); else tm0 = fmaxf(tm0, s[st][i]);
            }
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 2));
        const float mn0 = fmaxf(m0, tm0), mn1 = fmaxf(m1, tm1);
        const float c0 = mn0 == -FLT_MAX ? 1.f : exp2f(m0 - mn0), c1 = mn1 == -FLT_MAX ? 1.f : exp2f(m1 - mn1);
        float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
        for (int st = 0; st < 4; ++st)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const bool r1 = (i & 2) != 0;
                const float mn = r1 ? mn1 : mn0;
                const float p = (s[st][i] == -FLT_MAX || mn == -FLT_MAX) ? 0.f : exp2f(s[st][i] - mn);
                s[st][i] = p;
                if (r1) sum1 += p; else sum0 += p;
            }
        sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
        l0 = l0 * c0 + sum0;
        l1 = l1 * c1 + sum1;
        m0 = mn0;
        m1 = mn1;
        if (!__all_sync(0xffffffff, c0 == 1.f && c1 == 1.f)) {   // a row max moved: rescale O
#pragma unroll
            for (int n = 0; n < 8; ++n)
#pragma unroll
                for (int i = 0; i < 8; ++i) o[n][i] *= (i & 2) ? c1 : c0;
        }
        // ---- P to shared as [row][key] fp16 (quad pair 0 writes; the others hold the same values)
        if (qp == 0) {
#pragma unroll
            for (int st = 0; st < 4; ++st)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int rr = (i & 2) ? crow1 : crow0, key = st * 8 + (i & 4) + (lane & 2) + (i & 1);
                    ps[warp][rr * kKeys + key] = __float2half_rn(s[st][i]);
                }
        }
        __syncwarp();
        // ---- O += P V: A = P[row j][4 keys per step], B = V^T[column j of tile n][4 keys], 8 steps of 4 keys
        uint32_t pa[8][2];
#pragma unroll
        for (int sstep = 0; sstep < 8; ++sstep) {
            const uint2 v = *(const uint2 *) (ps[warp] + j * kKeys + sstep * 4);
            pa[sstep][0] = v.x;
            pa[sstep][1] = v.y;
        }
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const half * vr = vt + ((qp + 4 * n) * 8 + j) * kVS;
#pragma unroll
            for (int sstep = 0; sstep < 8; ++sstep) {
                const uint2 b = *(const uint2 *) (vr + sstep * 4);
                mma884(o[n], pa[sstep][0], pa[sstep][1], b.x, b.y);
            }
        }
    }
    // ---- out = O / l, gated; thread's C values: rows crow0 / crow1, column tile qp + 4 n, column within it
    const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int rr = (i & 2) ? crow1 : crow0;
        const int row = r0 + rr;
        if (row >= N) continue;
        const float inv = (i & 2) ? inv1 : inv0;
        const float * gate = qfull + ((int64_t) row * H + h) * 2 * kD + kD;
        float * orow = out + ((int64_t) row * H + h) * kD;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const int d = (qp + 4 * n) * 8 + (i & 4) + (lane & 2) + (i & 1);
            orow[d] = o[n][i] * inv * sigmoid_(gate[d]);
        }
    }
    (void) kv_stride;
}

}  // namespace

bool attention_prefill_tc(const float * q, const half * kcache, const half * vcache, const float * qfull_gate, float * out,
                          int N, int H, int Hkv, int D, int pos0, float scale, cudaStream_t s) {
    if (D != kD || H % Hkv != 0) return false;
    const int G = H / Hkv;
    const dim3 grid((N + kRows - 1) / kRows, Hkv);
    switch (G) {
        case 1: attn_tc_k<1><<<grid, 32 * 1, 0, s>>>(q, kcache, vcache, qfull_gate, out, N, H, Hkv, pos0, scale); break;
        case 2: attn_tc_k<2><<<grid, 32 * 2, 0, s>>>(q, kcache, vcache, qfull_gate, out, N, H, Hkv, pos0, scale); break;
        case 4: attn_tc_k<4><<<grid, 32 * 4, 0, s>>>(q, kcache, vcache, qfull_gate, out, N, H, Hkv, pos0, scale); break;
        case 6: attn_tc_k<6><<<grid, 32 * 6, 0, s>>>(q, kcache, vcache, qfull_gate, out, N, H, Hkv, pos0, scale); break;
        case 8: attn_tc_k<8><<<grid, 32 * 8, 0, s>>>(q, kcache, vcache, qfull_gate, out, N, H, Hkv, pos0, scale); break;
        default: return false;
    }
    (void) kMaxGroup;
    return true;
}

}  // namespace bnk
