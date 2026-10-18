// Attention on the tensor cores (sm_70 mma.m8n8k4) over the KV cache, for the dense engine's prompt chunks and
// decode windows. The one-warp-per-row prompt kernel and the split-KV decode kernel of kernels/ops.cu read the
// cache once per (row, head) on the CUDA cores; this one streams each 32-key tile once for 8 rows x the query
// heads of one KV head (the GQA group) through the tensor cores, with the online softmax in registers.
//
// Fragment layout (as gemv_r.cu / moe_tc.cu): a warp is 4 quad pairs (qp = (lane >> 2) & 3), each computing an
// 8 x 8 tile; thread j = (lane & 3) | ((lane >> 4) << 2) supplies 4 consecutive k values of row j (A) and of
// column j (B). Thread holds c[i] for row (lane & 1) + (i & 2) + ((lane >> 4) << 2) and column
// (i & 4) + (lane & 2) + (i & 1) of its quad pair's tile.
//
// Block: 8 query rows x G warps (one query head each) of a KV head, over the keys of one split (grid z): S = Q K^T
// as 4 sub-tiles of 8 keys (the quad pairs split the head dim and add up), the causal mask on the diagonal tiles,
// the row max / sum update and the rescale of O in registers, P as fp16 A fragments (through shared memory),
// O += P V with the quad pairs splitting the 256 output columns (8 column tiles each; V transposed in shared
// memory so the 4 keys of a B fragment are one 8-byte load). One split: out = O / l times the output gate;
// several: the split's (O, m, l) go to scratch and a second kernel combines them.
#include "dense/dense_attn.h"

#include <cfloat>
#include <cstdio>
#include <cstdlib>

#include "kernels/kvq.cuh"
#include "kernels/mma.cuh"
#include "kernels/ops.h"

namespace bnk {

namespace {

constexpr int kD = 256, kRows = 8, kKeys = 32;
constexpr int kKS = kD + 8;      // K tile row stride (halves): rows 4 banks apart, 16-byte loads of 8 rows conflict-free
constexpr int kVS = kKeys + 4;   // V^T row stride (halves): 8-byte aligned rows, few conflicts on the 8-byte loads

__device__ __forceinline__ float sigmoid_(float x) { return 1.f / (1.f + __expf(-x)); }

// pos0 from the host (prompt chunks) or from device memory (decode windows: the graph replays with new values)
struct Pos0 {
    int value;
    const int * dev;
    __device__ __forceinline__ int get() const { return dev ? *dev : value; }
};

template <int G, bool Q8>
__global__ void __launch_bounds__(32 * G) attn_tc_k(const float * __restrict__ q, const half * __restrict__ kc,
                                                    const half * __restrict__ vc, const float * __restrict__ qfull,
                                                    float * __restrict__ out, float * __restrict__ part_o,
                                                    float * __restrict__ part_ml, int N, int H, int Hkv, Pos0 p0,
                                                    float scale, int nsplit) {
    __shared__ __align__(16) half ks[kKeys * kKS];       // the K tile [key][d]
    __shared__ __align__(16) half vt[kD * kVS];          // the V tile transposed [d][key]
    __shared__ __align__(16) half ps[G][kRows * kKeys];  // each warp's P tile [row][key]
    constexpr int kThreads = 32 * G;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int qp = (lane >> 2) & 3, j = (lane & 3) | ((lane >> 4) << 2);
    const int hk = blockIdx.y, h = hk * G + warp, split = blockIdx.z;
    const int pos0 = p0.get();
    const int r0 = blockIdx.x * kRows;                   // the block's first query row
    const int my_row = r0 + j;                           // the row this thread's A fragments stand for
    const int n_kv = min(pos0 + N, pos0 + r0 + kRows);   // keys the block's rows may see (causal), exclusive
    // this split's keys: whole tiles, the last split takes the rest
    const int per = nsplit > 1 ? ((n_kv + nsplit - 1) / nsplit + kKeys - 1) / kKeys * kKeys : n_kv;
    const int kbeg = split * per, kend = min(n_kv, kbeg + per);

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

    for (int k0 = kbeg; k0 < kend; k0 += kKeys) {
        // ---- the K tile and the transposed V tile to shared memory (keys past the end read as the last valid key:
        // masked below). K: consecutive threads along a key's row (16-byte stores, conflict-free); V: consecutive
        // threads along the keys, so each transposed store of a warp is 32 consecutive halves of one d row
        __syncthreads();   // the previous tile's reads are done
        for (int idx = threadIdx.x; idx < kKeys * (kD / 8); idx += kThreads) {
            const int key = idx / (kD / 8), c8 = idx % (kD / 8);
            const int kk = min(k0 + key, kend - 1);
            *(uint4 *) (ks + key * kKS + c8 * 8) = kv_load8<Q8>(kc, (int64_t) kk * Hkv + hk, kD, c8 * 8);
        }
        for (int idx = threadIdx.x; idx < kKeys * (kD / 8); idx += kThreads) {
            const int key = idx % kKeys, c8 = idx / kKeys;
            const int kk = min(k0 + key, kend - 1);
            const uint4 v = kv_load8<Q8>(vc, (int64_t) kk * Hkv + hk, kD, c8 * 8);
            const half * vh = (const half *) &v;
#pragma unroll
            for (int i = 0; i < 8; ++i) vt[(c8 * 8 + i) * kVS + key] = vh[i];
        }
        __syncthreads();
        // ---- S = Q K^T for the 4 sub-tiles of 8 keys (two chains interleaved): thread j's B fragments are key
        // (8 st + j)'s dims
        float s[4][8];
#pragma unroll
        for (int st = 0; st < 4; ++st)
#pragma unroll
            for (int i = 0; i < 8; ++i) s[st][i] = 0.f;
#pragma unroll
        for (int st2 = 0; st2 < 4; st2 += 2) {
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
        // ---- the causal mask where the tile reaches the block's diagonal or the split's end (block-uniform)
        float tm0 = -FLT_MAX, tm1 = -FLT_MAX;
        if (k0 + kKeys > pos0 + r0 || k0 + kKeys > kend) {
#pragma unroll
            for (int st = 0; st < 4; ++st)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int key = k0 + st * 8 + (i & 4) + (lane & 2) + (i & 1);
                    const int qpos = (i & 2) ? qpos1 : qpos0;
                    if (key > qpos || key >= kend) s[st][i] = -FLT_MAX;
                }
        }
        // row max over the tile (this thread's 4 columns per row and the other thread's 4)
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
    // ---- the result: thread's C values are rows crow0 / crow1, column tile qp + 4 n, column within it
    if (nsplit == 1) {   // out = O / l, gated
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
    } else {   // this split's unnormalized O and (m, l) for the combine
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int rr = (i & 2) ? crow1 : crow0;
            const int row = r0 + rr;
            if (row >= N) continue;
            const int64_t th = ((int64_t) split * N + row) * H + h;
            float * po = part_o + th * kD;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int d = (qp + 4 * n) * 8 + (i & 4) + (lane & 2) + (i & 1);
                po[d] = o[n][i];
            }
            if (qp == 0 && (lane & 2) == 0 && (i & 5) == 0) {   // once per row
                part_ml[th * 2] = (i & 2) ? m1 : m0;
                part_ml[th * 2 + 1] = (i & 2) ? l1 : l0;
            }
        }
    }
}

// grid (N * H), block kD threads: out = (sum_s O_s c_s) / (sum_s l_s c_s), c_s = exp2(m_s - M), then the gate
__global__ void attn_tc_combine_k(const float * __restrict__ part_o, const float * __restrict__ part_ml,
                                  const float * __restrict__ qfull, float * __restrict__ out, int N, int H, int nsplit) {
    const int th = blockIdx.x, t = th / H, h = th % H, i = threadIdx.x;
    float M = -FLT_MAX;
    for (int s = 0; s < nsplit; ++s) {
        const int64_t e = ((int64_t) s * N * H + th) * 2;
        if (part_ml[e + 1] > 0.f) M = fmaxf(M, part_ml[e]);
    }
    float L = 0.f, o = 0.f;
    for (int s = 0; s < nsplit; ++s) {
        const int64_t e = ((int64_t) s * N * H + th) * 2;
        const float l = part_ml[e + 1];
        if (l == 0.f) continue;
        const float c = exp2f(part_ml[e] - M);
        L += l * c;
        o += part_o[((int64_t) s * N * H + th) * kD + i] * c;
    }
    const float gate = qfull[((int64_t) t * H + h) * 2 * kD + kD + i];
    out[((int64_t) t * H + h) * kD + i] = (L > 0.f ? o / L : 0.f) * sigmoid_(gate);
}

template <int G>
bool launch(const float * q, const half * kc, const half * vc, const float * qfull, float * out, float * part_o,
            float * part_ml, int N, int H, int Hkv, Pos0 p0, float scale, int nsplit, cudaStream_t s) {
    const dim3 grid((N + kRows - 1) / kRows, Hkv, nsplit);
    if (kv_q8()) attn_tc_k<G, true><<<grid, 32 * G, 0, s>>>(q, kc, vc, qfull, out, part_o, part_ml, N, H, Hkv, p0, scale, nsplit);
    else attn_tc_k<G, false><<<grid, 32 * G, 0, s>>>(q, kc, vc, qfull, out, part_o, part_ml, N, H, Hkv, p0, scale, nsplit);
    if (nsplit > 1) attn_tc_combine_k<<<N * H, kD, 0, s>>>(part_o, part_ml, qfull, out, N, H, nsplit);
    return true;
}

bool dispatch(const float * q, const half * kc, const half * vc, const float * qfull, float * out, float * part_o,
              float * part_ml, int N, int H, int Hkv, int D, Pos0 p0, float scale, int nsplit, cudaStream_t s) {
    if (D != kD || H % Hkv != 0 || N < 1) return false;
    switch (H / Hkv) {
        case 1: return launch<1>(q, kc, vc, qfull, out, part_o, part_ml, N, H, Hkv, p0, scale, nsplit, s);
        case 2: return launch<2>(q, kc, vc, qfull, out, part_o, part_ml, N, H, Hkv, p0, scale, nsplit, s);
        case 3: return launch<3>(q, kc, vc, qfull, out, part_o, part_ml, N, H, Hkv, p0, scale, nsplit, s);
        case 4: return launch<4>(q, kc, vc, qfull, out, part_o, part_ml, N, H, Hkv, p0, scale, nsplit, s);
        case 6: return launch<6>(q, kc, vc, qfull, out, part_o, part_ml, N, H, Hkv, p0, scale, nsplit, s);
        case 8: return launch<8>(q, kc, vc, qfull, out, part_o, part_ml, N, H, Hkv, p0, scale, nsplit, s);
        default: return false;
    }
}

}  // namespace

bool attention_prefill_tc(const float * q, const half * kcache, const half * vcache, const float * qfull_gate, float * out,
                          int N, int H, int Hkv, int D, int pos0, float scale, cudaStream_t s) {
    return dispatch(q, kcache, vcache, qfull_gate, out, nullptr, nullptr, N, H, Hkv, D, Pos0{pos0, nullptr}, scale, 1, s);
}

int decode_splits() {
    static const int n = [] {
        const int v = getenv("BNK_DEC_SPLITS") ? atoi(getenv("BNK_DEC_SPLITS")) : 64;
        return v < 1 ? 1 : v > kDecodeSplitsMax ? kDecodeSplitsMax : v;
    }();
    return n;
}

size_t attention_decode_tc_scratch_floats(int T, int H, int D) { return (size_t) kDecodeSplitsMax * T * H * (D + 2); }

bool attention_decode_tc(const float * q, const half * kcache, const half * vcache, const float * qfull_gate, float * out,
                         int T, int H, int Hkv, int D, const int * pos0, float scale, float * scratch, cudaStream_t s) {
    if (T < 1 || T > kRows) return false;
    const int ns = decode_splits();
    float * part_o = scratch;
    float * part_ml = scratch + (size_t) ns * T * H * D;
    return dispatch(q, kcache, vcache, qfull_gate, out, part_o, part_ml, T, H, Hkv, D, Pos0{0, pos0}, scale, ns, s);
}

}  // namespace bnk
