// Inkling 2's attention on the tensor cores (sm_70 mma.m8n8k4), after dense/dense_attn.cu: a block is 8 query rows
// x the G query heads of one KV head (a warp each), over one split of the keys; each 32-key tile of K and V goes
// through shared memory once for all of them. Inkling's parts: head dim 128, the caches as rings (a key's cell is
// its position modulo the ring), the window (keys at most window - 1 positions back), the relative-position bias
// gathered from a per-(row, head) table by distance (llama.cpp's dense bias: R . rel_bias[min(d, extent)] - sink,
// the sink logit folded in), and the sink's zero logit in the softmax's denominator. No output gate here.
//
// Fragment layout as dense_attn.cu: a warp is 4 quad pairs (qp = (lane >> 2) & 3) each computing an 8 x 8 tile;
// thread j = (lane & 3) | ((lane >> 4) << 2) supplies 4 consecutive k values of row j (A) and of column j (B); a
// thread holds c[i] for row (lane & 1) + (i & 2) + ((lane >> 4) << 2) and column (i & 4) + (lane & 2) + (i & 1).
#include "inkling/ink_attn.h"

#include <cfloat>
#include <stdexcept>

#include "kernels/gemv.h"
#include "kernels/mma.cuh"

namespace bnk {

namespace {

constexpr int kD = 128, kRows = 8, kKeys = 32;
constexpr int kKS = kD + 8;      // K tile row stride (halves)
constexpr int kVS = kKeys + 4;   // V^T row stride (halves)
constexpr float kLog2e = 1.4426950408889634f;

struct Pos0 {
    int value;
    const int * dev;
    __device__ __forceinline__ int get() const { return dev ? *dev : value; }
};

// table[t][h][d] = (R[t][h] . rel_bias[d] - sink[t][h]) * log2(e) for d < extent, -sink * log2(e) at d = extent;
// a block: 256 distances x 32 (t, h) rows, both operands in shared memory (n_rel = 16)
__global__ void k_rel_table(const float * R, const float * rel_bias, const float * sink, int TH, int n_rel, int extent,
                            float * table) {
    __shared__ float rb[256][17];
    __shared__ float rr[32][17];
    const int d0 = blockIdx.x * 256, th0 = blockIdx.y * 32;
    for (int i = threadIdx.x; i < 256 * n_rel; i += blockDim.x) {
        const int d = d0 + i / n_rel;
        rb[i / n_rel][i % n_rel] = d < extent ? rel_bias[(int64_t) d * n_rel + i % n_rel] : 0.f;
    }
    for (int i = threadIdx.x; i < 32 * n_rel; i += blockDim.x) {
        const int th = th0 + i / n_rel;
        rr[i / n_rel][i % n_rel] = th < TH ? R[(int64_t) th * n_rel + i % n_rel] : 0.f;
    }
    __syncthreads();
    const int d = d0 + threadIdx.x;
    if (d > extent) return;
    for (int r = 0; r < 32 && th0 + r < TH; ++r) {
        float b = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) b += rr[r][i] * rb[threadIdx.x][i];
        table[(int64_t) (th0 + r) * (extent + 1) + d] = (b - sink[th0 + r]) * kLog2e;
    }
}

template <int G>
__global__ void __launch_bounds__(32 * G) k_attn_tc(const float * __restrict__ q, const half * __restrict__ kc,
        const half * __restrict__ vc, const float * __restrict__ table, float * __restrict__ out,
        float * __restrict__ part_o, float * __restrict__ part_ml, int N, int H, int Hkv, int ring, int extent,
        int window, Pos0 p0, float scale, int nsplit) {
    __shared__ __align__(16) half ks[kKeys * kKS];
    __shared__ __align__(16) half vt[kD * kVS];
    __shared__ __align__(16) half ps[G][kRows * kKeys];
    constexpr int kThreads = 32 * G;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int qp = (lane >> 2) & 3, j = (lane & 3) | ((lane >> 4) << 2);
    const int hk = blockIdx.y, h = hk * G + warp, split = blockIdx.z;
    const int pos0 = p0.get();
    const int r0 = blockIdx.x * kRows;
    const int my_row = r0 + j;
    const int n_kv = min(pos0 + N, pos0 + r0 + kRows);   // keys the block's rows may see, exclusive
    const int lo = window > 0 ? max(0, pos0 + r0 - window + 1) : 0;
    const int per = nsplit > 1 ? ((n_kv - lo + nsplit - 1) / nsplit + kKeys - 1) / kKeys * kKeys : n_kv - lo;
    const int kbeg = lo + split * per, kend = min(n_kv, kbeg + per);
    const int KVD = Hkv * kD;

    const float scale2 = scale * kLog2e;
    uint32_t qa[16];
    {
        const int rr = min(my_row, N - 1);
        const float * qr = q + ((int64_t) rr * H + h) * kD + qp * 32;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 f = *(const float4 *) (qr + 4 * i);
            qa[2 * i] = h2u(__floats2half2_rn(sat_h(f.x * scale2), sat_h(f.y * scale2)));
            qa[2 * i + 1] = h2u(__floats2half2_rn(sat_h(f.z * scale2), sat_h(f.w * scale2)));
        }
    }
    const int crow0 = (lane & 1) + ((lane >> 4) << 2), crow1 = crow0 + 2;
    const int qpos0 = pos0 + r0 + crow0, qpos1 = qpos0 + 2;
    const int trow0 = min(r0 + crow0, N - 1), trow1 = min(r0 + crow1, N - 1);
    const float * tab0 = table + ((int64_t) trow0 * H + h) * (extent + 1);
    const float * tab1 = table + ((int64_t) trow1 * H + h) * (extent + 1);
    float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.f, l1 = 0.f;
    float o[4][8];
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int i = 0; i < 8; ++i) o[n][i] = 0.f;

    for (int k0 = kbeg; k0 < kend; k0 += kKeys) {
        __syncthreads();
        for (int idx = threadIdx.x; idx < kKeys * (kD / 8); idx += kThreads) {
            const int key = idx / (kD / 8), c8 = idx % (kD / 8);
            const int kk = min(k0 + key, kend - 1);
            *(uint4 *) (ks + key * kKS + c8 * 8) = *(const uint4 *) (kc + (int64_t) (kk % ring) * KVD + hk * kD + c8 * 8);
        }
        for (int idx = threadIdx.x; idx < kKeys * (kD / 8); idx += kThreads) {
            const int key = idx % kKeys, c8 = idx / kKeys;
            const int kk = min(k0 + key, kend - 1);
            const uint4 v = *(const uint4 *) (vc + (int64_t) (kk % ring) * KVD + hk * kD + c8 * 8);
            const half * vh = (const half *) &v;
#pragma unroll
            for (int i = 0; i < 8; ++i) vt[(c8 * 8 + i) * kVS + key] = vh[i];
        }
        __syncthreads();
        float s[4][8];
#pragma unroll
        for (int st = 0; st < 4; ++st)
#pragma unroll
            for (int i = 0; i < 8; ++i) s[st][i] = 0.f;
#pragma unroll
        for (int st2 = 0; st2 < 4; st2 += 2) {
            const half * kr0 = ks + (st2 * 8 + j) * kKS, * kr1 = kr0 + 8 * kKS;
            uint32_t kb0[16], kb1[16];
            const uint4 * kp0 = (const uint4 *) (kr0 + qp * 32), * kp1 = (const uint4 *) (kr1 + qp * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ((uint4 *) kb0)[i] = kp0[i];
                ((uint4 *) kb1)[i] = kp1[i];
            }
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                mma884(s[st2], qa[2 * kk], qa[2 * kk + 1], kb0[2 * kk], kb0[2 * kk + 1]);
                mma884(s[st2 + 1], qa[2 * kk], qa[2 * kk + 1], kb1[2 * kk], kb1[2 * kk + 1]);
            }
        }
#pragma unroll
        for (int st = 0; st < 4; ++st)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                s[st][i] += __shfl_xor_sync(0xffffffff, s[st][i], 4);
                s[st][i] += __shfl_xor_sync(0xffffffff, s[st][i], 8);
            }
        // the bias by distance, and the masks (causal, window, the split's end)
#pragma unroll
        for (int st = 0; st < 4; ++st)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int key = k0 + st * 8 + (i & 4) + (lane & 2) + (i & 1);
                const bool r1 = (i & 2) != 0;
                const int qpos = r1 ? qpos1 : qpos0;
                const int d = qpos - key;
                if (d < 0 || key >= kend || (window > 0 && d >= window)) {
                    s[st][i] = -FLT_MAX;
                } else {
                    const float * tab = r1 ? tab1 : tab0;
                    s[st][i] += tab[d < extent ? d : extent];
                }
            }
        float tm0 = -FLT_MAX, tm1 = -FLT_MAX;
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
        if (!__all_sync(0xffffffff, c0 == 1.f && c1 == 1.f)) {
#pragma unroll
            for (int n = 0; n < 4; ++n)
#pragma unroll
                for (int i = 0; i < 8; ++i) o[n][i] *= (i & 2) ? c1 : c0;
        }
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
        uint32_t pa[8][2];
#pragma unroll
        for (int sstep = 0; sstep < 8; ++sstep) {
            const uint2 v = *(const uint2 *) (ps[warp] + j * kKeys + sstep * 4);
            pa[sstep][0] = v.x;
            pa[sstep][1] = v.y;
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const half * vr = vt + ((qp + 4 * n) * 8 + j) * kVS;
#pragma unroll
            for (int sstep = 0; sstep < 8; ++sstep) {
                const uint2 b = *(const uint2 *) (vr + sstep * 4);
                mma884(o[n], pa[sstep][0], pa[sstep][1], b.x, b.y);
            }
        }
    }
    if (nsplit == 1) {   // with the sink's zero logit: M = max(m, 0), L = l 2^(m - M) + 2^(-M)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int rr = (i & 2) ? crow1 : crow0;
            const int row = r0 + rr;
            if (row >= N) continue;
            const float m = (i & 2) ? m1 : m0, l = (i & 2) ? l1 : l0;
            const float M = fmaxf(m, 0.f), c = l > 0.f ? exp2f(m - M) : 0.f;
            const float inv = 1.f / (l * c + exp2f(-M));
            float * orow = out + ((int64_t) row * H + h) * kD;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int d = (qp + 4 * n) * 8 + (i & 4) + (lane & 2) + (i & 1);
                orow[d] = o[n][i] * c * inv;
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int rr = (i & 2) ? crow1 : crow0;
            const int row = r0 + rr;
            if (row >= N) continue;
            const int64_t th = ((int64_t) split * N + row) * H + h;
            float * po = part_o + th * kD;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int d = (qp + 4 * n) * 8 + (i & 4) + (lane & 2) + (i & 1);
                po[d] = o[n][i];
            }
            if (qp == 0 && (lane & 2) == 0 && (i & 5) == 0) {
                part_ml[th * 2] = (i & 2) ? m1 : m0;
                part_ml[th * 2 + 1] = (i & 2) ? l1 : l0;
            }
        }
    }
}

// out = (sum_s O_s c_s) / (sum_s l_s c_s + 2^(-M)), c_s = 2^(m_s - M), M over the splits and the sink's 0
__global__ void k_attn_combine(const float * __restrict__ part_o, const float * __restrict__ part_ml,
                               float * __restrict__ out, int N, int H, int nsplit) {
    const int th = blockIdx.x, i = threadIdx.x;
    float M = 0.f;
    for (int s = 0; s < nsplit; ++s) {
        const int64_t e = ((int64_t) s * N * H + th) * 2;
        if (part_ml[e + 1] > 0.f) M = fmaxf(M, part_ml[e]);
    }
    float L = exp2f(-M), o = 0.f;
    for (int s = 0; s < nsplit; ++s) {
        const int64_t e = ((int64_t) s * N * H + th) * 2;
        const float l = part_ml[e + 1];
        if (l == 0.f) continue;
        const float c = exp2f(part_ml[e] - M);
        L += l * c;
        o += part_o[((int64_t) s * N * H + th) * kD + i] * c;
    }
    out[(int64_t) th * kD + i] = o / L;
}

template <int G>
void launch(const float * q, const half * kc, const half * vc, const float * table, float * out, float * part_o,
            float * part_ml, int N, int H, int Hkv, int ring, int extent, int window, Pos0 p0, float scale, int nsplit,
            cudaStream_t s) {
    const dim3 grid((N + kRows - 1) / kRows, Hkv, nsplit);
    k_attn_tc<G><<<grid, 32 * G, 0, s>>>(q, kc, vc, table, out, part_o, part_ml, N, H, Hkv, ring, extent, window, p0,
                                         scale, nsplit);
    if (nsplit > 1) k_attn_combine<<<N * H, kD, 0, s>>>(part_o, part_ml, out, N, H, nsplit);
}

}  // namespace

size_t ink_attn_scratch_floats(int T, int H, int extent, int max_splits) {
    return (size_t) T * H * (extent + 1) + (size_t) max_splits * T * H * (kD + 2);
}

void ink_attention_tc(const float * q, const float * R, const float * sink, const half * K, const half * V, int ring,
                      const float * rel_bias, int extent, int n_rel, int window, const int * pos0_dev, int pos0,
                      int T, int H, int KVH, int D, float scale, int nsplit, float * out, float * scratch,
                      cudaStream_t s) {
    if (D != kD || H % KVH || n_rel > 32) throw std::runtime_error("ink_attention_tc: unsupported shape");
    float * table = scratch;
    float * part_o = scratch + (size_t) T * H * (extent + 1);
    float * part_ml = part_o + (size_t) nsplit * T * H * kD;
    if (n_rel != 16) throw std::runtime_error("ink_attention_tc: 16 relative channels");
    k_rel_table<<<dim3((extent + 256) / 256, (T * H + 31) / 32), 256, 0, s>>>(R, rel_bias, sink, T * H, n_rel, extent, table);
    const Pos0 p0{pos0, pos0_dev};
    switch (H / KVH) {
        case 1: launch<1>(q, K, V, table, out, part_o, part_ml, T, H, KVH, ring, extent, window, p0, scale, nsplit, s); break;
        case 2: launch<2>(q, K, V, table, out, part_o, part_ml, T, H, KVH, ring, extent, window, p0, scale, nsplit, s); break;
        case 3: launch<3>(q, K, V, table, out, part_o, part_ml, T, H, KVH, ring, extent, window, p0, scale, nsplit, s); break;
        case 4: launch<4>(q, K, V, table, out, part_o, part_ml, T, H, KVH, ring, extent, window, p0, scale, nsplit, s); break;
        default: throw std::runtime_error("ink_attention_tc: query heads per KV head");
    }
    check_launch("ink_attention_tc");
}

}  // namespace bnk
