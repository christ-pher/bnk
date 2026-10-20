#include "vision/vision.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "core/util.h"
#include "ggml.h"

namespace bnk {

#define BLAS_CHECK(x)                                                                                    \
    do {                                                                                                 \
        cublasStatus_t s_ = (x);                                                                         \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                                 \
            throw std::runtime_error("cuBLAS error " + std::to_string((int) s_) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                                          \
    } while (0)

namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
// block-wide reduction (sum or max); `sh` holds 32 floats
template <bool MAX>
__device__ __forceinline__ float block_red(float v, float * sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = MAX ? warp_max(v) : warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = threadIdx.x < nw ? sh[threadIdx.x] : (MAX ? -FLT_MAX : 0.f);
    if (wid == 0) v = MAX ? warp_max(v) : warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

static inline int nblk(int64_t n, int t) { return (int) ((n + t - 1) / t); }

// patch p of the encoder's order (2x2 merge blocks row-major over the merged grid, each block row-major) ->
// its row and column in the patch grid
__host__ __device__ __forceinline__ void patch_rc(int p, int gw, int merge, int * r, int * c) {
    const int mm = merge * merge, blk = p / mm, in = p % mm, bw = gw / merge;
    *r = (blk / bw) * merge + in / merge;
    *c = (blk % bw) * merge + in % merge;
}

// x[p][:] = y[p][:] + bias + the position table bilinearly resized (corners aligned) to the gh x gw grid
__global__ void pos_embed_k(const float * y, const float * bias, const float * table, float * x, int E, int gh,
                            int gw, int side, int merge) {
    const int p = blockIdx.x;
    int r, c;
    patch_rc(p, gw, merge, &r, &c);
    const float fy = gh > 1 ? (float) r * (side - 1) / (gh - 1) : 0.f;
    const float fx = gw > 1 ? (float) c * (side - 1) / (gw - 1) : 0.f;
    const int y0 = (int) floorf(fy), x0 = (int) floorf(fx);
    const int y1 = min(y0 + 1, side - 1), x1 = min(x0 + 1, side - 1);
    const float dy = fy - y0, dx = fx - x0;
    const float w00 = (1 - dy) * (1 - dx), w01 = (1 - dy) * dx, w10 = dy * (1 - dx), w11 = dy * dx;
    const float * t00 = table + (int64_t) (y0 * side + x0) * E, * t01 = table + (int64_t) (y0 * side + x1) * E;
    const float * t10 = table + (int64_t) (y1 * side + x0) * E, * t11 = table + (int64_t) (y1 * side + x1) * E;
    for (int i = threadIdx.x; i < E; i += blockDim.x)
        x[(int64_t) p * E + i] = y[(int64_t) p * E + i] + bias[i] + w00 * t00[i] + w01 * t01[i] + w10 * t10[i] +
                                 w11 * t11[i];
}

// LayerNorm of fp32 rows (E wide) -> fp16
__global__ void layernorm_k(const float * x, const float * w, const float * b, half * y, int E, float eps) {
    __shared__ float sh[32];
    const float * xr = x + (int64_t) blockIdx.x * E;
    float s = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) s += xr[i];
    const float mean = block_red<false>(s, sh) / E;
    float v = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) { const float d = xr[i] - mean; v += d * d; }
    const float rstd = rsqrtf(block_red<false>(v, sh) / E + eps);
    for (int i = threadIdx.x; i < E; i += blockDim.x)
        y[(int64_t) blockIdx.x * E + i] = __float2half((xr[i] - mean) * rstd * w[i] + b[i]);
}

// qkv rows [N][3E] (fp32, bias not yet added) -> q, k, v [N][E] fp16; q and k rotated by the patch's row (first
// half of each head's rotary pairs) and column (second half), pairs (i, i + D/2)
__global__ void qkv_rope_k(const float * qkv, const float * bias, half * q, half * k, half * v, int E, int H,
                           int gw, int merge, float base) {
    const int p = blockIdx.x;
    int r, c;
    patch_rc(p, gw, merge, &r, &c);
    const int D = E / H, half_d = D / 2, quarter = D / 4;
    const float * row = qkv + (int64_t) p * 3 * E;
    for (int j = threadIdx.x; j < H * half_d; j += blockDim.x) {
        const int h = j / half_d, i = j % half_d;
        const int sec = i < quarter ? i : i - quarter;
        const float theta = (float) (i < quarter ? r : c) * powf(base, -2.f * sec / half_d);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        const int a = h * D + i, bb = a + half_d;
        for (int part = 0; part < 2; ++part) {
            const float x0 = row[part * E + a] + bias[part * E + a], x1 = row[part * E + bb] + bias[part * E + bb];
            half * o = part ? k : q;
            o[(int64_t) p * E + a] = __float2half(x0 * cs - x1 * sn);
            o[(int64_t) p * E + bb] = __float2half(x1 * cs + x0 * sn);
        }
    }
    for (int i = threadIdx.x; i < E; i += blockDim.x) v[(int64_t) p * E + i] = __float2half(row[2 * E + i] + bias[2 * E + i]);
}

// x += y + b
__global__ void residual_k(float * x, const float * y, const float * b, int E, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i] + b[i % E];
}

// y16 = gelu(y + b): tanh approximation (the ViT's MLP) or exact (the merger)
template <bool TANH>
__global__ void gelu_k(const float * y, const float * b, half * o, int E, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = y[i] + b[i % E];
    const float g = TANH ? 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)))
                         : 0.5f * x * (1.f + erff(x * 0.70710678f));
    o[i] = __float2half(g);
}

__global__ void add_bias_k(float * y, const float * b, int E, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += b[i % E];
}

// softmax of each row of S (n wide, fp16), in place: max and sum in one pass (online), then the probabilities
__global__ void softmax_k(half * S, int n) {
    __shared__ float shm[32], shl[32];
    half * s = S + (int64_t) blockIdx.x * n;
    float m = -FLT_MAX, l = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = __half2float(s[i]);
        if (x > m) { l = l * __expf(m - x) + 1.f; m = x; }
        else l += __expf(x - m);
    }
    const float gm = block_red<true>(m, shm);
    const float gl = block_red<false>(l * __expf(m - gm), shl);
    const float inv = 1.f / gl;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = __float2half(__expf(__half2float(s[i]) - gm) * inv);
}

float bf16_to_f32(uint16_t v) {
    const uint32_t u = (uint32_t) v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// a GGUF tensor's values as fp32 (F32, F16, BF16)
std::vector<float> to_f32(const TensorRef & t) {
    const int64_t n = t.ne[0] * t.nrows();
    std::vector<float> out(n);
    if (t.type == GGML_TYPE_F32) memcpy(out.data(), t.data, n * 4);
    else if (t.type == GGML_TYPE_BF16) for (int64_t i = 0; i < n; ++i) out[i] = bf16_to_f32(((const uint16_t *) t.data)[i]);
    else if (t.type == GGML_TYPE_F16) for (int64_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) t.data)[i]);
    else throw std::runtime_error("vision: " + t.name + " has an unsupported type " + ggml_type_str(t.type));
    return out;
}

}  // namespace

VisionEncoder::~VisionEncoder() {
    for (auto & s : stages_) if (s.host) cudaFreeHost(s.host);
    if (pos_table_) cudaFree(pos_table_);
    if (plain_) cudaFree(plain_);
    for (int i = 0; i < 2; ++i) {
        if (copied_[i]) cudaEventDestroy(copied_[i]);
        if (used_[i]) cudaEventDestroy(used_[i]);
    }
    if (blas_) cublasDestroy(blas_);
    if (st_) cudaStreamDestroy(st_);
    if (copy_) cudaStreamDestroy(copy_);
}

// parts: (tensor, as fp16); a null tensor is a placeholder filled by the caller (the combined patch weight)
void VisionEncoder::add_stage(const std::vector<std::pair<const TensorRef *, bool>> & parts) {
    Stage s;
    size_t off = 0;
    std::vector<std::vector<float>> vals;
    for (const auto & [t, f16] : parts) {
        vals.push_back(to_f32(*t));
        off = (off + 255) / 256 * 256;
        s.off.push_back(off);
        off += vals.back().size() * (f16 ? 2 : 4);
    }
    s.bytes = (off + 255) / 256 * 256;
    CUDA_CHECK(cudaHostAlloc((void **) &s.host, s.bytes, cudaHostAllocPortable));
    for (size_t i = 0; i < parts.size(); ++i) {
        const auto & v = vals[i];
        if (parts[i].second) {
            half * d = (half *) (s.host + s.off[i]);
            for (size_t j = 0; j < v.size(); ++j) {
                if (!(fabsf(v[j]) < 65504.f)) throw std::runtime_error("vision: " + parts[i].first->name + " does not fit fp16");
                d[j] = __float2half(v[j]);
            }
        } else {
            memcpy(s.host + s.off[i], v.data(), v.size() * 4);
        }
    }
    stage_max_ = std::max(stage_max_, s.bytes);
    stages_.push_back(std::move(s));
}

void VisionEncoder::load(const std::string & path, int device, VramBudget * budget, bool verbose) {
    DeviceScope ds(device);
    dev_ = device;
    budget_ = budget;
    const double t0 = now_ms();
    gguf_.open(path);
    if (gguf_.get_str("clip.projector_type") != "qwen3vl_merger")
        throw std::runtime_error("vision: projector '" + gguf_.get_str("clip.projector_type") + "' is not supported (qwen3vl_merger)");
    auto & c = cfg_;
    c.n_embd = (int) gguf_.get_u64("clip.vision.embedding_length");
    c.n_ff = (int) gguf_.get_u64("clip.vision.feed_forward_length");
    c.n_layer = (int) gguf_.get_u64("clip.vision.block_count");
    c.n_head = (int) gguf_.get_u64("clip.vision.attention.head_count");
    c.patch = (int) gguf_.get_u64("clip.vision.patch_size", 16);
    c.merge = (int) gguf_.get_u64("clip.vision.spatial_merge_size", 2);
    c.out_dim = (int) gguf_.get_u64("clip.vision.projection_dim");
    c.eps = (float) gguf_.get_f64("clip.vision.attention.layer_norm_epsilon", 1e-6);
    auto mean = gguf_.get_f32_arr("clip.vision.image_mean"), sd = gguf_.get_f32_arr("clip.vision.image_std");
    for (int i = 0; i < 3 && i < (int) mean.size() && i < (int) sd.size(); ++i) { c.mean[i] = mean[i]; c.std[i] = sd[i]; }
    for (int il = 0; il < c.n_layer; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        if (gguf_.find("v.deepstack." + std::to_string(il) + ".fc1.weight") || gguf_.find("v." + b + "deepstack_fc1.weight"))
            throw std::runtime_error("vision: deepstack layers are not supported");
    }
    const TensorRef & pos = gguf_.get("v.position_embd.weight");
    c.pos_side = (int) lrint(sqrt((double) pos.ne[1]));
    if (c.pos_side * c.pos_side != pos.ne[1] || pos.ne[0] != c.n_embd) throw std::runtime_error("vision: position table is not square");
    if (c.head_dim() % 4) throw std::runtime_error("vision: head dim not divisible by 4");

    // stage 0: the patch embedding (the two temporal taps summed: an image is its own second frame)
    {
        const TensorRef & w0 = gguf_.get("v.patch_embd.weight");
        add_stage({{&w0, true}, {&gguf_.get("v.patch_embd.bias"), false}});
        if (const TensorRef * w1 = gguf_.find("v.patch_embd.weight.1")) {
            std::vector<float> a = to_f32(w0), b = to_f32(*w1);
            half * d = (half *) (stages_[0].host + stages_[0].off[0]);
            for (size_t j = 0; j < a.size(); ++j) d[j] = __float2half(a[j] + b[j]);
        }
    }
    for (int il = 0; il < c.n_layer; ++il) {
        const std::string b = "v.blk." + std::to_string(il) + ".";
        auto T = [&](const char * n) { return &gguf_.get(b + n); };
        add_stage({{T("attn_qkv.weight"), true}, {T("attn_out.weight"), true}, {T("ffn_up.weight"), true},
                   {T("ffn_down.weight"), true}, {T("attn_qkv.bias"), false}, {T("attn_out.bias"), false},
                   {T("ffn_up.bias"), false}, {T("ffn_down.bias"), false}, {T("ln1.weight"), false},
                   {T("ln1.bias"), false}, {T("ln2.weight"), false}, {T("ln2.bias"), false}});
    }
    add_stage({{&gguf_.get("v.post_ln.weight"), false}, {&gguf_.get("v.post_ln.bias"), false},
               {&gguf_.get("mm.0.weight"), true}, {&gguf_.get("mm.0.bias"), false}, {&gguf_.get("mm.2.weight"), true},
               {&gguf_.get("mm.2.bias"), false}});
    if (gguf_.get("mm.0.weight").ne[0] != c.n_embd * c.merge * c.merge || gguf_.get("mm.2.weight").ne[1] != c.out_dim)
        throw std::runtime_error("vision: merger shapes do not match");

    std::vector<float> table = to_f32(pos);
    CUDA_CHECK(cudaMalloc(&pos_table_, table.size() * 4));
    CUDA_CHECK(cudaMemcpy(pos_table_, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&copied_[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&used_[i], cudaEventDisableTiming));
    }
    BLAS_CHECK(cublasCreate(&blas_));
    BLAS_CHECK(cublasSetStream(blas_, st_));
    BLAS_CHECK(cublasSetMathMode(blas_, CUBLAS_TENSOR_OP_MATH));
    if (budget_) arena_.reserve((size_t) 8 << 30, budget_, "vision scratch");
    loaded_ = true;
    size_t host = 0;
    for (auto & s : stages_) host += s.bytes;
    if (verbose)
        fprintf(stderr, "bnk: vision encoder %s: %d layers x %d, %d heads, out %d; %.0f MiB page-locked, %.1f s\n",
                gguf_.get_str("general.name", path).c_str(), c.n_layer, c.n_embd, c.n_head, c.out_dim, host / 1048576.0,
                (now_ms() - t0) / 1000);
}

void * VisionEncoder::scratch(size_t bytes) {
    if (budget_) {
        arena_.ensure(bytes);
        return arena_.ptr();
    }
    if (bytes > plain_bytes_) {
        if (plain_) cudaFree(plain_);
        plain_ = nullptr;
        CUDA_CHECK(cudaMalloc(&plain_, bytes));
        plain_bytes_ = bytes;
    }
    return plain_;
}

void VisionEncoder::release_scratch() {
    if (budget_) arena_.shrink_to(0);
}

void VisionEncoder::fetch(int s, int buf) {
    CUDA_CHECK(cudaStreamWaitEvent(copy_, used_[buf], 0));
    CUDA_CHECK(cudaMemcpyAsync(wbuf_[buf], stages_[s].host, stages_[s].bytes, cudaMemcpyHostToDevice, copy_));
    CUDA_CHECK(cudaEventRecord(copied_[buf], copy_));
}

void VisionEncoder::encode(const uint8_t * rgb, int H, int W, float * out) {
    if (!loaded_) throw std::runtime_error("vision: no encoder loaded");
    const auto & c = cfg_;
    if (H % c.unit() || W % c.unit() || H <= 0 || W <= 0) throw std::runtime_error("vision: image sides must be multiples of " + std::to_string(c.unit()));
    DeviceScope ds(dev_);
    const double t0 = now_ms();
    const int P = c.patch, gh = H / P, gw = W / P, N = gh * gw, mm = c.merge * c.merge, M = N / mm;
    const int E = c.n_embd, F = c.n_ff, D = c.head_dim(), Hn = c.n_head, K0 = 3 * P * P;
    // attention: query rows per pass, bounded so the scores (fp16, softmaxed in place) take ~128 MiB
    const int R = std::max(16, std::min(N, (int) (((size_t) 128 << 20) / ((size_t) Hn * N * 2))));

    size_t off = 0;
    auto carve = [&](size_t bytes) { off = (off + 255) / 256 * 256; const size_t o = off; off += bytes; return o; };
    const size_t o_w0 = carve(stage_max_), o_w1 = carve(stage_max_);
    const size_t o_x = carve((size_t) N * E * 4), o_h = carve((size_t) N * std::max({F, E, K0}) * 2);
    const size_t o_y = carve((size_t) N * std::max(3 * E, F) * 4);
    const size_t o_q = carve((size_t) N * E * 2), o_k = carve((size_t) N * E * 2), o_v = carve((size_t) N * E * 2);
    const size_t o_o = carve((size_t) N * E * 2);
    const size_t o_s = carve((size_t) Hn * R * N * 2);
    uint8_t * base = (uint8_t *) scratch(off);
    wbuf_[0] = base + o_w0;
    wbuf_[1] = base + o_w1;
    float * x = (float *) (base + o_x), * y = (float *) (base + o_y);
    half * h = (half *) (base + o_h), * q = (half *) (base + o_q), * k = (half *) (base + o_k), * v = (half *) (base + o_v);
    half * o = (half *) (base + o_o), * S = (half *) (base + o_s);

    // patches in the encoder's order, normalized, fp16
    std::vector<half> px((size_t) N * K0);
    for (int p = 0; p < N; ++p) {
        int r, cc;
        patch_rc(p, gw, c.merge, &r, &cc);
        half * d = px.data() + (size_t) p * K0;
        for (int ch = 0; ch < 3; ++ch)
            for (int ky = 0; ky < P; ++ky)
                for (int kx = 0; kx < P; ++kx) {
                    const uint8_t u = rgb[((size_t) (r * P + ky) * W + cc * P + kx) * 3 + ch];
                    d[(ch * P + ky) * P + kx] = __float2half((u / 255.f - c.mean[ch]) / c.std[ch]);
                }
    }
    CUDA_CHECK(cudaMemcpyAsync(h, px.data(), px.size() * 2, cudaMemcpyHostToDevice, st_));

    // Y[n][m] = X[n][k] . W[m][k]^T (fp16 in, fp32 out)
    auto gemm = [&](const half * Wm, int m, int kk, const half * X, int n, float * Y) {
        const float one = 1.f, zero = 0.f;
        BLAS_CHECK(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, m, n, kk, &one, Wm, CUDA_R_16F, kk, X, CUDA_R_16F, kk,
                                &zero, Y, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    const int ns = (int) stages_.size();
    fetch(0, 0);
    for (int s = 0; s < ns; ++s) {
        const int b = s & 1;
        if (s + 1 < ns) fetch(s + 1, b ^ 1);
        CUDA_CHECK(cudaStreamWaitEvent(st_, copied_[b], 0));
        const uint8_t * w = wbuf_[b];
        const auto & so = stages_[s].off;
        auto H16 = [&](int i) { return (const half *) (w + so[i]); };
        auto F32 = [&](int i) { return (const float *) (w + so[i]); };
        if (s == 0) {
            gemm(H16(0), E, K0, h, N, y);
            pos_embed_k<<<N, 256, 0, st_>>>(y, F32(1), pos_table_, x, E, gh, gw, c.pos_side, c.merge);
        } else if (s <= c.n_layer) {
            // 0 qkv, 1 out, 2 up, 3 down, 4-7 their biases, 8/9 ln1, 10/11 ln2
            layernorm_k<<<N, 256, 0, st_>>>(x, F32(8), F32(9), h, E, c.eps);
            gemm(H16(0), 3 * E, E, h, N, y);
            qkv_rope_k<<<N, 256, 0, st_>>>(y, F32(4), q, k, v, E, Hn, gw, c.merge, 10000.f);
            const float scale = 1.f / sqrtf((float) D), one = 1.f, zero = 0.f;
            for (int r0 = 0; r0 < N; r0 += R) {
                const int rn = std::min(R, N - r0);
                // per head: S^T (N x rn, column-major) = K_h . Q_h^T, then P = softmax over each query's column
                BLAS_CHECK(cublasGemmStridedBatchedEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, N, rn, D, &scale, k, CUDA_R_16F,
                                                      E, D, q + (size_t) r0 * E, CUDA_R_16F, E, D, &zero, S, CUDA_R_16F,
                                                      N, (long long) N * rn, Hn, CUBLAS_COMPUTE_32F,
                                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                softmax_k<<<Hn * rn, 256, 0, st_>>>(S, N);
                // O_h (D x rn, column-major in rows of E) = V_h . P_h
                BLAS_CHECK(cublasGemmStridedBatchedEx(blas_, CUBLAS_OP_N, CUBLAS_OP_N, D, rn, N, &one, v, CUDA_R_16F, E,
                                                      D, S, CUDA_R_16F, N, (long long) N * rn, &zero,
                                                      o + (size_t) r0 * E, CUDA_R_16F, E, D, Hn, CUBLAS_COMPUTE_32F,
                                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            gemm(H16(1), E, E, o, N, y);
            residual_k<<<nblk((int64_t) N * E, 256), 256, 0, st_>>>(x, y, F32(5), E, (int64_t) N * E);
            layernorm_k<<<N, 256, 0, st_>>>(x, F32(10), F32(11), h, E, c.eps);
            gemm(H16(2), F, E, h, N, y);
            gelu_k<true><<<nblk((int64_t) N * F, 256), 256, 0, st_>>>(y, F32(6), h, F, (int64_t) N * F);
            gemm(H16(3), E, F, h, N, y);
            residual_k<<<nblk((int64_t) N * E, 256), 256, 0, st_>>>(x, y, F32(7), E, (int64_t) N * E);
        } else {
            // 0/1 post norm, 2 mm0 weight, 3 its bias, 4 mm2 weight, 5 its bias; the norm is per patch, the
            // merger's rows are a block's mm patches side by side (they are adjacent in the encoder's order)
            const int EM = E * mm;
            layernorm_k<<<N, 256, 0, st_>>>(x, F32(0), F32(1), h, E, c.eps);
            gemm(H16(2), EM, EM, h, M, y);
            gelu_k<false><<<nblk((int64_t) M * EM, 256), 256, 0, st_>>>(y, F32(3), h, EM, (int64_t) M * EM);
            gemm(H16(4), c.out_dim, EM, h, M, y);
            add_bias_k<<<nblk((int64_t) M * c.out_dim, 256), 256, 0, st_>>>(y, F32(5), c.out_dim, (int64_t) M * c.out_dim);
        }
        CUDA_CHECK(cudaEventRecord(used_[b], st_));
        static const char * dump = getenv("BNK_VISION_DUMP");   // debugging: the residual after every stage
        if (dump && s < ns - 1) {
            std::vector<float> hx((size_t) N * E);
            CUDA_CHECK(cudaMemcpyAsync(hx.data(), x, hx.size() * 4, cudaMemcpyDeviceToHost, st_));
            CUDA_CHECK(cudaStreamSynchronize(st_));
            FILE * f = fopen((std::string(dump) + "/x" + std::to_string(s) + ".bin").c_str(), "wb");
            if (f) { fwrite(hx.data(), 4, hx.size(), f); fclose(f); }
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(out, y, (size_t) M * c.out_dim * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    CUDA_CHECK(cudaStreamSynchronize(copy_));
    release_scratch();
    ms += now_ms() - t0;
    ++images;
    tokens += M;
}

}  // namespace bnk
