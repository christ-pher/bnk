#include "inkling/ink_spark.h"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "ggml.h"
#include "inkling/ink_attn.h"
#include "inkling/ink_ops.h"
#include "kernels/ops.h"

namespace bnk {

namespace {

__device__ __forceinline__ float wsum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float bsum256(float v, float * red) {
    v = wsum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    return wsum(l < 8 ? red[l] : 0.f);
}

struct MixJob { float w[4]; const float * norm; float * out; };
struct MixJobs { MixJob j[5]; };

// out[r] = rms(sum_k w[k] feats[r][k]) * norm, a block per (row, job)
__global__ void k_mix_taps(const float * feats, int E, MixJobs J) {
    __shared__ float red[8];
    const MixJob & jb = J.j[blockIdx.y];
    const int r = blockIdx.x;
    const float * f = feats + (int64_t) r * 4 * E;
    float v[9];
    float ss = 0;
    int n = 0;
    for (int i = threadIdx.x; i < E; i += blockDim.x, ++n) {
        const float a = jb.w[0] * f[i] + jb.w[1] * f[E + i] + jb.w[2] * f[2 * E + i] + jb.w[3] * f[3 * E + i];
        v[n] = a;
        ss += a * a;
    }
    const float rr = rsqrtf(bsum256(ss, red) / E + 1e-6f);
    n = 0;
    for (int i = threadIdx.x; i < E; i += blockDim.x, ++n) jb.out[(int64_t) r * E + i] = v[n] * rr * jb.norm[i];
}

// a warp per (row, head) of D = 128: RMS norm with weight w (null: none), NeoX RoPE at position pos0 + row (rope),
// written back and (cache) stored as fp16 at cell (pos0 + row) % ring
__global__ void k_head_post(float * x, int64_t ldx, int T, int nh, const float * w, float eps, bool norm, bool rope,
                            int pos0, float base, half * cache, int ring, int64_t ldc) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= T * nh) return;
    const int t = row / nh, h = row % nh;
    float * xr = x + (int64_t) t * ldx + (int64_t) h * 128;
    float v[4] = {xr[lane], xr[lane + 32], xr[lane + 64], xr[lane + 96]};
    if (norm) {
        const float ss = wsum(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
        const float r = rsqrtf(ss / 128 + eps);
        for (int k = 0; k < 4; ++k) v[k] *= r * (w ? w[lane + 32 * k] : 1.f);
    }
    if (rope) {   // pairs (i, i + 64), i = lane and lane + 32
        const float p = (float) (pos0 + t);
        for (int k = 0; k < 2; ++k) {
            const int i = lane + 32 * k;
            const float th = p * powf(base, -2.f * i / 128.f);
            float sn, cs;
            sincosf(th, &sn, &cs);
            const float a = v[k], b = v[k + 2];
            v[k] = a * cs - b * sn;
            v[k + 2] = a * sn + b * cs;
        }
    }
    for (int k = 0; k < 4; ++k) xr[lane + 32 * k] = v[k];
    if (cache) {
        half * c = cache + (int64_t) ((pos0 + t) % ring) * ldc + (int64_t) h * 128;
        for (int k = 0; k < 4; ++k) c[lane + 32 * k] = __float2half(v[k]);
    }
}

// the block's input rows: rms(tok_embd[anchor]) * tok_norm, then the mask rows
__global__ void k_embed_block(const uint16_t * tok_embd, size_t row_bf16, const int32_t * anchor, const float * tok_norm,
                              const float * mask, int E, float * x) {
    __shared__ float red[8];
    const int r = blockIdx.x;
    if (r > 0) {
        for (int i = threadIdx.x; i < E; i += blockDim.x) x[(int64_t) r * E + i] = mask[(int64_t) (r - 1) * E + i];
        return;
    }
    const uint16_t * e = tok_embd + (int64_t) anchor[0] * row_bf16;
    float ss = 0;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        const float v = __uint_as_float((uint32_t) e[i] << 16);
        ss += v * v;
    }
    const float rr = rsqrtf(bsum256(ss, red) / E + 1e-6f);
    for (int i = threadIdx.x; i < E; i += blockDim.x) x[i] = __uint_as_float((uint32_t) e[i] << 16) * rr * tok_norm[i];
}

__global__ void k_swiglu_rows(const float * gu, float * h, int F) {
    const int t = blockIdx.y;
    for (int f = blockIdx.x * blockDim.x + threadIdx.x; f < F; f += gridDim.x * blockDim.x) {
        const float g = gu[(int64_t) t * 2 * F + f], u = gu[(int64_t) t * 2 * F + F + f];
        h[(int64_t) t * F + f] = g / (1.f + __expf(-g)) * u;
    }
}

// joint input [state | markov_w1[prev] | hidden | f_rnn]
__global__ void k_joint_in(const float * state, const uint16_t * w1, const int32_t * prev, const float * hid,
                           const float * f_rnn, int R, int E, float * out) {
    const int n = 2 * R + 2 * E;
    const uint16_t * wr = w1 + (int64_t) prev[0] * R;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v;
        if (i < R) v = state[i];
        else if (i < 2 * R) v = __uint_as_float((uint32_t) wr[i - R] << 16);
        else if (i < 2 * R + E) v = hid[i - 2 * R];
        else v = f_rnn[i - 2 * R - E];
        out[i] = v;
    }
}

// gate = sigmoid(c0), cand = tanh(c1), state = cand + gate (state - cand), markov = tanh(c2)
__global__ void k_joint_post(const float * c, const float * b, float * state, float * markov, int R) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    const float g = 1.f / (1.f + expf(-(c[i] + b[i])));
    const float cand = tanhf(c[R + i] + b[R + i]);
    state[i] = cand + g * (state[i] - cand);
    markov[i] = tanhf(c[2 * R + i] + b[2 * R + i]);
}

__global__ void k_col(const float * base, const float * bias, const float * b2, float cap, int V, float * col) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += gridDim.x * blockDim.x) {
        const float v = base[i] + bias[i] + b2[i];
        col[i] = cap > 0 ? cap * tanhf(v / cap) : v;
    }
}

// one block of 64: the candidates sorted by logit (descending), their softmax at 1/temperature, the drawn one
// (the first whose cumulative probability passes u; greedy: the first), into prev
__global__ void k_draw(const float * vals, const int32_t * ids, const float * u, const float * inv_temp, int32_t * prev,
                       int32_t * tok_out, int32_t * ids_out, float * probs_out) {
    __shared__ float sv[64];
    __shared__ int si[64];
    __shared__ float sp[64];
    const int t = threadIdx.x;
    sv[t] = vals[t];
    si[t] = ids[t];
    __syncthreads();
    // rank of each candidate (ties by id) -> its place
    const float v = sv[t];
    const int id = si[t];
    int rk = 0;
    for (int k = 0; k < 64; ++k) rk += sv[k] > v || (sv[k] == v && si[k] < id);
    __syncthreads();
    sv[rk] = v;
    si[rk] = id;
    __syncthreads();
    const float it = inv_temp[0];
    const float te = it > 0.f ? it : 1.f;   // greedy: the report's probabilities at temperature 1
    const float m = sv[0];
    sp[t] = expf((sv[t] - m) * te);
    __syncthreads();
    if (t == 0) {
        float z = 0;
        for (int k = 0; k < 64; ++k) z += sp[k];
        int pick = 0;
        if (it > 0.f) {
            float cum = 0;
            pick = 63;
            for (int k = 0; k < 64; ++k) {
                cum += sp[k] / z;
                if (cum > u[0]) { pick = k; break; }
            }
        }
        for (int k = 0; k < 64; ++k) sp[k] /= z;
        prev[0] = si[pick];
        tok_out[0] = si[pick];
    }
    __syncthreads();
    ids_out[t] = si[t];
    probs_out[t] = sp[t];
}

}  // namespace

InkSpark::~InkSpark() {
    for (void * p : allocs_) cudaFree(p);
    if (h_u_) cudaFreeHost(h_u_);
    if (h_out_) cudaFreeHost(h_out_);
    if (h_probs_) cudaFreeHost(h_probs_);
}

QMat InkSpark::upload_rows_of(const GgufModel & g, const std::string & name, int64_t r0, int64_t r1) {
    const TensorRef & t = g.get(name);
    const size_t rb = t.row_bytes(), bytes = (size_t) (r1 - r0) * rb;
    void * d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess || cudaMemcpy(d, t.data + (size_t) r0 * rb, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("inkspark: upload failed for " + name);
    allocs_.push_back(d);
    vram_bytes_ += bytes;
    QMat m;
    m.data = d;
    m.type = (int) t.type;
    m.rows = r1 - r0;
    m.cols = t.ne[0];
    m.row_bytes = rb;
    return m;
}

QMat InkSpark::upload(const std::string & name) {
    const TensorRef & t = gguf_.get(name);
    return upload_rows_of(gguf_, name, 0, t.nrows());
}

QMat InkSpark::upload_stack(const std::vector<std::string> & names) {
    std::vector<uint8_t> host;
    int64_t rows = 0, cols = -1;
    int type = -1;
    for (const auto & n : names) {
        const TensorRef & t = gguf_.get(n);
        if (cols >= 0 && (t.ne[0] != cols || (int) t.type != type)) throw std::runtime_error("inkspark: cannot stack " + n);
        cols = t.ne[0];
        type = (int) t.type;
        host.insert(host.end(), t.data, t.data + t.nbytes);
        rows += t.nrows();
    }
    void * d = nullptr;
    if (cudaMalloc(&d, host.size()) != cudaSuccess || cudaMemcpy(d, host.data(), host.size(), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("inkspark: upload failed for " + names[0]);
    allocs_.push_back(d);
    vram_bytes_ += host.size();
    QMat m;
    m.data = d;
    m.type = type;
    m.rows = rows;
    m.cols = cols;
    m.row_bytes = ggml_row_size((ggml_type) type, cols);
    return m;
}

const float * InkSpark::upload_f32(const std::string & name, std::vector<float> * host) {
    const TensorRef & t = gguf_.get(name);
    if (t.type != GGML_TYPE_F32) throw std::runtime_error("inkspark: " + name + " is not F32");
    if (host) host->assign((const float *) t.data, (const float *) t.data + t.nbytes / 4);
    return (const float *) upload(name).data;
}

void InkSpark::mm(const QMat & W, const float * x, int64_t ldx, int T, float * y, int64_t ldy) {
    if (T <= 0) return;
    if (T <= kMaxGemvRows) gemv_auto(W, x, ldx, T, y, ldy, false, act_, st_);
    else gemm_.run(W, x, ldx, T, y, ldy, false);
}

void InkSpark::load(const std::string & path, const GgufModel & target, const QMat & tok_embd, int n_vocab, int max_ctx,
                    int chunk, cudaStream_t st, bool verbose) {
    gguf_.open(path);
    if (gguf_.get_str("general.architecture") != "dflash" || !gguf_.get_bool("dflash.has_markov_context", false))
        throw std::runtime_error("inkspark: " + path + " is not an Inkling DSpark drafter (dflash with a Markov context)");
    st_ = st;
    CUDA_CHECK(cudaGetDevice(&dev_));
    n_vocab_ = n_vocab;
    max_ctx_ = max_ctx;
    chunk_ = chunk;
    n_layer_ = (int) gguf_.get_u64("dflash.block_count");
    E_ = (int) gguf_.get_u64("dflash.embedding_length");
    F_ = (int) gguf_.get_u64("dflash.feed_forward_length");
    H_ = (int) gguf_.get_u64("dflash.attention.head_count");
    KVH_ = (int) gguf_.get_u64("dflash.attention.head_count_kv");
    D_ = (int) gguf_.get_u64("dflash.attention.key_length");
    eps_ = (float) gguf_.get_f64("dflash.attention.layer_norm_rms_epsilon", 1e-6);
    rope_base_ = (float) gguf_.get_f64("dflash.rope.freq_base", 1e6);
    softcap_ = (float) gguf_.get_f64("dflash.final_logit_softcapping", 30.0);
    block_ = (int) gguf_.get_u64("dflash.block_size", 8);
    n_swa_ = (int) gguf_.get_u64("dflash.attention.sliding_window", 0);
    for (int64_t v : gguf_.get_int_arr("dflash.attention.sliding_window_pattern")) swa_.push_back(v != 0);
    for (int64_t v : gguf_.get_int_arr("dflash.target_layers")) targets_.push_back((int) v);
    if (gguf_.get_bool("dflash.sample_from_anchor", true)) throw std::runtime_error("inkspark: expected a bonus-anchor block");
    if ((int) swa_.size() != n_layer_ || (int) targets_.size() != kTaps || D_ != 128 || H_ % KVH_ || block_ > kMaxBlock ||
        (int) gguf_.get_u64("dflash.rope.dimension_count") != D_)
        throw std::runtime_error("inkspark: unexpected drafter shape");
    n_draft_ = std::max(1, std::min(block_ - 1, env_int("BNK_INK_SPARK_N", block_ - 1)));
    rank_ = (int) gguf_.get("markov_w1.weight").ne[0];

    CUDA_CHECK(cudaStreamSynchronize(st_));
    const int KVD = KVH_ * D_, QD = H_ * D_, S = QD + 2 * KVD;
    ring_swa_ = std::max(2048, n_swa_ + chunk + kMaxBlock);
    layers_.resize(n_layer_);
    for (int il = 0; il < n_layer_; ++il) {
        Layer & L = layers_[il];
        const std::string b = "blk." + std::to_string(il) + ".";
        L.qkv = upload_stack({b + "attn_q.weight", b + "attn_k.weight", b + "attn_v.weight"});
        L.kvctx = upload_stack({b + "attn_k_ctx.weight", b + "attn_v_ctx.weight"});
        L.wo = upload(b + "attn_output.weight");
        L.gu = upload_stack({b + "ffn_gate.weight", b + "ffn_up.weight"});
        L.down = upload(b + "ffn_down.weight");
        L.attn_norm = upload_f32(b + "attn_norm.weight");
        L.q_norm = upload_f32(b + "attn_q_norm.weight");
        L.k_norm = upload_f32(b + "attn_k_norm.weight");
        L.k_ctx_norm = upload_f32(b + "attn_k_ctx_norm.weight");
        L.ctx_mix = upload_f32(b + "ctx_mix.weight", &h_mix_[il]);
        L.ctx_norm = upload_f32(b + "ctx_norm.weight");
        L.ffn_norm = upload_f32(b + "ffn_norm.weight");
        L.window = swa_[il] ? n_swa_ : 0;
        L.ring = swa_[il] ? ring_swa_ : max_ctx;
        L.K.alloc((size_t) L.ring * KVD);
        L.V.alloc((size_t) L.ring * KVD);
        vram_bytes_ += 2 * L.K.n * sizeof(half);
        if (L.qkv.rows != S || L.gu.rows != 2 * F_ || L.kvctx.rows != 2 * KVD) throw std::runtime_error("inkspark: layer shapes");
    }
    tok_norm_ = upload_f32("token_embd_norm.weight");
    mask_embd_ = upload_f32("mask_embd.weight");
    out_norm_ = upload_f32("output_norm.weight");
    w1_ = upload("markov_w1.weight");
    w2_ = upload("markov_w2.weight");
    w2_b_ = upload_f32("markov_w2.bias");
    joint_ = upload("joint_proj.weight");
    joint_b_ = upload_f32("joint_proj.bias");
    mctx_mix_ = upload_f32("markov_ctx_mix.weight", &h_mix_[4]);
    mctx_norm_ = upload_f32("markov_ctx_norm.weight");
    if (w1_.type != GGML_TYPE_BF16 || w1_.cols != rank_ || w2_.rows != n_vocab || joint_.cols != 2 * rank_ + 2 * E_ ||
        joint_.rows != 3 * rank_ || (int) gguf_.get("mask_embd.weight").ne[1] < block_ - 1)
        throw std::runtime_error("inkspark: Markov head shapes");
    for (int k = 0; k < 5; ++k)   // the layers' (0 .. n_layer - 1) and the Markov head's (4)
        if ((k < n_layer_ || k == 4) && h_mix_[k].size() != (size_t) kTaps) throw std::runtime_error("inkspark: the mixes need 4 taps");
    if (E_ > 9 * 256 || n_layer_ > 4) throw std::runtime_error("inkspark: the mix kernel holds E <= 2304 and up to 4 layers");
    // the target's token table (its own) and head rows (a full copy: under tensor parallel this GPU holds half)
    tok_embd_ = tok_embd;
    if (tok_embd_.type != GGML_TYPE_BF16) throw std::runtime_error("inkspark: expected the target's BF16 token table");
    head_ = upload_rows_of(target, "output.weight", 0, n_vocab);

    const int W = std::max(chunk, kMaxBlock), B = kMaxBlock;
    mixed_.alloc((size_t) 5 * W * E_);
    kv_.alloc((size_t) W * 2 * KVD);
    x_.alloc((size_t) B * E_); xn_.alloc((size_t) B * E_); qkv_.alloc((size_t) B * S); attn_.alloc((size_t) B * QD);
    o_.alloc((size_t) B * E_); gu_.alloc((size_t) B * 2 * F_); h_.alloc((size_t) B * F_); hid_.alloc((size_t) B * E_);
    logits_.alloc((size_t) B * n_vocab);
    f_rnn_.alloc(E_); state_.alloc(rank_); joint_in_.alloc(2 * rank_ + 2 * E_); joint_out_.alloc(3 * rank_);
    markov_.alloc(rank_); bias_.alloc(n_vocab); col_.alloc(n_vocab);
    attn_scratch_.alloc((size_t) 16 * B * H_ * 130);
    topv_.alloc(kCand); topi_.alloc(kCand);
    draw_tok_.alloc(B); draw_ids_.alloc((size_t) B * kCand); draw_p_.alloc((size_t) B * kCand); draw_u_.alloc(B + 1);
    tok_dev_.alloc(1); prev_.alloc(1); pos_dev_.alloc(1);
    CUDA_CHECK(cudaHostAlloc((void **) &h_u_, (B + 1) * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **) &h_out_, (size_t) B * (1 + kCand) * sizeof(int32_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **) &h_probs_, (size_t) B * kCand * sizeof(float), cudaHostAllocDefault));
    const int64_t maxcols = std::max<int64_t>({(int64_t) F_, (int64_t) QD, (int64_t) 2 * rank_ + 2 * E_});
    actq_.alloc((size_t) kMaxGemvRows * maxcols); actd_.alloc((size_t) kMaxGemvRows * maxcols / 32);
    acts_.alloc((size_t) kMaxGemvRows * maxcols / 16); acth_.alloc((size_t) kMaxGemvRows * maxcols);
    act_.q = actq_; act_.d = actd_; act_.s = acts_; act_.h = acth_;
    gemm_.init(st_, (size_t) 2 * KVD * E_, (size_t) W * E_);
    if (const char * d = getenv("BNK_INK_SPARK_DUMP")) dump_dir_ = d;
    loaded_ = true;
    if (verbose)
        fprintf(stderr, "bnk: inkling: DSpark drafter %s: %d layers, blocks of %d (%d drafts), taps at visits %d/%d/%d/%d, "
                        "Markov rank %d, %.2f GiB in VRAM (with a copy of the head)\n",
                path.substr(path.find_last_of('/') + 1).c_str(), n_layer_, block_, n_draft_, targets_[0], targets_[1],
                targets_[2], targets_[3], rank_, vram_bytes_ / double(1 << 30));
}

// ------------------------------------------------------------------------------------------------ injection
void InkSpark::feed(const float * feats, int n, int pos0) {
    if (n <= 0) return;
    DeviceScope ds(dev_);
    if (n > chunk_) throw std::runtime_error("inkspark: feed larger than a chunk");
    const int KVD = KVH_ * D_;
    MixJobs J{};
    for (int k = 0; k < n_layer_; ++k) {
        for (int i = 0; i < 4; ++i) J.j[k].w[i] = h_mix_[k][i];
        J.j[k].norm = layers_[k].ctx_norm;
        J.j[k].out = mixed_.p + (size_t) k * chunk_ * E_;
    }
    // the layers' mixes on every row, the Markov head's (f_rnn) on the last only
    k_mix_taps<<<dim3(n, n_layer_), 256, 0, st_>>>(feats, E_, J);
    MixJobs M{};
    for (int i = 0; i < 4; ++i) M.j[0].w[i] = h_mix_[4][i];
    M.j[0].norm = mctx_norm_;
    M.j[0].out = f_rnn_.p;
    k_mix_taps<<<dim3(1, 1), 256, 0, st_>>>(feats + (size_t) (n - 1) * 4 * E_, E_, M);
    for (int il = 0; il < n_layer_; ++il) {
        Layer & L = layers_[il];
        mm(L.kvctx, mixed_.p + (size_t) il * chunk_ * E_, E_, n, kv_, 2 * KVD);
        const int rows = n * KVH_;
        k_head_post<<<(rows + 7) / 8, 256, 0, st_>>>(kv_, 2 * KVD, n, KVH_, L.k_ctx_norm, eps_, true, true, pos0, rope_base_,
                                                      L.K, L.ring, KVD);
        k_head_post<<<(rows + 7) / 8, 256, 0, st_>>>(kv_.p + KVD, 2 * KVD, n, KVH_, nullptr, eps_, false, false, pos0,
                                                      rope_base_, L.V, L.ring, KVD);
    }
    check_launch("inkspark feed");
    fed_hi_ = pos0 + n;
    if (!dump_dir_.empty() && dumps_ < 4) {   // the fed rows: int32 pos0, n, then [n][4 E] floats
        std::vector<float> h((size_t) n * 4 * E_);
        CUDA_CHECK(cudaMemcpyAsync(h.data(), feats, h.size() * 4, cudaMemcpyDeviceToHost, st_));
        CUDA_CHECK(cudaStreamSynchronize(st_));
        if (FILE * f = fopen((dump_dir_ + "/feed.bin").c_str(), "ab")) {
            fwrite(&pos0, 4, 1, f); fwrite(&n, 4, 1, f); fwrite(h.data(), 4, h.size(), f); fclose(f);
        }
    }
}

// ------------------------------------------------------------------------------------------------ the draft
void InkSpark::draft_block(int32_t anchor, int pos0) {
    DeviceScope ds(dev_);
    // the block: the anchor and the drafts the round will use (the layers see only these rows, as llama.cpp's block
    // of n_max masks: its quality does not depend on the masks after a row)
    const int nb = 1 + std::max(1, std::min(n_draft_, n_use_)), KVD = KVH_ * D_, QD = H_ * D_, S = QD + 2 * KVD, V = n_vocab_;
    // the draws: a uniform per row, 1 / temperature (0: greedy)
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    for (int i = 0; i <= kMaxBlock; ++i) h_u_[i] = std::min((float) ud(rng_), std::nextafter(1.0f, 0.0f));
    h_u_[kMaxBlock] = temperature_ > 0.f ? 1.f / temperature_ : 0.f;
    CUDA_CHECK(cudaMemcpyAsync(draw_u_.p, h_u_, (kMaxBlock + 1) * sizeof(float), cudaMemcpyHostToDevice, st_));
    h_out_[0] = anchor;
    CUDA_CHECK(cudaMemcpyAsync(tok_dev_.p, h_out_, 4, cudaMemcpyHostToDevice, st_));
    CUDA_CHECK(cudaMemcpyAsync(prev_.p, h_out_, 4, cudaMemcpyHostToDevice, st_));
    k_embed_block<<<nb, 256, 0, st_>>>((const uint16_t *) tok_embd_.data, tok_embd_.row_bytes / 2, tok_dev_, tok_norm_,
                                       mask_embd_, E_, x_);
    for (int il = 0; il < n_layer_; ++il) {
        Layer & L = layers_[il];
        ink_rms(x_, E_, L.attn_norm, xn_, E_, nb, E_, eps_, st_);
        mm(L.qkv, xn_, E_, nb, qkv_, S);
        k_head_post<<<(nb * H_ + 7) / 8, 256, 0, st_>>>(qkv_, S, nb, H_, L.q_norm, eps_, true, true, pos0, rope_base_,
                                                         nullptr, 0, 0);
        k_head_post<<<(nb * KVH_ + 7) / 8, 256, 0, st_>>>(qkv_.p + QD, S, nb, KVH_, L.k_norm, eps_, true, true, pos0,
                                                           rope_base_, L.K, L.ring, KVD);
        k_head_post<<<(nb * KVH_ + 7) / 8, 256, 0, st_>>>(qkv_.p + QD + KVD, S, nb, KVH_, nullptr, eps_, false, false,
                                                           pos0, rope_base_, L.V, L.ring, KVD);
        // every row over the context and the whole block; the window counted from the block's first row
        const int keys = L.window ? std::min(pos0 + nb, L.window + nb) : pos0 + nb;
        const int ns = std::max(1, std::min(16, keys / 2048));
        ink_attention_plain(qkv_, S, L.K, L.V, L.ring, L.window ? L.window + nb - 1 : 0, nullptr, pos0, nb, H_, KVH_,
                            1.f / sqrtf((float) D_), true, ns, attn_, attn_scratch_, st_);
        mm(L.wo, attn_, QD, nb, o_, E_);
        ink_add(x_, x_, o_, (int64_t) nb * E_, st_);
        ink_rms(x_, E_, L.ffn_norm, xn_, E_, nb, E_, eps_, st_);
        mm(L.gu, xn_, E_, nb, gu_, 2 * F_);
        k_swiglu_rows<<<dim3((F_ + 255) / 256, nb), 256, 0, st_>>>(gu_, h_, F_);
        mm(L.down, h_, F_, nb, o_, E_);
        ink_add(x_, x_, o_, (int64_t) nb * E_, st_);
    }
    ink_rms(x_, E_, out_norm_, hid_, E_, nb, E_, eps_, st_);
    mm(head_, hid_.p + E_, E_, nb - 1, logits_, V);   // the draft rows' base logits
    // the Markov chain over the draft rows
    CUDA_CHECK(cudaMemsetAsync(state_.p, 0, rank_ * sizeof(float), st_));
    for (int i = 1; i < nb; ++i) {
        k_joint_in<<<(2 * rank_ + 2 * E_ + 255) / 256, 256, 0, st_>>>(state_, (const uint16_t *) w1_.data, prev_,
                                                                      hid_.p + (size_t) i * E_, f_rnn_, rank_, E_, joint_in_);
        mm(joint_, joint_in_, 2 * rank_ + 2 * E_, 1, joint_out_, 3 * rank_);
        k_joint_post<<<(rank_ + 255) / 256, 256, 0, st_>>>(joint_out_, joint_b_, state_, markov_, rank_);
        mm(w2_, markov_, rank_, 1, bias_, V);
        k_col<<<1024, 256, 0, st_>>>(logits_.p + (size_t) (i - 1) * V, bias_, w2_b_, softcap_, V, col_);
        topk_rows(col_, 1, V, kCand, topi_, topv_, st_);
        k_draw<<<1, kCand, 0, st_>>>(topv_, topi_, draw_u_.p + i, draw_u_.p + kMaxBlock, prev_, draw_tok_.p + i,
                                     draw_ids_.p + (size_t) i * kCand, draw_p_.p + (size_t) i * kCand);
    }
    check_launch("inkspark draft");
    CUDA_CHECK(cudaMemcpyAsync(h_out_, draw_tok_.p, kMaxBlock * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(h_out_ + kMaxBlock, draw_ids_.p, (size_t) kMaxBlock * kCand * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaMemcpyAsync(h_probs_, draw_p_.p, (size_t) kMaxBlock * kCand * 4, cudaMemcpyDeviceToHost, st_));
    CUDA_CHECK(cudaStreamSynchronize(st_));
    if (!dump_dir_.empty() && dumps_ < 4) {   // the draft: anchor, pos0, nb, hidden [nb][E], base logits [nb-1][V], tokens
        ++dumps_;
        std::vector<float> hh((size_t) nb * E_), lg((size_t) (nb - 1) * V);
        CUDA_CHECK(cudaMemcpy(hh.data(), hid_.p, hh.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lg.data(), logits_.p, lg.size() * 4, cudaMemcpyDeviceToHost));
        if (FILE * f = fopen((dump_dir_ + "/draft" + std::to_string(dumps_) + ".bin").c_str(), "wb")) {
            fwrite(&anchor, 4, 1, f); fwrite(&pos0, 4, 1, f); fwrite(&nb, 4, 1, f);
            fwrite(hh.data(), 4, hh.size(), f); fwrite(lg.data(), 4, lg.size(), f); fwrite(h_out_ + 1, 4, nb - 1, f);
            fclose(f);
        }
    }
    blk_ = Block{};
    const bool sampled = temperature_ > 0.f;
    for (int i = 1; i < nb; ++i) {
        const int32_t t = h_out_[i];
        const int32_t * ids = h_out_ + kMaxBlock + (size_t) i * kCand;
        const float * pr = h_probs_ + (size_t) i * kCand;
        float p = 0.f;
        std::vector<std::pair<int32_t, float>> d;
        for (int k = 0; k < kCand; ++k) {
            if (ids[k] == t) p = pr[k];
            if (sampled) d.push_back({ids[k], pr[k]});
        }
        blk_.toks.push_back(t);
        blk_.prob.push_back(p);
        blk_.dist.push_back(std::move(d));
    }
}

int InkSpark::run(const float * R_rows, const int32_t * next_tokens, int n, int cell0, bool draft, float * prob) {
    DeviceScope ds(dev_);
    const double t0 = now_ms();
    if (prob) *prob = 0.f;
    blk_ = Block{};
    if (cell0 != fed_hi_) {   // a gap (or a rewind past the context): the context restarts here
        if (cell0 > fed_hi_) { fed_hi_ = cell0; }
    }
    feed(R_rows, n, cell0);
    int r = -1;
    if (draft && n_draft_ > 0) {
        draft_block(next_tokens[n - 1], cell0 + n);
        r = step(-1, cell0 + n, prob);
    } else {
        CUDA_CHECK(cudaStreamSynchronize(st_));
    }
    ms += now_ms() - t0;
    ++calls;
    return r;
}

int InkSpark::step(int32_t, int, float * prob) {
    if (prob) *prob = 0.f;
    if (blk_.next >= (int) blk_.toks.size()) return -1;
    if (prob) *prob = blk_.prob[blk_.next];
    return blk_.toks[blk_.next++];
}

void InkSpark::run_multi(const std::vector<RunItem> & items, int32_t * tok_out, float * prob_out) {
    for (size_t i = 0; i < items.size(); ++i) {
        const RunItem & it = items[i];
        tok_out[i] = run(it.R_rows, it.tokens, it.n, it.cell0, true, &prob_out[i]);
    }
}

void InkSpark::step_multi(const std::vector<StepItem> & items, int32_t * tok_out, float * prob_out) {
    for (size_t i = 0; i < items.size(); ++i) tok_out[i] = step(items[i].tok, items[i].cell, &prob_out[i]);
}

const std::vector<std::pair<int32_t, float>> * InkSpark::draft_dist(int, int i) const {
    if (i < 0 || i >= (int) blk_.dist.size() || blk_.dist[i].empty()) return nullptr;
    return &blk_.dist[i];
}

void InkSpark::save(std::vector<uint8_t> & out) const {
    DeviceScope ds(dev_);
    size_t bytes = (size_t) E_ * 4 + sizeof(int);
    for (const auto & L : layers_)
        if (L.window) bytes += (L.K.n + L.V.n) * sizeof(half);
    out.resize(bytes);
    size_t o = 0;
    CUDA_CHECK(cudaMemcpyAsync(out.data(), f_rnn_.p, (size_t) E_ * 4, cudaMemcpyDeviceToHost, st_));
    o += (size_t) E_ * 4;
    for (const auto & L : layers_) {
        if (!L.window) continue;
        CUDA_CHECK(cudaMemcpyAsync(out.data() + o, L.K.p, L.K.n * sizeof(half), cudaMemcpyDeviceToHost, st_));
        o += L.K.n * sizeof(half);
        CUDA_CHECK(cudaMemcpyAsync(out.data() + o, L.V.p, L.V.n * sizeof(half), cudaMemcpyDeviceToHost, st_));
        o += L.V.n * sizeof(half);
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    memcpy(out.data() + o, &fed_hi_, sizeof(int));
}

void InkSpark::load_state(const std::vector<uint8_t> & in) {
    if (in.empty()) { fed_hi_ = 0; return; }
    DeviceScope ds(dev_);
    size_t o = 0;
    CUDA_CHECK(cudaMemcpyAsync(f_rnn_.p, in.data(), (size_t) E_ * 4, cudaMemcpyHostToDevice, st_));
    o += (size_t) E_ * 4;
    for (auto & L : layers_) {
        if (!L.window) continue;
        CUDA_CHECK(cudaMemcpyAsync(L.K.p, in.data() + o, L.K.n * sizeof(half), cudaMemcpyHostToDevice, st_));
        o += L.K.n * sizeof(half);
        CUDA_CHECK(cudaMemcpyAsync(L.V.p, in.data() + o, L.V.n * sizeof(half), cudaMemcpyHostToDevice, st_));
        o += L.V.n * sizeof(half);
    }
    CUDA_CHECK(cudaStreamSynchronize(st_));
    memcpy(&fed_hi_, in.data() + o, sizeof(int));
    blk_ = Block{};
}

}  // namespace bnk
