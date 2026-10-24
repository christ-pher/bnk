// Checks of the DeepSeek-V4.1 kernels (src/deepseek/ds_ops.cu) against straightforward CPU references on random
// data, at the model's shapes but small contexts. Small enough to run beside a serving model.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "core/util.h"
#include "deepseek/ds_ops.h"

using namespace bnk;

static std::mt19937 rng(7);
static float rnd(float a = -1.f, float b = 1.f) { return std::uniform_real_distribution<float>(a, b)(rng); }
static int fails = 0;

template <typename T> static T * up(const std::vector<T> & v) {
    T * d;
    CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T> static std::vector<T> down(const T * d, size_t n) {
    std::vector<T> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
static void check(const char * what, const std::vector<float> & got, const std::vector<double> & ref, double tol) {
    double max_err = 0, max_ref = 0;
    size_t worst = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double e = std::fabs(got[i] - ref[i]);
        if (e > max_err) { max_err = e; worst = i; }
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }
    const bool ok = max_err <= tol * std::max(1.0, max_ref);
    printf("%-28s max err %.3e (ref max %.3e)%s", what, max_err, max_ref, ok ? "\n" : "  FAIL\n");
    if (!ok) { ++fails; printf("   at %zu: got %g ref %g\n", worst, (double) got[worst], ref[worst]); }
}
static double rms_ref(const double * x, int n, double eps) { double s = 0; for (int i = 0; i < n; ++i) s += x[i] * x[i]; return 1.0 / sqrt(s / n + eps); }
static void rot_ref(double * v, int n_rot, int pos, const std::vector<float> & f, bool inv) {
    for (int i = 0; i < n_rot / 2; ++i) {
        const double th = fmod((double) pos * (double) f[i], 2 * M_PI);
        const double c = cos(th), s = inv ? -sin(th) : sin(th);
        const double a = v[2 * i], b = v[2 * i + 1];
        v[2 * i] = a * c - b * s;
        v[2 * i + 1] = a * s + b * c;
    }
}

int main() {
    cudaStream_t st = nullptr;
    const int T = 5, hc = 4, E = 5120, HC = hc * E, D = 512, R = 64, H = 64, ID = 128, IH = 32, W = 128;
    const float eps = 1e-20f, hc_eps = 1e-6f;
    const auto fwin = ds_rope_inv_freq(R, 1e4f, 1.f, 0, 32, 1), fcomp = ds_rope_inv_freq(R, 1.6e5f, 16.f, 65536, 32, 1);
    float * d_fwin = up(fwin), * d_fcomp = up(fcomp);
    printf("yarn inv_freq[0..3] %.4g %.4g %.4g %.4g  [last] %.4g\n", fcomp[0], fcomp[1], fcomp[2], fcomp[3], fcomp.back());

    // ---- hyper-connections
    {
        std::vector<float> res((size_t) T * HC), fn((size_t) 24 * HC), scale{0.7f, 1.3f, 0.9f}, base(24), x((size_t) T * E);
        for (auto & v : res) v = rnd();
        for (auto & v : fn) v = rnd(-0.02f, 0.02f);
        for (auto & v : base) v = rnd();
        for (auto & v : x) v = rnd();
        float * d_res = up(res), * d_fn = up(fn), * d_scale = up(scale), * d_base = up(base), * d_x = up(x);
        float * d_pre, * d_post, * d_comb;
        CUDA_CHECK(cudaMalloc(&d_pre, T * hc * 4)); CUDA_CHECK(cudaMalloc(&d_post, T * hc * 4)); CUDA_CHECK(cudaMalloc(&d_comb, T * hc * hc * 4));
        ds_hc_mixes(d_res, d_fn, d_scale, d_base, d_pre, d_post, d_comb, T, hc, E, eps, hc_eps, 20, st);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> rpre(T * hc), rpost(T * hc), rcomb(T * hc * hc);
        for (int t = 0; t < T; ++t) {
            std::vector<double> xr(HC);
            for (int i = 0; i < HC; ++i) xr[i] = res[(size_t) t * HC + i];
            const double r = rms_ref(xr.data(), HC, eps);
            double m[24];
            for (int j = 0; j < 24; ++j) { double a = 0; for (int i = 0; i < HC; ++i) a += fn[(size_t) j * HC + i] * xr[i]; m[j] = a * r; }
            auto sig = [](double v) { return 1.0 / (1.0 + exp(-v)); };
            for (int s = 0; s < hc; ++s) { rpre[t * hc + s] = sig(m[s] * scale[0] + base[s]) + hc_eps; rpost[t * hc + s] = 2 * sig(m[hc + s] * scale[1] + base[hc + s]); }
            double c[4][4];
            for (int j = 0; j < 4; ++j) {
                double mx = -1e30; for (int k = 0; k < 4; ++k) { c[j][k] = m[8 + j * 4 + k] * scale[2] + base[8 + j * 4 + k]; mx = std::max(mx, c[j][k]); }
                double sum = 0; for (int k = 0; k < 4; ++k) { c[j][k] = exp(c[j][k] - mx); sum += c[j][k]; }
                for (int k = 0; k < 4; ++k) c[j][k] = c[j][k] / sum + hc_eps;
            }
            for (int it = 0; it < 20; ++it) {
                if (it > 0) for (int j = 0; j < 4; ++j) { double s = 0; for (int k = 0; k < 4; ++k) s += c[j][k]; for (int k = 0; k < 4; ++k) c[j][k] /= (s + hc_eps); }
                for (int k = 0; k < 4; ++k) { double s = 0; for (int j = 0; j < 4; ++j) s += c[j][k]; for (int j = 0; j < 4; ++j) c[j][k] /= (s + hc_eps); }
            }
            for (int j = 0; j < 4; ++j) for (int k = 0; k < 4; ++k) rcomb[(t * 4 + j) * 4 + k] = c[j][k];
        }
        check("hc_mixes pre", down(d_pre, T * hc), rpre, 1e-4);
        check("hc_mixes post", down(d_post, T * hc), rpost, 1e-4);
        check("hc_mixes comb", down(d_comb, T * hc * hc), rcomb, 1e-4);
        // pre / post
        float * d_mixed; CUDA_CHECK(cudaMalloc(&d_mixed, T * E * 4));
        ds_hc_pre(d_res, d_pre, d_mixed, T, hc, E, st);
        auto pre = down(d_pre, T * hc), post = down(d_post, T * hc), comb = down(d_comb, T * hc * hc);
        std::vector<double> rmixed((size_t) T * E);
        for (int t = 0; t < T; ++t) for (int e = 0; e < E; ++e) { double a = 0; for (int s = 0; s < hc; ++s) a += pre[t * hc + s] * res[((size_t) t * hc + s) * E + e]; rmixed[(size_t) t * E + e] = a; }
        check("hc_pre", down(d_mixed, (size_t) T * E), rmixed, 1e-5);
        ds_hc_post(d_x, d_res, d_post, d_comb, T, hc, E, st);
        std::vector<double> rres((size_t) T * HC);
        for (int t = 0; t < T; ++t) for (int e = 0; e < E; ++e) for (int k = 0; k < hc; ++k) {
            double v = post[t * hc + k] * x[(size_t) t * E + e];
            for (int j = 0; j < hc; ++j) v += comb[(t * hc + j) * hc + k] * res[((size_t) t * hc + j) * E + e];
            rres[((size_t) t * hc + k) * E + e] = v;
        }
        check("hc_post", down(d_res, (size_t) T * HC), rres, 1e-5);
        // engram gate
        std::vector<float> key((size_t) T * (hc + 1) * E), w((size_t) hc * E);
        for (auto & v : key) v = rnd();
        for (auto & v : w) v = rnd(0.5f, 1.5f);
        float * d_key = up(key), * d_w = up(w);
        CUDA_CHECK(cudaMemcpy(d_res, res.data(), res.size() * 4, cudaMemcpyHostToDevice));
        ds_engram_gate(d_res, d_key, (int64_t) (hc + 1) * E, d_key + (size_t) hc * E, (int64_t) (hc + 1) * E, d_w, T, hc, E, eps, st);
        std::vector<double> reg((size_t) T * HC);
        for (int t = 0; t < T; ++t) for (int s = 0; s < hc; ++s) {
            const float * h = &res[((size_t) t * hc + s) * E], * kk = &key[(size_t) t * (hc + 1) * E + (size_t) s * E], * vv = &key[(size_t) t * (hc + 1) * E + (size_t) hc * E];
            double sh = 0, sk = 0, dt = 0;
            for (int e = 0; e < E; ++e) { sh += (double) h[e] * h[e]; sk += (double) kk[e] * kk[e]; dt += (double) h[e] * w[(size_t) s * E + e] * kk[e]; }
            const double dot = dt / sqrt(sh / E + eps) / sqrt(sk / E + eps) / sqrt((double) E);
            const double g = 1.0 / (1.0 + exp(-copysign(sqrt(std::max(fabs(dot), 1e-6)), dot)));
            for (int e = 0; e < E; ++e) reg[((size_t) t * hc + s) * E + e] = h[e] + g * vv[e];
        }
        check("engram_gate", down(d_res, (size_t) T * HC), reg, 1e-4);
    }

    // ---- routing
    {
        const int NE = 384, k = 6;
        std::vector<float> lg((size_t) T * NE), bias(NE);
        for (auto & v : lg) v = rnd(-3, 3);
        for (auto & v : bias) v = rnd(-0.5f, 0.5f);
        float * d_lg = up(lg), * d_bias = up(bias), * d_w;
        int32_t * d_ids;
        CUDA_CHECK(cudaMalloc(&d_ids, T * k * 4)); CUDA_CHECK(cudaMalloc(&d_w, T * k * 4));
        ds_route(d_lg, d_bias, T, NE, k, 1.5f, true, d_ids, d_w, st);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto ids = down(d_ids, T * k); auto w = down(d_w, T * k);
        int bad = 0;
        double werr = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<std::pair<double, int>> sc(NE);
            std::vector<double> s(NE);
            for (int e = 0; e < NE; ++e) { const double x = lg[(size_t) t * NE + e]; s[e] = sqrt(x > 20 ? x : log1p(exp(x))); sc[e] = {-(s[e] + bias[e]), e}; }
            std::sort(sc.begin(), sc.end());
            double sum = 0;
            for (int j = 0; j < k; ++j) sum += s[sc[j].second];
            for (int j = 0; j < k; ++j) {
                if (ids[t * k + j] != sc[j].second) ++bad;
                werr = std::max(werr, fabs(w[t * k + j] - s[sc[j].second] / (sum + 1e-20) * 1.5));
            }
        }
        printf("%-28s wrong picks %d, max weight err %.3e%s\n", "route", bad, werr, bad || werr > 1e-5 ? "  FAIL" : "");
        if (bad || werr > 1e-5) ++fails;
    }

    // ---- attention: q rope, kv prep into the ring, compressed rows, sinks
    {
        const int pos0 = 300, n_comp = 700, k_sel = 512;
        std::vector<float> q((size_t) T * H * D), kv((size_t) T * D), kvw(D), sinks(H);
        for (auto & v : q) v = rnd();
        for (auto & v : kv) v = rnd();
        for (auto & v : kvw) v = rnd(0.5f, 1.5f);
        for (auto & v : sinks) v = rnd(-2, 2);
        // the ring holds every position < pos0 (slot = pos % kDsRing); the compressed rows random
        std::vector<half> ring((size_t) kDsRing * D), comp((size_t) n_comp * D);
        std::vector<double> ringf(ring.size()), compf(comp.size());
        for (int p = pos0 - kDsRing; p < pos0; ++p) if (p >= 0) for (int d = 0; d < D; ++d) { const float v = rnd(); ring[(size_t) (p % kDsRing) * D + d] = __float2half(v); ringf[(size_t) (p % kDsRing) * D + d] = __half2float(__float2half(v)); }
        for (size_t i = 0; i < comp.size(); ++i) { const float v = rnd(); comp[i] = __float2half(v); compf[i] = __half2float(comp[i]); }
        std::vector<int32_t> idx((size_t) T * k_sel);
        for (int t = 0; t < T; ++t) for (int j = 0; j < k_sel; ++j) idx[(size_t) t * k_sel + j] = j < 400 ? (j * 7 + t) % n_comp : -1;
        const int h_pos0 = pos0;
        int * d_pos0 = up(std::vector<int>{h_pos0});
        float * d_q = up(q), * d_kv = up(kv), * d_kvw = up(kvw), * d_sinks = up(sinks), * d_out;
        half * d_ring = up(ring), * d_comp = up(comp);
        int32_t * d_idx = up(idx);
        CUDA_CHECK(cudaMalloc(&d_out, (size_t) T * H * D * 4));
        ds_rope_heads(d_q, T, H, D, R, d_pos0, d_fcomp, st);
        ds_kv_prep(d_kv, d_kvw, d_ring, T, D, R, d_pos0, d_fcomp, eps, st);
        ds_attention(d_q, d_ring, d_comp, d_idx, k_sel, d_sinks, d_out, T, H, D, W, R, d_pos0, d_fcomp, 1.f / sqrtf((float) D), st);
        CUDA_CHECK(cudaDeviceSynchronize());
        // reference
        std::vector<double> qr((size_t) T * H * D);
        for (int t = 0; t < T; ++t) for (int h = 0; h < H; ++h) {
            double * v = &qr[((size_t) t * H + h) * D];
            for (int d = 0; d < D; ++d) v[d] = q[((size_t) t * H + h) * D + d];
            rot_ref(v + D - R, R, pos0 + t, fcomp, false);
        }
        check("rope_heads", down(d_q, (size_t) T * H * D), qr, 1e-4);
        for (int t = 0; t < T; ++t) {   // the window rows the kernel wrote
            std::vector<double> v(D);
            for (int d = 0; d < D; ++d) v[d] = kv[(size_t) t * D + d];
            const double r = rms_ref(v.data(), D, eps);
            for (int d = 0; d < D; ++d) v[d] = v[d] * r * kvw[d];
            rot_ref(v.data() + D - R, R, pos0 + t, fcomp, false);
            for (int d = 0; d < D; ++d) ringf[(size_t) ((pos0 + t) % kDsRing) * D + d] = __half2float(__float2half((float) v[d]));
        }
        {
            auto got = down(d_ring, ring.size());
            std::vector<float> g2(got.size()); for (size_t i = 0; i < got.size(); ++i) g2[i] = __half2float(got[i]);
            check("kv_prep ring", g2, ringf, 2e-3);
        }
        std::vector<double> ro((size_t) T * H * D);
        for (int t = 0; t < T; ++t) {
            const int p = pos0 + t, n_win = std::min(W, p + 1);
            std::vector<const double *> keys;
            for (int i = 0; i < n_win; ++i) keys.push_back(&ringf[(size_t) ((p - n_win + 1 + i) % kDsRing) * D]);
            for (int j = 0; j < k_sel; ++j) { const int c = idx[(size_t) t * k_sel + j]; if (c >= 0) keys.push_back(&compf[(size_t) c * D]); }
            for (int h = 0; h < H; ++h) {
                const double * qh = &qr[((size_t) t * H + h) * D];
                std::vector<double> s(keys.size());
                double m = -1e30;
                for (size_t i = 0; i < keys.size(); ++i) { double a = 0; for (int d = 0; d < D; ++d) a += qh[d] * keys[i][d]; s[i] = a / sqrt((double) D); m = std::max(m, s[i]); }
                double l = exp(sinks[h] - m);
                std::vector<double> o(D, 0.0);
                for (size_t i = 0; i < keys.size(); ++i) { const double pexp = exp(s[i] - m); l += pexp; for (int d = 0; d < D; ++d) o[d] += pexp * keys[i][d]; }
                for (int d = 0; d < D; ++d) o[d] /= l;
                rot_ref(o.data() + D - R, R, p, fcomp, true);
                for (int d = 0; d < D; ++d) ro[((size_t) t * H + h) * D + d] = o[d];
            }
        }
        check("attention (win + comp)", down(d_out, (size_t) T * H * D), ro, 3e-3);
        // window only
        ds_attention(d_q, d_ring, nullptr, nullptr, 0, d_sinks, d_out, T, H, D, W, R, d_pos0, d_fcomp, 1.f / sqrtf((float) D), st);
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int t = 0; t < T; ++t) {
            const int p = pos0 + t, n_win = std::min(W, p + 1);
            for (int h = 0; h < H; ++h) {
                const double * qh = &qr[((size_t) t * H + h) * D];
                std::vector<double> s(n_win);
                double m = -1e30;
                for (int i = 0; i < n_win; ++i) { const double * kk = &ringf[(size_t) ((p - n_win + 1 + i) % kDsRing) * D]; double a = 0; for (int d = 0; d < D; ++d) a += qh[d] * kk[d]; s[i] = a / sqrt((double) D); m = std::max(m, s[i]); }
                double l = exp(sinks[h] - m);
                std::vector<double> o(D, 0.0);
                for (int i = 0; i < n_win; ++i) { const double * kk = &ringf[(size_t) ((p - n_win + 1 + i) % kDsRing) * D]; const double pexp = exp(s[i] - m); l += pexp; for (int d = 0; d < D; ++d) o[d] += pexp * kk[d]; }
                for (int d = 0; d < D; ++d) o[d] /= l;
                rot_ref(o.data() + D - R, R, p, fcomp, true);
                for (int d = 0; d < D; ++d) ro[((size_t) t * H + h) * D + d] = o[d];
            }
        }
        check("attention (window only)", down(d_out, (size_t) T * H * D), ro, 3e-3);
    }

    // ---- compressor (ratio 2 with the state) and its commit
    {
        const int ratio = 2, pos0 = 41;   // odd: the first group takes the state
        std::vector<float> kv((size_t) T * D), sc((size_t) T * D), stk(D), sts(D), w(D), ikw(ID);
        std::vector<half> idxk((size_t) ID * D);
        std::vector<double> idxkf(idxk.size());
        for (auto & v : kv) v = rnd();
        for (auto & v : sc) v = rnd(-2, 2);
        for (auto & v : stk) v = rnd();
        for (auto & v : sts) v = rnd(-2, 2);
        for (auto & v : w) v = rnd(0.5f, 1.5f);
        for (auto & v : ikw) v = rnd(0.5f, 1.5f);
        for (size_t i = 0; i < idxk.size(); ++i) { const float v = rnd(-0.05f, 0.05f); idxk[i] = __float2half(v); idxkf[i] = __half2float(idxk[i]); }
        const int n_groups = 64;
        std::vector<half> ckc((size_t) n_groups * D, __float2half(0.f)), ikc((size_t) n_groups * ID, __float2half(0.f));
        int * d_pos0 = up(std::vector<int>{pos0});
        float * d_kv = up(kv), * d_sc = up(sc), * d_stk = up(stk), * d_sts = up(sts), * d_w = up(w), * d_ikw = up(ikw);
        half * d_idxk = up(idxk), * d_ckc = up(ckc), * d_ikc = up(ikc);
        ds_compress(d_kv, d_sc, d_stk, d_sts, d_w, d_idxk, d_ikw, d_ckc, d_ikc, T, ratio, D, ID, R, d_pos0, d_fcomp, eps, st);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto gck = down(d_ckc, ckc.size()); auto gik = down(d_ikc, ikc.size());
        std::vector<float> gckf(gck.size()), gikf(gik.size());
        for (size_t i = 0; i < gck.size(); ++i) gckf[i] = __half2float(gck[i]);
        for (size_t i = 0; i < gik.size(); ++i) gikf[i] = __half2float(gik[i]);
        std::vector<double> rck(ckc.size(), 0.0), rik(ikc.size(), 0.0);
        auto row_kv = [&](int p) { return p < pos0 ? stk.data() : &kv[(size_t) (p - pos0) * D]; };
        auto row_sc = [&](int p) { return p < pos0 ? sts.data() : &sc[(size_t) (p - pos0) * D]; };
        for (int p = pos0; p < pos0 + T; ++p) {
            if (p % ratio != ratio - 1) continue;
            const int j = p / ratio;
            std::vector<double> L(D);
            for (int d = 0; d < D; ++d) {
                const double s0 = row_sc(p - 1)[d], s1 = row_sc(p)[d], m = std::max(s0, s1);
                const double e0 = exp(s0 - m), e1 = exp(s1 - m);
                L[d] = (row_kv(p - 1)[d] * e0 + row_kv(p)[d] * e1) / (e0 + e1);
            }
            const double r = rms_ref(L.data(), D, eps);
            for (int d = 0; d < D; ++d) L[d] = L[d] * r * w[d];
            std::vector<double> ik(ID);
            for (int i = 0; i < ID; ++i) { double a = 0; for (int d = 0; d < D; ++d) a += idxkf[(size_t) i * D + d] * L[d]; ik[i] = a; }
            const double ri = rms_ref(ik.data(), ID, eps);
            for (int i = 0; i < ID; ++i) ik[i] = ik[i] * ri * ikw[i];
            rot_ref(L.data() + D - R, R, j * ratio, fcomp, false);
            rot_ref(ik.data() + ID - R, R, j * ratio, fcomp, false);
            for (int d = 0; d < D; ++d) rck[(size_t) j * D + d] = __half2float(__float2half((float) L[d]));
            for (int i = 0; i < ID; ++i) rik[(size_t) j * ID + i] = __half2float(__float2half((float) ik[i]));
        }
        check("compress latents", gckf, rck, 3e-3);
        check("compress index keys", gikf, rik, 3e-3);
        // commit 3 rows: pos0 + 3 = 44 even -> no open group (state untouched); commit 2 -> 43 odd: state = row 1
        ds_compress_commit(d_kv, d_sc, d_stk, d_sts, 3, ratio, D, d_pos0, st);
        CUDA_CHECK(cudaDeviceSynchronize());
        { auto s1 = down(d_stk, D); std::vector<double> r1(stk.begin(), stk.end()); check("compress_commit (even)", s1, r1, 0); }
        ds_compress_commit(d_kv, d_sc, d_stk, d_sts, 2, ratio, D, d_pos0, st);
        CUDA_CHECK(cudaDeviceSynchronize());
        { auto s1 = down(d_stk, D); std::vector<double> r1(kv.begin() + D, kv.begin() + 2 * D); check("compress_commit (odd)", s1, r1, 0); }
    }

    // ---- indexer: scores, candidate blocks, top-k
    {
        const int ratio = 1, pos0 = 3000, n_max = 3200, bs = 8, k_blocks = 80, topk = 512;
        const int64_t lds = 4096;
        std::vector<float> iq((size_t) T * IH * ID), iw((size_t) T * IH);
        std::vector<half> ikc((size_t) n_max * ID);
        std::vector<double> ikf(ikc.size());
        for (auto & v : iq) v = rnd();
        for (auto & v : iw) v = rnd(-1, 1);
        for (size_t i = 0; i < ikc.size(); ++i) { const float v = rnd(); ikc[i] = __float2half(v); ikf[i] = __half2float(ikc[i]); }
        int * d_pos0 = up(std::vector<int>{pos0});
        float * d_iq = up(iq), * d_iw = up(iw), * d_scores, * d_scratch;
        half * d_ikc = up(ikc);
        CUDA_CHECK(cudaMalloc(&d_scores, (size_t) T * lds * 4));
        const int64_t ldb = (lds + bs - 1) / bs;
        CUDA_CHECK(cudaMalloc(&d_scratch, (size_t) T * ldb * 4));
        uint8_t * d_cand; CUDA_CHECK(cudaMalloc(&d_cand, (size_t) T * ldb));
        int32_t * d_idx; CUDA_CHECK(cudaMalloc(&d_idx, (size_t) T * topk * 4));
        ds_index_scores(d_iq, d_ikc, d_iw, d_scores, T, IH, ID, ratio, n_max, lds, d_pos0, st);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto sc = down(d_scores, (size_t) T * lds);
        std::vector<double> rsc((size_t) T * n_max);
        std::vector<float> gsc((size_t) T * n_max);
        for (int t = 0; t < T; ++t) {
            const int nvis = (pos0 + t + 1) / ratio;
            for (int c = 0; c < n_max; ++c) {
                double a = -INFINITY;
                if (c < nvis) {
                    a = 0;
                    for (int h = 0; h < IH; ++h) { double d = 0; for (int i = 0; i < ID; ++i) d += iq[((size_t) t * IH + h) * ID + i] * ikf[(size_t) c * ID + i]; a += std::max(d, 0.0) * iw[t * IH + h]; }
                }
                rsc[(size_t) t * n_max + c] = a;
                gsc[(size_t) t * n_max + c] = sc[(size_t) t * lds + c];
            }
        }
        {   // compare finite entries only (the -inf tail is checked by equality)
            double maxe = 0; int infbad = 0;
            for (size_t i = 0; i < rsc.size(); ++i) {
                if (std::isinf(rsc[i])) { if (!std::isinf(gsc[i])) ++infbad; continue; }
                maxe = std::max(maxe, fabs(rsc[i] - gsc[i]));
            }
            printf("%-28s max err %.3e, bad -inf %d%s\n", "index_scores", maxe, infbad, maxe > 1e-3 || infbad ? "  FAIL" : "");
            if (maxe > 1e-3 || infbad) ++fails;
        }
        // candidate blocks from the reference scores, then top-k inside them
        ds_cand_blocks(d_scores, lds, T, ratio, bs, k_blocks, (n_max + bs - 1) / bs, d_pos0, d_scratch, d_cand, st);
        ds_topk_select(d_scores, lds, T, ratio, topk, d_pos0, d_cand, bs, d_idx, st);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto cand = down(d_cand, (size_t) T * ldb);
        auto idx = down(d_idx, (size_t) T * topk);
        int bad = 0;
        for (int t = 0; t < T; ++t) {
            const int nvis = (pos0 + t + 1) / ratio, nb = (nvis + bs - 1) / bs;
            std::vector<std::pair<double, int>> bsc(nb);
            for (int b = 0; b < nb; ++b) {
                double m = -INFINITY;
                for (int c = b * bs; c < std::min(nvis, (b + 1) * bs); ++c) m = std::max(m, (double) gsc[(size_t) t * n_max + c]);
                if (b == (nvis - 1) / bs) m = 1e38;
                bsc[b] = {-m, b};
            }
            std::sort(bsc.begin(), bsc.end());
            std::vector<char> ref_c(ldb, 0);
            for (int j = 0; j < std::min(k_blocks, nb); ++j) ref_c[bsc[j].second] = 1;
            // the mask may differ only among tied block scores: compare the set of scores
            int diff = 0;
            for (int b = 0; b < nb; ++b) if (ref_c[b] != cand[(size_t) t * ldb + b]) ++diff;
            // top-k: the chosen scores must equal the reference's k best scores among candidates
            std::vector<double> pool;
            for (int c = 0; c < nvis; ++c) if (cand[(size_t) t * ldb + c / bs]) pool.push_back(gsc[(size_t) t * n_max + c]);
            std::sort(pool.rbegin(), pool.rend());
            std::vector<double> got;
            for (int j = 0; j < topk; ++j) { const int c = idx[(size_t) t * topk + j]; if (c >= 0) got.push_back(gsc[(size_t) t * n_max + c]); }
            std::sort(got.rbegin(), got.rend());
            int kk = std::min<int>(topk, (int) pool.size());
            bool ok = (int) got.size() == kk;
            for (int j = 0; ok && j < kk; ++j) ok = got[j] == pool[j];
            // distinct indices
            std::vector<int> ids; for (int j = 0; j < topk; ++j) if (idx[(size_t) t * topk + j] >= 0) ids.push_back(idx[(size_t) t * topk + j]);
            std::sort(ids.begin(), ids.end());
            ok = ok && std::adjacent_find(ids.begin(), ids.end()) == ids.end();
            if (diff || !ok) { ++bad; printf("   t %d: cand diff %d, topk %s\n", t, diff, ok ? "ok" : "WRONG"); }
        }
        printf("%-28s %s\n", "cand_blocks + topk_select", bad ? "FAIL" : "ok");
        if (bad) ++fails;
        // top-k without candidates on a short context (n_vis < k)
        int * d_p2 = up(std::vector<int>{10});
        ds_topk_select(d_scores, lds, T, ratio, topk, d_p2, nullptr, bs, d_idx, st);
        CUDA_CHECK(cudaDeviceSynchronize());
        idx = down(d_idx, (size_t) T * topk);
        int bad2 = 0;
        for (int t = 0; t < T; ++t) {
            const int nvis = 10 + t + 1;
            std::vector<int> ids;
            for (int j = 0; j < topk; ++j) { const int c = idx[(size_t) t * topk + j]; if (j < nvis ? c < 0 || c >= nvis : c != -1) ++bad2; if (c >= 0) ids.push_back(c); }
            std::sort(ids.begin(), ids.end());
            if ((int) ids.size() != nvis || std::adjacent_find(ids.begin(), ids.end()) != ids.end()) ++bad2;
        }
        printf("%-28s %s\n", "topk_select (short)", bad2 ? "FAIL" : "ok");
        if (bad2) ++fails;
    }
    printf(fails ? "\n%d FAILURES\n" : "\nall ok\n", fails);
    return fails ? 1 : 0;
}
