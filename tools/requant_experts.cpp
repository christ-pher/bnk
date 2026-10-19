// Writes a model's cold experts in a smaller format (docs/EXPERT-MIX.md): the routed experts a routing-count file
// ranks lowest within each layer, dequantized from the model's own blocks (NVFP4's per-expert global scale folded
// in) and re-quantized with ggml's quantizers into one or two bands. A band is "<gate/up type>/<down type>:<percent
// of the layer's experts>"; the down matrices have 640 columns, which the 256-wide IQ / K formats cannot hold, so
// they take a 32- or 64-wide format (q2_0, iq4_nl, q4_0, q8_0). The engine loads the file beside the model
// (--cold-experts) and keeps every one of them resident.
//
//   build/requant_experts --model M.gguf --counts ~/.cache/bnk/counts-M.bnkc --out cold-M.gguf \
//                         [--bands iq2_s/iq4_nl:30,iq3_s/iq4_nl:40] [--down-q2-layers 1,8,10,...] [--threads N]
//
// --down-q2-layers: layers whose bands take Q2_0 down matrices instead of the band's down type (the layers whose
// down projections tolerate 2 bits; the IQ3_S release of Flash-Next uses Q2_0 down in 9 of its 48 layers).
// --borrow OTHER.gguf: no quantization of ours: each cold expert's blocks are copied from another GGUF of the same
// architecture (e.g. the IQ3_S release, quantized with its data-aware method), in whatever formats that file holds
// per layer; the bands then only give the percentages (e.g. --bands x:30,x:40).
// Bands are named b0, b1, ... in the output whatever their formats (tools/cold_drift.py matches by index).
//
// Output tensors per layer N and band B (= "<gate type>_<down type>"): blk.N.cold_B.experts (I32 [n]: the expert
// ids, ascending), blk.N.ffn_{gate,up}_exps.B (gate type [cols, rows, n]) and blk.N.ffn_down_exps.B (down type);
// metadata bnk.cold.model / bnk.cold.bands.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "core/expert_cache.h"
#include "core/gguf_model.h"
#include "core/model.h"
#include "core/util.h"
#include "ggml.h"
#include "gguf.h"

using namespace bnk;

namespace {

struct Band { ggml_type gate, down; int pct; std::string name; };

ggml_type type_of(const std::string & s) {
    static const std::map<std::string, ggml_type> t{{"iq2_xxs", GGML_TYPE_IQ2_XXS}, {"iq2_xs", GGML_TYPE_IQ2_XS},
                                                    {"iq2_s", GGML_TYPE_IQ2_S},     {"iq3_xxs", GGML_TYPE_IQ3_XXS},
                                                    {"iq3_s", GGML_TYPE_IQ3_S},     {"q4_k", GGML_TYPE_Q4_K},
                                                    {"q2_0", GGML_TYPE_Q2_0},       {"iq4_nl", GGML_TYPE_IQ4_NL},
                                                    {"q4_0", GGML_TYPE_Q4_0},       {"q8_0", GGML_TYPE_Q8_0}};
    auto it = t.find(s);
    if (it == t.end()) throw std::runtime_error("unknown band type " + s);
    return it->second;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model, counts, out, bands_s = "iq2_s/iq4_nl:30,iq3_s/iq4_nl:40", q2_layers_s, borrow;
    int threads = (int) std::thread::hardware_concurrency();
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { if (++i >= argc) throw std::runtime_error("missing value for " + a); return std::string(argv[i]); };
        if (a == "--model") model = next();
        else if (a == "--counts") counts = next();
        else if (a == "--out") out = next();
        else if (a == "--bands") bands_s = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--down-q2-layers") q2_layers_s = next();
        else if (a == "--borrow") borrow = next();
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    if (model.empty() || counts.empty() || out.empty()) {
        fprintf(stderr, "usage: requant_experts --model M.gguf --counts counts.bnkc --out cold.gguf [--bands iq2_s:30,iq3_s:40]\n");
        return 2;
    }
    std::vector<Band> bands;
    for (size_t p = 0; p < bands_s.size();) {
        size_t q = bands_s.find(',', p);
        if (q == std::string::npos) q = bands_s.size();
        const std::string item = bands_s.substr(p, q - p);
        const size_t c = item.find(':'), sl = item.find('/');
        if (c == std::string::npos) throw std::runtime_error("band needs gate/down:percent: " + item);
        Band b{GGML_TYPE_F32, GGML_TYPE_F32, std::stoi(item.substr(c + 1)), "b" + std::to_string(bands.size())};
        if (borrow.empty()) {
            if (sl == std::string::npos || sl > c) throw std::runtime_error("band needs gate/down:percent: " + item);
            b.gate = type_of(item.substr(0, sl));
            b.down = type_of(item.substr(sl + 1, c - sl - 1));
        }
        bands.push_back(b);
        p = q + 1;
    }
    std::vector<char> down_q2(4096, 0);
    for (size_t p = 0; p < q2_layers_s.size();) {
        size_t q = q2_layers_s.find(',', p);
        if (q == std::string::npos) q = q2_layers_s.size();
        if (q > p) down_q2.at(std::stoi(q2_layers_s.substr(p, q - p))) = 1;
        p = q + 1;
    }
    if (!q2_layers_s.empty()) ggml_quantize_init(GGML_TYPE_Q2_0);
    GgufModel g;
    g.open(model);
    const std::string arch = g.get_str("general.architecture");
    const int n_layer = (int) g.get_u64(arch + ".block_count") - (int) g.get_u64(arch + ".nextn_predict_layers", 0);
    const int n_expert = (int) g.get_u64(arch + ".expert_count");
    if (n_layer <= 0 || !n_expert) throw std::runtime_error("not a MoE model");
    const std::vector<uint32_t> cnt = load_counts(counts, n_layer, n_expert);
    if (cnt.empty()) throw std::runtime_error("cannot read the counts file (or it is for another model shape)");
    if (borrow.empty())
        for (const Band & b : bands) { ggml_quantize_init(b.gate); ggml_quantize_init(b.down); }
    GgufModel other;   // --borrow
    if (!borrow.empty()) {
        other.open(borrow);
        if (other.get_str("general.architecture") != arch) throw std::runtime_error("--borrow: another architecture");
    }

    // the output: one ggml context for the tensor metadata, data buffers kept until the write
    gguf_context * gc = gguf_init_empty();
    gguf_set_val_str(gc, "general.architecture", "bnk-cold-experts");
    gguf_set_val_str(gc, "bnk.cold.model", g.get_str("general.name", "").c_str());
    gguf_set_val_str(gc, "bnk.cold.bands", bands_s.c_str());
    if (!borrow.empty()) gguf_set_val_str(gc, "bnk.cold.borrow", borrow.c_str());
    if (!q2_layers_s.empty()) gguf_set_val_str(gc, "bnk.cold.down_q2_layers", q2_layers_s.c_str());
    ggml_init_params ip{ggml_tensor_overhead() * (size_t) n_layer * 4 * (bands.size() + 1) + 1024, nullptr, true};
    ggml_context * gx = ggml_init(ip);
    std::vector<std::vector<uint8_t>> bufs;   // tensor data, in order of creation
    std::vector<std::vector<int32_t>> idxs;
    bufs.reserve((size_t) n_layer * 3 * bands.size());
    idxs.reserve((size_t) n_layer * bands.size());
    size_t total_in = 0, total_out = 0;
    const double t0 = now_ms();
    for (int il = 0; il < n_layer; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        const TensorRef * mats[3] = {&g.get(b + "ffn_gate_exps.weight"), &g.get(b + "ffn_up_exps.weight"), &g.get(b + "ffn_down_exps.weight")};
        const TensorRef * scl[3] = {g.find(b + "ffn_gate_exps.scale"), g.find(b + "ffn_up_exps.scale"), g.find(b + "ffn_down_exps.scale")};
        // this layer's experts, coldest first (ties by id)
        std::vector<int> order(n_expert);
        for (int e = 0; e < n_expert; ++e) order[e] = e;
        std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return cnt[(size_t) il * n_expert + x] < cnt[(size_t) il * n_expert + y]; });
        int from = 0;
        for (const Band & band : bands) {
            const int n = n_expert * band.pct / 100;
            std::vector<int32_t> ids(order.begin() + from, order.begin() + from + n);
            from += n;
            std::sort(ids.begin(), ids.end());
            const char * tname = band.name.c_str();
            idxs.emplace_back(ids);
            ggml_tensor * ti = ggml_new_tensor_1d(gx, GGML_TYPE_I32, n);
            ggml_set_name(ti, (b + "cold_" + tname + ".experts").c_str());
            ti->data = idxs.back().data();
            gguf_add_tensor(gc, ti);
            static const char * mname[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
            for (int m = 0; m < 3; ++m) {
                const TensorRef & src = *mats[m];
                const int64_t cols = src.ne[0], rows = src.ne[1];
                const size_t in_per = src.nbytes / n_expert;
                const TensorRef * bsrc = borrow.empty() ? nullptr : &other.get(b + mname[m] + ".weight");
                const ggml_type ot = bsrc ? bsrc->type : m == 2 ? (down_q2[il] ? GGML_TYPE_Q2_0 : band.down) : band.gate;
                const size_t out_row = ggml_row_size(ot, cols), out_per = out_row * rows;
                bufs.emplace_back(out_per * n);
                std::vector<uint8_t> & dst = bufs.back();
                if (bsrc) {   // the other file's blocks, as they are
                    if (bsrc->ne[0] != cols || bsrc->ne[1] != rows || bsrc->ne[2] != n_expert || bsrc->nbytes / n_expert != out_per)
                        throw std::runtime_error("--borrow: tensor shape mismatch in " + bsrc->name);
                    for (int j = 0; j < n; ++j) memcpy(dst.data() + (size_t) j * out_per, bsrc->data + (size_t) ids[j] * out_per, out_per);
                    ggml_tensor * tw = ggml_new_tensor_3d(gx, ot, cols, rows, n);
                    ggml_set_name(tw, (b + mname[m] + "." + tname).c_str());
                    tw->data = dst.data();
                    gguf_add_tensor(gc, tw);
                    total_in += in_per * n;
                    total_out += out_per * n;
                    continue;
                }
                const auto * tt = ggml_get_type_traits(src.type);
                if (!tt->to_float) throw std::runtime_error(std::string("no dequantizer for ") + ggml_type_name(src.type));
                const float * scale = scl[m] ? (const float *) scl[m]->data : nullptr;
                std::atomic<int> nextj{0};
                auto worker = [&]() {
                    std::vector<float> f((size_t) rows * cols);
                    for (int j; (j = nextj.fetch_add(1)) < n;) {
                        const int e = ids[j];
                        tt->to_float(src.data + (size_t) e * in_per, f.data(), rows * cols);
                        if (scale) {
                            const float s = scale[e];
                            for (float & v : f) v *= s;
                        }
                        ggml_quantize_chunk(ot, f.data(), dst.data() + (size_t) j * out_per, 0, rows, cols, nullptr);
                    }
                };
                std::vector<std::thread> th;
                for (int t = 0; t < std::max(1, threads); ++t) th.emplace_back(worker);
                for (auto & t : th) t.join();
                ggml_tensor * tw = ggml_new_tensor_3d(gx, ot, cols, rows, n);
                ggml_set_name(tw, (b + mname[m] + "." + tname).c_str());
                tw->data = dst.data();
                gguf_add_tensor(gc, tw);
                total_in += in_per * n;
                total_out += out_per * n;
            }
        }
        fprintf(stderr, "\rlayer %d/%d: %.1f GiB -> %.1f GiB (%.0f s)", il + 1, n_layer, total_in / 1073741824.0,
                total_out / 1073741824.0, (now_ms() - t0) / 1000);
    }
    fprintf(stderr, "\nwriting %s ...\n", out.c_str());
    if (!gguf_write_to_file(gc, out.c_str(), false)) throw std::runtime_error("cannot write " + out);
    fprintf(stderr, "done: %.2f GiB of NVFP4 experts became %.2f GiB (%s), %.0f s\n", total_in / 1073741824.0,
            total_out / 1073741824.0, bands_s.c_str(), (now_ms() - t0) / 1000);
    gguf_free(gc);
    ggml_free(gx);
    return 0;
}
