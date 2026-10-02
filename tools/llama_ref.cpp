// Reference dumps from llama.cpp (CPU backend) for bisecting bnk:
//   llama_ref MODEL.gguf TOKENS_FILE OUT_PREFIX [tensor-name-filter ...]
// Writes OUT_PREFIX.logits (float32 [n_tokens][n_vocab]) and, for each named intermediate tensor
// whose name starts with one of the filters (e.g. "l_last-", "attn_output-3"), OUT_PREFIX.<name>
// as raw float32 in ggml's memory order. Default filter: "l_last-" (the HC residual after each layer).
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

struct Dump {
    std::string prefix;
    std::vector<std::string> filters;
};

static bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * d = (Dump *) ud;
    const std::string name = t->name;
    bool want = false;
    for (auto & f : d->filters)
        if (name.rfind(f, 0) == 0) want = true;
    if (ask) return want;
    if (!want || t->type != GGML_TYPE_F32) return true;
    std::vector<float> buf(ggml_nelements(t));
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    } else {
        return true;
    }
    std::string fn = d->prefix + "." + name;
    FILE * f = fopen(fn.c_str(), "wb");
    fwrite(buf.data(), 4, buf.size(), f);
    fclose(f);
    fprintf(stderr, "dumped %s [%lld %lld %lld %lld]\n", name.c_str(), (long long) t->ne[0], (long long) t->ne[1],
            (long long) t->ne[2], (long long) t->ne[3]);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL TOKENS OUT_PREFIX [filters...]\n", argv[0]);
        return 1;
    }
    std::vector<llama_token> toks;
    {
        std::ifstream in(argv[2]);
        std::string s((std::istreambuf_iterator<char>(in)), {});
        for (char & c : s) if (c == ',') c = ' ';
        std::istringstream is(s);
        long v;
        while (is >> v) toks.push_back((llama_token) v);
    }
    Dump d;
    d.prefix = argv[3];
    for (int i = 4; i < argc; ++i) d.filters.push_back(argv[i]);
    if (d.filters.empty()) d.filters.push_back("l_last-");

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 4096;
    cp.n_ubatch = 4096;
    cp.n_threads = 24;
    cp.n_threads_batch = 24;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &d;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 3;

    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i];
        b.pos[i] = (llama_pos) i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
    }
    b.n_tokens = (int) toks.size();
    if (llama_decode(ctx, b)) {
        fprintf(stderr, "decode failed\n");
        return 4;
    }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * f = fopen((d.prefix + ".logits").c_str(), "wb");
    for (size_t i = 0; i < toks.size(); ++i) {
        const float * lg = llama_get_logits_ith(ctx, (int) i);
        fwrite(lg, 4, nv, f);
        int am = 0;
        for (int j = 1; j < nv; ++j) if (lg[j] > lg[am]) am = j;
        if (i + 5 >= toks.size()) fprintf(stderr, "pos %zu argmax %d (%.4f)\n", i, am, lg[am]);
    }
    fclose(f);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
