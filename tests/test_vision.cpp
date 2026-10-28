// Runs the vision encoder on a raw RGB image (H x W x 3 bytes, sides multiples of the encoder's unit: 32 for Qwen,
// 14 for DeepSeek) and writes its embeddings (fp32 [cells][out_dim]): compared against llama.cpp's encoder by
// tests/compare_vision.py (Qwen) or tools/ds_vision_ref.py (DeepSeek). DeepSeek's encoder runs in the fixed workspace
// the engine gives it, sized for BNK_IMAGE_MAX_TOKENS (1,024) cells.
//   test_vision MMPROJ IMAGE.rgb W H OUT.emb [repeats]
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/util.h"
#include "vision/vision.h"

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: test_vision mmproj image.rgb W H out.emb [repeats]\n"); return 1; }
    const int W = atoi(argv[3]), H = atoi(argv[4]), reps = argc > 6 ? atoi(argv[6]) : 1;
    std::vector<uint8_t> rgb((size_t) W * H * 3);
    FILE * f = fopen(argv[2], "rb");
    if (!f || fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    fclose(f);
    bnk::VisionEncoder enc;
    enc.load(argv[1], 0, nullptr);
    const auto & c = enc.cfg();
    const int n = c.cells(H, W);
    void * ws = nullptr;
    if (c.kind == bnk::Projector::DeepSeek41V) {
        const size_t bytes = enc.workspace_bytes(getenv("BNK_IMAGE_MAX_TOKENS") ? atoi(getenv("BNK_IMAGE_MAX_TOKENS")) : 1024);
        CUDA_CHECK(cudaMalloc(&ws, bytes));
        enc.set_workspace(ws, bytes);
        printf("workspace %.0f MiB\n", bytes / 1048576.0);
    }
    std::vector<float> out((size_t) n * c.out_dim);
    for (int r = 0; r < reps; ++r) {
        const double t0 = bnk::now_ms();
        enc.encode(rgb.data(), H, W, out.data());
        printf("encode %d x %d: %d tokens, %.1f ms\n", W, H, n, bnk::now_ms() - t0);
    }
    f = fopen(argv[5], "wb");
    fwrite(out.data(), 4, out.size(), f);
    fclose(f);
    return 0;
}
