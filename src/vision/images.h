// The images of prompts. A prompt carries an image as a run of placeholder cells, one per merged patch, that
// all hold the same negative token id (-1 - key, the key a hash of the image's pixels): prefix reuse, parking
// and forking then compare images like tokens, and the engine maps the id back to the vocabulary's image token
// for the embedding row and the PLE n-grams (the reference hashes that token for image cells), then overwrites
// the embedding rows with the encoder's. The server registers each image (its preprocessed pixels in a file)
// before the request; embeddings are encoded on first use and kept in a host cache.
#pragma once

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "vision/vision.h"

namespace bnk {

inline bool is_image_id(int32_t t) { return t < 0; }

class ImageStore {
public:
    // loads the encoder (stage 0's device and VRAM budget)
    void open(const std::string & mmproj, int device, VramBudget * budget, bool verbose);
    bool enabled() const { return enc_.loaded(); }
    const VisionConfig & cfg() const { return enc_.cfg(); }
    // an image of H x W pixels (sides multiples of cfg().unit()) whose raw RGB bytes are in `path`
    void add(int32_t id, const std::string & path, int H, int W);
    // its merged grid (rows, columns); false: not registered
    bool grid(int32_t id, int * gh, int * gw) const;
    // its embeddings, [gh * gw][out_dim] (encoded now if not cached); valid until the next call
    const float * rows(int32_t id);
    // encodes the images among tokens[0, n) not cached yet (before buffers are sized around the VRAM it borrows)
    void encode_all(const int32_t * tokens, int n);
    size_t cache_limit = (size_t) 2 << 30;   // bytes of embeddings kept
    VisionEncoder & encoder() { return enc_; }

    // the image cells among seq[pos0, pos0 + N): their embeddings into rows of dst (device, [N][out_dim]) on `s`
    // (an image's cells may start before pos0: its earlier part came in an earlier chunk)
    void put_rows(const std::vector<int32_t> & seq, int pos0, int N, float * dst, cudaStream_t s);

    // M-RoPE positions (time, row, column) of cells [0, cells) for a token sequence `seq` (cells past it are
    // text): text advances one position per cell; an image of gh x gw takes its first position p for every cell's
    // time, p + row and p + column for the others, and the text after it continues at p + max(gh, gw)
    void positions(const std::vector<int32_t> & seq, int cells, std::vector<int32_t> & out) const;

private:
    struct Image { std::string path; int H = 0, W = 0, gh = 0, gw = 0; };
    struct Cached { std::vector<float> rows; std::list<int32_t>::iterator lru; };
    VisionEncoder enc_;
    std::unordered_map<int32_t, Image> images_;
    std::unordered_map<int32_t, Cached> cache_;
    std::list<int32_t> lru_;   // most recent first
    size_t cached_bytes_ = 0;
    mutable std::mutex mu_;
};

}  // namespace bnk
