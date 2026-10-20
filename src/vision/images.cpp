#include "vision/images.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "core/util.h"

namespace bnk {

void ImageStore::open(const std::string & mmproj, int device, VramBudget * budget, bool verbose) {
    enc_.load(mmproj, device, budget, verbose);
}

void ImageStore::add(int32_t id, const std::string & path, int H, int W) {
    if (!is_image_id(id)) throw std::runtime_error("image ids are negative");
    const int u = enc_.cfg().unit();
    if (H <= 0 || W <= 0 || H % u || W % u) throw std::runtime_error("image sides must be multiples of " + std::to_string(u));
    std::lock_guard<std::mutex> lk(mu_);
    images_[id] = Image{path, H, W, H / u, W / u};
}

bool ImageStore::grid(int32_t id, int * gh, int * gw) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = images_.find(id);
    if (it == images_.end()) return false;
    *gh = it->second.gh;
    *gw = it->second.gw;
    return true;
}

const float * ImageStore::rows(int32_t id) {
    std::lock_guard<std::mutex> lk(mu_);
    if (auto it = cache_.find(id); it != cache_.end()) {
        lru_.splice(lru_.begin(), lru_, it->second.lru);
        return it->second.rows.data();
    }
    auto im = images_.find(id);
    if (im == images_.end()) throw std::runtime_error("image " + std::to_string(id) + " is not registered");
    const Image & g = im->second;
    std::vector<uint8_t> rgb((size_t) g.H * g.W * 3);
    FILE * f = fopen(g.path.c_str(), "rb");
    const bool ok = f && fread(rgb.data(), 1, rgb.size(), f) == rgb.size();
    if (f) fclose(f);
    if (!ok) throw std::runtime_error("cannot read the image file " + g.path);
    std::vector<float> out((size_t) g.gh * g.gw * enc_.cfg().out_dim);
    enc_.encode(rgb.data(), g.H, g.W, out.data());
    const size_t bytes = out.size() * 4;
    while (!lru_.empty() && cached_bytes_ + bytes > cache_limit) {
        auto old = cache_.find(lru_.back());
        cached_bytes_ -= old->second.rows.size() * 4;
        cache_.erase(old);
        lru_.pop_back();
    }
    lru_.push_front(id);
    Cached & c = cache_[id];
    c.rows = std::move(out);
    c.lru = lru_.begin();
    cached_bytes_ += bytes;
    return c.rows.data();
}

void ImageStore::encode_all(const int32_t * t, int n) {
    for (int i = 0; i < n; ++i)
        if (is_image_id(t[i]) && (i == 0 || t[i - 1] != t[i])) rows(t[i]);
}

void ImageStore::put_rows(const std::vector<int32_t> & h, int pos0, int N, float * dst, cudaStream_t s) {
    const int E = enc_.cfg().out_dim;
    for (int t = 0; t < N;) {
        const int32_t id = h[pos0 + t];
        if (!is_image_id(id)) { ++t; continue; }
        int gh = 0, gw = 0;
        if (!grid(id, &gh, &gw)) throw std::runtime_error("image " + std::to_string(id) + " is not registered");
        int start = pos0 + t;
        while (start > 0 && h[start - 1] == id) --start;
        const float * src = rows(id);
        // cells of the same image in a row: one copy (a run longer than the image is the image again)
        const int j = (pos0 + t - start) % (gh * gw);
        int n = 0;
        while (t + n < N && h[pos0 + t + n] == id && j + n < gh * gw) ++n;
        CUDA_CHECK(cudaMemcpyAsync(dst + (size_t) t * E, src + (size_t) j * E, (size_t) n * E * 4, cudaMemcpyHostToDevice, s));
        t += n;
    }
}

void ImageStore::positions(const std::vector<int32_t> & seq, int cells, std::vector<int32_t> & out) const {
    out.resize((size_t) cells * 3);
    int32_t p = 0;   // the next text position
    int i = 0;
    const int n = std::min<int>(cells, (int) seq.size());
    while (i < cells) {
        if (i >= n || !is_image_id(seq[i])) {
            out[(size_t) i * 3] = out[(size_t) i * 3 + 1] = out[(size_t) i * 3 + 2] = p++;
            ++i;
            continue;
        }
        int gh = 0, gw = 0;
        if (!grid(seq[i], &gh, &gw)) throw std::runtime_error("image " + std::to_string(seq[i]) + " is not registered");
        // one image: gh * gw cells of the same id (a longer run is the same image again)
        const int32_t id = seq[i], p0 = p;
        int j = 0;
        for (; j < gh * gw && i < n && seq[i] == id; ++j, ++i) {
            out[(size_t) i * 3] = p0;
            out[(size_t) i * 3 + 1] = p0 + j / gw;
            out[(size_t) i * 3 + 2] = p0 + j % gw;
        }
        p = p0 + std::max(gh, gw);
    }
}

}  // namespace bnk
