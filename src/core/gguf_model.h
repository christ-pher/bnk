// Multi-shard GGUF access: metadata through ggml's gguf parser, tensor bytes through mmap.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"

struct gguf_context;

namespace bnk {

struct TensorRef {
    std::string name;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};
    int n_dims = 0;
    const uint8_t * data = nullptr;  // into the shard's mapping
    size_t nbytes = 0;
    int shard = 0;
    uint64_t file_offset = 0;        // absolute offset in its shard file

    size_t row_bytes() const { return ggml_row_size(type, ne[0]); }
    int64_t nrows() const { return ne[1] * ne[2] * ne[3]; }
};

class GgufModel {
public:
    GgufModel() = default;
    ~GgufModel();
    GgufModel(const GgufModel &) = delete;
    GgufModel & operator=(const GgufModel &) = delete;

    // Opens `path` and, when its split metadata says so, every sibling shard.
    void open(const std::string & path);

    const TensorRef * find(const std::string & name) const;
    const TensorRef & get(const std::string & name) const;  // throws when absent
    const std::vector<TensorRef> & tensors() const { return tensors_; }

    bool has_key(const std::string & key) const;
    std::string get_str(const std::string & key, const std::string & def = "") const;
    uint64_t get_u64(const std::string & key, uint64_t def = 0) const;  // any integer kind
    double get_f64(const std::string & key, double def = 0) const;      // any numeric kind
    bool get_bool(const std::string & key, bool def = false) const;
    std::vector<int64_t> get_int_arr(const std::string & key) const;
    std::vector<std::string> get_str_arr(const std::string & key) const;
    std::vector<float> get_f32_arr(const std::string & key) const;

    const std::vector<std::string> & shard_paths() const { return paths_; }
    gguf_context * meta() const { return ctx_.empty() ? nullptr : ctx_[0]; }

    // Advise the kernel about a byte range (MADV_WILLNEED / MADV_SEQUENTIAL ...).
    void advise(const TensorRef & t, int advice) const;

private:
    struct Map { void * base = nullptr; size_t size = 0; int fd = -1; };
    std::vector<std::string> paths_;
    std::vector<gguf_context *> ctx_;
    std::vector<Map> maps_;
    std::vector<TensorRef> tensors_;
    std::unordered_map<std::string, size_t> index_;
};

std::string ggml_type_str(ggml_type t);

}  // namespace bnk
