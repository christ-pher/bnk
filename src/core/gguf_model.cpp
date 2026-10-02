#include "core/gguf_model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <regex>
#include <stdexcept>

#include "gguf.h"

namespace bnk {

static std::string fmt_shard(const std::string & prefix, int i, int n, const std::string & suffix) {
    char buf[64];
    snprintf(buf, sizeof buf, "-%05d-of-%05d", i, n);
    return prefix + buf + suffix;
}

GgufModel::~GgufModel() {
    for (auto & m : maps_) {
        if (m.base) munmap(m.base, m.size);
        if (m.fd >= 0) close(m.fd);
    }
    for (auto * c : ctx_) gguf_free(c);
}

void GgufModel::open(const std::string & path) {
    paths_.clear();
    paths_.push_back(path);

    gguf_init_params p{true, nullptr};
    gguf_context * first = gguf_init_from_file(path.c_str(), p);
    if (!first) throw std::runtime_error("cannot parse GGUF: " + path);
    ctx_.push_back(first);

    int64_t k = gguf_find_key(first, "split.count");
    int n_split = k >= 0 ? (int) gguf_get_val_u16(first, k) : 1;
    if (n_split > 1) {
        std::smatch m;
        static const std::regex re("^(.*)-(\\d{5})-of-(\\d{5})(\\.gguf)$");
        if (!std::regex_match(path, m, re)) throw std::runtime_error("split GGUF with unexpected name: " + path);
        const std::string prefix = m[1], suffix = m[4];
        if (std::stoi(m[2]) != 1) throw std::runtime_error("open the first shard (-00001-of-...): " + path);
        for (int i = 2; i <= n_split; ++i) {
            std::string sp = fmt_shard(prefix, i, n_split, suffix);
            gguf_context * c = gguf_init_from_file(sp.c_str(), p);
            if (!c) throw std::runtime_error("cannot parse GGUF shard: " + sp);
            paths_.push_back(sp);
            ctx_.push_back(c);
        }
    }

    for (size_t s = 0; s < paths_.size(); ++s) {
        Map mp;
        mp.fd = ::open(paths_[s].c_str(), O_RDONLY);
        if (mp.fd < 0) throw std::runtime_error("open failed: " + paths_[s]);
        struct stat st;
        fstat(mp.fd, &st);
        mp.size = (size_t) st.st_size;
        mp.base = mmap(nullptr, mp.size, PROT_READ, MAP_SHARED, mp.fd, 0);
        if (mp.base == MAP_FAILED) throw std::runtime_error("mmap failed: " + paths_[s]);
        madvise(mp.base, mp.size, MADV_RANDOM);
        maps_.push_back(mp);

        gguf_context * c = ctx_[s];
        const size_t data_off = gguf_get_data_offset(c);
        for (int64_t t = 0; t < gguf_get_n_tensors(c); ++t) {
            TensorRef r;
            r.name = gguf_get_tensor_name(c, t);
            r.type = gguf_get_tensor_type(c, t);
            const int64_t * ne = gguf_get_tensor_ne(c, t);
            for (int d = 0; d < 4; ++d) r.ne[d] = ne[d];
            r.n_dims = 1;
            for (int d = 1; d < 4; ++d) if (ne[d] > 1) r.n_dims = d + 1;
            r.nbytes = gguf_get_tensor_size(c, t);
            r.shard = (int) s;
            r.file_offset = data_off + gguf_get_tensor_offset(c, t);
            if (r.file_offset + r.nbytes > mp.size) throw std::runtime_error("tensor out of file bounds: " + r.name);
            r.data = (const uint8_t *) mp.base + r.file_offset;
            index_[r.name] = tensors_.size();
            tensors_.push_back(std::move(r));
        }
    }
}

const TensorRef * GgufModel::find(const std::string & name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

const TensorRef & GgufModel::get(const std::string & name) const {
    const TensorRef * t = find(name);
    if (!t) throw std::runtime_error("tensor not found: " + name);
    return *t;
}

bool GgufModel::has_key(const std::string & key) const { return gguf_find_key(meta(), key.c_str()) >= 0; }

std::string GgufModel::get_str(const std::string & key, const std::string & def) const {
    int64_t k = gguf_find_key(meta(), key.c_str());
    if (k < 0) return def;
    return gguf_get_val_str(meta(), k);
}

static uint64_t scalar_int(gguf_context * c, int64_t k, gguf_type t, const void * data) {
    switch (t) {
        case GGUF_TYPE_UINT8: return *(const uint8_t *) data;
        case GGUF_TYPE_INT8: return (uint64_t) *(const int8_t *) data;
        case GGUF_TYPE_UINT16: return *(const uint16_t *) data;
        case GGUF_TYPE_INT16: return (uint64_t) *(const int16_t *) data;
        case GGUF_TYPE_UINT32: return *(const uint32_t *) data;
        case GGUF_TYPE_INT32: return (uint64_t) *(const int32_t *) data;
        case GGUF_TYPE_UINT64: return *(const uint64_t *) data;
        case GGUF_TYPE_INT64: return (uint64_t) *(const int64_t *) data;
        case GGUF_TYPE_BOOL: return *(const int8_t *) data ? 1 : 0;
        case GGUF_TYPE_FLOAT32: return (uint64_t) *(const float *) data;
        case GGUF_TYPE_FLOAT64: return (uint64_t) *(const double *) data;
        default: (void) c; (void) k; throw std::runtime_error("not an integer GGUF value");
    }
}

static size_t gguf_scalar_size(gguf_type t) {
    switch (t) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

uint64_t GgufModel::get_u64(const std::string & key, uint64_t def) const {
    int64_t k = gguf_find_key(meta(), key.c_str());
    if (k < 0) return def;
    gguf_type t = gguf_get_kv_type(meta(), k);
    if (t == GGUF_TYPE_ARRAY) {  // some writers store a per-layer array; take the first
        auto v = get_int_arr(key);
        return v.empty() ? def : (uint64_t) v[0];
    }
    return scalar_int(meta(), k, t, gguf_get_val_data(meta(), k));
}

double GgufModel::get_f64(const std::string & key, double def) const {
    int64_t k = gguf_find_key(meta(), key.c_str());
    if (k < 0) return def;
    gguf_type t = gguf_get_kv_type(meta(), k);
    if (t == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(meta(), k);
    if (t == GGUF_TYPE_FLOAT64) return gguf_get_val_f64(meta(), k);
    return (double) (int64_t) get_u64(key, (uint64_t) def);
}

bool GgufModel::get_bool(const std::string & key, bool def) const {
    int64_t k = gguf_find_key(meta(), key.c_str());
    if (k < 0) return def;
    return get_u64(key) != 0;
}

std::vector<int64_t> GgufModel::get_int_arr(const std::string & key) const {
    std::vector<int64_t> out;
    int64_t k = gguf_find_key(meta(), key.c_str());
    if (k < 0) return out;
    if (gguf_get_kv_type(meta(), k) != GGUF_TYPE_ARRAY) {
        out.push_back((int64_t) scalar_int(meta(), k, gguf_get_kv_type(meta(), k), gguf_get_val_data(meta(), k)));
        return out;
    }
    gguf_type at = gguf_get_arr_type(meta(), k);
    size_t n = gguf_get_arr_n(meta(), k), sz = gguf_scalar_size(at);
    const uint8_t * d = (const uint8_t *) gguf_get_arr_data(meta(), k);
    for (size_t i = 0; i < n; ++i) out.push_back((int64_t) scalar_int(meta(), k, at, d + i * sz));
    return out;
}

std::vector<float> GgufModel::get_f32_arr(const std::string & key) const {
    std::vector<float> out;
    int64_t k = gguf_find_key(meta(), key.c_str());
    if (k < 0) return out;
    if (gguf_get_kv_type(meta(), k) != GGUF_TYPE_ARRAY) { out.push_back((float) get_f64(key)); return out; }
    gguf_type at = gguf_get_arr_type(meta(), k);
    size_t n = gguf_get_arr_n(meta(), k);
    const uint8_t * d = (const uint8_t *) gguf_get_arr_data(meta(), k);
    for (size_t i = 0; i < n; ++i) {
        if (at == GGUF_TYPE_FLOAT32) out.push_back(((const float *) d)[i]);
        else if (at == GGUF_TYPE_FLOAT64) out.push_back((float) ((const double *) d)[i]);
        else out.push_back((float) (int64_t) scalar_int(meta(), k, at, d + i * gguf_scalar_size(at)));
    }
    return out;
}

std::vector<std::string> GgufModel::get_str_arr(const std::string & key) const {
    std::vector<std::string> out;
    int64_t k = gguf_find_key(meta(), key.c_str());
    if (k < 0) return out;
    size_t n = gguf_get_arr_n(meta(), k);
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.emplace_back(gguf_get_arr_str(meta(), k, i));
    return out;
}

void GgufModel::advise(const TensorRef & t, int advice) const {
    const size_t pg = 4096;
    uintptr_t a = (uintptr_t) t.data & ~(pg - 1);
    size_t len = (uintptr_t) t.data + t.nbytes - a;
    madvise((void *) a, len, advice);
}

std::string ggml_type_str(ggml_type t) { return ggml_type_name(t); }

}  // namespace bnk
