#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                                                    \
    do {                                                                                                 \
        cudaError_t e_ = (x);                                                                            \
        if (e_ != cudaSuccess)                                                                           \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " +    \
                                     __FILE__ + ":" + std::to_string(__LINE__));                        \
    } while (0)

namespace bnk {

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : def;
}

template <typename T>
struct DevBuf {
    T * p = nullptr;
    size_t n = 0;
    bool owned = true;   // false: a view into memory someone else manages (an ElasticBuf)
    void alloc(size_t count) {
        free();
        n = count;
        CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
        CUDA_CHECK(cudaMemset(p, 0, count * sizeof(T)));
    }
    void view(T * ptr, size_t count) {
        free();
        p = ptr;
        n = count;
        owned = false;
    }
    void free() {
        if (p && owned) cudaFree(p);
        p = nullptr;
        n = 0;
        owned = true;
    }
    ~DevBuf() { free(); }
    operator T *() const { return p; }
};

}  // namespace bnk
