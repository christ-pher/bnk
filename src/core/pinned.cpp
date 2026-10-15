#include "core/pinned.h"

#include <cuda_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstdio>
#include <thread>
#include <vector>

namespace bnk {

namespace {

// MemAvailable in bytes (0 when unreadable)
size_t mem_available() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
    fclose(f);
    return kb * 1024;
}

}  // namespace

void * pin_host(size_t bytes) {
    if (bytes == 0) return nullptr;
    // never let pinning push the host into the OOM killer: leave 2 GiB besides
    const size_t avail = mem_available();
    if (avail && bytes + (2ull << 30) > avail) return nullptr;
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    madvise(p, bytes, MADV_HUGEPAGE);
    // fault the pages in from several threads (registering faults them in on one)
    const size_t step = 2ull << 20;
    const int threads = (int) std::min<size_t>(8, (bytes + (64ull << 20) - 1) / (64ull << 20));
    if (threads > 1) {
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t)
            th.emplace_back([=] {
                for (size_t o = (size_t) t * step; o < bytes; o += (size_t) threads * step)
                    for (size_t q = o; q < std::min(bytes, o + step); q += 4096) ((volatile char *) p)[q] = 0;
            });
        for (auto & t : th) t.join();
    }
    if (cudaHostRegister(p, bytes, cudaHostRegisterPortable) != cudaSuccess) {
        cudaGetLastError();
        munmap(p, bytes);
        return nullptr;
    }
    return p;
}

void unpin_host(void * p, size_t bytes) {
    if (!p) return;
    cudaHostUnregister(p);
    munmap(p, bytes);
}

}  // namespace bnk
