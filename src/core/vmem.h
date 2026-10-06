// Elastic VRAM: buffers with a fixed virtual address range whose physical backing grows and shrinks (CUDA virtual
// memory management), all drawing on one budget.
//
// The expert cache holds whatever the budget has left. A buffer that needs more (the KV cache as a conversation
// grows, the prompt path's buffers for a long prompt) asks the budget, which takes the bytes back from the cache
// through its reclaim hook; bytes given back are offered to the cache again. Addresses never move, so captured
// CUDA graphs and device-side pointers stay valid; only the mapped extent changes.
#pragma once

#include <cuda.h>

#include <functional>
#include <string>
#include <vector>

namespace bnk {

// Allocation granularity of physical VRAM (2 MiB on Volta).
size_t vmem_granularity();
inline size_t vmem_round(size_t b) {
    const size_t g = vmem_granularity();
    return (b + g - 1) / g * g;
}

class VramBudget {
public:
    // limit: bytes of physical VRAM the elastic buffers and the cache may hold together
    void set_limit(size_t limit) { limit_ = limit; }
    // bytes to keep physically free beyond the budget (cuBLAS and graph workspaces, swap staging, ...)
    void set_reserve(size_t reserve) { reserve_ = reserve; }
    // re-derives the limit from what the GPU really has free: allocations made outside the budget (lazily created
    // graphs and workspaces) would otherwise let it promise memory that cuMemCreate then cannot deliver
    void sync();
    size_t limit() const { return limit_; }
    size_t used() const { return used_; }
    size_t free() const { return limit_ > used_ ? limit_ - used_ : 0; }
    // the cache's hook: give back at least `bytes` (returns how many it freed)
    std::function<size_t(size_t)> reclaim;
    // makes `bytes` available (reclaiming from the cache when needed); false when even that is not enough
    bool make_room(size_t bytes);
    // the state behind the last make_room (for error messages)
    std::string describe() const;
    void charge(size_t bytes) { used_ += bytes; }
    void refund(size_t bytes) { used_ -= bytes < used_ ? bytes : used_; }

private:
    size_t limit_ = 0, used_ = 0, reserve_ = 0, last_reclaimed_ = 0, last_driver_free_ = 0;
    bool reclaiming_ = false;
    friend class ElasticBuf;
};

class ElasticBuf {
public:
    ~ElasticBuf();
    // reserves `max_bytes` of address space (nothing mapped yet); reclaim_on_oom: when the driver is out of memory
    // although the budget agreed, re-sync the budget and reclaim from the cache once more (not for the cache itself)
    void reserve(size_t max_bytes, VramBudget * budget, const char * name, bool reclaim_on_oom = true);
    // maps [0, bytes) (rounded up to the granularity); throws when the budget cannot cover it
    void ensure(size_t bytes);
    // unmaps everything past `bytes` (rounded up)
    void shrink_to(size_t bytes);
    void * ptr() const { return (void *) va_; }
    size_t mapped() const { return mapped_; }
    size_t reserved() const { return reserved_; }
    template <typename T> T * as() const { return (T *) va_; }

private:
    CUdeviceptr va_ = 0;
    size_t reserved_ = 0, mapped_ = 0;
    VramBudget * budget_ = nullptr;
    std::string name_;
    bool reclaim_on_oom_ = true;
    bool map_pieces(size_t from, size_t to);
    struct Piece { size_t off, size; CUmemGenericAllocationHandle h; };
    std::vector<Piece> pieces_;
};

}  // namespace bnk
