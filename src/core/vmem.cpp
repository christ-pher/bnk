#include "core/vmem.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include <cuda_runtime.h>

namespace bnk {

namespace {

void cu_check(CUresult r, const char * what) {
    if (r != CUDA_SUCCESS) {
        const char * s = nullptr;
        cuGetErrorString(r, &s);
        throw std::runtime_error(std::string("CUDA VMM ") + what + ": " + (s ? s : "error"));
    }
}

CUmemAllocationProp prop_for_device() {
    CUdevice dev;
    int ord = 0;
    cudaGetDevice(&ord);
    cu_check(cuDeviceGet(&dev, ord), "cuDeviceGet");
    CUmemAllocationProp p = {};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = dev;
    return p;
}

}  // namespace

size_t vmem_granularity() {
    static size_t g = 0;
    if (!g) {
        CUmemAllocationProp p = prop_for_device();
        cu_check(cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), "granularity");
    }
    return g;
}

void VramBudget::sync() {
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    last_driver_free_ = f;
    // below the reserve, the limit drops under what is used: the deficit is owed back by the cache
    const size_t limit = used_ + f > reserve_ ? used_ + f - reserve_ : 0;
    static const bool log = getenv("BNK_VRAM_LOG") != nullptr;
    if (log && limit != limit_)
        fprintf(stderr, "bnk: VRAM budget sync: limit %+lld MiB (used %zu MiB, driver free %zu MiB)\n",
                ((long long) limit - (long long) limit_) / 1048576, used_ >> 20, f >> 20);
    limit_ = limit;
}

bool VramBudget::make_room(size_t bytes) {
    sync();
    // short by: what `bytes` needs beyond the limit, plus any amount the budget is already over it (memory taken
    // outside the budget since the last sync comes back out of the cache, restoring the reserve)
    auto short_by = [&] { return used_ + bytes > limit_ ? used_ + bytes - limit_ : (size_t) 0; };
    last_reclaimed_ = 0;
    if (!short_by()) return true;
    // a few rounds: the cache frees whole granules per layer, and the driver's count can move meanwhile
    for (int round = 0; round < 4 && short_by() && reclaim && !reclaiming_; ++round) {
        reclaiming_ = true;
        size_t got = 0;
        try {
            got = reclaim(short_by());
        } catch (...) {
            reclaiming_ = false;
            throw;
        }
        reclaiming_ = false;
        last_reclaimed_ += got;
        sync();
        if (!got) break;
    }
    return !short_by();
}

std::string VramBudget::describe() const {
    char b[200];
    snprintf(b, sizeof b, "budget used %zu / limit %zu MiB, driver free %zu MiB, reserve %zu MiB, cache gave %zu MiB%s",
             used_ >> 20, limit_ >> 20, last_driver_free_ >> 20, reserve_ >> 20, last_reclaimed_ >> 20,
             reclaiming_ ? ", inside a reclaim" : "");
    return b;
}

ElasticBuf::~ElasticBuf() {
    shrink_to(0);
    if (va_) cuMemAddressFree(va_, reserved_);
}

void ElasticBuf::reserve(size_t max_bytes, VramBudget * budget, const char * name, bool reclaim_on_oom) {
    budget_ = budget;
    name_ = name;
    reclaim_on_oom_ = reclaim_on_oom;
    reserved_ = vmem_round(max_bytes ? max_bytes : 1);
    cu_check(cuMemAddressReserve(&va_, reserved_, 0, 0, 0), "reserve");
}

void ElasticBuf::ensure(size_t bytes) {
    bytes = vmem_round(bytes);
    if (bytes <= mapped_) return;
    if (bytes > reserved_) throw std::runtime_error(name_ + ": " + std::to_string(bytes) + " bytes exceed the reservation");
    const size_t need = bytes - mapped_;
    if (budget_ && !budget_->make_room(need))
        throw std::runtime_error(name_ + ": not enough VRAM for " + std::to_string(need >> 20) + " MiB more (" +
                                 budget_->describe() + ")");
    if (!map_pieces(mapped_, bytes)) {
        // the budget agreed but the driver is out of memory: something allocated outside the budget since it was
        // last synced. Sync, reclaim the shortfall from the cache, and try once more.
        if (!budget_ || !reclaim_on_oom_ || budget_->reclaiming_ || !budget_->make_room(need) || !map_pieces(mapped_, bytes))
            throw std::runtime_error(name_ + ": out of VRAM mapping " + std::to_string(need >> 20) + " MiB");
    }
    CUmemAccessDesc acc = {};
    acc.location = prop_for_device().location;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cu_check(cuMemSetAccess(va_ + mapped_, need, &acc, 1), "access");
    cu_check(cuMemsetD8(va_ + mapped_, 0, need), "clear");
    mapped_ = bytes;
    if (budget_) budget_->charge(need);
}

// one physical allocation per granule, so any shrink lands on a piece boundary and keeps what is below it.
// Out of memory: unmaps what this call mapped and returns false (any other failure throws).
bool ElasticBuf::map_pieces(size_t from, size_t to) {
    const size_t g = vmem_granularity();
    CUmemAllocationProp p = prop_for_device();
    const size_t n0 = pieces_.size();
    auto unwind = [&] {
        while (pieces_.size() > n0) {
            const Piece pc = pieces_.back();
            pieces_.pop_back();
            cuMemUnmap(va_ + pc.off, pc.size);
            cuMemRelease(pc.h);
        }
    };
    for (size_t off = from; off < to; off += g) {
        Piece pc{off, g, 0};
        const CUresult r = cuMemCreate(&pc.h, g, &p, 0);
        if (r == CUDA_ERROR_OUT_OF_MEMORY) {
            unwind();
            return false;
        }
        if (r != CUDA_SUCCESS) unwind();
        cu_check(r, "create");
        const CUresult m = cuMemMap(va_ + off, g, 0, pc.h, 0);
        if (m != CUDA_SUCCESS) {
            cuMemRelease(pc.h);
            unwind();
            cu_check(m, "map");
        }
        pieces_.push_back(pc);
    }
    return true;
}

void ElasticBuf::shrink_to(size_t bytes) {
    bytes = vmem_round(bytes);
    if (bytes >= mapped_) return;
    cuCtxSynchronize();  // the GPU may still be reading these pages
    while (!pieces_.empty() && pieces_.back().off >= bytes) {
        const Piece pc = pieces_.back();
        pieces_.pop_back();
        cuMemUnmap(va_ + pc.off, pc.size);
        cuMemRelease(pc.h);
        if (budget_) budget_->refund(pc.size);
    }
    mapped_ = bytes;
}

}  // namespace bnk
