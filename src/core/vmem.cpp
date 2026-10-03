#include "core/vmem.h"

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

bool VramBudget::make_room(size_t bytes) {
    if (free() >= bytes) return true;
    if (reclaim) reclaim(bytes - free());
    return free() >= bytes;
}

ElasticBuf::~ElasticBuf() {
    shrink_to(0);
    if (va_) cuMemAddressFree(va_, reserved_);
}

void ElasticBuf::reserve(size_t max_bytes, VramBudget * budget, const char * name) {
    budget_ = budget;
    name_ = name;
    reserved_ = vmem_round(max_bytes ? max_bytes : 1);
    cu_check(cuMemAddressReserve(&va_, reserved_, 0, 0, 0), "reserve");
}

void ElasticBuf::ensure(size_t bytes) {
    bytes = vmem_round(bytes);
    if (bytes <= mapped_) return;
    if (bytes > reserved_) throw std::runtime_error(name_ + ": " + std::to_string(bytes) + " bytes exceed the reservation");
    const size_t need = bytes - mapped_;
    if (budget_ && !budget_->make_room(need))
        throw std::runtime_error(name_ + ": not enough VRAM for " + std::to_string(need >> 20) + " MiB more");
    // one physical allocation per granule, so any shrink lands on a piece boundary and keeps what is below it
    const size_t g = vmem_granularity();
    CUmemAllocationProp p = prop_for_device();
    CUmemAccessDesc acc = {};
    acc.location = p.location;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    for (size_t off = mapped_; off < bytes; off += g) {
        Piece pc{off, g, 0};
        cu_check(cuMemCreate(&pc.h, g, &p, 0), "create");
        cu_check(cuMemMap(va_ + off, g, 0, pc.h, 0), "map");
        pieces_.push_back(pc);
    }
    cu_check(cuMemSetAccess(va_ + mapped_, need, &acc, 1), "access");
    cu_check(cuMemsetD8(va_ + mapped_, 0, need), "clear");
    mapped_ = bytes;
    if (budget_) budget_->charge(need);
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
