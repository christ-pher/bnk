// Page-locked host buffers that pin fast: cudaHostAlloc pins 4 KiB pages one by one (~1.2 GB/s here, 1.7 s for
// 2 GiB, 0.7 s to free), and a conversation parked to RAM waited for that inside the request. Anonymous memory on
// transparent huge pages, faulted in by several threads and then registered, pins 2 GiB in ~0.1 s.
#pragma once

#include <cstddef>

namespace bnk {

// `bytes` of page-locked host RAM (portable to every device), or nullptr when RAM is short or pinning fails
void * pin_host(size_t bytes);
// frees a pin_host buffer of that size
void unpin_host(void * p, size_t bytes);

}  // namespace bnk
