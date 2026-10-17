#pragma once
// Simulator shim: capability-aware allocation is plain malloc on the PC.
#include <cstddef>
#include <cstdlib>

#define MALLOC_CAP_SPIRAM (1 << 10)
#define MALLOC_CAP_INTERNAL (1 << 11)
#define MALLOC_CAP_8BIT (1 << 2)

inline void* heap_caps_malloc(size_t size, unsigned) { return std::malloc(size); }
inline void heap_caps_free(void* p) { std::free(p); }
