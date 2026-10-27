#include "sound_tap.h"

#include <atomic>
#include <cstring>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/ringbuf.h>

namespace sound_tap {
namespace {
constexpr size_t kRingBytes = 96 * 1024;  // ~1.8 s of blocks: rides out a Wi-Fi stall
RingbufHandle_t s_ring = nullptr;
std::atomic<bool> s_on{false};
std::atomic<uint32_t> s_dropped{0};
}  // namespace

void feed(const int16_t* lr, size_t frames, uint32_t ms, bool muted, bool ownMotion)
{
    if (!s_on.load(std::memory_order_acquire)) return;
    Block b{};
    b.ms = ms;
    b.flags = static_cast<uint8_t>((muted ? kMuted : 0) | (ownMotion ? kOwnMotion : 0));
    const size_t n = frames < kFrames ? frames : kFrames;
    for (size_t i = 0; i < n; ++i)
        b.s[i] = static_cast<int16_t>((static_cast<int32_t>(lr[2 * i]) + static_cast<int32_t>(lr[2 * i + 1])) / 2);
    if (xRingbufferSend(s_ring, &b, sizeof(b), 0) != pdTRUE) s_dropped.fetch_add(1);
}

bool start()
{
    if (s_on.load()) return false;
    if (!s_ring) s_ring = xRingbufferCreateWithCaps(kRingBytes, RINGBUF_TYPE_NOSPLIT, MALLOC_CAP_SPIRAM);
    if (!s_ring) return false;
    size_t size = 0;
    while (void* item = xRingbufferReceive(s_ring, &size, 0)) vRingbufferReturnItem(s_ring, item);  // a late block
    s_dropped.store(0);
    s_on.store(true, std::memory_order_release);
    return true;
}

size_t take(void* out, size_t max, uint32_t waitMs)
{
    size_t size = 0;
    void* item = xRingbufferReceive(s_ring, &size, pdMS_TO_TICKS(waitMs));
    if (!item) return 0;
    const size_t n = size < max ? size : max;
    std::memcpy(out, item, n);
    vRingbufferReturnItem(s_ring, item);
    return n;
}

void stop()
{
    s_on.store(false, std::memory_order_release);
    size_t size = 0;
    while (void* item = xRingbufferReceive(s_ring, &size, 0)) vRingbufferReturnItem(s_ring, item);
}

uint32_t dropped() { return s_dropped.load(); }
}  // namespace sound_tap
