#pragma once
#include <atomic>
#include <cstdarg>
#include <cstdint>

#include "v4_spec.h"

class WifiEngine;

// The robot's diary on the wire: statistics, analysis and debugging of brain v4.
//
// Every brain tick (20 Hz) the main loop hands over the tick's values (v4::kDiaryValues, tools/v4/spec_v4.py DIARY:
// the raw sense readings, the 13 + 2 channels, the analyzer's 32 numbers, the main network's 61 outputs, what the body
// did); four ticks make one UDP datagram to the PC logger (tools/robot/lifelogd.py). Every ESP_LOG line goes out too,
// as text: the events (touches, faces, calls, gestures) and the 5 s stats lines. Nothing is sent from the main loop
// itself - it only copies into a ring buffer in PSRAM; a low-priority task on core 0 sends it while Wi-Fi is up; what
// does not fit the ring is dropped (counted).
//
// Wire format, little endian (tools/v4/diary.py reads it):
//   header (16 B): magic "SCL1", kind, n (ticks in this datagram), n_values (values per tick), boot_id, datagram seq
//   kind 3 (brain v4 ticks): int64 unix ms of the first tick (0 = clock not synced yet), then
//     n x { uint32 tick seq, uint32 ms since boot, uint16 flags, float16 values[n_values] }
//     flags: bit 0 brain v4 drives the body, bit 1 night rules (silent, dim)
//   kind 2 (a log line): uint32 ms since boot, then the text of the line (no terminator)
//   (kind 1 was platform B's int16 tick layout, until 2026-10-24: tools/v4/legacy_diary.py)
class LifeLog {
public:
    static constexpr int kValues = v4::kDiaryValues;
    static constexpr int kTicksPerDatagram = 4;

    // Starts the sender task and tees ESP_LOG into the diary. false if there is no host
    // configured (secrets/life_log_host.h) or no memory - then everything else is a no-op.
    bool begin(const WifiEngine& wifi);

    // One brain tick (main loop thread only).
    void tick(uint32_t tickSeq, uint32_t nowMs, uint16_t flags, const float* values);

    uint32_t sent() const { return sent_.load(); }
    uint32_t dropped() const { return dropped_.load(); }

private:
    static int teeLog(const char* fmt, va_list args);
    static void taskEntry(void* arg);
    void taskLoop();
    void push(const void* data, size_t size);

    struct __attribute__((packed)) Header {
        char magic[4];
        uint8_t kind, n;
        uint16_t nValues;
        uint32_t bootId, seq;
    };
    struct __attribute__((packed)) TickRec {
        uint32_t seq, ms;
        uint16_t flags;
        uint16_t v[kValues];  // IEEE 754 half precision
    };
    struct __attribute__((packed)) TickDatagram {
        Header h;
        int64_t unixMs;
        TickRec t[kTicksPerDatagram];
    };
    static_assert(sizeof(TickDatagram) <= 1472, "one tick datagram must fit an unfragmented UDP packet (MTU 1500)");

    void* ring_ = nullptr;  // RingbufHandle_t
    const WifiEngine* wifi_ = nullptr;
    int sock_ = -1;
    TickDatagram pending_{};
    int pending_n_ = 0;
    uint32_t boot_id_ = 0;
    std::atomic<uint32_t> seq_{0}, sent_{0}, dropped_{0};
};
