#pragma once
#include <atomic>
#include <cstdarg>
#include <cstdint>

#include "hybrid_spec.h"
#include "motivation.h"
#include "online_learner.h"
#include "world_model.h"
#include "sd_diary.h"

class WifiEngine;

// The robot's diary on the wire (platform B, stage 0: "eyes and ears for us").
//
// Every brain tick (20 Hz) the main loop hands over what the policy saw and
// what it did; four ticks make one UDP datagram to the PC logger
// (tools/life/lifelogd.py). Every ESP_LOG line goes out too, as text: the
// events (touches, learned links, limits) and the 5 s stats line. Nothing is
// sent from the main loop itself - it only copies into a ring buffer in
// PSRAM; a low-priority task on core 0 keeps every datagram on the microSD
// card (SdDiary) and sends it while Wi-Fi is up; what does not fit the ring
// is dropped (counted).
//
// Wire format, little endian (tools/life/lifedata.py reads it):
//   header (16 B): magic "SCL1", kind (1 ticks, 2 log line), n (ticks in this
//     datagram), n_values (int16 values per tick), boot_id, datagram seq
//   kind 1: int64 unix ms of the first tick (0 = clock not synced yet), then
//     n x { uint32 tick seq, uint32 ms since boot, uint16 flags, int16 values[n_values] }
//     values = policy input (100), action (51), efference copy (3), motivation (8: needs, reward, novelty,
//     head movement), world model (4: RMS error of vision, audio, body; learning progress x 10), scaled by kScale
//   kind 2: uint32 ms since boot, then the text of the line (no terminator)
class LifeLog {
public:
    static constexpr int kEmbValues = 64 + 32;  // bus v4: vision and hearing embeddings (x 1/4), 2026-10-17
    static constexpr int kValues =
        hybrid::kPolicyInputs + hybrid::ACT_COUNT + hybrid::EFF_COUNT + Motivation::kLogValues + WorldModel::kLogValues +
        OnlineLearner::kLogValues + kEmbValues + 1;  // + expect_talk
    static constexpr float kScale = 4096.0f;  // int16 = value x 4096: +-8, step 0.00024
    static constexpr int kTicksPerDatagram = 2;  // 4 until the embeddings (2026-10-17)

    // Starts the sender task and tees ESP_LOG into the diary. false if there is no host
    // configured (secrets/life_log_host.h) or no memory - then everything else is a no-op.
    bool begin(const WifiEngine& wifi);

    // One brain tick (main loop thread only).
    void tick(uint32_t tickSeq, uint32_t nowMs, uint16_t flags, const float* values);

    uint32_t sent() const { return sent_.load(); }
    uint32_t dropped() const { return dropped_.load(); }
    // datagrams kept on the microSD card / card failures (sd_diary.h)
    uint32_t sdWritten() const { return sd_written_.load(); }
    uint32_t sdErrors() const { return sd_errors_.load(); }

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
        int16_t v[kValues];
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
    std::atomic<uint32_t> seq_{0}, sent_{0}, dropped_{0}, sd_written_{0}, sd_errors_{0};
    SdDiary sd_;
};
