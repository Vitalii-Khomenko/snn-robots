#pragma once
#include <cstddef>
#include <cstdint>

// The microphone for the home PC (docs/CONTINGENCY.md step 2: what the ear calls speech). While a listener is
// connected (OtaServer GET /listen, tools/robot/listen.py) every microphone block goes, mono, into a PSRAM ring with its
// time (ms since boot, the diary's clock) and whether the speaker mute or the own-motion threshold was on; the HTTP
// task sends the ring on. Nothing is kept or sent when nobody listens. One listener at a time.
namespace sound_tap {
inline constexpr uint32_t kRate = 24000;
inline constexpr size_t kFrames = 128;  // per block, as the microphone task reads them (5.3 ms)
inline constexpr uint8_t kMuted = 1, kOwnMotion = 2;
struct Block {
    uint32_t ms;
    uint8_t flags;
    uint8_t pad[3];
    int16_t s[kFrames];  // mono: (left + right) / 2
};

// microphone task: never blocks; a full ring drops the block (counted)
void feed(const int16_t* lr, size_t frames, uint32_t ms, bool muted, bool ownMotion);
// HTTP task
bool start();                                       // false: no memory, or someone already listens
size_t take(void* out, size_t max, uint32_t waitMs);  // one block (sizeof(Block)) or 0
void stop();
uint32_t dropped();
}  // namespace sound_tap
