#pragma once
#include <cstdint>
#include <cstdio>

extern FILE* g_voice_dump;
namespace sim {
void seedPlatformRandom(uint32_t seed);

// Last LED command (LedEngine is recorded, not driven).
struct LedState {
    uint8_t r = 0, g = 0, b = 0;
    bool breathing = false;
};
extern LedState g_led;
}  // namespace sim
