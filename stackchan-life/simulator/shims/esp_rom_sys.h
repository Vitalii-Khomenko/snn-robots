#pragma once
// Host shim: busy-wait delays are not needed in the simulator.
#include <cstdint>
inline void esp_rom_delay_us(uint32_t) {}
