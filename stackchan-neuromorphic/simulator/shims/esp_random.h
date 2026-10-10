#pragma once
// Host shim: deterministic PRNG seeded from the scenario.
#include <cstdint>
uint32_t esp_random();
