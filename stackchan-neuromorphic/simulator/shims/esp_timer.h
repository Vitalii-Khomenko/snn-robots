#pragma once
// Host shim: esp_timer_get_time() returns the simulator's virtual clock (us).
#include <cstdint>
int64_t esp_timer_get_time();
