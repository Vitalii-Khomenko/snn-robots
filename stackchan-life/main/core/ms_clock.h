#pragma once
#include <cstdint>

// Millisecond timestamps are uint32_t (esp_timer_get_time() / 1000) and wrap
// after 49.7 days of uptime. Compare them only through these helpers:
// `now >= deadline` stops firing at the wrap for another 49 days, and
// `now < until` believes a window is open for 49 days.

// A periodic deadline has been reached. Valid while the deadline is kept
// within +-24.8 days of now (every periodic timer here is renewed within
// seconds or minutes).
inline bool msReached(uint32_t now, uint32_t deadline)
{
    return static_cast<int32_t>(now - deadline) >= 0;
}

// A short window ("until") is still open. Safe for windows that were set
// long ago and never renewed: anything farther ahead than maxWindowMs is
// treated as closed.
inline bool msWithin(uint32_t now, uint32_t until, uint32_t maxWindowMs = 600000)
{
    const uint32_t left = until - now;
    return left != 0 && left <= maxWindowMs;
}
