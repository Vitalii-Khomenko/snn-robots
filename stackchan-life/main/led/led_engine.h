#pragma once
#include <algorithm>
#include <cstdint>

// The body's 12x addressable RGB LED chain. Same PY32 IO expander as servo
// power (pin 13 vs pin 0 for "VM EN" - see ../power/power_rails.h), so this
// was confirmed working during bring-up (docs/HARDWARE.md) but was never
// wired into any actual behavior until now.
class LedEngine {
public:
    bool begin();
    void update();  // call periodically to drive the idle breathing effect

    // Static color, applied to all 12 LEDs immediately. Overrides the
    // breathing effect until setBreathing() is called again.
    void setColor(uint8_t r, uint8_t g, uint8_t b);

    // Slow sinusoidal brightness pulse around (r,g,b) - the idle "alive but
    // resting" glow.
    void setBreathing(uint8_t r, uint8_t g, uint8_t b);
    // The glow follows the virtual physiology: breathing rate (breaths per
    // minute) and a small flash on every heartbeat.
    void setRhythm(float breathsPerMin) { breath_period_s_ = 60.0f / std::max(breathsPerMin, 4.0f); }
    void beat() { beat_boost_ = 0.3f; }
    // How lively the glow is (the policy's led_activity, 0..1): a calm robot glows dim and breathes slowly, a lively one
    // glows bright, and above ~0.55 a wave runs round the twelve LEDs, faster the livelier it is.
    void setActivity(float activity01) { activity_ = std::clamp(activity01, 0.0f, 1.0f); }

private:
    bool breathing_ = false;
    uint8_t breath_r_ = 0, breath_g_ = 0, breath_b_ = 0;
    float breath_period_s_ = 4.0f, beat_boost_ = 0.0f;
    // The breathing phase is integrated, not fmod(uptime, period): the period follows the physiology every tick, and
    // fmod of a large uptime by a slightly different period jumps anywhere in the cycle (the glow flickered).
    float phase_ = 0.0f, wave_phase_ = 0.0f;
    float activity_ = 0.4f, activity_s_ = 0.4f;  // target and smoothed (no jumps from one tick's output)
    int64_t last_us_ = 0;
    bool sent_valid_ = false;
    uint16_t sent_[12] = {};  // colours on the LEDs (RGB565); unchanged frames are not sent again
};
