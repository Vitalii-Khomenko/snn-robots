#pragma once
#include <cstdint>
#include <functional>
#include <driver/i2c_master.h>

// LTR-553ALS-WA proximity + ambient-light sensor, next to the camera.
// Listed in the CoreS3 hardware spec but *not* exposed by the BSP at all -
// unlike servos/touch/IMU, there is no known-working sibling-project driver
// to copy from here, so this is a best-effort implementation of the
// standard LTR-5xx family register map, not a proven port. Treat the first
// hardware test as verifying the register map itself, not just the wiring.
//
// Presence is detected as "PS reading rose well above its recent quiet
// baseline", not an absolute threshold - ambient IR reflectivity varies too
// much by environment to hardcode one.
class PresenceEngine {
public:
    bool begin();
    void update();  // poll periodically; fires onPresence() here

    void onPresence(std::function<void()> cb) { on_presence_ = std::move(cb); }
    uint16_t lastRaw() const { return last_raw_; }

    // Ambient light, channel 0 (broad visible+IR), 0..1 normalized against
    // the sensor's 16-bit data register range. begin() already enables ALS
    // active mode (kRegAlsContr) - this just starts reading the data
    // registers the standard LTR-5xx map puts right before the PS ones
    // this class already uses, same "best-effort, not cross-checked
    // against the real datasheet" caveat as the rest of this class. Use
    // this raw absolute value for delta/jump detection (e.g.
    // RobotRuntime's StimEvent::BrightLight check, which compares
    // consecutive readings and doesn't care about absolute scale);
    // for anything that should adapt to "is it dark FOR THIS ROOM", use
    // relativeLightDelta() instead - a room with permanently-drawn
    // curtains never reads as "bright" in absolute terms, so an absolute
    // threshold would think it's always night.
    float lightLevel01() const { return light_level01_; }

    // Current reading minus a slow-moving (multi-hour time constant)
    // baseline - positive when meaningfully brighter than this room's own
    // recent typical level, near zero or negative otherwise. Same EMA
    // technique as the proximity baseline_ below, just much slower (day/
    // night adaptation, not seconds-scale presence detection).
    float relativeLightDelta() const { return light_level01_ - light_baseline_; }

private:
    esp_err_t writeReg(uint8_t reg, uint8_t value);
    esp_err_t readReg(uint8_t reg, uint8_t* value);
    bool readPsData(uint16_t* out);
    bool readAlsData(uint16_t* out);

    static constexpr uint16_t kAddr = 0x23;  // LTR-5xx family default 7-bit address

    i2c_master_dev_handle_t dev_ = nullptr;
    uint16_t last_raw_             = 0;
    float baseline_                = 0.0f;
    bool baseline_valid_            = false;
    bool present_                   = false;
    uint32_t last_trigger_ms_       = 0;
    uint32_t last_poll_ms_          = 0;
    static constexpr uint32_t kPollIntervalMs = 100;
    std::function<void()> on_presence_;
    float light_level01_            = 0.0f;
    float light_baseline_           = -1.0f;  // -1 = "not seeded yet"
    // ~4-hour time constant at the 10 Hz poll rate (kPollIntervalMs) - slow enough to represent "typical for
    // this room" rather than tracking the current moment, fast enough to
    // still adapt across seasons as days get longer/shorter.
    static constexpr float kLightBaselineAlpha = 0.000007f;  // per 100 ms poll (was 0.0000014 per 20 ms: same ~4 h)

    // How far above the slow-moving baseline a reading has to rise to count
    // as "something got close". Wide enough to ignore normal sensor noise.
    static constexpr float kTriggerMargin      = 60.0f;
    static constexpr float kReleaseMargin       = 25.0f;
    static constexpr uint32_t kRetriggerCooldownMs = 3000;
};
