#pragma once
#include <functional>
#include <Si12T.h>

// Head touch (Si12T capacitive sensor, 3 channels) - not the LCD
// touchscreen. Simple debounced press/release, no swipe-gesture
// classification yet.
class TouchEngine {
public:
    bool begin();
    // the pad's sensitivity (Si12T level 0..7, 3 until 2026-10-17); set at run time from the update server, kept in NVS
    bool setSensitivity(int level);
    int sensitivity() const { return level_; }
    void update();  // poll periodically

    void onTouch(std::function<void()> cb) { on_touch_ = std::move(cb); }
    void onRelease(std::function<void()> cb) { on_release_ = std::move(cb); }
    bool isTouched() const { return touched_; }
    // the pad reports contact in this very sample (no release delay): the servo-phantom check needs it
    bool contactNow() const { return contact_; }

private:
    si12t_handle_t handle_ = nullptr;
    bool touched_            = false;
    uint8_t touched_samples_  = 0;
    uint8_t released_samples_ = 0;
    std::function<void()> on_touch_;
    std::function<void()> on_release_;

    static constexpr uint8_t kThreshold      = 2;
public:
    uint8_t lastLevel() const { return last_level_; }
    // strongest level (2..3) of the current / last press, for the log
    uint8_t pressLevel() const { return press_level_; }
    // the three pads' strongest levels in the current / last press, and how long it lasted (ms): what a phantom
    // looks like next to a finger (2026-10-13: tens of touches an hour with nobody home)
    void lastPress(uint8_t levels[3], uint32_t& ms) const
    {
        for (int i = 0; i < 3; ++i) levels[i] = press_levels_[i];
        ms = press_ms_;
    }
private:
    uint8_t last_level_ = 0;
    static constexpr uint8_t kDebounceSamples = 2;
    // A finger that stays on the pad drops out of the sensor for a moment now and then (seen 2026-10-04: one held touch
    // came out as a new touch every ~0.9 s, the brain took it for a burst and grumbled). The press ends only after
    // ~0.4 s without contact (20 main-loop samples); separate strokes are seconds apart.
    static constexpr uint8_t kReleaseSamples = 20;
    uint8_t press_level_ = 0;
    int level_ = 3;
    uint8_t press_levels_[3] = {0, 0, 0};
    uint32_t press_start_ms_ = 0, press_ms_ = 0;
    bool contact_ = false;
};
