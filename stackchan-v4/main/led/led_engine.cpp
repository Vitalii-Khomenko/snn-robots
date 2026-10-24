#include "led_engine.h"

#include <algorithm>
#include <cmath>
#include <esp_log.h>
#include <esp_timer.h>

#include "../power/power_rails.h"

namespace {
constexpr const char* kTag = "led";
constexpr uint8_t kLedCount = 12;
constexpr float kPi          = 3.14159265358979323846f;
}  // namespace

bool LedEngine::begin()
{
    m5::PY32IOExpander_Class* io_expander = power::getBodyIoExpander();
    if (!io_expander) {
        ESP_LOGE(kTag, "PY32 expander unavailable");
        return false;
    }

    io_expander->setDirection(13, true);   // output
    io_expander->setPullMode(13, true);    // pull-up
    io_expander->setDriveMode(13, false);  // push-pull
    io_expander->setLedCount(kLedCount);
    return true;
}

void LedEngine::setColor(uint8_t r, uint8_t g, uint8_t b)
{
    breathing_ = false;
    m5::PY32IOExpander_Class* io_expander = power::getBodyIoExpander();
    if (!io_expander) {
        return;
    }
    for (uint8_t i = 0; i < kLedCount; ++i) {
        io_expander->setLedColor(i, r, g, b);
    }
    io_expander->refreshLeds();
    sent_valid_ = false;  // the next breathing frame is sent in full
}

void LedEngine::setBreathing(uint8_t r, uint8_t g, uint8_t b)
{
    breathing_ = true;
    pattern_ = -1;
    breath_r_  = r;
    breath_g_  = g;
    breath_b_  = b;
}

void LedEngine::update()
{
    if (!breathing_) {
        return;
    }
    // 25 frames a second is smooth for a breath of several seconds; every frame is one I2C burst on the bus the
    // IMU, touch pad, light sensor and power chip share (it used to be 14 transfers at 50 Hz).
    const int64_t now_us = esp_timer_get_time();
    if (last_us_ != 0 && now_us - last_us_ < 40000) {
        return;
    }
    const float dt = last_us_ != 0 ? std::min(0.5f, static_cast<float>(now_us - last_us_) * 1e-6f) : 0.0f;
    last_us_ = now_us;
    m5::PY32IOExpander_Class* io_expander = power::getBodyIoExpander();
    if (!io_expander) {
        return;
    }

    phase_ += dt / breath_period_s_;
    phase_ -= std::floor(phase_);
    activity_s_ += (activity_ - activity_s_) * std::min(1.0f, dt / 0.6f);
    const float act = activity_s_;
    // brightness: 0.15..1.0 breathing, scaled by the activity (a calm robot glows at ~45 %), plus the heartbeat flash
    const float breath = 0.15f + 0.85f * (0.5f - 0.5f * std::cos(2.0f * kPi * phase_));
    const float level = 0.35f + 0.65f * act;
    beat_boost_ *= std::pow(0.8f, dt / 0.02f);  // same fade as at the old 50 Hz
    // the running wave: 0.2 .. 1.5 turns a second, mixed in from activity 0.55 to 0.85
    const float wave_mix = std::clamp((act - 0.55f) / 0.3f, 0.0f, 1.0f);
    wave_phase_ += dt * (0.2f + 1.3f * wave_mix);
    wave_phase_ -= std::floor(wave_phase_);

    uint16_t colours[kLedCount];
    if (pattern_ >= 0) {
        // brain v4: colour and brightness glide over ~0.3 s, the pattern runs at the network's rate
        const float k = std::min(1.0f, dt / 0.3f);
        for (int c = 0; c < 3; ++c) colour_[c] += (target_[c] - colour_[c]) * k;
        pat_bright_s_ += (pat_bright_ - pat_bright_s_) * k;
        const float rate = pat_rate_;
        float hz = 0.25f;
        switch (pattern_) {
            case 1: hz = 0.15f + 0.35f * rate; break;  // breathe: 6.7 .. 2 s
            case 2: hz = 0.6f + 1.2f * rate; break;    // pulse: a heartbeat, 1.7 .. 0.55 s
            case 4: hz = 0.25f + 1.25f * rate; break;  // wave: turns a second
            default: break;
        }
        pat_phase_ += dt * hz;
        pat_phase_ -= std::floor(pat_phase_);
        for (uint8_t i = 0; i < kLedCount; ++i) {
            float br = 1.0f;
            switch (pattern_) {
                case 1: br = 0.25f + 0.75f * (0.5f - 0.5f * std::cos(2.0f * kPi * pat_phase_)); break;
                case 2: {  // two quick beats, then rest
                    const float p = pat_phase_;
                    br = 0.3f + 0.7f * std::max(std::exp(-p * p / 0.004f), 0.7f * std::exp(-(p - 0.22f) * (p - 0.22f) / 0.004f));
                    break;
                }
                case 3:  // flicker: each LED its own random level every frame, livelier at a higher rate
                    flicker_rng_ = flicker_rng_ * 1664525u + 1013904223u;
                    br = 1.0f - (0.3f + 0.6f * rate) * static_cast<float>(flicker_rng_ >> 24) / 255.0f;
                    break;
                case 4: br = 0.2f + 0.8f * (0.5f + 0.5f * std::cos(2.0f * kPi * (static_cast<float>(i) / kLedCount - pat_phase_))); break;
                default: break;
            }
            br *= pat_bright_s_;
            const uint8_t r = static_cast<uint8_t>(std::clamp(colour_[0] * br, 0.0f, 255.0f));
            const uint8_t g = static_cast<uint8_t>(std::clamp(colour_[1] * br, 0.0f, 255.0f));
            const uint8_t b = static_cast<uint8_t>(std::clamp(colour_[2] * br, 0.0f, 255.0f));
            colours[i] = static_cast<uint16_t>(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
        }
    }
    for (uint8_t i = 0; i < kLedCount && pattern_ < 0; ++i) {
        const float wave = 0.5f + 0.5f * std::cos(2.0f * kPi * (static_cast<float>(i) / kLedCount - wave_phase_));
        float br = level * (breath * (1.0f - wave_mix) + (0.25f + 0.75f * wave) * wave_mix);
        br = std::min(1.0f, br + beat_boost_);
        const uint8_t r = static_cast<uint8_t>(breath_r_ * br);
        const uint8_t g = static_cast<uint8_t>(breath_g_ * br);
        const uint8_t b = static_cast<uint8_t>(breath_b_ * br);
        // RGB565 like PY32IOExpander_Class::setLedColor(), low byte first
        colours[i] = static_cast<uint16_t>(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
    }
    if (sent_valid_ && std::equal(colours, colours + kLedCount, sent_)) {
        return;
    }
    std::copy(colours, colours + kLedCount, sent_);
    sent_valid_ = true;
    uint8_t data[kLedCount * 2];
    for (uint8_t i = 0; i < kLedCount; ++i) {
        data[2 * i] = static_cast<uint8_t>(colours[i] & 0xFF);
        data[2 * i + 1] = static_cast<uint8_t>(colours[i] >> 8);
    }
    io_expander->setLedData(data, sizeof(data));
    io_expander->refreshLeds();
}
