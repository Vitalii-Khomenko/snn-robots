#include "touch_engine.h"

#include <algorithm>
#include <bsp/esp-bsp.h>
#include <driver/i2c_master.h>
#include <esp_log.h>
#include <esp_timer.h>

namespace {
constexpr const char* kTag = "touch";
}

bool TouchEngine::begin()
{
    i2c_master_bus_handle_t bus = nullptr;
    if (i2c_master_get_bus_handle(BSP_I2C_NUM, &bus) != ESP_OK) {
        ESP_LOGE(kTag, "i2c_master_get_bus_handle failed");
        return false;
    }

    si12t_config_t cfg = {};
    cfg.i2c_bus          = bus;
    cfg.dev_addr         = SI12T_GND_ADDRESS;

    esp_err_t err = si12t_init(&cfg, &handle_);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "si12t_init failed: %s", esp_err_to_name(err));
        return false;
    }
    si12t_setup(handle_, SI12T_TYPE_LOW, static_cast<si12t_sensitivity_level_t>(level_));
    return true;
}

void TouchEngine::update()
{
    if (!handle_) {
        return;
    }

    uint8_t raw = 0;
    if (si12t_read_touch_result(handle_, &raw) != ESP_OK) {
        return;
    }
    uint8_t levels[3] = {0};
    si12t_parse_touch_result_to(raw, levels);
    uint8_t max_level = std::max({levels[0], levels[1], levels[2]});
    bool now_touched   = max_level >= kThreshold;
    contact_ = now_touched;
    if (now_touched) last_level_ = max_level;

    if (now_touched) {
        touched_samples_  = std::min<int>(touched_samples_ + 1, 10);
        released_samples_ = 0;
    } else {
        released_samples_ = std::min<int>(released_samples_ + 1, 100);
        touched_samples_  = 0;
    }

    const uint32_t now_ms = static_cast<uint32_t>(esp_timer_get_time() / 1000);
    // A pad that reports contact for 20 s is stuck, not held (58 s on 2026-10-13, right after a servo power cycle):
    // set the sensor up again and end the press.
    if (touched_ && now_touched && now_ms - press_start_ms_ > 20000) {
        ESP_LOGW(kTag, "pad stuck for 20 s (levels %u/%u/%u): sensor set up again", static_cast<unsigned>(levels[0]),
                 static_cast<unsigned>(levels[1]), static_cast<unsigned>(levels[2]));
        si12t_setup(handle_, SI12T_TYPE_LOW, static_cast<si12t_sensitivity_level_t>(level_));
        touched_ = false;
        touched_samples_ = 0;
        released_samples_ = kReleaseSamples;
        press_start_ms_ = now_ms;
        if (on_release_) on_release_();
        return;
    }
    if (touched_ && now_touched) {
        press_level_ = std::max(press_level_, max_level);
        for (int i = 0; i < 3; ++i) press_levels_[i] = std::max(press_levels_[i], levels[i]);
        press_ms_ = now_ms - press_start_ms_;
    }
    if (!touched_ && touched_samples_ >= kDebounceSamples) {
        touched_ = true;
        press_level_ = max_level;
        for (int i = 0; i < 3; ++i) press_levels_[i] = levels[i];
        press_start_ms_ = now_ms;
        press_ms_ = 0;
        if (on_touch_) {
            on_touch_();
        }
    } else if (touched_ && released_samples_ >= kReleaseSamples) {
        touched_ = false;
        ESP_LOGI(kTag, "press ended: %u ms, pads %u/%u/%u", static_cast<unsigned>(press_ms_),
                 static_cast<unsigned>(press_levels_[0]), static_cast<unsigned>(press_levels_[1]),
                 static_cast<unsigned>(press_levels_[2]));
        if (on_release_) {
            on_release_();
        }
    }
}

bool TouchEngine::setSensitivity(int level)
{
    if (level < 0 || level > 7) return false;
    level_ = level;
    if (!handle_) return true;  // applied in begin()
    const bool ok = si12t_setup(handle_, SI12T_TYPE_LOW, static_cast<si12t_sensitivity_level_t>(level_)) == ESP_OK;
    ESP_LOGI(kTag, "pad sensitivity level %d%s", level_, ok ? "" : " - FAILED");
    return ok;
}
