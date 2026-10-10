#include "touch_engine.h"

#include <algorithm>
#include <bsp/esp-bsp.h>
#include <driver/i2c_master.h>
#include <esp_log.h>

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
    si12t_setup(handle_, SI12T_TYPE_LOW, SI12T_SENSITIVITY_LEVEL_3);
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

    if (touched_ && now_touched) press_level_ = std::max(press_level_, max_level);
    if (!touched_ && touched_samples_ >= kDebounceSamples) {
        touched_ = true;
        press_level_ = max_level;
        if (on_touch_) {
            on_touch_();
        }
    } else if (touched_ && released_samples_ >= kReleaseSamples) {
        touched_ = false;
        if (on_release_) {
            on_release_();
        }
    }
}
