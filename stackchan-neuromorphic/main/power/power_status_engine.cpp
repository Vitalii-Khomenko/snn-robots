#include "power_status_engine.h"
#include "../core/ms_clock.h"

#include <bsp/esp-bsp.h>
#include <esp_log.h>
#include <esp_timer.h>

namespace {
constexpr const char* kTag = "power_status";
}

bool PowerStatusEngine::begin()
{
    i2c_master_bus_handle_t bus = nullptr;
    if (i2c_master_get_bus_handle(BSP_I2C_NUM, &bus) != ESP_OK) {
        ESP_LOGE(kTag, "i2c_master_get_bus_handle failed");
        return false;
    }

    i2c_device_config_t dev_cfg = {};
    dev_cfg.dev_addr_length     = I2C_ADDR_BIT_LEN_7;
    dev_cfg.device_address      = kAxp2101Addr;
    dev_cfg.scl_speed_hz        = 400000;
    if (i2c_master_bus_add_device(bus, &dev_cfg, &dev_) != ESP_OK) {
        ESP_LOGE(kTag, "i2c_master_bus_add_device(0x%02x) failed", kAxp2101Addr);
        return false;
    }
    return true;
}

void PowerStatusEngine::update()
{
    if (!dev_) {
        return;
    }
    uint32_t now = static_cast<uint32_t>(esp_timer_get_time() / 1000);
    if (!msReached(now, next_poll_ms_)) {
        return;
    }
    next_poll_ms_ = now + kPollIntervalMs;

    uint8_t reg   = 0x01;
    uint8_t value = 0;
    if (i2c_master_transmit_receive(dev_, &reg, 1, &value, 1, 1000) != ESP_OK) {
        return;
    }

    uint8_t current_direction = (value & 0b01100000) >> 5;  // 1 = charging, 2 = discharging
    bool charging_done          = (value & 0b00000111) == 0b00000100;

    charging_           = (current_direction == 1);
    externally_powered_ = (current_direction != 2) || charging_done;

    uint8_t battery_reg = 0xA4;
    uint8_t battery = 0;
    if (i2c_master_transmit_receive(dev_, &battery_reg, 1, &battery, 1, 1000) == ESP_OK) {
        battery_level01_ = static_cast<float>(battery) / 100.0f;
    }
}
