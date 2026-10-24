#include "power_status_engine.h"
#include "../core/ms_clock.h"

#include <bsp/esp-bsp.h>
#include <esp_log.h>
#include <esp_timer.h>

#include <algorithm>

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
    // Why did the board go off last time? The AXP2101 keeps its power-on (0x20) and power-off (0x21) sources across
    // the off state (XPowersLib: getPowerOnSource / getPowerOffSource). 0x21 bits: 0 PWRON key pulled low (off level),
    // 1 software (reg 0x10), 2 PWRON key off level, 3 VSYS under voltage, 4 VBUS over voltage, 5 DCDC under voltage,
    // 6 DCDC over voltage, 7 die over temperature. Asked for after the black-screen power-offs of 2026-10-22/23.
    uint8_t st[2] = {}, on = 0, off = 0;
    readReg(0x00, st, 2);
    readReg(0x20, &on, 1);
    readReg(0x21, &off, 1);
    ESP_LOGW(kTag, "pmu at boot: status 0x%02x 0x%02x, power-on source 0x%02x, last power-off source 0x%02x%s%s%s%s%s%s%s%s",
             st[0], st[1], on, off, off & 0x01 ? " [key low]" : "", off & 0x02 ? " [SOFTWARE]" : "",
             off & 0x04 ? " [key off level]" : "", off & 0x08 ? " [VSYS UNDER VOLTAGE]" : "",
             off & 0x10 ? " [VBUS over voltage]" : "", off & 0x20 ? " [DCDC under voltage]" : "",
             off & 0x40 ? " [DCDC over voltage]" : "", off & 0x80 ? " [OVER TEMPERATURE]" : "");
    return true;
}

bool PowerStatusEngine::readReg(uint8_t reg, uint8_t* out, size_t n)
{
    return dev_ && i2c_master_transmit_receive(dev_, &reg, 1, out, n, 50) == ESP_OK;
}

PowerStatusEngine::Rails PowerStatusEngine::takeRails()
{
    Rails r{vbus_min_ == 0xFFFF ? uint16_t{0} : vbus_min_, vsys_min_ == 0xFFFF ? uint16_t{0} : vsys_min_, vbat_, die_c_,
            status_};
    vbus_min_ = vsys_min_ = 0xFFFF;
    return r;
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

    // the rails (14-bit ADC values in mV; die temperature as in XPowersLib: 22 + (7274 - raw) / 20)
    uint8_t adc[10] = {};
    if (readReg(0x34, adc, 10)) {
        auto mv = [&](int i) { return static_cast<uint16_t>(((adc[i] & 0x3F) << 8) | adc[i + 1]); };
        vbat_ = mv(0);
        // (no VBUS at all, on the battery, is not a sag)
        sag_s_ = (status_ & 0x20) && mv(4) > 3000 && mv(4) < 4600 ? sag_s_ + 1 : 0;
        vbus_min_ = std::min(vbus_min_, mv(4));
        vsys_min_ = std::min(vsys_min_, mv(6));
        die_c_ = 22.0f + (7274.0f - static_cast<float>(mv(8))) / 20.0f;
    }
    uint8_t st = 0;
    if (readReg(0x00, &st, 1)) {
        if (status_read_ && ((st ^ status_) & 0x28))  // VBUS good or battery present changed
            ESP_LOGW(kTag, "pmu status 0x%02x -> 0x%02x (VBUS good %d, battery present %d), vbat %u mV", status_, st,
                     (st >> 5) & 1, (st >> 3) & 1, vbat_);
        status_ = st;
        status_read_ = true;
    }
}
