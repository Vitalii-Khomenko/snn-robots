#include "presence_engine.h"

#include <bsp/esp-bsp.h>
#include <esp_log.h>
#include <esp_timer.h>

namespace {
constexpr const char* kTag = "presence";

// LTR-5xx family registers (standard across LTR-506/553/558; not confirmed
// against the actual LTR-553ALS-WA datasheet - see header comment).
constexpr uint8_t kRegAlsContr   = 0x80;
constexpr uint8_t kRegPsContr    = 0x81;
constexpr uint8_t kRegPsLed      = 0x82;
constexpr uint8_t kRegPsNPulses  = 0x83;
constexpr uint8_t kRegPsMeasRate = 0x84;
constexpr uint8_t kRegPartId     = 0x86;
constexpr uint8_t kRegAlsDataCh1_0 = 0x88;  // IR-only channel, low byte - unused here
constexpr uint8_t kRegAlsDataCh1_1 = 0x89;
constexpr uint8_t kRegAlsDataCh0_0 = 0x8A;  // visible+IR channel, low byte
constexpr uint8_t kRegAlsDataCh0_1 = 0x8B;
constexpr uint8_t kRegPsData0    = 0x8D;  // low byte
constexpr uint8_t kRegPsData1    = 0x8E;  // high byte (top bits reserved/status)
}  // namespace

esp_err_t PresenceEngine::writeReg(uint8_t reg, uint8_t value)
{
    uint8_t buf[2] = {reg, value};
    return i2c_master_transmit(dev_, buf, sizeof(buf), 1000);
}

esp_err_t PresenceEngine::readReg(uint8_t reg, uint8_t* value)
{
    return i2c_master_transmit_receive(dev_, &reg, 1, value, 1, 1000);
}

bool PresenceEngine::begin()
{
    i2c_master_bus_handle_t bus = nullptr;
    if (i2c_master_get_bus_handle(BSP_I2C_NUM, &bus) != ESP_OK) {
        ESP_LOGE(kTag, "i2c_master_get_bus_handle failed");
        return false;
    }

    i2c_device_config_t dev_cfg = {};
    dev_cfg.dev_addr_length     = I2C_ADDR_BIT_LEN_7;
    dev_cfg.device_address      = kAddr;
    dev_cfg.scl_speed_hz        = 400000;
    if (i2c_master_bus_add_device(bus, &dev_cfg, &dev_) != ESP_OK) {
        ESP_LOGE(kTag, "i2c_master_bus_add_device(0x%02x) failed", kAddr);
        return false;
    }

    uint8_t part_id = 0;
    if (readReg(kRegPartId, &part_id) != ESP_OK) {
        ESP_LOGE(kTag, "PART_ID read failed - sensor not responding at 0x%02x", kAddr);
        return false;
    }
    // Datasheet-typical LTR-553 PART_ID is 0x92; log whatever we get either
    // way since this hasn't been cross-checked against real hardware yet.
    ESP_LOGI(kTag, "PART_ID=0x%02x (LTR-553 datasheet value is usually 0x92)", part_id);

    writeReg(kRegPsLed, 0x7F);       // 30 kHz pulse, 100% duty, ~100 mA drive
    writeReg(kRegPsNPulses, 0x01);   // 1 pulse per measurement
    writeReg(kRegPsMeasRate, 0x02);  // fast repeat rate
    writeReg(kRegPsContr, 0x03);     // proximity active mode
    writeReg(kRegAlsContr, 0x01);    // ALS active mode too (unused for now, cheap to enable)

    return true;
}

bool PresenceEngine::readPsData(uint16_t* out)
{
    uint8_t lo = 0, hi = 0;
    if (readReg(kRegPsData0, &lo) != ESP_OK) {
        return false;
    }
    if (readReg(kRegPsData1, &hi) != ESP_OK) {
        return false;
    }
    *out = static_cast<uint16_t>(((hi & 0x07) << 8) | lo);  // 11-bit PS value
    return true;
}

bool PresenceEngine::readAlsData(uint16_t* out)
{
    uint8_t lo = 0, hi = 0;
    if (readReg(kRegAlsDataCh0_0, &lo) != ESP_OK) {
        return false;
    }
    if (readReg(kRegAlsDataCh0_1, &hi) != ESP_OK) {
        return false;
    }
    *out = static_cast<uint16_t>((hi << 8) | lo);  // 16-bit ALS channel 0 value
    return true;
}

void PresenceEngine::update()
{
    // 10 Hz is plenty for "someone is at my face" and room light; every
    // 20 ms main-loop pass cost 4 I2C transactions on the bus shared with
    // the IMU, touch pad and power chip.
    const uint32_t poll_ms = static_cast<uint32_t>(esp_timer_get_time() / 1000);
    if (poll_ms - last_poll_ms_ < kPollIntervalMs) return;
    last_poll_ms_ = poll_ms;
    uint16_t als_raw = 0;
    if (readAlsData(&als_raw)) {
        // 16-bit register, so 65535 is the normalization ceiling - not a
        // calibrated lux value, just a relative brightness signal.
        light_level01_ = static_cast<float>(als_raw) / 65535.0f;

        if (light_baseline_ < 0.0f) {
            light_baseline_ = light_level01_;  // seed on first real reading
        } else {
            light_baseline_ += (light_level01_ - light_baseline_) * kLightBaselineAlpha;
        }
    }

    uint16_t raw = 0;
    if (!readPsData(&raw)) {
        return;
    }
    last_raw_ = raw;

    if (!baseline_valid_) {
        baseline_       = raw;
        baseline_valid_ = true;
        return;
    }

    uint32_t now = static_cast<uint32_t>(esp_timer_get_time() / 1000);

    if (!present_ && raw > baseline_ + kTriggerMargin) {
        present_ = true;
        if (now - last_trigger_ms_ >= kRetriggerCooldownMs) {
            last_trigger_ms_ = now;
            if (on_presence_) {
                on_presence_();
            }
        }
        return;  // don't fold a "someone is here" reading into the baseline
    }

    if (present_ && raw < baseline_ + kReleaseMargin) {
        present_ = false;
    }

    if (!present_) {
        // Slow exponential moving average so slow ambient drift (daylight
        // changing, etc.) doesn't get mistaken for presence.
        baseline_ += (static_cast<float>(raw) - baseline_) * 0.1f;  // per 100 ms poll (0.02 per 20 ms before: ~1 s)
    }
}
