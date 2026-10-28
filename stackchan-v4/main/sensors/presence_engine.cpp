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
constexpr uint8_t kRegAlsDataCh1_0 = 0x88;  // first of 4 ALS data bytes: CH1 (IR) lo/hi, CH0 (visible+IR) lo/hi
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

    // PS stayed 0..8 of 2047 with a hand at the face on 0x7F (60 kHz) and 1 pulse (2026-10-24): 40 kHz as in
    // M5Stack's CoreS3 example, 8 pulses per measurement (signal ~ pulses). Read back once, logged by update().
    writeReg(kRegPsLed, 0x3F);       // 40 kHz pulse, 100% duty, 100 mA drive
    writeReg(kRegPsNPulses, 0x08);   // 8 pulses per measurement
    writeReg(kRegPsMeasRate, 0x02);  // fast repeat rate
    writeReg(kRegPsContr, 0x03);     // proximity active mode
    writeReg(kRegAlsContr, 0x0D);    // ALS active mode, gain 8x (the room is often dark; brain v4 light channel)

    readReg(kRegPsLed, &cfg_[0]);
    readReg(kRegPsNPulses, &cfg_[1]);
    readReg(kRegPsMeasRate, &cfg_[2]);
    readReg(kRegPsContr, &cfg_[3]);
    return true;
}

bool PresenceEngine::readPsData(uint16_t* out)
{
    // one burst like the ALS bytes (read separately, PS stayed 0 even with a hand on the robot, 2026-10-24)
    uint8_t reg = kRegPsData0, d[2] = {0, 0};
    if (i2c_master_transmit_receive(dev_, &reg, 1, d, sizeof(d), 1000) != ESP_OK) {
        return false;
    }
    ps_status_ = d[1];
    *out = static_cast<uint16_t>(((d[1] & 0x07) << 8) | d[0]);  // 11-bit PS value (bit 7 of the high byte: saturated)
    return true;
}

bool PresenceEngine::readAlsData(uint16_t* out)
{
    // The four ALS data bytes are latched together when CH1's low byte (0x88) is read first: read them as one
    // burst. Reading channel 0 alone (0x8A, 0x8B) gave exactly 0 all the time (2026-10-24, brain v4 light channel).
    uint8_t reg = kRegAlsDataCh1_0, d[4] = {0, 0, 0, 0};
    if (i2c_master_transmit_receive(dev_, &reg, 1, d, sizeof(d), 1000) != ESP_OK) {
        return false;
    }
    als_ch1_ = static_cast<uint16_t>((d[1] << 8) | d[0]);
    *out = static_cast<uint16_t>((d[3] << 8) | d[2]);  // 16-bit ALS channel 0 value (visible + IR)
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
    if (!cfg_logged_ && poll_ms > 30000) {  // after the boot log, so it reaches the diary
        cfg_logged_ = true;
        ESP_LOGI(kTag, "PS config read back: led=0x%02x pulses=0x%02x rate=0x%02x contr=0x%02x", cfg_[0], cfg_[1],
                 cfg_[2], cfg_[3]);
    }
    uint16_t als_raw = 0;
    if (readAlsData(&als_raw)) {
        // 16-bit register, so 65535 is the normalization ceiling - not a
        // calibrated lux value, just a relative brightness signal.
        als_raw_ = als_raw;
        als_ok_ms_ = poll_ms ? poll_ms : 1;
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
    ps_ok_ms_ = poll_ms ? poll_ms : 1;

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
