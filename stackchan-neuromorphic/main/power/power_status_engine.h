#pragma once
#include <cstdint>
#include <driver/i2c_master.h>

// AXP2101 charge/power status, read over the same I2C bus
// power_rails.cpp's enableAxp2101PowerRails() already configures.
//
// charging(): register 0x01, bits [6:5] (battery current direction) == 1.
// Corroborated by two independent pieces of code in this project's
// dependency chain - upstream/m5stack-StackChan/firmware/xiaozhi-esp32's
// Axp2101::IsCharging() and upstream/m5stack-StackChan/firmware/main/hal/
// board/stackchan.cc's Pmic class - so reasonably trustworthy despite no
// datasheet being available locally to cite directly.
//
// externallyPowered(): there is no real VBUS-present bit read anywhere in
// this codebase's dependency chain, and no AXP2101 datasheet available
// locally to find one - only the same register 0x01 current-direction/
// charge-done bits. This is the exact heuristic
// stackchan.cc's IsExternalPowerConnected() already uses ("not actively
// discharging" -> assume externally powered), ported as-is rather than
// invented. It cannot distinguish "USB cable present but not charging"
// from "battery simply full while still plugged in", and can't tell USB
// power apart from any other external power source - it is NOT true
// USB-connection detection, just the best signal available without the
// real datasheet.
class PowerStatusEngine {
public:
    bool begin();
    void update();  // poll periodically

    bool charging() const { return charging_; }
    bool externallyPowered() const { return externally_powered_; }
    // AXP2101's battery-capacity register is the same one used by the
    // upstream CoreS3 board support.  1.0 is the neutral/fresh-boot value
    // until the first successful poll.
    float batteryLevel01() const { return battery_level01_; }

private:
    static constexpr uint16_t kAxp2101Addr = 0x34;
    // Charge state barely changes tick to tick - polling the real I2C
    // register at the ~20ms main-loop rate would just be chatty on a bus
    // shared with other sensors for no benefit. update() is cheap to call
    // every tick; the actual I2C read only happens this often.
    static constexpr uint32_t kPollIntervalMs = 1000;

    i2c_master_dev_handle_t dev_ = nullptr;
    bool charging_               = false;
    bool externally_powered_     = false;
    float battery_level01_       = 1.0f;
    uint32_t next_poll_ms_       = 0;
};
