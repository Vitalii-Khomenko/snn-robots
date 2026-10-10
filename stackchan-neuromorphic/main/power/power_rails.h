#pragma once
#include <cstdint>
#include <driver/i2c_master.h>
#include <PY32IOExpander_Class.hpp>

// Shared I2C power/enable plumbing for the whole robot. All of this was
// found the hard way during hardware bring-up (see docs/HARDWARE.md) - the
// generic CoreS3 BSP only turns on what it needs for the display, and the
// StackChan body's own add-ons (servos, LEDs) sit behind a second, separate
// I2C GPIO expander the BSP has no idea exists.
namespace power {

// Add a raw I2C device handle for `addr` on the shared bus. Caller owns the
// handle and must call i2c_master_bus_rm_device() when done with it.
i2c_master_dev_handle_t addI2cDevice(uint16_t addr);

// Runs the known-working AXP2101 rail-enable sequence
// (upstream/m5stack-StackChan/firmware/main/hal/board/stackchan.cc's `Pmic`
// constructor) that the generic BSP never runs on its own.
void enableAxp2101PowerRails();

// Raises AXP2101 ALDO1 (the AW88298 speaker amp's PA PVDD rail) from the
// BSP's default 1.8V to 3.3V. Must run *after* bsp_audio_codec_speaker_init()
// or whatever else last called bsp_enable_feature(BSP_FEATURE_SPEAKER),
// since that call is what drops it back to 1.8V.
void boostSpeakerPaVoltage();

// Lazily-constructed singleton for the StackChan body's PY32 IO expander
// (I2C addr 0x6F): pin 0 gates servo motor power ("VM EN"), pin 13 drives
// the 12x RGB LED chain. Returns nullptr if the chip can't be reached.
m5::PY32IOExpander_Class* getBodyIoExpander();

// Resets the servo rail through pin 0 ("VM EN") and turns it back on so the
// SCS servos start in the same clean state after a firmware flash as after a
// full power cycle. Safe to call more than once.
bool enableServoPower();

// Diagnostics for the servo watchdog: the PY32 expander's firmware version
// (read over I2C; -1 = no answer, a different value after a stall = the
// chip reset), its VM EN output latch and the pin's actual level (1 = on).
void readServoPowerState(int& version, int& vmOut, int& vmPin);

}  // namespace power
