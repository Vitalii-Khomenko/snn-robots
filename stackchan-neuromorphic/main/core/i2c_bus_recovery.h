#pragma once

// Bit-bangs the shared I2C bus (BSP_I2C_SCL/SDA - touch controller, PY32 IO
// expander, LTR-553, AXP2101, camera SCCB all share it) free of a stuck
// slave *before* anything claims those pins as an I2C peripheral. Call this
// as the very first thing in RobotRuntime::begin(), ahead of
// bsp_display_start() (the first real I2C user).
//
// Why this exists: `idf.py flash`'s hard_reset only pulses the ESP32's own
// EN pin - the rest of the board (and every I2C slave on it) stays
// continuously powered across it, unlike a real power-off/on cycle. If the
// previous firmware's reset landed mid I2C transaction, a slave can be left
// holding SDA low waiting for more clock pulses it'll now never get from a
// freshly-reset master, wedging the whole bus. Every I2C-backed begin() in
// this codebase (io_expander.begin() for servo power chief among them,
// since it silently returns false rather than crash) then fails quietly on
// the very next boot - matching "servos don't come on after a flash, only
// after a real power cycle" exactly, since only removing power actually
// resets the stuck slave too.
void recoverStuckI2cBus();
