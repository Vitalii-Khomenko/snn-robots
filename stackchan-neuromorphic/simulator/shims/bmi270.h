#pragma once
// Host shim for the BMI270 driver class: accelerometer readings come from
// the simulated head/body mechanics (gravity, motion, vibration, shaking).
#include <cstdint>
#include "driver/i2c_master.h"
class BMI270 {
public:
    BMI270(i2c_master_bus_handle_t bus, uint8_t address);
    bool begin();
    bool update();
    void getAccelerometer(float& x, float& y, float& z);
    void getGyroscope(float& x, float& y, float& z);  // the head's real angular rates (World::gyro)
private:
    float x_ = 0.0f, y_ = 0.0f, z_ = 9.8f;
};
