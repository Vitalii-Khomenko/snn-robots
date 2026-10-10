#include "imu_engine.h"

#include <algorithm>
#include <cmath>
#include <bsp/esp-bsp.h>
#include <driver/i2c_master.h>
#include <esp_log.h>
#include <esp_timer.h>

namespace {
constexpr const char* kTag = "imu";
}

bool ImuEngine::begin()
{
    i2c_master_bus_handle_t bus = nullptr;
    if (i2c_master_get_bus_handle(BSP_I2C_NUM, &bus) != ESP_OK) {
        ESP_LOGE(kTag, "i2c_master_get_bus_handle failed");
        return false;
    }

    imu_ = std::make_unique<BMI270>(bus, kImuAddr);
    if (!imu_->begin()) {
        ESP_LOGE(kTag, "BMI270 begin() failed");
        imu_.reset();
        return false;
    }
    return true;
}

void ImuEngine::getAccel(float& x, float& y, float& z) const
{
    if (!imu_) {
        x = y = z = 0.0f;
        return;
    }
    // std::unique_ptr<T>::get() returns T* regardless of the unique_ptr's
    // own constness, so this is fine even though getAccelerometer() itself
    // is a non-const member of the vendored BMI270 wrapper.
    imu_->getAccelerometer(x, y, z);
}

void ImuEngine::update(bool suppressShake)
{
    if (!imu_) return;
    if (!imu_->update()) {
        ++fails_;
        return;
    }
    ++reads_;

    float x, y, z;
    imu_->getAccelerometer(x, y, z);
    {
        // Real head rotation from the gyro (the IMU is in the head): the
        // only physical measurement of servo motion - the servos do not
        // answer read requests. Integrated |omega| over time, deg.
        float gx, gy, gz;
        imu_->getGyroscope(gx, gy, gz);
        const uint32_t now = static_cast<uint32_t>(esp_timer_get_time() / 1000);
        const float dt = last_gyro_ms_ ? std::min(0.2f, static_cast<float>(now - last_gyro_ms_) / 1000.0f) : 0.0f;
        last_gyro_ms_ = now;
        gyro_[0] = gx;
        gyro_[1] = gy;
        gyro_[2] = gz;
        const float omega = std::sqrt(gx * gx + gy * gy + gz * gz);
        last_omega_ = omega;
        max_omega_ = std::max(max_omega_, omega);
        if (omega > 3.0f) {  // ignore gyro noise/bias below 3 deg/s
            rotation_deg_ += omega * dt;
            rotation_watch_deg_ += omega * dt;
        }
    }

    if (suppressShake) {
        rebaseline_countdown_ = kRebaselineSamples;
    } else if (rebaseline_countdown_ > 0 && --rebaseline_countdown_ == 0) {
        baseline_captured_ = false;  // head settled after our own move
        tilt_streak_ = 0;
    }
    if (!baseline_captured_) {
        baseline_x_ = x;
        baseline_y_ = y;
        baseline_z_ = z;
        baseline_captured_ = true;
    }

    float diff = std::fabs(x - prev_x_) + std::fabs(y - prev_y_) + std::fabs(z - prev_z_);
    prev_x_ = x;
    prev_y_ = y;
    prev_z_ = z;

    // Resting is the reference: a body that has lain still for 3 s (wherever it was put down) is not tilted. Without
    // this a robot that was lifted and put back felt "tilted" for good (audit 2026-10-04: 37 % of an evening): the
    // baseline was re-taken only after its own head moved, but the startled robot held its head still.
    still_samples_ = (diff < 0.4f && last_omega_ < 5.0f && !suppressShake) ? still_samples_ + 1 : 0;
    if (still_samples_ >= kRestSamples) {
        baseline_x_ = x;
        baseline_y_ = y;
        baseline_z_ = z;
        still_samples_ = 0;
    }

    // Tilt: sustained deviation from the at-rest baseline above, distinct
    // from SHAKE's frame-to-frame jolt (diff, just computed) - see
    // onTilt()'s comment.
    const float tilt_deviation = std::fabs(x - baseline_x_) + std::fabs(y - baseline_y_) + std::fabs(z - baseline_z_);
    if (!suppressShake && rebaseline_countdown_ == 0 && tilt_deviation > kTiltThreshold) {
        if (++tilt_streak_ >= kTiltSustainTicks && !tilt_active_) {
            tilt_active_ = true;
            if (on_tilt_) {
                on_tilt_();
            }
        }
    } else {
        tilt_streak_ = 0;
        if (tilt_deviation < kTiltReleaseThreshold) {
            tilt_active_ = false;
        }
    }

    // While the head moves the jolt threshold is doubled rather than the
    // detector switched off: the robot turns much of the time, and a shake
    // during a turn went unnoticed. A smooth turn moves gravity in this
    // head-mounted IMU by well under 1 m/s^2 per sample; hands shaking the
    // robot give tens.
    if (diff <= (suppressShake ? 2.0f * kShakeThreshold : kShakeThreshold)) {
        return;
    }

    uint32_t now = static_cast<uint32_t>(esp_timer_get_time() / 1000);
    if (now - last_shake_peak_ms_ <= 100) {
        return;  // debounce
    }
    if (now - last_shake_peak_ms_ < 1000) {
        shake_count_++;
    } else {
        shake_count_ = 1;  // window expired, restart the sequence
    }
    last_shake_peak_ms_ = now;

    if (shake_count_ >= kShakeCountToTrigger) {
        shake_count_ = 0;
        if (on_shake_) {
            on_shake_();
        }
    }
}
