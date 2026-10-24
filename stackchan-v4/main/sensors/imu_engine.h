#pragma once
#include <functional>
#include <memory>
#include <bmi270.h>

// BMI270 6-axis IMU + simple shake detection (differential accelerometer
// threshold with debounce, same shape as the known-working
// MotionDetector in upstream/m5stack-StackChan/firmware/main/hal/utils/motion_detector).
class ImuEngine {
public:
    bool begin();
    // Poll periodically. Pass true (e.g. MotionEngine::isMoving()) while
    // the robot's own servos are actively moving it - a moving/head-mounted
    // IMU feels the robot's own commanded motion as real acceleration too,
    // and without this it can mistake its own gesture for an externally
    // applied shake (the same class of self-sensing bug the microphone had
    // - see MicrophoneEngine::lastPeakLevel()'s comment). Suppression only
    // gates the shake trigger itself; prev_x_/y_/z_ keep tracking every
    // call so there's no stale-baseline jump once motion stops.
    void update(bool suppressShake = false);

    void onShake(std::function<void()> cb) { on_shake_ = std::move(cb); }
    // A sustained deviation of the gravity vector from the at-rest baseline
    // captured on the first update() - being picked up/held at an angle,
    // not a jolt (SHAKE catches that, from the frame-to-frame difference
    // above, not this). Same suppression as onShake() and for the same
    // reason: the head's own commanded pitch/yaw would otherwise register
    // as a constant "tilt" too. Sustained over kTiltSustainTicks consecutive
    // update() calls (not just one noisy sample) before firing, then needs
    // to drop back under kTiltReleaseThreshold before it can fire again -
    // see the constants' comments. First-guess thresholds, not validated
    // against real handling on hardware.
    void onTilt(std::function<void()> cb) { on_tilt_ = std::move(cb); }
    void getAccel(float& x, float& y, float& z) const;
    // Latest angular rate, deg/s, sensor axes (see RobotRuntime for the
    // self-calibrated mapping to yaw / pitch).
    void getGyro(float& x, float& y, float& z) const
    {
        x = gyro_[0];
        y = gyro_[1];
        z = gyro_[2];
    }
    // Same as takeRotationDeg() but a separate accumulator (servo watchdog).
    float takeWatchRotationDeg()
    {
        const float r = rotation_watch_deg_;
        rotation_watch_deg_ = 0.0f;
        return r;
    }
    // Degrees the head physically rotated (gyro) since the last call.
    float takeRotationDeg()
    {
        const float r = rotation_deg_;
        rotation_deg_ = 0.0f;
        return r;
    }
    // Health for the stats line: sensor reads that succeeded / failed, and
    // the largest angular rate (deg/s) since the last call.
    void takeHealth(uint32_t& reads, uint32_t& fails, float& maxOmega)
    {
        reads = reads_;
        fails = fails_;
        maxOmega = max_omega_;
        reads_ = fails_ = 0;
        max_omega_ = 0.0f;
    }
    // True from the tilt trigger until the head is level again (hysteresis).
    bool tiltActive() const { return tilt_active_; }
    // The gyro inside the BMI270 goes silent now and then (2026-10-11: exact zeros for about half of the head's moves
    // in the afternoon, the accelerometer fine): the servo watchdog then saw "commanded motion, no rotation" and
    // power-cycled the servos every 12 s. Silent = exactly zero on all three axes for 1 s (a live gyro has a
    // zero-rate offset of a few 0.06 deg/s steps); after 2 s the chip is re-initialised (at most every 30 s).
    bool gyroAlive() const { return frozen_samples_ < 50; }
    uint32_t gyroRestarts() const { return gyro_restarts_; }
    // for the servo watchdog's log line: how many readings in a row were identical, bit for bit
    int gyroRepeats() const { return repeat_samples_; }

private:
    // The wrapper's default I2C address (BMI2_I2C_PRIM_ADDR, 0x68) collides
    // with the Si12T touch sensor on this board (SI12T_GND_ADDRESS is also
    // 0x68) - the secondary address (0x69) is what the known-working driver
    // uses, and what bring-up confirmed actually works here.
    static constexpr uint8_t kImuAddr = 0x69;

    std::unique_ptr<BMI270> imu_;
    float prev_x_ = 0.0f, prev_y_ = 0.0f, prev_z_ = 0.0f;
    int shake_count_             = 0;
    uint32_t last_shake_peak_ms_ = 0;
    std::function<void()> on_shake_;

    // Tilt (see onTilt()'s comment). Baseline captured once, on the first
    // successful update() - the robot is assumed to be at rest in its
    // normal mounting position at boot, same assumption
    // MotionEngine::begin()'s teleport(0.0f) already makes about yaw/pitch.
    float baseline_x_ = 0.0f, baseline_y_ = 0.0f, baseline_z_ = 0.0f;
    bool baseline_captured_ = false;
    int tilt_streak_             = 0;
    bool tilt_active_             = false;  // latched until deviation drops below the release threshold
    std::function<void()> on_tilt_;

    // Lowered from 16.0/3-in-a-row after a report that gently rocking the
    // body (not a firm deliberate shake) never registered - untested
    // against real hardware since (no way to bench-validate an IMU
    // threshold without the physical unit), so treat this as a first
    // guess to react to reports of "too sensitive" or "still not enough"
    // rather than a validated constant. Safe to lower further now that
    // suppressShake exists to rule out self-triggering from the robot's
    // own movement as the cause of any over-sensitivity.
    static constexpr float kShakeThreshold  = 8.0f;
    static constexpr int kShakeCountToTrigger = 2;

    // Accelerometer readings are in m/s^2 (BMI270::lsb_to_mps2()), so these
    // are on the same scale as kShakeThreshold above, just compared against
    // a fixed baseline instead of the previous frame. 3.0 m/s^2 is a
    // moderate, sustained tilt (well under gravity's full 9.8 m/s^2 swing
    // for a 90-degree tip) - a first guess, same caveat as kShakeThreshold's
    // own comment: no way to bench-validate an IMU threshold without the
    // physical unit.
    static constexpr float kTiltThreshold         = 3.0f;
    static constexpr float kTiltReleaseThreshold  = 1.5f;  // hysteresis to re-arm
    // ~300ms at the ~50Hz rate RobotRuntime::loop() calls update() at -
    // long enough that a brief jolt (already SHAKE's job) can't also read
    // as a tilt.
    static constexpr int kTiltSustainTicks = 15;
    // The IMU sits in the head, so pitching the head with its own servo
    // rotates gravity in the sensor and read as "picked up" (Tilt) once the
    // pitch passed ~16 deg (found with simulator/, 2026-09-26). After the
    // robot's own movement ends, re-capture the baseline once the head has
    // settled; a real tilt of the body still shows as a deviation from it.
    // Axis-agnostic on purpose: the IMU axes are not calibrated to the head.
    static constexpr int kRebaselineSamples = 15;  // ~300 ms of main-loop updates
    int rebaseline_countdown_ = 0;
    static constexpr int kRestSamples = 150;  // ~3 s of main-loop updates lying still: that is where the body rests
    int still_samples_ = 0;
    float last_omega_ = 0.0f;
    float rotation_deg_ = 0.0f, rotation_watch_deg_ = 0.0f;
    float gyro_[3] = {0.0f, 0.0f, 0.0f};
    uint32_t last_gyro_ms_ = 0;
    uint32_t reads_ = 0, fails_ = 0;
    int frozen_samples_ = 0, repeat_samples_ = 0;
    uint32_t last_restart_ms_ = 0, gyro_restarts_ = 0;
    float max_omega_ = 0.0f;
};
