#pragma once
#include <cstdint>

#include "../sensors/imu_engine.h"
#include "motion_engine.h"

// The servo watchdog and the jerk watch (docs/HARDWARE.md, servos). The bus is write-only on this board: the gyro is
// the only way to know the head really moves. Commanded motion the gyro does not see (two 2 s windows of >= 10 deg
// commanded and < 1.5 deg real, or kUnseenDeg of commanded degrees it never confirmed) means a hung servo: the rail
// must be power-cycled. A series (3 power cycles within 3 minutes: a head held or blocked) makes the next one wait 1,
// 2, 5, 10, 30, 60 minutes. A sagging supply (3 s) cuts the rail too. The robot (RobotRuntime) and the simulator
// (SimRuntime, whose servos may hang: `fault servo P`) run the same code (2026-10-28, an audit A21: it lived in
// RobotRuntime only, so nothing but the robot ever ran it).
class ServoSupervisor {
public:
    ServoSupervisor(MotionEngine& motion, ImuEngine& imu) : motion_(motion), imu_(imu) {}
    // Every main-loop pass. sagS: how long the USB input has sagged (PowerStatusEngine::sagSeconds; 0 without one).
    // -> why the servo rail must be power-cycled now, or nullptr. The caller cycles it and then calls repowered().
    const char* step(uint32_t now, int sagS);
    void repowered(uint32_t now);
    uint32_t repowers() const { return repowers_; }
    uint32_t jerks() const { return jerks_; }
    uint32_t lastRepowerMs() const { return last_repower_ms_; }

    static constexpr float kUnseenDeg = 25.0f;
    static constexpr uint32_t kJerkWatchMs = 2500;  // 1.5 s until 2026-10-28 (the servos now start for 1 s)

private:
    MotionEngine& motion_;
    ImuEngine& imu_;
    uint32_t watch_window_ms_ = 0, last_repower_ms_ = 0;
    float watch_travel_deg_ = 0.0f;
    int stalled_windows_ = 0;
    float unseen_deg_ = 0.0f;  // commanded degrees the gyro has not confirmed since it last saw the head turn
    uint32_t repowers_ = 0;
    // the head's own jerks (2026-10-27): the gyro's peak in the first kJerkWatchMs after each rail switch-on
    bool rail_was_on_ = true;
    uint32_t rail_on_at_ms_ = 0, jerks_ = 0;
    float rail_peak_ = 0.0f;
    float rail_peak_xyz_[3] = {0, 0, 0};  // the gyro (deg/s, signed) at that peak, and when (ms after the switch-on)
    uint32_t rail_peak_ms_ = 0;
    uint32_t repower_times_[3] = {0, 0, 0};
    int series_level_ = 0;
    bool blocked_logged_ = false;
    uint32_t sag_repower_ms_ = 0, sag_pause_until_ms_ = 0;
};
