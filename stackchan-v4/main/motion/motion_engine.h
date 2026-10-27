#pragma once
#include <SCSCL.h>
#include <esp_timer.h>
#include "spring.h"

// Smooth head motion over the SCS servo bus. Same bus parameters and servo
// IDs proven on this physical robot during bring-up (see docs/HARDWARE.md):
// UART1, 1 Mbaud, TX=GPIO6/RX=GPIO7, yaw=id 1 (zero_pos 460), pitch=id 2
// (zero_pos 620). Positions here are in degrees relative to center, not raw
// servo units - 0.3125 deg/raw-unit is the same conversion
// hal_servo.cpp's set_angle_impl uses.
//
// power::enableServoPower() must be called before begin() - this class only
// owns the UART bus, not the body's separate servo-power-enable line.
class MotionEngine {
public:
    bool begin();
    void update();  // call at ~50 Hz or faster; internally rate-limited

    // Move to (yaw, pitch) at up to `speed_deg_s` along the straight line,
    // with a smooth start and stop (trapezoid profile, full speed reached
    // after kRampS). Retargeting mid-move keeps the current speed: no jerk.
    // A move that takes kEaseMinS or longer goes as an eased one instead (kEasedMoves).
    void moveTo(float yaw_deg, float pitch_deg, float speed_deg_s);
    bool isMoving() const {
        return !yaw_spring_.done() || !pitch_spring_.done() ||
               std::fabs(yaw_spring_.target - commanded_yaw_target_) > kTargetEpsilonDeg ||
               std::fabs(pitch_spring_.target - commanded_pitch_target_) > kTargetEpsilonDeg;
    }
    void getCurrentDeg(float& yaw_deg, float& pitch_deg) const;
    // Model velocity of the head (deg/s), held between the 10 Hz updates.
    // Differentiating getCurrentDeg() in the 50 Hz main loop gives 0 four
    // times and a 5x spike once: every calm turn looked like a saccade.
    void getVelocityDegS(float& yaw_deg_s, float& pitch_deg_s) const
    {
        yaw_deg_s = yaw_spring_.velocity;
        pitch_deg_s = pitch_spring_.velocity;
    }

    // After the servo rail was power-cycled: re-enable torque and send the
    // current position again on the next update().
    void resyncAfterPowerCycle()
    {
        yaw_torque_enabled_ = pitch_torque_enabled_ = false;
        limits_written_[0] = limits_written_[1] = false;  // RAM copies reset with the power
        force_write_ = true;
    }
    // Position packets written to the bus since the last call (diagnostics;
    // the bus is write-only, so nothing tells whether a servo took one).
    uint32_t takePacketCount()
    {
        const uint32_t n = packets_sent_;
        packets_sent_ = 0;
        return n;
    }

    // Cumulative yaw+pitch degrees actually travelled since boot.
    float travelledDeg() const { return travelled_deg_; }
    // True while the head moves faster than a slow drift (saccades, nods).
    // The touch pad's servo coupling (phantom touches) comes from these.
    // (The servo rail switching on or off jolts the head a little too: kGatedPower.)
    bool isMovingFast(uint32_t nowMs) const { return nowMs - last_moved_ms_ < 300 || railSwitching(nowMs); }

    // Power-gated moves (kGatedPower): the servo rail was switched on or off within the last 400 ms (the servos
    // take hold or let go: a click in the microphone, a small jolt in the IMU and the head pad).
    bool railSwitching(uint32_t nowMs) const { return kGatedPower && nowMs - rail_change_ms_ < 400; }
    bool railOn() const { return rail_on_; }
    // what switches the rail (power::setServoPower on the robot)
    using RailFn = bool (*)(bool on);
    void setRailSwitch(RailFn fn) { rail_fn_ = fn; }
    // The servo watchdog in gated mode: the rail off now (it comes on again, after kRailMinOffMs, for the next move).
    void cutPower();
    // Rail switch-ons and the ms the rail was on since the last call (diagnostics).
    void takeRailStats(uint32_t& ons, uint32_t& onMs)
    {
        const uint32_t now = static_cast<uint32_t>(esp_timer_get_time() / 1000);
        if (rail_on_) rail_on_ms_ += now - rail_count_ms_;
        rail_count_ms_ = now;
        ons = rail_ons_;
        onMs = rail_on_ms_;
        rail_ons_ = rail_on_ms_ = 0;
    }

    // +-128 degrees: the limit the official StackChan firmware uses for
    // this yaw servo (hal_servo.cpp angleLimit -1280..1280 in 0.1 deg).
    // The servo itself spans ~312 degrees (0..1000 raw at 0.3125 deg per
    // unit, zero at 460), so it cannot turn a full 360.
    static constexpr float kYawMinDeg   = -128.0f;
    static constexpr float kYawMaxDeg   = 128.0f;
    static constexpr float kPitchMinDeg = -30.0f;
    static constexpr float kPitchMaxDeg = 60.0f;

    // Power-gated moves (2026-10-25, the owner's idea): the servo rail (VM EN) is on only while the head moves. A
    // hung servo (docs/HARDWARE.md: ~1.5 moves in 100, both servos at once, only a power cycle helps) then costs one
    // move instead of a dead head until the watchdog sees it; and a still head draws no current. The move waits
    // kRailBootMs for the servos to start, the rail goes off kRailHoldMs after the servo should have arrived, and stays
    // off at least kRailMinOffMs (long enough for a hung servo to reset). false = the rail always on, as before.
    static constexpr bool kGatedPower = true;
    static constexpr uint32_t kRailBootMs = 200;
    static constexpr uint32_t kRailHoldMs = 600;
    static constexpr uint32_t kRailMinOffMs = 300;

private:
    bool railReady(uint32_t now);
    void setRail(bool on, uint32_t now);
    void writeServos(bool yaw, bool pitch, float yaw_deg, float pitch_deg, uint16_t time_ms);
    void writeSparse(uint32_t now);

    SCSCL bus_;
    Spring yaw_spring_;
    Spring pitch_spring_;
    bool yaw_torque_enabled_   = false;
    bool pitch_torque_enabled_ = false;
    uint32_t last_update_ms_   = 0;
    uint32_t last_torque_ms_   = 0;
    uint32_t packets_sent_ = 0;
    bool force_write_ = false;
    bool limits_written_[2] = {false, false};  // position-mode limits sent since power-on
    static constexpr uint32_t kTorqueRefreshMs = 3000;
    // Sparse writes (2026-10-20): one packet per move with the destination and the time to get there (the servo
    // glides there by itself), one more when the move has settled, a new one only when the destination moves by
    // >= kRetargetRaw and not sooner than kMinPacketGapMs. The stalls grow with the packets a servo gets while it
    // moves (docs/HARDWARE.md); the old stream of 10 packets/s made ~1000 power cycles a day, and the camera showed
    // the head standing still in half of the commanded turns. false = the old 10 Hz stream of points.
    static constexpr bool kSparseWrites = true;
    static constexpr uint32_t kMinPacketGapMs = 300;
    // Positions as one broadcast SYNC_WRITE (servos never answer a broadcast) instead of addressed packets (each
    // answered on the same wire). false = addressed, as until 2026-10-22.
    static constexpr bool kSyncWrite = false;  // 2026-10-22: no fewer stalls; 3 full freezes the same evening
    static constexpr int kRetargetRaw = 3;  // ~1 degree
    int last_sent_raw_[2] = {-1, -1};       // yaw, pitch: the last destination sent
    bool unlatched_[2] = {false, false};    // a move went out; resend its end once it has settled
    uint32_t last_packet_ms_ = 0;
    uint32_t arrive_ms_ = 0;  // when the servo should have reached the last destination sent
    // the rail (kGatedPower); power::enableServoPower() leaves it on at boot
    RailFn rail_fn_ = nullptr;
    bool rail_on_ = true;
    uint32_t rail_change_ms_ = 0, rail_idle_ms_ = 0;
    uint32_t rail_ons_ = 0, rail_on_ms_ = 0, rail_count_ms_ = 0;

    // A gesture can request a large jump in one call. Keep that command as a
    // destination and slew the spring's target in small increments in update().
    // This is a second safety layer on top of the spring damping and the
    // 50 ms servo goal time, so a new gesture cannot kick the head sharply.
    float commanded_yaw_target_   = 0.0f;
    float commanded_pitch_target_ = 0.0f;
    float max_speed_ = 60.0f;   // deg/s of the current move
    static constexpr float kSpeedScale = 0.5f;  // moveTo() speeds are halved (2026-10-23); 1.0 = as asked
    float profile_speed_ = 0.0f;  // deg/s the moving target has right now
    static constexpr float kRampS = 0.25f;  // time to full speed (and to stop)
    // Eased moves (2026-10-25, the owner: "a short whizz of the motors every few seconds, it should move slowly and
    // smoothly"): the servo goes to one destination at one constant speed, so every move started and stopped with a
    // jerk. A gaze turn now follows half a cosine, (1 - cos(pi t / T)) / 2 - from rest, fastest in the middle, to rest -
    // `max_speed_` at its peak (T = pi/2 * distance / peak speed), and the servo gets it as kEaseStepS-long waypoints,
    // each with the time to reach it. Short moves (gesture keyframes) stay single packets. false = the trapezoid only.
    static constexpr bool kEasedMoves = true;
    static constexpr float kEaseMinS = 0.8f, kEaseStepS = 0.45f;
    static constexpr int kEaseMaxSteps = 8;
    bool ease_ = false, ease_started_ = false;
    float ease_from_[2] = {0, 0}, ease_to_[2] = {0, 0};
    uint32_t ease_start_ms_ = 0, ease_ms_ = 0;
    int ease_steps_ = 1, ease_sent_ = 0;
    float easeAt(int axis, float tau) const
    {
        const float s = tau >= 1.0f ? 1.0f : 0.5f - 0.5f * std::cos(3.14159265f * std::max(tau, 0.0f));
        return ease_from_[axis] + (ease_to_[axis] - ease_from_[axis]) * s;
    }
    static constexpr float kTargetEpsilonDeg      = 0.1f;

    float travelled_deg_ = 0.0f;
    uint32_t last_moved_ms_ = 0;
    // slower than this is rest (micro-drift, settling), not a fast move (isMovingFast)
    static constexpr float kRestSpeedDegPerSec = 8.0f;
};
