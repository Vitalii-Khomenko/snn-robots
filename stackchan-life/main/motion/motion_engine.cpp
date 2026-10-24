#include "motion_engine.h"

#include <algorithm>
#include <cmath>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <esp_log.h>
#include <esp_timer.h>
#include <esp_rom_sys.h>

namespace {
constexpr const char* kTag = "motion";

constexpr uint8_t kYawId    = 1;
constexpr uint8_t kPitchId  = 2;
constexpr int kYawZeroPos   = 460;
constexpr int kPitchZeroPos = 620;
// Packets go out every kUpdateMs; each asks the servo to take a bit longer
// than that to get there, so it glides from packet to packet instead of
// stopping in between.
// The servos hang (hold torque, ignore positions until their power is
// cycled) more often the more packets they get. Hardware A/B 2026-09-27,
// stalls: 50 Hz broadcast every ~24 s, 50 Hz addressed every ~36 s, 20 Hz
// every ~75 s, 10 Hz one in 5 minutes.
constexpr uint32_t kUpdateMs = 100;         // 10 packets/s per servo
constexpr uint16_t kServoGoalTimeMs = 120;
// hal_servo.cpp: "one step corresponds to 0.3125 degrees" - one raw servo step is 0.3125 degrees.
constexpr float kRawPerDeg = 1.0f / 0.3125f;  // 3.2
}  // namespace

bool MotionEngine::begin()
{
    if (!bus_.begin(UART_NUM_1, 1000000, 6, 7)) {
        ESP_LOGE(kTag, "servo bus begin() failed");
        return false;
    }
    // The StackChan body's SCS bus is effectively TX-only: its servos accept
    // position/torque commands but do not return reliable ACK or position
    // packets. Do not probe with Ping/ReadPos here and do not wait for ACKs
    // during the 50 Hz motion loop; the known-good v1 driver uses the same
    // fire-and-forget pattern.
    bus_.Level = 0;
    yaw_spring_.teleport(0.0f);
    pitch_spring_.teleport(0.0f);
    commanded_yaw_target_ = 0.0f;
    commanded_pitch_target_ = 0.0f;
    return true;
}

void MotionEngine::moveTo(float yaw_deg, float pitch_deg, float speed_deg_s)
{
    yaw_deg   = std::clamp(yaw_deg, kYawMinDeg, kYawMaxDeg);
    pitch_deg = std::clamp(pitch_deg, kPitchMinDeg, kPitchMaxDeg);
    // every move at half the asked speed (2026-10-23, the owner's call): the AXP2101 switched the board off for a
    // DCDC under-voltage (black screen, 5 times on 22-23.10), the evening ones in the middle of fast turns
    max_speed_ = std::clamp(speed_deg_s * kSpeedScale, 5.0f, 300.0f);
    // The spring only smooths the profiled target (stiff, critically
    // damped, ~50 ms): the profile decides how fast the head moves.
    applySpeedToSpring(yaw_spring_, 1000);
    applySpeedToSpring(pitch_spring_, 1000);
    commanded_yaw_target_   = yaw_deg;
    commanded_pitch_target_ = pitch_deg;
    // Re-assert torque at the start of a movement at most every few seconds.
    // SCS servos drop torque on their own (overload/overheat protection) and
    // then silently ignore position packets - seen on hardware 2026-09-27:
    // the head moved a few degrees, then stood still while commands kept
    // coming. Enabling once at boot was not enough.
    const uint32_t now = static_cast<uint32_t>(esp_timer_get_time() / 1000);
    if (now - last_torque_ms_ >= kTorqueRefreshMs) {
        yaw_torque_enabled_ = pitch_torque_enabled_ = false;  // writeServos() re-enables once
        last_torque_ms_ = now;
    }
}

void MotionEngine::getCurrentDeg(float& yaw_deg, float& pitch_deg) const
{
    yaw_deg   = yaw_spring_.value;
    pitch_deg = pitch_spring_.value;
}

// Mode and torque go out as broadcasts (id 0xFE: servos never answer them);
// positions as addressed packets with a 1 ms quiet gap after each, because a
// servo answers every addressed packet on the same wire - this firmware cannot
// hear it (the RX line is dead, docs/HARDWARE.md), and a packet sent right
// behind another ran into that reply: the prime suspect for the servos
// wedging (holding torque, ignoring positions until their power is cycled).
void MotionEngine::writeServos(bool yaw, bool pitch, float yaw_deg, float pitch_deg, uint16_t time_ms)
{
    constexpr uint8_t kBroadcast = 0xFE;
    if ((yaw && !yaw_torque_enabled_) || (pitch && !pitch_torque_enabled_)) {
        // Force POSITION mode. These SCS servos also have a motor (PWM) mode,
        // selected by angle limits 0/0 - the official StackChan firmware
        // switches the yaw servo into it. In that mode position packets are
        // ignored and the goal-time register we write (50) is read as 5%
        // motor power: the head creeps one way and stops. Written once per
        // power-on (EEPROM-backed registers; repeating them every 3 s, as
        // before, risked EEPROM wear and kept the servo busy).
        if (!limits_written_[0] || !limits_written_[1]) {
            bus_.writeWord(kBroadcast, SCSCL_MIN_ANGLE_LIMIT_L, 0);
            bus_.writeWord(kBroadcast, SCSCL_MAX_ANGLE_LIMIT_L, 1000);
            limits_written_[0] = limits_written_[1] = true;
        }
        bus_.EnableTorque(kBroadcast, 1);
        yaw_torque_enabled_ = pitch_torque_enabled_ = true;
    }
    uint8_t ids[2];
    uint16_t pos[2], time[2], speed[2];
    int n = 0;
    auto add = [&](uint8_t id, int zero_pos, float deg) {
        ids[n] = id;
        pos[n] = static_cast<uint16_t>(std::clamp(zero_pos + static_cast<int>(deg * kRawPerDeg), 0, 1000));
        time[n] = time_ms;
        speed[n] = 0;
        ++n;
    };
    if (yaw) add(kYawId, kYawZeroPos, yaw_deg);
    if (pitch) add(kPitchId, kPitchZeroPos, pitch_deg);
    packets_sent_ += static_cast<uint32_t>(n);
    // Addressed packets, like the official firmware - which waits for each
    // reply (up to 10 ms); we cannot hear it (the RX line is dead), so leave
    // the bus quiet for 1 ms instead so the reply never collides with the
    // next packet. (A broadcast SYNC_WRITE was A/B-tested 2026-09-27: what
    // matters is the packet rate; addressed stalled a little less.)
    // Since 2026-10-22 one broadcast SYNC_WRITE instead (kSyncWrite): with the sparse writes the packet rate no longer
    // explains the stalls (still ~1 per minute of motion, both servos at once, the head stiff), so the replies to
    // addressed packets - which nobody hears on this board - are the next suspect.
    if (kSyncWrite) {
        bus_.SyncWritePos(ids, static_cast<uint8_t>(n), pos, time, speed);
        return;
    }
    for (int i = 0; i < n; ++i) {
        bus_.WritePos(ids[i], pos[i], time[i], 0);
        esp_rom_delay_us(1000);
    }
}

void MotionEngine::update()
{
    uint32_t now = esp_timer_get_time() / 1000;
    if (now - last_update_ms_ < kUpdateMs) {
        return;  // 10 Hz: the servo packet rate (50 Hz made them stall)
    }
    // dt is the real time since the last update (~0.1 s), capped only
    // against a long loop stall. (It was capped at 0.05 s from the 50 Hz
    // days: at 10 Hz every profile, spring and stamina step ran at half the
    // real time - calm turns at 12.6 instead of 25 deg/s, nods at half
    // speed, rest recovery twice as slow. Found 2026-09-29.) The spring is
    // integrated in <= 20 ms substeps below, so a longer dt is safe.
    float dt          = (last_update_ms_ == 0) ? 0.02f : (now - last_update_ms_) / 1000.0f;
    dt                 = std::min(dt, 0.25f);
    last_update_ms_    = now;

    // Motion profile: the spring's target walks along the straight line to
    // the commanded pose; its speed ramps up and down with a fixed
    // acceleration (max speed / kRampS) and is capped by the braking
    // distance, so every move starts and ends softly. (It used to jump to
    // 75 deg/s at once: the head jerked.)
    {
        const float dy = commanded_yaw_target_ - yaw_spring_.target;
        const float dp = commanded_pitch_target_ - pitch_spring_.target;
        const float dist = std::hypot(dy, dp);
        if (dist < kTargetEpsilonDeg) {
            yaw_spring_.target = commanded_yaw_target_;
            pitch_spring_.target = commanded_pitch_target_;
            profile_speed_ = 0.0f;
        } else {
            const float accel = max_speed_ / kRampS;
            const float allowed = std::min(max_speed_, std::sqrt(2.0f * accel * dist));
            profile_speed_ += std::clamp(allowed - profile_speed_, -accel * dt, accel * dt);
            profile_speed_ = std::max(profile_speed_, 0.5f * accel * dt);  // never stall short of the goal
            const float step = std::min(profile_speed_ * dt, dist);
            yaw_spring_.target += dy / dist * step;
            pitch_spring_.target += dp / dist * step;
        }
    }

    bool yaw_was_done   = yaw_spring_.done();
    bool pitch_was_done = pitch_spring_.done();
    float yaw_before_deg   = yaw_spring_.value;
    float pitch_before_deg = pitch_spring_.value;
    // Sub-steps of <= 20 ms: the stiff tracking spring is explicit-Euler
    // stable only for short steps (at 50 ms it blew up to infinity).
    const int steps = std::max(1, static_cast<int>(std::ceil(dt / 0.02f)));
    for (int i = 0; i < steps; ++i) {
        yaw_spring_.update(dt / steps);
        pitch_spring_.update(dt / steps);
    }

    // Stamina: drain proportional to degrees actually travelled this tick
    // (not degrees requested - a spring easing toward a far target costs
    // the same per real step regardless of how big the original moveTo()
    // call was), regenerate at a flat rate whenever nothing moved. Both
    // happen every update() call so a robot that's constantly gesturing
    // never gets a regen tick in between.
    float moved_deg = std::abs(yaw_spring_.value - yaw_before_deg) + std::abs(pitch_spring_.value - pitch_before_deg);
    travelled_deg_ += moved_deg;
    stamina_ -= moved_deg * kStaminaCostPerDeg;
    // Rest = no FAST movement for a while. The slow micro-drift between
    // saccades (a few deg/s) is rest, like a resting animal still shifting
    // its gaze; requiring complete stillness meant stamina never recovered
    // once the drift existed (hardware 2026-09-27: 65% and falling).
    const float speed_deg_s = dt > 0.0f ? moved_deg / dt : 0.0f;
    if (speed_deg_s > kRestSpeedDegPerSec) {
        last_moved_ms_ = now;
    } else if (now - last_moved_ms_ >= kStaminaRestDelayMs) {
        stamina_ += kStaminaRegenPerSec * dt;
    }
    stamina_ = std::clamp(stamina_, 0.0f, 1.0f);

    // Keep sending packets while moving, plus exactly one more once a spring
    // settles (yaw_was_done/pitch_was_done catches that last "just finished"
    // frame) so the final position is actually latched.
    if (kSparseWrites) {
        writeSparse(now);
        return;
    }
    const bool yaw = force_write_ || !yaw_spring_.done() || !yaw_was_done;
    const bool pitch = force_write_ || !pitch_spring_.done() || !pitch_was_done;
    force_write_ = false;
    if (yaw || pitch) writeServos(yaw, pitch, yaw_spring_.value, pitch_spring_.value, kServoGoalTimeMs);
}

// The destination of the move goes out once, with the time the profile needs to get there, and the servo glides
// there by itself; the spring model keeps the same profile for the brain (efference copy, the gyro check). A
// destination that keeps moving (following a person, the slow drift) goes out again when it has moved by
// kRetargetRaw, at most every kMinPacketGapMs; when the move has settled its end is sent once more, in case a
// packet was lost (nothing on this bus says whether a servo took one).
void MotionEngine::writeSparse(uint32_t now)
{
    const float dest[2] = {commanded_yaw_target_, commanded_pitch_target_};
    const int zero[2] = {kYawZeroPos, kPitchZeroPos};
    int raw[2];
    for (int i = 0; i < 2; ++i) raw[i] = std::clamp(zero[i] + static_cast<int>(dest[i] * kRawPerDeg), 0, 1000);
    const bool settled = !isMoving();
    bool send[2] = {false, false};
    bool latch = false;
    for (int i = 0; i < 2; ++i) {
        if (force_write_) {
            send[i] = true;
        } else if (std::abs(raw[i] - last_sent_raw_[i]) >= kRetargetRaw || (settled && raw[i] != last_sent_raw_[i])) {
            send[i] = settled || now - last_packet_ms_ >= kMinPacketGapMs;
        } else if (settled && unlatched_[i]) {
            send[i] = latch = true;
        }
    }
    if (!send[0] && !send[1]) return;
    force_write_ = false;
    uint16_t time_ms = kServoGoalTimeMs;
    if (!latch) {
        // the trapezoid profile: the distance left at full speed, plus a ramp when it is not yet at full speed
        const float dist = std::hypot(dest[0] - yaw_spring_.value, dest[1] - pitch_spring_.value);
        float t = dist / max_speed_ + (profile_speed_ < 0.5f * max_speed_ ? kRampS : 0.0f);
        time_ms = static_cast<uint16_t>(std::clamp(t * 1000.0f, static_cast<float>(kServoGoalTimeMs), 4000.0f));
    }
    writeServos(send[0], send[1], dest[0], dest[1], time_ms);
    for (int i = 0; i < 2; ++i) {
        if (!send[i]) continue;
        last_sent_raw_[i] = raw[i];
        unlatched_[i] = !latch && !settled;
    }
    last_packet_ms_ = now;
}
