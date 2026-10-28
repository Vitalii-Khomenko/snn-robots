#include "servo_supervisor.h"

#include <algorithm>
#include <cmath>
#include <esp_log.h>

#include "../core/ms_clock.h"

namespace {
constexpr const char* kTag = "runtime";  // the robot's log lines as before (tools/v4/jerks.py, body.py read them)
}

// Safety net for the servos. Two 2 s windows of commanded motion without real motion -> cycle the servo rail (VM EN)
// and resend the position. The stalls had two causes (2026-09-27): PWM mode (fixed in MotionEngine::writeServos) and
// too many packets (50/s -> 10/s, see kUpdateMs); this catches whatever is left.
const char* ServoSupervisor::step(uint32_t now, int sagS)
{
    // The head's own jerks (2026-10-27, the owner): the commanded model never moves faster than ~40 deg/s, so a gyro
    // peak over 300 deg/s in the first kJerkWatchMs after the rail came on is the servo running off by itself.
    // 2026-10-28: the gyro axes at the peak, when it came and the packets before it (the bang is heard ~0.85 s after
    // the switch-on), to tell which servo runs off and on what packet.
    if (motion_.railOn() && !rail_was_on_) {
        rail_on_at_ms_ = motion_.railChangeMs() ? motion_.railChangeMs() : 1;
        rail_peak_ = 0.0f;
    }
    rail_was_on_ = motion_.railOn();
    if (rail_on_at_ms_) {
        float gx, gy, gz;
        imu_.getGyro(gx, gy, gz);
        const float g = std::sqrt(gx * gx + gy * gy + gz * gz);
        if (g > rail_peak_) {
            rail_peak_ = g;
            rail_peak_xyz_[0] = gx, rail_peak_xyz_[1] = gy, rail_peak_xyz_[2] = gz;
            rail_peak_ms_ = now - rail_on_at_ms_;
        }
        if (now - rail_on_at_ms_ >= kJerkWatchMs) {
            if (rail_peak_ > 300.0f) {
                ++jerks_;
                char packets[400];
                motion_.formatPackets(rail_on_at_ms_, packets, sizeof(packets));
                ESP_LOGW(kTag, "servos: a jerk of %.0f deg/s in the %.1f s after the rail came on (%s): at +%u ms, gyro "
                         "%.0f,%.0f,%.0f; packets (ms after the switch-on): %s",
                         static_cast<double>(rail_peak_), kJerkWatchMs / 1000.0,
                         now - last_repower_ms_ < 60000 ? "after a hang" : "no hang", static_cast<unsigned>(rail_peak_ms_), static_cast<double>(rail_peak_xyz_[0]),
                         static_cast<double>(rail_peak_xyz_[1]), static_cast<double>(rail_peak_xyz_[2]), packets);
            }
            rail_on_at_ms_ = 0;
        }
    }
    // (A planned power cycle 2 s after Wi-Fi association used to be here, a
    // workaround from before the real causes were found - PWM mode, then the
    // packet rate. Removed 2026-09-27: it only made the head go limp once.)
    bool repower = false;
    const char* why = "";
    if (now - watch_window_ms_ >= 2000) {
        const float commanded = motion_.travelledDeg() - watch_travel_deg_;
        const float real = imu_.takeWatchRotationDeg();
        watch_travel_deg_ = motion_.travelledDeg();
        watch_window_ms_ = now;
        // a silent gyro sees no rotation whatever the head does: that is not a stall (imu_engine.h)
        const bool stalled = commanded >= 10.0f && real < 1.5f && imu_.gyroAlive();
        // Two 2 s windows in a row with >= 10 deg commanded and < 1.5 deg
        // real are a stall. One window is not enough: slow micro-drift
        // glides below the gyro's 3 deg/s noise floor and adds commanded
        // degrees the gyro never counts (one false power cycle on hardware).
        stalled_windows_ = stalled ? stalled_windows_ + 1 : 0;
        // Sparse moves never make 10 deg in two windows in a row: since the gaze moves are rate-limited (one per
        // >= 2.5 s, 2026-10-24) a hung head got a step of 8..18 deg now and then and stood dead for minutes unseen
        // (2026-10-24: in 61 % of the windows with commanded motion the gyro saw none, runs of up to 3 minutes).
        // So the commanded degrees the gyro has not confirmed add up until it sees the head turn; windows under
        // 6 deg (a slow drift below the gyro's noise floor) do not count.
        if (real >= 3.0f) unseen_deg_ = 0.0f;
        else if (real < 1.5f && commanded >= 6.0f && imu_.gyroAlive()) unseen_deg_ += commanded;
        if (unseen_deg_ >= kUnseenDeg) stalled_windows_ = std::max(stalled_windows_, 2);
        // (A soft cure - position mode, torque and position resent - was
        // tried first for a while: it cleared 2 stalls of ~14. The servos
        // really hang; only their power brings them back.)
        // A series means the head is held or blocked (by a hand, a cover, the wall): power-cycling every 12 s did not
        // help and every restart is a current surge - the screen went black and twice the whole board froze
        // (2026-10-11). After 3 cycles within 3 minutes the next one waits 5 minutes.
        const bool series = repower_times_[0] != 0 && repower_times_[2] - repower_times_[0] < 180000;
        // a head that moved for real ends the series; each power cycle within a series makes the next wait longer:
        // 1, 2, 5, 10, 30, 60 min (it began with 5 min, and the owner saw a frozen head for 5 minutes after one
        // series, 2026-10-13; under a cover at night the head may be blocked for hours)
        static constexpr uint32_t kSeriesWaitMs[] = {60000, 120000, 300000, 600000, 1800000, 3600000};
        if (real >= 10.0f) series_level_ = 0;
        const uint32_t gap = series ? kSeriesWaitMs[std::min(series_level_, 5)] : 10000;
        if (stalled_windows_ >= 2 && now - last_repower_ms_ > gap) {
            repower = true;
            if (series) ++series_level_;
            why = "commanded motion but the gyro saw none";
        } else if (stalled_windows_ >= 2 && series && !blocked_logged_) {
            blocked_logged_ = true;
            ESP_LOGW(kTag, "servos: the head seems held or blocked (3 power cycles in 3 min): next power cycle in %u min",
                     static_cast<unsigned>(gap / 60000));
        }
    }
    // A hung servo pulls current for as long as it has power: the USB input sinks to the AXP's limit, the battery
    // makes up the rest, VSYS falls and the 3.3 V DCDC trips - the black-screen power-offs (2026-10-22/23, the
    // last one 4 min into a sag while the stall rule waited out a series pause). A sag of 3 s cuts the servo rail
    // whatever the stall rule says. If the sag outlives the cycle the servos were not the cause: no more sag
    // cycles for 10 minutes.
    const int sag = sagS;
    if (sag_repower_ms_ && now - sag_repower_ms_ > 5000 && now - sag_repower_ms_ < 8000 && sag >= 3) {
        sag_repower_ms_ = 0;
        sag_pause_until_ms_ = now + 600000;
        ESP_LOGW(kTag, "power: VBUS still sags after the servo rail cycle - not the servos; no sag cycles for 10 min");
    }
    if (!repower && sag >= 3 && msReached(now, sag_pause_until_ms_) && now - last_repower_ms_ > 15000) {
        repower = true;
        why = "the servo rail drains the supply: VBUS sagging";
        sag_repower_ms_ = now;
    }
    return repower ? why : nullptr;
}

void ServoSupervisor::repowered(uint32_t now)
{
    last_repower_ms_ = now;
    stalled_windows_ = 0;
    unseen_deg_ = 0.0f;
    ++repowers_;
    // the last three power cycles (oldest first): three within 3 minutes are a series
    repower_times_[0] = repower_times_[1];
    repower_times_[1] = repower_times_[2];
    repower_times_[2] = last_repower_ms_;
    blocked_logged_ = false;
}
