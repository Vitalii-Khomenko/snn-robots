#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <esp_log.h>

// Live statistics of how well the robot sees and follows a person, from the
// camera frames the brain actually receives (no ground truth is needed, so it
// works on the real robot as well as in the simulator). Event lines go out at
// once ("person acquired / centred / lost"); a summary line every 10 s
// while a person was around:
//   10 s | person in 72% of 70 frames | centred 81% (|x|<0.25) | err 0.14 | acquired 2, lost 1 |
//          to-centre 640 ms | head 12 deg
class TrackStats {
public:
    // Call for every camera result handed to the brain (before the tick uses it).
    void delivered() { ++delivered_; }
    // Call once per brain tick. newFrame: a camera frame arrived this tick.
    void update(uint32_t now, bool newFrame, float person, float targetX, float targetConf, float yawDeg)
    {
        if (!win_start_) win_start_ = now;
        ++ticks_;
        if (newFrame) {
            ++frames_;
            // hysteresis: acquired at 0.5, lost below 0.3 for 1 s
            if (person > 0.5f) {
                last_seen_ms_ = now;
                if (!seen_) {
                    seen_ = true;
                    seen_since_ms_ = now;
                    centred_ = false;
                    ++acquired_;
                    ESP_LOGI(kTag, "person acquired (p=%.2f x=%+.2f conf=%.2f)", person, targetX, targetConf);
                }
            } else if (seen_ && person < 0.3f && now - last_seen_ms_ > 1000) {
                seen_ = false;
                ++lost_;
                ESP_LOGI(kTag, "person lost after %.1f s", (now - seen_since_ms_) / 1000.0f);
            }
            if (seen_) {
                ++seen_frames_;
                const float ax = std::fabs(targetX);
                err_sum_ += ax;
                const bool centre = ax < 0.25f && targetConf > 0.2f;
                if (centre) ++centred_frames_;
                if (centre && !centred_) {
                    centred_ = true;
                    to_centre_sum_ms_ += now - seen_since_ms_;
                    ++to_centre_n_;
                    ESP_LOGI(kTag, "person centred after %u ms", static_cast<unsigned>(now - seen_since_ms_));
                } else if (!centre && ax > 0.4f) {
                    centred_ = false;
                }
            }
        }
        if (yaw_min_ > yaw_max_) yaw_min_ = yaw_max_ = yawDeg;
        yaw_min_ = std::min(yaw_min_, yawDeg);
        yaw_max_ = std::max(yaw_max_, yawDeg);
        if (now - win_start_ >= 10000) {
            if (seen_frames_ > 0 || acquired_ || lost_) {
                const float f = std::max<float>(1.0f, frames_);
                const float s = std::max<float>(1.0f, seen_frames_);
                ESP_LOGI(kTag,
                         "10 s | person in %u%% of %u frames | centred %u%% | err %.2f | acquired %u, lost %u | "
                         "to-centre %u ms | head %.0f..%.0f deg | total person %.0f s | camera frames delivered %u, used %u, ticks %u",
                         static_cast<unsigned>(100.0f * seen_frames_ / f), frames_,
                         static_cast<unsigned>(100.0f * centred_frames_ / s), err_sum_ / s, acquired_, lost_,
                         to_centre_n_ ? static_cast<unsigned>(to_centre_sum_ms_ / to_centre_n_) : 0u, yaw_min_, yaw_max_,
                         total_seen_s_ + seen_frames_ / 7.0f, delivered_, frames_, ticks_);
            }
            total_seen_s_ += seen_frames_ / 7.0f;
            win_start_ = now;
            delivered_ = ticks_ = 0;
            frames_ = seen_frames_ = centred_frames_ = acquired_ = lost_ = to_centre_n_ = 0;
            err_sum_ = 0.0f;
            to_centre_sum_ms_ = 0;
            yaw_min_ = 1e9f;
            yaw_max_ = -1e9f;
        }
    }

private:
    static constexpr const char* kTag = "track";
    uint32_t ticks_ = 0, delivered_ = 0, win_start_ = 0, frames_ = 0, seen_frames_ = 0, centred_frames_ = 0, acquired_ = 0, lost_ = 0;
    uint32_t to_centre_n_ = 0, to_centre_sum_ms_ = 0, last_seen_ms_ = 0, seen_since_ms_ = 0;
    float err_sum_ = 0.0f, yaw_min_ = 1e9f, yaw_max_ = -1e9f, total_seen_s_ = 0.0f;
    bool seen_ = false, centred_ = false;
};
