#pragma once
#include <array>
#include <cstdint>

#include "v4_spec.h"

// Online action-conditioned 1-second forward model on the robot (FOUNDATIONS §10: Predictive Processing,
// Barrett / Clark / Oudeyer). Runs at 10 Hz (< 15 us on ESP32-S3, 140 weights = 560 B):
//   input  x(t - 1 s) [14]: the 10 external sense channels + [said, gesture, headSpeed01, bias=1]
//   target y(t)       [10]: the 10 external sense channels 1 s later
// Tracks instantaneous prediction error (true predictive surprise) and Oudeyer's Learning Progress
// (slow error EMA - fast error EMA): positive while actively learning a predictable interaction, ~0 in random noise.
class PredictorV4 {
public:
    static constexpr int kY = 10;    // CH_LOUD .. CH_MOTION
    static constexpr int kX = 14;    // kY + [said, gesture, headSpeed01, 1.0]
    static constexpr int kLag = 10;  // 1.0 s at 10 Hz

    void step(const float* ch, bool said, bool gesture, float headSpeed01, bool active);

    float error() const { return err_now_; }
    float errorSlow() const { return err_slow_; }
    // Oudeyer learning progress (-1..1): positive when prediction error is falling over the last ~10-60 s.
    float learningProgress() const { return lp_; }
    // Predictive surprise (0..1): excess of current prediction error above the room's recent baseline.
    float surprise() const { return surprise_; }

private:
    std::array<std::array<float, kX>, kY> w_{};
    std::array<std::array<float, kX>, kLag> ring_{};
    uint32_t count_ = 0;
    float err_now_ = 0.0f, err_fast_ = 0.05f, err_slow_ = 0.05f, lp_ = 0.0f, surprise_ = 0.0f;
};
