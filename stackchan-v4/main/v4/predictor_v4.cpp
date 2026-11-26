#include "predictor_v4.h"

#include <algorithm>
#include <cmath>

void PredictorV4::step(const float* ch, bool said, bool gesture, float headSpeed01, bool active)
{
    std::array<float, kX> cur{};
    for (int i = 0; i < kY; ++i) cur[i] = std::clamp(ch[i], 0.0f, 1.0f);
    cur[kY + 0] = said ? 1.0f : 0.0f;
    cur[kY + 1] = gesture ? 1.0f : 0.0f;
    cur[kY + 2] = std::clamp(headSpeed01, 0.0f, 1.0f);
    cur[kY + 3] = 1.0f;

    const int slot = static_cast<int>(count_ % kLag);
    if (count_ >= kLag && active) {
        const auto& prev = ring_[slot];
        float x2 = 0.0f;
        for (int j = 0; j < kX; ++j) x2 += prev[j] * prev[j];
        const float lr = 0.02f / (x2 + 1.0f);

        float mae = 0.0f;
        for (int i = 0; i < kY; ++i) {
            float pred = 0.0f;
            auto& wi = w_[i];
            for (int j = 0; j < kX; ++j) pred += wi[j] * prev[j];
            const float diff = cur[i] - pred;
            mae += std::fabs(diff);
            const float g = lr * std::clamp(diff, -0.5f, 0.5f);
            for (int j = 0; j < kX; ++j) wi[j] = std::clamp(wi[j] + g * prev[j], -2.0f, 2.0f);
        }
        err_now_ = mae / static_cast<float>(kY);
        // Fast (~10 s) and slow (~60 s) error EMAs at 10 Hz
        err_fast_ += (err_now_ - err_fast_) * 0.01f;
        err_slow_ += (err_now_ - err_slow_) * 0.0017f;
        lp_ = std::clamp((err_slow_ - err_fast_) / (err_slow_ + 0.02f), -1.0f, 1.0f);
        surprise_ = std::clamp((err_now_ - err_slow_) / (err_slow_ + 0.03f), 0.0f, 1.0f);
    }
    ring_[slot] = cur;
    ++count_;
}
