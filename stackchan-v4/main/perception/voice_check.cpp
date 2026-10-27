#include "voice_check.h"

#include <cmath>

namespace {
constexpr double kRate = 24000.0;
constexpr float kPeriodic = 0.6f, kAboveDb = 6.0f, kFloorDown = 0.3f, kFloorUp = 0.002f, kVoiceRate = 0.1f;

// RBJ cookbook, Q = 1/sqrt(2); computed in double, kept as float (as tools/v4/voice_check.py)
void design(bool lowPass, double fc, float* c)
{
    const double w = 2.0 * 3.14159265358979323846 * fc / kRate;
    const double al = std::sin(w) / (2.0 * 0.70710678118654752440);
    const double cw = std::cos(w);
    const double a0 = 1.0 + al;
    const double b0 = lowPass ? (1.0 - cw) / 2.0 : (1.0 + cw) / 2.0;
    const double b1 = lowPass ? 1.0 - cw : -(1.0 + cw);
    c[0] = static_cast<float>(b0 / a0);
    c[1] = static_cast<float>(b1 / a0);
    c[2] = static_cast<float>(b0 / a0);
    c[3] = static_cast<float>(-2.0 * cw / a0);
    c[4] = static_cast<float>((1.0 - al) / a0);
}
}  // namespace

VoiceCheck::VoiceCheck()
{
    const bool low[3] = {false, true, true};
    const double fc[3] = {70.0, 900.0, 900.0};
    for (int k = 0; k < 3; ++k) {
        float c[5];
        design(low[k], fc[k], c);
        stages_[k] = Biquad{c[0], c[1], c[2], c[3], c[4]};
    }
}

void VoiceCheck::feed(const int16_t* mono, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        float x = static_cast<float>(mono[i]) / 32768.0f;
        for (auto& s : stages_) {  // transposed direct form II
            const float y = s.b0 * x + s.z1;
            s.z1 = s.b1 * x - s.a1 * y + s.z2;
            s.z2 = s.b2 * x - s.a2 * y;
            x = y;
        }
        if (++phase_ < kDecim) continue;
        phase_ = 0;
        ring_[n_ % kRing] = x;
        ++n_;
        if (n_ % kHop == 0 && n_ >= static_cast<uint32_t>(kWin + kLagHi)) voice_out_.store(frame(), std::memory_order_relaxed);
    }
}

float VoiceCheck::frame()
{
    // w[0] oldest .. w[kWin + kLagHi - 1] newest
    float w[kWin + kLagHi];
    for (int j = 0; j < kWin + kLagHi; ++j) w[j] = ring_[(n_ - kWin - kLagHi + j) % kRing];
    const float* cur = w + kLagHi;
    float e0 = 0.0f;
    for (int j = 0; j < kWin; ++j) e0 += cur[j] * cur[j];
    float best = 0.0f;
    for (int lag = kLagLo; lag <= kLagHi; ++lag) {
        const float* prev = w + kLagHi - lag;
        float el = 0.0f, r = 0.0f;
        for (int j = 0; j < kWin; ++j) {
            el += prev[j] * prev[j];
            r += cur[j] * prev[j];
        }
        const float c = r / (static_cast<float>(std::sqrt(static_cast<double>(e0) * static_cast<double>(el))) + 1e-12f);
        if (c > best) best = c;
    }
    const float level = 10.0f * std::log10(e0 / kWin + 1e-12f);
    if (!floor_set_) {
        floor_ = level;
        floor_set_ = true;
    }
    floor_ += (level - floor_) * (level < floor_ ? kFloorDown : kFloorUp);
    const bool voiced = best > kPeriodic && level > floor_ + kAboveDb;
    voice_ += ((voiced ? 1.0f : 0.0f) - voice_) * kVoiceRate;
    last_periodicity_ = best;
    last_level_ = level;
    return voice_;
}
