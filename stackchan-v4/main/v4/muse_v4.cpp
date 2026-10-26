#include "muse_v4.h"

#include <algorithm>
#include <cmath>

void MuseV4::setBias(const float* bias)
{
    for (int k = 0; k < muse_data::kOuts; ++k)
        bias_[k] = bias ? std::clamp(bias[k], -muse_data::kBiasMax, muse_data::kBiasMax) : 0.0f;
}

void MuseV4::step(const float* snn, const float* noise, Out& out)
{
    using namespace muse_data;
    float u[kDrives];
    for (int d = 0; d < kDrives; ++d) u[d] = snn[kDriveIdx[d]] - 0.5f;
    std::array<float, kN> pre{};
    for (int i = 0; i < kN; ++i) {
        float s = kB[i];
        const float* w = kW + i * kN;
        for (int j = 0; j < kN; ++j) s += w[j] * x_[j];
        const float* wi = kWin + i * kDrives;
        for (int d = 0; d < kDrives; ++d) s += wi[d] * u[d];
        pre[i] = s;
    }
    for (int i = 0; i < kN; ++i)
        x_[i] = std::clamp((1.0f - kLeak[i]) * x_[i] + kLeak[i] * std::tanh(pre[i]) + kNoise[i] * noise[i], -3.0f, 3.0f);
    for (int k = 0; k < kOuts; ++k) {
        float s = kBout[k] + bias_[k];
        const float* w = kWout + k * kN;
        for (int j = 0; j < kN; ++j) s += w[j] * x_[j];
        const float* wu = kWu + k * kDrives;
        for (int d = 0; d < kDrives; ++d) s += wu[d] * u[d];
        float y = std::tanh(s);
        if (k >= 3) {  // an idea: the Muse gets bored of what it keeps having
            float& h = hab_[k - 3];
            h += (std::max(y, 0.0f) - h) * kHabRate;
            if (y > 0.0f) y *= 1.0f - kHabCut * h;
        }
        out[k] = y;
    }
}
