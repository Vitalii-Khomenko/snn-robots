#include "body_v4.h"

#include <algorithm>
#include <cmath>

float BodyV4::usual(float hour)
{
    const double a = std::exp(-(24.0 - kDayEndH + kDayStartH) / kTauSleepH);
    const double b = std::exp(-(kDayEndH - kDayStartH) / kTauWakeH);
    const double evening = (1.0 - b) / (1.0 - a * b);
    const double morning = evening * a;
    if (hour >= kDayStartH && hour < kDayEndH)
        return static_cast<float>(1.0 - (1.0 - morning) * std::exp(-(hour - kDayStartH) / kTauWakeH));
    return static_cast<float>(evening * std::exp(-std::fmod(hour - kDayEndH + 24.0, 24.0) / kTauSleepH));
}

float BodyV4::load() const
{
    const double e = kWDeg * deg_ / kRefDegS + kWSay * say_ / kRefSayS + kWArousal * arousal_ / kRefArousal;
    return static_cast<float>(std::clamp(1.0 + kLoadGain * (e - 1.0), double{kLoadLo}, double{kLoadHi}));
}

void BodyV4::step(const In& in)
{
    if (in.clockOk && !clock_) s_ = usual(in.hour);
    clock_ = clock_ || in.clockOk;
    if (in.dtS <= 0.0f) return;
    const double dt = in.dtS;
    const double k = 1.0 - std::exp(-dt / kEffortTauS);
    deg_ += (std::min(in.movedDeg, kDegTickMax) / dt - deg_) * k;
    say_ += ((in.said ? 1.0 / dt : 0.0) - say_) * k;
    arousal_ += (in.arousal - arousal_) * k;
    if (in.asleep)
        s_ *= std::exp(-dt / (kTauSleepH * 3600.0));
    else
        s_ = 1.0 - (1.0 - s_) * std::exp(-dt * load() / (kTauWakeH * 3600.0));
}
