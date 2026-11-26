#include "memory_v4.h"

#include <algorithm>
#include <cmath>

void MemoryV4::set(const float* table)
{
    n_ = 0;
    day_recalled_.fill(-1);
    if (!table) return;
    for (int i = 0; i < kMax; ++i) {
        const float* r = table + i * kFields;
        Episode e{r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]};
        bool ok = true;
        for (int k = 0; k < kFields; ++k) ok = ok && std::isfinite(r[k]);
        if (!ok || e.strength < kMinStrength || e.minute < 0.0f || e.minute >= 1440.0f) continue;
        ep_[n_++] = e;
    }
}

int MemoryV4::candidate(uint32_t nowMs, bool clockOk, float hour, bool lit, bool around) const
{
    if (!clockOk || n_ == 0) return -1;
    if (recalled_once_ && nowMs - last_recall_ms_ < kGapMs) return -1;
    const int day = day_;
    const float minute = hour * 60.0f;
    int best = -1;
    for (int i = 0; i < n_; ++i) {
        const Episode& e = ep_[i];
        if (day_recalled_[i] == day) continue;
        float d = std::fabs(minute - e.minute);
        d = std::fmin(d, 1440.0f - d);
        if (d > kWindowMin || (e.lit > 0.5f) != lit || (e.around > 0.5f) != around) continue;
        if (best < 0 || e.strength > ep_[best].strength) best = i;
    }
    return best;
}

float MemoryV4::expectation(bool clockOk, float hour, bool lit, float& bestYawDeg) const
{
    bestYawDeg = 0.0f;
    if (!clockOk || n_ == 0) return 0.0f;
    const float minute = hour * 60.0f;
    float best = 0.0f;
    for (int i = 0; i < n_; ++i) {
        const Episode& e = ep_[i];
        if (e.valence <= 0.0f || (e.lit > 0.5f) != lit) continue;
        // Signed minutes ahead: positive when episode is upcoming (0..30 min ahead), small tail up to 10 min after.
        float ahead = e.minute - minute;
        if (ahead < -720.0f) ahead += 1440.0f;
        if (ahead > 720.0f) ahead -= 1440.0f;
        if (ahead < -10.0f || ahead > 30.0f) continue;
        // Peak anticipation at ~5 min ahead (sigma 14 min ahead, 6 min after)
        const float sigma = ahead >= 0.0f ? 14.0f : 6.0f;
        const float z = (ahead - 5.0f) / sigma;
        const float score = std::clamp(e.strength * std::exp(-0.5f * z * z), 0.0f, 1.0f);
        if (score > best) {
            best = score;
            bestYawDeg = e.yawDeg;
        }
    }
    return best;
}

void MemoryV4::clock(float hour)
{
    if (last_hour_ >= 0.0f && hour < last_hour_ - 12.0f) ++day_;
    last_hour_ = hour;
}

void MemoryV4::recalled(int index, uint32_t nowMs)
{
    if (index < 0 || index >= n_) return;
    day_recalled_[index] = day_;
    last_recall_ms_ = nowMs;
    recalled_once_ = true;
}

const char* MemoryV4::kindName(int kind)
{
    // tools/v4/episodes.py KINDS
    static const char* const names[] = {"pet", "tap", "talk", "near", "shake", "tilt", "poke", "fright"};
    return kind >= 0 && kind < static_cast<int>(sizeof(names) / sizeof(names[0])) ? names[kind] : "moment";
}
