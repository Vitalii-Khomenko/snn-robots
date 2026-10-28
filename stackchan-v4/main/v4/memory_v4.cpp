#include "memory_v4.h"

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
