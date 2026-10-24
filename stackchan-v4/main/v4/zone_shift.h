#pragma once
#include <array>

#include "v4_spec.h"

// Zones shifted by experience (stage 6, BRAIN_V4_PLAN.txt 8.4): for each channel one shift s of its "alarm border" a
// (the lower border of its first anxiety or anger zone). s > 0: habituated - a level that used to start the alarm is
// now ordinary (the fridge that always hums); s < 0: sensitized. |s| <= v4::kZoneShiftMax (0.15), learned slowly by
// the PC's night sleep (tools/v4/sleep_v4.py), kept in the learned blob.
//
// Applied as a monotone warp of the channel's value before the spiking analyzer, so the analyzer's innate zone wiring
// stays as trained: a + s maps onto a, 0 and 1 stay. tools/v4/zone_shift.py is the same function.
struct ZoneShift {
    std::array<float, v4::CH_COUNT> shift{};

    // the alarm border of channel c, or -1 when the channel has neither an anxiety nor an anger zone
    static float alarm(int c)
    {
        float a = 2.0f;
        for (int z : {v4::ZONE_ANXIETY, v4::ZONE_ANGER}) {
            const auto& b = v4::kZones[c][z];
            if (b.lo <= b.hi && b.lo < a) a = b.lo;
        }
        return a <= 1.0f ? a : -1.0f;
    }

    static float warp(float x, float a, float s)
    {
        const float m = a + s;  // the raw level that now reads as the border
        if (a <= 0.0f || m <= 0.01f || m >= 0.99f || s == 0.0f) return x;
        return x <= m ? x * (a / m) : a + (x - m) * ((1.0f - a) / (1.0f - m));
    }

    template <typename Channels>
    void apply(Channels& ch) const
    {
        for (int c = 0; c < v4::CH_COUNT; ++c) {
            if (shift[c] == 0.0f) continue;
            const float a = alarm(c);
            if (a > 0.0f) ch[c] = warp(ch[c], a, shift[c]);
        }
    }
};
