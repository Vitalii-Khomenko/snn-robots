#pragma once
#include <algorithm>
#include <array>

#include "v4_spec.h"

// Zones shifted by experience (stage 6, BRAIN_V4_PLAN.txt 8.4): for each channel one shift s of its "alarm border" a
// (the lower border of its first anxiety or anger zone). s > 0: habituated - a level that used to start the alarm is
// now ordinary (the fridge that always hums); s < 0: sensitized. |s| <= v4::kZoneShiftMax (0.15), learned slowly by
// the PC's night sleep (tools/v4/sleep_v4.py), kept in the learned blob.
//
// Applied as a monotone warp of the channel's value before the spiking analyzer, so the analyzer's innate zone wiring
// stays as trained: tolerant(c, a) + s maps onto a, 0 and 1 stay. tools/v4/learned.py warp() is the same function.
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

    // Innate tolerance (2026-10-25, the owner: "it gets cross too fast at sounds and touches - every input twice as
    // tolerable"): the raw level that reads as the alarm border a is min(2a, a + 0.6 (1 - a)) - the calm range below
    // the alarm twice as wide where the scale allows it (sharp 0.35 -> 0.70, touch_screen 0.40 -> 0.76, shake
    // 0.20 -> 0.40, loud and touch_head 0.70 -> 0.88). The learned shift moves this border. Sharp sounds are not
    // warped: their channel saturated at one bang, so their tolerance is a twice wider scale in the encoder
    // (channel_encoder.h kSharpFull). Hunger is not warped either (2026-10-27, BODY step 2): it is the body's own need,
    // not something the world does to the robot; warped, a body budget of 0.25 read 0.41 instead of 0.64.
    static constexpr float kTolerance = 2.0f;
    static float tolerant(int c, float a)
    {
        return c == v4::CH_SHARP || c == v4::CH_HUNGER ? a : std::min(kTolerance * a, a + 0.6f * (1.0f - a));
    }

    // the raw level m reads as a; 0 and 1 stay, linear in between
    static float warp(float x, float a, float m)
    {
        m = std::clamp(m, 0.02f, 0.98f);
        if (a <= 0.0f || m == a) return x;
        return x <= m ? x * (a / m) : a + (x - m) * ((1.0f - a) / (1.0f - m));
    }

    template <typename Channels>
    void apply(Channels& ch) const
    {
        for (int c = 0; c < v4::CH_COUNT; ++c) {
            const float a = alarm(c);
            if (a > 0.0f) ch[c] = warp(ch[c], a, tolerant(c, a) + shift[c]);
        }
    }
};
