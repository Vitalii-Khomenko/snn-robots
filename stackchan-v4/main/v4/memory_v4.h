#pragma once
#include <array>
#include <cstdint>

// The robot's own story (docs/EPISODES.md, 2026-10-28; the owner: "a short glance and a special face; it remembers the
// bad moments too"). The night (tools/v4/sleep_v4.py, episodes.py) picks the most vivid moments of the last days - a
// stroke, being shaken - with their context, and sends them in the learned blob ("mem.episodes"). By day, when the
// present is like a remembered moment - the same time of day (within kWindowMin), the room lit or dark as then,
// someone around or not as then - the robot remembers it: the executor turns the head for a moment to where it
// happened and shows the remembering face (good: a heart in a thought bubble; bad: a little storm). Each moment is
// remembered at most once a day, one at most every kGapMs.
class MemoryV4 {
public:
    // one remembered moment, as the blob has it: kFields floats
    struct Episode {
        float minute = 0;    // of the day, 0..1440
        float valence = 0;   // +1 good, -1 bad
        float strength = 0;  // 0..1 (the night's choice: how vivid)
        float yawDeg = 0;    // where the head pointed
        float around = 0;    // someone was around (1) or not (0)
        float lit = 0;       // the room was lit (1) or dark (0)
        float kind = 0;      // what it was (episodes.py KINDS)
        float date = 0;      // yymmdd
    };
    static constexpr int kMax = 24, kFields = 8;
    static constexpr float kWindowMin = 20.0f, kMinStrength = 0.3f;
    static constexpr uint32_t kGapMs = 600000;

    // the night's table (kMax x kFields floats, unused rows strength 0); nullptr: no memories
    void set(const float* table);
    int count() const { return n_; }
    // Every brain tick (with the clock): counts the days, past midnight.
    void clock(float hour);
    // By day: the moment to remember now (its index), or -1. Nothing at night or without a clock.
    int candidate(uint32_t nowMs, bool clockOk, float hour, bool lit, bool around) const;
    // the executor remembered it (it may not: a gesture, a reaction, something new in view come first)
    void recalled(int index, uint32_t nowMs);
    const Episode& episode(int index) const { return ep_[index]; }
    static const char* kindName(int kind);

private:
    std::array<Episode, kMax> ep_{};
    std::array<int, kMax> day_recalled_{};  // the day number it was last remembered (-1: never)
    int n_ = 0;
    int day_ = 0;            // counts the robot's days (the hour going back past midnight)
    float last_hour_ = -1.0f;
    uint32_t last_recall_ms_ = 0;
    bool recalled_once_ = false;
};
