#pragma once

// The body and its needs (docs/BODY.md, step 1; tools/v4/body.py - its twin): sleep pressure, Borbely's process S.
// Awake, it builds up towards 1 (kTauWakeH), faster the more the robot does: its head's travel, its calls and its
// arousal over the last ~10 minutes make the load (1 on a usual day, 2026-10-26); asleep (the owner's night, or a nap:
// the sleepy face shown) it drains (kTauSleepH). The clock (process C) holds the usual day's pressure off: the tired
// channel feels only the pressure above the usual curve at that hour (ChannelEncoder, usual()). A busy day makes the
// robot tired and a dark room then puts it to sleep; a usual day is as before; a nap or the night clears it.
// A reboot keeps nothing: the pressure starts on the usual curve once the clock is known.
class BodyV4 {
public:
    struct In {
        float dtS = 0;           // since the last tick
        bool clockOk = false;
        float hour = 12;
        float movedDeg = 0;      // yaw + pitch travel this tick
        bool said = false;       // a call started this tick
        float arousal = 0;       // the analyzer's
        bool asleep = false;     // night rules, or the sleepy face shown
    };

    void step(const In& in);
    float pressure() const { return static_cast<float>(s_); }
    float load() const;
    // the pressure of a usual day (load 1) at this hour
    static float usual(float hour);

    static constexpr float kTauWakeH = 12.0f, kTauSleepH = 3.0f, kEffortTauS = 600.0f;
    static constexpr float kRefDegS = 2.5f, kRefSayS = 4.0f / 60.0f, kRefArousal = 0.15f;
    static constexpr float kWDeg = 0.4f, kWSay = 0.3f, kWArousal = 0.3f;
    static constexpr float kLoadGain = 2.0f, kLoadLo = 0.25f, kLoadHi = 3.0f;
    static constexpr float kDayStartH = 6.0f, kDayEndH = 21.75f, kDegTickMax = 20.0f;

private:
    // double: a 50 ms step changes the pressure by ~1e-6, below a float's resolution near 1
    double s_ = 0.3;
    bool clock_ = false;
    double deg_ = kRefDegS, say_ = kRefSayS, arousal_ = kRefArousal;  // a usual load until it knows better
};
