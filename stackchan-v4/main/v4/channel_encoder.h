#pragma once
#include <array>
#include <cstdint>

#include "v4_spec.h"

// Brain v4's senses: the 13 channels 0..1 and the 2 "where" values the spiking analyzer reads
// (BRAIN_V4_INPUT_SCALES.txt). Deterministic code, no learning: normalizing, smoothing over each channel's window,
// the body clocks the spiking network cannot keep (minutes alone, night), the masks for our own noise (the inputs
// arrive already masked). Twin of tools/v4/channels.py - keep the two identical (tools/v4/parity_channels.py).
class ChannelEncoder {
public:
    struct Raw {
        float energy = 0, speech = 0;    // AuditoryNet: loudness over the room's floor, speech probability
        float sharpEv = 0;               // a sharp sound started this tick: strength 0.5..1.5, else 0
        float headEv = 0;                // a head touch started this tick: weight by pad level, else 0
        float headHold = 0;              // 1 while a finger stays on the head pad
        float screenEv = 0;              // a screen touch started this tick
        float accelSd = 0;               // sd of |acceleration| over the last 0.5 s, g
        float imuPitch = 0, imuRoll = 0; // gravity direction (BrainLoop: pitch /(pi/2), roll /pi)
        bool imuOk = false;
        float nearRaw = 0;               // proximity over its slow baseline, counts
        bool nearOk = false;
        float als = 0;                   // ambient light, raw counts
        bool alsOk = false;
        float motionFrac = 0, motionX = 0, motionY = 0;  // the picture's own movement and its centre
        bool frameOk = false;            // a fresh usable frame this tick (not blind, not during our own motion)
        float battery = 1, stamina = 1, hour = 12;
        bool clockOk = false;
    };
    using Out = std::array<float, v4::kSnnInputs>;

    void reset() { *this = ChannelEncoder(); }
    // One 50 ms brain tick.
    void update(const Raw& r, Out& out);

    // ---- constants (mirrored in tools/v4/channels.py)
    static constexpr float kDt = 0.05f;
    static constexpr float kLoudTau = 10.0f, kLoudGain = 1.6f;
    // kSharpFull 3 -> 6 (2026-10-25, the owner: twice as tolerable): a clap (1) read 0.33, two claps in 15 s or one
    // bang (up to 6) were the anger zone; now the same takes twice the sound. The zone warp leaves sharp alone.
    static constexpr float kSharpTau = 15.0f, kSharpFull = 6.0f;
    static constexpr float kVoiceTau = 60.0f;
    static constexpr float kTouchTau = 30.0f, kHeadK = 18.6f, kScreenK = 9.5f, kHoldRate = 0.5f;
    static constexpr float kShakeFloor = 0.03f, kShakeFull = 0.8f, kShakeDecay = 3.0f;
    static constexpr float kTiltFitTau = 1800.0f, kTiltTau = 0.5f, kTiltWarm = 1200.0f, kTiltRollBand = 0.18f,
                           kTiltAcceptS = 600.0f;
    static constexpr float kNearFull = 600.0f, kNearTau = 1.0f;
    static constexpr float kLightFull = 20000.0f, kLightTau = 2.0f;
    static constexpr float kMotionGain = 1.5f, kMotionTau = 5.0f, kMotionHoldDecay = 2.0f;
    static constexpr float kLonelyTau = 2400.0f, kLonelyMotionCap = 300.0f, kLonelyStart = 600.0f;
    static constexpr float kContactSpeech = 0.5f, kContactNear = 0.3f, kContactMotion = 0.1f;

    static float nightRamp(float hour);

private:
    double loud_ = 0, sharp_ = 0, voice_ = 0, head_ = 0, screen_ = 0, shake_ = 0;
    double n_ = 0, p0_ = 0, r0_ = 0, steady_ = 0, tilt_ = 0;
    double near_ = 0, light_ = 0, motion_ = 0, lonely_t_ = kLonelyStart, mx_ = 0, my_ = 0;
};
