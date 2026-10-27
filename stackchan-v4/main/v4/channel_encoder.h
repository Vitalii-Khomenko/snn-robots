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
        // a fresh frame good for the bright spot only: taken while the head turned slowly (2026-10-26: the turns are
        // slow and long, the head moves ~40 % of the time; the motion sense sees the room slide by then, the spot is
        // as good as still). The diary keeps both in frame_ok: 1 frameOk, 0.5 lightOk only, 0 neither.
        bool lightOk = false;
        float lightX = 0, lightY = 0, lightStr = 0;  // the picture's brightest spot (-1..1) and its excess (0..1)
        float headYaw = 0;               // where the head points, degrees (MotionEngine): places the spot in the room
        float battery = 1, hour = 12;
        float stamina = 1;               // 1 - the sleep pressure (BodyV4; until 2026-10-27 the servos' travel)
        float budget = 1;                // the body budget (BodyV4, docs/BODY.md step 2)
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
    // The motion sense (2026-10-25, the owner: nothing in the room interested it): a person moving shows in a few
    // frames only (2026-10-24 evening: 3 % of the frames, median share 0.015, p90 0.135), and a 5 s mean x1.5 kept
    // the channel at ~0.01 (attention ~0.03). Now x3, rising in 0.5 s and falling in 4 s: someone moving about reads
    // 0.05-0.25 (interest), p99.9 of a day ~0.25.
    static constexpr float kMotionGain = 3.0f, kMotionTau = 0.5f, kMotionFallTau = 4.0f, kMotionHoldDecay = 2.0f;
    // The bright spot (a lamp, a window; VisionFrontEnd::findLight): how NEW it is - its excess over how bright that
    // place in the room has looked (kSpotPlaces directions kSpotPlaceDeg apart, head yaw + the spot's place in the
    // picture; learned over kSpotSlowTau while in view, forgotten over kSpotForgetTau), less what was already looked
    // at (habituation over kSpotHabTau). One memory for the whole picture (until 2026-10-25 evening) made a lamp
    // switched on elsewhere read old after a white cup; every new view now reads a little new, so the habituation
    // rate went 0.005 -> 0.001 (it saturated and cut a lamp to 0.3). A place keeps its usual PEAK (up in kSpotSlowTau,
    // down in kSpotFallTau) and novelty starts kSpotMargin above it: on the robot one lamp read 0.1..0.63 (p10..p90)
    // from glance to glance, and against the mean its brighter half was "new" 35 % of the time (2026-10-25 17:00).
    static constexpr float kSpotSlowTau = 4.0f, kSpotNovGain = 2.0f, kSpotHabTau = 180.0f, kSpotHabRate = 0.001f,
                           kSpotHabCut = 0.7f, kSpotCamHalfDeg = 30.0f, kSpotPlaceDeg = 15.0f, kSpotForgetTau = 900.0f,
                           kSpotFallTau = 120.0f, kSpotMargin = 0.15f;
    static constexpr int kSpotPlaces = 13;  // -90 .. +90 deg
    static constexpr float kLonelyTau = 2400.0f, kLonelyMotionCap = 300.0f, kLonelyStart = 600.0f;
    static constexpr float kContactSpeech = 0.5f, kContactNear = 0.3f, kContactMotion = 0.1f;
    // the sleep pressure above a usual day's at this hour -> tired (docs/BODY.md step 1; 0.25 * (1 - stamina) before)
    static constexpr float kTiredFelt = 2.0f;
    // the body budget below this is felt as hunger (0.35 -> 0.5: valence lower; 0.17 -> 0.75: unwell; docs/BODY.md step 2)
    static constexpr float kBudgetFelt = 0.7f;

    static float nightRamp(float hour);

private:
    double loud_ = 0, sharp_ = 0, voice_ = 0, head_ = 0, screen_ = 0, shake_ = 0;
    double n_ = 0, p0_ = 0, r0_ = 0, steady_ = 0, tilt_ = 0;
    double near_ = 0, light_ = 0, motion_ = 0, lonely_t_ = kLonelyStart, mx_ = 0, my_ = 0;
    double lx_ = 0, ly_ = 0, spot_ = 0, spot_hab_ = 0;
    std::array<double, kSpotPlaces> place_{};  // how bright each direction of the room has looked
};
