#pragma once
#include <cstdint>

#include "../led/led_engine.h"
#include "../motion/motion_engine.h"
#include "../ui/face_engine.h"
#include "../voice/babble_voice.h"
#include "brain_v4.h"

// Brain v4's "muscles" (BRAIN_V4_PLAN.txt section 7): the main network decides WHAT (emotion, voice, LEDs, head
// mode); this only does HOW, inside physical and safety limits:
//   face   - the chosen emotion (a face stays at least 1.5 s so its morph can be seen); the eyes follow attention
//   voice  - BabbleVoice with the network's style, mood, register, length, rate; the call budget (~6 a minute,
//            reactions may spend the last tokens); SILENT at night 21:45-06:00 (the owner sleeps next to the robot)
//   LEDs   - colour, pattern (steady / breathe / pulse / flicker / wave), brightness, rate; dim at night
//   head   - gaze targets and the gesture library: nod "yes" (pitch), shake "no" (yaw), flinch, look down / up,
//            doze. Small amplitudes, at most one nod / shake every 4 s (current peaks: the AXP power-offs, 2026-10-23);
//            no nods, shakes or flinches at night.
class V4Executor {
public:
    V4Executor(FaceEngine& face, LedEngine& led, BabbleVoice& vocal, MotionEngine& motion)
        : face_(face), led_(led), vocal_(vocal), motion_(motion)
    {}
    // Every brain tick (20 Hz) with the latest decision; `decided` when the main network ran this tick.
    // `night`: the owner's sleep window, or the clock not synced yet (it might be night: quiet until it is known). Fills `done` with what really happened (BrainV4::Executed).
    void apply(const BrainV4& brain, bool decided, bool night, uint32_t nowMs, BrainV4::Executed& done);

    uint32_t calls() const { return calls_; }
    uint32_t gestures() const { return gestures_; }
    uint32_t nods() const { return nods_; }
    uint32_t shakes() const { return shakes_; }
    int shownEmotion() const { return static_cast<int>(shown_); }

    static constexpr float kYawLimitDeg = 100.0f, kPitchMinDeg = -20.0f, kPitchMaxDeg = 35.0f;
    static constexpr uint32_t kMinCommandGapMs = 120;  // servo bus: <= ~8 packets/s (gesture keyframes)
    // Gaze moves (look, scan, look up / down, doze): at most one every 2.5 s, and only for a visible change. The servos
    // hang more often the more packets they get (docs/HARDWARE.md); with every small gaze correction sent, the lively
    // network of 2026-10-24 sent 45-80 packets a minute, the servos hung 60-100 times an hour and the head barely moved.
    // 2026-10-25: every mood moves its head now (teacher_nn.py MANNER) and at 2.5 s / 8 deg the robot sent 50-120
    // packets a minute and the servos hung within a minute of every power cycle: one move per 4 s, >= 12 / 8 deg.
    static constexpr uint32_t kGazeGapMs = 4000;
    static constexpr float kGazeDeadYawDeg = 12.0f, kGazeDeadPitchDeg = 8.0f;
    static constexpr uint32_t kGestureGapMs = 4000;    // nod / shake at most every 4 s
    static constexpr float kMaxSpeedDegS = 110.0f;

private:
    enum class Gesture : uint8_t { None, Nod, Shake, Flinch };
    void applyFace(const BrainV4& brain, uint32_t now);
    void applyVoice(const BrainV4& brain, bool night, uint32_t now, BrainV4::Executed& done);
    void applyLeds(const BrainV4& brain, bool night);
    void applyHead(const BrainV4& brain, bool night, uint32_t now, BrainV4::Executed& done);
    bool command(float yaw, float pitch, float speed, uint32_t now, bool force = false);

    FaceEngine& face_;
    LedEngine& led_;
    BabbleVoice& vocal_;
    MotionEngine& motion_;

    // face
    Emotion shown_ = Emotion::Normal, candidate_ = Emotion::Normal;
    int candidate_ticks_ = 0;
    uint32_t face_since_ms_ = 0;
    // voice
    float tokens_ = 3.0f;
    uint32_t last_token_ms_ = 0, calls_ = 0;
    int reaction_ticks_ = 0;
    float prev_touch_ = 0.0f;
    // head
    Gesture gesture_ = Gesture::None;
    uint32_t gesture_start_ms_ = 0, last_gesture_ms_ = 0, last_command_ms_ = 0, gestures_ = 0, nods_ = 0, shakes_ = 0;
    int gesture_step_ = -1, gesture_reps_ = 2;
    float gesture_amp_ = 8.0f, gesture_half_s_ = 0.4f, base_yaw_ = 0.0f, base_pitch_ = 5.0f;
    float sent_yaw_ = 0.0f, sent_pitch_ = 5.0f;
    bool sent_valid_ = false;
    int prev_mode_ = 0;
    int head_cand_ = 0, head_cand_ticks_ = 0, head_mode_ = 0;  // the head mode, debounced
};
