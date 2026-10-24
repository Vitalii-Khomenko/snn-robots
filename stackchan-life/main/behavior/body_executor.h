#pragma once
#include <array>
#include <cstdint>

#include "../hybrid/body_sense.h"
#include "../hybrid/hybrid_brain.h"
#include "../led/led_engine.h"
#include "../motion/motion_engine.h"
#include "../ui/face_engine.h"
#include "../voice/babble_voice.h"

// The spinal cord and brainstem under the hybrid brain - not a behavior planner.
//
// It never decides WHAT to do. It reads the policy's action vector and
// executes it inside physical limits:
//   motor  - head yaw/pitch intent (-1..1) -> degrees inside the joint and
//            safety window, speed from the policy's speed/vigor scaled by
//            stamina, a minimum gap between servo commands, no jump bigger
//            than kMaxStepDeg. Two motor primitives the policy can trigger
//            (like central pattern generators): an inviting nod series and a
//            startle flinch. No songs: the policy has no output for it any more.
//   voice  - BabbleVoice: the policy's sound intent -> BabbleNet (syllables)
//            -> formant synthesizer -> speaker; a token bucket caps how often
//            a call may start.
//   face   - face valence/arousal/lids pick the closest pixel face
//            (debounced), gaze and blink outputs move the eyes.
//   LEDs   - the policy's colour (led_r/g/b) and liveliness (led_activity), breathing
//            and heartbeat from the virtual physiology.
class BodyExecutor {
public:
    // Platform B (mode life): drop the behaviour rules (call budget, invitation spacing); keep the physical limits
    void setFree(bool free)
    {
        free_ = free;
        vocal_.setWide(free);
    }
    // Mode life: the character of the next phrase (Emotion enum), -1 = the face being shown (the collector's choice)
    void setVoiceEmotion(int emotion) { voice_emotion_ = emotion; }
    BodyExecutor(FaceEngine& face, LedEngine& led, BabbleVoice& vocal, MotionEngine& motion)
        : face_(face), led_(led), vocal_(vocal), motion_(motion)
    {}
    void apply(HybridBrain& brain, BodySense& sense, uint32_t nowMs);

    uint32_t moves() const { return moves_; }
    uint32_t calls() const { return calls_; }
    uint32_t nods() const { return nod_series_; }
    uint32_t startles() const { return startles_; }
    bool voiceTrained() const { return vocal_.trained(); }

private:
    void applyVoice(HybridBrain& brain, BodySense& sense, const PolicyGru::Action& a, uint32_t now);
    void applyMotor(HybridBrain& brain, BodySense& sense, const PolicyGru::Action& a, uint32_t now);
    void applyFace(const hybrid::Bus& bus, const PolicyGru::Action& a, uint32_t now);
    void applyLeds(HybridBrain& brain);
    void command(BodySense& sense, float yaw, float pitch, float speed);

    FaceEngine& face_;
    LedEngine& led_;
    BabbleVoice& vocal_;
    MotionEngine& motion_;

    // ---- limits (mechanics and safety, not choices)
    static constexpr float kYawLimitDeg = 100.0f;      // the executor's window (joint: +-128)
    static constexpr float kPitchMinDeg = -20.0f;
    static constexpr float kPitchMaxDeg = 35.0f;
    static constexpr float kMaxStepDeg = 30.0f;        // no single command moves further than this
    // (70 until 2026-10-13: the servos hung after 14-17 % of the steps over 30 deg, 2-7 % of the smaller ones)
    static constexpr uint32_t kMinCommandGapMs = 120;  // servo bus: <= ~8 packets/s
    static constexpr float kMinSpeedDegS = 12.0f;
    static constexpr float kMaxSpeedDegS = 110.0f;
    static constexpr float kYawDeadDeg = 3.0f, kPitchDeadDeg = 2.5f;

    // ---- motor
    bool base_valid_ = false;
    float base_yaw_ = 0, base_pitch_ = 0;               // last commanded target
    uint32_t last_command_ms_ = 0;
    uint32_t moves_ = 0;
    std::array<std::array<float, 2>, 6> goal_hist_{};  // the network's head target over the last 6 ticks (saccade generator)
    size_t goal_n_ = 0;
    // primitives
    bool prev_invite_ = false, prev_startle_ = false;
    uint32_t nod_start_ms_ = 0, flinch_start_ms_ = 0, nod_series_ = 0, startles_ = 0;
    bool nodding_ = false, flinching_ = false;
    float nod_base_pitch_ = 0, flinch_base_pitch_ = 0;
    static constexpr float kNodDegS = 160.0f;

    // ---- voice
    float voice_tokens_ = 3.0f;
    bool free_ = false;          // mode life: only physical limits (setFree)
    int voice_emotion_ = -1, voice_character_ = 0;
    float talk_share_ = 0.0f;    // the voice's share of the last minute
    int reaction_ticks_ = 0;  // ticks left in which a call still counts as the answer to an event
    uint32_t last_token_ms_ = 0, calls_ = 0;
    static constexpr float kTokenCapacity = 5.0f;
    static constexpr float kTokenRefillMs = 10000.0f;  // <= 6 calls per minute on average (bursts of 4 for touch replies)

    // ---- face
    bool person_shown_ = false;
    uint32_t last_invite_ms_ = 0;
    Emotion shown_ = Emotion::Normal, candidate_ = Emotion::Normal;
    int candidate_ticks_ = 0;
    uint32_t face_since_ms_ = 0;
    bool prev_blink_ = false;
};
