#pragma once
#include <cstdint>

#include "../hybrid/bus.h"
#include "../hybrid/policy_gru.h"

// The data collector (platform B, 2026-10-11): instead of the policy network, "motor babbling" - what an infant does
// to find out what its own actions do. The world model learns from the diary what follows each action; the old
// policy (an imitation of hand-written teachers) moved the head in the same few patterns, this covers the space.
//
//   head      a new target every 15 s on average by day: anywhere within +-60 deg, or (half the time) a small saccade
//             near where it looks; random speed and vigor; still in between (~130 deg of travel a minute)
//   voice     now and then a short call with random voice parameters (about once a minute by day)
//   face      a random face every 20-90 s; LEDs drift through colours
//   people    a face or a person in view: looks at it half the time; a touch is always answered at once, with a
//             random one of: call + happy face, only a call, only a happy face, a loving face - so the diary shows
//             which answer keeps people coming (the actor learns that from the reward). No nods (phantom touches).
//   bang      a clap or a knock (not its own servos) is answered at once (the owner's wish, 2026-10-18): a quick glance
//             aside and up, a surprised face, a short squeal; at night only the face
//   handling  a shake: a dizzy face and a warbling sound; shaken again within a minute: annoyed, then angry with a growl;
//             tilted or lifted: a scared face and a squeal (the owner shook it and saw nothing, 2026-10-18); at night
//             only the faces
//   night     21:30-06:30 (the owner sleeps next to it): silent, a small move every few minutes, sleepy face, dim LEDs
//
// It writes the same action vector as the policy, so everything after it - the executor's physical limits, the voice,
// the face, the diary - is unchanged. Deterministic for a given seed.
class Collector {
public:
    explicit Collector(uint32_t seed = 0x2545f491u) : rng_(seed | 1u) {}
    void step(const hybrid::Bus& bus, uint32_t nowMs, PolicyGru::Action& a);
    // Platform B: the actor's outputs of this tick (actor_net.h, 7 values) - then when to move, where, how fast and
    // when to call are sampled from them instead of the collector's timers; set before every step(), nullptr = none.
    void setActor(const float* out24)
    {
        actor_ = out24 != nullptr;
        if (out24)
            for (int i = 0; i < 24; ++i) actor_out_[i] = out24[i];
    }
    // What was decided in the last step from the actor's outputs (for the online learning, online_learner.h):
    // whether a move could start (and did), the sampled target, whether a call could start (and did).
    struct Decision {
        bool move_possible = false, moved = false, call_possible = false, called = false;
        float yaw = 0.0f, pitch = 0.0f;
    };
    const Decision& lastDecision() const { return decision_; }
    // The character of the voice for the next phrase (Emotion enum, face_style.h), -1 = the face being shown: half of
    // the calls sound like another mood (giggle, squeal, warble ...), so the voice is not only the face's.
    int voiceEmotion() const { return voice_emo_; }

private:
    uint32_t rnd();
    float uni(float lo, float hi);
    float expo(float meanS);  // seconds
    float gauss();
    int sampleLogits(const float* logits, int n);  // softmax sample
    bool night(const hybrid::Bus& bus) const;
    void newMove(const hybrid::Bus& bus, uint32_t now, bool small);
    void newVoice(uint32_t now, float lengthS, bool happy);
    int voice_emo_ = -1;
    bool banged_ = false, shaken_ = false, tilted_ = false;
    uint32_t bang_ms_ = 0, rough_ms_[3] = {};  // the last three shakes

    uint32_t rng_;
    bool started_ = false;
    // head
    float yaw_ = 0.0f, pitch_ = 0.0f, speed_ = 0.5f, vigor_ = 0.5f;
    uint32_t next_move_ms_ = 0, move_until_ms_ = 0;
    // voice
    uint32_t next_call_ms_ = 0, call_until_ms_ = 0;
    float v_arousal_ = 0.5f, v_valence_ = 0.0f, v_register_ = 0.5f, v_slope_ = 0.0f, v_expr_ = 0.5f, v_length_ = 0.5f,
          v_rate_ = 0.5f;
    int v_style_ = 0;
    // face, LEDs, blinks
    int emo_ = 0;
    uint32_t next_face_ms_ = 0, happy_until_ms_ = 0, next_blink_ms_ = 0;
    float led_[3] = {0.3f, 0.5f, 0.8f}, led_to_[3] = {0.3f, 0.5f, 0.8f};
    // people
    bool touched_ = false, saw_face_ = false;
    int answer_ = 0;  // the last touch's answer: 0 call + happy face, 1 call, 2 happy face, 3 loving face
    int happy_face_ = 0;
    // the actor (platform B)
    bool actor_ = false;
    float actor_out_[24] = {};
    uint32_t last_actor_move_ms_ = 0;
    Decision decision_;
};
