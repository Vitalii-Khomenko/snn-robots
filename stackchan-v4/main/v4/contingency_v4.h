#pragma once
#include <array>
#include <cstdint>

#include "v4_spec.h"

// The sense of "I did that" (docs/CONTINGENCY.md, step 1; tools/v4/contingency.py - its twin): does the world answer
// the robot's acts? Every tick it takes what the robot just did (a call, a gesture, a face change) and what came from
// outside (a gentle pet or tap, a new speech onset (the ear holds back the robot's own voice), someone coming near, motion seen while
// the head is still). An answer 0.5-4 s after an act is credited to the latest act; how often each kind of act is
// answered is set against the chance of an answer landing in such a window anyway (the background rate at moments
// without acts). Acts within 3 s after an answer are reactions and do not count; nothing counts at night or while the
// networks do not drive the body. Step 1 only writes its numbers to the diary (ct_*); nothing acts on them yet.
class ContingencyV4 {
public:
    enum Act : uint8_t { kCall, kGesture, kFace, kActs };
    enum Answer : uint8_t { kPet, kTap, kSpeech, kNear, kMotion, kAnswers };
    struct In {
        uint32_t ms = 0;
        bool active = false;  // brain v4 drives the body and it is not night
        bool said = false, gesture = false;
        int face = 0;
        float headEv = 0, screenEv = 0, touchHead = 0, touchScreen = 0, nearRaw = 0, speech = 0, motion = 0, headYaw = 0;
    };
    // an act just credited with an answer (for the log)
    struct Credit {
        bool any = false;
        Act act = kCall;
        Answer answer = kPet;
        uint32_t latencyMs = 0;
    };
    using Out = std::array<float, v4::CT_COUNT>;  // agency, answered, ignored, call, gesture, face

    void step(const In& in, Out& out, Credit& credit);
    static const char* actName(int k);
    static const char* answerName(int a);

private:
    struct Pending {
        uint32_t ms = 0;
        Act act = kCall;
        bool around = false, live = false;
        std::array<bool, kAnswers> answered{};
    };
    void finish(Pending& p);

    std::array<Pending, 8> pending_{};
    std::array<float, kActs> n_{};                          // acts (leaky, ~5 min)
    std::array<std::array<float, kAnswers>, kActs> h_{};    // answered acts per kind of answer
    std::array<float, kAnswers> bg_n_{};                    // answers at moments without acts (leaky, ~30 min)
    float bg_s_ = 0;                                        // seconds of such moments
    std::array<uint32_t, kAnswers> last_true_{};            // last tick each answer's condition held
    std::array<bool, kAnswers> seen_{};
    float pulse_ = 0, ignored_ = 0;
    uint32_t last_ms_ = 0, last_said_ms_ = 0, last_gesture_ms_ = 0, last_act_ms_ = 0;
    uint32_t last_answer_ms_ = 0, last_around_ms_ = 0;
    bool started_ = false, said_seen_ = false, gesture_seen_ = false, act_seen_ = false, answer_seen_ = false;
    bool around_seen_ = false;
    int face_ = -1;
    float yaw_ = 0;
};
