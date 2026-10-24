#pragma once
#include <array>
#include <cstdint>

#include "bus.h"

// What the robot wants (platform B, stage 2): five needs and the reward of each 50 ms tick. Nothing here says what
// to do; it says what feels good and what does not (the plan's "Motivation and intrinsic reward").
//
// Every need d is 0..1 (0 = satisfied), grows by itself and falls with what satisfies it. Only raw senses are used
// (perception, body, context) - not the spiking core's readouts, which were trained to copy a hand-written teacher.
//
//   energy       the battery: empty at 10 %, satisfied above 80 % (charging lowers it)
//   social       grows in ~30 min without anybody (slower at night); a touch, being talked to, a face in view
//   stimulation  grows in ~5 min when nothing new happens; novelty (until the world model of stage 1 gives the
//                learning progress: then that is what satisfies it)
//   rest         the servos' fatigue (MotionEngine stamina) and the night
//   comfort      shaking, lifting, tilting, very loud sounds; fades in ~1 min
//
// The reward (homeostatic RL, Keramati & Gutkin 2014): r = sum_i w_i (D_i(t-1) - D_i(t)) + beta LP - c |dtheta|,
// D = d^2 (a strong need speaks louder than a weak one). In mode life the actor learns from it (the night: AWR on the
// diary; by day: the online learner).
class Motivation {
public:
    enum Need : uint8_t { Energy, Social, Stimulation, Rest, Comfort, kNeeds };
    static const char* needName(int i);
    // the values the diary records each tick (main/net/life_log.h, tools/life/lifedata.py), in this order:
    // the five needs, the reward, the novelty signal, the head movement (degrees / 100)
    static constexpr int kLogValues = kNeeds + 3;

    void tick(const hybrid::Bus& bus);
    // Curiosity (platform B): the world model's error of the last tick (RMS, vision and hearing; world_model.h). A
    // surprise - an error well above its habitual level - eases the need for stimulation like novelty does.
    void setWorldErrors(float rmsVision, float rmsAudio)
    {
        wm_vision_ = rmsVision;
        wm_audio_ = rmsAudio;
    }
    float curiosity() const { return curiosity_; }
    float need(int i) const { return d_[i]; }
    float reward() const { return r_; }
    void logValues(float* out) const;
    // the needs that carry over a reboot (RobotRuntime keeps them in NVS); energy and rest follow the body anyway
    const std::array<float, kNeeds>& needs() const { return d_; }
    void restore(const std::array<float, kNeeds>& d);

private:
    std::array<float, kNeeds> d_{0.2f, 0.3f, 0.3f, 0.0f, 0.0f};
    float r_ = 0.0f, novelty_ = 0.0f, move_deg_ = 0.0f;
    float wm_vision_ = 0.0f, wm_audio_ = 0.0f, hab_vision_ = 0.0f, hab_audio_ = 0.0f, curiosity_ = 0.0f;
    float prev_yaw_ = 0.0f, prev_pitch_ = 0.0f;
    bool started_ = false;
};
