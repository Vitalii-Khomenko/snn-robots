#pragma once
#include <array>
#include <cstdint>

#include "policy_gru.h"

// The actor of platform B (stage 3): trained on the robot's own diary by advantage-weighted regression
// (tools/life/train_life_actor.py) - the collector's random decisions, weighted by how they turned out.
//
//   70 inputs (what the robot senses and feels: vision, hearing, body, time, prediction errors, learned expectations,
//   the five needs, the world model's errors) -> dense 64 -> GRU 64 -> dense 48 -> 24:
//   move logit, yaw mean and log-std, pitch mean and log-std, speed, call logit, the value of the situation,
//   then 12 face logits (collector.cpp kDayFaces order) and 4 voice style logits (calm, ask, play, grumble)
//
// The policy's shape and kernels (INT8, PolicyGru::Params); compiled-in weights in actor_model_data.h, the nightly
// ones from the weights blob (actor.*). The collector samples its decisions from these outputs (collector.h).
class ActorNet {
public:
    static constexpr int kIn = 71, kOut = 24;  // 71: + the expectation of being talked to (2026-10-17)
    static constexpr int kFaces = 12, kStyles = 4, kFaceAt = 8, kStyleAt = 20;  // expression: logits (2026-10-17)
    enum Out : uint8_t { Move, YawMu, YawLogSig, PitchMu, PitchLogSig, Speed, Call, Value };
    static PolicyGru::Params compiledParams();
    static constexpr int kParamCount = PolicyGru::kParamCount;
    static const PolicyGru::ParamInfo* paramInfo();

    ActorNet() : p_(compiledParams()) {}
    void use(const PolicyGru::Params& p)
    {
        p_ = p;
        h_.fill(0.0f);
    }
    // false (out unchanged) on a non-finite result; the memory restarts then
    bool step(const float* in, float* out);
    // the hidden layer of the last step (48, after ReLU): what the online learning works on (online_learner.h)
    const float* features() const { return f_.data(); }
    uint32_t nanCount() const { return nan_count_; }

private:
    PolicyGru::Params p_;
    std::array<float, 64> h_{}, d_{};
    std::array<float, 48> f_{};
    std::array<float, 6 * 64> scratch_{};
    uint32_t nan_count_ = 0;
};
