#pragma once
#include <array>
#include <cstdint>

#include "hybrid_spec.h"
#include "policy_gru.h"

// The world model (platform B stage 1): every tick it predicts what the robot will sense in the next one - vision,
// hearing, body - from what it senses now and what it does now; the next tick shows how wrong it was.
//
//   154 inputs (the policy's input, its action, the efference copy)
//     -> dense 64 -> GRU 64 -> dense 48 -> 49 predicted changes of the sensed values
//
// The policy's shape and kernels (INT8 weights, PolicyGru::Params), trained on the diary by
// tools/life/train_world.py. The error per group (RMS) and its fall - the learning progress, the plan's curiosity:
// slow average of the error minus fast average, when positive - go to the diary; the motivation turns a surprise into
// curiosity (motivation.h).
class WorldModel {
public:
    static constexpr int kIn = hybrid::kPolicyInputs + hybrid::ACT_COUNT + hybrid::EFF_COUNT;
    static constexpr int kOut = hybrid::VIS_COUNT + hybrid::AUD_COUNT + hybrid::BODY_COUNT;  // the first 49 inputs
    // the diary's values: RMS error of vision, audio, body; learning progress x 10
    static constexpr int kLogValues = 4;

    static PolicyGru::Params compiledParams();
    static constexpr int kParamCount = PolicyGru::kParamCount;
    static const PolicyGru::ParamInfo* paramInfo();

    WorldModel() : p_(compiledParams()) {}
    void use(const PolicyGru::Params& p)
    {
        p_ = p;
        h_.fill(0.0f);
        have_pred_ = false;
        warmup_ = 0;
        lp_ = 0.0f;
    }
    bool trained() const;
    // One tick, after the policy acted and the executor ran: x = the diary's first kIn values of this tick (policy
    // input, action, efference copy - exactly what train_world.py learned from). Checks the last prediction against
    // what is sensed now, then predicts the next.
    void step(const float* x);
    float learningProgress() const { return lp_; }
    void logValues(float* out) const;
    uint32_t nanCount() const { return nan_count_; }

private:
    PolicyGru::Params p_;
    std::array<float, 64> h_{}, d_{};
    std::array<float, 48> f_{};
    std::array<float, 6 * 64> scratch_{};
    std::array<float, kOut> pred_{};
    bool have_pred_ = false;
    std::array<float, 3> rms_{};
    float e_fast_ = 0.0f, e_slow_ = 0.0f, lp_ = 0.0f;
    int warmup_ = 0;
    uint32_t nan_count_ = 0;
};
