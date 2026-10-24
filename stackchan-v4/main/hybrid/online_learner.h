#pragma once
#include <array>
#include <cstdint>

#include "actor_net.h"

// Stage 4 of platform B: the robot learns during the day, on the robot, from what happens to it.
//
// The actor's weights in flash do not change (the night sleep on the PC retrains the whole network from the diary).
// On top of its output layer the robot keeps a small plastic layer in RAM: a correction dW (8 x 48) of the actor's
// outputs from its hidden layer f (actor_net.h). Actor-critic with eligibility traces, the plan's three-factor rule:
//
//   value      V = out[Value] + dW[Value] . f                (the actor's own estimate, corrected)
//   TD error   delta = r / scale + gamma V(t) - V(t-1)        (the "dopamine": better or worse than expected)
//   critic     e_v = gamma lambda e_v + f(t-1);   dW[Value] += alpha_v delta e_v
//   actor      e_a = gamma lambda e_a + grad log pi(decision | f);   dW[Move, Call, YawMu, PitchMu] += alpha_a delta e_a
//
// grad log pi: a Bernoulli decision (start a move, start a call) - (taken - p); a sampled target - (a - mu) / sigma^2
// (times the tanh slope). Only ticks where a decision was possible count. Each correction is clamped (|dW| <= 1 in
// logit units) so a bad day cannot run away. The corrections are kept in NVS (every 10 min) and dropped when new
// nightly weights arrive: the night has learned the same day from the diary.
class OnlineLearner {
public:
    static constexpr int kF = 48, kRows = 8;  // the decisions and the value (not the expression logits)
    struct Decision {
        bool move_possible, moved, call_possible, called;
        float yaw, pitch;  // the sampled targets (-1..1) when moved
    };

    // Before the actor's outputs are used in a tick: adds the corrections to out (its first 8 values) for features f.
    void apply(const float* f, float* out) const;
    // After the decision of the tick: r = this tick's reward (motivation.h), out = the corrected outputs that were
    // used, f = their features. Learns from the previous tick's traces with this tick's TD error, then lays new traces.
    void learn(float reward, const float* f, const float* out, const Decision& d, bool night);

    void reset();
    static constexpr int kLogValues = 2;  // the diary: TD error, value
    void logValues(float* out) const
    {
        out[0] = delta_;
        out[1] = value_;
    }
    float delta() const { return delta_; }
    float value() const { return value_; }
    float size() const;  // mean |dW|, for the log
    // NVS
    using Weights = std::array<std::array<float, kF>, kRows>;
    const Weights& weights() const { return dw_; }
    void setWeights(const Weights& w);

private:
    Weights dw_{};
    Weights ea_{};  // actor traces (rows Move, YawMu, PitchMu, Call)
    std::array<float, kF> ev_{};
    std::array<float, kF> f_prev_{};
    float v_prev_ = 0.0f, delta_ = 0.0f, value_ = 0.0f;
    bool started_ = false;
};
