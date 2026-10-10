#pragma once
#include <array>
#include <cstdint>

#include "hybrid_spec.h"

// The policy: "given this situation, what should I do next?"
//
//   100 inputs (vision, audio, body, context, prediction error, noise pacemakers,
//   the spiking core's state, the previous action)
//     -> dense 64 -> GRU 64 -> dense 48 -> 51 normalized action values (hybrid::ACT_COUNT)
//
// Trained on the PC by imitation of a designed teacher, then corrected by
// DAgger rollouts in the digital twin (tools/hybrid/train_policy.py); ~39 k
// INT8 weights. It outputs motor INTENT, face and voice parameters - never
// PWM or servo packets: every value passes BodyExecutor's deterministic limits.
class PolicyGru {
public:
    using Input = std::array<float, hybrid::kPolicyInputs>;
    using Action = std::array<float, hybrid::ACT_COUNT>;

    PolicyGru() { reset(); }
    void reset();
    // One 50 ms step. false (action = neutral) if the network is missing or
    // produced a non-finite value; the hidden state restarts after a NaN.
    bool step(const Input& in, Action& out);
    bool trained() const;
    uint32_t nanCount() const { return nan_count_; }
    // Neutral action: hold still, calm face, silent.
    static Action neutral();

private:
    std::array<float, 64> h_{};
    std::array<float, 64> d_{};
    std::array<float, 48> f_{};
    std::array<float, 6 * 64> scratch_{};
    uint32_t nan_count_ = 0;
};
