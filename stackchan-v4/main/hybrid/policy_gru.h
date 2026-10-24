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

    // The weights the step reads (INT8 matrices with per-row float scales, float biases). By default the ones
    // compiled in (policy_model_data.h); the robot swaps in the nightly ones from flash (weight_store.h).
    struct Params {
        const int8_t *inp_w, *gru_i_w, *gru_h_w, *fc1_w, *fc2_w;
        const float *inp_s, *inp_b, *gru_i_s, *gru_i_b, *gru_h_s, *gru_h_b, *fc1_s, *fc1_b, *fc2_s, *fc2_b;
    };
    static Params compiledParams();
    // Array names and sizes, in the order of Params (for loading them by name).
    struct ParamInfo {
        const char* name;
        bool int8;
        int count;
    };
    static constexpr int kParamCount = 15;
    static const ParamInfo* paramInfo();

    PolicyGru() : p_(compiledParams()) { reset(); }
    void use(const Params& p) { p_ = p; reset(); }
    void reset();
    // One 50 ms step. false (action = neutral) if the network is missing or
    // produced a non-finite value; the hidden state restarts after a NaN.
    bool step(const Input& in, Action& out);
    bool trained() const;
    uint32_t nanCount() const { return nan_count_; }
    // Neutral action: hold still, calm face, silent.
    static Action neutral();

private:
    Params p_;
    std::array<float, 64> h_{};
    std::array<float, 64> d_{};
    std::array<float, 48> f_{};
    std::array<float, 6 * 64> scratch_{};
    uint32_t nan_count_ = 0;
};
