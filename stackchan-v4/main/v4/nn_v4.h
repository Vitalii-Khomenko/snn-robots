#pragma once
#include <array>
#include <cstdint>

#include "v4_spec.h"

// Brain v4's main network: every decision in one place (BRAIN_V4_PLAN.txt section 6). The spiking analyzer's 32
// numbers + 12 "what I just did" -> GRU 64 -> the emotion head (21 + strength) -> [GRU state + the emotion's
// probabilities] -> hidden 64 -> the action heads (voice, LEDs, head). Float weights, 10 Hz, ~31 k parameters.
// Mirrors tools/v4/nn_v4.py (activate(): softmax per choice group, tanh for signed values, sigmoid for the rest).
class NnV4 {
public:
    static constexpr int kIn = v4::kNnInputs, kHidden = 64, kOut = v4::NO_COUNT;
    using In = std::array<float, kIn>;
    using Out = std::array<float, kOut>;

    static constexpr int kEmo = 21, kAct = kOut - kEmo - 1;
    static constexpr int kAct1W = kHidden * (kHidden + kEmo), kAct1B = kHidden, kAct2W = kAct * kHidden, kAct2B = kAct;

    void reset() { h_.fill(0.0f); }
    bool trained() const;
    // false (out unchanged) without trained weights or on a non-finite result (the state is reset then)
    bool step(const In& in, Out& out);
    // The action heads learned from the robot's life (stage 6, learned_blob.h); all nullptr = the compiled-in ones.
    // Only between two steps (the main loop's thread).
    void useHeads(const float* act1W, const float* act1B, const float* act2W, const float* act2B);

private:
    std::array<float, kHidden> h_{};
    const float *act1_w_ = nullptr, *act1_b_ = nullptr, *act2_w_ = nullptr, *act2_b_ = nullptr;
    bool heads_set_ = false;
};
