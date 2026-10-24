#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

#include "v4_spec.h"

// Brain v4's spiking analyzer: 768 LIF neurons that turn the 13 sense channels into the 32 numbers of the
// FeatureBus-free v4 wire (v4_spec.h SnnOut) - "how pleasant, how excited, too little or too much, danger, news,
// what matters, what I need, where to look". It never decides what to do (BRAIN_V4_PLAN.txt, sections 4-5).
//
//   ENC   130  per channel 8 level neurons (Gaussian receptive fields) + ON / OFF; fixed
//   ZONE  260  per channel 5 innate zones x 4 adaptive neurons (habituation); fixed wiring from ENC
//   MIX   282  recurrent, 80 % excitatory / 20 % inhibitory; trained int8 input and recurrent weights
//   ATT    96  12 x 8 attention map: a bump where the movement is + trained int8 input from MIX
//   readout    32 leaky integrators over all spikes (trained int8), sigmoid / tanh
//
// Event driven: a spike adds its column of int8 weights (stored transposed, by presynaptic neuron); per-row float
// scales turn the integer sums into currents. 20 Hz ticks of 2 substeps. Mirrors tools/v4/snn_v4.py.
class SnnV4 {
public:
    static constexpr int kChannels = v4::CH_COUNT, kGrf = 8;
    static constexpr int kEnc = kChannels * (kGrf + 2), kZone = kChannels * v4::ZONE_COUNT * 4, kMix = 282,
                         kAtt = 96, kNeurons = kEnc + kZone + kMix + kAtt, kOut = v4::SO_COUNT;
    static constexpr int kSubsteps = 2;
    using In = std::array<float, v4::kSnnInputs>;
    using Out = std::array<float, v4::SO_COUNT>;

    SnnV4();
    void reset();
    // One 50 ms tick. false (out unchanged) when no trained weights are compiled in.
    bool tick(const In& in, Out& out);
    bool trained() const;
    // weights in faster memory (PSRAM copy of the flash tables), optional
    void useFastWeights();
    // diagnostics: mean spike rate of MIX (EMA), last tick's cost in microseconds
    float mixRate() const { return mix_rate_; }
    void setClock(int64_t (*clockUs)()) { clock_us_ = clockUs; }
    uint32_t lastUs() const { return last_us_; }

private:
    std::array<float, kEnc> v_enc_{};
    std::array<float, kZone> v_zone_{}, a_zone_{};
    std::array<float, kMix> v_mix_{};
    std::array<uint8_t, kMix> s_mix_{};
    std::array<float, kAtt> v_att_{};
    std::array<float, kOut> ro_{};
    std::array<float, kChannels> prev_{};
    bool has_prev_ = false;
    float mix_rate_ = 0.0f;
    std::array<uint16_t, kNeurons> spiked_{};
    std::array<int32_t, kMix> acc_in_{}, acc_rec_{};
    const int8_t* w_in_t_;   // [kEnc + kZone][kMix]
    const int8_t* w_rec_t_;  // [kMix][kMix]
    const int8_t* w_att_t_;  // [kMix][kAtt]
    const int8_t* w_ro_t_;   // [kNeurons][kOut]
    int64_t (*clock_us_)() = nullptr;
    uint32_t last_us_ = 0;
};
