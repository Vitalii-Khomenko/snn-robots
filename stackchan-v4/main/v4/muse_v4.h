#pragma once
#include <array>

#include "muse_v4_data.h"
#include "v4_spec.h"

// The Muse (docs/CREATIVITY.md, tools/v4/muse.py - its twin): brain v4's source of ideas. 64 leaky tanh neurons in
// three time scales (0.5 s / 3 s / 60 s), recurrently connected a little past the edge of chaos, pushed gently by the
// analyzer's mood. Its 8 outputs are the main network's pacemakers (pace_fast / pace_slow / pace_jitter, which used to
// be blind noise) and five ideas it sees (idea_play / show / still / social / surprise). It gets bored of an idea it
// keeps having (habituation over 90 s); the night sleep moves one bias per idea ("muse.bias" in the learned blob).
// The noise comes from the caller (BrainV4's generator), so the PC twin computes the same numbers.
class MuseV4 {
public:
    using Out = std::array<float, muse_data::kOuts>;
    static constexpr int kN = muse_data::kN;

    // one step at 10 Hz: snn = the analyzer's 32 numbers, noise = kN standard normal values
    void step(const float* snn, const float* noise, Out& out);
    // what the night learned (8 values, clipped to +-kBiasMax); nullptr = none
    void setBias(const float* bias);
    const std::array<float, muse_data::kOuts>& bias() const { return bias_; }

private:
    std::array<float, kN> x_{};
    std::array<float, muse_data::kOuts - 3> hab_{};
    std::array<float, muse_data::kOuts> bias_{};
};
