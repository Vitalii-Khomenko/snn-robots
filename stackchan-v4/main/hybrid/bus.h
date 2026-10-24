#pragma once
#include <algorithm>
#include <array>

#include "hybrid_spec.h"
#include "policy_gru.h"

namespace hybrid {

// The FeatureBus: every section the brain's modules read and write, in the
// wire order of tools/hybrid/spec.py. Perception and body encoders fill the
// first sections, the spiking core the SNN section, the policy reads all of
// it plus its own previous action.
struct Bus {
    std::array<float, VIS_COUNT> vision{};
    std::array<float, AUD_COUNT> audio{};
    std::array<float, BODY_COUNT> body{};
    std::array<float, CTX_COUNT> ctx{};
    std::array<float, PE_COUNT> pe{};
    std::array<float, EFF_COUNT> eff{};
    std::array<float, NOISE_COUNT> noise{};
    std::array<float, SNN_COUNT> snn{};
    std::array<float, PREV_COUNT> prev{};
    std::array<float, SLOW_COUNT> slow{};    // deterministic slow needs (boredom, social need); the policy only
    std::array<float, LEARN_COUNT> learn{};  // what the robot learned by itself (AssocMemory); the policy only

    static_assert(int(VIS_COUNT) + int(AUD_COUNT) + int(BODY_COUNT) + int(CTX_COUNT) + int(PE_COUNT) + int(NOISE_COUNT) +
                      int(SNN_COUNT) + int(PREV_COUNT) + int(SLOW_COUNT) + int(LEARN_COUNT) == kPolicyInputs,
                  "policy input = these sections (tools/hybrid/spec.py)");
    static_assert(int(VIS_COUNT) + int(AUD_COUNT) + int(BODY_COUNT) + int(CTX_COUNT) + int(PE_COUNT) + int(EFF_COUNT) +
                      int(NOISE_COUNT) == kSnnInputs,
                  "spiking core input = these sections (tools/hybrid/spec.py)");

    // The spiking core's input: vision, audio, body, ctx, pe, eff, noise.
    void snnInput(float* out) const
    {
        float* p = out;
        p = std::copy(vision.begin(), vision.end(), p);
        p = std::copy(audio.begin(), audio.end(), p);
        p = std::copy(body.begin(), body.end(), p);
        p = std::copy(ctx.begin(), ctx.end(), p);
        p = std::copy(pe.begin(), pe.end(), p);
        p = std::copy(eff.begin(), eff.end(), p);
        std::copy(noise.begin(), noise.end(), p);
    }
    // The policy's input: vision, audio, body, ctx, pe, noise, snn, prev, slow, learn.
    void policyInput(PolicyGru::Input& in) const
    {
        float* p = in.data();
        p = std::copy(vision.begin(), vision.end(), p);
        p = std::copy(audio.begin(), audio.end(), p);
        p = std::copy(body.begin(), body.end(), p);
        p = std::copy(ctx.begin(), ctx.end(), p);
        p = std::copy(pe.begin(), pe.end(), p);
        p = std::copy(noise.begin(), noise.end(), p);
        p = std::copy(snn.begin(), snn.end(), p);
        p = std::copy(prev.begin(), prev.end(), p);
        p = std::copy(slow.begin(), slow.end(), p);
        std::copy(learn.begin(), learn.end(), p);
    }
};

}  // namespace hybrid
