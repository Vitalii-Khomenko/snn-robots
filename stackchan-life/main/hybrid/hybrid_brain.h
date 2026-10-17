#pragma once
#include <array>
#include <cstdint>
#include <functional>

#include "assoc_memory.h"
#include "bus.h"
#include "motivation.h"
#include "policy_gru.h"
#include "spiking_core.h"

namespace hybrid {

// Virtual physiology: a heartbeat and breathing that follow the internal
// state (arousal, curiosity, sleep), nudged a little by CPU load. The face
// and the rear LEDs show it - they animate the state, not random decoration.
struct Physiology {
    float heartBpm = 60.0f, breathPerMin = 14.0f;
    float heartPhase = 0.0f;  // 0..1
    bool heartBeat = false;   // a beat began this tick
};

}  // namespace hybrid

// The hybrid brain: spiking core (internal state) + policy GRU (action
// selection) around the FeatureBus. Perception and the body encoders fill the
// bus before tick(); the executor reads action() after it.
class HybridBrain {
public:
    using ExternalPolicy = std::function<void(const PolicyGru::Input&, PolicyGru::Action&)>;

    HybridBrain();
    hybrid::Bus& bus() { return bus_; }
    const hybrid::Bus& bus() const { return bus_; }

    void tick(uint32_t nowMs);
    const PolicyGru::Action& action() const { return action_; }
    // The input the policy saw in the last tick (the bus changes after it: the previous-action features are
    // updated with the new action). Training dumps and parity tests need this one.
    const PolicyGru::Input& lastPolicyInput() const { return policy_in_; }
    const hybrid::Physiology& physiology() const { return physio_; }
    SpikingCore& snn() { return snn_; }
    const SpikingCore& snn() const { return snn_; }
    PolicyGru& policy() { return policy_; }
    AssocMemory& assoc() { return assoc_; }
    // the five needs and the reward (platform B: the actor of mode life learns from them)
    const Motivation& motivation() const { return motive_; }
    Motivation& motivation() { return motive_; }
    const AssocMemory& assoc() const { return assoc_; }
    // Replaces the trained policy (training rollouts in the simulator).
    void setExternalPolicy(ExternalPolicy fn) { external_ = std::move(fn); }
    // The v3 spiking core on or off (off in mode life: platform B does not read it)
    void setSpikingCore(bool on) { snn_on_ = on; }
    // CPU load 0..1 (the stats line), nudges the heart rate a little.
    void setLoad(float load01) { load_ = load01; }
    // Per-stage cost, us (EMA): snn, policy.
    void setClock(int64_t (*clockUs)()) { clock_us_ = clockUs; }
    uint32_t snnUs() const { return snn_us_; }
    uint32_t takeSnnMinUs() { const uint32_t m = snn_min_us_; snn_min_us_ = 0xffffffffu; return m; }
    uint32_t policyUs() const { return policy_us_; }
    uint32_t snnResets() const { return snn_resets_; }
    uint32_t policyFaults() const { return policy_faults_; }
    uint32_t tickCount() const { return ticks_; }

private:
    void updateNoise();
    void updatePhysiology();
    uint32_t rnd();
    float gauss();

    hybrid::Bus bus_;
    SpikingCore snn_;
    PolicyGru policy_;
    AssocMemory assoc_;
    Motivation motive_;
    PolicyGru::Action action_ = PolicyGru::neutral();
    hybrid::Physiology physio_;
    ExternalPolicy external_;
    std::array<float, hybrid::NOISE_COUNT> ou_{};
    uint32_t rng_ = 0x6c078965u;
    float load_ = 0.0f;
    bool snn_on_ = true;
    float annoy_evidence_ = 0.0f;
    uint32_t ticks_ = 0, snn_resets_ = 0, policy_faults_ = 0;
    PolicyGru::Input policy_in_{};
    void updateSlow();
    float stim_s_ = 0.3f, slow_boredom_ = 0.0f, slow_social_ = 0.2f;
    int64_t (*clock_us_)() = nullptr;
    uint32_t snn_us_ = 0, policy_us_ = 0, snn_min_us_ = 0xffffffffu;
};
