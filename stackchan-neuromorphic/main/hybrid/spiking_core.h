#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

#include "hybrid_spec.h"

// The compact spiking core: 768 leaky integrate-and-fire neurons in 13
// populations (attention map, novelty, habituation, curiosity, arousal,
// boredom, annoyance, social, playfulness, fatigue/sleep, music, sensory
// traces, prediction mismatch). It answers "what matters, how do I feel,
// what happened recently" - never "what do I do".
//
// Trained on the PC (tools/hybrid/train_snn.py, surrogate gradients) to embody
// the designed internal-state dynamics; frozen on the robot. INT8 weights:
// a dense input projection, a sparse recurrent matrix (CSR by presynaptic
// neuron, event driven) and readouts through leaky integrator cells, which
// hold the slow drives (seconds to minutes) that spikes alone cannot.
//
// Endogenous activity: every neuron has a tonic bias and a noise current,
// so the network never falls silent; the bus also carries slow random
// pacemaker drifts ("noise" group) that make attention wander.
//
// 50 ms per brain tick = 2 substeps of 25 ms. Mirrors tools/hybrid/snn_model.py.
class SpikingCore {
public:
    static constexpr int kNeurons = 768;
    static constexpr int kMapW = 12, kMapH = 8, kMapCells = kMapW * kMapH;
    static constexpr int kInputs = hybrid::kSnnInputs + 2 * kMapCells;
    static constexpr int kSubsteps = 2;
    using Values = std::array<float, hybrid::SNN_COUNT>;

    SpikingCore();
    // Endogenous noise current (std per neuron per substep), 0 = deterministic.
    void setNoise(float sigma) { noise_ = sigma; }
    void seed(uint32_t s) { rng_ = s ? s : 1; }
    void reset();
    // One 50 ms tick: bus (hybrid::kSnnInputs values) -> the SNN section of the FeatureBus.
    // false (values unchanged) if the network is missing.
    bool tick(const float* bus, Values& out);

    // Input weights as a copy in faster memory than flash (PSRAM on the robot). Optional.
    static constexpr size_t kInputWeightBytes = static_cast<size_t>(kInputs) * kNeurons;
    void useFastInputWeights(const int8_t* copy) { win_ptr_ = copy; }
    static const int8_t* inputWeights();  // the table in flash, to copy from
    bool trained() const;
    // Diagnostics: mean spikes per neuron per substep (EMA) and total synapses.
    float meanRate() const { return mean_rate_; }
    int synapseCount() const;
    // Unstable (runaway / NaN): the caller resets the core.
    bool unstable() const { return unstable_; }
    // Cost per stage in microseconds (EMA): input drive, neuron updates + synapses, readouts.
    void setClock(int64_t (*clockUs)()) { clock_us_ = clockUs; }
    const std::array<float, 3>& stageUs() const { return stage_us_; }

private:
    void encode(const float* bus);
    float gauss();
    uint32_t rnd();

    std::array<float, kInputs> in_{};
    std::array<float, kNeurons> drive_{};   // W_in * input, held over the substeps
    std::array<float, kNeurons> v_{}, acc_{};
    std::array<uint8_t, kNeurons> s_{};
    std::array<uint16_t, kNeurons> spiked_{};
    int n_spiked_ = 0;
    std::array<float, 32> z_{};              // readout integrators
    std::array<float, kMapCells> att_{};
    static constexpr float kMinInput = 0.02f;
    const int8_t* win_ptr_ = nullptr;
    float noise_ = 0.12f;
    uint32_t rng_ = 0x9e3779b9u;
    float mean_rate_ = 0.0f;
    int last_spikes_ = 0;
    int hot_ticks_ = 0;
    int64_t (*clock_us_)() = nullptr;
    std::array<float, 3> stage_us_{};
    bool unstable_ = false;
};
