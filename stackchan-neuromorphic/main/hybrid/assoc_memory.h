#pragma once
#include <array>
#include <cstdint>

#include "bus.h"

// What the robot learns by itself, while it lives - the one part of the brain
// that is not trained on the PC.
//
// A small plastic spiking circuit next to the spiking core:
//   - 11 cue groups (4 integrate-and-fire cells each) code the situation, at a
//     rate that follows the cue: a voice, a bang, a tone, a steady noise, a
//     person, a face, motion, something coming closer, a light change, the
//     owner at home, night. (Not the robot's own call: it answers a clap or a
//     touch at once, and its own answer then took the credit for what came next.)
//   - each cue keeps an eligibility trace of its recent spikes (rises at once,
//     fades over ~3.5 s: "what happened just before");
//   - two outcome groups (8 leaky integrate-and-fire cells each) - being
//     petted and being handled roughly - are driven only through the plastic
//     synapses cue -> outcome; their firing rate is the expectation the policy
//     reads (LEARN section of the FeatureBus);
//   - three-factor learning: the onset of an outcome (a touch while the robot is
//     not annoyed; a shake or a lift) is the teaching signal, and every synapse
//     moves by  eta * eligibility * (outcome - expectation)  - the
//     Rescorla-Wagner / dopamine prediction-error rule, with one error shared
//     by all cues. Only the onset teaches, with the traces of what came BEFORE
//     it, and nothing is learned for 3 s after it: the robot's own reaction (it
//     squeaks while it is shaken) must not take the credit. Extinction works
//     per episode: a cue's episode closes when its trace has faded (~5 s after
//     the cue ended), and only an episode that no outcome followed weakens the
//     link (judging it earlier punished the gap between a voice and the hand
//     that comes a second later). A cue that is there all the time (the owner at
//     home, a person in view) has no episodes: it loses its link slowly while it
//     is present without the outcome, so it ends up with the base rate; it also
//     learns at a third of the rate (lower salience), or it blocks the events.
//
// Deterministic, a few microseconds per tick. The weights are the robot's
// memory: RobotRuntime keeps them in NVS so they survive a reboot; they fade by
// themselves over days if nothing confirms them.
class AssocMemory {
public:
    static constexpr int kCues = 11, kOutcomes = 2;
    enum Outcome : uint8_t { Reward, Threat };
    static constexpr int kCueCells = 4, kOutCells = 8;
    static const char* cueName(int c);
    static const char* outcomeName(int o) { return o == Reward ? "petting" : "rough handling"; }

    using Weights = std::array<std::array<float, kOutcomes>, kCues>;

    // One 50 ms tick: reads the bus (perception, body, the spiking core's annoyance), learns, writes bus.learn.
    void tick(hybrid::Bus& bus);

    const Weights& weights() const { return w_; }
    void setWeights(const Weights& w);
    // Largest weight change since the last call (the runtime saves to flash when it is worth it).
    float takeChange();
    // A link that crossed 0.25 or 0.5 (up or down) since the last call, for the log. false if none.
    bool takeEvent(int& cue, int& outcome, float& weight, bool& rising);

    static constexpr float kMaxWeight = 1.5f;

private:
    void cues(const hybrid::Bus& bus, std::array<float, kCues>& c) const;

    Weights w_{};
    Weights logged_{};                          // the level last reported per link (0, 0.25, 0.5)
    std::array<std::array<float, kCueCells>, kCues> cue_v_{};  // integrate-and-fire membranes
    std::array<float, kCues> elig_{};
    std::array<std::array<float, kOutCells>, kOutcomes> out_v_{};
    std::array<float, kOutcomes> rate_{};
    std::array<bool, kOutcomes> was_on_{};
    std::array<bool, kCues> in_episode_{};
    std::array<std::array<bool, kOutcomes>, kCues> reinforced_{};  // this episode of the cue was followed by the outcome
    std::array<int, kOutcomes> since_onset_{{1000, 1000}};  // ticks
    float change_ = 0.0f;
};
