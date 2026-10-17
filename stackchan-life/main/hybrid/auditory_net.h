#pragma once
#include <array>
#include <cstdint>

#include "audio_frontend.h"
#include "hybrid_spec.h"
#include "layer_params.h"

// The Auditory NN ("what did I hear?") - separate from, and unrelated to,
// the vocal decoder that makes the robot's own voice. A tiny MLP trained on
// the PC (tools/hybrid/train_audio.py) over the filter-bank frames:
//
//   100 inputs (16 bands x 6 frames, balance, level, "I am speaking",
//   "my servos run") -> dense 64 -> dense 32 -> 8 outputs
//   (silence, speech-like, clap, sharp noise, continuous noise, low event,
//   high event, tonal)
//
// The rest of the Audio features (energy, novelty, approach, onset, loud,
// the two self flags) are exact measurements made here from the same frames.
class AuditoryNet {
public:
    static constexpr int kOutputs = 8;
    using Features = std::array<float, hybrid::AUD_COUNT>;

    struct Context {
        bool selfSpeaking = false;   // the speaker plays our own voice
        bool ownMotion = false;      // the head servos run (or ran a moment ago)
        float selfLevel01 = 0.0f;    // loudness of our own voice (SelfSoundAnalyzer)
        float onset01 = 0.0f;        // a sound onset since the last call (MicrophoneEngine)
        float loud01 = 0.0f;         // a loud-sound event since the last call
    };

    // Called once per brain tick. false (features zeroed) while the filter
    // bank has no history yet, or if the network is missing / produced NaN.
    bool run(const AudioFrontEnd& front, const Context& ctx, Features& out);
    // The network alone: raw input -> nonlinear outputs (parity tests, training tools).
    bool infer(const float* input);
    const std::array<float, AudioFrontEnd::kInputs>& lastInput() const { return in_; }
    // the second hidden layer of the last run (32, after ReLU): what it heard beyond the labels (bus v4)
    const std::array<float, 32>& hidden() const { return h2_; }
    const std::array<float, kOutputs>& lastOutputs() const { return out_; }
    bool trained() const;
    uint32_t nanCount() const { return nan_count_; }
    // Weights: compiled in, or the nightly ones from the blob (fc1..3); used by the brain task itself (no race).
    static constexpr int kLayers = 3;
    using Params = std::array<LayerParams, kLayers>;
    static Params compiledParams();
    static const LayerInfo* layerInfo();
    void use(const Params& p) { p_ = p; }

private:
    std::array<float, AudioFrontEnd::kInputs> in_{};
    std::array<float, kOutputs> out_{};
    std::array<float, 64> h1_{};
    std::array<float, 32> h2_{};
    // novelty background: per-band slow average of the log spectrum
    std::array<float, AudioFrontEnd::kBands> background_{};
    bool background_ready_ = false;
    float floor_ = 0.5f;  // quiet baseline of the level
    std::array<float, AudioFrontEnd::kBands> band_floor_{};  // ... and of every band
    bool band_floor_valid_ = false;
    // approach: low-band envelope trend
    std::array<float, 24> envelope_{};  // ~1.2 s of ticks
    int envelope_n_ = 0;
    uint32_t nan_count_ = 0;
    Params p_ = compiledParams();
};
