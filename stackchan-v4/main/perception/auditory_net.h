#pragma once
#include <array>
#include <cstdint>

#include "audio_frontend.h"
#include "layer_params.h"

// What the ear reports every brain tick (AuditoryNet::run). Brain v4 reads AUD_ENERGY and AUD_SPEECH (the loudness and
// voice channels); the rest is kept for the stats line and for debugging the ear.
enum Aud : uint8_t {
    AUD_SILENCE, AUD_SPEECH, AUD_CLAP, AUD_SHARP, AUD_CONTINUOUS, AUD_LOW, AUD_HIGH, AUD_TONAL,  // the network's classes
    AUD_APPROACH, AUD_ENERGY, AUD_NOVELTY, AUD_SELF_VOICE, AUD_OWN_MOTION, AUD_ONSET, AUD_LOUD,  // measured here
    AUD_COUNT
};

// The Auditory NN ("what did I hear?") - separate from, and unrelated to,
// the vocal decoder that makes the robot's own voice. A tiny MLP trained on
// the PC (platform B: stackchan-life/tools/hybrid/train_audio.py) over the filter-bank frames:
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
    using Features = std::array<float, AUD_COUNT>;

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
    const std::array<float, kOutputs>& lastOutputs() const { return out_; }
    bool trained() const;
    uint32_t nanCount() const { return nan_count_; }
    // Weights: compiled in (auditory_model_data.h, as trained by platform B's nights up to 2026-10-24).
    static constexpr int kLayers = 3;
    using Params = std::array<LayerParams, kLayers>;
    static Params compiledParams();

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
