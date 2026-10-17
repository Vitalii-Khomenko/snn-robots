#pragma once
#include <array>
#include <cstdint>

#include "droid_synth.h"

// The voice network: a small recurrent network (GRU 48, ~38 k INT8 weights, trained on the PC by
// tools/voice/train_babble.py) that composes a phrase note by note from a voice intent: which droid
// sound (one of ten kinds: whistle, chirp, pip, burble, trill, warble, giggle, squeal, growl, coo), its glide curve, an ending flick, duration, pitch, slope,
// loudness, FM depth, harmonics, vibrato, speed, gap - and when to stop. It runs once per note
// (2-8 times a second), not per audio sample; the synthesizer (droid_synth.h) turns each note into sound.
//
// The shape / ending / continuous heads exist once per kind (the output block of the sampled kind is
// used), so a pip never gets a whistle's duration. Kinds, shapes and endings are drawn from the
// network's distributions (no phrase is the same twice); continuous values get a little noise matched
// to the training residual.
class BabbleNet {
public:
    static constexpr int kIntent = 12;  // valence, arousal, register, slope, expr, length, rate, calm, ask, play, grumble, sing
    static constexpr int kKinds = 10, kShapes = 4, kEndings = 3, kCont = 9, kEmotions = 21;
    // inputs: intent 12, the face's emotion (one-hot, face_style.h order) 21, prev kind 10, prev shape 4, prev ending 3,
    // prev f0 / slope / dur / loud, step, first
    static constexpr int kInputs = 12 + kEmotions + kKinds + kShapes + kEndings + 4 + 2;
    // outputs: stop, kind 10, shape 10x4, ending 10x3, continuous 10x9
    static constexpr int kOutputs = 1 + kKinds + kKinds * kShapes + kKinds * kEndings + kKinds * kCont;
    static constexpr int kMaxNotes = 7;

    void seed(uint32_t s) { rng_ = s ? s : 1; }
    // Starts a phrase.
    void begin(const std::array<float, kIntent>& intent, int emotion);
    // Next note of the phrase; false when the network says stop (or after 7 notes).
    bool next(droid::Note& out);
    bool trained() const;
    uint32_t nanCount() const { return nan_count_; }

private:
    float uniform();
    std::array<float, kIntent> intent_{};
    int emotion_ = 0;
    std::array<float, 48> h_{};
    std::array<float, 64> d_{};
    std::array<float, 48> f_{};
    std::array<float, 6 * 48> scratch_{};
    int step_ = 0;
    bool have_prev_ = false;
    int prev_kind_ = 0, prev_shape_ = 0, prev_ending_ = 0;
    float prev_f0_ = 0, prev_slope_ = 0, prev_dur_ = 250, prev_loud_ = 0.7f;
    uint32_t rng_ = 0x2545f491u;
    uint32_t nan_count_ = 0;
};
