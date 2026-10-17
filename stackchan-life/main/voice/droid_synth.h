#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

// The robot's voice box: an analogue-style droid synthesizer (R2-D2 school), not a speech model.
//
// A "note" is one electronic utterance with a pitch glide and a character:
//   whistle  a long sliding tone with light FM and vibrato ("wiiii")
//   chirp    a fast exponential sweep rich in harmonics ("ri", "chrp")
//   pip      a very short click-attack beep with a metallic FM ring ("pip", "bib")
//   burble   rapid random steps of pitch, gated, strong inharmonic FM - computer chatter ("brbl")
//   trill    a tone whose pitch and level flutter 20-45 times a second ("brrr")
//   warble   a slow deep pitch wobble, 4-13 times a second ("wuwuwu": dizzy, crying, confused)
//   giggle   a run of short rising blips, 6-16 a second ("hihihi": laughing)
//   squeal   a short loud harmonic-rich rising scream ("eek": surprise, fear)
//   growl    a low buzzy tone with 35-75 Hz amplitude chop ("grrr": anger)
//   coo      a soft pure tone with a slow swell and a little vibrato ("mmm": love, sleep, sadness)
// plus an optional flick at the end (the pitch jumps up or down in the last 50 ms: a question or a sigh).
//
// Voices: a sine carrier with band-limited harmonics (2nd and 3rd, switched off before Nyquist) and
// phase modulation by a second oscillator at a fixed ratio; the modulation index is limited so no sideband
// can alias. Pitch follows the note's glide curve (linear, fast-then-slow, slow-then-fast, scoop).
// A high-pass takes out what the little speaker cannot play, a soft saturator keeps loud notes round.
//
// Deterministic DSP: it decides nothing (the BabbleNet network chooses the notes). It runs per sample in
// the audio task; tools/voice/voice_preview.cpp renders phrases to WAV on the PC.
namespace droid {

enum Kind : uint8_t { K_WHISTLE, K_CHIRP, K_PIP, K_BURBLE, K_TRILL, K_WARBLE, K_GIGGLE, K_SQUEAL, K_GROWL, K_COO, kKinds };
enum Shape : uint8_t { S_LINEAR, S_FAST, S_SLOW, S_SCOOP, kShapes };
enum Ending : uint8_t { E_NONE, E_UP, E_DOWN, kEndings };

struct Note {
    Kind kind = K_WHISTLE;
    Shape shape = S_LINEAR;
    Ending ending = E_NONE;
    float durMs = 250.0f;   // whole note including the ending flick
    float f0St = 0.0f;      // pitch at the start, semitones above the phrase's base pitch
    float slopeSt = 0.0f;   // pitch change over the note (burble: range of the steps), semitones
    float loud = 0.8f;      // 0..1
    float fm = 0.4f;        // 0..1 FM depth: 0 pure tone .. 1 metallic
    float harm = 0.3f;      // 0..1 harmonic content: 0 sine .. 1 buzzy
    float vib = 0.0f;       // 0..1 vibrato depth (whistle)
    float rate = 0.5f;      // 0..1 burble step rate / trill speed
    float gapMs = 25.0f;    // silence after the note
};

constexpr float kSampleRate = 24000.0f;

class Synth {
public:
    // Appends the note after what is queued. baseHz = the phrase's base pitch. false if the queue is full.
    bool enqueue(const Note& n, float baseHz);
    int queuedMs() const;
    bool active() const { return playing_ || count_ > 0; }
    void clear();
    // Renders n samples (silence when nothing is queued).
    void render(int16_t* out, size_t n);

    float level = 0.167f * 32767.0f;  // output scale after the saturator (peaks ~5k, RMS ~2.5k; 0.75 -> 0.375 -> x0.7 -> x0.85 -> x0.75 at the owner's request)

private:
    struct Item {
        Note note;
        float baseHz;
    };
    void start(const Item& it);
    float sampleOne();
    uint32_t rnd();
    float pitchSemis(float u, float tSec) const;

    static constexpr size_t kCapacity = 16;
    std::array<Item, kCapacity> q_{};
    size_t head_ = 0, count_ = 0;
    int queued_samples_ = 0;

    // current note
    bool playing_ = false;
    Item cur_{};
    int n_ = 0, total_ = 0, body_ = 0, gap_ = 0;
    float phase_ = 0.0f, mphase_ = 0.0f, vib_phase_ = 0.0f, lfo_phase_ = 0.0f;
    // burble steps
    int step_len_ = 1, step_n_ = 0;
    float step_pitch_ = 0.0f, step_target_ = 0.0f, step_amp_ = 1.0f;
    // output filter
    float hp_x_ = 0.0f, hp_y_ = 0.0f;
    uint32_t rng_ = 0x1234567u;
};

}  // namespace droid
