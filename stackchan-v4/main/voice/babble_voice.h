#pragma once
#include <array>
#include <atomic>
#include <cstdint>

#include "../audio/audio_output.h"
#include "babble_net.h"
#include "droid_synth.h"

// The robot's voice: the main network's sound intent -> BabbleNet (notes) -> droid synth -> speaker.
//
//   Intent (say, mood, register, slope, expressiveness, length, rate, 4 styles)
//     -> a call is started on a rising edge of `say` (armed, not too soon, not while sounding)
//     -> BabbleNet composes the phrase note by note, a quarter second ahead of the sound
//     -> droid::Synth renders 24 kHz audio, streamed in 20 ms chunks through AudioOutput
//
// Pitch register: a droid voice, base F0 about 520 .. 1200 Hz, syllables glide a long way above and below it.
class BabbleVoice {
public:
    // What to say, from brain v4's voice head (v4/v4_executor.cpp). Values 0..1 unless noted.
    struct Intent {
        float say = 0.0f;                                // a call starts when this rises above 0.55
        float valence = 0.0f;                            // -1..1
        float arousal = 0.3f;
        float reg = 0.5f;                                // pitch register
        float slope = 0.0f;                              // -1..1, the phrase's melodic direction
        float expr = 0.3f;                               // expressiveness
        float length = 0.5f, rate = 0.5f;
        float calm = 0.0f, ask = 0.0f, play = 0.0f, grumble = 0.0f;  // style
    };

    explicit BabbleVoice(AudioOutput& audio) : audio_(audio) {}

    // Starts the audio task (call after AudioOutput::begin()). The voice is rendered and written
    // to the speaker by its own task - core 1, above the microphone and camera tasks - so a busy
    // main loop (brain tick, display, Wi-Fi) can never starve the speaker and make the sound
    // stutter.
    bool begin();

    // Once per brain tick (50 ms): may start a phrase (`preempt`: a reaction - a touch, a shake - may cut an idle phrase short). `emotion` = the face being shown (Emotion enum), it picks the character of the voice. `allow` = the call budget (token bucket)
    // has room. Returns true when this tick started a phrase.
    bool step(const Intent& a, bool allow, int emotion, bool preempt = false);

    bool active() const { return active_.load(); }
    bool trained() const { return net_.trained(); }
    bool phraseEnded() { return ended_.exchange(false); }
    const char* phraseText() const { return text_; }
    float baseHz() const { return base_hz_; }
    uint32_t calls() const { return calls_; }
    uint32_t takeRenderUs() { return render_us_.exchange(0); }
    // Chunks that came later than 1.5 chunk durations while sounding (audible gaps), since boot.
    uint32_t gaps() const { return gaps_.load(); }
    void seed(uint32_t s) { net_.seed(s); }
    // A wider voice - phrase register 300-1580 Hz (default: 520-1190) and 1.5x the jumps between notes (platform B used
    // it; brain v4 keeps the default). Measured 2026-10-18: 60 % of the voice's energy sat between 400 and 1500 Hz.
    void setWide(bool wide)
    {
        wide_ = wide;
        net_.setPitchRange(wide ? 1.5f : 1.0f);
    }

    static constexpr float kSampleRate = static_cast<float>(AudioOutput::kSampleRate);

private:
    bool wide_ = false;
    static void taskEntry(void* arg);
    void taskLoop();
    void startPhrase();
    void topUp();

    AudioOutput& audio_;
    BabbleNet net_;
    droid::Synth synth_;
    std::array<int16_t, 480> chunk_{};
    // main thread -> audio task
    std::atomic<bool> pending_{false}, interrupt_{false};
    std::array<float, BabbleNet::kIntent> pending_intent_{};
    int pending_emotion_ = 0;
    float pending_base_hz_ = 250.0f;
    // audio task -> main thread
    std::atomic<bool> active_{false}, ended_{false};
    std::atomic<uint32_t> render_us_{0}, gaps_{0};
    bool phrase_active_ = false, armed_ = true, prev_preempt_ = false, reaction_open_ = false;
    int since_ticks_ = 1000;
    float base_hz_ = 250.0f;
    uint32_t calls_ = 0;
    char text_[96] = "";
    static constexpr int kRefractoryTicks = 14;  // 0.7 s between calls
};
