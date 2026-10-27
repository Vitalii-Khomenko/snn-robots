#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <esp_codec_dev.h>

#include "../perception/audio_frontend.h"
#include "../perception/voice_check.h"
#include "../core/ms_clock.h"

// Both ES7210 microphone channels at 24 kHz (the shared I2S bus), read by a
// background task in 128-frame stereo blocks (5.3 ms). Every block feeds the
// Auditory NN's filter bank (audioFront()); on top of that two exact events
// for the brain: a debounced "loud sound" (onLoudSound(), fired from update()
// on the main-loop thread) and sound onsets (takeOnset()). The task only
// writes atomics and single-producer rings.
//
// Sound direction is not measured: two microphones 5 cm apart gave nothing
// usable (docs/HARDWARE.md).
class MicrophoneEngine {
public:
    bool begin();
    void update();  // call every main-loop tick; fires onLoudSound() here

    void onLoudSound(std::function<void()> cb) { on_loud_sound_ = std::move(cb); }
    // Peak (int16) of the block that raised the last loud-sound event, and
    // whether the robot's own movement raised the threshold at the time -
    // logged so servo-click levels can be measured on hardware.
    int lastLoudPeak() const { return last_loud_peak_.load(); }
    bool lastLoudDuringOwnMotion() const { return last_loud_own_motion_.load(); }
    // Measurement of the robot's own servo noise: the main loop says which
    // phase the head is in (0 still, 1 within 400 ms of a start or stop,
    // 2 the steady middle of a turn) and the mic task keeps the largest
    // block peak per phase, taken by the stats line.
    void setMotionPhase(int phase) { motion_phase_.store(phase); }
    int takePhasePeak(int phase) { return phase_peak_[phase].exchange(0); }
    // A listener on the home PC (net/sound_tap.h): every block (interleaved L R), its time, whether the speaker mute
    // or the own-motion threshold was on. Called on the microphone task; it must not block.
    using Tap = void (*)(const int16_t* lr, size_t frames, uint32_t ms, bool muted, bool ownMotion);
    void setTap(Tap tap) { tap_.store(tap); }
    // Is it a voice (perception/voice_check.h): ~250 ms share of voiced 25 ms frames, 0..1.
    float voice() const { return voice_check_.voice(); }
    // The filter-bank frames the Auditory NN reads (perception/audio_frontend.h).
    const AudioFrontEnd& audioFront() const { return front_; }

    // Call while our own speaker plays so the robot does not mistake its own
    // voice for the world (no loud / onset events while the window is open).
    void suppressFor(uint32_t ms);
    // While the robot's own servos move, only sounds louder than
    // kLoudThresholdOwnMotion count as loud (servo clicks reach ~20k).
    // Muting instead left the robot deaf about half the time while it explored.
    void raiseThresholdFor(uint32_t ms);

    // Sound onsets (sharp energy rises: claps, taps, knocks), detected per
    // block in the mic task; the main loop drains them here.
    struct Onset {
        uint32_t ms = 0;
        float strength = 0.0f;  // 0.3..1
    };
    bool takeOnset(Onset& out);

    static constexpr uint32_t kSampleRate = 24000;


private:
    static void taskEntry(void* arg);
    void taskLoop();

    static constexpr size_t kChunkSamplesPerChannel = 128;

    esp_codec_dev_handle_t mic_ = nullptr;
    std::function<void()> on_loud_sound_;

    AudioFrontEnd front_;
    VoiceCheck voice_check_;
    std::atomic<bool> loud_sound_pending_{false};
    std::atomic<Tap> tap_{nullptr};
    std::atomic<uint32_t> ignore_until_ms_{0};
    // Speaker mute window open (wrap-safe; the longest mute is a song).
    bool muted(uint32_t now) const { return msWithin(now, ignore_until_ms_.load(), 60000); }
    std::atomic<uint32_t> raised_until_ms_{0};
    std::atomic<int> last_loud_peak_{0};
    std::atomic<bool> last_loud_own_motion_{false};
    std::atomic<int> motion_phase_{0};
    std::array<std::atomic<int>, 3> phase_peak_{};
    uint32_t last_trigger_ms_ = 0;
    // Onset detector (mic task only) and its single-producer ring.
    float onset_slow_ = 0.0f;
    uint32_t last_onset_ms_ = 0;
    std::array<Onset, 16> onsets_{};
    std::atomic<uint32_t> onset_head_{0};
    uint32_t onset_tail_ = 0;  // consumer side
    // An onset: block mean |x| above kOnsetRatio x the ~100 ms average and
    // a peak of at least kOnsetPeak (quiet taps count, room hiss does not).
    static constexpr float kOnsetRatio = 2.5f;
    static constexpr int kOnsetPeak = 1200;
    static constexpr uint32_t kOnsetRefractoryMs = 120;

    // Ambient noise measured during bring-up peaked around 5-110/32767;
    // loud speech/claps were 700-9800/32767. This sits well above ambient
    // noise without requiring a shout.
    static constexpr int kLoudThreshold          = 2000;
    // Above the servos' own clicks (hardware: up to ~19-22k when a turn
    // stops). A real clap nearby clips at 32767.
    static constexpr int kLoudThresholdOwnMotion = 24000;
    static constexpr uint32_t kRetriggerCooldownMs = 3000;
};
