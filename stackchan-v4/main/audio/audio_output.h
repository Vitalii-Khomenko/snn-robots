#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <esp_codec_dev.h>

#include "self_sound.h"

// Speaker output, built the way bring-up found actually works on this
// board (see docs/HARDWARE.md "speaker: done" for the full story): the
// generic BSP's bsp_audio_codec_speaker_init() alone is silent because its
// AW88298 hw_gain.pa_gain=15 calibration is ~14 dB quieter than this exact
// hardware needs, and its default 22050 Hz sample rate never gets the
// AW88298 into an active state. This class reuses bsp_audio_codec_speaker_init()
// only for its side effect (enabling the AXP2101 codec rails and the AW9523
// "enable codec" GPIO bit) and builds its own codec_if with pa_gain=1 on
// top, always at 24000 Hz.
//
// The codec is opened once in begin() and left open - opening/closing it
// per sound while MicrophoneEngine has the shared I2S bus's RX side
// continuously running caused repeated "Pending out channel for in channel
// running" bus reconfiguration, which fed a glitch into the mic's readings
// often enough to retrigger the robot's own "loud sound" reaction in a
// tight loop and eventually crash it. Reusing one open output channel for
// every chirp avoids that entirely.
class AudioOutput {
public:
    bool begin();  // must run after bsp_audio_init() with a stereo I2S config
    void playPcm(const int16_t* samples, size_t count, int volume = 90);

    // The owner's volume, 0 (silent) .. kLevels, set on the screen (FaceEngine's speaker button) and kept in NVS "v4"
    // "vol" (2026-10-25). kDefaultLevel is the codec volume 90 every sound had before.
    static constexpr int kLevels = 10, kDefaultLevel = 8;
    void setLevel(int level) { level_.store(std::clamp(level, 0, kLevels)); }
    int level() const { return level_.load(); }
    // the codec volume (0..100) of the current level: the codec's scale is in dB, so the low steps are farther apart
    int volume() const
    {
        static constexpr int8_t kVolume[kLevels + 1] = {0, 45, 55, 63, 70, 76, 81, 86, 90, 95, 100};
        return kVolume[level_.load()];
    }

    // What was really played since the last call (the SELF_SOUND sense).
    SelfSoundAnalyzer::Features takeSelfSound() { return self_sound_.take(); }

    static constexpr uint32_t kSampleRate = 24000;

private:
    esp_codec_dev_handle_t speaker_ = nullptr;
    int volume_ = -1;  // last volume written to the amplifier
    std::atomic<int> level_{kDefaultLevel};
    SelfSoundAnalyzer self_sound_;
};
