#pragma once
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

    // What was really played since the last call (the SELF_SOUND sense).
    SelfSoundAnalyzer::Features takeSelfSound() { return self_sound_.take(); }

    static constexpr uint32_t kSampleRate = 24000;

private:
    esp_codec_dev_handle_t speaker_ = nullptr;
    int volume_ = -1;  // last volume written to the amplifier
    SelfSoundAnalyzer self_sound_;
};
