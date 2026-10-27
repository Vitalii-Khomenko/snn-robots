#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

// Is it a voice? (docs/CONTINGENCY.md step 2; tools/v4/voice_check.py is its twin.) The Auditory NN's speech class
// fires ~250 times an hour on the robot's own sounds - the tail of its voice, the servos' whine (~6 kHz) and clicks
// (recordings of 2026-10-27). A human voice is periodic at its pitch (80-400 Hz) and stands above the room:
//
//   mono 24 kHz -> high-pass 70 Hz, low-pass 900 Hz twice -> every 6th sample (4 kHz) -> every 25 ms over the last
//   50 ms: periodicity = the best normalized autocorrelation at lags 10..50 (400..80 Hz), level in dB; the room's
//   floor follows the level down at once and up slowly (~12 s); a voiced frame = periodicity > 0.6 and level >
//   floor + 6 dB; voice() = a ~250 ms average of voiced frames (0..1).
//
// feed() runs on the microphone task (~1 % of a core), voice() is read by the brain loop.
class VoiceCheck {
public:
    VoiceCheck();
    void feed(const int16_t* mono, size_t n);
    float voice() const { return voice_out_.load(std::memory_order_relaxed); }
    // the last frame (for the twin's parity test)
    float periodicity() const { return last_periodicity_; }
    float levelDb() const { return last_level_; }

    static constexpr int kDecim = 6, kHop = 100, kWin = 200, kLagLo = 10, kLagHi = 50, kRing = 256;

private:
    float frame();

    struct Biquad {
        float b0, b1, b2, a1, a2, z1 = 0, z2 = 0;
    };
    std::array<Biquad, 3> stages_{};
    std::array<float, kRing> ring_{};
    std::array<float, kWin + kLagHi> win_{};  // frame()'s window: here, not on the microphone task's 4 KB stack
    uint32_t n_ = 0;
    int phase_ = 0;
    bool floor_set_ = false;
    float floor_ = 0, voice_ = 0, last_periodicity_ = 0, last_level_ = -120;
    std::atomic<float> voice_out_{0.0f};
};
