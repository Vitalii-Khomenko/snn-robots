#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

// Microphone blocks -> the Auditory NN's input: a 16-band cochlea-like
// filter bank (log-spaced band-pass biquads, 120 Hz .. 7.5 kHz at 24 kHz).
// The mic task feeds every 128-frame block; every 4 blocks (21.3 ms) one
// frame of 16 log band levels plus the left/right balance is published into
// a ring. The main loop reads the last 11 frames (~230 ms) when the brain
// ticks (single producer / single consumer, no lock).
class AudioFrontEnd {
public:
    static constexpr int kBands = 16;
    static constexpr int kFrameBlocks = 4;       // blocks per frame
    static constexpr int kHistory = 6;           // frames fed to the net
    static constexpr int kStride = 2;            // ... every second frame (~43 ms apart)
    static constexpr int kSpan = (kHistory - 1) * kStride + 1;  // frames needed
    static constexpr int kInputs = kBands * kHistory + 4;       // + balance, energy, self flag, motion flag
    static constexpr float kSampleRate = 24000.0f;

    struct Frame {
        std::array<float, kBands> band{};  // 0..~1.3, log of band RMS
        float balance = 0.0f;              // (R-L)/(R+L), -1..1
        float rms = 0.0f;                  // mono RMS, int16 units
        float peak = 0.0f;                 // int16 units
    };

    AudioFrontEnd();
    // Mic task: one stereo block (interleaved L R).
    void processBlock(const int16_t* stereo, size_t frames);
    // Main loop: the last kSpan frames, oldest first. false until enough
    // audio has arrived.
    bool latest(std::array<Frame, kSpan>& out) const;
    uint32_t frameCount() const { return head_.load(std::memory_order_acquire); }
    // Cost of one 128-frame block in microseconds (EMA, wall clock: includes preemption).
    uint32_t blockUs() const { return block_us_.load(); }

    // Net input from the frames plus the two flags the net needs to tell the
    // outside world from the robot itself.
    static void buildInput(const std::array<Frame, kSpan>& frames, bool selfSpeaking, bool ownMotion,
                           std::array<float, kInputs>& x);

private:
    struct Biquad {
        float b0 = 0, a1 = 0, a2 = 0, z1 = 0, z2 = 0;  // band-pass, b1 = 0, b2 = -b0
    };
    std::array<Biquad, kBands> bands_{};
    std::array<float, kBands> energy_{};
    float left_ = 0, right_ = 0, peak_ = 0, mono_sq_ = 0;
    int blocks_ = 0;
    size_t samples_ = 0;
    static constexpr int kRing = 16;
    std::array<Frame, kRing> ring_{};
    std::atomic<uint32_t> head_{0};
    std::atomic<uint32_t> block_us_{0};
};
