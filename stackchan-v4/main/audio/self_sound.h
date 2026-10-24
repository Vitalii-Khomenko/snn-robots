#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>

// What the speaker ACTUALLY produced, measured from the PCM on its way to
// the codec (AudioOutput::playPcm) - calls, songs, anything. Once per brain
// tick the runtime takes the eight features for the SELF_SOUND cells:
//
//   0 loudness (RMS)          4 high-band energy (> ~900 Hz)
//   1 voiced (periodic) part  5 pitch, normalized log 150..2400 Hz
//   2 noise (aperiodic) part  6 pitch motion since the last tick
//   3 low-band energy         7 sounding at all
//
// Pitch comes from zero crossings of a low-passed copy, periodicity from
// the normalized autocorrelation at that period: no knowledge of what the
// synthesizer was asked to do - "that was me" is heard, not assumed.
class SelfSoundAnalyzer {
public:
    static constexpr float kSampleRate = 24000.0f;
    // Level normalization; follows the vocal tract's output level
    // (x1.7 on 2026-09-30) so the SELF_SOUND features keep their scale.
    static constexpr float kScale = 1.7f;
    using Features = std::array<float, 8>;

    void process(const int16_t* x, size_t n)
    {
        if (n < 32) return;
        double sq = 0.0, lo = 0.0, hi = 0.0;
        int crossings = 0;
        bool nonzero = false;
        for (size_t i = 0; i < n; ++i) {
            const float s = x[i];
            nonzero |= x[i] != 0;
            sq += s * s;
            lp_ += (s - lp_) * 0.21f;    // ~900 Hz one-pole: low band
            zlp_ += (s - zlp_) * 0.45f;  // ~2.4 kHz: for zero crossings
            lo += lp_ * lp_;
            hi += (s - lp_) * (s - lp_);
            const bool pos = zlp_ >= 0.0f;
            if (pos != last_pos_) ++crossings;
            last_pos_ = pos;
        }
        if (!nonzero) return;
        const float rms = std::sqrt(static_cast<float>(sq / n));
        const float f0 = crossings * 0.5f * kSampleRate / static_cast<float>(n);
        // Periodicity: autocorrelation at the zero-crossing period.
        float periodic = 0.0f;
        const int lag = f0 > 40.0f ? static_cast<int>(kSampleRate / f0 + 0.5f) : 0;
        if (lag > 2 && static_cast<size_t>(lag) < n / 2) {
            double c = 0.0, e0 = 0.0, e1 = 0.0;
            for (size_t i = static_cast<size_t>(lag); i < n; ++i) {
                c += static_cast<double>(x[i]) * x[i - lag];
                e0 += static_cast<double>(x[i]) * x[i];
                e1 += static_cast<double>(x[i - lag]) * x[i - lag];
            }
            if (e0 > 0 && e1 > 0) periodic = std::clamp(static_cast<float>(c / std::sqrt(e0 * e1)), 0.0f, 1.0f);
        }
        const float w = static_cast<float>(n);
        // process() runs in the voice task, take() in the main loop (another core)
        std::lock_guard<std::mutex> lock(mutex_);
        acc_rms_ += rms * w;
        acc_lo_ += std::sqrt(static_cast<float>(lo / n)) * w;
        acc_hi_ += std::sqrt(static_cast<float>(hi / n)) * w;
        acc_periodic_ += periodic * w;
        acc_f0_ += f0 * w;
        acc_n_ += w;
    }

    // The features since the last call (zeros when nothing was played).
    Features take()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        Features f{};
        if (acc_n_ > 0.0f) {
            const float rms = acc_rms_ / acc_n_;
            const float per = acc_periodic_ / acc_n_;
            const float level = std::clamp(rms / (2500.0f * kScale), 0.0f, 1.0f);
            f[0] = level;
            f[1] = level * per;
            f[2] = level * (1.0f - per);
            f[3] = std::clamp(acc_lo_ / acc_n_ / (2000.0f * kScale), 0.0f, 1.0f);
            f[4] = std::clamp(acc_hi_ / acc_n_ / (2000.0f * kScale), 0.0f, 1.0f);
            const float f0 = acc_f0_ / acc_n_;
            const float pitch = per > 0.4f && f0 > 60.0f ? std::clamp(std::log2(f0 / 150.0f) / 4.0f, 0.0f, 1.0f) : 0.0f;
            f[5] = pitch;
            f[6] = pitch > 0.0f && last_pitch_ > 0.0f ? std::clamp(std::fabs(pitch - last_pitch_) * 8.0f, 0.0f, 1.0f) : 0.0f;
            f[7] = 1.0f;
            last_pitch_ = pitch;
        } else {
            last_pitch_ = 0.0f;
        }
        acc_rms_ = acc_lo_ = acc_hi_ = acc_periodic_ = acc_f0_ = acc_n_ = 0.0f;
        return f;
    }

private:
    std::mutex mutex_;  // guards the accumulators and last_pitch_
    float lp_ = 0.0f, zlp_ = 0.0f;
    bool last_pos_ = false;
    float acc_rms_ = 0.0f, acc_lo_ = 0.0f, acc_hi_ = 0.0f, acc_periodic_ = 0.0f, acc_f0_ = 0.0f, acc_n_ = 0.0f;
    float last_pitch_ = 0.0f;
};
