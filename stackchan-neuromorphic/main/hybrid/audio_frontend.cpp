#include "audio_frontend.h"

#include <algorithm>
#include <cmath>

#include <esp_timer.h>

AudioFrontEnd::AudioFrontEnd()
{
    constexpr float lo = 120.0f, hi = 7500.0f, q = 3.0f;
    for (int b = 0; b < kBands; ++b) {
        const float centre = lo * std::pow(hi / lo, (static_cast<float>(b) + 0.5f) / kBands);
        const float w = 6.283185307f * centre / kSampleRate;
        const float alpha = std::sin(w) / (2.0f * q);
        const float a0 = 1.0f + alpha;
        bands_[b].b0 = alpha / a0;
        bands_[b].a1 = -2.0f * std::cos(w) / a0;
        bands_[b].a2 = (1.0f - alpha) / a0;
    }
}

void AudioFrontEnd::processBlock(const int16_t* stereo, size_t frames)
{
    const int64_t t0 = esp_timer_get_time();
    for (size_t i = 0; i < frames; ++i) {
        const float l = stereo[2 * i], r = stereo[2 * i + 1];
        const float x = 0.5f * (l + r) * (1.0f / 32768.0f);
        left_ += std::fabs(l);
        right_ += std::fabs(r);
        peak_ = std::max(peak_, std::max(std::fabs(l), std::fabs(r)));
        mono_sq_ += x * x;
        for (int b = 0; b < kBands; ++b) {
            Biquad& f = bands_[b];
            const float y = f.b0 * x + f.z1;
            f.z1 = -f.a1 * y + f.z2;
            f.z2 = -f.b0 * x - f.a2 * y;
            energy_[b] += y * y;
        }
    }
    samples_ += frames;
    {
        const uint32_t us = static_cast<uint32_t>(esp_timer_get_time() - t0);
        const uint32_t prev = block_us_.load(std::memory_order_relaxed);
        block_us_.store(prev ? (prev * 15 + us) / 16 : us, std::memory_order_relaxed);
    }
    if (++blocks_ < kFrameBlocks) return;

    Frame& f = ring_[head_.load(std::memory_order_relaxed) % kRing];
    const float n = static_cast<float>(samples_);
    for (int b = 0; b < kBands; ++b) {
        const float rms = std::sqrt(energy_[b] / n) * 32768.0f;  // int16 units
        f.band[b] = std::log10(1.0f + rms * (1.0f / 8.0f)) * (1.0f / 3.0f);
        energy_[b] = 0.0f;
    }
    f.balance = left_ + right_ > 0.0f ? (right_ - left_) / (left_ + right_) : 0.0f;
    f.rms = std::sqrt(mono_sq_ / n) * 32768.0f;
    f.peak = peak_;
    left_ = right_ = peak_ = mono_sq_ = 0.0f;
    blocks_ = 0;
    samples_ = 0;
    head_.store(head_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
}

bool AudioFrontEnd::latest(std::array<Frame, kSpan>& out) const
{
    const uint32_t head = head_.load(std::memory_order_acquire);
    if (head < static_cast<uint32_t>(kSpan)) return false;
    for (int i = 0; i < kSpan; ++i) out[i] = ring_[(head - kSpan + i) % kRing];
    // A frame written meanwhile can only have overwritten the oldest slot.
    return true;
}

void AudioFrontEnd::buildInput(const std::array<Frame, kSpan>& frames, bool selfSpeaking, bool ownMotion,
                               std::array<float, kInputs>& x)
{
    // History slot h = frame (latest - h * kStride); slot 0 is the newest.
    for (int h = 0; h < kHistory; ++h) {
        const Frame& f = frames[kSpan - 1 - h * kStride];
        for (int b = 0; b < kBands; ++b) x[h * kBands + b] = f.band[b];
    }
    float wsum = 0.0f, bal = 0.0f, rms = 0.0f;
    for (int i = kSpan - 4; i < kSpan; ++i) {
        const float w = frames[i].rms + 1.0f;
        bal += frames[i].balance * w;
        wsum += w;
        rms = std::max(rms, frames[i].rms);
    }
    x[kBands * kHistory + 0] = std::clamp(bal / wsum, -1.0f, 1.0f);
    x[kBands * kHistory + 1] = std::clamp(std::log10(1.0f + rms * (1.0f / 8.0f)) * (1.0f / 3.0f), 0.0f, 1.5f);
    x[kBands * kHistory + 2] = selfSpeaking ? 1.0f : 0.0f;
    x[kBands * kHistory + 3] = ownMotion ? 1.0f : 0.0f;
}
