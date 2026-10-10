#include "droid_synth.h"

#include <algorithm>
#include <cmath>

namespace droid {
namespace {
constexpr float kPi = 3.14159265358979f;
constexpr float kTwoPi = 2.0f * kPi;
constexpr float kBandLimitHz = 10500.0f;  // nothing above this is generated (Nyquist is 12 kHz)
constexpr float kMinHz = 280.0f, kMaxHz = 3600.0f;
constexpr int kFlickMs = 55;

float smooth(float x) { return x <= 0.0f ? 0.0f : x >= 1.0f ? 1.0f : x * x * (3.0f - 2.0f * x); }

// 0..1 -> 0..1 glide curves
float curve(Shape s, float u)
{
    switch (s) {
        case S_FAST: return 1.0f - std::pow(1.0f - u, 2.6f);
        case S_SLOW: return std::pow(u, 2.6f);
        case S_SCOOP: return u + 0.9f * std::sin(kPi * u) * (1.0f - u) * -1.0f * 1.2f;  // dips first, then arrives
        default: return u;
    }
}

float modRatio(Kind k)
{
    switch (k) {
        case K_WHISTLE: return 1.004f;  // beating shimmer
        case K_CHIRP: return 2.0f;
        case K_PIP: return 1.5f;        // inharmonic: metallic
        case K_BURBLE: return 3.01f;
        case K_WARBLE: return 1.004f;
        case K_GIGGLE: return 2.0f;
        case K_SQUEAL: return 2.0f;
        case K_GROWL: return 0.5f;      // subharmonic: rough
        case K_COO: return 1.004f;
        default: return 1.5f;
    }
}

float fmMax(Kind k)
{
    switch (k) {
        case K_WHISTLE: return 0.9f;
        case K_CHIRP: return 3.0f;
        case K_PIP: return 4.0f;
        case K_BURBLE: return 5.0f;
        case K_WARBLE: return 0.9f;
        case K_GIGGLE: return 2.5f;
        case K_SQUEAL: return 4.5f;
        case K_GROWL: return 3.0f;
        case K_COO: return 0.35f;
        default: return 3.0f;
    }
}
}  // namespace

uint32_t Synth::rnd()
{
    rng_ ^= rng_ << 13;
    rng_ ^= rng_ >> 17;
    rng_ ^= rng_ << 5;
    return rng_;
}

void Synth::clear()
{
    head_ = count_ = 0;
    queued_samples_ = 0;
    playing_ = false;
    hp_x_ = hp_y_ = 0.0f;
}

int Synth::queuedMs() const
{
    const int remaining = playing_ ? std::max(0, total_ + gap_ - n_) : 0;
    return static_cast<int>((queued_samples_ + remaining) * 1000.0f / kSampleRate);
}

bool Synth::enqueue(const Note& n, float baseHz)
{
    if (count_ >= kCapacity) return false;
    q_[(head_ + count_) % kCapacity] = {n, baseHz};
    ++count_;
    queued_samples_ += static_cast<int>((n.durMs + n.gapMs) * kSampleRate / 1000.0f);
    return true;
}

void Synth::start(const Item& it)
{
    cur_ = it;
    const Note& n = it.note;
    total_ = std::max(1, static_cast<int>(n.durMs * kSampleRate / 1000.0f));
    gap_ = static_cast<int>(n.gapMs * kSampleRate / 1000.0f);
    n_ = 0;
    phase_ = mphase_ = 0.0f;
    vib_phase_ = 0.0f;
    lfo_phase_ = 0.0f;
    step_n_ = 0;
    step_len_ = 1;
    step_pitch_ = step_target_ = n.kind == K_GROWL ? 0.0f : n.f0St;
    step_amp_ = 1.0f;
    playing_ = true;
}

// Pitch in semitones above the base at note position u (0..1), t seconds in.
float Synth::pitchSemis(float u, float tSec) const
{
    const Note& n = cur_.note;
    float p = n.f0St + n.slopeSt * curve(n.shape, u);
    if (n.kind == K_PIP) p += 4.0f * std::exp(-tSec * 260.0f);  // the click: a quick drop onto the pitch
    if (n.ending != E_NONE) {
        const float flick = static_cast<float>(kFlickMs) / std::max(1.0f, n.durMs);
        const float v = (u - (1.0f - flick)) / flick;
        p += (n.ending == E_UP ? 8.0f : -8.0f) * smooth(v);
    }
    return p;
}

float Synth::sampleOne()
{
    if (!playing_) {
        if (count_ == 0) return 0.0f;
        const Item it = q_[head_];
        head_ = (head_ + 1) % kCapacity;
        --count_;
        queued_samples_ -= static_cast<int>((it.note.durMs + it.note.gapMs) * kSampleRate / 1000.0f);
        if (queued_samples_ < 0) queued_samples_ = 0;
        start(it);
    }
    if (n_ >= total_) {
        // the gap after the note
        if (++n_ >= total_ + gap_) playing_ = false;
        return 0.0f;
    }
    const Note& n = cur_.note;
    const float t = n_ / kSampleRate;
    const float u = static_cast<float>(n_) / static_cast<float>(total_);
    const float msLeft = (total_ - n_) * 1000.0f / kSampleRate;
    float semis = pitchSemis(u, t);
    float amp = 1.0f;
    float fmIndexScale = 1.0f;
    float harm = n.harm;

    switch (n.kind) {
        case kKinds: break;
        case K_WHISTLE: {
            vib_phase_ += (5.0f + 2.5f * n.rate) / kSampleRate;
            if (vib_phase_ >= 1.0f) vib_phase_ -= 1.0f;
            semis += n.vib * 0.9f * std::sin(kTwoPi * vib_phase_) * smooth(t / 0.12f);
            amp = smooth(t / 0.010f) * smooth(msLeft / 30.0f) * (0.92f + 0.08f * std::sin(kTwoPi * vib_phase_ * 0.5f));
            harm *= 0.5f;
            fmIndexScale = 0.7f + 0.5f * u;
            break;
        }
        case K_CHIRP: {
            amp = smooth(t / 0.003f) * smooth(msLeft / 25.0f) * (1.0f - 0.55f * u);
            fmIndexScale = 1.0f - 0.5f * u;
            break;
        }
        case K_PIP: {
            amp = smooth(t / 0.0012f) * std::exp(-4.2f * u) * smooth(msLeft / 6.0f);
            fmIndexScale = std::exp(-3.0f * u);
            harm = std::min(1.0f, harm + 0.3f);
            break;
        }
        case K_BURBLE: {
            if (step_n_ <= 0) {
                const float stepsPerSec = 7.0f + 17.0f * n.rate;
                step_len_ = std::max(static_cast<int>(kSampleRate / stepsPerSec * (0.75f + 0.5f * (rnd() & 1023) / 1023.0f)), 200);
                step_n_ = step_len_;
                const float r = (static_cast<float>(rnd() & 0xffff) / 65535.0f) - 0.5f;
                step_target_ = n.f0St + n.slopeSt * 0.5f + r * (std::fabs(n.slopeSt) + 6.0f);
                step_amp_ = 0.65f + 0.35f * (static_cast<float>(rnd() & 255) / 255.0f);
            }
            const int k = step_len_ - step_n_;
            --step_n_;
            step_pitch_ += (step_target_ - step_pitch_) * 0.012f;  // ~6 ms portamento
            semis = step_pitch_;
            const float kt = static_cast<float>(k) / static_cast<float>(step_len_);
            amp = smooth(k / (0.003f * kSampleRate)) * (1.0f - 0.35f * kt) * smooth((step_len_ - k) / (0.004f * kSampleRate)) * step_amp_ *
                  smooth(t / 0.004f) * smooth(msLeft / 12.0f);
            fmIndexScale = 1.0f - 0.4f * kt;
            break;
        }
        case K_WARBLE: {
            vib_phase_ += (4.0f + 9.0f * n.rate) / kSampleRate;
            if (vib_phase_ >= 1.0f) vib_phase_ -= 1.0f;
            semis += (1.5f + 5.0f * n.vib) * std::sin(kTwoPi * vib_phase_) * smooth(t / 0.05f);
            amp = smooth(t / 0.010f) * smooth(msLeft / 40.0f);
            harm *= 0.5f;
            break;
        }
        case K_GIGGLE: {
            const float pulses = 6.0f + 10.0f * n.rate;
            lfo_phase_ += pulses / kSampleRate;
            if (lfo_phase_ >= 1.0f) lfo_phase_ -= 1.0f;
            const float q = lfo_phase_;                       // position inside one blip
            semis += -3.0f + 6.0f * smooth(q / 0.7f);         // each blip rises
            amp = smooth(q / 0.06f) * (q < 0.78f ? 1.0f - 0.35f * q : smooth((0.9f - q) / 0.12f)) * smooth(t / 0.004f) * smooth(msLeft / 20.0f);
            fmIndexScale = 1.0f - 0.5f * q;
            break;
        }
        case K_SQUEAL: {
            vib_phase_ += 26.0f / kSampleRate;
            if (vib_phase_ >= 1.0f) vib_phase_ -= 1.0f;
            semis += 0.6f * std::sin(kTwoPi * vib_phase_);
            amp = smooth(t / 0.004f) * smooth(msLeft / 22.0f) * (1.0f - 0.25f * u);
            harm = std::max(harm, 0.75f);
            break;
        }
        case K_GROWL: {
            lfo_phase_ += (35.0f + 40.0f * n.rate) / kSampleRate;
            if (lfo_phase_ >= 1.0f) lfo_phase_ -= 1.0f;
            if ((n_ & 255) == 0) step_target_ = (static_cast<float>(rnd() & 0xffff) / 65535.0f - 0.5f) * 0.9f;  // slow random pitch jitter
            step_pitch_ += (step_target_ - step_pitch_) * 0.03f;
            semis += step_pitch_;
            amp = smooth(t / 0.015f) * smooth(msLeft / 40.0f) * (1.0f - 0.65f * (0.5f - 0.5f * std::cos(kTwoPi * lfo_phase_)));
            harm = std::max(harm, 0.8f);
            break;
        }
        case K_COO: {
            vib_phase_ += (4.0f + 2.0f * n.rate) / kSampleRate;
            if (vib_phase_ >= 1.0f) vib_phase_ -= 1.0f;
            semis += (0.2f + 0.6f * n.vib) * std::sin(kTwoPi * vib_phase_) * smooth(t / 0.15f);
            amp = 0.8f * smooth(t / 0.045f) * smooth(msLeft / 70.0f) * (0.85f + 0.15f * std::sin(kPi * u));
            harm *= 0.3f;
            break;
        }
        case K_TRILL: {
            const float hz = 20.0f + 25.0f * n.rate;
            lfo_phase_ += hz / kSampleRate;
            if (lfo_phase_ >= 1.0f) lfo_phase_ -= 1.0f;
            const float lfo = std::sin(kTwoPi * lfo_phase_);
            semis += (1.5f + 0.2f * std::fabs(n.slopeSt)) * lfo;
            amp = smooth(t / 0.008f) * smooth(msLeft / 25.0f) * (1.0f - 0.32f * (0.5f - 0.5f * std::cos(kTwoPi * lfo_phase_)));
            break;
        }
    }

    const float hz = std::clamp(cur_.baseHz * std::exp2(semis / 12.0f), kMinHz, kMaxHz);
    phase_ += hz / kSampleRate;
    if (phase_ >= 1.0f) phase_ -= 1.0f;
    const float ratio = modRatio(n.kind);
    mphase_ += hz * ratio / kSampleRate;
    if (mphase_ >= 1.0f) mphase_ -= 1.0f;

    // modulation index: never let a significant sideband (f * (1 + ratio * (index + 1))) pass the band limit
    float index = n.fm * fmMax(n.kind) * fmIndexScale;
    const float maxIndex = std::max(0.0f, (kBandLimitHz / hz - 1.0f) / ratio - 1.0f);
    index = std::min(index, maxIndex);

    const float ph = kTwoPi * phase_;
    float s = std::sin(ph + index * std::sin(kTwoPi * mphase_));
    // band-limited harmonics, anchored to the carrier phase
    if (2.0f * hz < kBandLimitHz) s += 0.55f * harm * std::sin(2.0f * ph + 0.8f * index * std::sin(kTwoPi * mphase_));
    if (3.0f * hz < kBandLimitHz) s += 0.33f * harm * std::sin(3.0f * ph);
    s *= amp * n.loud / (1.0f + 0.55f * harm);

    ++n_;
    return s;
}

void Synth::render(int16_t* out, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        float s = sampleOne();
        // high-pass ~280 Hz: the little speaker cannot play lower, and the energy would only distort
        const float y = 0.9265f * (hp_y_ + s - hp_x_);
        hp_x_ = s;
        hp_y_ = y;
        const float v = std::tanh(1.3f * y) * level;
        out[i] = static_cast<int16_t>(std::clamp(v, -32767.0f, 32767.0f));
    }
}

}  // namespace droid
