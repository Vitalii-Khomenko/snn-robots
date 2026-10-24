#include "motivation.h"

#include <algorithm>
#include <cmath>

using namespace hybrid;

namespace {
constexpr float kDt = 0.05f;
// the temperament: what each need is worth, curiosity, the cost of moving (the plan's starting values)
constexpr float kWeight[Motivation::kNeeds] = {0.5f, 1.0f, 0.7f, 0.5f, 1.5f};
constexpr float kMoveCost = 0.0002f;  // per degree the head travels: wear, and the robot is deaf while it moves
// (0.002 until 2026-10-14: a day's reward was almost all movement cost; tools/life/temperament.py uses the same)
// constexpr float kCuriosity = 0.3f;   // beta: learning progress, from the world model (stage 1)

float dz(float v, float th) { return std::max(0.0f, (v - th) / (1.0f - th)); }
float decay(float tau) { return std::exp(-kDt / tau); }
}  // namespace

const char* Motivation::needName(int i)
{
    static const char* const names[kNeeds] = {"energy", "social", "stimulation", "rest", "comfort"};
    return i >= 0 && i < kNeeds ? names[i] : "?";
}

void Motivation::tick(const Bus& bus)
{
    const auto& v = bus.vision;
    const auto& a = bus.audio;
    const auto& b = bus.body;
    const float night = bus.ctx[CTX_NIGHT];
    std::array<float, kNeeds> before = d_;

    // energy: straight from the battery
    d_[Energy] = std::clamp((0.8f - b[BODY_BATTERY]) / 0.7f, 0.0f, 1.0f);

    // social: grows alone, falls with attention (a touch fast, a voice with somebody there slower, a face slowly)
    const float touch = std::max(b[BODY_TOUCH_HEAD], b[BODY_TOUCH_SCREEN]);
    const float someone = std::max(v[VIS_PERSON], v[VIS_FACE]);
    const float talked_to = a[AUD_OWN_MOTION] > 0.5f ? 0.0f : dz(a[AUD_SPEECH], 0.4f) * (someone > 0.4f ? 1.0f : 0.5f);
    const float contact = std::max({touch, talked_to, 0.3f * dz(v[VIS_FACE], 0.5f)});
    float& social = d_[Social];
    social += kDt / 1800.0f * (1.0f - social) * (1.0f - contact) * (1.0f - 0.7f * night);
    if (touch > 0.5f) social *= decay(2.0f);
    social *= 1.0f - (1.0f - decay(30.0f)) * dz(talked_to, 0.0f);
    social *= 1.0f - (1.0f - decay(300.0f)) * dz(v[VIS_FACE], 0.5f);

    // stimulation: something new is happening (stand-in for the learning progress of the world model, stage 1)
    // its own voice and its own servos are not news: with 0.3 they counted a little, and the actor learned to call
    // ~6 times a minute to ease its boredom by itself (2026-10-15)
    const float self_heard = a[AUD_SELF_VOICE] > 0.5f || a[AUD_OWN_MOTION] > 0.5f ? 0.0f : 1.0f;
    novelty_ = std::clamp(std::max({dz(a[AUD_NOVELTY], 0.3f) * self_heard, dz(v[VIS_ACTIVITY], 0.2f),
                                    dz(std::fabs(v[VIS_LIGHT_CHANGE]), 0.2f), dz(bus.pe[PE_VISION], 0.3f), touch}),
                          0.0f, 1.0f);
    // curiosity: the world model was surprised - its error well above what is usual (the habitual level adapts in
    // ~1 min, so steady noise, a television, stops being a surprise; what the nightly sleep teaches it stops too).
    // 2026-10-15 (diary): vision RMS still ~0.003, turning ~0.007, the top tenth 0.04+.
    {
        constexpr float kHab = kDt / 60.0f;
        if (hab_vision_ <= 0.0f) hab_vision_ = std::max(wm_vision_, 1e-3f), hab_audio_ = std::max(wm_audio_, 1e-3f);
        const float sv = std::clamp((wm_vision_ / std::max(hab_vision_, 1e-3f) - 1.5f) / 4.0f, 0.0f, 1.0f);
        const bool own = a[AUD_SELF_VOICE] > 0.5f || a[AUD_OWN_MOTION] > 0.5f;
        const float sa = own ? 0.0f : 0.5f * std::clamp((wm_audio_ / std::max(hab_audio_, 1e-3f) - 1.5f) / 4.0f, 0.0f, 1.0f);
        curiosity_ = std::max(sv, sa);
        hab_vision_ += kHab * (wm_vision_ - hab_vision_);
        hab_audio_ += kHab * (wm_audio_ - hab_audio_);
        novelty_ = std::max(novelty_, curiosity_);
    }
    float& stim = d_[Stimulation];
    stim += kDt / 300.0f * (1.0f - stim) * (1.0f - novelty_) * (1.0f - 0.8f * night);
    stim *= 1.0f - (1.0f - decay(10.0f)) * novelty_;

    // rest: the servos' fatigue and the night
    d_[Rest] = std::clamp(1.0f - b[BODY_STAMINA] + 0.4f * night, 0.0f, 1.0f);

    // comfort: being shaken, lifted, tilted; very loud sounds that are not our own
    float& comfort = d_[Comfort];
    comfort += kDt * (2.0f * std::max(b[BODY_SHAKE], b[BODY_TILT]) +
                      (a[AUD_OWN_MOTION] > 0.5f ? 0.0f : 0.5f * dz(a[AUD_LOUD], 0.5f)));
    comfort = std::clamp(comfort * decay(60.0f), 0.0f, 1.0f);

    for (float& x : d_) x = std::clamp(x, 0.0f, 1.0f);

    // the head's movement this tick (BodySense: yaw / 100 deg, pitch / 35 deg)
    const float yaw = b[BODY_YAW] * 100.0f, pitch = b[BODY_PITCH] * 35.0f;
    move_deg_ = started_ ? std::hypot(yaw - prev_yaw_, pitch - prev_pitch_) : 0.0f;
    prev_yaw_ = yaw;
    prev_pitch_ = pitch;

    float r = -kMoveCost * move_deg_;
    if (started_)
        for (int i = 0; i < kNeeds; ++i) r += kWeight[i] * (before[i] * before[i] - d_[i] * d_[i]);
    // + kCuriosity * learning progress, once the world model exists (stage 1)
    r_ = r;
    started_ = true;
}

void Motivation::logValues(float* out) const
{
    for (int i = 0; i < kNeeds; ++i) out[i] = d_[i];
    out[kNeeds] = r_;
    out[kNeeds + 1] = novelty_;
    out[kNeeds + 2] = move_deg_ / 100.0f;
}

void Motivation::restore(const std::array<float, kNeeds>& d)
{
    for (int i = 0; i < kNeeds; ++i)
        if (std::isfinite(d[i])) d_[i] = std::clamp(d[i], 0.0f, 1.0f);
}
