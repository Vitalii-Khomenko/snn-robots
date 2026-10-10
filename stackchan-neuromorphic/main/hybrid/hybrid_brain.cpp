#include "hybrid_brain.h"

#include <algorithm>
#include <cmath>

#include <esp_random.h>
#include <esp_timer.h>

using namespace hybrid;

HybridBrain::HybridBrain()
{
    rng_ = esp_random() | 1u;
    snn_.setClock([]() -> int64_t { return esp_timer_get_time(); });
    snn_.seed(esp_random() | 1u);
}

uint32_t HybridBrain::rnd()
{
    rng_ ^= rng_ << 13;
    rng_ ^= rng_ >> 17;
    rng_ ^= rng_ << 5;
    return rng_;
}

float HybridBrain::gauss()
{
    constexpr float k = 1.0f / 16777216.0f;
    float s = 0.0f;
    for (int i = 0; i < 4; ++i) s += static_cast<float>(rnd() >> 8) * k;
    return (s - 2.0f) * 1.7320508f;
}

// Endogenous pacemakers (tools/hybrid/synth.py ou()): Ornstein-Uhlenbeck
// drifts with time constants 6 s, 8 s, 90 s and 0.6 s.
void HybridBrain::updateNoise()
{
    static constexpr float kTau[NOISE_COUNT] = {6.0f, 8.0f, 90.0f, 0.6f};
    static constexpr float kSigma[NOISE_COUNT] = {0.5f, 0.4f, 0.5f, 0.4f};
    for (int i = 0; i < NOISE_COUNT; ++i) {
        const float a = std::exp(-0.05f / kTau[i]);
        ou_[i] = a * ou_[i] + kSigma[i] * std::sqrt(1.0f - a * a) * gauss();
        bus_.noise[i] = std::clamp(ou_[i], -1.0f, 1.0f);
    }
}

void HybridBrain::updatePhysiology()
{
    const auto& s = bus_.snn;
    Physiology& p = physio_;
    const float arousal = std::clamp(0.7f * s[SNN_AROUSAL] + 0.3f * s[SNN_ALERT], 0.0f, 1.0f);
    float bpm = 58.0f + 62.0f * std::pow(arousal, 1.3f) + 14.0f * s[SNN_CURIOSITY];
    const float sleep = std::clamp(s[SNN_SLEEP] * 1.4f, 0.0f, 1.0f);
    bpm = bpm * (1.0f - sleep) + 40.0f * sleep;
    bpm += 4.0f * std::clamp(load_, 0.0f, 1.0f);  // the load only nudges, it must not dominate
    p.heartBpm += (bpm - p.heartBpm) * 0.03f;
    const float breath = 10.0f + 12.0f * arousal;
    p.breathPerMin += (breath * (1.0f - 0.35f * sleep) - p.breathPerMin) * 0.03f;
    p.heartPhase += p.heartBpm / 60.0f * 0.05f;
    p.heartBeat = false;
    if (p.heartPhase >= 1.0f) {
        p.heartPhase -= 1.0f;
        p.heartBeat = true;
    }
}

// The slow needs, exactly (the same dynamics as tools/hybrid/teacher_drives.py, where the spiking core's boredom and social need
// come from; the core cannot count minutes of silence, so the policy gets these two as inputs). Weak sensor noise (below 0.2) is not
// stimulation and not company: in an empty quiet room boredom climbs to ~0.8 in a quarter of an hour, social need follows.
void HybridBrain::updateSlow()
{
    constexpr float dt = 0.05f;
    auto dz = [](float v, float th = 0.2f) { return std::max(0.0f, (v - th) / (1.0f - th)); };
    const auto& v = bus_.vision;
    const auto& a = bus_.audio;
    const auto& s = bus_.snn;
    const float touch = std::max(bus_.body[BODY_TOUCH_HEAD], bus_.body[BODY_TOUCH_SCREEN]);
    const float vis_ev = std::max({v[VIS_ACTIVITY], 0.7f * std::fabs(v[VIS_LIGHT_CHANGE]), 0.8f * v[VIS_LOOM]});
    const float loud = std::max({a[AUD_LOUD], 0.9f * a[AUD_CLAP], a[AUD_SHARP] > 0.5f ? 0.7f * a[AUD_SHARP] : 0.0f});
    const float speech = a[AUD_SPEECH];
    const float person = std::max(v[VIS_PERSON], v[VIS_FACE]);
    const float nov = std::max({s[SNN_NOV_VIS], s[SNN_NOV_AUD], s[SNN_NOV_TOUCH]});
    // the spiking core's sleep readout idles at 0.01-0.04 in the day; undamped, that drained the slow needs (social need stuck
    // at 0.3, boredom low: per second it took 0.9 x sleep, more than the needs grow) - only real sleep counts
    const float sleep = dz(s[SNN_SLEEP]);
    const float stim = std::max({dz(vis_ev), 0.7f * dz(std::max(loud, speech)), dz(touch), 0.6f * dz(person, 0.3f), dz(nov),
                                 0.15f * a[AUD_SELF_VOICE]});
    stim_s_ += (stim - stim_s_) * std::min(1.0f, dt / 5.0f);
    const float quiet = std::max(0.0f, 1.0f - 2.5f * stim_s_);
    slow_boredom_ += dt * (0.003f * (1.0f - slow_boredom_) * quiet -
                           0.1f * slow_boredom_ * std::clamp((stim_s_ - 0.05f) / 0.3f, 0.0f, 1.0f));
    slow_boredom_ = std::clamp(slow_boredom_ * (1.0f - 0.5f * sleep * dt), 0.0f, 1.0f);
    // Attention satisfies the social need - a touch, being talked to; somebody who is merely in view barely does (2026-10-04:
    // with the owner sitting at the PC in view the need never grew, and the robot never asked them to play). This part
    // differs from teacher_drives.py, which only shapes the spiking core's own social readout.
    const float presence = std::max({0.02f * dz(s[SNN_TR_PERSON], 0.5f), dz(s[SNN_TR_TOUCH]), dz(s[SNN_TR_SOUND]) * 0.5f * (speech > 0.3f ? 1.0f : 0.0f)});
    slow_social_ += dt * (0.0035f * (1.0f - slow_social_) * (1.0f - presence) - 0.15f * slow_social_ * presence -
                          0.002f * slow_social_ * a[AUD_SELF_VOICE]);
    slow_social_ = std::clamp(slow_social_ + 0.3f * bus_.ctx[CTX_OWNER_ARRIVED], 0.0f, 1.0f) * (1.0f - 0.9f * dt * sleep);
    bus_.slow[SLOW_BOREDOM] = slow_boredom_;
    bus_.slow[SLOW_SOCIAL] = slow_social_;
}

void HybridBrain::tick(uint32_t)
{
    ++ticks_;
    updateNoise();

    // --- spiking core: internal state
    {
        const int64_t t0 = clock_us_ ? clock_us_() : 0;
        float in[kSnnInputs];
        bus_.snnInput(in);
        snn_.tick(in, bus_.snn);
        if (snn_.unstable()) {  // saturated or non-finite: restart from rest
            snn_.reset();
            ++snn_resets_;
            bus_.snn.fill(0.0f);
        }
        // Plausibility of anger: annoyance needs a reason - a run of touches, shaking, loud
        // sounds that keep coming. The evidence builds with each such event and
        // fades in ~30 s; the spiking core's annoyance can not exceed it by much.
        {
            const float touch = std::max(bus_.body[BODY_TOUCH_HEAD], bus_.body[BODY_TOUCH_SCREEN]);
            // Touches and loud sounds are short pulses (3 ticks); a shake is felt for 2 s and a tilt for 1.5 s (40 / 30 ticks),
            // so they add a sixth of the evidence per tick: one shake is a reason to be cross, not for a minute of rage. This is
            // only a safety net; that two or three strokes are petting and not a reason for anger is the spiking core's own
            // (trained since 2026-10-04 on a drive teacher where only a run of touches annoys).
            const float event = std::max(touch, bus_.audio[AUD_LOUD] > 0.5f ? bus_.audio[AUD_LOUD] : 0.0f);
            const float held = std::max(bus_.body[BODY_SHAKE], bus_.body[BODY_TILT] * 0.5f);
            annoy_evidence_ = std::min(1.0f, annoy_evidence_ * 0.9983f + 0.06f * event + 0.01f * held);
            bus_.snn[SNN_ANNOYANCE] = std::min(bus_.snn[SNN_ANNOYANCE], 0.12f + 1.6f * annoy_evidence_);
        }
        if (clock_us_) {
            const uint32_t us = static_cast<uint32_t>(clock_us_() - t0);
            snn_us_ = snn_us_ ? (snn_us_ * 7 + us) / 8 : us;
            snn_min_us_ = std::min(snn_min_us_, us);
        }
    }

    updateSlow();
    assoc_.tick(bus_);  // learns from what just happened (needs the spiking core's annoyance), before the policy reads it

    // --- policy: action selection
    {
        const int64_t t0 = clock_us_ ? clock_us_() : 0;
        PolicyGru::Input& in = policy_in_;
        bus_.policyInput(in);
        PolicyGru::Action a = PolicyGru::neutral();
        bool ok = true;
        if (external_) external_(in, a);
        else ok = policy_.step(in, a);
        if (!ok) ++policy_faults_;
        // Reject / clamp: no value leaves this class outside its range.
        bool bad = false;
        for (int i = 0; i < ACT_COUNT; ++i) {
            if (!std::isfinite(a[i])) {
                a[i] = PolicyGru::neutral()[i];
                bad = true;
            }
            const bool sig = i == ACT_YAW || i == ACT_PITCH || i == ACT_GAZE_X || i == ACT_GAZE_Y ||
                             i == ACT_SOUND_VALENCE || i == ACT_VOICE_SLOPE || i == ACT_FACE_VALENCE;
            a[i] = std::clamp(a[i], sig ? -1.0f : 0.0f, 1.0f);
        }
        if (bad) ++policy_faults_;
        action_ = a;
        bus_.prev[PREV_YAW] = a[ACT_YAW];
        bus_.prev[PREV_PITCH] = a[ACT_PITCH];
        bus_.prev[PREV_SPEED] = a[ACT_SPEED];
        bus_.prev[PREV_SOUND] = a[ACT_SOUND_ACT];
        bus_.prev[PREV_HOLD] = a[ACT_HOLD];
        bus_.prev[PREV_EXPLORE] = a[ACT_EXPLORE];
        bus_.prev[PREV_VALENCE] = a[ACT_FACE_VALENCE];
        bus_.prev[PREV_AROUSAL] = a[ACT_FACE_AROUSAL];
        if (clock_us_) {
            const uint32_t us = static_cast<uint32_t>(clock_us_() - t0);
            policy_us_ = policy_us_ ? (policy_us_ * 7 + us) / 8 : us;
        }
    }
    updatePhysiology();
}
