#include "body_executor.h"
#include "../core/debug_flags.h"
#include "../core/ms_clock.h"

#include <algorithm>
#include <cmath>
#include <esp_log.h>
#include <esp_timer.h>
#include <cstdio>

using namespace hybrid;

namespace {
constexpr const char* kTag = "executor";
float lerp(float a, float b, float t) { return a + (b - a) * t; }
}  // namespace

void BodyExecutor::apply(HybridBrain& brain, BodySense& sense, uint32_t now)
{
    const PolicyGru::Action& a = brain.action();
    if constexpr (!kParkHead) applyVoice(brain, sense, a, now);
    if constexpr (kVisionRec == VisionRec::Sweep) {
        if (now - last_command_ms_ >= 250) {  // a recording aid: the policy does not move the head
            const float t = static_cast<float>(now) * 0.001f;
            command(sense, 90.0f * std::sin(t * 6.2831853f / 70.0f), 8.0f + 14.0f * std::sin(t * 6.2831853f / 29.0f), 22.0f);
            last_command_ms_ = now;
        }
    } else {
        applyMotor(brain, sense, a, now);
    }
    applyFace(brain.bus(), a, now);
    applyLeds(brain);
}

void BodyExecutor::command(BodySense& sense, float yaw, float pitch, float speed)
{
    if constexpr (kParkHead) {  // one move to the front, then the servos are left alone
        static bool parked = false;
        if (parked) return;
        parked = true;
        yaw = 0.0f;
        pitch = 5.0f;
        speed = 40.0f;
    }
    if constexpr (kVisionRec == VisionRec::Still) {
        yaw = 0.0f;
        pitch = 5.0f;
        speed = 40.0f;
    }
    yaw = std::clamp(yaw, -kYawLimitDeg, kYawLimitDeg);
    pitch = std::clamp(pitch, kPitchMinDeg, kPitchMaxDeg);
    motion_.moveTo(yaw, pitch, std::clamp(speed, kMinSpeedDegS, kMaxSpeedDegS * 1.5f));
    sense.observeMotorCommand(yaw, pitch);
    base_yaw_ = yaw;
    base_pitch_ = pitch;
}

void BodyExecutor::applyMotor(HybridBrain& brain, BodySense& sense, const PolicyGru::Action& a, uint32_t now)
{
    float yaw = 0, pitch = 0;
    motion_.getCurrentDeg(yaw, pitch);
    if (!base_valid_) {
        base_yaw_ = yaw;
        base_pitch_ = pitch;
        base_valid_ = true;
    }
    const auto& bus = brain.bus();
    const float stamina = motion_.staminaLevel();

    // ---- motor primitives, triggered by the policy on a rising edge
    // (the invitation's 4-minute spacing is a behaviour rule: in mode life no invitation comes from the actor anyway)
    const bool invite = !free_ && a[ACT_INVITE] > 0.5f, startle = a[ACT_STARTLE] > 0.6f;
    // an invitation to play (a nod series and a call) at most every 4 minutes, however often the network wishes it
    if (invite && !prev_invite_ && !nodding_ && (last_invite_ms_ == 0 || now - last_invite_ms_ >= 240000)) {
        last_invite_ms_ = now;
        nodding_ = true;
        nod_start_ms_ = now;
        nod_base_pitch_ = pitch;
        ++nod_series_;
        ESP_LOGI(kTag, "invite: nodding (social=%.2f)", bus.snn[SNN_SOCIAL]);
    }
    if (startle && !prev_startle_ && !flinching_ && !nodding_) {
        flinching_ = true;
        flinch_start_ms_ = now;
        flinch_base_pitch_ = pitch;
        ++startles_;
        ESP_LOGI(kTag, "startle: flinch (loud=%.2f loom=%.2f)", bus.audio[AUD_LOUD], bus.vision[VIS_LOOM]);
    }
    prev_invite_ = invite;
    prev_startle_ = startle;
    if (nodding_) {
        // three nods, 0.45 s each: down, up
        const uint32_t t = now - nod_start_ms_;
        if (t >= 1350) {
            nodding_ = false;
            command(sense, base_yaw_, nod_base_pitch_, 60.0f);
        } else if (now - last_command_ms_ >= 90) {
            const bool down = (t / 225) % 2 == 0;
            command(sense, base_yaw_, nod_base_pitch_ + (down ? -14.0f : 8.0f), kNodDegS);
            last_command_ms_ = now;
        }
        return;
    }
    if (flinching_) {
        const uint32_t t = now - flinch_start_ms_;
        if (t >= 700) {
            flinching_ = false;
            command(sense, base_yaw_, flinch_base_pitch_, 90.0f);
        } else if (t < 120 && now - last_command_ms_ >= 60) {
            command(sense, base_yaw_, flinch_base_pitch_ + 14.0f, 220.0f);  // the head jerks back
            last_command_ms_ = now;
        }
        return;
    }

    // ---- the policy's head intent
    float want_yaw = std::clamp(a[ACT_YAW], -1.0f, 1.0f) * kYawLimitDeg;
    float want_pitch = lerp(kPitchMinDeg, kPitchMaxDeg, (std::clamp(a[ACT_PITCH], -1.0f, 1.0f) + 1.0f) * 0.5f);
    goal_hist_[goal_n_ % goal_hist_.size()] = {want_yaw, want_pitch};  // for the saccade generator below
    ++goal_n_;
    const bool hold = a[ACT_HOLD] > 0.6f;
    if (hold || now - last_command_ms_ < kMinCommandGapMs) return;
    float dy = want_yaw - base_yaw_, dp = want_pitch - base_pitch_;
    // A saccade generator, like the brainstem's: the head goes once the network's target has settled (it moved less than
    // 2 deg over the last 0.3 s), in one move, instead of creeping after a target that glides (a creeping head keeps the
    // servos loud and the robot deaf: 13000 deg of creep in 20 idle minutes). Big turns (40 deg or more) go at once; a
    // target that keeps moving (a person walking) is followed in steps at most every 0.6 s.
    // Where and when to look is the network's; this only shapes the movement. Plus a motor dead band of a few degrees.
    const auto& old_goal = goal_hist_[goal_n_ % goal_hist_.size()];  // the oldest of the last 6 ticks (0.3 s)
    const bool settled = goal_n_ >= goal_hist_.size() && std::hypot(want_yaw - old_goal[0], want_pitch - old_goal[1]) < 2.0f;
    if (std::fabs(dy) < kYawDeadDeg && std::fabs(dp) < kPitchDeadDeg) return;
    if (!settled && std::hypot(dy, dp) < 40.0f && now - last_command_ms_ < 600) return;
    // never a jump bigger than kMaxStepDeg: the head walks there in steps
    const float len = std::hypot(dy, dp);
    if (len > kMaxStepDeg) {
        dy *= kMaxStepDeg / len;
        dp *= kMaxStepDeg / len;
    }
    const float vigor = a[ACT_VIGOR];
    float speed = kMinSpeedDegS + (kMaxSpeedDegS - kMinSpeedDegS) * a[ACT_SPEED] * a[ACT_SPEED] * (0.75f + 0.25f * vigor);
    speed *= 0.7f + 0.3f * stamina;
    const float ty = std::clamp(base_yaw_ + dy, -kYawLimitDeg, kYawLimitDeg);
    const float tp = std::clamp(base_pitch_ + dp, kPitchMinDeg, kPitchMaxDeg);
    const float step = std::hypot(ty - base_yaw_, tp - base_pitch_);
    command(sense, ty, tp, speed);
    last_command_ms_ = now;
    if (step >= 12.0f) {
        ++moves_;
        ESP_LOGI(kTag, "move d=%+.0f,%+.0f to %.0f,%.0f speed=%.0f vigor=%.2f explore=%.2f att=%+.2f,%+.2f", ty - yaw,
                 tp - pitch, ty, tp, speed, vigor, a[ACT_EXPLORE], bus.snn[SNN_ATT_X], bus.snn[SNN_ATT_Y]);
    }
}

void BodyExecutor::applyVoice(HybridBrain& brain, BodySense& sense, const PolicyGru::Action& a, uint32_t now)
{
    if (last_token_ms_ == 0) last_token_ms_ = now;
    voice_tokens_ = std::min(kTokenCapacity, voice_tokens_ + static_cast<float>(now - last_token_ms_) / kTokenRefillMs);
    last_token_ms_ = now;
    // The call budget (~6 a minute): reactions (a touch, a shake, a startle, a loud sound) may spend the last tokens; idle
    // babble keeps two in reserve, so a chatty robot still answers the hand that touches it. (Teaching this to the network
    // through a teacher with memory made it go silent: the call signal is a sparse pulse, and labels that drop some of the
    // pulses in situations that look the same pushed it below its threshold - 2026-10-04.)
    const auto& bus = brain.bus();
    const bool event = bus.body[BODY_TOUCH_HEAD] > 0.5f || bus.body[BODY_TOUCH_SCREEN] > 0.5f || bus.body[BODY_SHAKE] > 0.5f ||
                       a[ACT_STARTLE] > 0.5f || bus.audio[AUD_LOUD] > 0.5f ||
                       std::max(bus.audio[AUD_CLAP], bus.audio[AUD_SHARP]) > 0.7f;
    // a reaction stays a reaction for 0.5 s after its event: the network's answer can come a tick or two after the touch
    // (the audit of 2026-10-04: 39 of 70 touches went unanswered while sound_act was 0.99 - it rose one tick too late)
    reaction_ticks_ = event ? 10 : std::max(0, reaction_ticks_ - 1);
    const bool reaction = reaction_ticks_ > 0;
    // Platform B (mode life): no behaviour budget - only the physical limit, the voice at most half of the last minute
    // (the speaker and the battery); how often to call is the actor's.
    talk_share_ += (static_cast<float>(vocal_.active() ? 1 : 0) - talk_share_) * (0.05f / 60.0f);
    const bool allowed = free_ ? talk_share_ < 0.5f : voice_tokens_ >= (reaction ? 1.0f : 2.5f);
    if (vocal_.step(a, allowed, static_cast<int>(shown_), reaction)) {
        voice_tokens_ -= 1.0f;
        ++calls_;
        ESP_LOGI(kTag, "call: start (sound_act=%.2f)", a[ACT_SOUND_ACT]);
    }
    if (vocal_.phraseEnded()) {
        const auto& s = brain.bus().snn;
        ESP_LOGI(kTag, "voice: %s (%.0f Hz) face=%s val=%+.2f aro=%.2f cur=%.2f style=%s", vocal_.phraseText(),
                 static_cast<double>(vocal_.baseHz()), emotionName(shown_), a[ACT_SOUND_VALENCE], a[ACT_SOUND_AROUSAL], s[SNN_CURIOSITY],
                 a[ACT_STYLE_ASK] > 0.6f ? "ask" : a[ACT_STYLE_PLAY] > 0.6f ? "play"
                 : a[ACT_STYLE_GRUMBLE] > 0.6f ? "grumble" : a[ACT_STYLE_CALM] > 0.6f ? "calm" : "mixed");
    }
    sense.observeSoundCommand(vocal_.active());
}

void BodyExecutor::applyFace(const Bus& bus, const PolicyGru::Action& a, uint32_t now)
{
    face_.setGaze(a[ACT_GAZE_X], a[ACT_GAZE_Y]);
    // person symbol on the screen: on above 0.5, off below 0.3 (the held sighting fades over 1.2 s)
    {
        const float p = bus.vision[VIS_PERSON];
        if (p > 0.5f) person_shown_ = true;
        else if (p < 0.3f) person_shown_ = false;
        face_.setPersonSeen(person_shown_);
    }
    const bool blink = a[ACT_BLINK] > 0.6f;
    if (blink && !prev_blink_) face_.blink();
    prev_blink_ = blink;

    // The face is chosen by the policy network: 21 outputs, one per Emotion (face_style.h order), the
    // strongest wins. Nothing here decides a face; the debounce below only keeps a face from flickering.
    static_assert(ACT_EMO_SICK - ACT_EMO_NORMAL + 1 == kEmotionCount, "one policy output per face (face_style.h order)");
    int best = 0;
    for (int i = 1; i < kEmotionCount; ++i)
        if (a[ACT_EMO_NORMAL + i] > a[ACT_EMO_NORMAL + best]) best = i;
    const Emotion want = static_cast<Emotion>(best);
    // The network holds its face itself (it is trained against flicker, tools/hybrid/train_policy.py); here only a
    // face that lasts 4 ticks is shown, and a shown face stays at least 1.5 s so its morph can be seen.
    if (want != candidate_) {
        candidate_ = want;
        candidate_ticks_ = 0;
    }
    ++candidate_ticks_;
    if (want == shown_ || candidate_ticks_ < 4 || now - face_since_ms_ < 1500) return;
    shown_ = want;
    face_since_ms_ = now;
    face_.setEmotion(want);
    ESP_LOGI(kTag, "emotion=%d %s val=%+.2f aro=%.2f lids=%.2f", static_cast<int>(want), emotionName(want),
             a[ACT_FACE_VALENCE], a[ACT_FACE_AROUSAL], a[ACT_LIDS]);
}

void BodyExecutor::applyLeds(HybridBrain& brain)
{
    const Physiology& p = brain.physiology();
    const auto& a = brain.action();
    // colour and liveliness are the policy's (led_r/g/b, led_activity); breathing and heartbeat come from the physiology
    auto level = [](float x) { return static_cast<uint8_t>(std::clamp(x, 0.0f, 1.0f) * 64.0f + 0.5f); };
    led_.setBreathing(level(a[ACT_LED_R]), level(a[ACT_LED_G]), level(a[ACT_LED_B]));
    led_.setActivity(a[ACT_LED_ACTIVITY]);
    led_.setRhythm(p.breathPerMin);
    if (p.heartBeat) led_.beat();
}
