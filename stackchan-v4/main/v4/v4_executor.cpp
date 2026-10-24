#include "v4_executor.h"

#include <algorithm>
#include <cmath>
#include <esp_log.h>

#include "../core/ms_clock.h"

using namespace v4;

namespace {
constexpr const char* kTag = "v4";

// the strongest of a choice group (softmax outputs), as an index inside the group
int argmaxRange(const NnV4::Out& o, int at, int count)
{
    int best = at;
    for (int i = at + 1; i < at + count; ++i)
        if (o[i] > o[best]) best = i;
    return best - at;
}
}  // namespace

void V4Executor::apply(const BrainV4& brain, bool decided, bool night, uint32_t now, BrainV4::Executed& done)
{
    if (!brain.nnValid()) return;
    applyFace(brain, now);
    applyVoice(brain, night, now, done);
    applyLeds(brain, night);
    applyHead(brain, night, now, done);
    if (decided) {
        done.emotion = static_cast<int>(shown_);
        float yaw = 0, pitch = 0;
        motion_.getCurrentDeg(yaw, pitch);
        done.headYawDeg = yaw;
        done.headPitchDeg = pitch;
    }
}

void V4Executor::applyFace(const BrainV4& brain, uint32_t now)
{
    const auto& o = brain.nn();
    const auto& s = brain.snn();
    face_.setGaze(std::clamp(s[SO_ATT_X] * 1.2f, -1.0f, 1.0f), std::clamp(s[SO_ATT_Y] * 1.2f, -1.0f, 1.0f));
    const Emotion want = static_cast<Emotion>(argmaxRange(o, kNoEmoAt, kNoEmoCount));
    // Nothing here picks a face: a choice that lasts 2 ticks is shown, and a shown face stays 1.5 s.
    if (want == candidate_) ++candidate_ticks_;
    else {
        candidate_ = want;
        candidate_ticks_ = 1;
    }
    if (want != shown_ && candidate_ticks_ >= 2 && now - face_since_ms_ >= 1500) {
        shown_ = want;
        face_since_ms_ = now;
        face_.setEmotion(shown_);
        ESP_LOGI(kTag, "face: %s (valence %.2f arousal %.2f over %.2f threat %.2f under %.2f)", emotionName(shown_),
                 s[SO_VALENCE], s[SO_AROUSAL], s[SO_OVER], s[SO_THREAT], s[SO_UNDER]);
    }
}

void V4Executor::applyVoice(const BrainV4& brain, bool night, uint32_t now, BrainV4::Executed& done)
{
    const auto& o = brain.nn();
    const auto& s = brain.snn();
    if (last_token_ms_ == 0) last_token_ms_ = now;
    tokens_ = std::min(5.0f, tokens_ + static_cast<float>(now - last_token_ms_) / 10000.0f);
    last_token_ms_ = now;
    // a reaction (a touch, a fright) may spend the last tokens and cut idle babble short; it stays a reaction 0.5 s
    const float touch = std::max(s[SO_SAL_TOUCH_HEAD], s[SO_SAL_TOUCH_SCREEN]);
    const bool event = (touch > 0.12f && touch - prev_touch_ > 0.02f) || s[SO_SURPRISE] > 0.35f || s[SO_THREAT] > 0.45f;
    prev_touch_ = touch;
    reaction_ticks_ = event ? 10 : std::max(0, reaction_ticks_ - 1);
    const bool reaction = reaction_ticks_ > 0;
    // the voice's intent, from the main network's voice head
    BabbleVoice::Intent a;
    a.say = o[NO_SAY];
    a.valence = o[NO_SND_VALENCE];
    a.arousal = o[NO_SND_AROUSAL];
    a.reg = o[NO_SND_REGISTER];
    a.slope = o[NO_SND_VALENCE] * 0.5f;
    a.expr = o[NO_SND_AROUSAL];
    a.length = o[NO_SND_LENGTH];
    a.rate = o[NO_SND_RATE];
    a.calm = o[NO_STYLE_CALM];
    a.ask = o[NO_STYLE_ASK];
    a.play = o[NO_STYLE_PLAY];
    a.grumble = o[NO_STYLE_GRUMBLE];
    // Night: silent, whatever the network wants (the owner sleeps next to the robot) - a safety rule, not a choice.
    const bool allowed = !night && tokens_ >= (reaction ? 1.0f : 2.5f);
    if (vocal_.step(a, allowed, static_cast<int>(shown_), reaction)) {
        tokens_ -= 1.0f;
        ++calls_;
        done.said = true;
        ESP_LOGI(kTag, "call (face=%s%s)", emotionName(shown_), reaction ? ", reaction" : "");
    }
    if (vocal_.phraseEnded())
        ESP_LOGI(kTag, "voice: %s (%.0f Hz) face=%s", vocal_.phraseText(), static_cast<double>(vocal_.baseHz()),
                 emotionName(shown_));
}

void V4Executor::applyLeds(const BrainV4& brain, bool night)
{
    const auto& o = brain.nn();
    const int c = argmaxRange(o, kNoColorAt, kNoColorCount);
    const int p = argmaxRange(o, kNoPatternAt, kNoPatternCount);
    const float bright = night ? std::min(o[NO_LED_BRIGHT], 0.06f) : o[NO_LED_BRIGHT];
    led_.setPattern(kLedRgb[c][0], kLedRgb[c][1], kLedRgb[c][2], p, bright, o[NO_LED_RATE]);
}

bool V4Executor::command(float yaw, float pitch, float speed, uint32_t now, bool force)
{
    yaw = std::clamp(yaw, -kYawLimitDeg, kYawLimitDeg);
    pitch = std::clamp(pitch, kPitchMinDeg, kPitchMaxDeg);
    if (!force && now - last_command_ms_ < kMinCommandGapMs) return false;
    if (!force && sent_valid_ && std::fabs(yaw - sent_yaw_) < 3.0f && std::fabs(pitch - sent_pitch_) < 2.5f) return false;
    motion_.moveTo(yaw, pitch, std::clamp(speed, 8.0f, kMaxSpeedDegS));
    sent_yaw_ = yaw;
    sent_pitch_ = pitch;
    sent_valid_ = true;
    last_command_ms_ = now;
    return true;
}

void V4Executor::applyHead(const BrainV4& brain, bool night, uint32_t now, BrainV4::Executed& done)
{
    const auto& o = brain.nn();
    // A head mode counts once the network has kept it for two decisions (3 ticks): at a touch's very start the analyzer's
    // surprise peaks and the network wished a flinch for one decision before the nod (2026-10-24: 9 flinches on 11
    // strokes); the face has the same rule.
    const int wish = argmaxRange(o, kNoHeadAt, kNoHeadCount);
    if (wish == head_cand_) ++head_cand_ticks_;
    else head_cand_ = wish, head_cand_ticks_ = 1;
    if (head_cand_ticks_ >= 3) head_mode_ = head_cand_;
    const int mode = head_mode_;
    const float amp = o[NO_GEST_AMP], speed = o[NO_GEST_SPEED], reps = o[NO_GEST_REPS];
    float yaw = 0, pitch = 0;
    motion_.getCurrentDeg(yaw, pitch);
    if (!sent_valid_) {
        sent_yaw_ = base_yaw_ = yaw;
        sent_pitch_ = base_pitch_ = pitch;
        sent_valid_ = true;
    }

    // ---- start a gesture on the network's wish (a rising edge of the mode)
    const bool edge = mode != prev_mode_;
    prev_mode_ = mode;
    if (gesture_ == Gesture::None && edge) {
        const bool gap_ok = last_gesture_ms_ == 0 || now - last_gesture_ms_ >= kGestureGapMs;
        Gesture g = Gesture::None;
        if (mode == kHeadNod && gap_ok && !night) g = Gesture::Nod;
        else if (mode == kHeadShake && gap_ok && !night) g = Gesture::Shake;
        // not at night either: a jerk of the head is servo noise next to the sleeping owner
        else if (mode == kHeadFlinch && !night && (last_gesture_ms_ == 0 || now - last_gesture_ms_ >= 1500)) g = Gesture::Flinch;
        if (g != Gesture::None) {
            gesture_ = g;
            gesture_start_ms_ = now;
            gesture_step_ = -1;
            base_yaw_ = sent_yaw_;
            base_pitch_ = sent_pitch_;
            // nod 5..12 deg, shake 5..14 deg; a half swing 0.45 .. 0.3 s; 1..3 repeats
            gesture_amp_ = g == Gesture::Nod ? 5.0f + 7.0f * amp : g == Gesture::Shake ? 5.0f + 9.0f * amp : 12.0f;
            gesture_half_s_ = 0.45f - 0.15f * speed;
            gesture_reps_ = 1 + static_cast<int>(std::lround(std::clamp(reps, 0.0f, 1.0f) * 2.0f));
            last_gesture_ms_ = now;
            ++gestures_;
            if (g == Gesture::Nod) ++nods_;
            if (g == Gesture::Shake) ++shakes_;
            done.gesture = true;
            ESP_LOGI(kTag, "gesture: %s amp %.0f deg x%d", g == Gesture::Nod ? "nod (yes)" : g == Gesture::Shake ? "shake (no)" : "flinch",
                     static_cast<double>(gesture_amp_), gesture_reps_);
        }
    }

    // ---- a running gesture: keyframes, each reached at a speed that fits its half swing (soft: the servo eases)
    if (gesture_ != Gesture::None) {
        const float t = static_cast<float>(now - gesture_start_ms_) * 0.001f;
        if (gesture_ == Gesture::Flinch) {
            if (gesture_step_ < 0) {
                command(base_yaw_, base_pitch_ + gesture_amp_, 160.0f, now, true);  // the head jerks back
                gesture_step_ = 0;
            } else if (gesture_step_ == 0 && t > 0.6f) {
                command(base_yaw_, base_pitch_, 50.0f, now, true);
                gesture_step_ = 1;
            } else if (t > 1.2f) {
                gesture_ = Gesture::None;
            }
            return;
        }
        const int frames = 2 * gesture_reps_;  // nod: down / up per repeat; shake: one side / the other per repeat
        const int step = static_cast<int>(t / gesture_half_s_);
        if (step != gesture_step_) {
            gesture_step_ = step;
            const float speed_deg_s = 2.0f * gesture_amp_ / gesture_half_s_;
            if (step >= frames) {
                command(base_yaw_, base_pitch_, speed_deg_s * 0.6f, now, true);
                gesture_ = Gesture::None;
            } else if (gesture_ == Gesture::Nod) {
                const float target = step % 2 == 0 ? base_pitch_ - gesture_amp_ : base_pitch_ + gesture_amp_ * 0.25f;
                command(base_yaw_, target, speed_deg_s, now, true);
            } else {
                const float target = step % 2 == 0 ? base_yaw_ + gesture_amp_ : base_yaw_ - gesture_amp_;
                command(target, base_pitch_ - (shown_ == Emotion::Sad ? 5.0f : 0.0f), speed_deg_s, now, true);
            }
        }
        return;
    }

    // ---- where to look
    const float move_speed = (night ? 8.0f : 18.0f) + 50.0f * speed;
    const float gy = o[NO_GAZE_YAW] * BrainV4::kGazeYawDeg, gp = o[NO_GAZE_PITCH] * BrainV4::kGazePitchDeg;
    switch (mode) {
        case kHeadLook: command(gy, 5.0f - gp, move_speed, now); break;
        case kHeadScan: command(gy, 5.0f - gp, move_speed * 0.6f, now); break;
        case kHeadLookDown: command(sent_yaw_, -15.0f, 15.0f, now); break;
        case kHeadLookUp: command(sent_yaw_, 22.0f, 15.0f, now); break;
        case kHeadDoze: command(sent_yaw_ * 0.5f, -18.0f, 8.0f, now); break;
        default: break;  // hold
    }
}
