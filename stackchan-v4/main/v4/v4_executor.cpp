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
    // the Muse's ideas as they come (the log line only: the network decides what to make of them)
    static constexpr const char* kIdeaNames[5] = {"play", "show", "still", "social", "surprise"};
    for (int k = 0; k < 5; ++k) {
        const float v = brain.ideas()[3 + k];
        const bool on = idea_on_[k] ? v > 0.3f : v > 0.5f;  // hysteresis: an idea near 0.5 is one idea, not ten
        if (on && !idea_on_[k] && !night && (idea_log_ms_[k] == 0 || now - idea_log_ms_[k] >= 10000)) {
            ESP_LOGI(kTag, "muse: idea %s (%.2f)", kIdeaNames[k], static_cast<double>(v));
            idea_log_ms_[k] = now;
        }
        idea_on_[k] = on;
    }
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

// The head's own chance (glances, how long a place is kept): a generator of its own, seeded by the moment it is
// first needed - the simulator's world keeps its random numbers to itself, and no two days on the robot are alike.
float V4Executor::rnd01(uint32_t now)
{
    if (rng_ == 0) rng_ = ((now * 2654435761u) ^ 0x9E3779B9u) | 1u;
    rng_ ^= rng_ << 13;
    rng_ ^= rng_ >> 17;
    rng_ ^= rng_ << 5;
    return static_cast<float>(rng_ & 0xFFFFFF) / 16777216.0f;
}

uint32_t V4Executor::rndMs(uint32_t lo, uint32_t hi, uint32_t now)
{
    return lo + static_cast<uint32_t>(rnd01(now) * static_cast<float>(hi - lo));
}

bool V4Executor::command(float yaw, float pitch, float speed, uint32_t now, bool force, uint32_t gap_ms,
                         float dead_yaw_deg)
{
    yaw = std::clamp(yaw, -kYawLimitDeg, kYawLimitDeg);
    pitch = std::clamp(pitch, kPitchMinDeg, kPitchMaxDeg);
    if (!force && now - last_command_ms_ < gap_ms) return false;
    if (!force && sent_valid_ && std::fabs(yaw - sent_yaw_) < dead_yaw_deg &&
        std::fabs(pitch - sent_pitch_) < kGazeDeadPitchDeg)
        return false;
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
    if (motion_.isMoving()) still_since_ms_ = now;  // the orienting reflex aims from a still head only
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
        else if (mode == kHeadFlinch && !night && (last_flinch_ms_ == 0 || now - last_flinch_ms_ >= kFlinchGapMs) &&
                 (last_gesture_ms_ == 0 || now - last_gesture_ms_ >= 1500))
            g = Gesture::Flinch;
        if (g != Gesture::None) {
            gesture_ = g;
            gesture_start_ms_ = now;
            gesture_step_ = -1;
            base_yaw_ = sent_yaw_;
            base_pitch_ = sent_pitch_;
            // nod 5..12 deg, shake 5..14 deg; a half swing 0.45 .. 0.3 s; 1..3 repeats
            gesture_amp_ = g == Gesture::Nod ? 5.0f + 7.0f * amp : g == Gesture::Shake ? 5.0f + 9.0f * amp : 6.0f;
            if (g == Gesture::Flinch) last_flinch_ms_ = now;
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
                // the head starts back (12 deg at 160 deg/s until 2026-10-25: a jerk the owner heard as a whizz; 8 at 90 until 10-26)
                command(base_yaw_, base_pitch_ + gesture_amp_, 60.0f, now, true);
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
    // the peak speed of a gaze turn (eased: MotionEngine::kEasedMoves; halved there: 7 .. 20 deg/s)
    const float move_speed = (night ? 8.0f : 14.0f) + 26.0f * speed;
    const float gy = o[NO_GAZE_YAW] * BrainV4::kGazeYawDeg;
    const float gp = o[NO_GAZE_PITCH] * BrainV4::kGazePitchDeg;
    if (brain.snn()[SO_ATT_STRENGTH] > kAttLook) attended_ms_ = now, attended_ = true;
    const bool wander = !attended_ || now - attended_ms_ > kWatchMs;
    uint32_t gap = kGazeGapMs;
    float dead = kGazeDeadYawDeg;
    float speed_k = 1.0f;
    if (wander) {
        gap = wander_hold_ms_;
        dead = kWanderDeadDeg;
    } else {
        speed_k = kOrientSpeedK;  // something caught its attention: it turns there quickly (still eased)
    }
    // At night the head only sinks into the doze: a clap at 03:51 (2026-10-25) lifted it, the servos hung, and the
    // power cycle that freed them clicked next to the sleeping owner.
    if (night && mode != kHeadDoze) return;

    // ---- the orienting reflex: something new in view (the analyzer's attention) - the head turns to it and keeps
    // watching that place kWatchMs (teacher_nn.py: ATT_LOOK, WATCH_S), whatever the network's gaze says meanwhile
    const bool free_head = mode == kHeadLook || mode == kHeadScan || mode == kHeadHold;
    if (!night && free_head && brain.snn()[SO_ATT_STRENGTH] > kAttLook) {
        const bool watching = watch_until_ms_ != 0 && !msReached(now, watch_until_ms_);
        if (!watching || now - still_since_ms_ >= kAimStillMs) {  // the picture is fresh: aim (again)
            float hy = 0, hp = 0;
            motion_.getCurrentDeg(hy, hp);
            watch_yaw_ = std::clamp(hy + brain.snn()[SO_ATT_X] * kCamHalfDeg, -kOrientLimitDeg, kOrientLimitDeg);
            watch_pitch_ = BrainV4::kGazePitchCenterDeg - 0.4f * brain.snn()[SO_ATT_Y] * BrainV4::kGazePitchDeg;
        }
        watch_until_ms_ = now + kWatchMs;
        if (glance_until_ms_ != 0) {
            glance_until_ms_ = 0;
            next_glance_ms_ = now + rndMs(kGlanceGapMinMs, kGlanceGapMaxMs, now);
        }
        command(watch_yaw_, watch_pitch_, move_speed * kOrientSpeedK, now, false, kOrientGapMs, kOrientDeadDeg);
        return;
    }
    if (!night && free_head && watch_until_ms_ != 0 && !msReached(now, watch_until_ms_)) return;  // still watching

    // ---- a curious glance (kGlanceGapMinMs): only while it wanders by day, looking or scanning
    const bool idle = wander && !night && (mode == kHeadLook || mode == kHeadScan);
    if (next_glance_ms_ == 0) next_glance_ms_ = now + rndMs(kGlanceGapMinMs / 2, kGlanceGapMaxMs / 2, now);
    if (glance_until_ms_ != 0) {
        if (idle && !msReached(now, glance_until_ms_)) return;  // looking there
        glance_until_ms_ = 0;  // over (or something to attend to, a mood with its own pose): the network's gaze again
        next_glance_ms_ = now + rndMs(kGlanceGapMinMs, kGlanceGapMaxMs, now);
        wander_hold_ms_ = 0;
    } else if (idle && msReached(now, next_glance_ms_) && now - last_command_ms_ >= kGazeGapMs) {
        const float side = std::fabs(sent_yaw_) > 30.0f ? (sent_yaw_ > 0 ? -1.0f : 1.0f) : (rnd01(now) < 0.5f ? -1.0f : 1.0f);
        const float yaw = side * (kGlanceYawMinDeg + (kGlanceYawMaxDeg - kGlanceYawMinDeg) * rnd01(now));
        const float pitch = -5.0f + 23.0f * rnd01(now);
        // the turn there takes T = pi/2 * distance / peak speed (MotionEngine: eased, at half the asked speed)
        const float travel_s = 1.5708f * std::hypot(yaw - sent_yaw_, pitch - sent_pitch_) / (0.5f * kGlanceSpeedDegS);
        if (command(yaw, pitch, kGlanceSpeedDegS, now, true)) {
            const uint32_t hold = rndMs(kGlanceHoldMinMs, kGlanceHoldMaxMs, now);
            glance_until_ms_ = now + static_cast<uint32_t>(travel_s * 1000.0f) + hold;
            if (glance_until_ms_ == 0) glance_until_ms_ = 1;
            ESP_LOGI(kTag, "glance: yaw %.0f pitch %.0f, looks %.1f s", static_cast<double>(yaw),
                     static_cast<double>(pitch), static_cast<double>(hold) * 0.001);
            return;
        }
    }
    bool moved = false;
    switch (mode) {
        case kHeadLook: moved = command(gy, BrainV4::kGazePitchCenterDeg - gp, move_speed * speed_k, now, false, gap, dead); break;
        case kHeadScan: moved = command(gy, BrainV4::kGazePitchCenterDeg - gp, move_speed * 0.6f * speed_k, now, false, gap, dead); break;
        case kHeadLookDown: command(sent_yaw_, -15.0f, 15.0f, now); break;
        case kHeadLookUp: command(sent_yaw_, 22.0f, 15.0f, now); break;
        case kHeadDoze: command(sent_yaw_ * 0.5f, -18.0f, 8.0f, now); break;
        default: break;  // hold
    }
    if (moved) {
        // how long the new place is kept: shorter in a lively mood, and by chance
        const float k = std::clamp(0.6f * speed + 0.4f * rnd01(now), 0.0f, 1.0f);
        wander_hold_ms_ = kWanderHoldMaxMs - static_cast<uint32_t>((kWanderHoldMaxMs - kWanderHoldMinMs) * k);
    }
}
