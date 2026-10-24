#include "collector.h"

#include <algorithm>
#include <cmath>

using namespace hybrid;

namespace {
constexpr float kYawLimitDeg = 100.0f;    // BodyExecutor::kYawLimitDeg: ACT_YAW = target / this
constexpr float kHalfViewYaw = 30.0f;     // the camera's half field of view (BrainLoop)
constexpr float kHalfViewPitch = 22.0f;
// the executor's pitch window: ACT_PITCH -1..1 = kPitchMinDeg..kPitchMaxDeg (-20..35)
float pitchAction(float deg) { return std::clamp((deg + 20.0f) / 55.0f * 2.0f - 1.0f, -1.0f, 1.0f); }
float pitchDeg(float action) { return -20.0f + (action + 1.0f) * 0.5f * 55.0f; }
// faces to wander through by day (Sleepy is the night's); the emo_* outputs are one-hot
constexpr int kDayFaces[] = {ACT_EMO_NORMAL, ACT_EMO_HAPPY, ACT_EMO_CURIOUS, ACT_EMO_THINKING, ACT_EMO_COOL,
                             ACT_EMO_WINK, ACT_EMO_CONFUSED, ACT_EMO_SURPRISED, ACT_EMO_BORED, ACT_EMO_VERY_HAPPY,
                             ACT_EMO_SHY, ACT_EMO_LOVE};
}  // namespace

uint32_t Collector::rnd()
{
    rng_ ^= rng_ << 13;
    rng_ ^= rng_ >> 17;
    rng_ ^= rng_ << 5;
    return rng_;
}

float Collector::uni(float lo, float hi) { return lo + (hi - lo) * static_cast<float>(rnd() >> 8) / 16777216.0f; }

int Collector::sampleLogits(const float* logits, int n)
{
    float mx = logits[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, logits[i]);
    float p[16], sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += p[i] = std::exp(logits[i] - mx);
    float u = uni(0.0f, sum);
    for (int i = 0; i < n; ++i)
        if ((u -= p[i]) <= 0.0f) return i;
    return n - 1;
}

float Collector::gauss()
{
    float s = 0.0f;
    for (int i = 0; i < 4; ++i) s += uni(0.0f, 1.0f);
    return (s - 2.0f) * 1.7320508f;
}

float Collector::expo(float meanS) { return -meanS * std::log(std::max(1e-6f, uni(0.0f, 1.0f))); }

bool Collector::night(const Bus& bus) const
{
    const float s = bus.ctx[CTX_HOUR_SIN], c = bus.ctx[CTX_HOUR_COS];
    if (s == 0.0f && c == 0.0f) return false;  // clock not synced yet
    float h = std::atan2(s, c) / 6.2831853f * 24.0f;
    if (h < 0.0f) h += 24.0f;
    return h >= 21.5f || h < 6.5f;
}

void Collector::newMove(const Bus& bus, uint32_t now, bool small)
{
    const float cur_yaw = bus.body[BODY_YAW] * 100.0f, cur_pitch = bus.body[BODY_PITCH] * 35.0f;
    float y, p;
    if (small) {
        y = cur_yaw + uni(-20.0f, 20.0f);
        p = cur_pitch + uni(-10.0f, 10.0f);
    } else {
        y = uni(-60.0f, 60.0f);
        p = uni(-15.0f, 30.0f);
    }
    yaw_ = std::clamp(y / kYawLimitDeg, -0.9f, 0.9f);
    pitch_ = pitchAction(p);
    speed_ = uni(0.25f, 1.0f);
    vigor_ = uni(0.2f, 1.0f);
    move_until_ms_ = now + 3000;  // the executor walks big turns in steps of up to 70 deg
}

void Collector::newVoice(uint32_t now, float lengthS, bool happy)
{
    call_until_ms_ = now + static_cast<uint32_t>(lengthS * 1000.0f);
    v_arousal_ = happy ? uni(0.5f, 0.9f) : uni(0.1f, 0.9f);
    v_valence_ = happy ? uni(0.4f, 1.0f) : uni(-0.6f, 0.8f);
    v_register_ = uni(0.0f, 1.0f);
    v_slope_ = uni(-0.8f, 0.8f);
    v_expr_ = uni(0.1f, 0.9f);
    v_length_ = uni(0.1f, 0.9f);
    v_rate_ = uni(0.2f, 0.9f);
    v_style_ = happy ? 2 : actor_ ? sampleLogits(actor_out_ + 20, 4) : static_cast<int>(rnd() % 4);
    // the voice's character: the face's (a touch answer always), or half the time another mood's - the droid sounds of
    // the 21 faces differ (giggles, squeals, growls, warbles, coos), and the day's 12 faces leave many out
    static constexpr int kVoiceMoods[] = {ACT_EMO_LAUGHING, ACT_EMO_SURPRISED, ACT_EMO_SAD, ACT_EMO_DIZZY, ACT_EMO_THINKING,
                                          ACT_EMO_CURIOUS, ACT_EMO_VERY_HAPPY, ACT_EMO_LOVE, ACT_EMO_ANNOYED, ACT_EMO_CONFUSED,
                                          ACT_EMO_SCARED, ACT_EMO_SHY, ACT_EMO_ANGRY, ACT_EMO_WINK};
    constexpr int kMoods = static_cast<int>(sizeof(kVoiceMoods) / sizeof(kVoiceMoods[0]));
    voice_emo_ = !happy && uni(0.0f, 1.0f) < 0.5f ? kVoiceMoods[rnd() % kMoods] - ACT_EMO_NORMAL : -1;
}

void Collector::step(const Bus& bus, uint32_t now, PolicyGru::Action& a)
{
    a = PolicyGru::neutral();
    const bool sleep = night(bus);
    if (!started_) {
        started_ = true;
        next_move_ms_ = now + 2000;
        next_call_ms_ = now + 10000;
        next_face_ms_ = now;
        next_blink_ms_ = now + 3000;
        yaw_ = bus.body[BODY_YAW] * 100.0f / kYawLimitDeg;
        pitch_ = pitchAction(bus.body[BODY_PITCH] * 35.0f);
    }

    // ---- people: a touch is answered at once; a face or a person in view is often looked at
    const bool touch = std::max(bus.body[BODY_TOUCH_HEAD], bus.body[BODY_TOUCH_SCREEN]) > 0.5f;
    if (touch && !touched_) {
        // Every touch is answered (the owner's rule), the way chosen at random so that the diary shows which answer
        // keeps people coming: a call with a happy face, only a call, only a happy face, a loving face. No nod: the
        // touch pad sits next to the pitch servo, a slow nod made a phantom touch, which made a nod... (2026-10-13,
        // the head nodded all night).
        const int how = static_cast<int>(rnd() % 4);
        if (!sleep && how <= 1) newVoice(now, uni(0.4f, 1.0f), true);
        happy_face_ = how == 3 ? ACT_EMO_LOVE : ACT_EMO_HAPPY;
        if (how != 1) happy_until_ms_ = now + 4000;
        answer_ = how;
    }
    touched_ = touch;
    // ---- a bang (clap, knock) is answered at once, like a touch: the collector's rule, masked in the actor's training.
    // Not the robot's own noise: its servos (own motion) and its voice were heard as claps before (2026-10-03).
    const bool bang = std::max(bus.audio[AUD_CLAP], bus.audio[AUD_SHARP]) > 0.7f && bus.audio[AUD_OWN_MOTION] < 0.5f &&
                      bus.audio[AUD_SELF_VOICE] < 0.3f;
    if (bang && !banged_ && static_cast<int32_t>(now - bang_ms_) >= 3000) {
        bang_ms_ = now;
        happy_face_ = ACT_EMO_SURPRISED;
        happy_until_ms_ = now + 2500;
        if (!sleep) {
            const float side = uni(0.0f, 1.0f) < 0.5f ? -1.0f : 1.0f;
            yaw_ = std::clamp((bus.body[BODY_YAW] * 100.0f + side * uni(15.0f, 30.0f)) / kYawLimitDeg, -0.9f, 0.9f);
            pitch_ = pitchAction(bus.body[BODY_PITCH] * 35.0f + uni(5.0f, 12.0f));
            speed_ = 0.9f;
            vigor_ = 0.9f;
            move_until_ms_ = now + 1500;
            last_actor_move_ms_ = now;  // the actor's next move waits
            newVoice(now, uni(0.3f, 0.6f), false);
            v_arousal_ = uni(0.7f, 1.0f);
            voice_emo_ = ACT_EMO_SURPRISED - ACT_EMO_NORMAL;
        }
    }
    banged_ = bang;
    // ---- handled roughly: shaken, tilted, lifted (BodySense holds a shake 2 s, a tilt while it lasts). The collector's
    // rule like a touch: the face at once, a sound by day; masked in the actor's training. The more it is shaken within
    // a minute, the crosser: dizzy, then annoyed, then angry with a growl.
    const bool shake = bus.body[BODY_SHAKE] > 0.5f, tilt = bus.body[BODY_TILT] > 0.5f;
    if (shake && !shaken_) {
        rough_ms_[0] = rough_ms_[1];
        rough_ms_[1] = rough_ms_[2];
        rough_ms_[2] = now | 1u;
        int recent = 0;
        for (uint32_t t : rough_ms_) recent += t && now - t < 60000 ? 1 : 0;
        const int face = recent >= 3 ? ACT_EMO_ANGRY : recent == 2 ? ACT_EMO_ANNOYED : ACT_EMO_DIZZY;
        happy_face_ = face;
        happy_until_ms_ = now + 4000;
        if (!sleep) {
            newVoice(now, uni(0.5f, 1.0f), false);
            v_arousal_ = uni(0.6f, 1.0f);
            v_valence_ = face == ACT_EMO_DIZZY ? uni(-0.3f, 0.3f) : uni(-1.0f, -0.5f);
            v_style_ = face == ACT_EMO_DIZZY ? 0 : 3;  // grumble when cross
            voice_emo_ = face - ACT_EMO_NORMAL;
        }
    } else if (tilt && !tilted_) {
        happy_face_ = ACT_EMO_SCARED;
        happy_until_ms_ = now + 3000;
        if (!sleep) {
            newVoice(now, uni(0.3f, 0.6f), false);
            v_arousal_ = uni(0.7f, 1.0f);
            voice_emo_ = ACT_EMO_SCARED - ACT_EMO_NORMAL;
        }
    }
    if (tilt && static_cast<int32_t>(happy_until_ms_ - now) < 500 && happy_face_ == ACT_EMO_SCARED)
        happy_until_ms_ = now + 500;  // scared while it lasts
    shaken_ = shake;
    tilted_ = tilt;
    const bool face = bus.vision[VIS_FACE] > 0.5f ||
                      (bus.vision[VIS_PERSON] > 0.6f && bus.vision[VIS_TARGET_CONF] > 0.5f);
    // (with an actor, looking at people is the actor's: it learned that from the reward)
    if (!actor_ && face && !saw_face_ && !sleep && uni(0.0f, 1.0f) < 0.5f) {  // half the time: does looking pay off?
        const float y = bus.body[BODY_YAW] * 100.0f + bus.vision[VIS_TARGET_X] * kHalfViewYaw;
        const float p = bus.body[BODY_PITCH] * 35.0f - bus.vision[VIS_TARGET_Y] * kHalfViewPitch;
        yaw_ = std::clamp(y / kYawLimitDeg, -0.9f, 0.9f);
        pitch_ = pitchAction(p);
        speed_ = uni(0.4f, 0.8f);
        move_until_ms_ = now + 2000;
        next_move_ms_ = now + static_cast<uint32_t>(1000.0f * (2.0f + expo(6.0f)));
    }
    saw_face_ = face;
    // ---- the actor's decisions (platform B): sampled from what it learned; the night stays quiet (the owner sleeps)
    bool actor_call = false;
    decision_ = Decision{};
    if (actor_) {
        const float* o = actor_out_;
        const auto p = [](float logit) { return 1.0f / (1.0f + std::exp(-logit)); };
        const bool may_move = static_cast<int32_t>(now - last_actor_move_ms_) >= (sleep ? 60000 : 1500);
        decision_.move_possible = may_move;
        if (may_move && uni(0.0f, 1.0f) < p(o[0])) {
            last_actor_move_ms_ = now;
            // the floor of exploration (the plan's "entropy floor"): the actor narrowed its targets to +-20 deg around
            // where it likes to look, and the owner saw a head that hardly moved (2026-10-15): at least +-30 deg
            // (yaw, 0.3 of the window) and +-25 deg (pitch, ~0.9 of the window's half)
            const float ys = std::max(0.3f, std::exp(std::clamp(o[2], -3.0f, 1.0f)));
            const float ps = std::max(0.45f, std::exp(std::clamp(o[4], -3.0f, 1.0f)));
            yaw_ = std::clamp(std::tanh(o[1]) + ys * gauss(), -0.9f, 0.9f);
            pitch_ = std::clamp(std::tanh(o[3]) + ps * gauss(), -1.0f, 1.0f);
            speed_ = sleep ? 0.25f : std::clamp(p(o[5]), 0.15f, 1.0f);
            vigor_ = uni(0.2f, 1.0f);
            move_until_ms_ = now + 3000;
            decision_.moved = true;
            decision_.yaw = yaw_;
            decision_.pitch = pitch_;
        }
        decision_.call_possible = !sleep && static_cast<int32_t>(call_until_ms_ - now) <= 0;
        actor_call = decision_.call_possible && uni(0.0f, 1.0f) < p(o[6]);
        decision_.called = actor_call;
    }
    // ---- head babbling (the collector's own timer, when there is no actor)
    if (!actor_ && static_cast<int32_t>(now - next_move_ms_) >= 0) {
        if (sleep) {
            newMove(bus, now, true);
            speed_ = 0.25f;
            next_move_ms_ = now + static_cast<uint32_t>(1000.0f * (60.0f + expo(180.0f)));
        } else {
            // moderate: ~130 deg a minute (the servos hang more the more they move, 2026-10-11)
            newMove(bus, now, uni(0.0f, 1.0f) < 0.5f);
            next_move_ms_ = now + static_cast<uint32_t>(1000.0f * (3.0f + expo(12.0f)));
        }
    }
    const bool moving = static_cast<int32_t>(move_until_ms_ - now) > 0;
    a[ACT_YAW] = yaw_;
    a[ACT_PITCH] = pitch_;
    a[ACT_SPEED] = speed_;
    a[ACT_VIGOR] = vigor_;
    a[ACT_HOLD] = moving ? 0.0f : 1.0f;
    a[ACT_EXPLORE] = 0.5f;
    a[ACT_GAZE_X] = std::clamp((yaw_ * kYawLimitDeg - bus.body[BODY_YAW] * 100.0f) / 60.0f, -1.0f, 1.0f);
    a[ACT_GAZE_Y] = std::clamp((pitchDeg(pitch_) - bus.body[BODY_PITCH] * 35.0f) / 40.0f, -1.0f, 1.0f);

    // ---- voice: now and then by day, never at night (the owner sleeps next to it)
    if (actor_call) {
        newVoice(now, uni(0.3f, 1.2f), false);
    } else if (!actor_ && !sleep && static_cast<int32_t>(now - next_call_ms_) >= 0) {
        newVoice(now, uni(0.3f, 1.2f), false);
        next_call_ms_ = now + static_cast<uint32_t>(1000.0f * (10.0f + expo(50.0f)));
    }
    if (sleep && static_cast<int32_t>(call_until_ms_ - now) > 0) call_until_ms_ = now;
    const bool calling = static_cast<int32_t>(call_until_ms_ - now) > 0;
    a[ACT_SOUND_ACT] = calling ? 1.0f : 0.0f;
    a[ACT_SOUND_AROUSAL] = v_arousal_;
    a[ACT_SOUND_VALENCE] = v_valence_;
    a[ACT_VOICE_REGISTER] = v_register_;
    a[ACT_VOICE_SLOPE] = v_slope_;
    a[ACT_VOICE_EXPR] = v_expr_;
    a[ACT_VOICE_LENGTH] = v_length_;
    a[ACT_VOICE_RATE] = v_rate_;
    a[ACT_STYLE_CALM] = v_style_ == 0 ? 1.0f : 0.0f;
    a[ACT_STYLE_ASK] = v_style_ == 1 ? 1.0f : 0.0f;
    a[ACT_STYLE_PLAY] = v_style_ == 2 ? 1.0f : 0.0f;
    a[ACT_STYLE_GRUMBLE] = v_style_ == 3 ? 1.0f : 0.0f;

    // ---- face, eyelids, blinks, LEDs
    if (static_cast<int32_t>(now - next_face_ms_) >= 0) {
        // the actor chooses the face (12 logits after its 8 decision outputs); without it, at random
        emo_ = kDayFaces[actor_ ? sampleLogits(actor_out_ + 8, 12) : static_cast<int>(rnd() % 12)];
        next_face_ms_ = now + static_cast<uint32_t>(1000.0f * uni(20.0f, 90.0f));
    }
    int shown = emo_;
    if (static_cast<int32_t>(happy_until_ms_ - now) > 0) shown = happy_face_;
    else if (sleep) shown = ACT_EMO_SLEEPY;
    for (int i = ACT_EMO_NORMAL; i <= ACT_EMO_SICK; ++i) a[i] = i == shown ? 1.0f : 0.0f;
    const bool cross = shown == ACT_EMO_ANGRY || shown == ACT_EMO_ANNOYED || shown == ACT_EMO_SCARED;
    a[ACT_FACE_VALENCE] = shown == ACT_EMO_HAPPY ? 0.6f : cross ? -0.6f : 0.0f;
    a[ACT_FACE_AROUSAL] = cross || shown == ACT_EMO_DIZZY ? 0.8f : sleep ? 0.1f : 0.4f;
    a[ACT_LIDS] = sleep ? 0.7f : 0.0f;
    a[ACT_BLINK] = 0.0f;
    if (static_cast<int32_t>(now - next_blink_ms_) >= 0) {
        a[ACT_BLINK] = 1.0f;
        next_blink_ms_ = now + static_cast<uint32_t>(1000.0f * (1.5f + expo(3.0f)));
    }
    for (int k = 0; k < 3; ++k) {
        if (std::fabs(led_[k] - led_to_[k]) < 0.01f) led_to_[k] = uni(0.0f, 1.0f);
        led_[k] += (led_to_[k] - led_[k]) * 0.005f;
    }
    a[ACT_LED_R] = led_[0];
    a[ACT_LED_G] = led_[1];
    a[ACT_LED_B] = led_[2];
    a[ACT_LED_ACTIVITY] = sleep ? 0.05f : 0.3f;
    a[ACT_INVITE] = 0.0f;
    a[ACT_STARTLE] = 0.0f;
}
