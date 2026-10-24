#include "brain_v4.h"

#include <algorithm>
#include <cmath>

BrainV4::BrainV4()
{
    for (auto& v : pace_v_) v = 0.0f;
}

float BrainV4::gauss()
{
    // xorshift32 + Box-Muller
    auto u = [this]() {
        rng_ ^= rng_ << 13;
        rng_ ^= rng_ >> 17;
        rng_ ^= rng_ << 5;
        return (static_cast<float>(rng_ >> 8) + 0.5f) / 16777216.0f;
    };
    const float a = u(), b = u();
    return std::sqrt(-2.0f * std::log(a)) * std::cos(6.2831853f * b);
}

void BrainV4::prevInputs(float* x) const
{
    using namespace v4;
    static constexpr float kEmoValence[21] = {0.55f, 0.80f, 0.90f, 0.90f, 0.20f, 0.10f, 0.45f, 0.55f, 0.10f, 0.25f, 0.95f,
                                              0.70f, 0.80f, 0.65f, 0.45f, 0.55f, 0.35f, 0.10f, 0.35f, 0.70f, 0.20f};
    static constexpr float kEmoArousal[21] = {0.30f, 0.55f, 0.80f, 0.85f, 0.25f, 0.50f, 0.05f, 0.85f, 0.90f, 0.65f, 0.45f,
                                              0.35f, 0.55f, 0.55f, 0.50f, 0.35f, 0.15f, 0.90f, 0.70f, 0.35f, 0.15f};
    x[PV_PREV_VALENCE] = kEmoValence[emo_];
    x[PV_PREV_AROUSAL] = kEmoArousal[emo_];
    x[PV_EMO_AGE] = std::min(emo_age_ / 20.0f, 1.0f);
    x[PV_SINCE_SOUND] = std::min(since_sound_ / 30.0f, 1.0f);
    x[PV_SINCE_GESTURE] = std::min(since_gesture_ / 30.0f, 1.0f);
    x[PV_HEAD_YAW] = head_yaw_;
    x[PV_HEAD_PITCH] = head_pitch_;
    x[PV_LED_LEVEL] = led_level_;
    x[PV_TOUCH_ANSWERED] = touch_open_ ? 0.0f : 1.0f;
    x[PV_PACE_FAST] = pace_[0];
    x[PV_PACE_SLOW] = pace_[1];
    x[PV_PACE_JITTER] = pace_[2];
}

bool BrainV4::tick(const ChannelEncoder::Out& channels)
{
    snn_.tick(channels, snn_out_);
    // the analyzer's first 5 s only settle its state (its readouts start at 0.5: "half threatened"); no decisions yet
    if (++ticks_ % 2 || ticks_ < kSettleTicks) return false;
    // pacemakers: Ornstein-Uhlenbeck at 10 Hz, ~3 s / ~60 s / ~0.5 s
    static const float keep[3] = {std::exp(-kNnDt / 3.0f), std::exp(-kNnDt / 60.0f), std::exp(-kNnDt / 0.5f)};
    for (int k = 0; k < 3; ++k) {
        pace_v_[k] = keep[k] * pace_v_[k] + std::sqrt(1.0f - keep[k] * keep[k]) * gauss();
        pace_[k] = std::tanh(pace_v_[k] * 0.9f);
    }
    // a new touch opens a question the robot has to answer (teacher_nn.Teacher.step)
    const float touch_now = std::max(snn_out_[v4::SO_SAL_TOUCH_HEAD], snn_out_[v4::SO_SAL_TOUCH_SCREEN]);
    if (touch_now > 0.12f && touch_now - prev_touch_ > 0.04f) {
        touch_open_ = true;
        touch_age_ = 0.0f;
    }
    prev_touch_ = touch_now;
    NnV4::In in;
    std::copy(snn_out_.begin(), snn_out_.end(), in.begin());
    prevInputs(in.data() + v4::SO_COUNT);
    std::copy(in.begin() + v4::SO_COUNT, in.end(), prev_.begin());
    nn_valid_ = nn_.step(in, nn_out_);
    if (!nn_valid_ && nn_.trained()) ++nn_faults_;
    return nn_valid_;
}

void BrainV4::executed(const Executed& e)
{
    if (e.emotion != emo_) {
        emo_ = std::clamp(e.emotion, 0, 20);
        emo_age_ = 0.0f;
    } else {
        emo_age_ += kNnDt;
    }
    if (e.said) {
        if (since_sound_ > 0.7f) touch_open_ = false;
        since_sound_ = 0.0f;
    } else {
        since_sound_ += kNnDt;
    }
    if (e.gesture) {
        if (since_gesture_ > 0.5f) touch_open_ = false;
        since_gesture_ = 0.0f;
    } else {
        since_gesture_ += kNnDt;
    }
    touch_age_ += kNnDt;
    if (touch_age_ > 3.0f) touch_open_ = false;
    head_yaw_ = std::clamp(e.headYawDeg / kGazeYawDeg, -1.5f, 1.5f);
    head_pitch_ = std::clamp(e.headPitchDeg / kGazePitchDeg, -1.5f, 1.5f);
    led_level_ = e.ledLevel;
}
