#include "channel_encoder.h"

#include <algorithm>
#include <cmath>

namespace {
double clamp01(double x) { return x < 0.0 ? 0.0 : x > 1.0 ? 1.0 : x; }
double alpha(double tau) { return 1.0 - std::exp(-ChannelEncoder::kDt / tau); }
double keep(double tau) { return std::exp(-ChannelEncoder::kDt / tau); }
}  // namespace

float ChannelEncoder::nightRamp(float h)
{
    if (h >= 21.0f && h < 21.75f) return (h - 21.0f) / 0.75f;
    if (h >= 21.75f || h < 6.0f) return 1.0f;
    if (h >= 6.0f && h < 6.75f) return 1.0f - (h - 6.0f) / 0.75f;
    return 0.0f;
}

void ChannelEncoder::update(const Raw& r, Out& out)
{
    static const double a_loud = alpha(kLoudTau), a_voice = alpha(kVoiceTau), k_sharp = keep(kSharpTau),
                        k_touch = keep(kTouchTau), k_shake = keep(kShakeDecay), a_fit = alpha(kTiltFitTau),
                        a_tilt = alpha(kTiltTau), a_near = alpha(kNearTau), a_light = alpha(kLightTau),
                        a_motion = alpha(kMotionTau), a_mfall = alpha(kMotionFallTau), k_mhold = keep(kMotionHoldDecay),
                        a_spot = alpha(kSpotSlowTau), k_spot_hab = keep(kSpotHabTau), k_forget = keep(kSpotForgetTau),
                        a_spot_fall = alpha(kSpotFallTau),
                        log_light_full = std::log10(1.0 + kLightFull);
    const double dt = kDt;

    loud_ += (r.energy - loud_) * a_loud;
    sharp_ = sharp_ * k_sharp + (r.sharpEv <= 0.0f ? 0.0 : 1.0 + 5.0 * clamp01((r.sharpEv - 1.0) / 0.5));
    voice_ += ((r.speech > 0.4f ? 1.0 : 0.0) - voice_) * a_voice;
    head_ = head_ * k_touch + r.headEv + r.headHold * kHoldRate * dt;
    screen_ = screen_ * k_touch + r.screenEv;
    shake_ = std::max(clamp01((r.accelSd - kShakeFloor) / kShakeFull), shake_ * k_shake);

    // Tilt. Sideways the IMU reads clean (0.5 deg noise when upright). Forward / back it turns with the pitch servo,
    // and the commanded pitch does not tell where the head really is (the servos often stop short: roll residual
    // 13 deg, 2026-10-19..21), so that axis only counts beyond the band the head itself can make (its roll spans 0.25..0.53 over a day, 2026-10-24).
    double tilt_now = 0.0;
    if (r.imuOk) {
        const double dev = std::max(std::fabs(r.imuPitch - p0_),
                                    2.0 * std::max(0.0, std::fabs(r.imuRoll - r0_) - kTiltRollBand));
        tilt_now = n_ >= kTiltWarm ? clamp01(dev) : 0.0;
        steady_ = (tilt_now > 0.11 && shake_ < 0.05 && std::fabs(tilt_now - tilt_) < 0.03) ? steady_ + dt : 0.0;
        // upright and still (or the first minute after boot, or the same lean for 10 minutes: the new normal)
        if (n_ < kTiltWarm || (tilt_now < 0.11 && shake_ < 0.05) || steady_ > kTiltAcceptS) {
            n_ = std::min(n_ + 1.0, 1e9);
            const double a = steady_ <= kTiltAcceptS ? std::max(a_fit, 1.0 / n_) : 0.01;
            p0_ += (r.imuPitch - p0_) * a;
            r0_ += (r.imuRoll - r0_) * a;
        }
    }
    tilt_ += (tilt_now - tilt_) * a_tilt;

    if (r.nearOk) near_ += (clamp01(r.nearRaw / kNearFull) - near_) * a_near;
    if (r.alsOk) light_ += (clamp01(std::log10(1.0 + std::max(r.als, 0.0f)) / log_light_full) - light_) * a_light;
    if (r.frameOk) {
        const double target = clamp01(r.motionFrac * kMotionGain);
        motion_ += (target - motion_) * (target > motion_ ? a_motion : a_mfall);
        if (r.motionFrac > 0.01f) {
            mx_ = r.motionX;
            my_ = r.motionY;
        }
        const double s = clamp01(r.lightStr);
        double nov = 0.0;
        if (s > 0.0) {
            lx_ = r.lightX;
            ly_ = r.lightY;
            // the two nearest directions, linearly
            const double p = std::clamp((r.headYaw + r.lightX * kSpotCamHalfDeg + 90.0) / kSpotPlaceDeg, 0.0,
                                        kSpotPlaces - 1.0);
            const int i = std::min(static_cast<int>(p), kSpotPlaces - 2);
            const double w = p - i;
            const double known = place_[i] * (1.0 - w) + place_[i + 1] * w;
            nov = std::max(0.0, s - known - kSpotMargin);
            // both by the same step: the read value learns at a_spot (up) / a_spot_fall (down)
            const double step = (s - known) * (s > known ? a_spot : a_spot_fall);
            place_[i] += step;
            place_[i + 1] += step;
        }
        spot_ = clamp01(kSpotNovGain * nov) * (1.0 - kSpotHabCut * spot_hab_);
    } else {
        motion_ *= 1.0 - (1.0 - k_mhold) * 0.25;  // no usable frame: fade slowly, do not jump to 0
        spot_ *= 1.0 - (1.0 - k_mhold) * 0.25;
    }
    spot_hab_ = clamp01(spot_hab_ * k_spot_hab + spot_ * kSpotHabRate);
    for (double& b : place_) b *= k_forget;
    if (motion_ < 0.02) {
        mx_ *= 0.98;
        my_ *= 0.98;
    }
    if (spot_ < 0.02) {
        lx_ *= 0.98;
        ly_ *= 0.98;
    }

    const double hunger = clamp01(1.0 - r.battery);
    const double tired = clamp01(0.75 * (r.clockOk ? nightRamp(r.hour) : 0.0) + 0.25 * (1.0 - r.stamina));
    lonely_t_ += dt;
    if (r.headEv > 0 || r.screenEv > 0 || r.headHold > 0 || r.speech > kContactSpeech || near_ > kContactNear)
        lonely_t_ = 0.0;
    else if (motion_ > kContactMotion && lonely_t_ > kLonelyMotionCap)
        lonely_t_ = kLonelyMotionCap;

    using namespace v4;
    out[CH_LOUD] = static_cast<float>(clamp01(loud_ * kLoudGain));
    out[CH_SHARP] = static_cast<float>(clamp01(sharp_ / kSharpFull));
    out[CH_VOICE] = static_cast<float>(clamp01(voice_));
    out[CH_TOUCH_HEAD] = static_cast<float>(1.0 - std::exp(-head_ / kHeadK));
    out[CH_TOUCH_SCREEN] = static_cast<float>(1.0 - std::exp(-screen_ / kScreenK));
    out[CH_SHAKE] = static_cast<float>(shake_);
    out[CH_TILT] = static_cast<float>(clamp01(tilt_));
    out[CH_NEAR] = static_cast<float>(near_);
    out[CH_LIGHT] = static_cast<float>(light_);
    out[CH_MOTION] = static_cast<float>(motion_);
    out[CH_HUNGER] = static_cast<float>(hunger);
    out[CH_TIRED] = static_cast<float>(tired);
    out[CH_LONELY] = static_cast<float>(1.0 - std::exp(-lonely_t_ / kLonelyTau));
    out[int(CH_COUNT) + int(WH_MOTION_X)] = static_cast<float>(mx_);
    out[int(CH_COUNT) + int(WH_MOTION_Y)] = static_cast<float>(my_);
    out[int(CH_COUNT) + int(WH_LIGHT_X)] = static_cast<float>(lx_);
    out[int(CH_COUNT) + int(WH_LIGHT_Y)] = static_cast<float>(ly_);
    out[int(CH_COUNT) + int(WH_LIGHT_SPOT)] = static_cast<float>(spot_);
}
