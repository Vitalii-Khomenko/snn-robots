#include "body_sense.h"

#include <algorithm>
#include <cmath>

#include "../core/ms_clock.h"

using namespace hybrid;

void BodySense::update(const Raw& r, uint32_t now, Bus& bus)
{
    auto on = [&](Event e) { return msWithin(now, until_[static_cast<int>(e)]) ? 1.0f : 0.0f; };
    auto& b = bus.body;
    b[BODY_YAW] = std::clamp(r.yawDeg / kYawScaleDeg, -1.3f, 1.3f);
    b[BODY_PITCH] = std::clamp(r.pitchDeg / kPitchScaleDeg, -1.3f, 1.3f);
    b[BODY_YAW_VEL] = std::clamp(r.yawVelDegS / kVelScaleDegS, -1.0f, 1.0f);
    b[BODY_PITCH_VEL] = std::clamp(r.pitchVelDegS / kVelScaleDegS, -1.0f, 1.0f);
    b[BODY_REAL_YAW_VEL] = r.gyroValid ? std::clamp(r.realYawVelDegS / kVelScaleDegS, -1.0f, 1.0f) : 0.0f;
    b[BODY_REAL_PITCH_VEL] = r.gyroValid ? std::clamp(r.realPitchVelDegS / kVelScaleDegS, -1.0f, 1.0f) : 0.0f;
    b[BODY_IMU_PITCH] = std::clamp(r.imuPitch, -1.0f, 1.0f);
    b[BODY_IMU_ROLL] = std::clamp(r.imuRoll, -1.0f, 1.0f);
    b[BODY_TOUCH_HEAD] = on(Event::HeadTouch);
    b[BODY_TOUCH_SCREEN] = on(Event::ScreenTouch);
    b[BODY_SHAKE] = on(Event::Shake);
    b[BODY_TILT] = on(Event::Tilt);
    b[BODY_BATTERY] = std::clamp(r.battery01, 0.0f, 1.0f);
    b[BODY_STAMINA] = std::clamp(r.stamina01, 0.0f, 1.0f);
    b[BODY_LIGHT_DELTA] = std::clamp(r.lightDelta, -1.0f, 1.0f);
    b[BODY_SPEAKER] = r.speaker ? 1.0f : 0.0f;

    // Time of day (22:00-06:30 night, 06:30-20:00 day; two song windows with
    // a 2 minute ramp: dawn 06:40-07:07, dusk 21:25-21:52).
    auto& c = bus.ctx;
    c.fill(0.0f);
    if (r.timeSynced) {
        const float h = r.hour;
        c[CTX_NIGHT] = (h >= 22.0f || h < 6.5f) ? 1.0f : 0.0f;
        c[CTX_DAY] = (h >= 6.5f && h < 20.0f) ? 1.0f : 0.0f;
        auto window = [h](float from, float to) {
            return (h < from || h >= to) ? 0.0f : std::min(1.0f, (h - from) / (2.0f / 60.0f));
        };
        c[CTX_DAWN] = window(6.0f + 40.0f / 60.0f, 7.0f + 7.0f / 60.0f);
        c[CTX_DUSK] = window(21.0f + 25.0f / 60.0f, 21.0f + 52.0f / 60.0f);
        c[CTX_HOUR_SIN] = std::sin(h / 24.0f * 6.2831853f);
        c[CTX_HOUR_COS] = std::cos(h / 24.0f * 6.2831853f);
    }
    c[CTX_OWNER_NEAR] = r.ownerNear ? 1.0f : 0.0f;
    c[CTX_OWNER_ARRIVED] = on(Event::OwnerArrived);

    // Efference copy.
    bus.eff[EFF_YAW] = std::clamp((cmd_yaw_ - r.yawDeg) / 60.0f, -1.0f, 1.0f);
    bus.eff[EFF_PITCH] = std::clamp((cmd_pitch_ - r.pitchDeg) / 40.0f, -1.0f, 1.0f);
    bus.eff[EFF_SOUND] = cmd_sound_ ? 1.0f : 0.0f;

    // Motor prediction error: what the gyro says versus what the motion
    // model expected. The model leads the servos by up to a few ticks, so
    // the error is the smallest difference to any of the last few model
    // velocities; small velocities are inside the gyro noise.
    for (size_t i = model_hist_.size() - 1; i > 0; --i) model_hist_[i] = model_hist_[i - 1];
    model_hist_[0] = {r.yawVelDegS, r.pitchVelDegS};
    hist_n_ = std::min<int>(hist_n_ + 1, static_cast<int>(model_hist_.size()));
    float err = 0.0f;
    if (r.gyroValid && hist_n_ == static_cast<int>(model_hist_.size())) {
        float ey = 1e9f, ep = 1e9f;
        for (const auto& m : model_hist_) {
            ey = std::min(ey, std::fabs(r.realYawVelDegS - m[0]));
            ep = std::min(ep, std::fabs(r.realPitchVelDegS - m[1]));
        }
        err = std::max(0.0f, std::max(ey, ep) - 8.0f) / 60.0f;
    }
    pe_motor_ += (std::clamp(err, 0.0f, 1.0f) - pe_motor_) * 0.3f;
    bus.pe[PE_MOTOR] = pe_motor_;
    bus.pe[PE_VISION] = pe_vision_;
}
