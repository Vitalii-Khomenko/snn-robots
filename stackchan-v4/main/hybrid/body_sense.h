#pragma once
#include <array>
#include <cstdint>

#include "bus.h"

// The deterministic sensor/body encoders ("spinal" input side): normalizes
// head pose and speed, IMU, touch events, battery, stamina, time of day and
// the owner into the Body and Ctx sections of the FeatureBus, keeps the
// efference copy (what the brain asked the body to do) and the motor
// prediction error (what was asked versus what the gyro says happened).
// No learning, no decisions.
class BodySense {
public:
    enum class Event : uint8_t { HeadTouch, ScreenTouch, Shake, Tilt, OwnerArrived };

    struct Raw {
        float yawDeg = 0, pitchDeg = 0;
        float yawVelDegS = 0, pitchVelDegS = 0;            // the motion model's velocity
        bool gyroValid = false;
        float realYawVelDegS = 0, realPitchVelDegS = 0;    // gyro, mapped to yaw / pitch
        float imuPitch = 0, imuRoll = 0;                   // -1..1
        float battery01 = 1, stamina01 = 1, lightDelta = 0;
        bool speaker = false;                              // our own voice is playing
        bool timeSynced = false;
        float hour = 12.0f;
        bool ownerNear = false;
    };

    // Latched until the next tick reads it (events can arrive between ticks);
    // each stays visible for ~150 ms like the touch pad's own pulse.
    // Touches are short pulses; a shake is felt for 2 s and a lift / tilt for 1.5 s (the brain and the
    // policy need more than a 150 ms blip to answer it: another face, another state).
    void event(Event e, uint32_t nowMs)
    {
        // The owner's arrival is a one-tick pulse, as in training (tools/hybrid/synth.py; HybridBrain::updateSlow adds
        // 0.3 social need per tick of it): BrainLoop raises it in the tick that reads it, and 1 ms covers that tick only.
        const uint32_t hold = e == Event::Shake ? 2000 : e == Event::Tilt ? 1500 : e == Event::OwnerArrived ? 1 : 150;
        until_[static_cast<int>(e)] = nowMs + hold;
    }
    // Efference copy: the executor accepted a head move to this target.
    void observeMotorCommand(float yawDeg, float pitchDeg) { cmd_yaw_ = yawDeg; cmd_pitch_ = pitchDeg; }
    void observeSoundCommand(bool on) { cmd_sound_ = on; }
    void setVisionError(float e) { pe_vision_ = e; }

    void update(const Raw& raw, uint32_t nowMs, hybrid::Bus& bus);

    static constexpr float kYawScaleDeg = 100.0f, kPitchScaleDeg = 35.0f, kVelScaleDegS = 180.0f;

private:
    std::array<uint32_t, 5> until_{};
    float cmd_yaw_ = 0, cmd_pitch_ = 0;
    bool cmd_sound_ = false;
    std::array<std::array<float, 2>, 5> model_hist_{};  // last model velocities (lag tolerance)
    int hist_n_ = 0;
    float pe_motor_ = 0, pe_vision_ = 0;
};
