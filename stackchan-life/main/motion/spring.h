#pragma once
#include <algorithm>
#include <cmath>

// Minimal critically-damped-ish spring for smooth head motion. Same shape as
// the known-working driver's spring math
// (upstream/m5stack-StackChan/firmware/main/stackchan/motion/servo.cpp,
// map_speed_to_spring_options) so speed values feel the same.
struct Spring {
    float value      = 0.0f;
    float velocity    = 0.0f;
    float target       = 0.0f;
    float stiffness    = 170.0f;
    float damping       = 26.0f;
    float mass          = 1.0f;
    float rest_delta    = 0.1f;
    float rest_speed    = 0.1f;

    void teleport(float v)
    {
        value    = v;
        velocity = 0.0f;
        target   = v;
    }

    bool done() const { return std::fabs(target - value) < rest_delta && std::fabs(velocity) < rest_speed; }

    void update(float dt)
    {
        if (done()) {
            value    = target;
            velocity = 0.0f;
            return;
        }
        float force      = -stiffness * (value - target);
        float damp_force = -damping * velocity;
        float accel       = (force + damp_force) / mass;
        velocity += accel * dt;
        value += velocity * dt;
    }
};

// speed: 0..1000, matching the known-working driver's convention.
inline void applySpeedToSpring(Spring& s, int speed)
{
    speed              = std::clamp(speed, 0, 1000);
    constexpr float k_min = 10.0f;
    // Keep the response lively without the sharp acceleration that the old
    // 650 stiffness produced when a large gesture target arrived at once.
    constexpr float k_max = 420.0f;
    float n                = speed / 1000.0f;
    s.stiffness             = k_min + (n * n) * (k_max - k_min);
    s.damping               = 2.0f * std::sqrt(s.mass * s.stiffness);
    // Settle within ~1 servo step (0.3125 deg). A tighter rest (0.1) kept
    // the spring creeping for over a second, sending 1-raw-unit steps whose
    // clicks the microphone heard as loud sounds every ~3 s, while muting
    // the microphone for that whole creep made the robot deaf (hardware +
    // simulator, 2026-09-26).
    s.rest_delta = 0.4f;
    s.rest_speed = 1.0f;
}
