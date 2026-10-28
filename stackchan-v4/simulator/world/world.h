#pragma once
// The physical world around the simulated robot: the room (texture, window,
// lamps, daylight, ceiling light), people, sounds, the robot's own body
// (servo mechanics, head-mounted IMU, speaker), and its sensors' raw
// readings. Everything the firmware's drivers would read comes from here;
// everything its actuators do lands here.
//
// Conventions (match the firmware as verified on hardware 2026-09-26):
//   yaw +   = robot's right; image x grows to the right, MOTOR_LOOK_RIGHT
//   pitch + = up; image y grows downward
//   azimuth of things in the room uses the same axis as yaw.
//
// Several constants are estimates, not measurements (servo click level,
// camera FOV, microphone coupling). They are listed in docs/DEVELOPMENT.md (Fidelity)
// and can be changed in the scenario where it matters.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "scenario.h"

namespace sim {

class World {
public:
    static World& get();

    void load(const Scenario& scenario);
    const Scenario& scenario() const { return sc_; }

    // Advances physics and applies scenario events up to `t_us`.
    void advanceTo(uint64_t t_us);
    // The robot did something a person may answer ("responder" in a scenario; docs/CONTINGENCY.md):
    // 0 a call, 1 a gesture, 2 a face change.
    void robotActed(int kind);
    int answersGiven() const { return answers_given_; }
    uint64_t now() const { return now_us_; }

    // --- clock ------------------------------------------------------------
    float hourOfDay(uint64_t t_us) const;   // wall clock at boot + t
    float daylight(uint64_t t_us) const;    // 0 night .. 1 full day
    bool ownerHome() const { return owner_home_; }

    // --- servos (SCSCL bus) -------------------------------------------------
    void servoWrite(uint8_t id, int raw, uint16_t goal_ms);
    float headYawDeg() const { return yaw_deg_; }
    float headPitchDeg() const { return pitch_deg_; }
    float headSpeedDegPerS() const { return std::fabs(yaw_vel_) + std::fabs(pitch_vel_); }
    double travelDeg() const { return travel_deg_; }
    int movementsStarted() const { return movements_; }  // moves faster than 10 deg/s

    // --- camera -------------------------------------------------------------
    // Renders a QQVGA RGB565 frame (high byte first, like esp32-camera) of
    // what the head sees at `t_us`.
    void renderFrame(uint8_t* rgb565, int width, int height, uint64_t t_us);
    void lockExposure(uint64_t t_us);  // AEC/AGC off, as CameraEngine::begin() does
    // Where is lamp `i` in the current view? (-1..1 image coords, visible?)
    bool lampInView(size_t i, float& x, float& y) const;

    // --- microphones / speaker ---------------------------------------------
    // Interleaved stereo int16 for [t_end - frames/24 kHz, t_end).
    void micRead(int16_t* stereo, size_t frames, uint64_t t_end_us);
    void speakerPush(uint64_t t_us, const int16_t* samples, size_t count, int volume);

    // --- IMU, touch, presence/ALS, power -------------------------------------
    void imuAccel(float& x, float& y, float& z);
    uint8_t headTouchRaw() const;   // Si12T raw result (3 x 2-bit levels)
    bool takeScreenTap();           // one pending LCD tap, if any
    uint16_t alsRaw() const;        // LTR-553 ALS channel 0 counts
    uint16_t psRaw();               // LTR-553 proximity counts
    bool charging() const { return charging_; }
    // injected faults (scenario `fault imu|light|ntp on|off`, 2026-10-28, an audit A21): the IMU and the light /
    // proximity chip stop answering on the bus, the clock never syncs
    bool faultImu() const { return fault_imu_; }
    bool faultLight() const { return fault_light_; }
    bool faultNtp() const { return fault_ntp_; }
    int batteryPct() const { return battery_pct_; }

    // --- gyroscope: the head's real angular rates, deg/s -----------------
    void gyro(float& pitch_dps, float& yaw_dps);

    // Human-readable world log line (prefixed with the virtual time).
    void note(const char* fmt, ...) __attribute__((format(printf, 2, 3)));

private:
    struct Person {
        std::string name;
        float az = 0, dist = 1.5f;
        float walk_from = 0, walk_to = 0;
        uint64_t walk_start = 0, walk_end = 0;
        float dist_from = 1.5f;                  // gradual approach / retreat
        uint64_t dist_start = 0, dist_end = 0;
        bool present = false;
    };
    struct Sound {  // scheduled scenario sound
        enum Kind { Clap, Talk, Bang, Hiss, Tone, Whistle, Clip } kind = Clap;
        uint64_t start = 0, end = 0;
        float az = 0, level = 1;
        uint32_t seed = 0;
        Sound() = default;
        Sound(Kind k, uint64_t st, uint64_t en, float a, float lv, uint32_t sd)
            : kind(k), start(st), end(en), az(a), level(lv), seed(sd) {}
        bool ramp = false;  // level grows from 10% to 100% over the sound (someone approaching)
        float hz0 = 0, hz1 = 0;  // Tone / Whistle
        // Clip: a recorded sound (raw int16, 24 kHz mono), its gain, which classes it
        // belongs to (bit mask, see kClip*) and when it is actually audible (50 ms slots).
        std::shared_ptr<const std::vector<int16_t>> clip;
        std::shared_ptr<const std::vector<uint8_t>> active;
        float gain = 1.0f;
        uint8_t mask = 0;
    };
    struct Thing {  // a non-person object (ball, hand, pet) that moves through the view
        std::string name;
        bool present = false;
        float az = 0, el = 0, az_from = 0, el_from = 0, size_deg = 8, reflect = 0.7f;
        uint64_t move_start = 0, move_end = 0;
    };
    struct Pulse {  // servo click transient
        uint64_t start;
        float level;
    };

    void applyEvent(const Event& e);
    void stepPhysics(double dt_s);
    float reflectance(float az, float el) const;
    float radiance(float az, float el, uint64_t t_us) const;
    float ambient(uint64_t t_us) const;
    Person* person(const std::string& name);
    float personAz(const Person& p, uint64_t t_us) const;
    float personDist(const Person& p, uint64_t t_us) const;
    float relAz(float az) const;  // azimuth relative to where the head points, -180..180

    Scenario sc_;
    uint64_t now_us_ = 0;
    size_t next_event_ = 0;
    std::mt19937 rng_{1};
    std::normal_distribution<float> gauss_{0.0f, 1.0f};

    // room
    static constexpr int kTexW = 720, kTexH = 240;  // 0.5 deg cells, el -60..60
    std::vector<uint8_t> texture_;
    std::vector<LampSpec> lamps_;
    bool room_light_ = false;
    float extra_noise_ = 0.0f;  // "noise" event: TV/fan level
    std::vector<Person> people_;
    std::vector<Thing> things_;
    float extra_tilt_ = 0.0f;   // 0 white .. 1 low-pass (a fan)
    float bg_lp_[2] = {0, 0};
    float hiss_prev_ = 0.0f;
    double tone_phase_ = 0.0;
    float thingAz(const Thing& t, uint64_t t_us) const;
    float thingEl(const Thing& t, uint64_t t_us) const;
    std::vector<Sound> sounds_;
    std::map<std::string, std::pair<std::shared_ptr<std::vector<int16_t>>, std::shared_ptr<std::vector<uint8_t>>>> clips_;
    bool loadClip(const std::string& path, Sound& s);
    std::map<std::string, float> clip_ref_;
public:
    static constexpr uint8_t kClipSpeech = 1, kClipClap = 2, kClipSharp = 4, kClipContinuous = 8, kClipLow = 16,
                             kClipHigh = 32, kClipTonal = 64;
private:
    bool owner_home_ = false;
    bool charging_ = true;
    bool fault_imu_ = false, fault_light_ = false, fault_ntp_ = false;
    int battery_pct_ = 100;

    // servos
    float yaw_deg_ = 0, pitch_deg_ = 0;
    float yaw_target_ = 0, pitch_target_ = 0;
    float yaw_vel_ = 0, pitch_vel_ = 0;
    uint16_t yaw_goal_ms_ = 50, pitch_goal_ms_ = 50;
    bool was_moving_ = false;
    float move_peak_speed_ = 0.0f;
    double travel_deg_ = 0;
    int movements_ = 0;
    std::deque<Pulse> clicks_;
    float prev_yaw_vel_ = 0, prev_pitch_vel_ = 0;
    float yaw_accel_ = 0, pitch_accel_ = 0;

    // a person who answers the robot's acts (responder MODE ANSWER P LAT_LO LAT_HI) or at random (responder random
    // ANSWER MEAN_S), with what: pet / talk / tap
    struct Responder {
        bool on = false, random = false;
        bool acts[3] = {false, false, false};
        std::string answer = "pet";
        float p = 0.8f, lat_lo = 0.8f, lat_hi = 2.0f, mean_s = 20.0f;
        uint64_t next_random_us = 0;
    } responder_;
    std::vector<std::pair<uint64_t, std::string>> pending_answers_;
    int answers_given_ = 0;
    void answer(const std::string& what);

    // body handling
    uint64_t touch_until_ = 0;
    int pending_taps_ = 0;
    std::deque<uint64_t> tap_times_;
    uint64_t shake_until_ = 0;
public:
    // What the robot's IMU detector would report (the sim's own IMU path is held back by 'moving'): a hand shaking
    // the body, and the body lifted / turned over by more than ~20 degrees.
    bool shaking() const { return now_us_ < shake_until_; }
    bool bodyLifted() const { return std::fabs(body_pitch_deg_) > 20.0f || std::fabs(body_roll_deg_) > 20.0f; }
private:
    float body_pitch_deg_ = 0, body_roll_deg_ = 0, body_pitch_target_ = 0, body_roll_target_ = 0;

    // camera exposure
    bool exposure_locked_ = false;
    float exposure_gain_ = 1.0f;

    // speaker ring buffer (24 kHz)
    struct SpeakerChunk {
        uint64_t start_us;
        std::vector<int16_t> samples;
        float gain;
    };
    std::deque<SpeakerChunk> speaker_;

    // proximity: a hand or face in front of the LTR-553
    uint64_t near_until_ = 0;
};

}  // namespace sim
