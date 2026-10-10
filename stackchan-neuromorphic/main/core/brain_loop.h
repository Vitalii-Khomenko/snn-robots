#pragma once
#include <array>
#include <cstdint>

#include "track_stats.h"
#include "../audio/audio_output.h"
#include "../behavior/body_executor.h"
#include "../hybrid/auditory_net.h"
#include "../hybrid/body_sense.h"
#include "../hybrid/hybrid_brain.h"
#include "../led/led_engine.h"
#include "../motion/motion_engine.h"
#include "../power/power_status_engine.h"
#include "../sensors/camera_engine.h"
#include "../sensors/imu_engine.h"
#include "../sensors/microphone_engine.h"
#include "../sensors/presence_engine.h"
#include "../sensors/touch_engine.h"
#include "../ui/face_engine.h"
#include "../voice/babble_voice.h"

// Everything between the hardware engines and the hybrid brain, shared by
// the robot (RobotRuntime) and the digital twin (SimRuntime) so the two can
// not drift apart: the coupling of servo noise into the microphone and IMU,
// the vestibular (gyro) calibration, the body / vision / audio encoders that
// fill the FeatureBus, the 20 Hz brain tick and the executor call. It owns
// the Auditory NN, BodySense, HybridBrain and BodyExecutor.
class BrainLoop {
public:
    struct Engines {
        MotionEngine& motion;
        FaceEngine& face;
        LedEngine& led;
        AudioOutput& audio;
        BabbleVoice& vocal;
        ImuEngine& imu;
        MicrophoneEngine& mic;
        PresenceEngine& presence;
        CameraEngine& camera;
        PowerStatusEngine& power;
        TouchEngine& touch;
    };
    struct Env {
        bool timeSynced = false;
        float hour = 12.0f;
        bool ownerNear = false;
    };
    static constexpr uint32_t kTickIntervalMs = 50;

    explicit BrainLoop(Engines e);

    // ---- events from the engines' callbacks
    void headTouch(uint32_t now);
    void screenTouch(uint32_t now) { sense_.event(BodySense::Event::ScreenTouch, now); }
    void shake(uint32_t now) { sense_.event(BodySense::Event::Shake, now); }
    void tilt(uint32_t now) { sense_.event(BodySense::Event::Tilt, now); }
    // intensity as MicrophoneEngine reports it (0.6 .. 1.5), voice-like flag
    void loudSound(float intensity01) { loud_event_ = std::max(loud_event_, intensity01); }
    void soundOnset(float strength01) { onset_event_ = std::max(onset_event_, strength01); }
    void vision(const CameraEngine::VisionFrame& f, uint32_t now)
    {
        vision_frame_ = f;
        vision_frame_ms_ = now;
        vision_pending_ = true;
        track_.delivered();
    }

    // ---- per main-loop pass
    // Before the microphone / IMU update(): our own servos make noise.
    void coupleMotion(uint32_t now);
    // After every engine has updated: body encoders, and on the brain's
    // clock the Auditory NN, the brain tick and the executor.
    void step(uint32_t now, const Env& env);
    // After VocalTract::update(): keep the microphone from hearing our voice.
    void afterVoice(uint32_t now);

    HybridBrain& brain() { return brain_; }
    BodyExecutor& executor() { return executor_; }
    BodySense& sense() { return sense_; }
    const AuditoryNet& auditory() const { return auditory_; }
    const BodySense::Raw& raw() const { return raw_; }
    bool gyroValid() const { return raw_.gyroValid; }
    // diagnostics
    uint64_t tickLastUs() const { return tick_last_us_; }
    uint64_t takeTickMaxUs() { const uint64_t m = tick_max_us_; tick_max_us_ = 0; return m; }
    uint32_t missedTicks() const { return missed_ticks_; }
    uint32_t tickPeriodMs() const { return tick_period_ms_; }
    uint32_t audioUs() const { return audio_us_; }
    // Concept-style telemetry (docs/ARCHITECTURE.md): one compact line.
    int formatTelemetry(char* buf, int size) const;
    // Dataset line for offline retraining (concept phase 10): the policy input and
    // the action of the last tick as signed 16-bit hex, value x 1000.
    int formatDataset(uint32_t nowMs, char* buf, int size) const;
    // The Auditory NN's last input (100 values) as hex, for recording the real microphone.
    int formatAudioRecord(uint32_t nowMs, char* buf, int size) const;

private:
    void updateBody(uint32_t now);
    void feedVision(uint32_t now);
    void holdPerson();
    void tick(uint32_t now, const Env& env);

    Engines e_;
    AuditoryNet auditory_;
    BodySense sense_;
    HybridBrain brain_;
    BodyExecutor executor_;
    BodySense::Raw raw_{};

    CameraEngine::VisionFrame vision_frame_{};
    uint32_t vision_frame_ms_ = 0;
    bool vision_pending_ = false;
    int vision_age_ticks_ = 1000;
    float pe_vision_ = 0.0f;
    struct PersonHold {
        float person = 0, face = 0, conf = 0, worldYaw = 0, worldPitch = 0;
        int age = 1000;
    } hold_;

    float gyro_yaw_corr_[3] = {0, 0, 0}, gyro_pitch_corr_[3] = {0, 0, 0};
    bool was_moving_ = false;
    uint32_t motion_edge_ms_ = 0, own_motion_until_ms_ = 0, self_voice_until_ms_ = 0;
    uint32_t pending_touch_ms_ = 0;  // a touch that arrived during motion, waiting to be told from a servo phantom
    float loud_event_ = 0.0f, onset_event_ = 0.0f, loud_hold_ = 0.0f, onset_hold_ = 0.0f;
    bool owner_was_nearby_ = false;

    uint32_t next_tick_ms_ = 0, last_tick_ms_ = 0, tick_period_ms_ = 0, missed_ticks_ = 0;
    uint64_t tick_last_us_ = 0, tick_max_us_ = 0;
    uint32_t audio_us_ = 0;
    std::array<bool, 3> heard_{};  // clap / sharp / speech above 0.5 (rising edges are logged)
    TrackStats track_;
    bool frame_this_tick_ = false;
};
