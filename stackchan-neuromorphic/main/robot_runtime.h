#pragma once
#include <cstdint>
#include "audio/audio_output.h"
#include "core/brain_loop.h"
#include "led/led_engine.h"
#include "motion/motion_engine.h"
#include "net/wifi_engine.h"
#include "power/power_status_engine.h"
#include "sensors/camera_engine.h"
#include "sensors/imu_engine.h"
#include "sensors/microphone_engine.h"
#include "sensors/presence_engine.h"
#include "sensors/touch_engine.h"
#include "ui/face_engine.h"
#include "voice/babble_voice.h"

// Hardware bring-up, polling and the robot-only safety nets. Sensors feed the
// hybrid brain's FeatureBus through BrainLoop (Vision NN in the camera task,
// Auditory NN, deterministic body encoders); every actuator request comes
// out of the policy's action vector through BodyExecutor's physical limits.
// There is no behavior code here. The simulator's SimRuntime wires the same
// BrainLoop around the same engines.
class RobotRuntime {
public:
    RobotRuntime();
    bool begin();
    void loop();
private:
    uint32_t nowMs() const;
    void logStats(uint32_t now);
    void superviseServos(uint32_t now);
    // What the robot learned by itself (AssocMemory) lives in NVS: loaded at boot, saved every 10 min when it changed.
    void loadLearned();
    void saveLearned(uint32_t now);
    MotionEngine motion_;
    FaceEngine face_;
    LedEngine led_;
    AudioOutput audio_;
    BabbleVoice vocal_{audio_};
    TouchEngine touch_;
    ImuEngine imu_;
    MicrophoneEngine mic_;
    PresenceEngine presence_;
    CameraEngine camera_;
    PowerStatusEngine power_status_;
    WifiEngine wifi_;
    BrainLoop loop_;

    uint32_t next_learn_save_ms_ = 0;
    float learn_change_ = 0.0f;
    uint32_t next_telemetry_ms_ = 0, next_dataset_ms_ = 0, next_audio_rec_ms_ = 0;
    float last_travel_deg_ = 0.0f;
    // Servo watchdog (the bus is write-only on this board: the gyro is the
    // only way to know the head really moves).
    uint32_t watch_window_ms_ = 0, last_repower_ms_ = 0;
    float watch_travel_deg_ = 0.0f;
    int stalled_windows_ = 0;
    uint32_t servo_repowers_ = 0;
    uint32_t next_diagnostics_ms_ = 0;
    uint64_t diagnostics_window_start_us_ = 0;
    uint64_t diagnostics_busy_us_ = 0;
};
