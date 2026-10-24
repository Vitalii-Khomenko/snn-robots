#pragma once
#include <atomic>
#include <cstdint>
#include "audio/audio_output.h"
#include "core/brain_loop.h"
#include "led/led_engine.h"
#include "motion/motion_engine.h"
#include "net/life_log.h"
#include "behavior/collector.h"
#include "hybrid/actor_net.h"
#include "hybrid/online_learner.h"
#include "hybrid/world_model.h"
#include "net/ota_server.h"
#include "net/weight_store.h"
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
    LifeLog life_;
    WeightStore weights_;
    WorldModel world_;
    // platform B data collector instead of the policy (behavior/collector.h); NVS life/collector, PUT /mode
    // who acts: 0 the v3 policy network, 1 the collector (random babbling), 2 life (platform B: the actor's decisions)
    int mode_ = 0;
    bool collector_mode_ = false;  // modes 1 and 2: the collector writes the action vector
    Collector collector_;
    ActorNet actor_;
    OnlineLearner learner_;  // stage 4: learning by day on top of the actor (hybrid/online_learner.h)
    float wm_log_[WorldModel::kLogValues] = {};
    // the Auditory NN's inputs of the last 2 s (40 ticks x 100), for GET /audio (PSRAM)
    float* audio_feat_ = nullptr;
    std::atomic<uint32_t> audio_feat_n_{0};
    int formatMedia(int kind, uint8_t* buf, int cap);  // the world model's last values: part of the actor's state
    void loadMode();
    bool setMode(int mode);  // stage 1: predicts the next tick for the diary (main/hybrid/world_model.h)
    uint32_t world_us_ = 0;
    OtaServer ota_;
    int formatStatus(char* buf, int size);
    void loadWeights();
    void logBootReason();
    void applyWeights();
    void superviseWeights(uint32_t now);
    uint32_t weights_trial_ms_ = 0, weights_faults_ = 0, release_old_ms_ = 0;
    BrainLoop loop_;
    uint32_t logged_tick_ = 0;
    void logTick(uint32_t now);
    void confirmFirmware(uint32_t now);
    bool firmware_confirmed_ = false;

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
    uint32_t panel_recover_ms_ = 0, next_panel_refresh_ms_ = 60000;
    uint32_t repower_times_[3] = {0, 0, 0};
    int series_level_ = 0;
    bool blocked_logged_ = false;
    uint32_t sag_repower_ms_ = 0, sag_pause_until_ms_ = 0;
    uint32_t next_diagnostics_ms_ = 0;
    uint64_t diagnostics_window_start_us_ = 0;
    uint64_t diagnostics_busy_us_ = 0;
};
