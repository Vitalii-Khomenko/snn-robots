#pragma once
#include <cstdint>
#include "audio/audio_output.h"
#include "core/brain_loop.h"
#include "led/led_engine.h"
#include "motion/motion_engine.h"
#include "motion/servo_supervisor.h"
#include "net/learned_store.h"
#include "net/life_log.h"
#include "net/ota_server.h"
#include "net/wifi_engine.h"
#include "power/power_status_engine.h"
#include "sensors/camera_engine.h"
#include "sensors/imu_engine.h"
#include "sensors/microphone_engine.h"
#include "sensors/presence_engine.h"
#include "sensors/touch_engine.h"
#include "ui/face_engine.h"
#include "voice/babble_voice.h"

// Hardware bring-up, polling and the robot-only safety nets. The senses feed brain v4 through BrainLoop; every
// actuator request comes out of the main network through V4Executor's limits. There is no behaviour code here. What
// else runs is only for statistics, analysis and debugging: the diary (net/life_log.h), the 5 s stats lines, the
// update server (net/ota_server.h). The simulator's SimRuntime wires the same BrainLoop around the same engines.
class RobotRuntime {
public:
    RobotRuntime();
    bool begin();
    void loop();

private:
    uint32_t nowMs() const;
    void logStats(uint32_t now);
    void logTick();
    void superviseServos(uint32_t now);
    void logBootReason();
    void confirmFirmware(uint32_t now);
    int formatStatus(char* buf, int size);
    void forgetOldProject();
    void loadLearned();
    void superviseLearned(uint32_t now);
    void keepBody(uint32_t now);

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
    OtaServer ota_;
    LearnedStore learned_;
    BrainLoop loop_;
    uint32_t learned_trial_ms_ = 0, learned_faults_ = 0, release_old_ms_ = 0;

    uint32_t logged_tick_ = 0;
    bool firmware_confirmed_ = false;
    uint32_t confirm_log_ms_ = 0;
    static constexpr uint32_t kConfirmGiveUpMs = 180000;
    float last_travel_deg_ = 0.0f;
    uint32_t volume_save_ms_ = 0;  // the speaker volume goes into NVS at this time (0 = nothing to save)
    // the body's slow state across reboots (keepBody): given back once the clock is known, saved every kBodySaveMs
    bool body_restored_ = false;
    uint32_t body_save_ms_ = 0;
    static constexpr uint32_t kBodySaveMs = 600000;
    ServoSupervisor servos_{motion_, imu_};  // the servo watchdog (motion/servo_supervisor.h)
    uint32_t panel_recover_ms_ = 0, next_panel_refresh_ms_ = 60000;
    uint32_t next_diagnostics_ms_ = 0;
    uint64_t diagnostics_window_start_us_ = 0;
    uint64_t diagnostics_busy_us_ = 0;
    static constexpr int kStages = 8;  // the main loop's stages (RobotRuntime::loop, stats line loop_us)
    uint64_t stage_sum_us_[kStages] = {};
    uint32_t stage_max_us_[kStages] = {};
    uint32_t stage_passes_ = 0;
};
