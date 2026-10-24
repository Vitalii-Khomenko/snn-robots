#pragma once
#include <cstdint>
#include "audio/audio_output.h"
#include "core/brain_loop.h"
#include "led/led_engine.h"
#include "motion/motion_engine.h"
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
    BrainLoop loop_;

    uint32_t logged_tick_ = 0;
    bool firmware_confirmed_ = false;
    float last_travel_deg_ = 0.0f;
    // Servo watchdog (the bus is write-only on this board: the gyro is the only way to know the head really moves).
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
