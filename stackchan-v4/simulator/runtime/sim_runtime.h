#pragma once
// Host counterpart of main/robot_runtime.{h,cpp}: the same engines (real firmware classes) wired into the same
// BrainLoop, polled in the same order. RobotRuntime itself pulls in the display, NVS, Wi-Fi and I2S bring-up, so this
// file mirrors its begin()/loop() instead. Keep them in sync.

#include <cstdint>
#include <cstdio>
#include <random>

#include "../../main/audio/audio_output.h"
#include "../../main/core/brain_loop.h"
#include "../../main/led/led_engine.h"
#include "../../main/motion/motion_engine.h"
#include "../../main/power/power_status_engine.h"
#include "../../main/sensors/camera_engine.h"
#include "../../main/sensors/imu_engine.h"
#include "../../main/sensors/microphone_engine.h"
#include "../../main/sensors/presence_engine.h"
#include "../../main/sensors/touch_engine.h"
#include "../../main/ui/face_engine.h"
#include "../../main/voice/babble_voice.h"

namespace sim {

struct RunOptions {
    FILE* v4_dump = nullptr;  // every brain tick: 15 channels, 32 analyzer outputs, 61 main-network outputs (float32)
    const uint8_t* learned = nullptr;  // a learned blob (main/v4/learned_blob.h), checked; nullptr = innate
    FILE* trace = nullptr;    // CSV, one row per 100 ms
    FILE* summary = nullptr;  // per-minute table
    float brain_cost_ms = 4.0f;  // modelled brain-tick time on the ESP32-S3
    float loop_cost_ms = 1.5f;   // other work per main-loop pass
    uint32_t seed = 1;
};

class SimRuntime {
public:
    explicit SimRuntime(const RunOptions& options);
    bool begin();                 // RobotRuntime::begin()
    void run(uint64_t until_us);  // RobotRuntime::loop() until the given time
    void finish();

private:
    uint32_t nowMs() const;
    void updateWifi(uint32_t now);
    void logStats(uint32_t now);
    void writeTrace(uint32_t now);
    void writeMinute(uint32_t now);

    RunOptions opt_;
    FaceEngine face_;
    int last_face_ = -1;
    LedEngine led_;
    AudioOutput audio_;
    BabbleVoice vocal_{audio_};
    MotionEngine motion_;
    TouchEngine touch_;
    ImuEngine imu_;
    MicrophoneEngine mic_;
    PresenceEngine presence_;
    CameraEngine camera_;
    PowerStatusEngine power_status_;
    BrainLoop loop_;

    uint32_t next_diagnostics_ms_ = 0;
    // Wi-Fi model (WifiEngine): connect, NTP sync
    bool wifi_connected_ = false, time_synced_ = false;

    std::mt19937 cost_rng_;
    // trace / summary bookkeeping
    uint32_t next_trace_ms_ = 0, next_minute_ms_ = 60000, minute_index_ = 0;
    uint32_t start_ms_ = 0;  // the firmware clock at begin() (--ms-offset: near its wrap)
    double minute_travel_start_ = 0;
    int minute_moves_start_ = 0;
    double host_tick_s_ = 0.0;
    uint64_t host_ticks_ = 0;
};

// Boot time before RobotRuntime::begin() starts (bootloader, BSP, PSRAM).
constexpr uint64_t kBootUs = 4000000;

}  // namespace sim
