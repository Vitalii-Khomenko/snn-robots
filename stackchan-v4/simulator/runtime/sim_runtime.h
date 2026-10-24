#pragma once
// Host counterpart of main/robot_runtime.{h,cpp}: the same engines (real
// firmware classes) wired into the same BrainLoop, polled in the same order.
// RobotRuntime itself pulls in the display, NVS, Wi-Fi and I2S bring-up, so
// this file mirrors its begin()/loop() instead. Keep them in sync.

#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <string>

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
    bool v4 = false;               // brain v4 drives the body (mode v4)
    FILE* v4_dump = nullptr;       // every tick: 15 channels, 32 analyzer outputs, 61 main-network outputs (float32)
    FILE* trace = nullptr;    // CSV, one row per 100 ms
    FILE* summary = nullptr;  // per-minute table
    FILE* bus_dump = nullptr; // per brain tick: the policy input and the action (tools/hybrid training)
    // External policy (training rollouts): replaces the trained GRU. The
    // callback gets the policy input and fills the 28 action values.
    HybridBrain::ExternalPolicy policy;
    float brain_cost_ms = 4.0f;   // modelled brain-tick time on the ESP32-S3
    float loop_cost_ms = 1.5f;    // other work per main-loop pass
    uint32_t seed = 1;
};

class SimRuntime {
public:
    explicit SimRuntime(const RunOptions& options);
    bool begin();                 // RobotRuntime::begin()
    void run(uint64_t until_us);  // RobotRuntime::loop() until the given time
    void finish();

    HybridBrain& brain() { return loop_.brain(); }

private:
    uint32_t nowMs() const;
    void updateWifi(uint32_t now);
    void logStats(uint32_t now);
    void writeTrace(uint32_t now);
    void writeMinute(uint32_t now);

    RunOptions opt_;
    FaceEngine face_;
    LedEngine led_;
    AudioOutput audio_;
    BabbleVoice vocal_{audio_};
    MotionEngine motion_;
    TouchEngine touch_;
    ImuEngine imu_;
    uint32_t last_sim_shake_ms_ = 0, last_sim_tilt_ms_ = 0;
    bool sim_tilted_ = false;
    MicrophoneEngine mic_;
    PresenceEngine presence_;
    CameraEngine camera_;
    PowerStatusEngine power_status_;
    BrainLoop loop_;

    uint32_t next_telemetry_ms_ = 0;
    uint32_t next_diagnostics_ms_ = 0;

    // Wi-Fi model (WifiEngine): connect, NTP sync, 20-minute presence ping
    bool wifi_connected_ = false, time_synced_ = false, owner_nearby_ = false;
    uint32_t next_owner_check_ms_ = 0;

    std::mt19937 cost_rng_;
    // trace / summary bookkeeping
    uint32_t next_trace_ms_ = 0, next_minute_ms_ = 60000, minute_index_ = 0;
    double minute_travel_start_ = 0;
    int minute_moves_start_ = 0;
    int minute_ticks_ = 0, minute_lamp_view_ = 0, minute_lamp_centred_ = 0;
    uint64_t last_dump_tick_ = 0;
    double host_tick_s_ = 0.0;
    uint64_t host_ticks_ = 0;
};

// Boot time before RobotRuntime::begin() starts (bootloader, BSP, PSRAM).
constexpr uint64_t kBootUs = 4000000;

}  // namespace sim
