#pragma once
#include <cstdint>
#include <functional>

class WeightStore;

// Updates over Wi-Fi (platform B: the nightly training sends new weights while the robot sleeps). A small HTTP
// server on port 80; every request needs the header X-Token = LIFE_OTA_TOKEN (secrets/life_log_host.h). Used by
// tools/life/ota.py.
//
//   GET /status    one line of JSON: firmware slot and state, weights slot, version and trial, uptime
//   PUT /weights   a blob (tools/life/pack_weights.py) into the idle weights slot; the robot switches to it at once,
//                  without a reboot, on trial for a minute (WeightStore: a reboot during the trial boots it again, two
//                  such boots and it goes back)
//   PUT /mode      body "brain", "collector" or "life": who acts (v3 policy, babbling, the platform B actor); reboots
//   PUT /touch     body "0".."7": the head pad's sensitivity, applied at once and kept (no reboot)
//   GET /snapshot  the next camera frame: raw RGB565 + the front end's frame + the Vision NN's outputs (camera_engine.h)
//   GET /audio     the last 2 s of sound (int16, 24 kHz) + the Auditory NN's inputs of the last 2 s
//                  (both: the perception networks learn from them on the PC; the owner allowed it, 2026-10-17)
//   PUT /firmware  an app image into the idle app slot, booted on trial (bootloader rollback: RobotRuntime confirms
//                  it after a healthy minute)
//
// A firmware upload reboots the robot when it is complete and checked; weights do not.
class OtaServer {
public:
    using StatusFn = std::function<int(char* buf, int size)>;
    using ModeFn = std::function<bool(int mode)>;  // 0 brain, 1 collector, 2 life
    using SettingFn = std::function<bool(int value)>;
    // media for the PC: kind 0 a camera snapshot, 1 the last 2 s of sound; returns the bytes written into buf
    using MediaFn = std::function<int(int kind, uint8_t* buf, int cap)>;
    bool begin(WeightStore& weights, StatusFn status, ModeFn mode, SettingFn touchSensitivity, MediaFn media);

private:
    WeightStore* weights_ = nullptr;
    StatusFn status_;
    ModeFn mode_;
    SettingFn touch_;
    MediaFn media_;
};
