#pragma once
#include <cstdint>
#include <functional>

class LearnedStore;

// Updates and checks over Wi-Fi: a small HTTP server on port 80; every request needs the header X-Token =
// LIFE_OTA_TOKEN (secrets/life_log_host.h). Used by tools/robot/ota.py.
//
//   GET /status    one line of JSON: firmware slot and state, uptime, diary counters, touch sensitivity
//   PUT /touch     body "0".."7": the head pad's sensitivity, applied at once and kept (no reboot)
//   PUT /learned   what the night sleep learned (v4/learned_blob.h) into the idle slot; the robot switches to it at once,
//                  without a reboot, on trial for a minute (net/learned_store.h)
//   PUT /firmware  an app image into the idle app slot, booted on trial (bootloader rollback: RobotRuntime confirms
//                  it after a healthy minute)
//
// A firmware upload reboots the robot when it is complete and checked; a learned blob does not.
class OtaServer {
public:
    using StatusFn = std::function<int(char* buf, int size)>;
    using SettingFn = std::function<bool(int value)>;
    bool begin(LearnedStore& learned, StatusFn status, SettingFn touchSensitivity);

private:
    LearnedStore* learned_ = nullptr;
    StatusFn status_;
    SettingFn touch_;
};
