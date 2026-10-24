#pragma once
#include <cstdint>
#include <functional>

// Updates and checks over Wi-Fi: a small HTTP server on port 80; every request needs the header X-Token =
// LIFE_OTA_TOKEN (secrets/life_log_host.h). Used by tools/robot/ota.py.
//
//   GET /status    one line of JSON: firmware slot and state, uptime, diary counters, touch sensitivity
//   PUT /touch     body "0".."7": the head pad's sensitivity, applied at once and kept (no reboot)
//   PUT /firmware  an app image into the idle app slot, booted on trial (bootloader rollback: RobotRuntime confirms
//                  it after a healthy minute)
//
// A firmware upload reboots the robot when it is complete and checked.
class OtaServer {
public:
    using StatusFn = std::function<int(char* buf, int size)>;
    using SettingFn = std::function<bool(int value)>;
    bool begin(StatusFn status, SettingFn touchSensitivity);

private:
    StatusFn status_;
    SettingFn touch_;
};
