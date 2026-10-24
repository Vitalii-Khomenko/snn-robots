#pragma once
// Scenario files (*.scn): a room, the robot's situation, and a timeline of
// things people and the environment do. See docs/DEVELOPMENT.md for the
// full command list.

#include <cstdint>
#include <string>
#include <vector>

namespace sim {

struct LampSpec {
    std::string name;
    float az_deg = 0.0f;       // azimuth around the robot; + = robot's right
    float el_deg = 10.0f;      // elevation; + = up
    float radius_deg = 3.0f;   // apparent size of the bright part
    float brightness = 6.0f;   // emissive radiance, room reflectance scale (1 = white wall in daylight)
    bool on = true;
};

struct WindowSpec {
    bool present = false;
    float az_deg = 160.0f, el_deg = 15.0f, width_deg = 40.0f, height_deg = 30.0f;
    float gain = 4.0f;  // radiance at full daylight
};

struct Event {
    uint64_t at_us = 0;
    std::string verb;                 // touch_head, clap, lamp, person, ...
    std::vector<std::string> args;
    int line = 0;
};

struct Scenario {
    std::string name;
    std::string path;
    uint64_t duration_us = 10ull * 60 * 1000000;
    float clock_min = 12 * 60;        // wall clock at boot, minutes since midnight
    float sunrise_min = 7 * 60 + 30;
    float sunset_min = 19 * 60 + 30;
    uint32_t seed = 1;
    bool room_light = false;
    bool charging = true;             // the robot normally lives on USB
    int battery_pct = 100;
    bool owner_home = false;
    float noise_floor = 25.0f;        // microphone noise, int16 LSB
    float snn_cost_ms = 8.0f;         // mean NeuroBrain::tick() cost on the ESP32-S3 (estimate; see stats snn_tick_us)
    float loop_cost_ms = 1.5f;        // rest of one main-loop pass: I2C sensor polls etc.
    float fov_deg = 60.0f;            // camera horizontal field of view
    std::vector<LampSpec> lamps;
    WindowSpec window;
    std::vector<Event> events;        // sorted by time
};

// Throws std::runtime_error with "file:line: message" on errors.
Scenario loadScenario(const std::string& path);

// "90", "1:30", "1:02:03" -> microseconds.
uint64_t parseDuration(const std::string& text);
// "07:20" -> minutes since midnight.
float parseClock(const std::string& text);

}  // namespace sim
