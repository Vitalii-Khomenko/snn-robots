#include "scenario.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>


namespace sim {
namespace {

std::vector<std::string> split(const std::string& line)
{
    std::istringstream in(line);
    std::vector<std::string> out;
    std::string word;
    while (in >> word) out.push_back(word);
    return out;
}

[[noreturn]] void fail(const std::string& path, int line, const std::string& msg)
{
    throw std::runtime_error(path + ":" + std::to_string(line) + ": " + msg);
}

float num(const std::string& path, int line, const std::string& text)
{
    try {
        size_t used = 0;
        const float value = std::stof(text, &used);
        if (used != text.size()) throw std::invalid_argument(text);
        return value;
    } catch (const std::exception&) {
        fail(path, line, "expected a number, got '" + text + "'");
    }
}

bool onOff(const std::string& path, int line, const std::string& text)
{
    if (text == "on" || text == "yes" || text == "home") return true;
    if (text == "off" || text == "no" || text == "away") return false;
    fail(path, line, "expected on/off, got '" + text + "'");
}

// Commands allowed after "at"/"every", with their minimum argument count.
const std::vector<std::pair<std::string, size_t>> kEventVerbs = {
    {"touch_head", 0}, {"tap_screen", 0}, {"double_tap_screen", 0}, {"triple_tap_screen", 0},
    {"pet", 0}, {"shake", 0}, {"pick_up", 0}, {"put_down", 0}, {"clap", 0}, {"talk", 0}, {"talk_ramp", 0},
    {"noise", 1}, {"bang", 0}, {"hiss", 0}, {"tone", 2}, {"whistle", 3}, {"thing", 2}, {"clip", 2}, {"lamp", 2}, {"room_light", 1}, {"person", 2}, {"owner", 1},
    {"charging", 1}, {"battery", 1}, {"log", 1}, {"responder", 1}, {"fault", 1},
};

}  // namespace

uint64_t parseDuration(const std::string& text)
{
    std::vector<double> parts;
    std::string part;
    std::istringstream in(text);
    while (std::getline(in, part, ':')) parts.push_back(std::stod(part));
    double seconds = 0.0;
    for (double p : parts) seconds = seconds * 60.0 + p;
    return static_cast<uint64_t>(seconds * 1e6 + 0.5);
}

float parseClock(const std::string& text)
{
    const auto colon = text.find(':');
    if (colon == std::string::npos) throw std::invalid_argument(text);
    return std::stof(text.substr(0, colon)) * 60.0f + std::stof(text.substr(colon + 1));
}

Scenario loadScenario(const std::string& path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open scenario " + path);
    Scenario sc;
    sc.path = path;
    sc.name = path.substr(path.find_last_of('/') + 1);
    sc.name = sc.name.substr(0, sc.name.find('.'));

    struct Repeat {
        uint64_t first, period, until;
        Event event;
    };
    std::vector<Repeat> repeats;
    std::string raw;
    int line_no = 0;
    while (std::getline(file, raw)) {
        ++line_no;
        const auto hash = raw.find('#');
        if (hash != std::string::npos) raw.resize(hash);
        auto w = split(raw);
        if (w.empty()) continue;
        const std::string& key = w[0];
        auto need = [&](size_t n) {
            if (w.size() < n + 1) fail(path, line_no, key + ": expected " + std::to_string(n) + " argument(s)");
        };
        try {
            if (key == "duration") { need(1); sc.duration_us = parseDuration(w[1]); }
            else if (key == "clock") { need(1); sc.clock_min = parseClock(w[1]); }
            else if (key == "sunrise") { need(1); sc.sunrise_min = parseClock(w[1]); }
            else if (key == "sunset") { need(1); sc.sunset_min = parseClock(w[1]); }
            else if (key == "seed") { need(1); sc.seed = static_cast<uint32_t>(num(path, line_no, w[1])); }
            else if (key == "room_light") { need(1); sc.room_light = onOff(path, line_no, w[1]); }
            else if (key == "charging") { need(1); sc.charging = onOff(path, line_no, w[1]); }
            else if (key == "battery") { need(1); sc.battery_pct = static_cast<int>(num(path, line_no, w[1])); }
            else if (key == "owner") { need(1); sc.owner_home = onOff(path, line_no, w[1]); }
            else if (key == "noise_floor") { need(1); sc.noise_floor = num(path, line_no, w[1]); }
            else if (key == "snn_cost") { need(1); sc.snn_cost_ms = num(path, line_no, w[1]); }
            else if (key == "loop_cost") { need(1); sc.loop_cost_ms = num(path, line_no, w[1]); }
            else if (key == "fov") { need(1); sc.fov_deg = num(path, line_no, w[1]); }
            else if (key == "lamp") {
                // lamp NAME AZ EL [on|off] [brightness B] [radius R]
                need(3);
                LampSpec lamp;
                lamp.name = w[1];
                lamp.az_deg = num(path, line_no, w[2]);
                lamp.el_deg = num(path, line_no, w[3]);
                for (size_t i = 4; i < w.size(); ++i) {
                    if (w[i] == "on" || w[i] == "off") lamp.on = w[i] == "on";
                    else if (w[i] == "brightness" && i + 1 < w.size()) lamp.brightness = num(path, line_no, w[++i]);
                    else if (w[i] == "radius" && i + 1 < w.size()) lamp.radius_deg = num(path, line_no, w[++i]);
                    else fail(path, line_no, "lamp: unknown option '" + w[i] + "'");
                }
                sc.lamps.push_back(lamp);
            } else if (key == "window") {
                // window AZ EL WIDTH HEIGHT [gain G]
                need(4);
                sc.window.present = true;
                sc.window.az_deg = num(path, line_no, w[1]);
                sc.window.el_deg = num(path, line_no, w[2]);
                sc.window.width_deg = num(path, line_no, w[3]);
                sc.window.height_deg = num(path, line_no, w[4]);
                if (w.size() >= 7 && w[5] == "gain") sc.window.gain = num(path, line_no, w[6]);
            } else if (key == "at" || key == "every") {
                // at T VERB ARGS...
                // every PERIOD [from T] [until T] VERB ARGS...
                need(2);
                size_t i = 1;
                uint64_t first = 0, period = 0, until = UINT64_MAX;
                if (key == "at") {
                    first = parseDuration(w[i++]);
                } else {
                    period = parseDuration(w[i++]);
                    if (period == 0) fail(path, line_no, "every: period must be > 0");
                    first = period;
                    while (i + 1 < w.size() && (w[i] == "from" || w[i] == "until")) {
                        if (w[i] == "from") first = parseDuration(w[i + 1]);
                        else until = parseDuration(w[i + 1]);
                        i += 2;
                    }
                }
                if (i >= w.size()) fail(path, line_no, key + ": missing event");
                const std::string verb = w[i++];
                const auto known = std::find_if(kEventVerbs.begin(), kEventVerbs.end(),
                                                [&](const auto& v) { return v.first == verb; });
                if (known == kEventVerbs.end()) fail(path, line_no, "unknown event '" + verb + "'");
                std::vector<std::string> args(w.begin() + static_cast<long>(i), w.end());
                if (args.size() < known->second) fail(path, line_no, verb + ": missing arguments");
                if (period == 0) {
                    sc.events.push_back(Event{first, verb, args, line_no});
                } else {
                    repeats.push_back(Repeat{first, period, until, Event{0, verb, args, line_no}});
                }
            } else {
                fail(path, line_no, "unknown command '" + key + "'");
            }
        } catch (const std::invalid_argument&) {
            fail(path, line_no, "bad time/number in '" + raw + "'");
        }
    }
    // Expand repeats only now: "duration" may come after them in the file.
    for (const auto& r : repeats) {
        for (uint64_t t = r.first; t <= std::min(r.until, sc.duration_us); t += r.period) {
            Event e = r.event;
            e.at_us = t;
            sc.events.push_back(e);
        }
    }
    sc.events.erase(std::remove_if(sc.events.begin(), sc.events.end(),
                                   [&](const Event& e) { return e.at_us > sc.duration_us; }),
                    sc.events.end());
    std::stable_sort(sc.events.begin(), sc.events.end(),
                     [](const Event& a, const Event& b) { return a.at_us < b.at_us; });
    return sc;
}

}  // namespace sim
