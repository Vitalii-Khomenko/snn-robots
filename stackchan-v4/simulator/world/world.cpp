#include "world.h"

#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <fstream>

#include "../core/log.h"

namespace sim {
namespace {

constexpr float kPi = 3.14159265f;
constexpr float kDeg = kPi / 180.0f;
constexpr double kMicRate = 24000.0;

// --- Estimated physical constants (see README "What is estimated") --------
constexpr float kServoMaxSpeed = 500.0f;     // deg/s, SCS-class servo, no load
constexpr float kServoResponse = 0.015f;     // s, velocity smoothing
constexpr float kServoMovingDegS = 2.0f;     // below this the servo is "still"
// Estimated burst sigma, updated from the 2026-09-26 behavior-fix capture:
// repeated own-motion events at 8-11k, self_noise_peak around 12.7k.
// The old 1-4k model hid this feedback from simulator tests.
constexpr float kClickStartLevel = 3500.0f;
constexpr float kClickStopLevel = 3000.0f;
constexpr float kWhinePerDegS = 5.0f;        // mic amplitude per deg/s of head speed
constexpr float kSpeakerCoupling = 0.55f;    // speaker -> mic, both in the head
constexpr float kClapLevel = 14000.0f;
constexpr float kTalkLevel = 2600.0f;        // at 1 m, peak
constexpr float kIldDepth = 0.35f;           // interaural level difference
constexpr float kImuArmM = 0.04f;            // IMU distance from the rotation axes
constexpr float kCameraTargetMean = 118.0f;  // AEC target before the lock
constexpr float kSensorNoise = 1.8f;         // 8-bit levels, per pixel

float wrap180(float a)
{
    while (a > 180.0f) a -= 360.0f;
    while (a < -180.0f) a += 360.0f;
    return a;
}

float smooth01(float x) { x = std::clamp(x, 0.0f, 1.0f); return x * x * (3.0f - 2.0f * x); }

// Deterministic value noise on a periodic (in azimuth) lattice.
struct ValueNoise {
    int cells_x, cells_y;
    std::vector<float> v;
    ValueNoise(int cx, int cy, std::mt19937& rng) : cells_x(cx), cells_y(cy), v(static_cast<size_t>(cx * (cy + 1)))
    {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (auto& x : v) x = u(rng);
    }
    float at(float fx, float fy) const  // fx in [0,1) periodic, fy in [0,1]
    {
        const float x = fx * cells_x, y = fy * cells_y;
        const int x0 = static_cast<int>(std::floor(x)), y0 = std::min(static_cast<int>(std::floor(y)), cells_y - 1);
        const float tx = smooth01(x - x0), ty = smooth01(y - y0);
        auto g = [&](int ix, int iy) { return v[static_cast<size_t>(((ix % cells_x + cells_x) % cells_x) + iy * cells_x)]; };
        const float a = g(x0, y0) + (g(x0 + 1, y0) - g(x0, y0)) * tx;
        const float b = g(x0, y0 + 1) + (g(x0 + 1, y0 + 1) - g(x0, y0 + 1)) * tx;
        return a + (b - a) * ty;
    }
};

}  // namespace

World& World::get()
{
    static World world;
    return world;
}

void World::load(const Scenario& scenario)
{
    sc_ = scenario;
    rng_.seed(sc_.seed);
    lamps_ = sc_.lamps;
    people_.clear();
    things_.clear();
    sounds_.clear();
    extra_noise_ = extra_tilt_ = 0.0f;
    room_light_ = sc_.room_light;
    owner_home_ = sc_.owner_home;
    charging_ = sc_.charging;
    battery_pct_ = sc_.battery_pct;

    // Room texture: walls/furniture as layered value noise plus a few
    // rectangular objects, so the camera sees edges (optic flow works) and
    // turning the head changes the image, like a real room.
    std::mt19937 tex_rng(sc_.seed * 7919u + 17u);
    ValueNoise coarse(24, 6, tex_rng), mid(90, 20, tex_rng), fine(360, 60, tex_rng);
    texture_.assign(static_cast<size_t>(kTexW * kTexH), 0);
    for (int y = 0; y < kTexH; ++y) {
        for (int x = 0; x < kTexW; ++x) {
            const float fx = static_cast<float>(x) / kTexW, fy = static_cast<float>(y) / kTexH;
            float r = 0.18f + 0.35f * coarse.at(fx, fy) + 0.2f * mid.at(fx, fy) + 0.08f * fine.at(fx, fy);
            const float el = 60.0f - y * 0.5f;
            if (el < -25.0f) r *= 0.75f;  // floor/desk is darker
            texture_[static_cast<size_t>(y * kTexW + x)] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f);
        }
    }
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int k = 0; k < 14; ++k) {  // shelves, frames, doors
        const int w = 20 + static_cast<int>(u(tex_rng) * 70), h = 16 + static_cast<int>(u(tex_rng) * 90);
        const int x0 = static_cast<int>(u(tex_rng) * kTexW), y0 = static_cast<int>(u(tex_rng) * (kTexH - h));
        const float refl = 0.1f + 0.7f * u(tex_rng);
        for (int y = y0; y < y0 + h; ++y)
            for (int x = x0; x < x0 + w; ++x)
                texture_[static_cast<size_t>(y * kTexW + x % kTexW)] = static_cast<uint8_t>(refl * 255.0f);
    }
}

float World::hourOfDay(uint64_t t_us) const
{
    const double minutes = std::fmod(sc_.clock_min + static_cast<double>(t_us) / 60e6, 1440.0);
    return static_cast<float>(minutes / 60.0);
}

float World::daylight(uint64_t t_us) const
{
    const float m = hourOfDay(t_us) * 60.0f;
    const float rise = smooth01((m - (sc_.sunrise_min - 30.0f)) / 120.0f);
    const float set = 1.0f - smooth01((m - (sc_.sunset_min - 60.0f)) / 90.0f);
    return std::min(rise, set);
}

float World::ambient(uint64_t t_us) const
{
    float a = 0.02f;  // standby LEDs, street light
    a += daylight(t_us) * (sc_.window.present ? 0.45f : 0.15f);
    if (room_light_) a += 0.55f;
    for (const auto& lamp : lamps_)
        if (lamp.on) a += 0.03f * lamp.brightness / 6.0f;
    return a;
}

float World::reflectance(float az, float el) const
{
    float a = std::fmod(az, 360.0f);
    if (a < 0) a += 360.0f;
    const int x = std::min(kTexW - 1, static_cast<int>(a * 2.0f));
    const int y = std::clamp(static_cast<int>((60.0f - el) * 2.0f), 0, kTexH - 1);
    return texture_[static_cast<size_t>(y * kTexW + x)] / 255.0f;
}

float World::relAz(float az) const { return wrap180(az - yaw_deg_); }

float World::personAz(const Person& p, uint64_t t_us) const
{
    if (t_us >= p.walk_end || p.walk_end == p.walk_start) return p.az;
    if (t_us <= p.walk_start) return p.walk_from;
    const float f = static_cast<float>(t_us - p.walk_start) / static_cast<float>(p.walk_end - p.walk_start);
    return p.walk_from + wrap180(p.walk_to - p.walk_from) * f;
}

float World::personDist(const Person& p, uint64_t t_us) const
{
    if (t_us >= p.dist_end || p.dist_end == p.dist_start) return p.dist;
    if (t_us <= p.dist_start) return p.dist_from;
    const float f = static_cast<float>(t_us - p.dist_start) / static_cast<float>(p.dist_end - p.dist_start);
    return p.dist_from + (p.dist - p.dist_from) * f;
}

float World::radiance(float az, float el, uint64_t t_us) const
{
    float value = reflectance(az, el);
    // People occlude the background.
    for (const auto& p : people_) {
        if (!p.present) continue;
        const float dist = personDist(p, t_us);
        const float half_w = std::atan(0.22f / dist) / kDeg;
        const float top = std::atan((1.65f - 0.3f) / dist) / kDeg;
        const float d = wrap180(az - personAz(p, t_us));
        if (std::fabs(d) < half_w && el < top && el > -35.0f) {
            value = 0.32f + 0.12f * std::sin((el - top) * 0.35f) + (el > top - 12.0f ? 0.15f : 0.0f);
        }
    }
    for (const auto& th : things_) {
        if (!th.present) continue;
        const float dx = wrap180(az - thingAz(th, t_us)) * std::cos(el * kDeg);
        const float dy = el - thingEl(th, t_us);
        if (dx * dx + dy * dy < th.size_deg * th.size_deg * 0.25f) value = th.reflect;
    }
    value *= ambient(t_us);
    if (sc_.window.present) {
        const auto& w = sc_.window;
        if (std::fabs(wrap180(az - w.az_deg)) < w.width_deg / 2 && std::fabs(el - w.el_deg) < w.height_deg / 2) {
            // Mullions keep the window from being one flat blob.
            const bool frame = std::fabs(wrap180(az - w.az_deg)) < 0.8f || std::fabs(el - w.el_deg) < 0.8f;
            value = frame ? 0.08f * ambient(t_us) : 0.03f + w.gain * daylight(t_us);
        }
    }
    for (const auto& lamp : lamps_) {
        if (!lamp.on) continue;
        const float dx = wrap180(az - lamp.az_deg) * std::cos(el * kDeg);
        const float dy = el - lamp.el_deg;
        const float d = std::sqrt(dx * dx + dy * dy);
        if (d < lamp.radius_deg) value = std::max(value, lamp.brightness);
        else if (d < 3.0f * lamp.radius_deg)
            value += lamp.brightness * 0.12f * (1.0f - (d - lamp.radius_deg) / (2.0f * lamp.radius_deg));
    }
    return value;
}

void World::renderFrame(uint8_t* rgb565, int width, int height, uint64_t t_us)
{
    advanceTo(t_us);
    const float half_x = sc_.fov_deg / 2.0f;
    const float half_y = half_x * static_cast<float>(height) / static_cast<float>(width);
    static std::vector<float> rad;
    rad.resize(static_cast<size_t>(width * height));
    double sum = 0.0;
    for (int y = 0; y < height; ++y) {
        const float el = pitch_deg_ - ((y + 0.5f) / height * 2.0f - 1.0f) * half_y;
        for (int x = 0; x < width; ++x) {
            const float az = yaw_deg_ + ((x + 0.5f) / width * 2.0f - 1.0f) * half_x;
            const float r = radiance(az, el, t_us);
            rad[static_cast<size_t>(y * width + x)] = r;
            sum += std::min(r, 1.0f);
        }
    }
    float gain = exposure_gain_;
    if (!exposure_locked_) {
        const float mean = static_cast<float>(sum / (width * height));
        gain = std::clamp(kCameraTargetMean / std::max(mean * 255.0f, 1e-3f), 0.3f, 60.0f);
    }
    for (int i = 0; i < width * height; ++i) {
        const float v = rad[static_cast<size_t>(i)] * 255.0f * gain + gauss_(rng_) * kSensorNoise;
        const int level = std::clamp(static_cast<int>(v + 0.5f), 0, 255);
        const uint16_t pixel = static_cast<uint16_t>(((level >> 3) << 11) | ((level >> 2) << 5) | (level >> 3));
        rgb565[i * 2] = static_cast<uint8_t>(pixel >> 8);
        rgb565[i * 2 + 1] = static_cast<uint8_t>(pixel & 0xff);
    }
}

void World::lockExposure(uint64_t t_us)
{
    static std::vector<uint8_t> frame(160 * 120 * 2);
    exposure_locked_ = false;
    renderFrame(frame.data(), 160, 120, t_us);
    double sum = 0.0;
    const float half_x = sc_.fov_deg / 2.0f, half_y = half_x * 0.75f;
    for (int y = 0; y < 120; ++y)
        for (int x = 0; x < 160; ++x)
            sum += std::min(radiance(yaw_deg_ + ((x + 0.5f) / 160 * 2 - 1) * half_x,
                                     pitch_deg_ - ((y + 0.5f) / 120 * 2 - 1) * half_y, t_us), 1.0f);
    const float mean = static_cast<float>(sum / (160 * 120));
    exposure_gain_ = std::clamp(kCameraTargetMean / std::max(mean * 255.0f, 1e-3f), 0.3f, 60.0f);
    exposure_locked_ = true;
    note("camera exposure locked: scene mean %.3f, gain %.1f", mean, exposure_gain_);
}

bool World::lampInView(size_t i, float& x, float& y) const
{
    if (i >= lamps_.size() || !lamps_[i].on) return false;
    const float half_x = sc_.fov_deg / 2.0f, half_y = half_x * 0.75f;
    x = relAz(lamps_[i].az_deg) / half_x;
    y = -(lamps_[i].el_deg - pitch_deg_) / half_y;
    return std::fabs(x) <= 1.0f && std::fabs(y) <= 1.0f;
}

void World::speakerPush(uint64_t t_us, const int16_t* samples, size_t count, int volume)
{
    SpeakerChunk chunk;
    chunk.start_us = t_us;
    chunk.samples.assign(samples, samples + count);
    chunk.gain = kSpeakerCoupling * static_cast<float>(volume) / 100.0f;
    speaker_.push_back(std::move(chunk));
    while (!speaker_.empty() && speaker_.front().start_us + 2000000 < t_us) speaker_.pop_front();
}

void World::micRead(int16_t* stereo, size_t frames, uint64_t t_end_us)
{
    advanceTo(t_end_us);
    const double t0 = static_cast<double>(t_end_us) - frames * 1e6 / kMicRate;
    const float whine = kWhinePerDegS * headSpeedDegPerS();
    const float noise = sc_.noise_floor;
    for (size_t k = 0; k < frames; ++k) {
        const double t_us = t0 + k * 1e6 / kMicRate;
        const double ts = t_us / 1e6;
        float common = 0.0f, left = 0.0f, right = 0.0f;
        // The robot itself: servo whine and start/stop clicks, speaker bleed.
        if (whine > 0.0f) {
            common += whine * (0.6f * std::sin(2 * kPi * 2300.0 * ts) + 0.4f * std::sin(2 * kPi * 3400.0 * ts)) +
                      whine * 0.3f * gauss_(rng_);
        }
        for (const auto& c : clicks_) {
            const double age = (t_us - static_cast<double>(c.start)) / 1e6;
            if (age >= 0 && age < 0.015) common += c.level * static_cast<float>(std::exp(-age / 0.003)) * gauss_(rng_);
        }
        for (const auto& chunk : speaker_) {
            if (t_us < chunk.start_us) continue;
            const size_t idx = static_cast<size_t>((t_us - chunk.start_us) * kMicRate / 1e6);
            if (idx < chunk.samples.size()) common += chunk.samples[idx] * chunk.gain;
        }
        // The room: scenario sounds from a direction.
        for (const auto& s : sounds_) {
            if (t_us < s.start || t_us >= s.end) continue;
            const double age = (t_us - s.start) / 1e6;
            const float ramp = s.ramp ? 0.1f + 0.9f * static_cast<float>((t_us - s.start) / static_cast<double>(s.end - s.start)) : 1.0f;
            float v = 0.0f;
            const double total = (s.end - s.start) / 1e6;
            const float env = static_cast<float>(std::min({1.0, age / 0.01, (total - age) / 0.03 + 0.0}));
            if (s.kind == Sound::Clap) {
                v = kClapLevel * s.level * static_cast<float>(std::exp(-age / 0.006)) * gauss_(rng_);
            } else if (s.kind == Sound::Bang) {  // a door slam / thud: low, ringing
                v = 11000.0f * s.level * static_cast<float>(std::exp(-age / 0.05)) *
                    static_cast<float>(0.6 * std::sin(2 * kPi * 70.0 * ts) + 0.4 * std::sin(2 * kPi * 130.0 * ts + 1.0)) +
                    1500.0f * s.level * static_cast<float>(std::exp(-age / 0.01)) * gauss_(rng_);
            } else if (s.kind == Sound::Hiss) {  // a burst of high-passed noise (steam, spray)
                const float n = gauss_(rng_);
                v = 1800.0f * s.level * std::max(0.0f, env) * (n - hiss_prev_) * 0.7f;
                hiss_prev_ = n;
            } else if (s.kind == Sound::Clip) {
                const size_t idx = static_cast<size_t>(age * kMicRate);
                if (idx < s.clip->size()) v = static_cast<float>((*s.clip)[idx]) * s.gain;
            } else if (s.kind == Sound::Tone || s.kind == Sound::Whistle) {
                const double f = s.kind == Sound::Tone ? s.hz0 : s.hz0 + (s.hz1 - s.hz0) * (age / std::max(total, 1e-3));
                if (s.kind == Sound::Whistle) tone_phase_ += 2 * kPi * f / kMicRate;
                const double ph = s.kind == Sound::Tone ? 2 * kPi * f * ts : tone_phase_;
                v = 2600.0f * s.level * std::max(0.0f, env) * static_cast<float>(std::sin(ph));
            } else {
                const double f0 = 120.0 + (s.seed % 60);
                const float syllable = static_cast<float>(std::max(0.0, std::sin(kPi * 3.6 * age)));
                double voiced = 0.0;
                for (int h = 1; h <= 8; ++h) voiced += std::sin(2 * kPi * f0 * h * ts + h) / h;
                v = kTalkLevel * s.level * std::sqrt(syllable) * static_cast<float>(voiced / 2.7) +
                    40.0f * syllable * gauss_(rng_);
            }
            v *= ramp;
            const float rel = relAz(s.az) * kDeg;
            left += v * (1.0f - kIldDepth * std::sin(rel));
            right += v * (1.0f + kIldDepth * std::sin(rel));
        }
        float bgl = 0.0f, bgr = 0.0f;
        if (extra_noise_ > 0.0f) {
            const float a = extra_tilt_ * 0.95f, comp = std::sqrt((1.0f + a) / (1.0f - a));
            bg_lp_[0] = a * bg_lp_[0] + (1.0f - a) * gauss_(rng_);
            bg_lp_[1] = a * bg_lp_[1] + (1.0f - a) * gauss_(rng_);
            bgl = extra_noise_ * comp * bg_lp_[0];
            bgr = extra_noise_ * comp * bg_lp_[1];
        }
        const float l = common + left + noise * gauss_(rng_) + bgl;
        const float r = common + right + noise * gauss_(rng_) + bgr;
        stereo[k * 2] = static_cast<int16_t>(std::clamp(l, -32768.0f, 32767.0f));
        stereo[k * 2 + 1] = static_cast<int16_t>(std::clamp(r, -32768.0f, 32767.0f));
    }
}

void World::imuAccel(float& x, float& y, float& z)
{
    const float g = 9.81f;
    // SIM_IMU_V3=1: the old mounting the v3 policy was trained with (head pitch on x) - the v3 behaviour tests use it.
    static const bool v3_mounting = std::getenv("SIM_IMU_V3") != nullptr;
    if (v3_mounting) {
        const float pitch = (pitch_deg_ + body_pitch_deg_) * kDeg;
        const float roll = body_roll_deg_ * kDeg;
        x = -g * std::sin(pitch);
        y = g * std::sin(roll) * std::cos(pitch);
        z = g * std::cos(pitch) * std::cos(roll);
        x += pitch_accel_ * kDeg * kImuArmM;
        y += yaw_accel_ * kDeg * kImuArmM;
    } else {
        // Mounted as on the real robot (diary 2026-10-19..24): the head's pitch turns gravity in the y-z plane (the
        // "roll" BrainLoop reads: 0.44 at pitch 0, 0.25 at +35, 0.55 at -20), a sideways lean shows on x.
        const float a = (80.0f - (pitch_deg_ + body_pitch_deg_)) * kDeg;
        const float side = body_roll_deg_ * kDeg;
        x = -g * std::sin(side);
        y = g * std::sin(a) * std::cos(side);
        z = g * std::cos(a) * std::cos(side);
        y += pitch_accel_ * kDeg * kImuArmM;
        x += yaw_accel_ * kDeg * kImuArmM;
    }
    const float vib = std::min(1.0f, headSpeedDegPerS() / 60.0f) * 0.35f;
    x += vib * gauss_(rng_);
    y += vib * gauss_(rng_);
    z += vib * gauss_(rng_);
    if (now_us_ < shake_until_) {
        const float ts = static_cast<float>(now_us_) / 1e6f;
        x += 14.0f * std::sin(2 * kPi * 4.5f * ts);
        y += 11.0f * std::sin(2 * kPi * 4.5f * ts + 1.3f);
    }
    x += 0.04f * gauss_(rng_);
    y += 0.04f * gauss_(rng_);
    z += 0.04f * gauss_(rng_);
}

uint8_t World::headTouchRaw() const
{
    // Channel 1 (bits 2..3) at level 3 while a hand is on the head.
    return now_us_ < touch_until_ ? static_cast<uint8_t>(3 << 2) : 0;
}

bool World::takeScreenTap()
{
    if (tap_times_.empty() || tap_times_.front() > now_us_) return false;
    tap_times_.pop_front();
    return true;
}

uint16_t World::alsRaw() const
{
    // Light reaching the sensor on the face: ambient plus sources the face points at.
    float lux = ambient(now_us_) * 300.0f;
    auto facing = [&](float az, float el) {
        const float c = std::cos(relAz(az) * kDeg) * std::cos((el - pitch_deg_) * kDeg);
        return c > 0 ? c * c : 0.0f;
    };
    for (const auto& lamp : lamps_)
        if (lamp.on) lux += lamp.brightness * 25.0f * facing(lamp.az_deg, lamp.el_deg);
    if (sc_.window.present) lux += sc_.window.gain * daylight(now_us_) * 120.0f * facing(sc_.window.az_deg, sc_.window.el_deg);
    // counts as the robot's LTR-553 reads them (2026-10-28, its diary 10-26/27): a lit room by day 2-25 (median ~8), a
    // dark night 0. It was lux x 20 - a dark room read 120 counts (light channel 0.44, the robot's night is 0) and a
    // lit one 3400 (0.82, near the channel's anger zone; the robot's day is 0.1-0.33).
    return static_cast<uint16_t>(std::clamp(lux * 0.05f, 0.0f, 65535.0f));
}

uint16_t World::psRaw()
{
    float counts = 35.0f + 2.0f * gauss_(rng_);
    if (now_us_ < near_until_) counts += 420.0f;
    for (const auto& p : people_)
        if (p.present && personDist(p, now_us_) < 0.45f && std::fabs(relAz(personAz(p, now_us_))) < 30.0f)
            counts += 400.0f * (0.45f - personDist(p, now_us_)) / 0.45f;
    return static_cast<uint16_t>(std::clamp(counts, 0.0f, 2047.0f));
}

void World::servoPower(bool on)
{
    if (on && !servo_power_ && hang_p_ > 0.0f) {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        servos_hung_ = u(rng_) < hang_p_;
        if (servos_hung_) note("servos hung at the switch-on");
    }
    if (!on) servos_hung_ = false;
    servo_power_ = on;
}

void World::servoWrite(uint8_t id, int raw, uint16_t goal_ms)
{
    if (!servo_power_ || servos_hung_) return;  // limp, or hung: the packet is lost
    // Same calibration as MotionEngine: zero 460 (yaw) / 620 (pitch), 0.3125 deg/unit.
    if (id == 1) {
        yaw_target_ = (raw - 460) * 0.3125f;
        yaw_goal_ms_ = goal_ms;
    } else if (id == 2) {
        pitch_target_ = (raw - 620) * 0.3125f;
        pitch_goal_ms_ = goal_ms;
    }
}

void World::stepPhysics(double dt_s)
{
    const float dt = static_cast<float>(dt_s);
    auto axis = [&](float& pos, float& vel, float target, uint16_t goal_ms) {
        const float goal_s = std::max(goal_ms / 1000.0f, dt);
        const float desired = std::clamp((target - pos) / goal_s, -kServoMaxSpeed, kServoMaxSpeed);
        vel += (desired - vel) * std::min(1.0f, dt / kServoResponse);
        const float step = vel * dt;
        pos += step;
        travel_deg_ += std::fabs(step);
    };
    prev_yaw_vel_ = yaw_vel_;
    prev_pitch_vel_ = pitch_vel_;
    axis(yaw_deg_, yaw_vel_, yaw_target_, yaw_goal_ms_);
    axis(pitch_deg_, pitch_vel_, pitch_target_, pitch_goal_ms_);
    yaw_accel_ = (yaw_vel_ - prev_yaw_vel_) / dt;
    pitch_accel_ = (pitch_vel_ - prev_pitch_vel_) / dt;

    // Every motor start/stop clicks loud enough to cross the microphone's
    // loud threshold - including the 1-raw-unit steps of a slow spring
    // creep. Confirmed on hardware 2026-09-26: with the microphone no longer
    // muted during the creep, "loud sound: centre" fired every ~3 s (the
    // detector's cooldown) while the head moved.
    const bool moving = headSpeedDegPerS() > kServoMovingDegS;
    if (moving && !was_moving_) {
        clicks_.push_back(Pulse{now_us_, kClickStartLevel});
        move_peak_speed_ = 0.0f;
    } else if (!moving && was_moving_) {
        clicks_.push_back(Pulse{now_us_, kClickStopLevel});
        if (move_peak_speed_ > 10.0f) ++movements_;
    }
    if (moving) move_peak_speed_ = std::max(move_peak_speed_, headSpeedDegPerS());
    was_moving_ = moving;
    while (!clicks_.empty() && clicks_.front().start + 50000 < now_us_) clicks_.pop_front();

    // The body is lifted/tilted by a person, smoothly.
    body_pitch_deg_ += (body_pitch_target_ - body_pitch_deg_) * std::min(1.0f, dt / 0.4f);
    body_roll_deg_ += (body_roll_target_ - body_roll_deg_) * std::min(1.0f, dt / 0.4f);
}

void World::robotActed(int kind)
{
    if (!responder_.on || responder_.random || kind < 0 || kind > 2 || !responder_.acts[kind]) return;
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    if (u(rng_) >= responder_.p) return;
    const float lat = responder_.lat_lo + (responder_.lat_hi - responder_.lat_lo) * u(rng_);
    pending_answers_.emplace_back(now_us_ + static_cast<uint64_t>(lat * 1e6), responder_.answer);
}

void World::answer(const std::string& what)
{
    Event e;
    e.verb = what == "tap" ? "tap_screen" : what;
    if (what == "pet") e.args = {"1200"};
    if (what == "talk") e.args = {"0", "1.5", "1"};
    ++answers_given_;
    note("responder: %s", what.c_str());
    applyEvent(e);
}

void World::advanceTo(uint64_t t_us)
{
    constexpr uint64_t kStep = 1000;  // 1 ms physics step
    while (now_us_ < t_us) {
        while (next_event_ < sc_.events.size() && sc_.events[next_event_].at_us <= now_us_) {
            applyEvent(sc_.events[next_event_++]);
        }
        for (size_t i = 0; i < pending_answers_.size();) {
            if (pending_answers_[i].first <= now_us_) {
                const std::string what = pending_answers_[i].second;
                pending_answers_.erase(pending_answers_.begin() + static_cast<long>(i));
                answer(what);
            } else {
                ++i;
            }
        }
        if (responder_.on && responder_.random && now_us_ >= responder_.next_random_us) {
            if (responder_.next_random_us) answer(responder_.answer);
            std::exponential_distribution<float> gap(1.0f / responder_.mean_s);
            responder_.next_random_us = now_us_ + static_cast<uint64_t>(gap(rng_) * 1e6);
        }
        const uint64_t step = std::min(kStep, t_us - now_us_);
        now_us_ += step;
        stepPhysics(step / 1e6);
    }
    while (next_event_ < sc_.events.size() && sc_.events[next_event_].at_us <= now_us_) {
        applyEvent(sc_.events[next_event_++]);
    }
    while (!sounds_.empty() && sounds_.front().end + 1000000 < now_us_) sounds_.erase(sounds_.begin());
}

World::Person* World::person(const std::string& name)
{
    for (auto& p : people_)
        if (p.name == name) return &p;
    people_.push_back(Person{});
    people_.back().name = name;
    return &people_.back();
}

void World::applyEvent(const Event& e)
{
    const auto& a = e.args;
    auto argf = [&](size_t i, float def) { return i < a.size() ? std::stof(a[i]) : def; };
    const std::string& v = e.verb;
    if (v == "responder") {
        // responder off | responder random ANSWER MEAN_S | responder ACTS ANSWER P LAT_LO LAT_HI (ACTS: call,gesture,face)
        responder_ = Responder{};
        pending_answers_.clear();
        if (!a.empty() && a[0] != "off") {
            responder_.on = true;
            responder_.answer = a.size() > 1 ? a[1] : "pet";
            if (a[0] == "random") {
                responder_.random = true;
                responder_.mean_s = argf(2, 20.0f);
            } else {
                responder_.acts[0] = a[0].find("call") != std::string::npos;
                responder_.acts[1] = a[0].find("gesture") != std::string::npos;
                responder_.acts[2] = a[0].find("face") != std::string::npos;
                responder_.p = argf(2, 0.8f);
                responder_.lat_lo = argf(3, 0.8f);
                responder_.lat_hi = argf(4, 2.0f);
            }
        }
        note("responder %s", a.empty() ? "off" : a[0].c_str());
    } else if (v == "fault") {
        // fault imu|light|ntp on|off;  fault servo P (the chance of a hang at each rail switch-on; 0 / off: none)
        const bool on = a.size() < 2 || a[1] != "off";
        const std::string what = a.empty() ? "" : a[0];
        if (what == "servo") hang_p_ = on && a.size() > 1 ? std::clamp(std::stof(a[1]), 0.0f, 1.0f) : 0.0f;
        else if (what == "imu") fault_imu_ = on;
        else if (what == "light") fault_light_ = on;
        else if (what == "ntp") fault_ntp_ = on;
        note("fault %s %s", what.c_str(), on ? "on" : "off");
    } else if (v == "touch_head") {
        touch_until_ = now_us_ + static_cast<uint64_t>(argf(0, 400) * 1000);
        note("touch_head %.0f ms", argf(0, 400));
    } else if (v == "pet") {
        const uint64_t ms = static_cast<uint64_t>(argf(0, 1500));
        near_until_ = now_us_ + ms * 1000;
        touch_until_ = now_us_ + 300000 + ms * 1000;
        note("pet: hand near the face and on the head for %llu ms", static_cast<unsigned long long>(ms));
    } else if (v == "tap_screen" || v == "double_tap_screen" || v == "triple_tap_screen") {
        const int n = v == "tap_screen" ? 1 : v == "double_tap_screen" ? 2 : 3;
        for (int i = 0; i < n; ++i) tap_times_.push_back(now_us_ + static_cast<uint64_t>(i) * 250000);
        note("%s", v.c_str());
    } else if (v == "shake") {
        shake_until_ = now_us_ + static_cast<uint64_t>(argf(0, 2) * 1e6);
        note("shake %.1f s", argf(0, 2));
    } else if (v == "pick_up") {
        // pick_up [PITCH [ROLL]]: held tilted forward / back, and sideways
        body_pitch_target_ = argf(0, 25);
        body_roll_target_ = argf(1, 0);
        note("pick_up: body tilted %.0f deg, %.0f sideways", body_pitch_target_, body_roll_target_);
    } else if (v == "put_down") {
        body_pitch_target_ = 0;
        body_roll_target_ = 0;
        note("put_down");
    } else if (v == "clap") {
        sounds_.push_back(Sound{Sound::Clap, now_us_, now_us_ + 40000, argf(0, 0), argf(1, 1), 0});
        note("clap from %.0f deg", argf(0, 0));
    } else if (v == "talk") {
        const float seconds = argf(1, 3);
        sounds_.push_back(Sound{Sound::Talk, now_us_, now_us_ + static_cast<uint64_t>(seconds * 1e6), argf(0, 0),
                                argf(2, 1), static_cast<uint32_t>(rng_())});
        note("talk from %.0f deg for %.1f s", argf(0, 0), seconds);
    } else if (v == "talk_ramp") {
        const float seconds = argf(1, 4);
        Sound s{Sound::Talk, now_us_, now_us_ + static_cast<uint64_t>(seconds * 1e6), argf(0, 0), argf(2, 1),
                static_cast<uint32_t>(rng_())};
        s.ramp = true;
        sounds_.push_back(s);
        note("talk coming closer from %.0f deg over %.1f s", argf(0, 0), seconds);
    } else if (v == "bang") {
        sounds_.push_back(Sound{Sound::Bang, now_us_, now_us_ + 300000, argf(0, 0), argf(1, 1), 0});
        note("bang from %.0f deg", argf(0, 0));
    } else if (v == "hiss") {
        sounds_.push_back(Sound{Sound::Hiss, now_us_, now_us_ + static_cast<uint64_t>(argf(1, 0.4f) * 1e6), argf(0, 0),
                                argf(2, 1), 0});
        note("hiss from %.0f deg", argf(0, 0));
    } else if (v == "tone" || v == "whistle") {
        Sound s{v == "tone" ? Sound::Tone : Sound::Whistle, now_us_,
                now_us_ + static_cast<uint64_t>(argf(v == "tone" ? 2 : 3, 1.0f) * 1e6), argf(0, 0), 1.0f, 0};
        s.hz0 = argf(1, 1000);
        s.hz1 = v == "tone" ? s.hz0 : argf(2, 2000);
        s.level = argf(v == "tone" ? 3 : 4, 1.0f);
        sounds_.push_back(s);
        note("%s from %.0f deg at %.0f Hz", v.c_str(), s.az, s.hz0);
    } else if (v == "clip") {
        // clip AZ FILE LEVEL CLASSES   (classes: speech+clap+sharp+continuous+low+high+tonal)
        Sound s{Sound::Clip, now_us_, now_us_, argf(0, 0), argf(2, 1.0f), 0};
        if (a.size() > 1 && loadClip(a[1], s)) {
            uint8_t m = 0;
            const std::string cls = a.size() > 3 ? a[3] : "";
            auto has = [&](const char* w) { return cls.find(w) != std::string::npos; };
            if (has("speech")) m |= kClipSpeech;
            if (has("clap")) m |= kClipClap;
            if (has("sharp")) m |= kClipSharp;
            if (has("continuous")) m |= kClipContinuous;
            if (has("low")) m |= kClipLow;
            if (has("high")) m |= kClipHigh;
            if (has("tonal")) m |= kClipTonal;
            s.mask = m;
            s.end = now_us_ + static_cast<uint64_t>(s.clip->size() * 1e6 / kMicRate);
            sounds_.push_back(s);
            note("clip %s from %.0f deg (%s)", a[1].c_str(), s.az, cls.c_str());
        }
    } else if (v == "thing") {
        Thing* th = nullptr;
        for (auto& t : things_)
            if (t.name == a[0]) th = &t;
        if (!th) {
            things_.push_back(Thing{});
            th = &things_.back();
            th->name = a[0];
        }
        const std::string& act = a[1];
        if (act == "enter") {
            th->present = true;
            th->az = th->az_from = argf(2, 0);
            th->el = th->el_from = argf(3, 0);
            th->size_deg = argf(4, 8);
            th->reflect = argf(5, 0.7f);
            th->move_start = th->move_end = 0;
        } else if (act == "move") {
            th->az_from = thingAz(*th, now_us_);
            th->el_from = thingEl(*th, now_us_);
            th->az = argf(2, 0);
            th->el = argf(3, 0);
            th->move_start = now_us_;
            th->move_end = now_us_ + static_cast<uint64_t>(argf(4, 2) * 1e6);
        } else if (act == "leave") {
            th->present = false;
        }
    } else if (v == "noise") {
        extra_noise_ = argf(0, 0);
        extra_tilt_ = std::clamp(argf(1, 0.0f), 0.0f, 1.0f);
        note("background noise level %.0f tilt %.2f", extra_noise_, extra_tilt_);
    } else if (v == "lamp") {
        for (auto& lamp : lamps_) {
            if (lamp.name != a[0]) continue;
            if (a[1] == "on" || a[1] == "off") lamp.on = a[1] == "on";
            else if (a[1] == "brightness" && a.size() > 2) lamp.brightness = std::stof(a[2]);
            note("lamp %s %s", lamp.name.c_str(), lamp.on ? "on" : "off");
        }
    } else if (v == "room_light") {
        room_light_ = a[0] == "on";
        note("room light %s", a[0].c_str());
    } else if (v == "person") {
        Person* p = person(a[0]);
        const std::string& act = a[1];
        if (act == "enter") {
            p->present = true;
            p->az = argf(2, 0);
            p->dist = argf(3, 1.5f);
            p->walk_start = p->walk_end = 0;
            note("person %s enters at %.0f deg, %.1f m", a[0].c_str(), p->az, p->dist);
        } else if (act == "walk") {
            p->walk_from = personAz(*p, now_us_);
            p->walk_to = argf(2, 0);
            p->walk_start = now_us_;
            p->walk_end = now_us_ + static_cast<uint64_t>(argf(3, 3) * 1e6);
            p->az = p->walk_to;
            note("person %s walks to %.0f deg", a[0].c_str(), p->walk_to);
        } else if (act == "come_close") {
            p->dist = argf(2, 0.3f);
            note("person %s comes close (%.2f m)", a[0].c_str(), p->dist);
        } else if (act == "approach") {
            // approach NAME DIST SECONDS: walk towards (or away from) the robot
            p->dist_from = personDist(*p, now_us_);
            p->dist = argf(2, 0.3f);
            p->dist_start = now_us_;
            p->dist_end = now_us_ + static_cast<uint64_t>(argf(3, 2) * 1e6);
            note("person %s approaches to %.2f m over %.1f s", a[0].c_str(), p->dist, argf(3, 2));
        } else if (act == "step_back") {
            p->dist = argf(2, 1.5f);
            note("person %s steps back (%.2f m)", a[0].c_str(), p->dist);
        } else if (act == "leave") {
            p->present = false;
            note("person %s leaves", a[0].c_str());
        }
    } else if (v == "owner") {
        owner_home_ = a[0] == "home";
        note("owner's phone %s", owner_home_ ? "at home" : "away");
    } else if (v == "charging") {
        charging_ = a[0] == "on";
        note("charging %s", a[0].c_str());
    } else if (v == "battery") {
        battery_pct_ = static_cast<int>(argf(0, 100));
        note("battery %d%%", battery_pct_);
    } else if (v == "log") {
        std::string text;
        for (const auto& w : a) text += (text.empty() ? "" : " ") + w;
        note("%s", text.c_str());
    }
}


float World::thingAz(const Thing& t, uint64_t t_us) const
{
    if (t_us >= t.move_end || t.move_end == t.move_start) return t.az;
    if (t_us <= t.move_start) return t.az_from;
    const float f = static_cast<float>(t_us - t.move_start) / static_cast<float>(t.move_end - t.move_start);
    return t.az_from + wrap180(t.az - t.az_from) * f;
}

float World::thingEl(const Thing& t, uint64_t t_us) const
{
    if (t_us >= t.move_end || t.move_end == t.move_start) return t.el;
    if (t_us <= t.move_start) return t.el_from;
    const float f = static_cast<float>(t_us - t.move_start) / static_cast<float>(t.move_end - t.move_start);
    return t.el_from + (t.el - t.el_from) * f;
}

void World::gyro(float& pitch_dps, float& yaw_dps)
{
    pitch_dps = pitch_vel_ + 0.4f * gauss_(rng_);
    yaw_dps = yaw_vel_ + 0.4f * gauss_(rng_);
}

bool World::loadClip(const std::string& path, Sound& s)
{
    auto it = clips_.find(path);
    if (it == clips_.end()) {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) return false;
        const size_t bytes = static_cast<size_t>(in.tellg());
        in.seekg(0);
        auto data = std::make_shared<std::vector<int16_t>>(bytes / 2);
        in.read(reinterpret_cast<char*>(data->data()), static_cast<std::streamsize>(data->size() * 2));
        // 50 ms slots; a slot is "audible" when its RMS is above 20 % of the loud ones'
        const size_t slot = 1200, n = data->size() / slot + 1;
        std::vector<float> rms(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            double e = 0;
            size_t cnt = 0;
            for (size_t k = i * slot; k < std::min(data->size(), (i + 1) * slot); ++k, ++cnt) e += double((*data)[k]) * (*data)[k];
            rms[i] = cnt ? static_cast<float>(std::sqrt(e / cnt)) : 0.0f;
        }
        std::vector<float> sorted = rms;
        std::sort(sorted.begin(), sorted.end());
        const float loud = sorted[static_cast<size_t>(0.9 * (n - 1))];
        auto active = std::make_shared<std::vector<uint8_t>>(n, 0);
        for (size_t i = 0; i < n; ++i) (*active)[i] = rms[i] > 0.2f * loud && rms[i] > 30.0f ? 1 : 0;
        // reference level: the RMS of the audible slots
        double e = 0;
        size_t cnt = 0;
        for (size_t i = 0; i < n; ++i)
            if ((*active)[i]) { e += double(rms[i]) * rms[i]; ++cnt; }
        const float ref = cnt ? static_cast<float>(std::sqrt(e / cnt)) : 1.0f;
        clip_ref_[path] = ref;
        it = clips_.emplace(path, std::make_pair(data, active)).first;
    }
    s.clip = it->second.first;
    s.active = it->second.second;
    // gain: audible RMS -> 2500 * level (a person talking at ~1 m), peak limited
    s.gain = 2500.0f * s.level / std::max(clip_ref_[path], 1.0f);
    return true;
}

void World::note(const char* fmt, ...)
{
    char buf[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    logLine('W', "world", buf);
}

}  // namespace sim
