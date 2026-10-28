// Host implementations of the platform APIs the firmware sources call.
// Each one forwards to the virtual RTOS (time, tasks) or to the physical
// world model (sensors, actuators), so the real firmware .cpp files run
// unmodified on top.

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <SCSCL.h>
#include <Si12T.h>
#include <bmi270.h>
#include <bsp/esp-bsp.h>
#include <driver/i2c_master.h>
#include <esp_camera.h>
#include <esp_codec_dev.h>
#include <esp_err.h>
#include <esp_log.h>
#include <esp_random.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "../../main/audio/audio_output.h"
#include "../../main/led/led_engine.h"
#include "../../main/ui/face_engine.h"
#include "../core/log.h"
#include "../core/vrtos.h"
#include "../world/world.h"
#include "platform.h"

using sim::World;

namespace {
std::mt19937 g_random{1};
World& world()
{
    World& w = World::get();
    w.advanceTo(vrtos::nowUs());
    return w;
}
}  // namespace

namespace sim {
void seedPlatformRandom(uint32_t seed) { g_random.seed(seed * 2654435761u + 1u); }
LedState g_led;
}  // namespace sim

// --- esp_timer / esp_random / esp_err / esp_log ----------------------------
int64_t esp_timer_get_time() { return static_cast<int64_t>(vrtos::nowUs()); }
uint32_t esp_random() { return static_cast<uint32_t>(g_random()); }

const char* esp_err_to_name(esp_err_t code)
{
    switch (code) {
        case ESP_OK: return "ESP_OK";
        case ESP_FAIL: return "ESP_FAIL";
        case ESP_ERR_TIMEOUT: return "ESP_ERR_TIMEOUT";
        default: return "ESP_ERR";
    }
}

void sim_esp_log(char level, const char* tag, const char* fmt, ...)
{
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    sim::logLine(level, tag, buf);
}

// --- FreeRTOS on vrtos --------------------------------------------------------
void vTaskDelay(TickType_t ticks)
{
    if (ticks == 0) {
        vrtos::sleepFor(0);  // yield
        return;
    }
    // Real FreeRTOS wakes on tick boundaries (10 ms here).
    constexpr uint64_t kTickUs = 1000000 / configTICK_RATE_HZ;
    const uint64_t now = vrtos::nowUs();
    vrtos::sleepUntil((now / kTickUs + ticks) * kTickUs);
}

BaseType_t xTaskCreate(TaskFunction_t fn, const char* name, uint32_t, void* arg, UBaseType_t, TaskHandle_t* out)
{
    if (out) *out = nullptr;
    return vrtos::spawn(name, [fn, arg] { fn(arg); }) ? pdPASS : pdFAIL;
}

BaseType_t xTaskCreatePinnedToCore(TaskFunction_t fn, const char* name, uint32_t stack, void* arg,
                                   UBaseType_t prio, TaskHandle_t* out, BaseType_t)
{
    return xTaskCreate(fn, name, stack, arg, prio, out);
}

UBaseType_t uxTaskGetStackHighWaterMark(TaskHandle_t) { return 0; }

// --- BSP -------------------------------------------------------------------
struct sim_codec { int kind; };
static sim_codec g_mic_codec{0};

esp_err_t bsp_i2c_init() { return ESP_OK; }
esp_codec_dev_handle_t bsp_audio_codec_microphone_init() { return &g_mic_codec; }

// --- ES7210 microphone codec: reads block for the audio they return ------
namespace {
double g_mic_cursor_us = -1.0;
}

int esp_codec_dev_open(esp_codec_dev_handle_t, esp_codec_dev_sample_info_t*) { return 0; }
int esp_codec_dev_set_in_gain(esp_codec_dev_handle_t, float) { return 0; }
int esp_codec_dev_set_out_vol(esp_codec_dev_handle_t, int) { return 0; }
int esp_codec_dev_write(esp_codec_dev_handle_t, void*, int) { return 0; }

int esp_codec_dev_read(esp_codec_dev_handle_t, void* data, int len)
{
    const size_t frames = static_cast<size_t>(len) / (2 * sizeof(int16_t));
    const double now = static_cast<double>(vrtos::nowUs());
    if (g_mic_cursor_us < now) g_mic_cursor_us = now;  // first read, or the task fell behind
    g_mic_cursor_us += frames * 1e6 / 24000.0;
    vrtos::sleepUntil(static_cast<uint64_t>(g_mic_cursor_us));
    World::get().micRead(static_cast<int16_t*>(data), frames, vrtos::nowUs());
    return 0;
}

// --- GC0308 camera (esp32-camera, fb_count 1, grab when empty) -----------
namespace {
constexpr uint64_t kCameraFrameUs = 143000;  // sensor's native ~7 fps at QQVGA
uint64_t g_capture_start_us = 0;
std::vector<uint8_t> g_frame_buf(160 * 120 * 2);
camera_fb_t g_fb{g_frame_buf.data(), g_frame_buf.size(), 160, 120, PIXFORMAT_RGB565};
int setExposureCtrl(sensor_t*, int enable)
{
    if (!enable) World::get().lockExposure(vrtos::nowUs());
    return 0;
}
int setNoop(sensor_t*, int) { return 0; }
sensor_t g_sensor{setExposureCtrl, setNoop, setNoop};
}  // namespace

esp_err_t esp_camera_init(const camera_config_t*)
{
    g_capture_start_us = vrtos::nowUs();
    return ESP_OK;
}
esp_err_t esp_camera_deinit() { return ESP_OK; }
sensor_t* esp_camera_sensor_get() { return &g_sensor; }

camera_fb_t* esp_camera_fb_get()
{
    // The driver starts capturing after the previous buffer is returned and
    // hands it over when the frame completes.
    const uint64_t done = g_capture_start_us + kCameraFrameUs;
    if (vrtos::nowUs() < done) vrtos::sleepUntil(done);
    World::get().renderFrame(g_frame_buf.data(), 160, 120, vrtos::nowUs());
    return &g_fb;
}

void esp_camera_fb_return(camera_fb_t*) { g_capture_start_us = vrtos::nowUs(); }

// --- I2C chips: LTR-553 (0x23) and AXP2101 (0x34) register models -----------
struct sim_i2c_bus { int port; };
struct sim_i2c_dev {
    uint16_t addr;
    uint16_t latch = 0;
};
static sim_i2c_bus g_bus{BSP_I2C_NUM};

esp_err_t i2c_master_get_bus_handle(int, i2c_master_bus_handle_t* out)
{
    *out = &g_bus;
    return ESP_OK;
}

esp_err_t i2c_master_bus_add_device(i2c_master_bus_handle_t, const i2c_device_config_t* cfg,
                                    i2c_master_dev_handle_t* out)
{
    if (cfg->device_address != 0x23 && cfg->device_address != 0x34) return ESP_ERR_NOT_FOUND;
    *out = new sim_i2c_dev{cfg->device_address};
    return ESP_OK;
}

esp_err_t i2c_master_transmit(i2c_master_dev_handle_t, const uint8_t*, size_t, int) { return ESP_OK; }

esp_err_t i2c_master_transmit_receive(i2c_master_dev_handle_t dev, const uint8_t* tx, size_t, uint8_t* rx,
                                      size_t rx_len, int)
{
    World& w = world();
    const uint8_t reg = tx[0];
    if (dev->addr == 0x23 && reg == 0x88 && rx_len == 4) {
        // the ALS data as one burst (PresenceEngine::readAlsData, 2026-10-24): CH1 (IR) lo/hi, CH0 lo/hi; the room's
        // IR share taken as a quarter
        const uint16_t ch0 = w.alsRaw(), ch1 = static_cast<uint16_t>(ch0 / 4);
        rx[0] = ch1 & 0xff;
        rx[1] = ch1 >> 8;
        rx[2] = ch0 & 0xff;
        rx[3] = ch0 >> 8;
        return ESP_OK;
    }
    if (dev->addr == 0x23) {  // LTR-553 ALS/PS: low byte latches a 16-bit sample
        switch (reg) {
            case 0x86: *rx = 0x92; break;
            case 0x8A: dev->latch = w.alsRaw(); *rx = dev->latch & 0xff; break;
            case 0x8B: *rx = dev->latch >> 8; break;
            case 0x8D: dev->latch = w.psRaw(); *rx = dev->latch & 0xff; break;
            case 0x8E: *rx = (dev->latch >> 8) & 0x07; break;
            default: *rx = 0; break;
        }
    } else {  // AXP2101: reg 0x01 status, 0xA4 battery percent
        if (reg == 0x01) {
            // On USB with a full battery the PMIC reports standby + charge
            // done (externally powered, not charging); below full, charging.
            if (!w.charging()) *rx = 0b01000000;                 // discharging
            else if (w.batteryPct() >= 100) *rx = 0b00000100;    // standby, charge done
            else *rx = 0b00100000;                               // charging
        } else if (reg == 0xA4) {
            *rx = static_cast<uint8_t>(w.batteryPct());
        } else {
            *rx = 0;
        }
    }
    return ESP_OK;
}

// --- Si12T head-touch driver (driver API replaced; parsing identical) -------
struct si12t_dev_t { int unused; };
static si12t_dev_t g_si12t{0};

esp_err_t si12t_init(const si12t_config_t*, si12t_handle_t* handle)
{
    *handle = &g_si12t;
    return ESP_OK;
}
esp_err_t si12t_setup(si12t_handle_t, si12t_type_t, si12t_sensitivity_level_t) { return ESP_OK; }
esp_err_t si12t_read_touch_result(si12t_handle_t, uint8_t* touch_result)
{
    *touch_result = world().headTouchRaw();
    return ESP_OK;
}
void si12t_parse_touch_result_to(uint8_t touch_result, uint8_t* parsed_result)
{
    int index = 0;  // same as components/si12t/src/Si12T.cpp
    for (int j = 0; j < 6; j += 2) parsed_result[index++] = (touch_result >> j) & 0x03;
}

// --- BMI270 accelerometer ------------------------------------------------------
BMI270::BMI270(i2c_master_bus_handle_t, uint8_t) {}
bool BMI270::begin() { return true; }
bool BMI270::update()
{
    world().imuAccel(x_, y_, z_);
    return true;
}
void BMI270::getAccelerometer(float& x, float& y, float& z)
{
    x = x_;
    y = y_;
    z = z_;
}

void BMI270::getGyroscope(float& x, float& y, float& z)
{
    // Sensor axes as mounted in the head: x = pitch rate, z = yaw rate (the
    // firmware learns the mapping by correlation, like on the robot).
    float pitch = 0.0f, yaw = 0.0f;
    world().gyro(pitch, yaw);
    x = pitch;
    y = 0.0f;
    z = -yaw;
}

// --- SCS servo bus -------------------------------------------------------------
// SIM_SERVO_LOG=FILE: every packet on the bus, one line each (ms, kind, id, values) - what the servos are told
static FILE* servoLog()
{
    static FILE* f = [] { const char* p = std::getenv("SIM_SERVO_LOG"); return p ? std::fopen(p, "w") : nullptr; }();
    return f;
}
bool SCSCL::begin(int, int, int, int) { return true; }
int SCSCL::WritePos(uint8_t id, int16_t position, uint16_t time_ms, uint16_t)
{
    if (FILE* f = servoLog()) std::fprintf(f, "%lld pos %u %d %u\n", static_cast<long long>(esp_timer_get_time() / 1000), id, position, time_ms);
    world().servoWrite(id, position, time_ms);
    return 1;
}
int SCSCL::EnableTorque(uint8_t id, uint8_t enable)
{
    if (FILE* f = servoLog()) std::fprintf(f, "%lld torque %u %u\n", static_cast<long long>(esp_timer_get_time() / 1000), id, enable);
    return 1;
}
int SCSCL::writeWord(uint8_t id, uint8_t addr, uint16_t value)
{
    if (FILE* f = servoLog()) std::fprintf(f, "%lld word %u %u %u\n", static_cast<long long>(esp_timer_get_time() / 1000), id, addr, value);
    return 1;
}

// --- Face, LED, speaker: recorded, not drawn ------------------------------------
void FaceEngine::begin() {}
void FaceEngine::update()
{
    // LVGL would raise LV_EVENT_PRESSED on the pixel face; the world tells
    // us when a finger taps the screen.
    if (world().takeScreenTap()) screen_touch_pending_.store(true);
}
void FaceEngine::setEmotion(Emotion e) { emotion_ = e; }
void FaceEngine::setGaze(float x, float y)
{
    gaze_dx_ = static_cast<int>(std::lround(std::clamp(x, -1.0f, 1.0f) * kGazeRangeX));
    gaze_dy_ = static_cast<int>(std::lround(std::clamp(y, -1.0f, 1.0f) * kGazeRangeY));
}

bool LedEngine::begin() { return true; }
void LedEngine::update() {}
void LedEngine::setColor(uint8_t r, uint8_t g, uint8_t b)
{
    breathing_ = false;
    sim::g_led = {r, g, b, false};
}
void LedEngine::setBreathing(uint8_t r, uint8_t g, uint8_t b)
{
    breathing_ = true;
    breath_r_ = r;
    breath_g_ = g;
    breath_b_ = b;
    sim::g_led = {r, g, b, true};
}

bool AudioOutput::begin() { return true; }
FILE* g_voice_dump = nullptr;  // --voice-wav: everything the speaker plays, raw int16 24 kHz
static uint64_t g_voice_end_us = 0;

void AudioOutput::playPcm(const int16_t* samples, size_t count, int volume)
{
    if (g_voice_dump) {
        if (vrtos::nowUs() > g_voice_end_us + 400000) {  // a pause between phrases
            static const int16_t zeros[8400] = {};
            std::fwrite(zeros, sizeof(int16_t), 8400, g_voice_dump);
        }
        std::fwrite(samples, sizeof(int16_t), count, g_voice_dump);
        g_voice_end_us = vrtos::nowUs() + static_cast<uint64_t>(count * 1e6 / kSampleRate);
    }
    // The I2S DMA paces writes at the sample rate; the robot's microphones
    // hear the speaker (World::micRead mixes it back in).
    self_sound_.process(samples, count);
    world().speakerPush(vrtos::nowUs(), samples, count, volume);
    const uint64_t us = static_cast<uint64_t>(count * 1e6 / kSampleRate);
    vrtos::sleepFor(us);
}
