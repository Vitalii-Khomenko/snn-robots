#include "camera_engine.h"

#include <cstring>
#include "../core/debug_flags.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <bsp/esp-bsp.h>
#include <esp_camera.h>
#include <esp_log.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

namespace {
constexpr const char* kTag = "camera";
// ~7 FPS: this sensor's native rate at QQVGA / this XCLK. The retina's
// direction cells are tuned in frames (1 frame lag ~26 deg/s at 143 ms).
constexpr uint32_t kFrameIntervalMs = 143;
}

bool CameraEngine::begin()
{
    // "I2C must be enabled by bsp_i2c_init(), before camera is
    // initialized" (BSP header) - other sensors already bring the shared
    // I2C bus up, and bsp_i2c_init() is safe to call again.
    bsp_i2c_init();

    camera_config_t config = BSP_CAMERA_DEFAULT_CONFIG;
    // The BSP default is QVGA (320x240) for matching the display; the
    // retina samples 64x48, so QQVGA is plenty and keeps the camera's
    // DMA / frame-buffer footprint small.
    config.frame_size = FRAMESIZE_QQVGA;  // 160x120
    // One buffer: the retina finishes well within a frame period, so
    // double-buffering would only cost memory.
    config.fb_count = 1;
    esp_err_t err     = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "esp_camera_init failed: %d", err);
        return false;
    }

    // Let auto-exposure / gain / white balance converge on the scene once,
    // then lock them. Free-running, they keep hunting and shift the whole
    // frame's brightness from capture to capture, which a change-based
    // retina reads as motion everywhere (seen on hardware: false motion
    // every 1-2 s at rest).
    sensor_t* sensor = esp_camera_sensor_get();
    if (sensor) {
        vTaskDelay(pdMS_TO_TICKS(300));
        sensor->set_exposure_ctrl(sensor, 0);
        sensor->set_gain_ctrl(sensor, 0);
        sensor->set_whitebal(sensor, 0);
    } else {
        ESP_LOGE(kTag, "esp_camera_sensor_get failed - AEC/AGC/AWB left free-running");
    }

    // Core 1 with the microphone task, below it in priority: a 25 ms network run must
    // never keep the microphone DMA waiting, and must not preempt the main loop (core 0).
    if (xTaskCreatePinnedToCore(&CameraEngine::taskEntry, "camera_vision", 5120, this, 2, nullptr, 1) != pdPASS) {
        ESP_LOGE(kTag, "camera task creation failed");
        esp_camera_deinit();
        return false;
    }
    return true;
}

void CameraEngine::taskEntry(void* arg)
{
    static_cast<CameraEngine*>(arg)->taskLoop();
}

void CameraEngine::taskLoop()
{
    uint32_t next_ms = static_cast<uint32_t>(esp_timer_get_time() / 1000);
    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (float* scratch = pending_scratch_.exchange(nullptr)) net_.useFastScratch(scratch);
        if (fb) {
            const int64_t t0 = esp_timer_get_time();
            if (front_.process(fb->buf, fb->width, fb->height, frame_)) {
                VisionFrame result;
                result.networkOk = net_.run(frame_, result.features);
                result.embedding = net_.hidden();
                result.shiftX = frame_.shiftX;
                result.shiftY = frame_.shiftY;
                result.shiftValid = frame_.shiftValid;
                result.blind = frame_.blind();
                result.motionFrac = frame_.motionFrac;
                result.motionX = frame_.motionX;
                result.motionY = frame_.motionY;
                const uint32_t us = static_cast<uint32_t>(esp_timer_get_time() - t0);
                vision_us_.store(us);
                if (us < vision_min_us_.load()) vision_min_us_.store(us);
                portENTER_CRITICAL(&lock_);
                published_ = result;
                portEXIT_CRITICAL(&lock_);
                last_ms_.store(static_cast<uint32_t>(esp_timer_get_time() / 1000));
                frames_.fetch_add(1);
                pending_.store(true);
                if constexpr (kVisionRec != VisionRec::Off) dumpFrame();
                if (uint8_t* out = snap_out_.load()) {  // a snapshot was asked for (snapshot())
                    const int cap = snap_cap_.load();
                    struct __attribute__((packed)) Head {
                        char magic[4];
                        uint16_t w, h;
                        uint32_t raw_len, ms;
                    } head{{'S', 'N', 'P', '1'}, static_cast<uint16_t>(fb->width), static_cast<uint16_t>(fb->height),
                           static_cast<uint32_t>(fb->len), static_cast<uint32_t>(esp_timer_get_time() / 1000)};
                    const int need = static_cast<int>(sizeof(head) + fb->len + sizeof(frame_.image) +
                                                      sizeof(frame_.side) + sizeof(float) + sizeof(result.features));
                    int n = -2;
                    if (need <= cap) {
                        uint8_t* p = out;
                        std::memcpy(p, &head, sizeof(head)), p += sizeof(head);
                        std::memcpy(p, fb->buf, fb->len), p += fb->len;
                        std::memcpy(p, frame_.image.data(), sizeof(frame_.image)), p += sizeof(frame_.image);
                        std::memcpy(p, frame_.side.data(), sizeof(frame_.side)), p += sizeof(frame_.side);
                        std::memcpy(p, &frame_.contrast, sizeof(float)), p += sizeof(float);
                        std::memcpy(p, result.features.data(), sizeof(result.features)), p += sizeof(result.features);
                        n = static_cast<int>(p - out);
                    }
                    snap_out_.store(nullptr);
                    snap_len_.store(n);
                }
            }
            esp_camera_fb_return(fb);
        }
        // Paced from the start of the previous frame, so the cost of the
        // front end and the network does not lower the frame rate.
        next_ms += kFrameIntervalMs;
        const uint32_t now_ms = static_cast<uint32_t>(esp_timer_get_time() / 1000);
        const int32_t wait = static_cast<int32_t>(next_ms - now_ms);
        if (wait > 0) vTaskDelay(pdMS_TO_TICKS(static_cast<uint32_t>(wait)));
        else next_ms = now_ms;
    }
}

bool CameraEngine::isFresh() const
{
    const uint32_t now = static_cast<uint32_t>(esp_timer_get_time() / 1000);
    return static_cast<int32_t>(now - last_ms_.load()) < 300;  // signed: the camera task may stamp a frame just after `now` was read
}

void CameraEngine::update()
{
    if (pending_.exchange(false) && on_vision_) {
        portENTER_CRITICAL(&lock_);
        delivered_ = published_;
        portEXIT_CRITICAL(&lock_);
        on_vision_(delivered_);
    }
}

// "vframe: " + int8 image [3][24][32] (value x 60) as hex, then side[4] and contrast as int16 (x 1000) hex.
void CameraEngine::dumpFrame()
{
    static uint32_t n = 0;
    if ((n++ & 1) != 0) return;
    static char line[2 * (3 * 768 + 10) + 16];
    static const char* hex = "0123456789abcdef";
    int p = 0;
    auto put = [&](uint8_t b) {
        line[p++] = hex[b >> 4];
        line[p++] = hex[b & 15];
    };
    for (int i = 0; i < 3 * 768; ++i) {
        const long v = std::lround(frame_.image[i] * 60.0f);
        put(static_cast<uint8_t>(static_cast<int8_t>(v < -127 ? -127 : v > 127 ? 127 : v)));
    }
    auto put16 = [&](float f) {
        const long v = std::clamp(std::lround(f * 1000.0f), -32768L, 32767L);
        put(static_cast<uint8_t>((v >> 8) & 0xff));
        put(static_cast<uint8_t>(v & 0xff));
    };
    for (float s : frame_.side) put16(s);
    put16(frame_.contrast / 100.0f);
    line[p] = 0;
    std::printf("vframe: %s\n", line);
}

int CameraEngine::snapshot(uint8_t* out, int cap, int timeoutMs)
{
    snap_len_.store(-1);
    snap_cap_.store(cap);
    snap_out_.store(out);
    for (int t = 0; t < timeoutMs; t += 20) {
        const int n = snap_len_.load();
        if (n != -1) return n > 0 ? n : 0;
        vTaskDelay(pdMS_TO_TICKS(20));
    }
    snap_out_.store(nullptr);
    return 0;
}
