#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>

#include <freertos/FreeRTOS.h>

#include "../perception/vision_frontend.h"

// GC0308 camera (esp32-camera driver via the CoreS3 BSP): RGB565 QQVGA, one frame buffer, ~7 FPS. Every frame goes
// through the vision front end (perception/vision_frontend.h): a 32x24 luminance image whose ego-motion compensated
// difference gives brain v4's motion sense - how much of the picture moves and where. No recognition, no frames leave
// the robot.
//
// Runs its own FreeRTOS task (capture is too slow for the 20 ms main loop, like MicrophoneEngine's task). update(), on
// the main-loop thread, hands the newest result to onVision(), so the brain is only ever touched from one thread.
class CameraEngine {
public:
    struct VisionFrame {
        bool blind = false;                                        // too dark to see anything
        float motionFrac = 0.0f, motionX = 0.0f, motionY = 0.0f;  // share of the picture that moves, its centre
    };

    bool begin();
    void update();  // call every main-loop tick; delivers a new frame here

    // Delivers the latest result from the main-loop thread. The camera task never calls application code directly.
    void onVision(std::function<void(const VisionFrame&)> cb) { on_vision_ = std::move(cb); }
    bool isFresh() const;
    // Frames processed since the last call (stats line), and the cost of the last front-end run in microseconds.
    uint32_t takeFrameCount() { return frames_.exchange(0); }
    uint32_t frontUs() const { return front_us_.load(); }

private:
    static void taskEntry(void* arg);
    void taskLoop();

    std::function<void(const VisionFrame&)> on_vision_;
    VisionFrontEnd front_;
    std::atomic<bool> pending_{false};
    std::atomic<uint32_t> last_ms_{0};
    std::atomic<uint32_t> frames_{0};
    std::atomic<uint32_t> front_us_{0};
    // Camera-task scratch and the lock-protected hand-off copy. Members rather than locals so neither task's small
    // stack carries a frame.
    VisionFrontEnd::Frame frame_{};
    VisionFrame published_{};
    VisionFrame delivered_{};
    portMUX_TYPE lock_ = portMUX_INITIALIZER_UNLOCKED;
};
