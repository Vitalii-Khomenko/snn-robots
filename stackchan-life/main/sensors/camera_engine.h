#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>

#include <freertos/FreeRTOS.h>

#include "../hybrid/vision_frontend.h"
#include "../hybrid/vision_net.h"

// GC0308 camera (esp32-camera driver via the CoreS3 BSP): RGB565 QQVGA, one
// frame buffer, ~7 FPS. Every frame goes through the Vision front end (a
// 32x24 luminance image with ego-motion compensated difference) and the
// Vision NN (hybrid/vision_net.h), which reports what visually matters as
// the Vision section of the FeatureBus.
//
// Runs its own FreeRTOS task (capture + network are too slow for the 20 ms
// main loop, like MicrophoneEngine's task). update(), on the main-loop
// thread, hands the newest result to onVision(), so the brain is only ever
// touched from one thread.
class CameraEngine {
public:
    struct VisionFrame {
        VisionNet::Features features{};
        float shiftX = 0.0f, shiftY = 0.0f;  // global image shift (pooled pixels): ego-motion
        bool shiftValid = false;
        bool networkOk = false;
        std::array<float, 64> embedding{};  // the Vision NN's hidden layer (platform B, bus v4)
    };

    bool begin();
    void update();  // call every main-loop tick; delivers a new frame here

    // Delivers the latest vision result from the main-loop thread. The
    // camera task never calls application code directly.
    void onVision(std::function<void(const VisionFrame&)> cb) { on_vision_ = std::move(cb); }
    bool isFresh() const;
    // Frames processed since the last call (stats line), and the cost of the
    // last front end + network run in microseconds.
    uint32_t takeFrameCount() { return frames_.exchange(0); }
    uint32_t visionUs() const { return vision_us_.load(); }
    uint32_t takeVisionMinUs() { return vision_min_us_.exchange(0xffffffffu); }
    // Layer activations in faster memory (taken over by the camera task between frames).
    void useFastScratch(float* buffer) { pending_scratch_.store(buffer); }
    const VisionNet& net() const { return net_; }
    void useVisionWeights(const VisionNet::Params& p) { net_.use(p); }  // the nightly weights (any task)
    // Media for the PC (platform B: the perception networks learn from what this camera really sees; the owner
    // allowed frames to the home PC, 2026-10-17). Called from another task: asks the camera task for the next frame and
    // waits up to timeoutMs. out gets: the raw frame (RGB565, w x h), the front end's frame (3 x 32 x 24 float, side,
    // contrast) and the Vision NN's outputs. Returns the bytes written, 0 if no frame came.
    int snapshot(uint8_t* out, int cap, int timeoutMs);

private:
    static void taskEntry(void* arg);
    void taskLoop();

    std::function<void(const VisionFrame&)> on_vision_;
    VisionFrontEnd front_;
    VisionNet net_;
    std::atomic<bool> pending_{false};
    std::atomic<uint32_t> last_ms_{0};
    void dumpFrame();
    std::atomic<uint32_t> frames_{0};
    std::atomic<uint32_t> vision_us_{0}, vision_min_us_{0xffffffffu};
    std::atomic<float*> pending_scratch_{nullptr};
    std::atomic<uint8_t*> snap_out_{nullptr};
    std::atomic<int> snap_cap_{0}, snap_len_{-1};
    // Camera-task scratch and the lock-protected hand-off copy. Members
    // rather than locals so neither task's small stack carries a frame.
    VisionFrontEnd::Frame frame_{};
    VisionFrame published_{};
    VisionFrame delivered_{};
    portMUX_TYPE lock_ = portMUX_INITIALIZER_UNLOCKED;
};
