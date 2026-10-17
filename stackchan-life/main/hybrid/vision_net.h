#pragma once
#include <array>
#include <atomic>
#include <cstdint>

#include "layer_params.h"

#include "hybrid_spec.h"
#include "vision_frontend.h"

// The Vision NN: a small CNN trained on the PC (tools/hybrid/train_vision.py)
// on frames of the digital twin, INT8 weights in flash, float activations.
//
//   3x24x32 -> conv 8 -> pool -> conv 16 -> pool -> conv 24 -> pool (24x3x4)
//   + 4 side values -> dense 64 -> 13 outputs
//
// It answers "what visually important thing is happening" - motion by
// region, a person / face, something approaching or receding, where the
// most salient target is - and nothing about what to do about it.
// The photometric features (brightness, light change, brightest spot) come
// from the front end exactly; the net's outputs and those fill the Vision
// section of the FeatureBus.
class VisionNet {
public:
    static constexpr int kOutputs = 13;
    static constexpr int kP1 = 8 * 12 * 16, kP2 = 16 * 6 * 8;  // layer activations (floats)
    // NN output order (after the output nonlinearity).
    enum Out : uint8_t {
        O_MOTION_L, O_MOTION_C, O_MOTION_R, O_MOTION_UP, O_MOTION_DOWN, O_PERSON, O_FACE, O_LOOM, O_RECEDE,
        O_TARGET_X, O_TARGET_Y, O_TARGET_CONF, O_ACTIVITY
    };
    using Features = std::array<float, hybrid::VIS_COUNT>;

    // Runs the net on one front-end frame and fills the 18 Vision features.
    // false (features zeroed) when the network is missing or produced NaN.
    bool run(const VisionFrontEnd::Frame& frame, Features& out);
    // Raw network outputs of the last run (parity tests, training).
    const std::array<float, kOutputs>& lastOutputs() const { return out_; }
    // Moves the 9 KB of layer activations to faster memory (internal SRAM on
    // the robot); the network itself lives in PSRAM. Optional.
    static constexpr size_t kFastScratchBytes = sizeof(float) * (kP1 + kP2);
    void useFastScratch(float* buffer)
    {
        p1_ = buffer;
        p2_ = buffer + kP1;
    }
    bool trained() const;
    uint32_t nanCount() const { return nan_count_; }
    // Weights: compiled in, or the nightly ones from the blob (conv1..3, fc1..2). use() is called from another task
    // than run(): the new set goes to the other buffer and is switched with one atomic store (the old arrays must stay
    // mapped until the run that started on them is over - a frame takes ~20 ms).
    static constexpr int kLayers = 5;
    using Params = std::array<LayerParams, kLayers>;
    static Params compiledParams();
    static const LayerInfo* layerInfo();
    void use(const Params& p)
    {
        const int next = 1 - active_.load();
        params_[next] = p;
        active_.store(next);
    }
    // the hidden layer of the last run (64, after ReLU): what it saw beyond the labels (platform B, bus v4)
    const std::array<float, 64>& hidden() const { return hidden_; }

private:
    std::array<float, kOutputs> out_{};
    // Activations: members (the camera task's stack is small).
    std::array<float, kP1 + kP2> scratch_storage_{};
    float* p1_ = scratch_storage_.data();         // after layer 1 (8 x 12 x 16)
    float* p2_ = scratch_storage_.data() + kP1;   // after layer 2 (16 x 6 x 8)
    std::array<float, 24 * 3 * 4 + VisionFrontEnd::kSide> feat_{};
    std::array<float, 64> hidden_{};
    uint32_t nan_count_ = 0;
    Params params_[2] = {compiledParams(), compiledParams()};
    std::atomic<int> active_{0};
};
