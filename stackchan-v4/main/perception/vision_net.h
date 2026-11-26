#pragma once
#include <array>
#include <cstdint>

#include "vision_frontend.h"

// Compact int8 CNN (Core 1, < 12 ms on ESP32-S3) trained on real room snapshots (.snp) + COCO + real camera sessions.
// Runs on VisionFrontEnd::Frame::image ([3, 24, 32]) and side ([4]) using nn::conv3x3Pool and nn::dense.
// Outputs: [person, face, target_x, target_y, target_conf].
class VisionNet {
public:
    static constexpr int kOutputs = 5;
    struct Out {
        float person = 0.0f;      // 0..1
        float face = 0.0f;        // 0..1
        float targetX = 0.0f;     // -1..1 (left..right)
        float targetY = 0.0f;     // -1..1 (top..bottom)
        float targetConf = 0.0f;  // 0..1
    };

    bool trained() const;
    bool infer(const float* image, const float* side, Out& out);
    bool run(const VisionFrontEnd::Frame& frame, Out& out);

    uint32_t nanCount() const { return nan_count_; }

private:
    std::array<float, 8 * 12 * 16> c1_{};
    std::array<float, 16 * 6 * 8> c2_{};
    std::array<float, 24 * 3 * 4 + 4> c3_{};
    std::array<float, 64> fc1_{};
    std::array<float, kOutputs> raw_{};
    uint32_t nan_count_ = 0;
};
