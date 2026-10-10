#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

// Camera frame -> the Vision NN's input tensor, in plain deterministic C++.
//
// No recognition happens here. A QQVGA RGB565 frame is sampled to 64x48, 2x2
// pooled to a 32x24 luminance image and turned into three channels:
//
//   0  luminance with the frame mean removed (exposure drift cancels out)
//   1  temporal difference after the global image shift (ego-motion: the
//      head turning) has been searched out and subtracted - only things that
//      move in the room itself remain
//   2  plain temporal difference (what the net needs to tell its own motion
//      from the world's)
//
// plus four side values (shift x/y, shift trusted, mean brightness) and a
// few exact photometric features (brightness, light change, brightest spot)
// that need no learning. The CNN (vision_net.h) turns it into the Vision
// features of the FeatureBus.
class VisionFrontEnd {
public:
    static constexpr int kWidth = 32;
    static constexpr int kHeight = 24;
    static constexpr int kCells = kWidth * kHeight;
    static constexpr int kChannels = 3;
    static constexpr int kSide = 4;

    struct Frame {
        std::array<float, kChannels * kCells> image{};  // [3][24][32]
        std::array<float, kSide> side{};
        float brightness = 0.0f;      // 0..1
        float lightChange = 0.0f;     // -1..1, mean luminance step since the last frame
        float lightX = 0.0f, lightY = 0.0f, lightStrength = 0.0f;  // brightest spot, image coords, 0..1
        float shiftX = 0.0f, shiftY = 0.0f;  // global image shift in pooled pixels
        bool shiftValid = false;
        // Spread of the luminance image (8-bit levels). A dark room is sensor noise
        // only (< ~3.5): nothing can be seen, and the network must not guess.
        float contrast = 0.0f;
        bool blind() const { return contrast < 3.5f; }
    };

    // False while the first frame primes the temporal reference.
    bool process(const uint8_t* rgb565, size_t width, size_t height, Frame& out);

private:
    static constexpr int kSampleW = kWidth * 2, kSampleH = kHeight * 2, kSampleCount = kSampleW * kSampleH;
    static constexpr int kBlock = 8, kGridW = kSampleW / kBlock, kGridH = kSampleH / kBlock;
    static constexpr int kLightMinExcess = 25, kLightFull = 60;
    static constexpr int kMaxShift = 3;
    void findLight(Frame& out) const;

    std::array<uint8_t, kSampleCount> sample_{};
    std::array<uint8_t, kCells> current_{}, previous_{};
    float previous_mean_ = 0.0f;
    bool primed_ = false;
};
