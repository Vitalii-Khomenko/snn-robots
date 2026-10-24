#include "vision_frontend.h"

#include <algorithm>
#include <cmath>

namespace {
uint8_t luminance(uint16_t pixel)
{
    const unsigned r = (pixel >> 11) & 0x1f, g = (pixel >> 5) & 0x3f, b = pixel & 0x1f;
    return static_cast<uint8_t>(((r * 255U / 31U) * 77U + (g * 255U / 63U) * 150U + (b * 255U / 31U) * 29U) >> 8);
}
}  // namespace

bool VisionFrontEnd::process(const uint8_t* buffer, size_t width, size_t height, Frame& out)
{
    if (!buffer || !width || !height) return false;
    for (int row = 0; row < kSampleH; ++row) {
        const size_t y = std::min(height - 1, static_cast<size_t>((static_cast<float>(row) + 0.5f) * height / kSampleH));
        for (int col = 0; col < kSampleW; ++col) {
            const size_t x = std::min(width - 1, static_cast<size_t>((static_cast<float>(col) + 0.5f) * width / kSampleW));
            const size_t o = (y * width + x) * 2;
            // esp32-camera's RGB565 is high byte first.
            sample_[row * kSampleW + col] = luminance(static_cast<uint16_t>((buffer[o] << 8) | buffer[o + 1]));
        }
    }
    int sum = 0;
    for (int row = 0; row < kHeight; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            const int t = (row * 2) * kSampleW + col * 2;
            const int v = (sample_[t] + sample_[t + 1] + sample_[t + kSampleW] + sample_[t + kSampleW + 1] + 2) / 4;
            current_[row * kWidth + col] = static_cast<uint8_t>(v);
            sum += v;
        }
    }
    const float mean = static_cast<float>(sum) / kCells;
    if (!primed_) {
        previous_ = current_;
        previous_mean_ = mean;
        primed_ = true;
        return false;
    }

    {
        float var = 0.0f;
        for (int i = 0; i < kCells; ++i) var += (current_[i] - mean) * (current_[i] - mean);
        out.contrast = std::sqrt(var / kCells);
    }
    out.brightness = mean / 255.0f;
    out.lightChange = std::clamp((mean - previous_mean_) / 12.0f, -1.0f, 1.0f);
    findLight(out);

    // Global image shift: the (dx, dy) that makes the previous frame match
    // the current one best. A shift is only believed when it explains the
    // image clearly better than no shift at all (uniform walls explain
    // nothing; a person walking in a still room must not read as ego-motion).
    int best_cost = 1 << 30, best_dx = 0, best_dy = 0, zero_cost = 0;
    constexpr int kMargin = kMaxShift;
    for (int dy = -kMaxShift; dy <= kMaxShift; ++dy) {
        for (int dx = -kMaxShift; dx <= kMaxShift; ++dx) {
            int cost = 0;
            for (int y = kMargin; y < kHeight - kMargin; ++y) {
                const uint8_t* c = &current_[y * kWidth];
                const uint8_t* p = &previous_[(y - dy) * kWidth - dx];
                for (int x = kMargin; x < kWidth - kMargin; ++x) cost += std::abs(static_cast<int>(c[x]) - p[x]);
            }
            if (dx == 0 && dy == 0) zero_cost = cost;
            if (cost < best_cost) {
                best_cost = cost;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }
    constexpr int kSamples = (kWidth - 2 * kMargin) * (kHeight - 2 * kMargin);
    const bool trusted = (best_dx != 0 || best_dy != 0) && best_cost * 100 < zero_cost * 80 && zero_cost - best_cost > kSamples * 2;
    const int dx = trusted ? best_dx : 0, dy = trusted ? best_dy : 0;
    out.shiftX = static_cast<float>(dx);
    out.shiftY = static_cast<float>(dy);
    out.shiftValid = trusted;

    float* lum = out.image.data();
    float* comp = lum + kCells;
    float* raw = comp + kCells;
    constexpr float kLumScale = 1.0f / 64.0f, kDiffScale = 1.0f / 48.0f;
    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const int i = y * kWidth + x;
            const float cur = current_[i];
            lum[i] = std::clamp((cur - mean) * kLumScale, -2.0f, 2.0f);
            raw[i] = std::clamp((cur - previous_[i]) * kDiffScale, -2.0f, 2.0f);
            const int px = x - dx, py = y - dy;
            const bool inside = px >= 0 && px < kWidth && py >= 0 && py < kHeight;
            comp[i] = inside ? std::clamp((cur - previous_[py * kWidth + px]) * kDiffScale, -2.0f, 2.0f) : 0.0f;
        }
    }
    {
        int moving = 0;
        float sx = 0.0f, sy = 0.0f;
        for (int y = kMaxShift; y < kHeight - kMaxShift; ++y)
            for (int x = kMaxShift; x < kWidth - kMaxShift; ++x)
                if (std::fabs(comp[y * kWidth + x]) > kMotionCellThr) {
                    ++moving;
                    sx += static_cast<float>(x);
                    sy += static_cast<float>(y);
                }
        const bool light_jump = std::fabs(mean - previous_mean_) > 6.0f;
        out.motionFrac = light_jump ? 0.0f : static_cast<float>(moving) / kSamples;
        out.motionX = moving ? (sx / moving + 0.5f) / (kWidth * 0.5f) - 1.0f : 0.0f;
        out.motionY = moving ? (sy / moving + 0.5f) / (kHeight * 0.5f) - 1.0f : 0.0f;
    }
    out.side = {out.shiftX / kMaxShift, out.shiftY / kMaxShift, trusted ? 1.0f : 0.0f, mean / 255.0f};
    previous_ = current_;
    previous_mean_ = mean;
    return true;
}

// Brightest spot of the frame (a lamp, a window), static or not: the 8x8-
// sample block excess over the frame mean, centroid of the blocks close to
// the peak.
void VisionFrontEnd::findLight(Frame& out) const
{
    std::array<int, kGridW * kGridH> block{};
    int total = 0;
    for (int row = 0; row < kSampleH; ++row) {
        for (int col = 0; col < kSampleW; ++col) {
            const int v = sample_[row * kSampleW + col];
            block[(row / kBlock) * kGridW + col / kBlock] += v;
            total += v;
        }
    }
    const int mean = total / kSampleCount;
    int peak = 0;
    for (auto& b : block) {
        b = b / (kBlock * kBlock) - mean;
        peak = std::max(peak, b);
    }
    out.lightStrength = 0.0f;
    out.lightX = out.lightY = 0.0f;
    if (peak < kLightMinExcess) return;
    float weight = 0.0f, sx = 0.0f, sy = 0.0f;
    for (int y = 0; y < kGridH; ++y) {
        for (int x = 0; x < kGridW; ++x) {
            const int excess = block[y * kGridW + x] - peak * 3 / 5;
            if (excess <= 0) continue;
            weight += static_cast<float>(excess);
            sx += static_cast<float>(excess) * (static_cast<float>(x) + 0.5f);
            sy += static_cast<float>(excess) * (static_cast<float>(y) + 0.5f);
        }
    }
    out.lightX = sx / weight / (kGridW * 0.5f) - 1.0f;
    out.lightY = sy / weight / (kGridH * 0.5f) - 1.0f;
    out.lightStrength = std::clamp(static_cast<float>(peak - kLightMinExcess) / (kLightFull - kLightMinExcess), 0.0f, 1.0f);
}
