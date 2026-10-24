#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

// Small inference kernels of the INT8 networks (the Auditory NN, the voice's BabbleNet). Weights are INT8 with one
// float scale per output row / channel, activations are float. The Python side (platform B:
// stackchan-life/tools/hybrid/nn.py) quantizes with the same rule and mirrors every function here, so a trained
// network behaves identically on the PC, in the simulator and on the robot.
namespace nn {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// y[r] = act(scale[r] * sum_c w[r][c] x[c] + bias[r]), w row-major [rows][cols].
inline void dense(const int8_t* w, const float* scale, const float* bias, const float* x, int rows, int cols,
                  float* y, bool relu)
{
    for (int r = 0; r < rows; ++r) {
        const int8_t* row = w + r * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; ++c) acc += static_cast<float>(row[c]) * x[c];
        const float v = acc * scale[r] + bias[r];
        y[r] = relu ? std::max(v, 0.0f) : v;
    }
}

// Fused conv3x3 (stride 1, zero padding 1) + ReLU + 2x2 max pooling: only two
// output rows exist at a time, so the full-resolution activation map (24 KB
// for the first layer) is never stored. in [C][H][W] -> out [K][H/2][W/2].
// W <= 32.
inline void conv3x3Pool(const float* in, int channels, int h, int w_, const int8_t* w, const float* scale,
                        const float* bias, int kernels, float* out)
{
    float row[2][32];
    const int oh = h / 2, ow = w_ / 2;
    for (int py = 0; py < oh; ++py) {
        for (int k = 0; k < kernels; ++k) {
            for (int r = 0; r < 2; ++r)
                for (int x = 0; x < w_; ++x) row[r][x] = 0.0f;
            for (int c = 0; c < channels; ++c) {
                const int8_t* wk = w + (k * channels + c) * 9;
                const float* ic = in + c * h * w_;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const float wv = static_cast<float>(wk[(dy + 1) * 3 + (dx + 1)]);
                        if (wv == 0.0f) continue;
                        const int x0 = std::max(0, -dx), x1 = std::min(w_, w_ - dx);
                        for (int r = 0; r < 2; ++r) {
                            const int y = 2 * py + r + dy;
                            if (y < 0 || y >= h) continue;
                            const float* src = ic + y * w_ + dx;
                            float* dst = row[r];
                            for (int x = x0; x < x1; ++x) dst[x] += wv * src[x];
                        }
                    }
                }
            }
            const float s = scale[k], b = bias[k];
            for (int px = 0; px < ow; ++px) {
                const float a = std::max(row[0][2 * px], row[0][2 * px + 1]);
                const float c2 = std::max(row[1][2 * px], row[1][2 * px + 1]);
                out[(k * oh + py) * ow + px] = std::max(std::max(a, c2) * s + b, 0.0f);
            }
        }
    }
}

// One GRU step, PyTorch gate order (r, z, n). wi [3H][I], wh [3H][H].
inline void gruStep(const int8_t* wi, const float* si, const float* bi, const int8_t* wh, const float* sh,
                    const float* bh, const float* x, int inputs, float* h, int hidden, float* scratch /* 6H */)
{
    float* gi = scratch;
    float* gh = scratch + 3 * hidden;
    dense(wi, si, bi, x, 3 * hidden, inputs, gi, false);
    dense(wh, sh, bh, h, 3 * hidden, hidden, gh, false);
    for (int j = 0; j < hidden; ++j) {
        const float r = sigmoid(gi[j] + gh[j]);
        const float z = sigmoid(gi[hidden + j] + gh[hidden + j]);
        const float n = std::tanh(gi[2 * hidden + j] + r * gh[2 * hidden + j]);
        h[j] = (1.0f - z) * n + z * h[j];
    }
}

inline bool finite(const float* x, int n)
{
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(x[i])) return false;
    return true;
}

}  // namespace nn
