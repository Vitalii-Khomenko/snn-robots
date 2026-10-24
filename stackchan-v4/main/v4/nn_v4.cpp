#include "nn_v4.h"

#include <algorithm>
#include <cmath>

#include "nn_v4_data.h"

namespace {
constexpr int kEmo = NnV4::kEmo;
constexpr int kAct = NnV4::kAct;
static_assert(sizeof(nnv4::kWAct1) / sizeof(float) == NnV4::kAct1W && sizeof(nnv4::kWAct2) / sizeof(float) == NnV4::kAct2W,
              "action heads' shape (tools/v4/nn_v4.py)");
float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void denseF(const float* w, const float* b, const float* x, int rows, int cols, float* y)
{
    for (int r = 0; r < rows; ++r) {
        const float* row = w + static_cast<size_t>(r) * cols;
        float acc = b[r];
        for (int c = 0; c < cols; ++c) acc += row[c] * x[c];
        y[r] = acc;
    }
}
}  // namespace

bool NnV4::trained() const { return nnv4::kTrained; }

void NnV4::useHeads(const float* act1W, const float* act1B, const float* act2W, const float* act2B)
{
    heads_set_ = act1W && act1B && act2W && act2B;
    act1_w_ = act1W;
    act1_b_ = act1B;
    act2_w_ = act2W;
    act2_b_ = act2B;
}

bool NnV4::step(const In& in, Out& out)
{
    if (!nnv4::kTrained) return false;
    using namespace nnv4;
    constexpr int H = kHidden;
    // GRU, PyTorch gate order (r, z, n)
    float gi[3 * H], gh[3 * H];
    denseF(kWih, kBih, in.data(), 3 * H, kIn, gi);
    denseF(kWhh, kBhh, h_.data(), 3 * H, H, gh);
    for (int j = 0; j < H; ++j) {
        const float r = sigmoidf(gi[j] + gh[j]);
        const float z = sigmoidf(gi[H + j] + gh[H + j]);
        const float n = std::tanh(gi[2 * H + j] + r * gh[2 * H + j]);
        h_[j] = (1.0f - z) * n + z * h_[j];
    }
    float raw[kOut];
    float e[kEmo + 1];
    denseF(kWEmo, kBEmo, h_.data(), kEmo + 1, H, e);
    // the emotion's probabilities feed the action heads
    float cat[H + kEmo];
    std::copy(h_.begin(), h_.end(), cat);
    {
        float mx = e[0];
        for (int k = 1; k < kEmo; ++k) mx = std::max(mx, e[k]);
        float sum = 0.0f;
        for (int k = 0; k < kEmo; ++k) sum += cat[H + k] = std::exp(e[k] - mx);
        for (int k = 0; k < kEmo; ++k) cat[H + k] /= sum;
    }
    float hid[H], act[kAct];
    denseF(heads_set_ ? act1_w_ : kWAct1, heads_set_ ? act1_b_ : kBAct1, cat, H, H + kEmo, hid);
    for (float& v : hid) v = std::tanh(v);
    denseF(heads_set_ ? act2_w_ : kWAct2, heads_set_ ? act2_b_ : kBAct2, hid, kAct, H, act);
    for (int k = 0; k < kEmo; ++k) raw[kEmoIdx[k]] = e[k];
    raw[kStrengthIdx] = e[kEmo];
    for (int k = 0; k < kAct; ++k) raw[kActIdx[k]] = act[k];
    // activations: per output kind (0 sigmoid, 1 tanh, 2.. a softmax group)
    for (int i = 0; i < kOut; ++i)
        if (kKind[i] == 0) out[i] = sigmoidf(raw[i]);
        else if (kKind[i] == 1) out[i] = std::tanh(raw[i]);
    for (uint8_t g = 2; g < 8; ++g) {
        float mx = -1e30f, sum = 0.0f;
        for (int i = 0; i < kOut; ++i)
            if (kKind[i] == g) mx = std::max(mx, raw[i]);
        if (mx < -1e29f) continue;
        for (int i = 0; i < kOut; ++i)
            if (kKind[i] == g) sum += out[i] = std::exp(raw[i] - mx);
        for (int i = 0; i < kOut; ++i)
            if (kKind[i] == g) out[i] /= sum;
    }
    for (float v : out)
        if (!std::isfinite(v)) {
            reset();
            return false;
        }
    return true;
}
