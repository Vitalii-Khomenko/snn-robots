#include "nn_v4.h"

#include <algorithm>
#include <cmath>

#include "nn_v4_data.h"

#if __has_include(<esp_heap_caps.h>)
#include <cstring>
#include <esp_heap_caps.h>
#define NNV4_HAVE_PSRAM 1
#endif

namespace {
constexpr int kEmo = NnV4::kEmo;
constexpr int kAct = NnV4::kAct;
static_assert(sizeof(nnv4::kWAct1) / sizeof(float) == NnV4::kAct1W && sizeof(nnv4::kWAct2) / sizeof(float) == NnV4::kAct2W,
              "action heads' shape (tools/v4/nn_v4.py)");
// the GRU's input weights must match the spec's input count: a new input (spec_v4.PREV) needs a retrained, re-exported
// network, or the GRU reads past kWih (2026-10-27: 55 inputs in the spec, 49 in the old weights)
static_assert(sizeof(nnv4::kWih) / sizeof(float) == 3 * NnV4::kHidden * NnV4::kIn,
              "GRU input weights' shape: retrain and export_nn_v4.py after a change of spec_v4.PREV");
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
uint32_t NnV4::id() { return nnv4::kNnId; }

NnV4::NnV4()
    : wih_(nnv4::kWih), whh_(nnv4::kWhh), wemo_(nnv4::kWEmo), act1_w_(nnv4::kWAct1), act1_b_(nnv4::kBAct1),
      act2_w_(nnv4::kWAct2), act2_b_(nnv4::kBAct2), base_act1_w_(nnv4::kWAct1), base_act1_b_(nnv4::kBAct1),
      base_act2_w_(nnv4::kWAct2), base_act2_b_(nnv4::kBAct2)
{
}

namespace {
const float* copyToPsram(const float* p, size_t n)
{
#ifdef NNV4_HAVE_PSRAM
    if (void* m = heap_caps_malloc(n * sizeof(float), MALLOC_CAP_SPIRAM)) {
        std::memcpy(m, p, n * sizeof(float));
        return static_cast<const float*>(m);
    }
#endif
    (void)n;
    return p;
}
}  // namespace

void NnV4::useFastWeights()
{
    wih_ = copyToPsram(nnv4::kWih, sizeof(nnv4::kWih) / sizeof(float));
    whh_ = copyToPsram(nnv4::kWhh, sizeof(nnv4::kWhh) / sizeof(float));
    wemo_ = copyToPsram(nnv4::kWEmo, sizeof(nnv4::kWEmo) / sizeof(float));
    const bool learned = act1_w_ != base_act1_w_;
    base_act1_w_ = copyToPsram(nnv4::kWAct1, kAct1W);
    base_act2_w_ = copyToPsram(nnv4::kWAct2, kAct2W);
    if (!learned) {
        act1_w_ = base_act1_w_;
        act2_w_ = base_act2_w_;
    }
}

void NnV4::useHeads(const float* act1W, const float* act1B, const float* act2W, const float* act2B)
{
    if (!(act1W && act1B && act2W && act2B)) {
        act1_w_ = base_act1_w_, act1_b_ = base_act1_b_, act2_w_ = base_act2_w_, act2_b_ = base_act2_b_;
        return;
    }
    constexpr size_t n = kAct1W + kAct1B + kAct2W + kAct2B;
    if (!heads_buf_) {
#ifdef NNV4_HAVE_PSRAM
        heads_buf_ = static_cast<float*>(heap_caps_malloc(n * sizeof(float), MALLOC_CAP_SPIRAM));
#else
        static float buf[n];
        heads_buf_ = buf;
#endif
    }
    if (!heads_buf_) {  // no memory: read them where they are
        act1_w_ = act1W, act1_b_ = act1B, act2_w_ = act2W, act2_b_ = act2B;
        return;
    }
    float* p = heads_buf_;
    act1_w_ = p, p = std::copy(act1W, act1W + kAct1W, p);
    act1_b_ = p, p = std::copy(act1B, act1B + kAct1B, p);
    act2_w_ = p, p = std::copy(act2W, act2W + kAct2W, p);
    act2_b_ = p, std::copy(act2B, act2B + kAct2B, p);
}

bool NnV4::step(const In& in, Out& out)
{
    if (!nnv4::kTrained) return false;
    using namespace nnv4;
    const int64_t t0 = clock_us_ ? clock_us_() : 0;
    constexpr int H = kHidden;
    // GRU, PyTorch gate order (r, z, n)
    float gi[3 * H], gh[3 * H];
    denseF(wih_, kBih, in.data(), 3 * H, kIn, gi);
    denseF(whh_, kBhh, h_.data(), 3 * H, H, gh);
    for (int j = 0; j < H; ++j) {
        const float r = sigmoidf(gi[j] + gh[j]);
        const float z = sigmoidf(gi[H + j] + gh[H + j]);
        const float n = std::tanh(gi[2 * H + j] + r * gh[2 * H + j]);
        h_[j] = (1.0f - z) * n + z * h_[j];
    }
    float raw[kOut];
    float e[kEmo + 1];
    denseF(wemo_, kBEmo, h_.data(), kEmo + 1, H, e);
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
    denseF(act1_w_, act1_b_, cat, H, H + kEmo, hid);
    for (float& v : hid) v = std::tanh(v);
    denseF(act2_w_, act2_b_, hid, kAct, H, act);
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
    if (clock_us_) last_us_ = static_cast<uint32_t>(clock_us_() - t0);
    for (float v : out)
        if (!std::isfinite(v)) {
            reset();
            return false;
        }
    return true;
}
