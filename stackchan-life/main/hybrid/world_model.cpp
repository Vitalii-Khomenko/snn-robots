#include "world_model.h"

#include <algorithm>
#include <cmath>
#include <iterator>

#include "nn_ops.h"
#include "world_model_data.h"

using namespace hybrid;

namespace {
constexpr float kFast = 0.05f / 10.0f;   // ~10 s average of the error
constexpr float kSlow = 0.05f / 120.0f;  // ~2 min
constexpr int kWarmupTicks = 200;        // 10 s
constexpr int kGroupAt[4] = {0, VIS_COUNT, VIS_COUNT + AUD_COUNT, VIS_COUNT + AUD_COUNT + BODY_COUNT};
static_assert(kPolVisionAt == 0 && kPolAudioAt == VIS_COUNT && kPolBodyAt == VIS_COUNT + AUD_COUNT,
              "the sensed values are the first 49 of the policy input");
}  // namespace

bool WorldModel::trained() const { return world_model::kTrained; }

PolicyGru::Params WorldModel::compiledParams()
{
    using namespace world_model;
    return PolicyGru::Params{inp_w, gru_i_w, gru_h_w, fc1_w, fc2_w, inp_s, inp_b, gru_i_s, gru_i_b,
                             gru_h_s, gru_h_b, fc1_s, fc1_b, fc2_s, fc2_b};
}

const PolicyGru::ParamInfo* WorldModel::paramInfo()
{
    using namespace world_model;
    static const PolicyGru::ParamInfo info[kParamCount] = {
        {"inp_w", true, int(std::size(inp_w))},     {"gru_i_w", true, int(std::size(gru_i_w))},
        {"gru_h_w", true, int(std::size(gru_h_w))}, {"fc1_w", true, int(std::size(fc1_w))},
        {"fc2_w", true, int(std::size(fc2_w))},     {"inp_s", false, int(std::size(inp_s))},
        {"inp_b", false, int(std::size(inp_b))},    {"gru_i_s", false, int(std::size(gru_i_s))},
        {"gru_i_b", false, int(std::size(gru_i_b))}, {"gru_h_s", false, int(std::size(gru_h_s))},
        {"gru_h_b", false, int(std::size(gru_h_b))}, {"fc1_s", false, int(std::size(fc1_s))},
        {"fc1_b", false, int(std::size(fc1_b))},    {"fc2_s", false, int(std::size(fc2_s))},
        {"fc2_b", false, int(std::size(fc2_b))}};
    return info;
}

void WorldModel::step(const float* x)
{
    const float* in = x;  // the sensed values are its first kOut
    if (!world_model::kTrained) return;
    // how wrong was the last prediction?
    if (have_pred_) {
        float total = 0.0f;
        for (int g = 0; g < 3; ++g) {
            float e = 0.0f;
            for (int i = kGroupAt[g]; i < kGroupAt[g + 1]; ++i) {
                const float d = in[i] - pred_[i];
                e += d * d;
            }
            total += e;
            rms_[g] = std::sqrt(e / static_cast<float>(kGroupAt[g + 1] - kGroupAt[g]));
        }
        const float rms = std::sqrt(total / kOut);
        // the first 10 s after a (re)start the memory is empty and the errors are large: they would sit in the slow
        // average for minutes and read as progress. Both averages start after that, from the fast one.
        if (warmup_ < kWarmupTicks) {
            ++warmup_;
            e_fast_ += (warmup_ == 1 ? 1.0f : 0.1f) * (rms - e_fast_);
            e_slow_ = e_fast_;
        } else {
            e_fast_ += kFast * (rms - e_fast_);
            e_slow_ += kSlow * (rms - e_slow_);
            lp_ = std::max(0.0f, e_slow_ - e_fast_);
        }
    }
    // the next prediction. The v3 spiking core's section is zeroed: it is off in mode life, and train_world.py zeroes
    // it in every day (2026-10-17)
    float xin[kIn];
    std::copy(x, x + kIn, xin);
    std::fill(xin + kPolSnnAt, xin + kPolSnnAt + SNN_COUNT, 0.0f);
    const PolicyGru::Params& p = p_;
    nn::dense(p.inp_w, p.inp_s, p.inp_b, xin, 64, kIn, d_.data(), true);
    nn::gruStep(p.gru_i_w, p.gru_i_s, p.gru_i_b, p.gru_h_w, p.gru_h_s, p.gru_h_b, d_.data(), 64, h_.data(), 64,
                scratch_.data());
    nn::dense(p.fc1_w, p.fc1_s, p.fc1_b, h_.data(), 48, 64, f_.data(), true);
    float delta[kOut];
    nn::dense(p.fc2_w, p.fc2_s, p.fc2_b, f_.data(), kOut, 48, delta, false);
    if (!nn::finite(delta, kOut) || !nn::finite(h_.data(), 64)) {
        ++nan_count_;
        h_.fill(0.0f);
        have_pred_ = false;
        warmup_ = 0;
        lp_ = 0.0f;
        return;
    }
    for (int i = 0; i < kOut; ++i) pred_[i] = in[i] + delta[i];
    have_pred_ = true;
}

void WorldModel::logValues(float* out) const
{
    out[0] = rms_[0];
    out[1] = rms_[1];
    out[2] = rms_[2];
    out[3] = lp_ * 10.0f;
}
