#include "online_learner.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr float kGamma = 0.995f;      // per 50 ms tick: ~10 s ahead (as the critic on the PC)
constexpr float kLambda = 0.95f;
constexpr float kRewardScale = 0.027f;  // the critic's return scale (critic_net.pt "scale", 2026-10-14: 0.0268)
constexpr float kAlphaV = 2e-4f;
constexpr float kAlphaA = 5e-5f;
constexpr float kMaxDw = 1.0f;
constexpr float kMaxDelta = 2.0f;      // one surprising tick cannot move everything
}  // namespace

void OnlineLearner::apply(const float* f, float* out) const
{
    for (int r = 0; r < kRows; ++r) {
        float s = 0.0f;
        for (int k = 0; k < kF; ++k) s += dw_[r][k] * f[k];
        out[r] += s;
    }
}

void OnlineLearner::learn(float reward, const float* f, const float* out, const Decision& d, bool night)
{
    using O = ActorNet::Out;
    value_ = out[O::Value];
    if (started_) {
        delta_ = std::clamp(reward / kRewardScale + kGamma * value_ - v_prev_, -kMaxDelta, kMaxDelta);
        for (int k = 0; k < kF; ++k)
            dw_[O::Value][k] = std::clamp(dw_[O::Value][k] + kAlphaV * delta_ * ev_[k], -kMaxDw, kMaxDw);
        // the actor learns by day only: at night the collector's quiet rules decide, not the actor's samples
        if (!night)
            for (int r : {O::Move, O::YawMu, O::PitchMu, O::Call})
                for (int k = 0; k < kF; ++k)
                    dw_[r][k] = std::clamp(dw_[r][k] + kAlphaA * delta_ * ea_[r][k], -kMaxDw, kMaxDw);
    }
    // new traces: the value's from these features, the actor's from the decision just taken
    const auto sig = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    float g[kRows] = {};
    if (d.move_possible) g[O::Move] = (d.moved ? 1.0f : 0.0f) - sig(out[O::Move]);
    if (d.call_possible) g[O::Call] = (d.called ? 1.0f : 0.0f) - sig(out[O::Call]);
    if (d.moved) {
        const float my = std::tanh(out[O::YawMu]), mp = std::tanh(out[O::PitchMu]);
        const float sy = std::exp(std::clamp(out[O::YawLogSig], -3.0f, 1.0f));
        const float sp = std::exp(std::clamp(out[O::PitchLogSig], -3.0f, 1.0f));
        g[O::YawMu] = std::clamp((d.yaw - my) / (sy * sy) * (1.0f - my * my), -5.0f, 5.0f);
        g[O::PitchMu] = std::clamp((d.pitch - mp) / (sp * sp) * (1.0f - mp * mp), -5.0f, 5.0f);
    }
    for (int k = 0; k < kF; ++k) {
        ev_[k] = kGamma * kLambda * ev_[k] + f[k];
        for (int r : {O::Move, O::YawMu, O::PitchMu, O::Call}) ea_[r][k] = kGamma * kLambda * ea_[r][k] + g[r] * f[k];
        f_prev_[k] = f[k];
    }
    v_prev_ = value_;
    started_ = true;
}

void OnlineLearner::reset()
{
    for (auto& row : dw_) row.fill(0.0f);
    for (auto& row : ea_) row.fill(0.0f);
    ev_.fill(0.0f);
    started_ = false;
}

float OnlineLearner::size() const
{
    float s = 0.0f;
    for (const auto& row : dw_)
        for (float x : row) s += std::fabs(x);
    return s / (kRows * kF);
}

void OnlineLearner::setWeights(const Weights& w)
{
    for (int r = 0; r < kRows; ++r)
        for (int k = 0; k < kF; ++k) dw_[r][k] = std::isfinite(w[r][k]) ? std::clamp(w[r][k], -kMaxDw, kMaxDw) : 0.0f;
}
