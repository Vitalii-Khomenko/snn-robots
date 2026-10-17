#include "babble_net.h"

#include <algorithm>
#include <cmath>

#include "../hybrid/nn_ops.h"
#include "babble_model_data.h"


bool BabbleNet::trained() const { return babble_model::kTrained; }

float BabbleNet::uniform()
{
    rng_ ^= rng_ << 13;
    rng_ ^= rng_ >> 17;
    rng_ ^= rng_ << 5;
    return static_cast<float>(rng_ >> 8) * (1.0f / 16777216.0f);
}

void BabbleNet::begin(const std::array<float, kIntent>& intent, int emotion)
{
    intent_ = intent;
    emotion_ = std::clamp(emotion, 0, kEmotions - 1);
    h_.fill(0.0f);
    step_ = 0;
    have_prev_ = false;
}

bool BabbleNet::next(droid::Note& out)
{
    using namespace babble_model;
    if (!kTrained || step_ >= kMaxNotes) return false;
    float x[kInputs] = {};
    for (int i = 0; i < kIntent; ++i) x[i] = intent_[i];
    constexpr int kEmoAt = 12, kPrevKind = kEmoAt + kEmotions, kPrevShape = kPrevKind + kKinds, kPrevEnd = kPrevShape + kShapes,
                  kPrevCont = kPrevEnd + kEndings, kStepAt = kPrevCont + 4;
    x[kEmoAt + emotion_] = 1.0f;
    if (have_prev_) {
        x[kPrevKind + prev_kind_] = 1.0f;
        x[kPrevShape + prev_shape_] = 1.0f;
        x[kPrevEnd + prev_ending_] = 1.0f;
        x[kPrevCont] = prev_f0_ / 5.0f;
        x[kPrevCont + 1] = prev_slope_ / 8.0f;
        x[kPrevCont + 2] = (prev_dur_ - 250.0f) / 150.0f;
        x[kPrevCont + 3] = prev_loud_;
    }
    x[kStepAt] = static_cast<float>(step_) / 6.0f;
    x[kStepAt + 1] = step_ == 0 ? 1.0f : 0.0f;

    nn::dense(inp_w, inp_s, inp_b, x, 64, kInputs, d_.data(), true);
    nn::gruStep(gru_i_w, gru_i_s, gru_i_b, gru_h_w, gru_h_s, gru_h_b, d_.data(), 64, h_.data(), 48, scratch_.data());
    nn::dense(fc1_w, fc1_s, fc1_b, h_.data(), 48, 48, f_.data(), true);
    float o[kOutputs];
    nn::dense(fc2_w, fc2_s, fc2_b, f_.data(), kOutputs, 48, o, false);
    if (!nn::finite(o, kOutputs) || !nn::finite(h_.data(), 48)) {
        ++nan_count_;
        h_.fill(0.0f);
        return false;
    }
    if (step_ >= 1 && nn::sigmoid(o[0]) > uniform()) return false;  // the phrase is over

    auto draw = [&](const float* logits, int n) {
        float m = logits[0];
        for (int i = 1; i < n; ++i) m = std::max(m, logits[i]);
        float p[16], sum = 0.0f;
        for (int i = 0; i < n; ++i) sum += p[i] = std::exp(logits[i] - m);
        float pick = uniform() * sum;
        for (int i = 0; i < n; ++i) {
            pick -= p[i];
            if (pick <= 0.0f) return i;
        }
        return n - 1;
    };
    constexpr int kShapeAt = 1 + kKinds, kEndingAt = kShapeAt + kKinds * kShapes, kContAt = kEndingAt + kKinds * kEndings;
    const int kind = draw(o + 1, kKinds);
    const int shape = draw(o + kShapeAt + kind * kShapes, kShapes);
    const int ending = draw(o + kEndingAt + kind * kEndings, kEndings);
    const float* c = o + kContAt + kind * kCont;
    const float* sd = cont_std + kind * kCont;
    auto noisy = [&](int i) { return c[i] + (uniform() - 0.5f) * 1.7320508f * sd[i] * 0.5f; };  // uniform, same std
    out.kind = static_cast<droid::Kind>(kind);
    out.shape = static_cast<droid::Shape>(shape);
    out.ending = static_cast<droid::Ending>(ending);
    out.durMs = std::clamp(noisy(0) * 150.0f + 250.0f, 45.0f, 800.0f);
    out.f0St = std::clamp(noisy(1) * 5.0f, -9.0f, 12.0f);
    out.slopeSt = std::clamp(noisy(2) * 8.0f, -18.0f, 18.0f);
    out.loud = std::clamp(noisy(3), 0.3f, 1.0f);
    out.fm = std::clamp(noisy(4), 0.0f, 1.0f);
    out.harm = std::clamp(noisy(5), 0.0f, 1.0f);
    out.vib = std::clamp(noisy(6), 0.0f, 1.0f);
    out.rate = std::clamp(noisy(7), 0.0f, 1.0f);
    out.gapMs = std::clamp((noisy(8) + 0.5f) * 80.0f, 10.0f, 130.0f);
    if (ending != 0 && out.durMs < 140.0f) out.durMs = 140.0f;
    prev_kind_ = kind;
    prev_shape_ = shape;
    prev_ending_ = ending;
    prev_f0_ = out.f0St;
    prev_slope_ = out.slopeSt;
    prev_dur_ = out.durMs;
    prev_loud_ = out.loud;
    have_prev_ = true;
    ++step_;
    return true;
}
