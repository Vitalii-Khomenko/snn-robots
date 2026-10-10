#include "policy_gru.h"

#include <algorithm>
#include <cmath>

#include "nn_ops.h"
#include "policy_model_data.h"

using namespace hybrid;

namespace {
bool signedAction(int i)
{
    return i == ACT_YAW || i == ACT_PITCH || i == ACT_GAZE_X || i == ACT_GAZE_Y || i == ACT_SOUND_VALENCE ||
           i == ACT_VOICE_SLOPE || i == ACT_FACE_VALENCE;
}
}  // namespace

bool PolicyGru::trained() const { return policy_model::kTrained; }

void PolicyGru::reset() { h_.fill(0.0f); }

PolicyGru::Action PolicyGru::neutral()
{
    Action a{};
    a[ACT_SPEED] = 0.3f;
    a[ACT_VOICE_REGISTER] = 0.5f;
    a[ACT_HOLD] = 1.0f;
    a[ACT_STYLE_ASK] = 0.5f;
    return a;
}

bool PolicyGru::step(const Input& in, Action& out)
{
    using namespace policy_model;
    out = neutral();
    if (!kTrained) return false;
    nn::dense(inp_w, inp_s, inp_b, in.data(), 64, kPolicyInputs, d_.data(), true);
    nn::gruStep(gru_i_w, gru_i_s, gru_i_b, gru_h_w, gru_h_s, gru_h_b, d_.data(), 64, h_.data(), 64, scratch_.data());
    nn::dense(fc1_w, fc1_s, fc1_b, h_.data(), 48, 64, f_.data(), true);
    float raw[ACT_COUNT];
    nn::dense(fc2_w, fc2_s, fc2_b, f_.data(), ACT_COUNT, 48, raw, false);
    if (!nn::finite(raw, ACT_COUNT) || !nn::finite(h_.data(), 64)) {
        ++nan_count_;
        h_.fill(0.0f);
        return false;
    }
    for (int i = 0; i < ACT_COUNT; ++i) out[i] = signedAction(i) ? std::tanh(raw[i]) : nn::sigmoid(raw[i]);
    return true;
}
