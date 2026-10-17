#include "policy_gru.h"

#include <algorithm>
#include <cmath>
#include <iterator>

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

PolicyGru::Params PolicyGru::compiledParams()
{
    using namespace policy_model;
    return Params{inp_w, gru_i_w, gru_h_w, fc1_w, fc2_w, inp_s, inp_b, gru_i_s, gru_i_b,
                  gru_h_s, gru_h_b, fc1_s, fc1_b, fc2_s, fc2_b};
}

const PolicyGru::ParamInfo* PolicyGru::paramInfo()
{
    using namespace policy_model;
    static const ParamInfo info[kParamCount] = {
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
    out = neutral();
    if (!policy_model::kTrained) return false;
    const Params& p = p_;
    nn::dense(p.inp_w, p.inp_s, p.inp_b, in.data(), 64, kPolicyInputs, d_.data(), true);
    nn::gruStep(p.gru_i_w, p.gru_i_s, p.gru_i_b, p.gru_h_w, p.gru_h_s, p.gru_h_b, d_.data(), 64, h_.data(), 64,
                scratch_.data());
    nn::dense(p.fc1_w, p.fc1_s, p.fc1_b, h_.data(), 48, 64, f_.data(), true);
    float raw[ACT_COUNT];
    nn::dense(p.fc2_w, p.fc2_s, p.fc2_b, f_.data(), ACT_COUNT, 48, raw, false);
    if (!nn::finite(raw, ACT_COUNT) || !nn::finite(h_.data(), 64)) {
        ++nan_count_;
        h_.fill(0.0f);
        return false;
    }
    for (int i = 0; i < ACT_COUNT; ++i) out[i] = signedAction(i) ? std::tanh(raw[i]) : nn::sigmoid(raw[i]);
    return true;
}
