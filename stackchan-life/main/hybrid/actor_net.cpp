#include "actor_net.h"

#include <iterator>

#include "actor_model_data.h"
#include "nn_ops.h"

PolicyGru::Params ActorNet::compiledParams()
{
    using namespace actor_model;
    return PolicyGru::Params{inp_w, gru_i_w, gru_h_w, fc1_w, fc2_w, inp_s, inp_b, gru_i_s, gru_i_b,
                             gru_h_s, gru_h_b, fc1_s, fc1_b, fc2_s, fc2_b};
}

const PolicyGru::ParamInfo* ActorNet::paramInfo()
{
    using namespace actor_model;
    static const PolicyGru::ParamInfo info[kParamCount] = {
        {"inp_w", true, int(std::size(inp_w))},     {"gru_i_w", true, int(std::size(gru_i_w))},
        {"gru_h_w", true, int(std::size(gru_h_w))}, {"fc1_w", true, int(std::size(fc1_w))},
        {"fc2_w", true, int(std::size(fc2_w))},     {"inp_s", false, int(std::size(inp_s))},
        {"inp_b", false, int(std::size(inp_b))},    {"gru_i_s", false, int(std::size(gru_i_s))},
        {"gru_i_b", false, int(std::size(gru_i_b))}, {"gru_h_s", false, int(std::size(gru_h_s))},
        {"gru_h_b", false, int(std::size(gru_h_b))}, {"fc1_s", false, int(std::size(fc1_s))},
        {"fc1_b", false, int(std::size(fc1_b))},    {"fc2_s", false, int(std::size(fc2_s))},
        {"fc2_b", false, int(std::size(fc2_b))}};
    static_assert(std::size(inp_w) == 64 * kIn && std::size(fc2_b) == kOut, "actor_model_data.h: 71 -> 24");
    return info;
}

bool ActorNet::step(const float* in, float* out)
{
    const PolicyGru::Params& p = p_;
    nn::dense(p.inp_w, p.inp_s, p.inp_b, in, 64, kIn, d_.data(), true);
    nn::gruStep(p.gru_i_w, p.gru_i_s, p.gru_i_b, p.gru_h_w, p.gru_h_s, p.gru_h_b, d_.data(), 64, h_.data(), 64,
                scratch_.data());
    nn::dense(p.fc1_w, p.fc1_s, p.fc1_b, h_.data(), 48, 64, f_.data(), true);
    float o[kOut];
    nn::dense(p.fc2_w, p.fc2_s, p.fc2_b, f_.data(), kOut, 48, o, false);
    if (!nn::finite(o, kOut) || !nn::finite(h_.data(), 64)) {
        ++nan_count_;
        h_.fill(0.0f);
        return false;
    }
    for (int i = 0; i < kOut; ++i) out[i] = o[i];
    return true;
}
