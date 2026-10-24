#include "vision_net.h"

#include <algorithm>
#include <cmath>
#include <iterator>

#include "nn_ops.h"
#include "vision_model_data.h"

bool VisionNet::trained() const { return vision_model::kTrained; }

VisionNet::Params VisionNet::compiledParams()
{
    using namespace vision_model;
    return {{{conv1_w, conv1_s, conv1_b}, {conv2_w, conv2_s, conv2_b}, {conv3_w, conv3_s, conv3_b}, {fc1_w, fc1_s, fc1_b},
             {fc2_w, fc2_s, fc2_b}}};
}

const LayerInfo* VisionNet::layerInfo()
{
    using namespace vision_model;
    static const LayerInfo info[kLayers] = {{"conv1", int(std::size(conv1_w)), int(std::size(conv1_b))},
                                            {"conv2", int(std::size(conv2_w)), int(std::size(conv2_b))},
                                            {"conv3", int(std::size(conv3_w)), int(std::size(conv3_b))},
                                            {"fc1", int(std::size(fc1_w)), int(std::size(fc1_b))},
                                            {"fc2", int(std::size(fc2_w)), int(std::size(fc2_b))}};
    return info;
}

bool VisionNet::run(const VisionFrontEnd::Frame& f, Features& out)
{
    using namespace hybrid;
    // exact photometric features need no network
    auto photometric = [&] {
        out.fill(0.0f);
        out[VIS_BRIGHTNESS] = f.brightness;
        out[VIS_LIGHT_CHANGE] = f.lightChange;
        out[VIS_LIGHT_X] = f.lightX;
        out[VIS_LIGHT_Y] = f.lightY;
        out[VIS_LIGHT_STRENGTH] = f.lightStrength;
    };
    photometric();
    if (!vision_model::kTrained) return false;
    const Params& p = params_[active_.load()];
    // max-pool(ReLU(conv)) fused, layer by layer; scale and bias are applied
    // after the pool, which is the same because both are monotone per channel
    // (the scale is positive).
    nn::conv3x3Pool(f.image.data(), 3, 24, 32, p[0].w, p[0].s, p[0].b, 8, p1_);
    nn::conv3x3Pool(p1_, 8, 12, 16, p[1].w, p[1].s, p[1].b, 16, p2_);
    nn::conv3x3Pool(p2_, 16, 6, 8, p[2].w, p[2].s, p[2].b, 24, feat_.data());  // 24 x 3 x 4
    for (int i = 0; i < VisionFrontEnd::kSide; ++i) feat_[24 * 3 * 4 + i] = f.side[i];
    nn::dense(p[3].w, p[3].s, p[3].b, feat_.data(), 64, 24 * 3 * 4 + VisionFrontEnd::kSide, hidden_.data(), true);
    nn::dense(p[4].w, p[4].s, p[4].b, hidden_.data(), kOutputs, 64, out_.data(), false);
    if (!nn::finite(out_.data(), kOutputs)) {
        ++nan_count_;
        photometric();
        return false;
    }
    for (int i = 0; i < kOutputs; ++i) {
        const bool signedOut = i == O_TARGET_X || i == O_TARGET_Y;
        out_[i] = signedOut ? std::tanh(out_[i]) : nn::sigmoid(out_[i]);
    }
    // Dead zone: a network never says exactly zero. Everything under 0.15 is
    // "nothing there", so a quiet room reads as quiet (a constant 0.1 "person"
    // would keep boredom and social need down for ever).
    auto live = [](float x) { return std::clamp((x - 0.15f) / 0.85f, 0.0f, 1.0f); };
    out_[O_MOTION_L] = live(out_[O_MOTION_L]);
    out_[O_MOTION_C] = live(out_[O_MOTION_C]);
    out_[O_MOTION_R] = live(out_[O_MOTION_R]);
    out_[O_MOTION_UP] = live(out_[O_MOTION_UP]);
    out_[O_MOTION_DOWN] = live(out_[O_MOTION_DOWN]);
    // a person / face is believed only above 0.3: a chair and a monitor give 0.2-0.5 in an empty room
    auto livePerson = [](float x) { return std::clamp((x - 0.30f) / 0.70f, 0.0f, 1.0f); };
    out_[O_PERSON] = livePerson(out_[O_PERSON]);
    out_[O_FACE] = livePerson(out_[O_FACE]);
    out_[O_LOOM] = live(out_[O_LOOM]);
    out_[O_RECEDE] = live(out_[O_RECEDE]);
    out_[O_TARGET_CONF] = live(out_[O_TARGET_CONF]);
    out_[O_ACTIVITY] = live(out_[O_ACTIVITY]);
    out[VIS_MOTION_L] = out_[O_MOTION_L];
    out[VIS_MOTION_C] = out_[O_MOTION_C];
    out[VIS_MOTION_R] = out_[O_MOTION_R];
    out[VIS_MOTION_UP] = out_[O_MOTION_UP];
    out[VIS_MOTION_DOWN] = out_[O_MOTION_DOWN];
    out[VIS_PERSON] = out_[O_PERSON];
    out[VIS_FACE] = out_[O_FACE];
    out[VIS_LOOM] = out_[O_LOOM];
    out[VIS_RECEDE] = out_[O_RECEDE];
    out[VIS_TARGET_X] = out_[O_TARGET_X];
    out[VIS_TARGET_Y] = out_[O_TARGET_Y];
    out[VIS_TARGET_CONF] = out_[O_TARGET_CONF];
    out[VIS_ACTIVITY] = out_[O_ACTIVITY];
    if (f.blind()) {  // too dark to see anything: no person, no target, no approach
        for (int i : {VIS_PERSON, VIS_FACE, VIS_LOOM, VIS_RECEDE, VIS_TARGET_CONF, VIS_TARGET_X, VIS_TARGET_Y})
            out[i] = 0.0f;
    }
    return true;
}
