#include "vision_net.h"

#include <algorithm>
#include <cmath>

#include "nn_ops.h"
#include "vision_model_data.h"

bool VisionNet::trained() const { return vision_model::kTrained; }

bool VisionNet::infer(const float* image, const float* side, Out& out)
{
    if (!vision_model::kTrained) return false;
    using namespace vision_model;
    nn::conv3x3Pool(image, 3, 24, 32, conv1_w, conv1_s, conv1_b, 8, c1_.data());
    nn::conv3x3Pool(c1_.data(), 8, 12, 16, conv2_w, conv2_s, conv2_b, 16, c2_.data());
    nn::conv3x3Pool(c2_.data(), 16, 6, 8, conv3_w, conv3_s, conv3_b, 24, c3_.data());
    for (int i = 0; i < 4; ++i) c3_[24 * 3 * 4 + i] = side[i];
    nn::dense(fc1_w, fc1_s, fc1_b, c3_.data(), 64, 24 * 3 * 4 + 4, fc1_.data(), true);
    nn::dense(fc2_w, fc2_s, fc2_b, fc1_.data(), kOutputs, 64, raw_.data(), false);
    if (!nn::finite(raw_.data(), kOutputs)) {
        ++nan_count_;
        return false;
    }
    out.person = nn::sigmoid(raw_[0]);
    out.face = nn::sigmoid(raw_[1]);
    out.targetX = std::tanh(raw_[2]);
    out.targetY = std::tanh(raw_[3]);
    out.targetConf = nn::sigmoid(raw_[4]);
    return true;
}

bool VisionNet::run(const VisionFrontEnd::Frame& frame, Out& out)
{
    out = {};
    if (frame.blind()) return true;
    return infer(frame.image.data(), frame.side.data(), out);
}
