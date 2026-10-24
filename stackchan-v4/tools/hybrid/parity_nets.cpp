// C++ <-> PyTorch parity for the perception nets: reads tools/hybrid/data/parity.bin
// (written by parity.py) and compares VisionNet / AuditoryNet outputs.
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "../../main/hybrid/auditory_net.h"
#include "../../main/hybrid/vision_net.h"
#include <esp_timer.h>

int64_t esp_timer_get_time() { return 0; }

int main(int argc, char** argv)
{
    FILE* f = std::fopen(argc > 1 ? argv[1] : "data/parity.bin", "rb");
    if (!f) return 1;
    int n_vis = 0, n_aud = 0;
    std::fread(&n_vis, 4, 1, f);
    std::fread(&n_aud, 4, 1, f);
    VisionNet vision;
    auto frame = std::make_unique<VisionFrontEnd::Frame>();
    VisionNet::Features feat;
    double worst_v = 0;
    for (int i = 0; i < n_vis; ++i) {
        std::vector<float> img(3 * 768), side(4), exp(13);
        std::fread(img.data(), 4, img.size(), f);
        std::fread(side.data(), 4, 4, f);
        std::fread(exp.data(), 4, 13, f);
        for (int k = 0; k < 3 * 768; ++k) frame->image[k] = img[k];
        for (int k = 0; k < 4; ++k) frame->side[k] = side[k];
        vision.run(*frame, feat);
        for (int k = 0; k < 13; ++k) worst_v = std::max(worst_v, std::fabs(double(vision.lastOutputs()[k]) - exp[k]));
    }
    AudioFrontEnd front;
    AuditoryNet net;
    double worst_a = 0;
    for (int i = 0; i < n_aud; ++i) {
        std::vector<float> x(AudioFrontEnd::kInputs), exp(AuditoryNet::kOutputs);
        std::fread(x.data(), 4, x.size(), f);
        std::fread(exp.data(), 4, AuditoryNet::kOutputs, f);
        net.infer(x.data());
        for (int k = 0; k < AuditoryNet::kOutputs; ++k) worst_a = std::max(worst_a, std::fabs(double(net.lastOutputs()[k]) - exp[k]));
    }
    std::printf("vision max |diff| %.2e over %d, audio max |diff| %.2e over %d\n", worst_v, n_vis, worst_a, n_aud);
    return (worst_v < 1e-3 && worst_a < 1e-3) ? 0 : 1;
}
