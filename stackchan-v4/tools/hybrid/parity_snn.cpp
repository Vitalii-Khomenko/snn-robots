// C++ <-> PyTorch parity for the spiking core: noise off, same input sequence.
#include <cmath>
#include <cstdio>
#include <vector>

#include "../../main/hybrid/spiking_core.h"

int main(int argc, char** argv)
{
    FILE* f = std::fopen(argc > 1 ? argv[1] : "data/parity_snn.bin", "rb");
    if (!f) return 1;
    int T = 0;
    std::fread(&T, 4, 1, f);
    std::vector<float> bus(static_cast<size_t>(T) * hybrid::kSnnInputs), exp(static_cast<size_t>(T) * hybrid::SNN_COUNT);
    std::fread(bus.data(), 4, bus.size(), f);
    std::fread(exp.data(), 4, exp.size(), f);
    SpikingCore core;
    core.setNoise(0.0f);
    SpikingCore::Values v{};
    double err[hybrid::SNN_COUNT] = {};
    double worst = 0;
    int n = 0;
    for (int t = 0; t < T; ++t) {
        core.tick(&bus[static_cast<size_t>(t) * hybrid::kSnnInputs], v);
        if (t < 200) continue;  // let both settle (spike flips early on do not matter)
        for (int k = 0; k < hybrid::SNN_COUNT; ++k) {
            const double d = std::fabs(v[k] - exp[static_cast<size_t>(t) * hybrid::SNN_COUNT + k]);
            err[k] += d;
            worst = std::max(worst, d);
        }
        ++n;
    }
    double mean = 0;
    for (int k = 0; k < hybrid::SNN_COUNT; ++k) {
        std::printf("%-14s mean |diff| %.4f\n", hybrid::kSnnNames[k], err[k] / n);
        mean += err[k] / n / hybrid::SNN_COUNT;
    }
    std::printf("mean %.4f worst %.3f, spikes/neuron/substep %.3f\n", mean, worst, core.meanRate());
    return mean < 0.02 ? 0 : 1;
}
