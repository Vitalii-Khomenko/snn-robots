#pragma once
#include <cstdint>

// The weights of a small INT8 layer (the Auditory NN): int8 weights, a float scale per output, float biases.
struct LayerParams {
    const int8_t* w;
    const float* s;  // per output channel scale
    const float* b;
};
