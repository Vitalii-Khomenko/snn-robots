#pragma once
#include <cstdint>

// The weights of a small INT8 net, layer by layer: what the Vision and Auditory nets read instead of their compiled-in
// arrays, so the nightly retraining on the robot's own frames and sounds (tools/life/train_perception.py) reaches them
// through the weights blob (net/weight_store.h) like the actor's.
struct LayerParams {
    const int8_t* w;
    const float* s;  // per output channel scale
    const float* b;
};
struct LayerInfo {
    const char* name;  // "<name>_w", "<name>_s", "<name>_b" in the blob, after "<net>."
    int weights, outputs;
};
