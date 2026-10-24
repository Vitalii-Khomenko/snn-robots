// Runs the firmware's Vision front end over recorded frame pairs (real photos
// prepared by tools/hybrid/make_coco_vision.py).
//   frames_feed IN.bin OUT.bin
// IN : records of [previous RGB565 160x120][current RGB565 160x120] (38400 bytes each)
// OUT: records of [int8 image 3*768 (value*60)][float side[4]][float contrast]
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "../../main/hybrid/vision_frontend.h"

int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    FILE* in = std::fopen(argv[1], "rb");
    FILE* out = std::fopen(argv[2], "wb");
    if (!in || !out) return 1;
    std::vector<uint8_t> prev(38400), cur(38400);
    auto frame = std::make_unique<VisionFrontEnd::Frame>();
    long n = 0;
    while (std::fread(prev.data(), 1, prev.size(), in) == prev.size() && std::fread(cur.data(), 1, cur.size(), in) == cur.size()) {
        VisionFrontEnd front;  // fresh reference per pair
        front.process(prev.data(), 160, 120, *frame);
        if (!front.process(cur.data(), 160, 120, *frame)) continue;
        int8_t q[3 * 768];
        for (int i = 0; i < 3 * 768; ++i) {
            long v = std::lround(frame->image[i] * 60.0f);
            q[i] = static_cast<int8_t>(v < -127 ? -127 : v > 127 ? 127 : v);
        }
        std::fwrite(q, 1, sizeof(q), out);
        std::fwrite(frame->side.data(), sizeof(float), 4, out);
        std::fwrite(&frame->contrast, sizeof(float), 1, out);
        ++n;
    }
    std::fclose(out);
    std::printf("%ld samples\n", n);
    return 0;
}
