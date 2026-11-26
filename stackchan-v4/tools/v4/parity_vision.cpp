#include <cstdio>
#include <cstdlib>

#include "vision_net.h"

int main()
{
    VisionNet net;
    if (!net.trained()) {
        std::fprintf(stderr, "VisionNet not trained\n");
        return 1;
    }
    constexpr size_t kImg = 3 * 24 * 32;
    float img[kImg], side[4];
    while (std::fread(img, sizeof(float), kImg, stdin) == kImg &&
           std::fread(side, sizeof(float), 4, stdin) == 4) {
        VisionNet::Out out;
        if (!net.infer(img, side, out)) return 2;
        const float o[5] = {out.person, out.face, out.targetX, out.targetY, out.targetConf};
        std::fwrite(o, sizeof(float), 5, stdout);
    }
    return 0;
}
