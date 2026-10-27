// The contingency detector's C++ twin over a stream: per tick uint32 ms + float32 inputs [12] (tools/v4/contingency.py
// INPUTS) -> outputs [6] (float32). tools/v4/parity_contingency.py
#include <cstdint>
#include <cstdio>

#include "contingency_v4.h"

int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    FILE* f = std::fopen(argv[1], "rb");
    FILE* g = std::fopen(argv[2], "wb");
    if (!f || !g) return 1;
    ContingencyV4 det;
    uint32_t ms;
    float r[12];
    ContingencyV4::Out out;
    ContingencyV4::Credit credit;
    while (std::fread(&ms, sizeof(ms), 1, f) == 1 && std::fread(r, sizeof(float), 12, f) == 12) {
        if (r[0] < 0) {  // a new boot: a fresh detector
            det = ContingencyV4{};
            continue;
        }
        ContingencyV4::In in;
        in.ms = ms;
        in.active = r[0] > 0;
        in.said = r[1] > 0;
        in.gesture = r[2] > 0;
        in.face = static_cast<int>(r[3]);
        in.headEv = r[4];
        in.screenEv = r[5];
        in.touchHead = r[6];
        in.touchScreen = r[7];
        in.nearRaw = r[8];
        in.speech = r[9];
        in.motion = r[10];
        in.headYaw = r[11];
        det.step(in, out, credit);
        std::fwrite(out.data(), sizeof(float), out.size(), g);
        std::fwrite(&credit.any, 1, 1, g);
    }
    std::fclose(f);
    std::fclose(g);
    return 0;
}
