// The sleep pressure's C++ twin over a stream: per tick float32 [7] (tools/v4/body.py inputs: dt, clock_ok, hour,
// moved_deg, said, arousal, asleep; dt < 0 marks a new boot) -> pressure, load (float32). tools/v4/parity_body.py
#include <cstdio>

#include "body_v4.h"

int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    FILE* f = std::fopen(argv[1], "rb");
    FILE* g = std::fopen(argv[2], "wb");
    if (!f || !g) return 1;
    BodyV4 body;
    float r[7];
    while (std::fread(r, sizeof(float), 7, f) == 7) {
        if (r[0] < 0) {  // a new boot: a fresh body
            body = BodyV4{};
            continue;
        }
        BodyV4::In in;
        in.dtS = r[0];
        in.clockOk = r[1] > 0;
        in.hour = r[2];
        in.movedDeg = r[3];
        in.said = r[4] > 0;
        in.arousal = r[5];
        in.asleep = r[6] > 0;
        body.step(in);
        const float out[2] = {body.pressure(), body.load()};
        std::fwrite(out, sizeof(float), 2, g);
    }
    std::fclose(f);
    std::fclose(g);
    return 0;
}
