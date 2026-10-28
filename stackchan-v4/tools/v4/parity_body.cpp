// BodyV4 over a stream: per tick float32 [16] (tools/v4/body.py COLUMNS; dt < 0 marks a new boot; a finite off_s a
// restore before the tick) -> pressure, load, budget (float32). tools/v4/parity_body.py
#include <cmath>
#include <cstdio>

#include "body_v4.h"

int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    FILE* f = std::fopen(argv[1], "rb");
    FILE* g = std::fopen(argv[2], "wb");
    if (!f || !g) return 1;
    BodyV4 body;
    float r[16];
    while (std::fread(r, sizeof(float), 16, f) == 16) {
        if (r[0] < 0) {  // a new boot: a fresh body
            body = BodyV4{};
            continue;
        }
        if (std::isfinite(r[15])) body.restore({r[13], r[14]}, r[15]);
        BodyV4::In in;
        in.dtS = r[0];
        in.clockOk = r[1] > 0;
        in.hour = r[2];
        in.movedDeg = r[3];
        in.said = r[4] > 0;
        in.arousal = r[5];
        in.night = r[6] > 0;
        in.drowsy = r[7] > 0;
        in.threat = r[8];
        in.over = r[9];
        in.pet = r[10] > 0;
        in.nearby = r[11] > 0;
        in.hurt = r[12] > 0;
        body.step(in);
        const float out[3] = {body.pressure(), body.load(), body.budget()};
        std::fwrite(out, sizeof(float), 3, g);
    }
    std::fclose(f);
    std::fclose(g);
    return 0;
}
