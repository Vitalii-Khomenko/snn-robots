// Host build of main/v4/channel_encoder.cpp for tools/v4/parity_channels.py: raw rows in (float32), channels out.
#include <cstdio>
#include <vector>
#include "../../main/v4/channel_encoder.h"
int main(int argc, char** argv)
{
    if (argc < 3) return 1;
    FILE* f = std::fopen(argv[1], "rb");
    FILE* g = std::fopen(argv[2], "wb");
    float row[22];
    ChannelEncoder enc;
    ChannelEncoder::Out out{};
    while (std::fread(row, sizeof(float), 22, f) == 22) {
        ChannelEncoder::Raw r;
        r.energy = row[0]; r.speech = row[1]; r.sharpEv = row[2]; r.headEv = row[3]; r.headHold = row[4];
        r.screenEv = row[5]; r.accelSd = row[6]; r.imuPitch = row[7]; r.imuRoll = row[8]; r.imuOk = row[9] > 0;
        r.nearRaw = row[10]; r.nearOk = row[11] > 0; r.als = row[12]; r.alsOk = row[13] > 0; r.motionFrac = row[14];
        r.motionX = row[15]; r.motionY = row[16]; r.frameOk = row[17] > 0; r.battery = row[18]; r.stamina = row[19];
        r.hour = row[20]; r.clockOk = row[21] > 0;
        enc.update(r, out);
        std::fwrite(out.data(), sizeof(float), out.size(), g);
    }
    std::fclose(f);
    std::fclose(g);
    return 0;
}
