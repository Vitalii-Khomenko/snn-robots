// The Muse's C++ twin over a stream: snn [T, 32] + noise [T, 64] (float32) -> outputs [T, 8]. tools/v4/parity_muse.py
#include <cstdio>

#include "muse_v4.h"

int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    FILE* f = std::fopen(argv[1], "rb");
    FILE* g = std::fopen(argv[2], "wb");
    if (!f || !g) return 1;
    MuseV4 m;
    if (argc > 3) {
        float b[muse_data::kOuts];
        FILE* bf = std::fopen(argv[3], "rb");
        if (!bf || std::fread(b, sizeof(float), muse_data::kOuts, bf) != muse_data::kOuts) return 1;
        std::fclose(bf);
        m.setBias(b);
    }
    float row[v4::SO_COUNT + MuseV4::kN];
    MuseV4::Out out;
    while (std::fread(row, sizeof(float), v4::SO_COUNT + MuseV4::kN, f) == v4::SO_COUNT + MuseV4::kN) {
        m.step(row, row + v4::SO_COUNT, out);
        std::fwrite(out.data(), sizeof(float), out.size(), g);
    }
    std::fclose(f);
    std::fclose(g);
    return 0;
}
