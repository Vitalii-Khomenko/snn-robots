// Host build of main/v4/snn_v4.cpp for tools/v4/parity_snn_v4.py: channel rows in, the 32 outputs out.
#include <cstdio>
#include "../../main/v4/snn_v4.h"
int main(int argc, char** argv)
{
    if (argc < 3) return 1;
    FILE* f = std::fopen(argv[1], "rb");
    FILE* g = std::fopen(argv[2], "wb");
    static SnnV4 snn;
    SnnV4::In in;
    SnnV4::Out out{};
    while (std::fread(in.data(), sizeof(float), in.size(), f) == in.size()) {
        snn.tick(in, out);
        std::fwrite(out.data(), sizeof(float), out.size(), g);
    }
    std::fclose(f);
    std::fclose(g);
    return 0;
}
