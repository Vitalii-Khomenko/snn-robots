// Host build of main/v4/nn_v4.cpp for tools/v4/parity_nn_v4.py: input rows (44) in, outputs (61) out.
#include <cstdio>
#include "../../main/v4/nn_v4.h"
int main(int argc, char** argv)
{
    if (argc < 3) return 1;
    FILE* f = std::fopen(argv[1], "rb");
    FILE* g = std::fopen(argv[2], "wb");
    NnV4 nn;
    NnV4::In in;
    NnV4::Out out{};
    while (std::fread(in.data(), sizeof(float), in.size(), f) == in.size()) {
        nn.step(in, out);
        std::fwrite(out.data(), sizeof(float), out.size(), g);
    }
    std::fclose(f);
    std::fclose(g);
    return 0;
}
