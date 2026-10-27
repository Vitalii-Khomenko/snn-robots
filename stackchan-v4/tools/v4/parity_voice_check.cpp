// VoiceCheck's C++ twin over a mono int16 stream -> per 25 ms frame: voice, periodicity, level (float32).
// tools/v4/parity_voice_check.py
#include <cstdint>
#include <cstdio>

#include "voice_check.h"

int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    FILE* f = std::fopen(argv[1], "rb");
    FILE* g = std::fopen(argv[2], "wb");
    if (!f || !g) return 1;
    static VoiceCheck vc;
    int16_t s;
    uint32_t n = 0;
    while (std::fread(&s, sizeof(s), 1, f) == 1) {
        vc.feed(&s, 1);
        if (++n % (VoiceCheck::kDecim * VoiceCheck::kHop) == 0 &&
            n >= static_cast<uint32_t>(VoiceCheck::kDecim * (VoiceCheck::kWin + VoiceCheck::kLagHi))) {
            const float row[3] = {vc.voice(), vc.periodicity(), vc.levelDb()};
            std::fwrite(row, sizeof(float), 3, g);
        }
    }
    std::fclose(f);
    std::fclose(g);
    return 0;
}
