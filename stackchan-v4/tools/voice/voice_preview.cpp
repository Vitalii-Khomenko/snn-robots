// Renders droid phrases to a WAV file (24 kHz mono) for listening and measurements.
//   voice_preview out.wav spec.txt
// spec.txt: one note per line, "baseHz kind shape ending durMs f0St slopeSt loud fm harm vib rate gapMs",
// an empty line ends a phrase (0.5 s of silence follows).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../main/voice/droid_synth.h"

int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    FILE* in = std::fopen(argv[2], "r");
    if (!in) return 1;
    droid::Synth synth;
    std::vector<int16_t> pcm;
    int16_t buf[480];
    auto pump = [&](bool all) {
        while (synth.queuedMs() > (all ? 0 : 100) || (all && synth.active())) {
            synth.render(buf, 480);
            pcm.insert(pcm.end(), buf, buf + 480);
        }
    };
    char line[256];
    while (std::fgets(line, sizeof line, in)) {
        if (line[0] == '\n' || line[0] == '\r') {
            pump(true);
            pcm.insert(pcm.end(), 12000, 0);
            continue;
        }
        float base, kind, shape, ending, dur, f0, slope, loud, fm, harm, vib, rate, gap;
        if (std::sscanf(line, "%f %f %f %f %f %f %f %f %f %f %f %f %f", &base, &kind, &shape, &ending, &dur, &f0, &slope, &loud, &fm,
                        &harm, &vib, &rate, &gap) != 13)
            continue;
        droid::Note n;
        n.kind = static_cast<droid::Kind>(static_cast<int>(kind));
        n.shape = static_cast<droid::Shape>(static_cast<int>(shape));
        n.ending = static_cast<droid::Ending>(static_cast<int>(ending));
        n.durMs = dur; n.f0St = f0; n.slopeSt = slope; n.loud = loud; n.fm = fm; n.harm = harm; n.vib = vib; n.rate = rate; n.gapMs = gap;
        pump(false);
        while (!synth.enqueue(n, base)) pump(false);
    }
    pump(true);
    pcm.insert(pcm.end(), 4800, 0);
    FILE* f = std::fopen(argv[1], "wb");
    if (!f) return 1;
    const uint32_t n = static_cast<uint32_t>(pcm.size() * 2), sr = 24000, br = 48000, fmt = 16;
    const uint16_t pcmfmt = 1, ch = 1, ba = 2, bits = 16;
    std::fwrite("RIFF", 1, 4, f);
    const uint32_t riff = 36 + n;
    std::fwrite(&riff, 4, 1, f);
    std::fwrite("WAVEfmt ", 1, 8, f);
    std::fwrite(&fmt, 4, 1, f);
    std::fwrite(&pcmfmt, 2, 1, f);
    std::fwrite(&ch, 2, 1, f);
    std::fwrite(&sr, 4, 1, f);
    std::fwrite(&br, 4, 1, f);
    std::fwrite(&ba, 2, 1, f);
    std::fwrite(&bits, 2, 1, f);
    std::fwrite("data", 1, 4, f);
    std::fwrite(&n, 4, 1, f);
    std::fwrite(pcm.data(), 2, pcm.size(), f);
    std::fclose(f);
    std::printf("%zu samples (%.2f s)\n", pcm.size(), pcm.size() / 24000.0);
    return 0;
}
