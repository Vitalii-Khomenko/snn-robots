#include <cstdio>
#include <cstdlib>

#include "auditory_net.h"

int64_t esp_timer_get_time() { return 0; }

int main()
{
    AuditoryNet net;
    if (!net.trained()) {
        std::fprintf(stderr, "AuditoryNet not trained\n");
        return 1;
    }
    float in[AudioFrontEnd::kInputs];
    while (std::fread(in, sizeof(float), AudioFrontEnd::kInputs, stdin) == AudioFrontEnd::kInputs) {
        if (!net.infer(in)) return 2;
        const auto& out = net.lastOutputs();
        std::fwrite(out.data(), sizeof(float), AuditoryNet::kOutputs, stdout);
    }
    return 0;
}
