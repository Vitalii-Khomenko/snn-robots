// Training data for the perception networks, from the digital twin.
//
//   stackchan_datagen vision|audio SCENARIO.scn --out FILE [--minutes N] [--seed N]
//
// The room, people, objects and sounds come from the scenario; the head is
// moved at random (like a robot looking around, plus stillness) so the
// networks also learn ego-motion and the robot's own servo noise. Inputs are
// produced by the REAL firmware front ends (hybrid/vision_frontend,
// hybrid/audio_frontend), labels by the world's ground truth.
//
// vision record (per camera frame): int8 image[3*768] (value*60), float side[4],
//   float label[13] = VisionNet output order (motion l/c/r/up/down, person, face,
//   loom, recede, target x/y/conf, activity), float head yaw/pitch speed
// audio record (per 50 ms): float input[100], float label[9]
//   (silence, speech, clap, sharp, continuous, low, high, tonal, direction)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../../main/hybrid/audio_frontend.h"
#include "../../main/hybrid/vision_frontend.h"
#include "../../main/motion/motion_engine.h"
#include "../core/log.h"
#include "../core/vrtos.h"
#include "../hal/platform.h"
#include "../world/world.h"

namespace {
using sim::World;

struct HeadWalker {
    std::mt19937 rng;
    MotionEngine motion;
    uint64_t next_ms = 0;
    explicit HeadWalker(uint32_t seed) : rng(seed * 31u + 7u) {}
    void update(uint64_t now_ms)
    {
        motion.update();
        if (now_ms < next_ms) return;
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        const float r = u(rng);
        if (r < 0.35f) {  // hold still
            next_ms = now_ms + 500 + static_cast<uint64_t>(u(rng) * 4000);
            return;
        }
        float yaw, pitch;
        motion.getCurrentDeg(yaw, pitch);
        const float step = r < 0.7f ? 15.0f : 60.0f;
        const float ty = std::clamp(yaw + (u(rng) * 2 - 1) * step, -100.0f, 100.0f);
        const float tp = std::clamp(pitch + (u(rng) * 2 - 1) * step * 0.4f, -20.0f, 35.0f);
        static const float speeds[] = {8.0f, 25.0f, 25.0f, 60.0f, 120.0f};
        const float speed = speeds[rng() % 5];
        motion.moveTo(ty, tp, speed);
        const float dist = std::hypot(ty - yaw, tp - pitch);
        next_ms = now_ms + static_cast<uint64_t>(dist / speed * 1000.0f) + 300 + static_cast<uint64_t>(u(rng) * 2500);
    }
    bool moving() const { return motion.isMoving(); }
};

void runVision(FILE* out, uint64_t duration_us, uint32_t seed)
{
    World& w = World::get();
    HeadWalker head(seed);
    head.motion.begin();
    VisionFrontEnd front;
    auto frame = std::make_unique<VisionFrontEnd::Frame>();
    std::vector<uint8_t> rgb(160 * 120 * 2);
    w.lockExposure(vrtos::nowUs());
    uint64_t next_frame = vrtos::nowUs();
    uint64_t count = 0;
    while (vrtos::nowUs() < duration_us) {
        const uint64_t now = vrtos::nowUs();
        head.update(now / 1000);
        if (now >= next_frame) {
            next_frame = now + 143000;
            w.renderFrame(rgb.data(), 160, 120, now);
            const World::VisionTruth t = w.visionTruth(now);
            if (front.process(rgb.data(), 160, 120, *frame) && t.valid) {
                int8_t q[3 * 768];
                for (int i = 0; i < 3 * 768; ++i)
                    q[i] = static_cast<int8_t>(std::clamp(std::lround(frame->image[i] * 60.0f), -127L, 127L));
                // in the dark nothing can be seen, whoever is there
                const bool blind = frame->blind();
                float label[13] = {t.motion_l, t.motion_c, t.motion_r, t.motion_up, t.motion_down, blind ? 0.0f : t.person, blind ? 0.0f : t.face,
                                   blind ? 0.0f : t.loom, blind ? 0.0f : t.recede, blind ? 0.0f : t.target_x,
                                   blind ? 0.0f : t.target_y, blind ? 0.0f : t.target_conf, t.activity};
                float vel[2];
                head.motion.getVelocityDegS(vel[0], vel[1]);
                std::fwrite(q, 1, sizeof(q), out);
                std::fwrite(frame->side.data(), sizeof(float), 4, out);
                std::fwrite(label, sizeof(float), 13, out);
                std::fwrite(vel, sizeof(float), 2, out);
                ++count;
            }
        }
        vrtos::sleepFor(20000);
    }
    std::printf("vision: %llu frames\n", static_cast<unsigned long long>(count));
}

void runAudio(FILE* out, uint64_t duration_us, uint32_t seed)
{
    World& w = World::get();
    HeadWalker head(seed);
    head.motion.begin();
    AudioFrontEnd front;
    std::mt19937 rng(seed * 977u + 3u);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    int16_t block[128 * 2];
    double t_us = static_cast<double>(vrtos::nowUs());
    uint64_t next_tick = vrtos::nowUs() + 300000;
    uint64_t own_until = 0, self_until = 0, next_self = vrtos::nowUs() + 2000000;
    uint64_t moving_until = 0;
    uint64_t count = 0;
    while (vrtos::nowUs() < duration_us) {
        t_us += 128 * 1e6 / 24000.0;
        vrtos::sleepUntil(static_cast<uint64_t>(t_us));
        const uint64_t now = vrtos::nowUs();
        head.update(now / 1000);
        if (head.moving()) moving_until = now + 350000;
        // The robot's own voice: a chirp or two now and then.
        if (now >= next_self) {
            const int n = 2400 + static_cast<int>(u(rng) * 9600);  // 0.1 .. 0.5 s
            std::vector<int16_t> pcm(static_cast<size_t>(n));
            double phase = 0.0;
            const float f0 = 500.0f + u(rng) * 1500.0f, f1 = f0 * (0.6f + u(rng) * 1.2f);
            const float amp = 3000.0f + u(rng) * 9000.0f;
            for (int i = 0; i < n; ++i) {
                const float f = f0 + (f1 - f0) * i / n;
                phase += 2 * M_PI * f / 24000.0;
                const float env = std::min({1.0f, i / 300.0f, (n - i) / 600.0f});
                pcm[static_cast<size_t>(i)] = static_cast<int16_t>(amp * env * (std::sin(phase) + 0.3f * std::sin(2 * phase)));
            }
            w.speakerPush(now, pcm.data(), pcm.size(), 80);
            self_until = now + static_cast<uint64_t>(n * 1e6 / 24000.0) + 300000;
            next_self = now + 3000000 + static_cast<uint64_t>(u(rng) * 25000000.0);
        }
        w.micRead(block, 128, now);
        front.processBlock(block, 128);
        if (now >= next_tick) {
            next_tick += 50000;
            std::array<AudioFrontEnd::Frame, AudioFrontEnd::kSpan> frames;
            if (!front.latest(frames)) continue;
            const bool self = now < self_until, own = now < moving_until;
            std::array<float, AudioFrontEnd::kInputs> x;
            AudioFrontEnd::buildInput(frames, self, own, x);
            const World::AudioTruth t = w.audioTruth(now);
            // Self sound / own servos are not the world: external silence.
            const bool ext = t.speech || t.clap || t.sharp || t.continuous || t.low || t.high || t.tonal;
            float y[9] = {ext ? 0.0f : 1.0f, t.speech ? 1.0f : 0.0f, t.clap ? 1.0f : 0.0f, t.sharp ? 1.0f : 0.0f,
                          t.continuous ? 1.0f : 0.0f, t.low ? 1.0f : 0.0f, t.high ? 1.0f : 0.0f, t.tonal ? 1.0f : 0.0f,
                          t.direction};
            std::fwrite(x.data(), sizeof(float), x.size(), out);
            std::fwrite(y, sizeof(float), 9, out);
            ++count;
        }
    }
    std::printf("audio: %llu records\n", static_cast<unsigned long long>(count));
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s vision|audio SCENARIO.scn --out FILE [--minutes N] [--seed N]\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    sim::Scenario scenario;
    try {
        scenario = sim::loadScenario(argv[2]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
    std::string out_path;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--minutes" && i + 1 < argc) scenario.duration_us = static_cast<uint64_t>(std::stod(argv[++i]) * 60e6);
        else if (a == "--seed" && i + 1 < argc) scenario.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
    }
    FILE* out = std::fopen(out_path.c_str(), "wb");
    if (!out) return 1;
    FILE* log = std::fopen("/dev/null", "w");
    sim::setLogFile(log);
    sim::setLogEcho(false);
    sim::seedPlatformRandom(scenario.seed);
    World::get().load(scenario);
    const uint64_t duration = scenario.duration_us;
    const uint32_t seed = scenario.seed;
    vrtos::start([&] {
        if (mode == "vision") runVision(out, duration, seed);
        else runAudio(out, duration, seed);
        std::fflush(out);
    });
    std::fclose(out);
    std::_Exit(0);
}
