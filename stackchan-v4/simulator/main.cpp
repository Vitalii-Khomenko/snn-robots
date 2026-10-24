// StackChan digital twin: runs the real firmware (the ear, the 13 channels, brain v4's spiking analyzer and main
// network, the executor, motion, voice, sensor engines) in a simulated room.
//
//   simulator/run.sh scenarios/alone_dark_morning.scn [options]
//
// Options:
//   --out DIR          output folder (default simulator/out/<scenario>)
//   --minutes N        override the scenario duration
//   --seed N           override the scenario seed
//   --echo             also print the log to the terminal
//   --voice-wav F      write everything the speaker plays (raw int16, 24 kHz) to F
//   --brain-cost MS    modelled brain-tick CPU time on the ESP32-S3 (default 4)
//   --v4-dump F        every brain tick: brain v4's channels, analyzer and main-network outputs (float32 stream)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

#include "core/log.h"
#include "core/vrtos.h"
#include "hal/platform.h"
#include "runtime/sim_runtime.h"
#include "world/world.h"

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s SCENARIO.scn [--out DIR] [--minutes N] [--seed N] [--echo] "
                             "[--voice-wav F] [--brain-cost MS] [--v4-dump F]\n", argv[0]);
        return 2;
    }
    sim::Scenario scenario;
    try {
        scenario = sim::loadScenario(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
    std::string out_dir;
    sim::RunOptions options;
    bool echo = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--out") out_dir = value();
        else if (a == "--minutes") scenario.duration_us = static_cast<uint64_t>(std::stod(value()) * 60e6);
        else if (a == "--seed") scenario.seed = static_cast<uint32_t>(std::stoul(value()));
        else if (a == "--echo") echo = true;
        else if (a == "--voice-wav") {
            g_voice_dump = std::fopen(value().c_str(), "wb");
            if (!g_voice_dump) return 1;
        } else if (a == "--brain-cost") options.brain_cost_ms = std::stof(value());
        else if (a == "--v4") {
            // brain v4 is the only brain now; accepted so older command lines still run
        } else if (a == "--v4-dump") {
            options.v4_dump = std::fopen(value().c_str(), "wb");
            if (!options.v4_dump) return 1;
        }
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (out_dir.empty()) out_dir = std::string(SIM_DEFAULT_OUT) + "/" + scenario.name;
    std::filesystem::create_directories(out_dir);

    FILE* log = std::fopen((out_dir + "/log.txt").c_str(), "w");
    options.trace = std::fopen((out_dir + "/trace.csv").c_str(), "w");
    options.summary = std::fopen((out_dir + "/summary.txt").c_str(), "w");
    options.loop_cost_ms = scenario.loop_cost_ms;
    options.seed = scenario.seed;
    if (!log || !options.trace || !options.summary) {
        std::fprintf(stderr, "cannot write to %s\n", out_dir.c_str());
        return 1;
    }
    sim::setLogFile(log);
    sim::setLogEcho(echo);
    sim::seedPlatformRandom(scenario.seed);
    sim::World::get().load(scenario);

    std::printf("scenario %s: %.1f min, seed %u -> %s\n", scenario.name.c_str(), scenario.duration_us / 60e6,
                scenario.seed, out_dir.c_str());
    std::fflush(stdout);
    auto runtime = std::make_unique<sim::SimRuntime>(options);
    vrtos::start([&] {
        vrtos::sleepFor(sim::kBootUs);
        runtime->begin();
        runtime->run(scenario.duration_us);
        runtime->finish();
    });
    std::fclose(log);
    std::fclose(options.trace);
    std::fclose(options.summary);
    if (options.v4_dump) std::fclose(options.v4_dump);
    if (g_voice_dump) std::fclose(g_voice_dump);
    std::printf("done: %s/summary.txt, trace.csv, log.txt\n", out_dir.c_str());
    std::fflush(stdout);
    // Firmware tasks (mic, camera) never end; leave without joining them.
    std::_Exit(0);
}
