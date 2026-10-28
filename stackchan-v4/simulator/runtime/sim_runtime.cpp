#include "../../main/core/ms_clock.h"
#include "sim_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

#include <esp_log.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "../core/log.h"
#include "../core/vrtos.h"
#include "../hal/platform.h"
#include "../world/world.h"

namespace sim {
namespace {
constexpr const char* kTag = "runtime";
// Same constants as robot_runtime.cpp.
constexpr uint32_t kMainLoopIntervalMs = 20;
// WifiEngine model: association and NTP after boot (seen on hardware: wifi up ~9 s, time synced ~40 s).
constexpr uint32_t kWifiUpMs = 9000;
constexpr uint32_t kTimeSyncMs = 40000;
}  // namespace

SimRuntime::SimRuntime(const RunOptions& options)
    : opt_(options),
      loop_(BrainLoop::Engines{motion_, face_, led_, audio_, vocal_, imu_, mic_, presence_, camera_, power_status_, touch_}),
      cost_rng_(options.seed * 97u + 5u)
{
}

uint32_t SimRuntime::nowMs() const { return static_cast<uint32_t>(esp_timer_get_time() / 1000); }

// Mirrors RobotRuntime::begin().
bool SimRuntime::begin()
{
    start_ms_ = nowMs();  // 0, or near the wrap (--ms-offset)
    ESP_LOGI(kTag, "nvs");
    ESP_LOGI(kTag, "brain v4: spiking analyzer %s, main network %s, auditory NN %s",
             loop_.brain().analyzer().trained() ? "trained" : "UNTRAINED", loop_.brain().trained() ? "trained" : "UNTRAINED",
             loop_.auditory().trained() ? "trained" : "MISSING");
    if (opt_.learned) {  // as RobotRuntime::loadLearned
        const learned::View v{opt_.learned};
        const int n = loop_.useLearned(v);
        ESP_LOGI(kTag, "learned: version %u (%d arrays: %s%s)", static_cast<unsigned>(v.version()), n,
                 n >= 4 ? "action heads" : "innate heads", n % 4 == 1 ? ", zone shifts" : "");
    }
    ESP_LOGI(kTag, "screen");
    face_.begin();
    ESP_LOGI(kTag, "LEDs");
    led_.begin();
    face_.setEmotion(Emotion::Normal);
    led_.setBreathing(0, 25, 30);
    ESP_LOGI(kTag, "speaker");
    audio_.begin();
    vocal_.begin();
    ESP_LOGI(kTag, "servos");
    if (!motion_.begin()) ESP_LOGE(kTag, "motion begin failed");

    ESP_LOGI(kTag, "touch");
    if (touch_.begin()) touch_.onTouch([this]() { loop_.headTouch(nowMs()); });

    ESP_LOGI(kTag, "IMU");
    imu_.begin();

    ESP_LOGI(kTag, "microphone");
    if (mic_.begin()) {
        mic_.onLoudSound([this]() {
            float intensity = std::clamp((0.6f + static_cast<float>(mic_.lastLoudPeak()) / 20000.0f) / 1.5f, 0.4f, 1.0f);
            if (mic_.lastLoudDuringOwnMotion()) intensity *= 0.3f;
            loop_.loudSound(intensity);
            ESP_LOGI(kTag, "loud sound: peak=%d%s", mic_.lastLoudPeak(), mic_.lastLoudDuringOwnMotion() ? " own-motion" : "");
        });
    }

    ESP_LOGI(kTag, "presence");
    presence_.begin();

    ESP_LOGI(kTag, "camera");
    if (camera_.begin()) {
        camera_.onVision([this](const CameraEngine::VisionFrame& frame) { loop_.vision(frame, nowMs()); });
    }

    ESP_LOGI(kTag, "power status");
    power_status_.begin();
    ESP_LOGI(kTag, "wifi");

    const uint32_t now = nowMs();
    next_diagnostics_ms_ = now + 5000;
    next_trace_ms_ = now;
    ESP_LOGI(kTag, "RobotRuntime ready");
    return true;
}

// WifiEngine's observable behaviour: connect, NTP.
void SimRuntime::updateWifi(uint32_t now)
{
    if (!wifi_connected_ && msReached(now, kWifiUpMs + start_ms_)) {
        wifi_connected_ = true;
        ESP_LOGI("wifi", "got IP");
    }
    if (!time_synced_ && msReached(now, kTimeSyncMs + start_ms_) && !World::get().faultNtp()) {
        time_synced_ = true;
        ESP_LOGI("wifi", "time synced");
    }
}

// Mirrors RobotRuntime::logStats() (the brain line).
void SimRuntime::logStats(uint32_t now)
{
    const auto& c = loop_.channels();
    auto& b = loop_.brain();
    const auto& so = b.snn();
    const auto& ex = loop_.executor();
    ESP_LOGI(kTag, "v4: ch=%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f snn=rate%.3f val=%.2f aro=%.2f "
             "under=%.2f over=%.2f threat=%.2f nn=faults%u night=%d face=%s calls=%u nods=%u shakes=%u gestures=%u "
             "missed_ticks=%u",
             c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12],
             static_cast<double>(b.analyzer().mixRate()), so[v4::SO_VALENCE], so[v4::SO_AROUSAL], so[v4::SO_UNDER],
             so[v4::SO_OVER], so[v4::SO_THREAT], static_cast<unsigned>(b.nnFaults()), loop_.night() ? 1 : 0,
             emotionName(static_cast<Emotion>(ex.shownEmotion())), static_cast<unsigned>(ex.calls()),
             static_cast<unsigned>(ex.nods()), static_cast<unsigned>(ex.shakes()), static_cast<unsigned>(ex.gestures()),
             static_cast<unsigned>(loop_.missedTicks()));
    next_diagnostics_ms_ = now + 5000;
}

void SimRuntime::writeTrace(uint32_t now)
{
    if (!opt_.trace) return;
    World& w = World::get();
    const auto& so = loop_.brain().snn();
    const auto& c = loop_.channels();
    float cmd_yaw = 0, cmd_pitch = 0;
    motion_.getCurrentDeg(cmd_yaw, cmd_pitch);
    std::fprintf(opt_.trace, "%.1f,%.3f,%.1f,%.1f,%.1f,%.1f,%s,%d", now / 1000.0, w.hourOfDay(vrtos::nowUs()),
                 w.headYawDeg(), w.headPitchDeg(), cmd_yaw, cmd_pitch, emotionName(face_.emotion()),
                 vocal_.active() ? 1 : 0);
    for (int i = 0; i < v4::CH_COUNT; ++i) std::fprintf(opt_.trace, ",%.3f", c[i]);
    for (int i : {v4::SO_VALENCE, v4::SO_AROUSAL, v4::SO_UNDER, v4::SO_OVER, v4::SO_THREAT, v4::SO_SURPRISE,
                  v4::SO_ATT_X, v4::SO_ATT_Y, v4::SO_ATT_STRENGTH})
        std::fprintf(opt_.trace, ",%.3f", so[i]);
    std::fprintf(opt_.trace, "\n");
}

void SimRuntime::writeMinute(uint32_t now)
{
    World& w = World::get();
    const auto& counts = logCounts();
    auto count = [&](const std::string& key) {
        int n = 0;
        for (const auto& kv : counts)
            if (kv.first.rfind(key, 0) == 0) n += kv.second;
        return n;
    };
    if (opt_.summary) {
        std::fprintf(opt_.summary, "%3u | %5.2f | %5.0f | %4d | %4d | %4d | %4d | %4d | %4d | %4.0f%% | %4.0f | %4.0f | %5.2f\n",
                     minute_index_ + 1, w.hourOfDay(vrtos::nowUs()), w.travelDeg() - minute_travel_start_,
                     w.movementsStarted() - minute_moves_start_, count("v4: face:"), count("v4: call"),
                     count("v4: gesture:"), count("runtime: loud sound"),
                     count("runtime: head touch (") + count("runtime: screen touch"), loop_.body().pressure() * 100.0f,
                     w.headYawDeg(), w.headPitchDeg(), static_cast<double>(loop_.body().budget()));
        std::fflush(opt_.summary);
    }
    resetLogCounts();
    minute_travel_start_ = w.travelDeg();
    minute_moves_start_ = w.movementsStarted();
    ++minute_index_;
    next_minute_ms_ = now + 60000;
}

// Mirrors RobotRuntime::loop(), one iteration per pass until `until_us`.
void SimRuntime::run(uint64_t until_us)
{
    if (opt_.summary)
        std::fprintf(opt_.summary, "min |  hour | travel | moves | faces | calls | gestures | loud | touches | sleep_p | yaw | pitch | budget\n");
    if (opt_.trace) {
        std::fprintf(opt_.trace, "t_s,hour,head_yaw,head_pitch,cmd_yaw,cmd_pitch,face,sound");
        for (const char* n : v4::kChannelNames) std::fprintf(opt_.trace, ",%s", n);
        std::fprintf(opt_.trace, ",valence,arousal,under,over,threat,surprise,att_x,att_y,att_strength\n");
    }
    next_minute_ms_ = nowMs() + 60000;
    while (vrtos::nowUs() < until_us) {
        const uint32_t now = nowMs();
        motion_.update();
        face_.update();
        led_.update();
        if (face_.takeScreenTouch()) {
            loop_.screenTouch(now);
            ESP_LOGI(kTag, "screen touch");
        }
        touch_.update();
        loop_.coupleMotion(now);
        mic_.update();
        presence_.update();
        camera_.update();
        power_status_.update();
        // I2C polls of touch/IMU/ALS/PMIC and the rest of the pass.
        vrtos::busyFor(static_cast<uint64_t>(opt_.loop_cost_ms * 1000.0f));
        updateWifi(now);
        BrainLoop::Env env;
        env.timeSynced = time_synced_;
        env.hour = time_synced_ ? World::get().hourOfDay(vrtos::nowUs()) : 12.0f;
        const uint32_t ticks_before = loop_.tickCount();
        const auto host_start = std::chrono::steady_clock::now();
        loop_.step(now, env);
        if (loop_.tickCount() != ticks_before) {
            // what a person in the scenario may answer (World "responder")
            if (loop_.saidThisTick()) World::get().robotActed(0);
            if (loop_.gestureThisTick()) World::get().robotActed(1);
            const int face = loop_.executor().shownEmotion();
            if (last_face_ >= 0 && face != last_face_) World::get().robotActed(2);
            last_face_ = face;
            host_tick_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - host_start).count();
            ++host_ticks_;
            // The tick costs ESP32-S3 CPU time: the main task stays busy for it while the camera/mic tasks keep running.
            std::uniform_real_distribution<float> jitter(0.7f, 1.3f);
            vrtos::busyFor(static_cast<uint64_t>(opt_.brain_cost_ms * jitter(cost_rng_) * 1000.0f));
            if (opt_.v4_dump) {
                const auto& b = loop_.brain();
                std::fwrite(loop_.channels().data(), sizeof(float), loop_.channels().size(), opt_.v4_dump);
                std::fwrite(b.snn().data(), sizeof(float), b.snn().size(), opt_.v4_dump);
                std::fwrite(b.nn().data(), sizeof(float), b.nn().size(), opt_.v4_dump);
            }
        }
        loop_.afterVoice(now);

        if (msReached(now, next_diagnostics_ms_)) logStats(now);
        if (msReached(now, next_trace_ms_)) {
            writeTrace(now);
            next_trace_ms_ = now + 100;
        }
        if (msReached(now, next_minute_ms_)) writeMinute(now);
        vTaskDelay(pdMS_TO_TICKS(kMainLoopIntervalMs));
    }
}

void SimRuntime::finish()
{
    // Wall-clock, so only on stdout: the log must stay bit-reproducible.
    std::printf("host brain tick (ear + channels + SNN + NN + executor) mean %.1f us over %llu ticks\n",
                host_ticks_ ? host_tick_s_ / host_ticks_ * 1e6 : 0.0, static_cast<unsigned long long>(host_ticks_));
}

}  // namespace sim
