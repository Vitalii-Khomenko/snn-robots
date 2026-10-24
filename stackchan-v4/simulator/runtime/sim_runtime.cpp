#include "../../main/core/ms_clock.h"
#include "sim_runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <vector>

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
// WifiEngine model: association and NTP after boot (seen on hardware:
// wifi up ~9 s, time synced ~40 s), owner ping every 20 minutes.
constexpr uint32_t kWifiUpMs = 9000;
constexpr uint32_t kTimeSyncMs = 40000;
constexpr uint32_t kPresenceCheckIntervalMs = 20 * 60 * 1000;
}  // namespace

SimRuntime::SimRuntime(const RunOptions& options)
    : opt_(options),
      loop_(BrainLoop::Engines{motion_, face_, led_, audio_, vocal_, imu_, mic_, presence_, camera_, power_status_, touch_}),
      cost_rng_(options.seed * 97u + 5u)
{
    if (opt_.policy) loop_.brain().setExternalPolicy(opt_.policy);
    if (opt_.v4) {  // as RobotRuntime::loadMode for mode v4
        loop_.brain().setExternalPolicy([](const PolicyGru::Input&, PolicyGru::Action& a) { a = PolicyGru::neutral(); });
        loop_.brain().setSpikingCore(false);
        loop_.setV4Active(true);
    }
}

uint32_t SimRuntime::nowMs() const { return static_cast<uint32_t>(esp_timer_get_time() / 1000); }

// Mirrors RobotRuntime::begin().
bool SimRuntime::begin()
{
    ESP_LOGI(kTag, "nvs");
    ESP_LOGI(kTag, "hybrid brain: SNN %d neurons / %d synapses, policy GRU, vision NN %s, auditory NN %s",
             SpikingCore::kNeurons, loop_.brain().snn().synapseCount(), camera_.net().trained() ? "trained" : "MISSING",
             loop_.auditory().trained() ? "trained" : "MISSING");
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
    if (imu_.begin()) {
        imu_.onShake([this]() {
            loop_.shake(nowMs());
            ESP_LOGI(kTag, "shake detected");
        });
        imu_.onTilt([this]() {
            loop_.tilt(nowMs());
            ESP_LOGI(kTag, "tilt detected");
        });
    }

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
    if (presence_.begin()) presence_.onPresence([this]() { ESP_LOGI(kTag, "presence detected"); });

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
    {  // the experience the scenario gives the robot (on the robot: NVS, RobotRuntime::loadLearned)
        AssocMemory::Weights w{};
        for (const auto& link : World::get().scenario().learned) w[link.cue] = {link.reward, link.threat};
        if (!World::get().scenario().learned.empty()) loop_.brain().assoc().setWeights(w);
    }
    ESP_LOGI(kTag, "RobotRuntime ready");
    return true;
}

// WifiEngine's observable behaviour: connect, NTP, periodic owner ping.
void SimRuntime::updateWifi(uint32_t now)
{
    if (!wifi_connected_ && now >= kWifiUpMs) {
        wifi_connected_ = true;
        ESP_LOGI("wifi", "got IP");
        next_owner_check_ms_ = now + 1000;
    }
    if (!time_synced_ && now >= kTimeSyncMs) {
        time_synced_ = true;
        ESP_LOGI("wifi", "time synced");
    }
    if (wifi_connected_ && now >= next_owner_check_ms_) {
        const bool home = World::get().ownerHome();
        if (home != owner_nearby_) ESP_LOGI("wifi", "known device %s", home ? "nearby" : "gone");
        owner_nearby_ = home;
        next_owner_check_ms_ = now + kPresenceCheckIntervalMs;
    }
}

// Mirrors RobotRuntime::logStats() (behaviour fields only).
void SimRuntime::logStats(uint32_t now)
{
    float head_yaw = 0.0f, head_pitch = 0.0f;
    motion_.getCurrentDeg(head_yaw, head_pitch);
    auto& brain = loop_.brain();
    auto& ex = loop_.executor();
    const auto& s = brain.bus().snn;
    auto pct = [](float x) { return static_cast<unsigned>(std::clamp(x, 0.0f, 1.0f) * 100.0f + 0.5f); };
    ESP_LOGI(kTag,
             "stats: wifi=%s time=%s owner=%s snn_rate=%.3f snn_resets=%u policy_faults=%u missed_ticks=%u "
             "cur=%.2f aro=%.2f bor=%.2f ann=%.2f soc=%.2f play=%.2f fat=%.2f slp=%.2f mus=%.2f "
             "moves=%u calls=%u nods=%u startles=%u stamina=%u%% travel=%udeg head=%d,%d voice=%s heart=%.0f",
             wifi_connected_ ? "up" : "down", time_synced_ ? "synced" : "unsynced", owner_nearby_ ? "near" : "away",
             static_cast<double>(brain.snn().meanRate()), static_cast<unsigned>(brain.snnResets()),
             static_cast<unsigned>(brain.policyFaults()), static_cast<unsigned>(loop_.missedTicks()),
             s[hybrid::SNN_CURIOSITY], s[hybrid::SNN_BOREDOM], s[hybrid::SNN_AROUSAL], s[hybrid::SNN_ANNOYANCE],
             s[hybrid::SNN_SOCIAL], s[hybrid::SNN_PLAYFUL], s[hybrid::SNN_FATIGUE], s[hybrid::SNN_SLEEP],
             s[hybrid::SNN_MUSIC], static_cast<unsigned>(ex.moves()), static_cast<unsigned>(ex.calls()),
             static_cast<unsigned>(ex.nods()), static_cast<unsigned>(ex.startles()),
             pct(motion_.staminaLevel()), static_cast<unsigned>(motion_.travelledDeg()), static_cast<int>(head_yaw),
             static_cast<int>(head_pitch), ex.voiceTrained() ? "babble" : "silent",
             static_cast<double>(brain.physiology().heartBpm));
    next_diagnostics_ms_ = now + 5000;
}

void SimRuntime::writeTrace(uint32_t now)
{
    World& w = World::get();
    float lx = 0, ly = 0;
    bool lamp_seen = false;
    for (size_t i = 0; i < w.scenario().lamps.size(); ++i)
        if (w.lampInView(i, lx, ly)) {
            lamp_seen = true;
            break;
        }
    ++minute_ticks_;
    if (lamp_seen) {
        ++minute_lamp_view_;
        if (std::fabs(lx) < 0.25f && std::fabs(ly) < 0.4f) ++minute_lamp_centred_;
    }
    if (!opt_.trace) return;
    using namespace hybrid;
    const auto& b = loop_.brain().bus();
    const auto& a = loop_.brain().action();
    float cmd_yaw = 0, cmd_pitch = 0;
    motion_.getCurrentDeg(cmd_yaw, cmd_pitch);
    float person_x = 9.0f;  // 9 = nobody in view
    w.personImageX(vrtos::nowUs(), person_x);
    std::fprintf(opt_.trace,
                 "%.1f,%.3f,%.1f,%.1f,%.1f,%.1f,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                 "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                 now / 1000.0, w.hourOfDay(vrtos::nowUs()), w.headYawDeg(), w.headPitchDeg(), cmd_yaw, cmd_pitch,
                 emotionName(face_.emotion()), vocal_.active() ? 1 : 0, motion_.staminaLevel(),
                 a[ACT_FACE_VALENCE], a[ACT_FACE_AROUSAL], a[ACT_LIDS], b.snn[SNN_CURIOSITY], b.snn[SNN_BOREDOM],
                 b.snn[SNN_AROUSAL], b.snn[SNN_ANNOYANCE], b.snn[SNN_SOCIAL], b.snn[SNN_SLEEP], b.snn[SNN_MUSIC],
                 b.snn[SNN_ATT_STRENGTH], b.snn[SNN_ATT_X], b.snn[SNN_ATT_Y], a[ACT_HOLD], a[ACT_EXPLORE],
                 a[ACT_SOUND_ACT], a[ACT_INVITE], lamp_seen ? 1.0f : 0.0f, person_x,
                 b.learn[LEARN_EXPECT_REWARD], b.learn[LEARN_EXPECT_THREAT], a[ACT_LED_ACTIVITY]);
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
        std::fprintf(opt_.summary,
                     "%3u | %5.2f | %5.0f | %4d | %4d | %4d | %4d | %4d | %4d | %4d | %3.0f%% | %3.0f%% | %4.0f%% | %4.0f | %4.0f\n",
                     minute_index_ + 1, w.hourOfDay(vrtos::nowUs()), w.travelDeg() - minute_travel_start_,
                     w.movementsStarted() - minute_moves_start_, count("executor: emotion="),
                     count("executor: voice:"), count("executor: sing: start"), count("runtime: loud sound"),
                     count("runtime: head touch:") + count("runtime: screen touch"),
                     count("runtime: tilt") + count("runtime: shake"),
                     minute_ticks_ ? 100.0f * minute_lamp_view_ / minute_ticks_ : 0.0f,
                     minute_ticks_ ? 100.0f * minute_lamp_centred_ / minute_ticks_ : 0.0f,
                     motion_.staminaLevel() * 100.0f, w.headYawDeg(), w.headPitchDeg());
        std::fflush(opt_.summary);
    }
    resetLogCounts();
    minute_travel_start_ = w.travelDeg();
    minute_moves_start_ = w.movementsStarted();
    minute_ticks_ = minute_lamp_view_ = minute_lamp_centred_ = 0;
    ++minute_index_;
    next_minute_ms_ = now + 60000;
}

// Mirrors RobotRuntime::loop(), one iteration per pass until `until_us`.
void SimRuntime::run(uint64_t until_us)
{
    if (opt_.summary) {
        std::fprintf(opt_.summary,
                     "min |  hour | travel | moves | faces | calls | songs | loud | touches | tilt/shake | lamp_in_view | "
                     "lamp_centred | stamina | yaw | pitch\n");
    }
    if (opt_.trace) {
        std::fprintf(opt_.trace,
                     "t_s,hour,head_yaw,head_pitch,cmd_yaw,cmd_pitch,emotion,sound,stamina,face_valence,face_arousal,lids,"
                     "curiosity,boredom,arousal,annoyance,social,sleep,music,att_strength,att_x,att_y,hold,explore,"
                     "sound_act,invite,lamp_in_view,person_x,expect_reward,expect_threat,led_activity\n");
    }
    next_minute_ms_ = nowMs() + 60000;
    uint64_t dumped = 0;
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
        if (World::get().shaking() && now - last_sim_shake_ms_ > 300) {
            last_sim_shake_ms_ = now;
            loop_.shake(now);
            ESP_LOGI(kTag, "shake detected");
        }
        if (World::get().bodyLifted() && now - last_sim_tilt_ms_ > 300) {
            last_sim_tilt_ms_ = now;
            loop_.tilt(now);
            if (!sim_tilted_) ESP_LOGI(kTag, "tilt detected");
            sim_tilted_ = true;
        } else if (!World::get().bodyLifted()) {
            sim_tilted_ = false;
        }
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
        env.ownerNear = owner_nearby_;
        const uint32_t ticks_before = loop_.brain().tickCount();
        const auto host_start = std::chrono::steady_clock::now();
        loop_.step(now, env);
        if (loop_.brain().tickCount() != ticks_before) {
            host_tick_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - host_start).count();
            ++host_ticks_;
            // The tick costs ESP32-S3 CPU time: the main task stays busy for
            // it while the camera/mic tasks keep running.
            std::uniform_real_distribution<float> jitter(0.7f, 1.3f);
            vrtos::busyFor(static_cast<uint64_t>(opt_.brain_cost_ms * jitter(cost_rng_) * 1000.0f));
            if (opt_.v4_dump) {
                const auto& v4b = loop_.brainV4();
                std::fwrite(loop_.channels().data(), sizeof(float), loop_.channels().size(), opt_.v4_dump);
                std::fwrite(v4b.snn().data(), sizeof(float), v4b.snn().size(), opt_.v4_dump);
                std::fwrite(v4b.nn().data(), sizeof(float), v4b.nn().size(), opt_.v4_dump);
            }
            if (opt_.bus_dump) {
                // what the policy saw this tick (the bus has moved on: its previous-action features now hold this tick's action)
                const PolicyGru::Input& in = loop_.brain().lastPolicyInput();
                std::fwrite(in.data(), sizeof(float), in.size(), opt_.bus_dump);
                // the efference copy is an input of the spiking core only (SNN fine-tuning from rollouts)
                std::fwrite(loop_.brain().bus().eff.data(), sizeof(float), hybrid::EFF_COUNT, opt_.bus_dump);
                std::fwrite(loop_.brain().action().data(), sizeof(float), hybrid::ACT_COUNT, opt_.bus_dump);
                ++dumped;
            }
        }
        loop_.afterVoice(now);

        if (msReached(now, next_telemetry_ms_)) {
            static char line[768];
            loop_.formatTelemetry(line, sizeof(line));
            ESP_LOGI("brain", "%s", line);
            next_telemetry_ms_ = now + 1000;
        }
        if (msReached(now, next_diagnostics_ms_)) logStats(now);
        if (now >= next_trace_ms_) {
            writeTrace(now);
            next_trace_ms_ = now + 100;
        }
        if (now >= next_minute_ms_) writeMinute(now);
        vTaskDelay(pdMS_TO_TICKS(kMainLoopIntervalMs));
    }
    (void)dumped;
}

void SimRuntime::finish()
{
    // Wall-clock, so only on stdout: the log must stay bit-reproducible.
    std::printf("host brain tick (front ends' main-thread part + SNN + policy + executor) mean %.1f us over %llu ticks\n",
                host_ticks_ ? host_tick_s_ / host_ticks_ * 1e6 : 0.0, static_cast<unsigned long long>(host_ticks_));
}

}  // namespace sim
