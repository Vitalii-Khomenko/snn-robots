#include "robot_runtime.h"
#include "core/ms_clock.h"

#include <algorithm>
#include <cstring>
#include <array>
#include <cmath>
#include <bsp/esp-bsp.h>
#include <driver/i2s_std.h>
#include <esp_attr.h>
#include <esp_heap_caps.h>
#include <esp_log.h>
#include <esp_random.h>
#include <esp_task_wdt.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <nvs.h>
#include <nvs_flash.h>

#include "core/debug_flags.h"
#include "core/i2c_bus_recovery.h"
#include "hybrid/spiking_core.h"
#include "power/power_rails.h"

namespace {
constexpr const char* kTag = "runtime";
constexpr uint32_t kMainLoopIntervalMs = 20;  // 50 Hz is enough; motion is capped at 50 Hz and SNN at 20 Hz
constexpr bool kCameraEnabled = true;
// Switches for hardware diagnosis (2026-09-27 servo investigation). Both
// on in normal use.
constexpr bool kMicrophoneEnabled = true;
constexpr bool kWifiEnabled = true;
// Dataset logging mode (concept phase 10): 4 lines a second of policy input +
// action on the serial console, for offline retraining (tools/hybrid/serial_dataset.py).
constexpr bool kDatasetLogging = false;

// bsp_audio_init(NULL) brings the raw I2S channel up in MONO slot mode.
// The known-working driver always brings it up in STEREO with both slots,
// even though the codec_dev-level open() still requests a mono stream -
// see docs/HARDWARE.md.
i2s_std_config_t buildStereoI2sConfig(uint32_t sample_rate)
{
    i2s_std_config_t cfg      = {};
    cfg.clk_cfg.sample_rate_hz = sample_rate;
    cfg.clk_cfg.clk_src        = I2S_CLK_SRC_DEFAULT;
    cfg.clk_cfg.mclk_multiple  = I2S_MCLK_MULTIPLE_256;

    cfg.slot_cfg.data_bit_width = I2S_DATA_BIT_WIDTH_16BIT;
    cfg.slot_cfg.slot_bit_width = I2S_SLOT_BIT_WIDTH_AUTO;
    cfg.slot_cfg.slot_mode      = I2S_SLOT_MODE_STEREO;
    cfg.slot_cfg.slot_mask      = I2S_STD_SLOT_BOTH;
    cfg.slot_cfg.ws_width       = I2S_DATA_BIT_WIDTH_16BIT;
    cfg.slot_cfg.ws_pol         = false;
    cfg.slot_cfg.bit_shift      = true;
    cfg.slot_cfg.left_align     = true;
    cfg.slot_cfg.big_endian     = false;
    cfg.slot_cfg.bit_order_lsb  = false;

    cfg.gpio_cfg.mclk = BSP_I2S_MCLK;
    cfg.gpio_cfg.bclk = BSP_I2S_SCLK;
    cfg.gpio_cfg.ws   = BSP_I2S_LCLK;
    cfg.gpio_cfg.dout = BSP_I2S_DOUT;
    cfg.gpio_cfg.din  = BSP_I2S_DSIN;
    return cfg;
}

}  // namespace

RobotRuntime::RobotRuntime()
    : loop_(BrainLoop::Engines{motion_, face_, led_, audio_, vocal_, imu_, mic_, presence_, camera_, power_status_, touch_})
{
}

uint32_t RobotRuntime::nowMs() const
{
    return static_cast<uint32_t>(esp_timer_get_time() / 1000);
}

bool RobotRuntime::begin()
{
    // Must run before anything (bsp_display_start() right below, first)
    // claims BSP_I2C_SCL/SDA as a real I2C peripheral - see
    // core/i2c_bus_recovery.h for why a flash-triggered reset (as opposed
    // to a real power cycle) can leave this shared bus wedged.
    recoverStuckI2cBus();

    ESP_LOGI(kTag, "nvs");
    esp_err_t nvs_err = nvs_flash_init();
    if (nvs_err == ESP_ERR_NVS_NO_FREE_PAGES || nvs_err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs_err = nvs_flash_init();
    }
    if (nvs_err != ESP_OK) {
        ESP_LOGE(kTag, "nvs_flash_init failed: %s", esp_err_to_name(nvs_err));
    }

    loadLearned();

    ESP_LOGI(kTag, "hybrid brain: SNN %d neurons / %d synapses, policy GRU, vision NN %s, auditory NN %s",
             SpikingCore::kNeurons, loop_.brain().snn().synapseCount(), camera_.net().trained() ? "trained" : "MISSING",
             loop_.auditory().trained() ? "trained" : "MISSING");

    ESP_LOGI(kTag, "screen");
    bsp_display_start();
    bsp_display_backlight_on();
    face_.begin();

    ESP_LOGI(kTag, "LEDs");
    if (!led_.begin()) {
        ESP_LOGE(kTag, "LED begin failed");
    }
    face_.setEmotion(Emotion::Normal);
    led_.setBreathing(0, 25, 30);

    ESP_LOGI(kTag, "speaker");
    power::enableAxp2101PowerRails();
    i2s_std_config_t i2s_cfg = buildStereoI2sConfig(AudioOutput::kSampleRate);
    bsp_audio_init(&i2s_cfg);
    if (!audio_.begin()) {
        ESP_LOGE(kTag, "audio begin failed");
    } else if (!vocal_.begin()) {
        ESP_LOGE(kTag, "voice begin failed");
    }

    ESP_LOGI(kTag, "servos");
    if (!power::enableServoPower()) {
        ESP_LOGE(kTag, "servo power enable failed");
    }
    if (!motion_.begin()) {
        ESP_LOGE(kTag, "motion begin failed");
    }

    ESP_LOGI(kTag, "touch");
    if (touch_.begin()) {
        touch_.onTouch([this]() { loop_.headTouch(nowMs()); });
    } else {
        ESP_LOGE(kTag, "touch begin failed");
    }

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
    } else {
        ESP_LOGE(kTag, "IMU begin failed");
    }

    ESP_LOGI(kTag, "microphone");
    if (!kMicrophoneEnabled) {
        ESP_LOGW(kTag, "microphone disabled (kMicrophoneEnabled = false)");
    } else if (mic_.begin()) {
        mic_.onLoudSound([this]() {
            // Heard while our own servos ran: mostly ourselves. (Their clicks
            // can clip the ADC too - 32664 seen on hardware - so the peak
            // cannot tell a clap from a click during a turn.)
            float intensity = std::clamp((0.6f + static_cast<float>(mic_.lastLoudPeak()) / 20000.0f) / 1.5f, 0.4f, 1.0f);
            if (mic_.lastLoudDuringOwnMotion()) intensity *= 0.3f;
            loop_.loudSound(intensity);
            ESP_LOGI(kTag, "loud sound: peak=%d%s", mic_.lastLoudPeak(), mic_.lastLoudDuringOwnMotion() ? " own-motion" : "");
        });
    } else {
        ESP_LOGE(kTag, "microphone begin failed");
    }

    ESP_LOGI(kTag, "presence");
    if (presence_.begin()) {
        presence_.onPresence([this]() { ESP_LOGI(kTag, "presence detected"); });
    } else {
        ESP_LOGE(kTag, "presence begin failed");
    }

    if constexpr (kCameraEnabled) {
        ESP_LOGI(kTag, "camera");
        if (camera_.begin()) {
            camera_.onVision([this](const CameraEngine::VisionFrame& frame) {
                // CameraEngine invokes this from update(), never from its
                // capture task, so this copy stays on the main-loop thread.
                loop_.vision(frame, nowMs());
            });
        } else {
            ESP_LOGE(kTag, "camera begin failed");
        }
    }

    // Continuous readings, sampled by tickBrain() as NeuroBrain::Environment.
    ESP_LOGI(kTag, "power status");
    if (!power_status_.begin()) {
        ESP_LOGE(kTag, "power status begin failed");
    }

    // Wi-Fi (net/wifi_engine.h) - station connect + NTP + LAN device-
    // presence ping, feeding the NIGHT/DAY/OWNER input cells. Non-blocking:
    // association and sync happen in the background.
    ESP_LOGI(kTag, "wifi");
    if (!kWifiEnabled) {
        ESP_LOGW(kTag, "wifi disabled (diagnostic)");
    } else if (!wifi_.begin()) {
        ESP_LOGE(kTag, "wifi begin failed");
    }

    const uint32_t now = nowMs();
    // The Vision NN's layer activations (9 KB) go to internal SRAM when - and
    // only when - Wi-Fi and the camera have already taken theirs and a safe
    // reserve remains (Wi-Fi allocates rx/tx buffers dynamically at runtime).
    {
        constexpr size_t kInternalReserve = 36 * 1024;
        const size_t internal_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (internal_free > VisionNet::kFastScratchBytes + kInternalReserve) {
            float* scratch = static_cast<float*>(
                heap_caps_malloc(VisionNet::kFastScratchBytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
            if (scratch) camera_.useFastScratch(scratch);
        }
        ESP_LOGI(kTag, "vision activations in %s (internal free %u B)",
                 internal_free > VisionNet::kFastScratchBytes + kInternalReserve ? "internal SRAM" : "PSRAM",
                 static_cast<unsigned>(internal_free));
    }
    // The spiking core's input weights (231 KB) are read ~30 columns per tick:
    // PSRAM is about twice as fast as this board's flash (DIO mode).
    {
        void* copy = heap_caps_malloc(SpikingCore::kInputWeightBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (copy) {
            std::memcpy(copy, SpikingCore::inputWeights(), SpikingCore::kInputWeightBytes);
            loop_.brain().snn().useFastInputWeights(static_cast<const int8_t*>(copy));
        }
        ESP_LOGI(kTag, "SNN input weights in %s", copy ? "PSRAM" : "flash");
    }
    diagnostics_window_start_us_ = static_cast<uint64_t>(esp_timer_get_time());
    next_diagnostics_ms_ = now + 5000;
    next_learn_save_ms_ = now + 600000;

    ESP_LOGI(kTag, "RobotRuntime ready");
    return true;
}

// Safety net for the servos (the bus is write-only on this board: the gyro
// is the only way to know the head really moves). Two 2 s windows of
// commanded motion without real motion -> cycle the servo rail (VM EN) and
// resend the position. The stalls had two causes (2026-09-27): PWM mode
// (fixed in MotionEngine::writeServos) and too many packets (50/s -> 10/s,
// see kUpdateMs); this catches whatever is left.
void RobotRuntime::superviseServos(uint32_t now)
{
    // (A planned power cycle 2 s after Wi-Fi association used to be here, a
    // workaround from before the real causes were found - PWM mode, then the
    // packet rate. Removed 2026-09-27: it only made the head go limp once.)
    bool repower = false;
    const char* why = "";
    if (now - watch_window_ms_ >= 2000) {
        const float commanded = motion_.travelledDeg() - watch_travel_deg_;
        const float real = imu_.takeWatchRotationDeg();
        watch_travel_deg_ = motion_.travelledDeg();
        watch_window_ms_ = now;
        const bool stalled = commanded >= 10.0f && real < 1.5f;
        // Two 2 s windows in a row with >= 10 deg commanded and < 1.5 deg
        // real are a stall. One window is not enough: slow micro-drift
        // glides below the gyro's 3 deg/s noise floor and adds commanded
        // degrees the gyro never counts (one false power cycle on hardware).
        stalled_windows_ = stalled ? stalled_windows_ + 1 : 0;
        // (A soft cure - position mode, torque and position resent - was
        // tried first for a while: it cleared 2 stalls of ~14. The servos
        // really hang; only their power brings them back.)
        if (stalled_windows_ >= 2 && now - last_repower_ms_ > 10000) {
            repower = true;
            why = "commanded motion but the gyro saw none";
        }
    }
    if (!repower) return;
    int py32, vm_out, vm_pin;
    power::readServoPowerState(py32, vm_out, vm_pin);
    ESP_LOGW(kTag, "servo rail power cycle (%s); py32=0x%02x vm_en out=%d", why, py32, vm_out);
    power::enableServoPower();  // VM EN off 500 ms, on
    motion_.resyncAfterPowerCycle();
    last_repower_ms_ = nowMs();
    stalled_windows_ = 0;
    ++servo_repowers_;
}

namespace {
constexpr const char* kLearnNs = "assoc";
constexpr uint32_t kLearnMagic = 0x41534d31;  // "ASM1": the layout of AssocMemory::Weights below
struct LearnedBlob {
    uint32_t magic = kLearnMagic;
    AssocMemory::Weights w{};
};
}  // namespace

void RobotRuntime::loadLearned()
{
    nvs_handle_t h;
    if (nvs_open(kLearnNs, NVS_READONLY, &h) != ESP_OK) {
        ESP_LOGI(kTag, "learned: nothing yet (a fresh memory)");
        return;
    }
    LearnedBlob blob;
    size_t size = sizeof(blob);
    if (nvs_get_blob(h, "w", &blob, &size) == ESP_OK && size == sizeof(blob) && blob.magic == kLearnMagic) {
        loop_.brain().assoc().setWeights(blob.w);
        int links = 0;
        for (const auto& cue : blob.w)
            for (float x : cue) links += x >= 0.25f ? 1 : 0;
        ESP_LOGI(kTag, "learned: memory loaded (%d links >= 0.25)", links);
    } else {
        ESP_LOGW(kTag, "learned: stored memory unreadable, starting fresh");
    }
    nvs_close(h);
}

void RobotRuntime::saveLearned(uint32_t now)
{
    learn_change_ += loop_.brain().assoc().takeChange();
    if (!msReached(now, next_learn_save_ms_)) return;
    next_learn_save_ms_ = now + 600000;
    if (learn_change_ < 0.01f) return;  // nothing worth a flash write
    learn_change_ = 0.0f;
    LearnedBlob blob;
    blob.w = loop_.brain().assoc().weights();
    nvs_handle_t h;
    if (nvs_open(kLearnNs, NVS_READWRITE, &h) != ESP_OK) return;
    const bool ok = nvs_set_blob(h, "w", &blob, sizeof(blob)) == ESP_OK && nvs_commit(h) == ESP_OK;
    nvs_close(h);
    ESP_LOGI(kTag, "learned: memory saved%s", ok ? "" : " - FAILED");
}

void RobotRuntime::logStats(uint32_t now)
{
    const uint32_t servo_sent = motion_.takePacketCount();
    uint32_t imu_reads = 0, imu_fails = 0;
    float gyro_max = 0.0f;
    imu_.takeHealth(imu_reads, imu_fails, gyro_max);
    const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
    const uint32_t window_us = static_cast<uint32_t>(now_us - diagnostics_window_start_us_);
    // The voice streams from its own task, so the main loop never waits for the speaker any more: all of its busy
    // time is work. (Subtracting AudioOutput's DMA wait - measured in the voice task - read ~0 % while it talked.)
    const uint64_t active_us = diagnostics_busy_us_;
    const unsigned busy = window_us ? static_cast<unsigned>(diagnostics_busy_us_ * 1000ULL / window_us) : 0;
    const unsigned active = window_us ? static_cast<unsigned>(active_us * 1000ULL / window_us) : 0;
    float head_yaw = 0.0f, head_pitch = 0.0f;
    motion_.getCurrentDeg(head_yaw, head_pitch);
    auto& brain = loop_.brain();
    auto& executor = loop_.executor();
    auto pct = [](float x) { return static_cast<unsigned>(std::clamp(x, 0.0f, 1.0f) * 100.0f + 0.5f); };
    uint32_t face_frames = 0, face_max_us = 0;
    face_.takeDrawStats(face_frames, face_max_us);
    const unsigned tract_permille = window_us ? static_cast<unsigned>(vocal_.takeRenderUs() * 1000ULL / window_us) : 0;
    ESP_LOGI(kTag,
             "stats: busy=%u.%u%% active=%u.%u%% internal_free=%uB internal_min=%uB psram_free=%uB stack_headroom=%uB "
             "wifi=%s time=%s owner=%s snn_us=%u/%u snn_stage_us=%.0f/%.0f/%.0f policy_us=%u vision_us=%u/%u audio_us=%u mic_block_us=%u tick_us=%u tick_max_us=%u "
             "missed_ticks=%u tick_ms=%u snn_rate=%.3f snn_resets=%u policy_faults=%u vision_nan=%u audio_nan=%u "
             "moves=%u calls=%u nods=%u startles=%u stamina=%u%% travel=%udeg battery=%u%% head=%d,%d "
             "servo_packets=%u cmd_travel=%.0fdeg real_rotation=%.0fdeg gyro=%s imu=%u/%u gyro_max=%.0f servo_repowers=%u "
             "face_fps=%u face_frame_max_us=%u cam_fps=%.1f mic_peak=%d/%d/%d voice=%s voice_gaps=%u voice_load=%u.%u%% heart=%.0f",
             busy / 10, busy % 10, active / 10, active % 10,
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
             static_cast<unsigned>(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)),
             static_cast<unsigned>(uxTaskGetStackHighWaterMark(NULL) * 4),
             wifi_.isConnected() ? "up" : "down", wifi_.isTimeSynced() ? "synced" : "unsynced",
             wifi_.knownDeviceNearby() ? "near" : "away", static_cast<unsigned>(brain.snnUs()),
             static_cast<unsigned>(brain.takeSnnMinUs()), static_cast<double>(brain.snn().stageUs()[0]),
             static_cast<double>(brain.snn().stageUs()[1]), static_cast<double>(brain.snn().stageUs()[2]),
             static_cast<unsigned>(brain.policyUs()),
             static_cast<unsigned>(camera_.visionUs()), static_cast<unsigned>(camera_.takeVisionMinUs()),
             static_cast<unsigned>(loop_.audioUs()), static_cast<unsigned>(mic_.audioFront().blockUs()), static_cast<unsigned>(loop_.tickLastUs()),
             static_cast<unsigned>(loop_.takeTickMaxUs()), static_cast<unsigned>(loop_.missedTicks()),
             static_cast<unsigned>(loop_.tickPeriodMs()), static_cast<double>(brain.snn().meanRate()),
             static_cast<unsigned>(brain.snnResets()), static_cast<unsigned>(brain.policyFaults()),
             static_cast<unsigned>(camera_.net().nanCount()), static_cast<unsigned>(loop_.auditory().nanCount()),
             static_cast<unsigned>(executor.moves()), static_cast<unsigned>(executor.calls()),
             static_cast<unsigned>(executor.nods()),
             static_cast<unsigned>(executor.startles()), pct(motion_.staminaLevel()),
             static_cast<unsigned>(motion_.travelledDeg()), pct(loop_.raw().battery01), static_cast<int>(head_yaw),
             static_cast<int>(head_pitch), static_cast<unsigned>(servo_sent), motion_.travelledDeg() - last_travel_deg_,
             imu_.takeRotationDeg(), loop_.gyroValid() ? "calibrated" : "learning", static_cast<unsigned>(imu_reads),
             static_cast<unsigned>(imu_fails), static_cast<double>(gyro_max),
             static_cast<unsigned>(servo_repowers_),
             static_cast<unsigned>(face_frames / 5), static_cast<unsigned>(face_max_us),
             static_cast<double>(camera_.takeFrameCount() * 1e6f / static_cast<float>(window_us)),
             mic_.takePhasePeak(0), mic_.takePhasePeak(1), mic_.takePhasePeak(2),
             executor.voiceTrained() ? "babble" : "silent", static_cast<unsigned>(vocal_.gaps()), tract_permille / 10, tract_permille % 10,
             static_cast<double>(brain.physiology().heartBpm));
    last_travel_deg_ = motion_.travelledDeg();
    diagnostics_window_start_us_ = now_us;
    diagnostics_busy_us_ = 0;
    brain.setLoad(std::clamp(static_cast<float>(active) / 1000.0f, 0.0f, 1.0f));
    next_diagnostics_ms_ = now + 5000;
}

void RobotRuntime::loop()
{
    // Subscribe to the task watchdog: a real hang becomes a clean reboot.
    esp_err_t wdt_err = esp_task_wdt_add(NULL);
    if (wdt_err != ESP_OK) {
        ESP_LOGE(kTag, "esp_task_wdt_add failed: %s", esp_err_to_name(wdt_err));
    }

    while (true) {
        esp_task_wdt_reset();
        const uint64_t loop_start_us = static_cast<uint64_t>(esp_timer_get_time());
        const uint32_t now = nowMs();
        motion_.update();
        face_.update();
        led_.update();
        if (face_.takeScreenTouch()) {
            loop_.screenTouch(now);
            ESP_LOGI(kTag, "screen touch");
        }
        touch_.update();
        // Our own servos shake the IMU and click into the microphone: raise
        // those thresholds while the head moves (BrainLoop::coupleMotion).
        loop_.coupleMotion(now);
        if constexpr (kMicrophoneEnabled) {
            mic_.update();
        }
        presence_.update();
        if constexpr (kCameraEnabled) {
            camera_.update();
        }
        power_status_.update();
        BrainLoop::Env env;
        env.timeSynced = wifi_.isTimeSynced();
        env.hour = env.timeSynced ? wifi_.hourOfDay() : 12.0f;
        env.ownerNear = wifi_.knownDeviceNearby();
        loop_.step(now, env);
        loop_.afterVoice(now);
        superviseServos(now);
        saveLearned(now);

        const uint64_t loop_elapsed_us = static_cast<uint64_t>(esp_timer_get_time()) - loop_start_us;
        diagnostics_busy_us_ += loop_elapsed_us;
        if (msReached(now, next_telemetry_ms_)) {
            static char line[768];
            loop_.formatTelemetry(line, sizeof(line));
            ESP_LOGI("brain", "%s", line);
            next_telemetry_ms_ = now + 1000;
        }
        if (msReached(now, next_diagnostics_ms_)) {
            logStats(now);
        }
        if constexpr (kAudioRecording) {
            if (msReached(now, next_audio_rec_ms_)) {
                static char line[480];
                loop_.formatAudioRecord(now, line, sizeof(line));
                ESP_LOGI("audio", "%s", line + 9);
                next_audio_rec_ms_ = now + 50;
            }
        }
        if constexpr (kDatasetLogging) {
            if (msReached(now, next_dataset_ms_)) {
                static char line[640];
                loop_.formatDataset(now, line, sizeof(line));
                ESP_LOGI("dataset", "%s", line + 9);
                next_dataset_ms_ = now + 250;
            }
        }
        vTaskDelay(pdMS_TO_TICKS(kMainLoopIntervalMs));
    }
}
