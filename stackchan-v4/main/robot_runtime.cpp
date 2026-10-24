#include "robot_runtime.h"
#include "core/ms_clock.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <array>
#include <cmath>
#include <bsp/esp-bsp.h>
#include <driver/gpio.h>
#include <driver/i2s_std.h>
#include <esp_attr.h>
#include <esp_heap_caps.h>
#include <esp_log.h>
#include <esp_core_dump.h>
#include <esp_ota_ops.h>
#include <esp_system.h>
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
    // The diary first, so the boot lines are in it too: they wait in its ring buffer until Wi-Fi is up.
    if constexpr (kWifiEnabled) life_.begin(wifi_);

    ESP_LOGI(kTag, "nvs");
    esp_err_t nvs_err = nvs_flash_init();
    if (nvs_err == ESP_ERR_NVS_NO_FREE_PAGES || nvs_err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs_err = nvs_flash_init();
    }
    if (nvs_err != ESP_OK) {
        ESP_LOGE(kTag, "nvs_flash_init failed: %s", esp_err_to_name(nvs_err));
    }

    logBootReason();
    loadLearned();
    loadWeights();
    loadMode();
    {
        const esp_partition_t* run = esp_ota_get_running_partition();
        esp_ota_img_states_t state = ESP_OTA_IMG_UNDEFINED;
        esp_ota_get_state_partition(run, &state);
        ESP_LOGI(kTag, "firmware in %s (%s)", run ? run->label : "?",
                 state == ESP_OTA_IMG_PENDING_VERIFY ? "new, on trial for a minute" : "confirmed");
    }

    ESP_LOGI(kTag, "hybrid brain: SNN %d neurons / %d synapses, policy GRU, vision NN %s, auditory NN %s",
             SpikingCore::kNeurons, loop_.brain().snn().synapseCount(), camera_.net().trained() ? "trained" : "MISSING",
             loop_.auditory().trained() ? "trained" : "MISSING");

    // The microSD slot shares the screen's SPI wires: with its chip select floating an inserted card answers the
    // screen's traffic and the picture flickers or vanishes (seen 2026-10-11). Deselected until it is mounted.
    gpio_reset_pin(BSP_SD_SPI_CS);
    gpio_set_direction(BSP_SD_SPI_CS, GPIO_MODE_OUTPUT);
    gpio_set_level(BSP_SD_SPI_CS, 1);

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
    {
        nvs_handle_t h;
        uint8_t level = 3;
        if (nvs_open("life", NVS_READONLY, &h) == ESP_OK) {
            nvs_get_u8(h, "touch", &level);
            nvs_close(h);
        }
        touch_.setSensitivity(level);
        ESP_LOGI(kTag, "touch pad sensitivity level %u", static_cast<unsigned>(level));
    }
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
    } else {
        ota_.begin(weights_, [this](char* buf, int size) { return formatStatus(buf, size); },
                   [this](int mode) { return setMode(mode); },
                   [this](int level) {
                       if (!touch_.setSensitivity(level)) return false;
                       nvs_handle_t h;
                       if (nvs_open("life", NVS_READWRITE, &h) == ESP_OK) {
                           nvs_set_u8(h, "touch", static_cast<uint8_t>(level));
                           nvs_commit(h);
                           nvs_close(h);
                       }
                       return true;
                   },
                   [this](int kind, uint8_t* buf, int cap) { return formatMedia(kind, buf, cap); });
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
            loop_.brainV4().analyzer().useFastWeights();  // brain v4's int8 tables (236 KB) to PSRAM as well
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
        // a silent gyro sees no rotation whatever the head does: that is not a stall (imu_engine.h)
        const bool stalled = commanded >= 10.0f && real < 1.5f && imu_.gyroAlive();
        // Two 2 s windows in a row with >= 10 deg commanded and < 1.5 deg
        // real are a stall. One window is not enough: slow micro-drift
        // glides below the gyro's 3 deg/s noise floor and adds commanded
        // degrees the gyro never counts (one false power cycle on hardware).
        stalled_windows_ = stalled ? stalled_windows_ + 1 : 0;
        // (A soft cure - position mode, torque and position resent - was
        // tried first for a while: it cleared 2 stalls of ~14. The servos
        // really hang; only their power brings them back.)
        // A series means the head is held or blocked (by a hand, a cover, the wall): power-cycling every 12 s did not
        // help and every restart is a current surge - the screen went black and twice the whole board froze
        // (2026-10-11). After 3 cycles within 3 minutes the next one waits 5 minutes.
        const bool series = repower_times_[0] != 0 && repower_times_[2] - repower_times_[0] < 180000;
        // a head that moved for real ends the series; each power cycle within a series makes the next wait longer:
        // 1, 2, 5, 10, 30, 60 min (it began with 5 min, and the owner saw a frozen head for 5 minutes after one
        // series, 2026-10-13; under a cover at night the head may be blocked for hours)
        static constexpr uint32_t kSeriesWaitMs[] = {60000, 120000, 300000, 600000, 1800000, 3600000};
        if (real >= 10.0f) series_level_ = 0;
        const uint32_t gap = series ? kSeriesWaitMs[std::min(series_level_, 5)] : 10000;
        if (stalled_windows_ >= 2 && now - last_repower_ms_ > gap) {
            repower = true;
            if (series) ++series_level_;
            why = "commanded motion but the gyro saw none";
        } else if (stalled_windows_ >= 2 && series && !blocked_logged_) {
            blocked_logged_ = true;
            ESP_LOGW(kTag, "servos: the head seems held or blocked (3 power cycles in 3 min): next power cycle in %u min",
                     static_cast<unsigned>(gap / 60000));
        }
    }
    // A hung servo pulls current for as long as it has power: the USB input sinks to the AXP's limit, the battery
    // makes up the rest, VSYS falls and the 3.3 V DCDC trips - the black-screen power-offs (2026-10-22/23, the
    // last one 4 min into a sag while the stall rule waited out a series pause). A sag of 3 s cuts the servo rail
    // whatever the stall rule says. If the sag outlives the cycle the servos were not the cause: no more sag
    // cycles for 10 minutes.
    const int sag = power_status_.sagSeconds();
    if (sag_repower_ms_ && now - sag_repower_ms_ > 5000 && now - sag_repower_ms_ < 8000 && sag >= 3) {
        sag_repower_ms_ = 0;
        sag_pause_until_ms_ = now + 600000;
        ESP_LOGW(kTag, "power: VBUS still sags after the servo rail cycle - not the servos; no sag cycles for 10 min");
    }
    if (!repower && sag >= 3 && msReached(now, sag_pause_until_ms_) && now - last_repower_ms_ > 15000) {
        repower = true;
        why = "the servo rail drains the supply: VBUS sagging";
        sag_repower_ms_ = now;
    }
    if (!repower) return;
    int py32, vm_out, vm_pin;
    power::readServoPowerState(py32, vm_out, vm_pin);
    float gx, gy, gz, ax, ay, az;
    imu_.getGyro(gx, gy, gz);
    imu_.getAccel(ax, ay, az);
    ESP_LOGW(kTag, "servo rail power cycle (%s); py32=0x%02x vm_en out=%d; gyro %.3f,%.3f,%.3f same for %d reads, "
             "accel %.2f,%.2f,%.2f", why, py32, vm_out, static_cast<double>(gx), static_cast<double>(gy),
             static_cast<double>(gz), imu_.gyroRepeats(), static_cast<double>(ax), static_cast<double>(ay),
             static_cast<double>(az));
    power::enableServoPower();  // VM EN off 500 ms, on
    motion_.resyncAfterPowerCycle();
    last_repower_ms_ = nowMs();
    loop_.servoRepowered(last_repower_ms_);  // brain v4: the head's jolt is not a shake
    stalled_windows_ = 0;
    ++servo_repowers_;
    // the last three power cycles (oldest first): three within 3 minutes are a series
    repower_times_[0] = repower_times_[1];
    repower_times_[1] = repower_times_[2];
    repower_times_[2] = last_repower_ms_;
    blocked_logged_ = false;
    panel_recover_ms_ = last_repower_ms_ + 1000;  // when the rail has settled
}

namespace {
constexpr const char* kLearnNs = "assoc";
constexpr uint32_t kLearnMagic = 0x41534d32;  // "ASM2": 11 cues x 3 outcomes (ASM1 had 2, 2026-10-17)
constexpr uint32_t kLearnMagicV1 = 0x41534d31;
struct LearnedBlobV1 {
    uint32_t magic;
    std::array<std::array<float, 2>, AssocMemory::kCues> w;
};
struct LearnedBlob {
    uint32_t magic = kLearnMagic;
    AssocMemory::Weights w{};
};
constexpr uint32_t kNeedsMagic = 0x4e454431;  // "NED1": Motivation's five needs
struct NeedsBlob {
    uint32_t magic = kNeedsMagic;
    std::array<float, Motivation::kNeeds> d{};
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
    blob.magic = 0;  // nothing read yet
    size_t size = sizeof(blob);
    LearnedBlobV1 v1;
    size_t size_v1 = sizeof(v1);
    if (nvs_get_blob(h, "w", &v1, &size_v1) == ESP_OK && size_v1 == sizeof(v1) && v1.magic == kLearnMagicV1) {
        for (int c = 0; c < AssocMemory::kCues; ++c) blob.w[c] = {v1.w[c][0], v1.w[c][1], 0.0f};
        blob.magic = kLearnMagic;
        size = sizeof(blob);
        ESP_LOGI(kTag, "learned: the 2-outcome memory carried over into 3 outcomes");
    }
    if ((blob.magic == kLearnMagic && size == sizeof(blob)) ||
        (nvs_get_blob(h, "w", &blob, &size) == ESP_OK && size == sizeof(blob) && blob.magic == kLearnMagic)) {
        loop_.brain().assoc().setWeights(blob.w);
        int links = 0;
        for (const auto& cue : blob.w)
            for (float x : cue) links += x >= 0.25f ? 1 : 0;
        ESP_LOGI(kTag, "learned: memory loaded (%d links >= 0.25)", links);
    } else {
        ESP_LOGW(kTag, "learned: stored memory unreadable, starting fresh");
    }
    {
        OnlineLearner::Weights w{};
        size = sizeof(w);
        if (nvs_get_blob(h, "online", &w, &size) == ESP_OK && size == sizeof(w)) {
            learner_.setWeights(w);
            ESP_LOGI(kTag, "online learning: the day's corrections restored (mean |dW| %.4f)",
                     static_cast<double>(learner_.size()));
        }
    }
    NeedsBlob needs;
    size = sizeof(needs);
    if (nvs_get_blob(h, "needs", &needs, &size) == ESP_OK && size == sizeof(needs) && needs.magic == kNeedsMagic) {
        loop_.brain().motivation().restore(needs.d);
        ESP_LOGI(kTag, "needs restored: social %.2f, stimulation %.2f, comfort %.2f",
                 static_cast<double>(needs.d[Motivation::Social]), static_cast<double>(needs.d[Motivation::Stimulation]),
                 static_cast<double>(needs.d[Motivation::Comfort]));
    }
    nvs_close(h);
}

void RobotRuntime::saveLearned(uint32_t now)
{
    learn_change_ += loop_.brain().assoc().takeChange();
    if (!msReached(now, next_learn_save_ms_)) return;
    next_learn_save_ms_ = now + 600000;
    nvs_handle_t h;
    if (nvs_open(kLearnNs, NVS_READWRITE, &h) != ESP_OK) return;
    // the needs every 10 min (24 bytes: a reboot does not make the robot forget it was lonely)
    NeedsBlob needs;
    needs.d = loop_.brain().motivation().needs();
    nvs_set_blob(h, "needs", &needs, sizeof(needs));
    if (mode_ == 2) nvs_set_blob(h, "online", &learner_.weights(), sizeof(OnlineLearner::Weights));
    if (learn_change_ >= 0.01f) {  // the memory only when it changed
        learn_change_ = 0.0f;
        LearnedBlob blob;
        blob.w = loop_.brain().assoc().weights();
        nvs_set_blob(h, "w", &blob, sizeof(blob));
        ESP_LOGI(kTag, "learned: memory saved");
    }
    if (nvs_commit(h) != ESP_OK) ESP_LOGW(kTag, "learned/needs: NVS commit FAILED");
    nvs_close(h);
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
             "face_fps=%u face_frame_max_us=%u cam_fps=%.1f mic_peak=%d/%d/%d voice=%s voice_gaps=%u voice_load=%u.%u%% heart=%.0f diary=%u/%u sd=%u/%u world_us=%u gyro_restarts=%u online_dw=%.4f",
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
             static_cast<double>(brain.physiology().heartBpm), static_cast<unsigned>(life_.sent()),
             static_cast<unsigned>(life_.dropped()), static_cast<unsigned>(life_.sdWritten()),
             static_cast<unsigned>(life_.sdErrors()), static_cast<unsigned>(world_us_), static_cast<unsigned>(imu_.gyroRestarts()),
             static_cast<double>(learner_.size()));
    {
        // brain v4's senses (v4/channel_encoder.h): the 13 channels now, and the raw light / proximity readings
        const auto& c = loop_.channels();
        auto& v4b = loop_.brainV4();
        const auto& so = v4b.snn();
        const auto& ex = loop_.executorV4();
        ESP_LOGI(kTag, "v4: als=%u ir=%u ps=%u/%02x near=%.0f ch=%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f "
                 "snn=%s/%uus/rate%.3f val=%.2f aro=%.2f under=%.2f over=%.2f threat=%.2f nn=%s/faults%u active=%d face=%s "
                 "calls=%u nods=%u shakes=%u",
                 static_cast<unsigned>(presence_.alsRaw()), static_cast<unsigned>(presence_.alsIrRaw()),
                 static_cast<unsigned>(presence_.lastRaw()), static_cast<unsigned>(presence_.psHigh()),
                 static_cast<double>(presence_.nearOverBaseline()), c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12],
                 v4b.analyzer().trained() ? "on" : "untrained", static_cast<unsigned>(v4b.analyzer().lastUs()),
                 static_cast<double>(v4b.analyzer().mixRate()), so[v4::SO_VALENCE], so[v4::SO_AROUSAL], so[v4::SO_UNDER],
                 so[v4::SO_OVER], so[v4::SO_THREAT], v4b.trained() ? "on" : "untrained",
                 static_cast<unsigned>(v4b.nnFaults()), loop_.v4Active() ? 1 : 0, emotionName(static_cast<Emotion>(ex.shownEmotion())),
                 static_cast<unsigned>(ex.calls()), static_cast<unsigned>(ex.nods()), static_cast<unsigned>(ex.shakes()));
    }
    {
        // the rails of the black-screen power-offs (2026-10-22/23): lowest VBUS / VSYS over the last stats window
        const auto r = power_status_.takeRails();
        ESP_LOGI(kTag, "power: vbus_min=%umV vsys_min=%umV vbat=%umV pmu_die=%.0fC pmu_status=0x%02x", r.vbus_min, r.vsys_min,
                 r.vbat, static_cast<double>(r.die_c), r.status);
    }
    last_travel_deg_ = motion_.travelledDeg();
    diagnostics_window_start_us_ = now_us;
    diagnostics_busy_us_ = 0;
    brain.setLoad(std::clamp(static_cast<float>(active) / 1000.0f, 0.0f, 1.0f));
    next_diagnostics_ms_ = now + 5000;
}

// Who acts: the policy network (the v3 brain), the data collector (motor babbling, behavior/collector.h), or life
// (platform B: the collector's decisions sampled from the actor trained on the diary, hybrid/actor_net.h).
void RobotRuntime::loadMode()
{
    nvs_handle_t h;
    uint8_t v = 0;
    if (nvs_open("life", NVS_READONLY, &h) == ESP_OK) {
        nvs_get_u8(h, "collector", &v);
        nvs_close(h);
    }
    mode_ = v <= 3 ? v : 0;
    collector_mode_ = mode_ == 1 || mode_ == 2;
    if (collector_mode_) {
        collector_ = Collector(esp_random());
        loop_.brain().setExternalPolicy([this](const PolicyGru::Input&, PolicyGru::Action& a) {
            const auto& bus = loop_.brain().bus();
            if (mode_ == 2) {
                // the actor's state, in the order of tools/life/train_critic.py STATE_NAMES
                float x[ActorNet::kIn];
                float* p = x;
                p = std::copy(bus.vision.begin(), bus.vision.end(), p);
                p = std::copy(bus.audio.begin(), bus.audio.end(), p);
                p = std::copy(bus.body.begin(), bus.body.end(), p);
                p = std::copy(bus.ctx.begin(), bus.ctx.end(), p);
                p = std::copy(bus.pe.begin(), bus.pe.end(), p);
                p = std::copy(bus.learn.begin(), bus.learn.end(), p);
                const auto& needs = loop_.brain().motivation().needs();
                p = std::copy(needs.begin(), needs.end(), p);
                p = std::copy(wm_log_, wm_log_ + WorldModel::kLogValues, p);
                *p = loop_.brain().assoc().expectation(AssocMemory::Talk);
                float out[ActorNet::kOut];
                const bool ok = actor_.step(x, out);
                if (ok) learner_.apply(actor_.features(), out);
                collector_.setActor(ok ? out : nullptr);
                collector_.step(bus, nowMs(), a);
                loop_.executor().setVoiceEmotion(collector_.voiceEmotion());
                if (ok) {
                    const auto& d = collector_.lastDecision();
                    learner_.learn(loop_.brain().motivation().reward(), actor_.features(), out,
                                   {d.move_possible, d.moved, d.call_possible, d.called, d.yaw, d.pitch},
                                   bus.ctx[hybrid::CTX_NIGHT] > 0.5f);
                }
                return;
            }
            collector_.step(bus, nowMs(), a);
        });
    }
    if (mode_ == 3) {
        // brain v4 drives the body (v4/brain_v4.h); the v3 policy is not computed (its slot gets a neutral action)
        loop_.brain().setExternalPolicy([](const PolicyGru::Input&, PolicyGru::Action& a) { a = PolicyGru::neutral(); });
        loop_.setV4Active(true);
    }
    loop_.brain().setSpikingCore(mode_ == 0 || mode_ == 1);  // modes life and v4 do not read the v3 spiking core
    loop_.executor().setFree(mode_ == 2);      // ... and has no behaviour budgets in the executor, only physical limits
    static const char* const names[] = {"brain (the v3 policy network)", "collector (motor babbling)",
                                        "life (platform B: the actor learned from the diary)",
                                        "v4 (brain v4: spiking analyzer + main network)"};
    ESP_LOGI(kTag, "mode: %s", names[mode_]);
}

// PUT /mode from the update server: stored, then the robot reboots into it.
bool RobotRuntime::setMode(int mode)
{
    nvs_handle_t h;
    if (mode < 0 || mode > 3 || nvs_open("life", NVS_READWRITE, &h) != ESP_OK) return false;
    const bool ok = nvs_set_u8(h, "collector", static_cast<uint8_t>(mode)) == ESP_OK && nvs_commit(h) == ESP_OK;
    nvs_close(h);
    ESP_LOGI(kTag, "mode -> %d after the reboot", mode);
    return ok;
}

// Why the robot (re)started, into the diary: a freeze without a trace (2026-10-11) is what this is for. A panic or
// a watchdog leaves a core dump in flash: its summary goes in once, then the dump is erased.
void RobotRuntime::logBootReason()
{
    static const char* const names[] = {"unknown", "power on", "external pin", "software restart", "panic",
                                        "interrupt watchdog", "task watchdog", "other watchdog", "deep sleep",
                                        "brownout (power dip)", "SDIO", "USB", "JTAG", "eFuse", "power glitch",
                                        "CPU lockup"};
    const esp_reset_reason_t r = esp_reset_reason();
    const int i = static_cast<int>(r);
    ESP_LOGI(kTag, "boot reason: %s (%d)", i >= 0 && i < static_cast<int>(std::size(names)) ? names[i] : "?", i);
    if (esp_core_dump_image_check() != ESP_OK) return;
    esp_core_dump_summary_t* sum = static_cast<esp_core_dump_summary_t*>(
        heap_caps_malloc(sizeof(esp_core_dump_summary_t), MALLOC_CAP_SPIRAM));
    if (sum && esp_core_dump_get_summary(sum) == ESP_OK) {
        char bt[160];
        int n = 0;
        for (uint32_t k = 0; k < sum->exc_bt_info.depth && k < 8 && n < static_cast<int>(sizeof(bt)) - 12; ++k)
            n += std::snprintf(bt + n, sizeof(bt) - static_cast<size_t>(n), " 0x%08lx",
                               static_cast<unsigned long>(sum->exc_bt_info.bt[k]));
        bt[n] = 0;
        ESP_LOGW(kTag, "crash of the last run: task %s, pc 0x%08lx, cause %lu, backtrace%s", sum->exc_task,
                 static_cast<unsigned long>(sum->exc_pc), static_cast<unsigned long>(sum->ex_info.exc_cause), bt);
    } else {
        ESP_LOGW(kTag, "a core dump from the last run is in flash (summary unreadable)");
    }
    heap_caps_free(sum);
    // logged once: the next boot must not report the same crash again (2026-10-17: an old dump read as a new one)
    esp_core_dump_image_erase();
}

// The nightly weights (net/weight_store.h), net by net: a net takes them only when every one of its arrays is there
// with the size this firmware expects; otherwise it keeps the weights compiled into the firmware.
namespace {
// A net of the policy's shape from the blob ("<net>.<array>"): all its arrays with the expected sizes, or nothing.
bool loadParams(const WeightStore& store, const char* net, const PolicyGru::ParamInfo* info, PolicyGru::Params& out)
{
    const void* ptr[PolicyGru::kParamCount];
    static_assert(sizeof(PolicyGru::Params) == sizeof(ptr), "Params = the pointers in paramInfo() order");
    for (int i = 0; i < PolicyGru::kParamCount; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "%s.%s", net, info[i].name);
        ptr[i] = store.find(name, info[i].int8, info[i].count);
        if (!ptr[i]) {
            ESP_LOGW(kTag, "weights: %s missing in the blob: %s keeps its compiled-in weights", name, net);
            return false;
        }
    }
    std::memcpy(&out, ptr, sizeof(out));
    return true;
}

// A layered INT8 net (Vision, Auditory) from the blob: every layer's weights, scales and biases, or nothing.
template <size_t N>
bool loadLayers(const WeightStore& store, const char* net, const LayerInfo* info, std::array<LayerParams, N>& out)
{
    std::array<LayerParams, N> p;
    for (size_t i = 0; i < N; ++i) {
        char w[32], s[32], b[32];
        std::snprintf(w, sizeof(w), "%s.%s_w", net, info[i].name);
        std::snprintf(s, sizeof(s), "%s.%s_s", net, info[i].name);
        std::snprintf(b, sizeof(b), "%s.%s_b", net, info[i].name);
        p[i].w = static_cast<const int8_t*>(store.find(w, true, info[i].weights));
        p[i].s = static_cast<const float*>(store.find(s, false, info[i].outputs));
        p[i].b = static_cast<const float*>(store.find(b, false, info[i].outputs));
        if (!p[i].w || !p[i].s || !p[i].b) {
            ESP_LOGW(kTag, "weights: %s.%s missing in the blob: %s keeps its compiled-in weights", net, info[i].name, net);
            return false;
        }
    }
    out = p;
    return true;
}
}  // namespace

// The nightly weights (net/weight_store.h), net by net: a net takes them only when every one of its arrays is there
// with the size this firmware expects; otherwise it keeps the weights compiled into the firmware.
void RobotRuntime::loadWeights()
{
    if (weights_.open()) applyWeights();
}

void RobotRuntime::applyWeights()
{
    PolicyGru::Params p;
    if (loadParams(weights_, "policy", PolicyGru::paramInfo(), p)) {
        loop_.brain().policy().use(p);
        ESP_LOGI(kTag, "weights: policy from flash (slot %c, version %u)", weights_.slot(),
                 static_cast<unsigned>(weights_.version()));
    }
    if (loadParams(weights_, "actor", ActorNet::paramInfo(), p)) {
        actor_.use(p);
        ESP_LOGI(kTag, "weights: actor from flash");
    }
    if (loadParams(weights_, "world", WorldModel::paramInfo(), p)) {
        world_.use(p);
        ESP_LOGI(kTag, "weights: world model from flash");
    }
    // perception retrained on the robot's own frames and sounds (tools/life/train_perception.py)
    VisionNet::Params vp;
    if (loadLayers(weights_, "vision", VisionNet::layerInfo(), vp)) {
        camera_.useVisionWeights(vp);
        ESP_LOGI(kTag, "weights: vision from flash");
    }
    AuditoryNet::Params ap;
    if (loadLayers(weights_, "auditory", AuditoryNet::layerInfo(), ap)) {
        loop_.auditory().use(ap);
        ESP_LOGI(kTag, "weights: auditory from flash");
    }
}

// New weights over Wi-Fi are switched in here, between two ticks, and watched for a minute: a net that produces
// NaN on them reboots the robot (the trial then counts that boot; two and the old weights come back). After a clean
// minute they are confirmed.
void RobotRuntime::superviseWeights(uint32_t now)
{
    const uint32_t faults = loop_.brain().policyFaults() + world_.nanCount() + actor_.nanCount() +
                            camera_.net().nanCount() + loop_.auditory().nanCount();
    // the camera task may still be inside a frame on the old arrays: they are unmapped a moment later
    if (release_old_ms_ && now - release_old_ms_ >= 1000) {
        weights_.releaseOld();
        release_old_ms_ = 0;
    }
    if (weights_.takeSwitchRequest()) {
        if (weights_.open()) {
            applyWeights();
            release_old_ms_ = now | 1;
            learner_.reset();  // the night learned the same days from the diary: the day's corrections start again
            ESP_LOGI(kTag, "weights: switched to slot %c (version %u) without a reboot, on trial for a minute",
                     weights_.slot(), static_cast<unsigned>(weights_.version()));
        }
        weights_trial_ms_ = now;
        weights_faults_ = faults;
    }
    if (!weights_.onTrial()) return;
    if (faults > weights_faults_ + 3) {
        ESP_LOGW(kTag, "weights: the nets fail on the new weights (%u NaN): reboot, the trial counts it",
                 static_cast<unsigned>(faults - weights_faults_));
        vTaskDelay(pdMS_TO_TICKS(200));
        esp_restart();
    }
    if (now - weights_trial_ms_ >= 60000) weights_.confirm();
}

// A firmware that came over the air is on trial (bootloader rollback): after a healthy minute it stays. One that
// crashes or hangs before that boots back into the previous one.
void RobotRuntime::confirmFirmware(uint32_t now)
{
    if (firmware_confirmed_ || now < 60000) return;
    firmware_confirmed_ = true;
    const esp_partition_t* run = esp_ota_get_running_partition();
    esp_ota_img_states_t state = ESP_OTA_IMG_UNDEFINED;
    if (esp_ota_get_state_partition(run, &state) == ESP_OK && state == ESP_OTA_IMG_PENDING_VERIFY) {
        esp_ota_mark_app_valid_cancel_rollback();
        ESP_LOGI(kTag, "firmware in %s confirmed after a healthy minute", run->label);
    }
}

// GET /status of the update server (net/ota_server.h, tools/life/ota.py). Runs on the server's task: only values
// that are set once at boot or are single words.
int RobotRuntime::formatStatus(char* buf, int size)
{
    const esp_partition_t* run = esp_ota_get_running_partition();
    esp_ota_img_states_t state = ESP_OTA_IMG_UNDEFINED;
    esp_ota_get_state_partition(run, &state);
    return std::snprintf(buf, static_cast<size_t>(size),
                         "{\"firmware\":\"%s\",\"firmware_trial\":%s,\"weights_slot\":\"%c\",\"weights_version\":%u,"
                         "\"weights_trial\":%s,\"uptime_s\":%u,\"diary_sent\":%u,\"diary_dropped\":%u,\"mode\":\"%s\",\"touch_level\":%d}\n",
                         run ? run->label : "?", state == ESP_OTA_IMG_PENDING_VERIFY ? "true" : "false",
                         weights_.slot(), static_cast<unsigned>(weights_.version()),
                         weights_.onTrial() ? "true" : "false", static_cast<unsigned>(nowMs() / 1000),
                         static_cast<unsigned>(life_.sent()), static_cast<unsigned>(life_.dropped()),
                         mode_ == 3 ? "v4" : mode_ == 2 ? "life" : mode_ == 1 ? "collector" : "brain", touch_.sensitivity());
}

// GET /snapshot, /audio of the update server (on its task): what the perception networks see and hear, for their
// training on the PC (the owner allowed frames and sound to the home PC, 2026-10-17).
int RobotRuntime::formatMedia(int kind, uint8_t* buf, int cap)
{
    if (kind == 0) return camera_.snapshot(buf, cap, 1500);
    struct __attribute__((packed)) Head {
        char magic[4];
        uint32_t rate, samples, feat_ticks, feat_dim, ms;
    } head{{'A', 'U', 'D', '1'}, MicrophoneEngine::kSampleRate, 0, 0, 100, nowMs()};
    const int max_samples = (cap - static_cast<int>(sizeof(head)) - 40 * 100 * 4) / 2;
    if (max_samples <= 0) return 0;
    int16_t* pcm = reinterpret_cast<int16_t*>(buf + sizeof(head));
    head.samples = static_cast<uint32_t>(mic_.recentSamples(pcm, std::min(max_samples, MicrophoneEngine::kRingSamples)));
    float* feat = reinterpret_cast<float*>(buf + sizeof(head) + head.samples * 2);
    const uint32_t k = audio_feat_n_.load();
    const uint32_t n = audio_feat_ ? std::min<uint32_t>(k, 40) : 0;
    for (uint32_t i = 0; i < n; ++i) std::copy(audio_feat_ + ((k - n + i) % 40) * 100, audio_feat_ + ((k - n + i) % 40) * 100 + 100, feat + i * 100);
    head.feat_ticks = n;
    std::memcpy(buf, &head, sizeof(head));
    return static_cast<int>(sizeof(head) + head.samples * 2 + n * 100 * 4);
}

// The diary (net/life_log.h): what the policy saw and did in every brain tick.
void RobotRuntime::logTick(uint32_t now)
{
    auto& brain = loop_.brain();
    const uint32_t tick = brain.tickCount();
    if (tick == logged_tick_) return;
    logged_tick_ = tick;
    static float values[LifeLog::kValues];
    float* p = std::copy(brain.lastPolicyInput().begin(), brain.lastPolicyInput().end(), values);
    p = std::copy(brain.action().begin(), brain.action().end(), p);
    p = std::copy(brain.bus().eff.begin(), brain.bus().eff.end(), p);
    static_assert(WorldModel::kIn == hybrid::kPolicyInputs + hybrid::ACT_COUNT + hybrid::EFF_COUNT, "diary order");
    const int64_t t0 = esp_timer_get_time();
    world_.step(values);
    world_us_ = static_cast<uint32_t>(esp_timer_get_time() - t0);
    brain.motivation().logValues(p);
    world_.logValues(p + Motivation::kLogValues);
    world_.logValues(wm_log_);
    if (!audio_feat_) audio_feat_ = static_cast<float*>(heap_caps_malloc(40 * 100 * sizeof(float), MALLOC_CAP_SPIRAM));
    if (audio_feat_) {  // the Auditory NN's input of this tick, for GET /audio
        const auto& in = loop_.auditory().lastInput();
        const uint32_t k = audio_feat_n_.load();
        std::copy(in.begin(), in.begin() + 100, audio_feat_ + (k % 40) * 100);
        audio_feat_n_.store(k + 1);
    }
    brain.motivation().setWorldErrors(wm_log_[0], wm_log_[1]);  // curiosity in the next tick (motivation.h)
    learner_.logValues(p + Motivation::kLogValues + WorldModel::kLogValues);
    {  // bus v4: the perception networks' hidden layers, x 1/4 (ReLU activations reach ~20; the diary holds +-8)
        float* e = p + Motivation::kLogValues + WorldModel::kLogValues + OnlineLearner::kLogValues;
        for (float x : loop_.visionEmbedding()) *e++ = 0.25f * x;
        for (float x : loop_.auditory().hidden()) *e++ = 0.25f * x;
        *e++ = loop_.brain().assoc().expectation(AssocMemory::Talk);  // the third learned expectation (2026-10-17)
        // brain v4: senses, then the analyzer's report and the main network's decision (zeros until they run)
        e = std::copy(loop_.channels().begin(), loop_.channels().end(), e);
        const auto& v4b = loop_.brainV4();
        e = std::copy(v4b.snn().begin(), v4b.snn().end(), e);
        e = std::copy(v4b.nn().begin(), v4b.nn().end(), e);
        static_assert(LifeLog::kV4Values == v4::kSnnInputs + v4::SO_COUNT + v4::NO_COUNT, "diary order");
    }
    // flags bit 0: the action came from the collector's random decisions; bit 1: from the actor (mode life)
    // flags: 1 the collector's random decisions, 2 the actor (mode life), 4 brain v4 drives (mode v4)
    life_.tick(tick, now, static_cast<uint16_t>(mode_ == 1 ? 1 : mode_ == 2 ? 2 : mode_ == 3 ? 4 : 0), values);
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
        logTick(now);
        loop_.afterVoice(now);
        superviseServos(now);
        if (panel_recover_ms_ && msReached(now, panel_recover_ms_)) {
            panel_recover_ms_ = 0;
            face_.recoverPanel();
            ESP_LOGI(kTag, "screen re-initialised after the servo rail power cycle");
            next_panel_refresh_ms_ = now + 60000;
        }
        // Once a minute the panel is set up again, without a reset: a black screen from any power dip comes back by
        // itself (2026-10-13: black until the owner pressed the power button, once after a recovery).
        if (msReached(now, next_panel_refresh_ms_)) {
            next_panel_refresh_ms_ = now + 60000;
            face_.recoverPanel(false);
        }
        saveLearned(now);
        confirmFirmware(now);
        superviseWeights(now);

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
