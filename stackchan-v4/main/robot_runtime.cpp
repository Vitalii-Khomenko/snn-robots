#include "robot_runtime.h"
#include "core/ms_clock.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <bsp/esp-bsp.h>
#include <driver/gpio.h>
#include <driver/i2s_std.h>
#include <esp_attr.h>
#include <esp_heap_caps.h>
#include <esp_log.h>
#include <esp_core_dump.h>
#include <esp_ota_ops.h>
#include <esp_partition.h>
#include <esp_system.h>
#include <esp_task_wdt.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <nvs.h>
#include <nvs_flash.h>

#include "core/i2c_bus_recovery.h"
#include "power/power_rails.h"

namespace {
constexpr const char* kTag = "runtime";
constexpr uint32_t kMainLoopIntervalMs = 20;  // 50 Hz is enough; motion is capped at 50 Hz and the brain at 20 Hz
constexpr bool kCameraEnabled = true;
// Switches for hardware diagnosis (2026-09-27 servo investigation). Both
// on in normal use.
constexpr bool kMicrophoneEnabled = true;
constexpr bool kWifiEnabled = true;

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
    forgetOldProject();
    loadLearned();
    {
        const esp_partition_t* run = esp_ota_get_running_partition();
        esp_ota_img_states_t state = ESP_OTA_IMG_UNDEFINED;
        esp_ota_get_state_partition(run, &state);
        ESP_LOGI(kTag, "firmware in %s (%s)", run ? run->label : "?",
                 state == ESP_OTA_IMG_PENDING_VERIFY ? "new, on trial for a minute" : "confirmed");
    }

    ESP_LOGI(kTag, "brain v4: spiking analyzer %s, main network %s, auditory NN %s",
             loop_.brain().analyzer().trained() ? "trained" : "UNTRAINED", loop_.brain().trained() ? "trained" : "UNTRAINED",
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
        if (nvs_open("v4", NVS_READONLY, &h) == ESP_OK) {
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
        // brain v4 reads the accelerometer itself (shake, tilt channels); these events are for the log only
        imu_.onShake([]() { ESP_LOGI(kTag, "shake detected"); });
        imu_.onTilt([]() { ESP_LOGI(kTag, "tilt detected"); });
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

    ESP_LOGI(kTag, "power status");
    if (!power_status_.begin()) {
        ESP_LOGE(kTag, "power status begin failed");
    }

    // Wi-Fi (net/wifi_engine.h): station connect + NTP (the clock of the night rules and the tired channel) + the LAN
    // device ping (owner=near/away in the stats line). Non-blocking: association and sync happen in the background.
    ESP_LOGI(kTag, "wifi");
    if (!kWifiEnabled) {
        ESP_LOGW(kTag, "wifi disabled (diagnostic)");
    } else if (!wifi_.begin()) {
        ESP_LOGE(kTag, "wifi begin failed");
    } else {
        ota_.begin(learned_, [this](char* buf, int size) { return formatStatus(buf, size); },
                   [this](int level) {
                       if (!touch_.setSensitivity(level)) return false;
                       nvs_handle_t h;
                       if (nvs_open("v4", NVS_READWRITE, &h) == ESP_OK) {
                           nvs_set_u8(h, "touch", static_cast<uint8_t>(level));
                           nvs_commit(h);
                           nvs_close(h);
                       }
                       return true;
                   });
    }

    const uint32_t now = nowMs();
    // brain v4's int8 tables (236 KB) are read every tick: PSRAM is about twice as fast as this board's flash (DIO)
    loop_.brain().analyzer().useFastWeights();
    diagnostics_window_start_us_ = static_cast<uint64_t>(esp_timer_get_time());
    next_diagnostics_ms_ = now + 5000;

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
    loop_.servoRepowered(last_repower_ms_);  // the head's jolt is not a shake
    stalled_windows_ = 0;
    ++servo_repowers_;
    // the last three power cycles (oldest first): three within 3 minutes are a series
    repower_times_[0] = repower_times_[1];
    repower_times_[1] = repower_times_[2];
    repower_times_[2] = last_repower_ms_;
    blocked_logged_ = false;
    panel_recover_ms_ = last_repower_ms_ + 1000;  // when the rail has settled
}

void RobotRuntime::logStats(uint32_t now)
{
    const uint32_t servo_sent = motion_.takePacketCount();
    uint32_t imu_reads = 0, imu_fails = 0;
    float gyro_max = 0.0f;
    imu_.takeHealth(imu_reads, imu_fails, gyro_max);
    const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
    const uint32_t window_us = static_cast<uint32_t>(now_us - diagnostics_window_start_us_);
    // The voice streams from its own task, so all of the main loop's busy time is work.
    const unsigned busy = window_us ? static_cast<unsigned>(diagnostics_busy_us_ * 1000ULL / window_us) : 0;
    float head_yaw = 0.0f, head_pitch = 0.0f;
    motion_.getCurrentDeg(head_yaw, head_pitch);
    auto pct = [](float x) { return static_cast<unsigned>(std::clamp(x, 0.0f, 1.0f) * 100.0f + 0.5f); };
    uint32_t face_frames = 0, face_max_us = 0;
    face_.takeDrawStats(face_frames, face_max_us);
    const unsigned voice_permille = window_us ? static_cast<unsigned>(vocal_.takeRenderUs() * 1000ULL / window_us) : 0;
    // the machine: CPU, memory, timing, servos, sensors, links
    ESP_LOGI(kTag,
             "stats: busy=%u.%u%% internal_free=%uB internal_min=%uB psram_free=%uB stack_headroom=%uB wifi=%s time=%s "
             "owner=%s tick_us=%u tick_max_us=%u missed_ticks=%u tick_ms=%u audio_us=%u mic_block_us=%u cam_fps=%.1f "
             "cam_us=%u audio_nan=%u stamina=%u%% travel=%udeg battery=%u%% head=%d,%d servo_packets=%u cmd_travel=%.0fdeg "
             "real_rotation=%.0fdeg imu=%u/%u gyro_max=%.0f gyro_restarts=%u servo_repowers=%u face_fps=%u "
             "face_frame_max_us=%u mic_peak=%d/%d/%d voice=%s voice_gaps=%u voice_load=%u.%u%% touch_storms=%u diary=%u/%u",
             busy / 10, busy % 10, static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
             static_cast<unsigned>(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)),
             static_cast<unsigned>(uxTaskGetStackHighWaterMark(NULL) * 4), wifi_.isConnected() ? "up" : "down",
             wifi_.isTimeSynced() ? "synced" : "unsynced", wifi_.knownDeviceNearby() ? "near" : "away",
             static_cast<unsigned>(loop_.tickLastUs()), static_cast<unsigned>(loop_.takeTickMaxUs()),
             static_cast<unsigned>(loop_.missedTicks()), static_cast<unsigned>(loop_.tickPeriodMs()),
             static_cast<unsigned>(loop_.audioUs()), static_cast<unsigned>(mic_.audioFront().blockUs()),
             static_cast<double>(camera_.takeFrameCount() * 1e6f / static_cast<float>(window_us)),
             static_cast<unsigned>(camera_.frontUs()), static_cast<unsigned>(loop_.auditory().nanCount()),
             pct(motion_.staminaLevel()), static_cast<unsigned>(motion_.travelledDeg()),
             pct(power_status_.batteryLevel01()), static_cast<int>(head_yaw), static_cast<int>(head_pitch),
             static_cast<unsigned>(servo_sent), motion_.travelledDeg() - last_travel_deg_, imu_.takeRotationDeg(),
             static_cast<unsigned>(imu_reads), static_cast<unsigned>(imu_fails), static_cast<double>(gyro_max),
             static_cast<unsigned>(imu_.gyroRestarts()), static_cast<unsigned>(servo_repowers_),
             static_cast<unsigned>(face_frames / 5), static_cast<unsigned>(face_max_us), mic_.takePhasePeak(0),
             mic_.takePhasePeak(1), mic_.takePhasePeak(2), vocal_.trained() ? "babble" : "silent",
             static_cast<unsigned>(vocal_.gaps()), voice_permille / 10, voice_permille % 10,
             static_cast<unsigned>(loop_.stormDropped()), static_cast<unsigned>(life_.sent()),
             static_cast<unsigned>(life_.dropped()));
    {
        // the brain: the 13 channels now, the raw light / proximity readings, the analyzer's mood, what was done
        const auto& c = loop_.channels();
        auto& b = loop_.brain();
        const auto& so = b.snn();
        const auto& ex = loop_.executor();
        ESP_LOGI(kTag, "v4: als=%u ir=%u ps=%u/%02x near=%.0f ch=%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f/%.2f "
                 "snn=%s/%uus/rate%.3f val=%.2f aro=%.2f under=%.2f over=%.2f threat=%.2f nn=%s/faults%u drives=%d "
                 "night=%d face=%s calls=%u nods=%u shakes=%u gestures=%u learned=%u%s",
                 static_cast<unsigned>(presence_.alsRaw()), static_cast<unsigned>(presence_.alsIrRaw()),
                 static_cast<unsigned>(presence_.lastRaw()), static_cast<unsigned>(presence_.psHigh()),
                 static_cast<double>(presence_.nearOverBaseline()), c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8],
                 c[9], c[10], c[11], c[12], b.analyzer().trained() ? "on" : "untrained",
                 static_cast<unsigned>(b.analyzer().lastUs()), static_cast<double>(b.analyzer().mixRate()),
                 so[v4::SO_VALENCE], so[v4::SO_AROUSAL], so[v4::SO_UNDER], so[v4::SO_OVER], so[v4::SO_THREAT],
                 b.trained() ? "on" : "untrained", static_cast<unsigned>(b.nnFaults()), loop_.driving() ? 1 : 0,
                 loop_.night() ? 1 : 0, emotionName(static_cast<Emotion>(ex.shownEmotion())),
                 static_cast<unsigned>(ex.calls()), static_cast<unsigned>(ex.nods()), static_cast<unsigned>(ex.shakes()),
                 static_cast<unsigned>(ex.gestures()), static_cast<unsigned>(learned_.version()),
                 learned_.onTrial() ? "(trial)" : "");
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
    next_diagnostics_ms_ = now + 5000;
}

// The diary (net/life_log.h): every brain tick in the order of tools/v4/spec_v4.py DIARY.
void RobotRuntime::logTick()
{
    const uint32_t tick = loop_.tickCount();
    if (tick == logged_tick_) return;
    logged_tick_ = tick;
    static float values[LifeLog::kValues];
    const auto& r = loop_.raw();
    const float raw[] = {r.energy, r.speech, r.sharpEv, r.headEv, r.headHold, r.screenEv, r.accelSd, r.imuPitch,
                         r.imuRoll, r.imuOk ? 1.0f : 0.0f, r.nearRaw, r.nearOk ? 1.0f : 0.0f, r.als,
                         r.alsOk ? 1.0f : 0.0f, r.motionFrac, r.motionX, r.motionY, r.frameOk ? 1.0f : 0.0f,
                         r.battery, r.stamina, r.hour, r.clockOk ? 1.0f : 0.0f};
    static_assert(sizeof(raw) / sizeof(float) == v4::RAW_COUNT, "diary order: tools/v4/spec_v4.py RAW");
    float* p = std::copy(std::begin(raw), std::end(raw), values);
    p = std::copy(loop_.channels().begin(), loop_.channels().end(), p);
    p = std::copy(loop_.brain().snn().begin(), loop_.brain().snn().end(), p);
    p = std::copy(loop_.brain().nn().begin(), loop_.brain().nn().end(), p);
    float yaw = 0, pitch = 0;
    motion_.getCurrentDeg(yaw, pitch);
    *p++ = yaw;
    *p++ = pitch;
    *p++ = static_cast<float>(loop_.executor().shownEmotion());
    *p++ = loop_.saidThisTick() ? 1.0f : 0.0f;
    *p++ = loop_.gestureThisTick() ? 1.0f : 0.0f;
    p = std::copy(loop_.brain().lastPrev().begin(), loop_.brain().lastPrev().end(), p);
    static_assert(v4::RAW_COUNT + v4::kSnnInputs + v4::SO_COUNT + v4::NO_COUNT + v4::BODY_COUNT + v4::PV_COUNT ==
                      LifeLog::kValues,
                  "diary order: tools/v4/spec_v4.py DIARY");
    // flags: bit 0 brain v4 drives the body, bit 1 night rules, bit 2 the main network decided this tick
    life_.tick(tick, nowMs(),
               static_cast<uint16_t>((loop_.driving() ? 1 : 0) | (loop_.night() ? 2 : 0) | (loop_.decidedThisTick() ? 4 : 0)),
               values);
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

// A clean start for brain v4 (2026-10-24): what the earlier brains left in flash is removed once - the NVS namespaces
// of v3 / platform B ("life": mode and touch level, "assoc": the associative memory and the needs, "weights": the
// weights slot) and their weights blobs (magic "SCW1") in weights_a / weights_b. Only the owner's head pad
// sensitivity is carried over (into "v4"). Does nothing once it is all gone.
void RobotRuntime::forgetOldProject()
{
    nvs_handle_t h;
    if (nvs_open("life", NVS_READWRITE, &h) == ESP_OK) {
        uint8_t level = 0;
        if (nvs_get_u8(h, "touch", &level) == ESP_OK) {
            nvs_handle_t v;
            if (nvs_open("v4", NVS_READWRITE, &v) == ESP_OK) {
                nvs_set_u8(v, "touch", level);
                nvs_commit(v);
                nvs_close(v);
            }
        }
        nvs_close(h);
    }
    int removed = 0;
    for (const char* ns : {"life", "assoc", "weights"}) {
        if (nvs_open(ns, NVS_READWRITE, &h) != ESP_OK) continue;  // not there
        nvs_erase_all(h);
        nvs_commit(h);
        nvs_close(h);
        ESP_LOGI(kTag, "clean start: old NVS namespace '%s' erased", ns);
        ++removed;
    }
    for (const char* label : {"weights_a", "weights_b"}) {
        const esp_partition_t* part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, label);
        char magic[4] = {};
        if (!part || esp_partition_read(part, 0, magic, sizeof(magic)) != ESP_OK || std::memcmp(magic, "SCW1", 4) != 0)
            continue;
        if (esp_partition_erase_range(part, 0, part->erase_size) == ESP_OK) {
            ESP_LOGI(kTag, "clean start: platform B's weights blob in %s erased", label);
            ++removed;
        }
    }
    if (removed) ESP_LOGI(kTag, "clean start: %d leftovers of the earlier brains removed", removed);
}

// What the robot learned from its life (stage 6, net/learned_store.h): the blob in flash, or the innate networks.
void RobotRuntime::loadLearned()
{
    if (!learned_.open()) return;
    const int n = loop_.useLearned(learned_.view());
    ESP_LOGI(kTag, "learned: version %u (%d arrays: %s%s)", static_cast<unsigned>(learned_.version()), n,
             n >= 4 ? "action heads" : "innate heads", n % 4 == 1 ? ", zone shifts" : "");
}

// A new blob over Wi-Fi is switched in here, between two ticks, without a reboot, and watched for a minute: if the
// main network fails on it (non-finite outputs), the robot goes back to the blob it had - also without a reboot (the
// owner sleeps next to it when the night sleep sends one). After a clean minute it is confirmed.
void RobotRuntime::superviseLearned(uint32_t now)
{
    const uint32_t faults = loop_.brain().nnFaults();
    if (release_old_ms_ && now - release_old_ms_ >= 1000) {
        learned_.releaseOld();
        release_old_ms_ = 0;
    }
    if (learned_.takeSwitchRequest()) {
        learned_.open();
        loop_.useLearned(learned_.view());
        release_old_ms_ = now | 1;
        learned_trial_ms_ = now;
        learned_faults_ = faults;
        ESP_LOGI(kTag, "learned: switched to version %u (slot %c) without a reboot, on trial for a minute",
                 static_cast<unsigned>(learned_.version()), learned_.slot());
    }
    if (!learned_.onTrial()) return;
    if (faults > learned_faults_ + 3) {
        ESP_LOGW(kTag, "learned: the main network fails on version %u (%u faults): back to the previous one",
                 static_cast<unsigned>(learned_.version()), static_cast<unsigned>(faults - learned_faults_));
        learned_.revert();
        learned_.open();
        loop_.useLearned(learned_.view());
        release_old_ms_ = now | 1;
        return;
    }
    if (now - learned_trial_ms_ >= 60000) learned_.confirm();
}

// GET /status of the update server (net/ota_server.h, tools/robot/ota.py). Runs on the server's task: only values
// that are set once at boot or are single words.
int RobotRuntime::formatStatus(char* buf, int size)
{
    const esp_partition_t* run = esp_ota_get_running_partition();
    esp_ota_img_states_t state = ESP_OTA_IMG_UNDEFINED;
    esp_ota_get_state_partition(run, &state);
    return std::snprintf(buf, static_cast<size_t>(size),
                         "{\"firmware\":\"%s\",\"firmware_trial\":%s,\"brain\":\"v4\",\"learned_version\":%u,"
                         "\"learned_slot\":\"%c\",\"learned_trial\":%s,\"uptime_s\":%u,\"diary_sent\":%u,"
                         "\"diary_dropped\":%u,\"touch_level\":%d}\n",
                         run ? run->label : "?", state == ESP_OTA_IMG_PENDING_VERIFY ? "true" : "false",
                         static_cast<unsigned>(learned_.version()), learned_.slot(), learned_.onTrial() ? "true" : "false",
                         static_cast<unsigned>(nowMs() / 1000), static_cast<unsigned>(life_.sent()),
                         static_cast<unsigned>(life_.dropped()), touch_.sensitivity());
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
        loop_.step(now, env);
        logTick();
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
        confirmFirmware(now);
        superviseLearned(now);

        const uint64_t loop_elapsed_us = static_cast<uint64_t>(esp_timer_get_time()) - loop_start_us;
        diagnostics_busy_us_ += loop_elapsed_us;
        if (msReached(now, next_diagnostics_ms_)) logStats(now);
        vTaskDelay(pdMS_TO_TICKS(kMainLoopIntervalMs));
    }
}
