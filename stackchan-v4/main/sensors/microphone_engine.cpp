#include "microphone_engine.h"

#include <esp_heap_caps.h>

#include <algorithm>
#include <cmath>
#include <array>
#include <cstdlib>
#include <bsp/esp-bsp.h>
#include <esp_log.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

namespace {
constexpr const char* kTag = "mic";
}

bool MicrophoneEngine::begin()
{
    mic_ = bsp_audio_codec_microphone_init();
    if (!mic_) {
        ESP_LOGE(kTag, "mic codec init failed");
        return false;
    }

    esp_codec_dev_sample_info_t fs = {};
    fs.bits_per_sample              = 16;
    fs.channel                      = 2;  // both ES7210 inputs (the filter bank reports their balance)
    fs.channel_mask                 = 0;
    fs.sample_rate                  = kSampleRate;
    fs.mclk_multiple                = 0;

    if (esp_codec_dev_open(mic_, &fs) != 0) {
        ESP_LOGE(kTag, "esp_codec_dev_open failed");
        return false;
    }
    // Input gain defaults to ~0 dB; the ES7210 mics are inaudibly quiet
    // without this boost (see docs/HARDWARE.md "microphones: done").
    esp_codec_dev_set_in_gain(mic_, 60.0f);

    BaseType_t ok = xTaskCreatePinnedToCore(&MicrophoneEngine::taskEntry, "mic_engine", 4096, this, 3, nullptr, 1);
    if (ok != pdPASS) {
        ESP_LOGE(kTag, "failed to start mic task");
        return false;
    }
    return true;
}

void MicrophoneEngine::suppressFor(uint32_t ms)
{
    uint32_t until = static_cast<uint32_t>(esp_timer_get_time() / 1000) + ms;
    ignore_until_ms_.store(until);
    // A block captured just before suppression can already be waiting for
    // update(). Do not let that stale self-speaker event escape after a sound
    // starts; future blocks are filtered by ignore_until_ms_ in taskLoop().
    loud_sound_pending_.store(false);
}

void MicrophoneEngine::raiseThresholdFor(uint32_t ms)
{
    raised_until_ms_.store(static_cast<uint32_t>(esp_timer_get_time() / 1000) + ms);
}

void MicrophoneEngine::taskEntry(void* arg)
{
    static_cast<MicrophoneEngine*>(arg)->taskLoop();
}

void MicrophoneEngine::taskLoop()
{
    std::array<int16_t, kChunkSamplesPerChannel * 2> buf{};  // interleaved L R

    while (true) {
        // blocks on the I2S DMA: paces the task
        if (esp_codec_dev_read(mic_, buf.data(), buf.size() * sizeof(int16_t)) != 0) {
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        int32_t peak = 0;
        int32_t total_energy = 0;
        for (const int16_t v : buf) {
            const int32_t a = std::abs(static_cast<int32_t>(v));
            peak = std::max(peak, a);
            total_energy += a;
        }

        front_.processBlock(buf.data(), kChunkSamplesPerChannel);

        const uint32_t now = static_cast<uint32_t>(esp_timer_get_time() / 1000);
        const bool own_motion = msWithin(now, raised_until_ms_.load());
        const int loud_threshold = own_motion ? kLoudThresholdOwnMotion : kLoudThreshold;
        const bool muted_now = muted(now);
        std::atomic<int>& peak_slot = phase_peak_[std::clamp(motion_phase_.load(), 0, 2)];
        if (!muted_now && peak > peak_slot.load()) peak_slot.store(peak);

        // Onset (see takeOnset()).
        const float mean_abs = static_cast<float>(total_energy) / (2.0f * kChunkSamplesPerChannel);
        const int onset_peak = own_motion ? kLoudThresholdOwnMotion : kOnsetPeak;
        if (!muted_now && peak >= onset_peak && mean_abs > kOnsetRatio * onset_slow_ &&
            now - last_onset_ms_ >= kOnsetRefractoryMs) {
            last_onset_ms_ = now;
            const uint32_t head = onset_head_.load(std::memory_order_relaxed);
            Onset o;
            o.ms = now;
            o.strength = std::clamp(mean_abs / (onset_slow_ + 1.0f) / 10.0f, 0.3f, 1.0f);
            onsets_[head % onsets_.size()] = o;
            onset_head_.store(head + 1, std::memory_order_release);
        }
        onset_slow_ += (mean_abs - onset_slow_) * 0.05f;  // ~100 ms

        if (peak >= loud_threshold && !muted_now && !loud_sound_pending_.load() &&
            now - last_trigger_ms_ >= kRetriggerCooldownMs) {
            last_trigger_ms_ = now;
            // latched here, at the triggering block, not when update() runs
            last_loud_peak_.store(peak);
            last_loud_own_motion_.store(own_motion);
            loud_sound_pending_.store(true);
        }
        vTaskDelay(0);  // yield (the read above already paces the task)
    }
}

bool MicrophoneEngine::takeOnset(Onset& out)
{
    const uint32_t head = onset_head_.load(std::memory_order_acquire);
    if (head - onset_tail_ > onsets_.size()) onset_tail_ = head - onsets_.size();  // overrun: skip the oldest
    if (onset_tail_ == head) return false;
    out = onsets_[onset_tail_ % onsets_.size()];
    ++onset_tail_;
    return true;
}

void MicrophoneEngine::update()
{
    if (loud_sound_pending_.exchange(false) && on_loud_sound_) {
        on_loud_sound_();
    }
}
