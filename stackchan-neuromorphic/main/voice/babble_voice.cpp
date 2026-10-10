#include "babble_voice.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <esp_log.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

using namespace hybrid;

bool BabbleVoice::begin()
{
    if (xTaskCreatePinnedToCore(&BabbleVoice::taskEntry, "babble_voice", 6144, this, 5, nullptr, 1) != pdPASS) {
        ESP_LOGE("voice", "audio task creation failed");
        return false;
    }
    return true;
}

void BabbleVoice::taskEntry(void* arg) { static_cast<BabbleVoice*>(arg)->taskLoop(); }

bool BabbleVoice::step(const PolicyGru::Action& a, bool allow, int emotion, bool preempt)
{
    ++since_ticks_;
    const float act = a[ACT_SOUND_ACT];
    if (act < 0.3f) armed_ = true;
    const bool free_now = !active_.load() && !pending_.load();
    // a reaction (touch, shake, ...) is answered once, even in the middle of an idle phrase: it stays open from its onset
    // until it is answered or the reaction window (BodyExecutor) closes - the network's answer may come a tick later
    if (preempt && !prev_preempt_) reaction_open_ = true;
    if (!preempt) reaction_open_ = false;
    prev_preempt_ = preempt;
    const bool onset = reaction_open_;
    const bool may = onset ? (act > 0.55f && allow && !pending_.load() && since_ticks_ >= 3)
                           : (armed_ && act > 0.55f && allow && free_now && since_ticks_ >= kRefractoryTicks);
    if (!may) return false;
    reaction_open_ = false;
    if (onset && active_.load()) interrupt_.store(true);  // cut the idle phrase: the reaction comes first
    armed_ = false;
    since_ticks_ = 0;
    ++calls_;
    pending_intent_ = {a[ACT_SOUND_VALENCE], a[ACT_SOUND_AROUSAL], a[ACT_VOICE_REGISTER], a[ACT_VOICE_SLOPE], a[ACT_VOICE_EXPR],
                       a[ACT_VOICE_LENGTH], a[ACT_VOICE_RATE], a[ACT_STYLE_CALM], a[ACT_STYLE_ASK], a[ACT_STYLE_PLAY],
                       a[ACT_STYLE_GRUMBLE], 0.0f};  // BabbleNet's 'sing' input: no songs (2026-10-04)
    pending_emotion_ = emotion;
    pending_base_hz_ = 520.0f * std::exp2(1.2f * std::clamp(a[ACT_VOICE_REGISTER], 0.0f, 1.0f));  // droid register
    base_hz_ = pending_base_hz_;
    active_.store(true);
    pending_.store(true, std::memory_order_release);
    return true;
}

void BabbleVoice::startPhrase()
{
    net_.begin(pending_intent_, pending_emotion_);
    text_[0] = '\0';
    phrase_active_ = true;
}

// Keeps a quarter second of syllables ahead of the sound (audio task).
void BabbleVoice::topUp()
{
    while (phrase_active_ && synth_.queuedMs() < 250) {
        droid::Note s;
        if (!net_.next(s)) {
            phrase_active_ = false;
            ended_.store(true, std::memory_order_release);
            break;
        }
        if (!synth_.enqueue(s, pending_base_hz_)) break;
        // phrase text: W whistle, C chirp, P pip, B burble, T trill, V warble, G giggle, S squeal, R growl, O coo; + rising, - falling; ? ends upward, . ends downward
        static const char kKind[] = {'W', 'C', 'P', 'B', 'T', 'V', 'G', 'S', 'R', 'O'};
        char one[5];
        int n = 0;
        one[n++] = kKind[s.kind];
        if (s.slopeSt > 1.0f) one[n++] = '+';
        else if (s.slopeSt < -1.0f) one[n++] = '-';
        if (s.ending == droid::E_UP) one[n++] = '?';
        else if (s.ending == droid::E_DOWN) one[n++] = '.';
        one[n] = '\0';
        const size_t len = std::strlen(text_);
        if (len + 6 < sizeof(text_)) std::snprintf(text_ + len, sizeof(text_) - len, "%s%s", len ? " " : "", one);
    }
}

void BabbleVoice::taskLoop()
{
    int64_t last_start_us = 0;
    while (true) {
        if (interrupt_.exchange(false)) {
            phrase_active_ = false;
            synth_.clear();
        }
        if (pending_.exchange(false, std::memory_order_acquire)) {
            startPhrase();
            // step() set it already, but the idle branch below may have cleared it in between (it read pending_ just
            // before step() raised it): a phrase played with active() false is heard by the robot as the world
            active_.store(true);
        }
        topUp();
        if (synth_.active()) {
            const int64_t t0 = esp_timer_get_time();
            // 20 ms chunks: a start-to-start spacing over 30 ms means the speaker ran dry
            if (last_start_us && t0 - last_start_us > 30000) gaps_.fetch_add(1);
            last_start_us = t0;
            synth_.render(chunk_.data(), chunk_.size());
            render_us_.fetch_add(static_cast<uint32_t>(esp_timer_get_time() - t0));
            audio_.playPcm(chunk_.data(), chunk_.size(), 90);  // paced by the DMA: blocks ~20 ms
        } else {
            last_start_us = 0;
            if (!phrase_active_ && !pending_.load()) active_.store(false);
            vTaskDelay(pdMS_TO_TICKS(10));
        }
    }
}
