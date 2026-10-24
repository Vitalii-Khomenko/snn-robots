#include "brain_loop.h"
#include "ms_clock.h"

#include <algorithm>
#include <cmath>
#include <esp_log.h>
#include <esp_timer.h>

namespace {
constexpr const char* kTag = "runtime";
}

BrainLoop::BrainLoop(Engines e) : e_(e), exec_(e.face, e.led, e.vocal, e.motion)
{
    brain_.analyzer().setClock([]() -> int64_t { return esp_timer_get_time(); });
    brain_.network().setClock([]() -> int64_t { return esp_timer_get_time(); });
}

// A storm of touches with nobody around is the pad, not a hand: it fired every 2-4 s all night while the owner slept
// (2026-10-13; the pad sits next to the pitch servo). A first version (6 in a minute, nobody in this tick) dropped
// the owner's own petting the same evening - he stroked it from above, out of the camera, silent. Now: more than 8
// touches in 10 minutes, and nobody heard, seen moving or near in all those 10 minutes (the nightly phantoms come
// ~15 in 10 min; whoever pets the robot was heard or seen coming up to it).
bool BrainLoop::touchStorm(uint32_t now)
{
    touch_times_[touch_n_++ % touch_times_.size()] = now;
    const uint32_t oldest = touch_times_[touch_n_ % touch_times_.size()];  // 9 touches back
    const bool quiet_10min = now - last_somebody_ms_ > 600000;
    if (touch_n_ <= touch_times_.size() || now - oldest >= 600000 || !quiet_10min) return false;
    if (++storm_dropped_ == 1 || now - storm_logged_ms_ > 60000) {
        ESP_LOGW(kTag, "head touch storm with nobody around for 10 min: dropped as pad noise (%u so far)",
                 static_cast<unsigned>(storm_dropped_));
        storm_logged_ms_ = now;
    }
    return true;
}

void BrainLoop::headTouch(uint32_t now)
{
    // The Si12T pad sits next to the pitch servo: a moving head couples into it as a level 2-3 phantom "touch" (seen
    // on hardware). Only fast movement (saccade, nod) and 300 ms after it: slow drift must not make real touches
    // disappear. Maybe a servo phantom, maybe a hand on the head while it moves: decided in step() - a brief blip is
    // dropped, a touch that is held long enough is real.
    if (e_.motion.isMovingFast(now)) {
        if (pending_touch_ms_ == 0) pending_touch_ms_ = now ? now : 1;
        return;
    }
    if (touchStorm(now)) return;
    acceptHeadTouch(now);
    ESP_LOGI(kTag, "head touch (level %u)", static_cast<unsigned>(e_.touch.pressLevel()));
}

void BrainLoop::acceptHeadTouch(uint32_t now)
{
    // a firmer press weighs more (pad levels 1..3)
    const unsigned level = e_.touch.pressLevel();
    head_ev_ = std::max(head_ev_, level >= 3 ? 1.3f : level == 2 ? 1.0f : 0.7f);
    touch_ms_ = now ? now : 1;
}

void BrainLoop::vision(const CameraEngine::VisionFrame& f, uint32_t now)
{
    // a frame taken while our own head turned shows the room sliding past: not the world moving
    frame_ok_ = !f.blind && !msWithin(now, own_motion_until_ms_);
    motion_[0] = f.motionFrac;
    motion_[1] = f.motionX;
    motion_[2] = f.motionY;
    frame_this_tick_ = true;
}

void BrainLoop::coupleMotion(uint32_t now)
{
    // Motion phase for the microphone's own-noise measurement: servo clicks are expected at the start and the stop of
    // every movement.
    const bool moving = e_.motion.isMoving();
    if (moving != was_moving_) {
        was_moving_ = moving;
        motion_edge_ms_ = now;
    }
    e_.mic.setMotionPhase(now - motion_edge_ms_ < 400 ? 1 : moving ? 2 : 0);
    if (moving) {
        // The IMU and the microphone hear our own servos: raise their thresholds while the head moves and 350 ms
        // after (the servo stops ~120 ms after the last packet).
        e_.mic.raiseThresholdFor(350);
        own_motion_until_ms_ = now + 350;
    }
    e_.imu.update(moving);
    MicrophoneEngine::Onset onset;
    while (e_.mic.takeOnset(onset)) soundOnset(std::clamp(onset.strength, 0.0f, 1.0f));
}

// The ear, once per tick: loudness, speech and the sharp-sound event of this tick. Our own voice and servos are
// flagged so the network can tell the world from the robot.
void BrainLoop::hear(uint32_t now, float& sharpEv)
{
    const auto self = e_.audio.takeSelfSound();
    if (e_.vocal.active()) self_voice_until_ms_ = now + 300;
    AuditoryNet::Context ctx;
    ctx.selfSpeaking = msWithin(now, self_voice_until_ms_);
    ctx.ownMotion = msWithin(now, own_motion_until_ms_);
    ctx.selfLevel01 = self[0];
    // Our own servos click and whine loud enough to look like claps (peaks of 20-32k on hardware): while the head
    // moves, and just after, such events are mostly ourselves and are not passed on at all.
    if (ctx.ownMotion) loud_event_ = onset_event_ = 0.0f;
    // A sharp sound that started this tick: the microphone's loud or onset event (the Auditory NN's clap / sharp
    // classes fire 20-40 times an hour on nothing - tools/v4/calibrate.py, 2026-10-24). Loud events come with
    // intensity 0.4 (peak ~2000: a tap on the table, a word) .. 1.0 (peak >= 20000, a clap at the robot); only peaks
    // from ~6000 are sharp sounds (2026-10-24: 9 weak ones in 2 minutes of the owner moving about kept the channel at
    // 1.0 and the analyzer's threat at 0.8).
    sharpEv = loud_event_ >= 0.6f ? 0.5f + (loud_event_ - 0.6f) / 0.4f : onset_event_ > 0.5f ? onset_event_ : 0.0f;
    // the boot's own click (amplifier / servo power on: peak 32768 at 4.4 s, 2026-10-24) is not a bang at the robot
    if (now < 10000) sharpEv = 0.0f;
    // the network was trained on events that fade over ~3 ticks: hold them the same way
    onset_hold_ = std::max(onset_event_, onset_hold_ * 0.6f);
    loud_hold_ = std::max(loud_event_, loud_hold_ * 0.6f);
    ctx.onset01 = onset_hold_ > 0.05f ? onset_hold_ : 0.0f;
    ctx.loud01 = loud_hold_ > 0.05f ? loud_hold_ : 0.0f;
    onset_event_ = loud_event_ = 0.0f;
    const int64_t t0 = esp_timer_get_time();
    auditory_.run(e_.mic.audioFront(), ctx, ear_);
    const uint32_t us = static_cast<uint32_t>(esp_timer_get_time() - t0);
    audio_us_ = audio_us_ ? (audio_us_ * 7 + us) / 8 : us;
    // what the ear makes of it, once per sound (rising edge), for live tests
    const float heard[3] = {ear_[AUD_CLAP], ear_[AUD_SHARP], ear_[AUD_SPEECH]};
    static const char* const names[3] = {"clap", "sharp", "speech"};
    for (int i = 0; i < 3; ++i) {
        if (heard[i] > 0.5f && !heard_[i])
            ESP_LOGI(kTag, "hear: %s %.2f (energy %.2f%s)", names[i], static_cast<double>(heard[i]),
                     static_cast<double>(ear_[AUD_ENERGY]), ctx.ownMotion ? ", own motion" : "");
        heard_[i] = heard[i] > 0.35f ? (heard[i] > 0.5f || heard_[i]) : false;
    }
}

// Brain v4's senses (v4/channel_encoder.h), once per tick after the ear ran.
void BrainLoop::updateChannels(uint32_t now, float sharpEv, const Env& env)
{
    ChannelEncoder::Raw& r = raw_;
    r.energy = ear_[AUD_ENERGY];
    r.speech = ear_[AUD_SPEECH];
    r.sharpEv = sharpEv;
    r.headEv = head_ev_;
    // A finger that stays on the pad after an accepted touch keeps "touching" (petting); a contact that never became an
    // accepted touch (a servo phantom) does not.
    const bool contact = e_.touch.contactNow() && touch_ms_ != 0 && now - touch_ms_ < 3000;
    if (contact) touch_ms_ = now ? now : 1;
    r.headHold = contact ? 1.0f : 0.0f;
    r.screenEv = screen_ev_;
    head_ev_ = screen_ev_ = 0.0f;
    float ax, ay, az;
    e_.imu.getAccel(ax, ay, az);
    {
        accel_[accel_n_++ % accel_.size()] = std::sqrt(ax * ax + ay * ay + az * az) / 9.81f;
        float m = 0.0f, q = 0.0f;
        for (float a : accel_) m += a;
        m /= accel_.size();
        for (float a : accel_) q += (a - m) * (a - m);
        const float sd = accel_n_ >= accel_.size() ? std::sqrt(q / accel_.size()) : 0.0f;
        // our own servos shake the head a little too: their share is taken off while they move
        r.accelSd = e_.motion.isMoving() ? std::max(0.0f, sd - 0.15f) : sd;
    }
    const bool imu_masked = msWithin(now, imu_mask_until_ms_);
    if (imu_masked) r.accelSd = 0.0f;
    // gravity direction: pitch / (pi/2), roll / pi
    r.imuPitch = std::atan2(-ax, std::sqrt(ay * ay + az * az)) / 1.57079633f;
    r.imuRoll = std::atan2(ay, az) / 3.14159265f;
    r.imuOk = accel_n_ >= accel_.size() && !imu_masked;
    r.nearRaw = e_.presence.nearOverBaseline();
    r.nearOk = e_.presence.proximityOk();
    r.als = e_.presence.alsRaw();
    r.alsOk = e_.presence.lightOk();
    r.frameOk = frame_this_tick_ && frame_ok_;
    r.motionFrac = motion_[0];
    r.motionX = motion_[1];
    r.motionY = motion_[2];
    r.battery = e_.power.batteryLevel01();
    r.stamina = e_.motion.staminaLevel();
    r.hour = env.hour;
    r.clockOk = env.timeSynced;
    encoder_.update(r, channels_);
    zones_.apply(channels_);  // the zones as experience shifted them (zone_shift.h)
    frame_this_tick_ = false;
}

void BrainLoop::tick(uint32_t now, const Env& env)
{
    if (last_tick_ms_ != 0 && now - next_tick_ms_ > kTickIntervalMs * 3 / 2) ++missed_ticks_;
    tick_period_ms_ = last_tick_ms_ == 0 ? 0 : now - last_tick_ms_;
    last_tick_ms_ = now;
    const uint64_t start_us = static_cast<uint64_t>(esp_timer_get_time());
    ++ticks_;

    float sharp_ev = 0.0f;
    hear(now, sharp_ev);
    updateChannels(now, sharp_ev, env);
    using namespace v4;
    if (ear_[AUD_SPEECH] > 0.4f || channels_[CH_MOTION] > 0.1f || channels_[CH_NEAR] > 0.3f) last_somebody_ms_ = now;

    const bool decided = brain_.tick(channels_);
    decided_tick_ = decided;
    // the owner's sleep (21:45-06:00): the executor keeps silent and dim; before the clock is synced it might be night
    // too (a reboot at 3 am): quiet for the first 2 minutes, then a clock that never comes means daytime rules
    night_ = env.timeSynced ? (env.hour >= 21.75f || env.hour < 6.0f) : now < 120000;
    const uint32_t calls = exec_.calls(), gestures = exec_.gestures();
    if (brain_.trained()) {
        exec_.apply(brain_, decided, night_, now, done_);
    }
    said_tick_ = exec_.calls() != calls;
    gesture_tick_ = exec_.gestures() != gestures;
    if (decided) {
        if (!brain_.trained()) {
            // untrained networks drive nothing: their own choices stand in for what was done
            const auto& o = brain_.nn();
            int best = 0;
            for (int i = 1; i < kNoEmoCount; ++i)
                if (o[kNoEmoAt + i] > o[kNoEmoAt + best]) best = i;
            done_.emotion = best;
            done_.said = o[NO_SAY] > 0.5f;
            e_.motion.getCurrentDeg(done_.headYawDeg, done_.headPitchDeg);
        }
        done_.ledLevel = brain_.nn()[NO_LED_BRIGHT];
        brain_.executed(done_);
        done_.said = done_.gesture = false;
    }
    tick_last_us_ = static_cast<uint64_t>(esp_timer_get_time()) - start_us;
    tick_max_us_ = std::max(tick_max_us_, tick_last_us_);
}

void BrainLoop::step(uint32_t now, const Env& env)
{
    if (pending_touch_ms_) {
        // signed: the touch callback stamps its own, slightly later clock reading
        const int32_t held_s = static_cast<int32_t>(now - pending_touch_ms_);
        const uint32_t held = held_s > 0 ? static_cast<uint32_t>(held_s) : 0;
        // A real finger is a full-strength touch (level 3) of 180-250 ms or more (measured 2026-10-04); a servo phantom
        // is a short blip. Level 3 is believed from 120 ms, a weaker level-2 contact from 250 ms.
        const uint32_t needed = e_.touch.pressLevel() >= 3 ? 120 : 250;
        if (!e_.touch.contactNow() && held < needed) {
            ESP_LOGI(kTag, "head touch ignored: brief, during own motion (%u ms, level %u)", static_cast<unsigned>(held),
                     static_cast<unsigned>(e_.touch.pressLevel()));
            pending_touch_ms_ = 0;
        } else if (held >= needed) {  // still on the pad, or a full-strength tap that lasted long enough
            if (!touchStorm(now)) {
                acceptHeadTouch(now);
                ESP_LOGI(kTag, "head touch (held during motion, level %u)", static_cast<unsigned>(e_.touch.pressLevel()));
            }
            pending_touch_ms_ = 0;
        } else if (held > 3000) {
            pending_touch_ms_ = 0;
        }
    }
    if (static_cast<int32_t>(now - next_tick_ms_) < 0) return;
    tick(now, env);
    next_tick_ms_ += kTickIntervalMs;
    // Never run a catch-up burst after a slow tick.
    if (static_cast<int32_t>(static_cast<uint32_t>(esp_timer_get_time() / 1000) - next_tick_ms_) >= 0)
        next_tick_ms_ = static_cast<uint32_t>(esp_timer_get_time() / 1000) + kTickIntervalMs;
}

int BrainLoop::useLearned(const learned::View& v)
{
    int taken = 0;
    const float* a1w = v.find("nn.act1_w", NnV4::kAct1W);
    const float* a1b = v.find("nn.act1_b", NnV4::kAct1B);
    const float* a2w = v.find("nn.act2_w", NnV4::kAct2W);
    const float* a2b = v.find("nn.act2_b", NnV4::kAct2B);
    if (a1w && a1b && a2w && a2b) {
        brain_.network().useHeads(a1w, a1b, a2w, a2b);
        taken += 4;
    } else {
        brain_.network().useHeads(nullptr, nullptr, nullptr, nullptr);
    }
    zones_.shift.fill(0.0f);
    if (const float* s = v.find("zones.shift", v4::CH_COUNT)) {
        for (int c = 0; c < v4::CH_COUNT; ++c)
            zones_.shift[c] = std::isfinite(s[c]) ? std::clamp(s[c], -v4::kZoneShiftMax, v4::kZoneShiftMax) : 0.0f;
        ++taken;
    }
    return taken;
}

void BrainLoop::afterVoice(uint32_t now)
{
    (void)now;
    // Keep the mic's self-speaker suppression running while a phrase streams so the robot does not hear its own voice
    // as the world.
    if (e_.vocal.active()) e_.mic.suppressFor(300);
}
