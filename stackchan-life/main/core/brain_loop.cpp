#include "brain_loop.h"
#include "ms_clock.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <esp_log.h>
#include <esp_timer.h>

using namespace hybrid;

namespace {
constexpr const char* kTag = "runtime";
}

BrainLoop::BrainLoop(Engines e)
    : e_(e), executor_(e.face, e.led, e.vocal, e.motion)
{
    brain_.setClock([]() -> int64_t { return esp_timer_get_time(); });
}

// A storm of touches with nobody around is the pad, not a hand: it fired every 2-4 s all night while the owner slept
// (2026-10-13; the pad sits next to the pitch servo). A first version (6 in a minute, nobody in this tick) dropped
// the owner's own petting the same evening - he stroked it from above, out of the camera, silent. Now: more than 8
// touches in 10 minutes, and nobody seen or heard in all those 10 minutes (2026-10-17: the nightly phantoms come
// ~15 in 10 min; whoever pets the robot was seen or heard coming up to it).
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
    // The Si12T pad sits next to the pitch servo: a moving head couples into
    // it as a level 2-3 phantom "touch" (seen on hardware). Not a touch.
    // Only fast movement (saccade, nod) and 300 ms after it: slow drift must
    // not make real touches disappear.
    if (e_.motion.isMovingFast(now)) {
        // Maybe a servo phantom, maybe a hand on the head while it moves: decided in step() -
        // a brief blip is dropped, a touch that is held long enough (see step()) is real.
        if (pending_touch_ms_ == 0) pending_touch_ms_ = now ? now : 1;
        return;
    }
    if (touchStorm(now)) return;
    sense_.event(BodySense::Event::HeadTouch, now);
    ESP_LOGI(kTag, "head touch (level %u)", static_cast<unsigned>(e_.touch.pressLevel()));
}

void BrainLoop::coupleMotion(uint32_t now)
{
    // Motion phase for the microphone's own-noise measurement: servo clicks
    // are expected at the start and the stop of a movement.
    const bool moving = e_.motion.isMoving();  // any move: the servos click at the start and the stop of every one
    if (moving != was_moving_) {
        was_moving_ = moving;
        motion_edge_ms_ = now;
    }
    e_.mic.setMotionPhase(now - motion_edge_ms_ < 400 ? 1 : moving ? 2 : 0);
    if (moving) {
        // The IMU and the microphone hear our own servos: raise their
        // thresholds while the head moves and 350 ms after (the servo stops
        // ~120 ms after the last packet).
        e_.mic.raiseThresholdFor(350);
        own_motion_until_ms_ = now + 350;
    }
    e_.imu.update(moving);
    // Interaural / onset events from the mic task
    MicrophoneEngine::Onset onset;
    while (e_.mic.takeOnset(onset)) soundOnset(std::clamp(onset.strength, 0.0f, 1.0f));
}

void BrainLoop::updateBody(uint32_t now)
{
    (void)now;
    BodySense::Raw& r = raw_;
    float yaw = 0, pitch = 0;
    e_.motion.getCurrentDeg(yaw, pitch);
    // The motion model's own velocity (command model, TX-only servo bus).
    float vy = 0, vp = 0;
    e_.motion.getVelocityDegS(vy, vp);
    r.yawDeg = yaw;
    r.pitchDeg = pitch;
    r.yawVelDegS = vy;
    r.pitchVelDegS = vp;
    float ax, ay, az;
    e_.imu.getAccel(ax, ay, az);
    r.imuPitch = std::atan2(-ax, std::sqrt(ay * ay + az * az)) / 1.57079633f;
    r.imuRoll = std::atan2(ay, az) / 3.14159265f;
    {
        // Vestibular sense. Which gyro axis is yaw / pitch (and its sign)
        // depends on how the IMU sits in the head, so it is learned: each
        // axis is correlated with the commanded head velocity while the head
        // moves, and the best-matching axis is used once the correlation is
        // clear.
        float g[3];
        e_.imu.getGyro(g[0], g[1], g[2]);
        for (int a = 0; a < 3; ++a) {
            if (std::fabs(vy) > 5.0f) gyro_yaw_corr_[a] = gyro_yaw_corr_[a] * 0.995f + g[a] * vy * 0.005f;
            if (std::fabs(vp) > 5.0f) gyro_pitch_corr_[a] = gyro_pitch_corr_[a] * 0.995f + g[a] * vp * 0.005f;
        }
        auto best = [](const float* c, int skip) {
            int b = -1;
            for (int a = 0; a < 3; ++a)
                if (a != skip && (b < 0 || std::fabs(c[a]) > std::fabs(c[b]))) b = a;
            return b;
        };
        const int ya = best(gyro_yaw_corr_, -1), pa = best(gyro_pitch_corr_, ya);
        r.gyroValid = std::fabs(gyro_yaw_corr_[ya]) > 50.0f && std::fabs(gyro_pitch_corr_[pa]) > 20.0f;
        r.realYawVelDegS = gyro_yaw_corr_[ya] > 0 ? g[ya] : -g[ya];
        r.realPitchVelDegS = gyro_pitch_corr_[pa] > 0 ? g[pa] : -g[pa];
    }
    r.battery01 = e_.power.batteryLevel01();
    r.stamina01 = e_.motion.staminaLevel();
    r.lightDelta = e_.presence.relativeLightDelta();
    r.speaker = e_.vocal.active();
}

void BrainLoop::feedVision(uint32_t now)
{
    Bus& bus = brain_.bus();
    // signed age: the frame can be stamped a few ms after the loop iteration's `now` was read
    if (vision_pending_ && e_.camera.isFresh() && static_cast<int32_t>(now - vision_frame_ms_) < 350) {
        bus.vision = vision_frame_.features;
        // While the head turns, "something comes closer / goes away" is the robot's own motion: a person sliding into the
        // picture as the head comes back looked like approach (audit 2026-10-04: 143 startles in a TV evening, every one
        // during its own movement). Not passed on, like the microphone's loud events during own motion.
        if (msWithin(now, own_motion_until_ms_)) bus.vision[VIS_LOOM] = bus.vision[VIS_RECEDE] = 0.0f;
        vision_age_ticks_ = 0;
        frame_this_tick_ = true;
        // Prediction error of the eyes: the image shift the camera saw
        // versus what the head's real turn should have produced (32 px over
        // 60 degrees). Sign conventions differ per mounting, so magnitudes.
        float err = 0.0f;
        if (raw_.gyroValid && vision_frame_.shiftValid) {
            const float expected = std::hypot(raw_.realYawVelDegS, raw_.realPitchVelDegS) * 0.143f * (32.0f / 60.0f);
            const float measured = std::hypot(vision_frame_.shiftX, vision_frame_.shiftY);
            err = std::clamp(std::fabs(measured - expected) / 3.0f - 0.1f, 0.0f, 1.0f);
        }
        pe_vision_ += (err - pe_vision_) * 0.5f;
        sense_.setVisionError(pe_vision_);
    } else {
        // No new frame this tick (the camera runs at ~7 fps): events fade
        // after a frame's worth of ticks, a stalled camera goes blind.
        ++vision_age_ticks_;
        auto& v = bus.vision;
        if (vision_age_ticks_ > 3) {
            for (int i : {VIS_MOTION_L, VIS_MOTION_C, VIS_MOTION_R, VIS_MOTION_UP, VIS_MOTION_DOWN, VIS_LOOM, VIS_RECEDE,
                          VIS_ACTIVITY, VIS_LIGHT_CHANGE})
                v[i] *= 0.5f;
            for (int i : {VIS_PERSON, VIS_FACE, VIS_TARGET_CONF}) v[i] *= 0.9f;
        }
        if (vision_age_ticks_ > 40) {
            const float b = v[VIS_BRIGHTNESS];
            v.fill(0.0f);
            v[VIS_BRIGHTNESS] = b;
        }
        pe_vision_ *= 0.8f;
        sense_.setVisionError(pe_vision_);
    }
    vision_pending_ = false;
    holdPerson();
}

// A person the eyes lose for a moment (a blink of the detector: real-camera confidence hovers
// around the threshold) is not gone: the last sighting is kept for ~1.2 s with fading confidence.
// Its position is remembered in the room (head angle + image offset), so it keeps pointing the
// right way while the head turns. Deterministic, same code in the simulator.
void BrainLoop::holdPerson()
{
    auto& v = brain_.bus().vision;
    constexpr int kHoldTicks = 24;
    constexpr float kHalfViewYaw = 30.0f, kHalfViewPitch = 22.0f;
    if (v[VIS_PERSON] > 0.4f && v[VIS_TARGET_CONF] > 0.4f) {
        // Anchored in the room only by a frame of this tick: between frames the bus keeps the last frame's image
        // position while the head turns on, and re-anchoring it to the new head angle walked the remembered person
        // along with the turn.
        if (frame_this_tick_) {
            hold_.person = v[VIS_PERSON];
            hold_.face = v[VIS_FACE];
            hold_.conf = v[VIS_TARGET_CONF];
            hold_.worldYaw = raw_.yawDeg + v[VIS_TARGET_X] * kHalfViewYaw;
            hold_.worldPitch = raw_.pitchDeg - v[VIS_TARGET_Y] * kHalfViewPitch;
            hold_.age = 0;
        }
        return;
    }
    if (hold_.age >= kHoldTicks) return;
    ++hold_.age;
    const float fade = 1.0f - static_cast<float>(hold_.age) / kHoldTicks;
    const float x = std::clamp((hold_.worldYaw - raw_.yawDeg) / kHalfViewYaw, -1.0f, 1.0f);
    const float y = std::clamp((raw_.pitchDeg - hold_.worldPitch) / kHalfViewPitch, -1.0f, 1.0f);
    if (std::fabs(x) >= 1.0f || std::fabs(y) >= 1.0f) {  // it left the picture
        hold_.age = kHoldTicks;
        return;
    }
    v[VIS_PERSON] = std::max(v[VIS_PERSON], hold_.person * fade);
    v[VIS_FACE] = std::max(v[VIS_FACE], hold_.face * fade);
    v[VIS_TARGET_CONF] = std::max(v[VIS_TARGET_CONF], hold_.conf * fade);
    v[VIS_TARGET_X] = x;
    v[VIS_TARGET_Y] = y;
}

void BrainLoop::tick(uint32_t now, const Env& env)
{
    {
        const auto& bus = brain_.bus();
        if (bus.vision[hybrid::VIS_PERSON] > 0.4f || bus.vision[hybrid::VIS_FACE] > 0.3f ||
            bus.audio[hybrid::AUD_SPEECH] > 0.4f)
            last_somebody_ms_ = now;
    }
    Bus& bus = brain_.bus();
    raw_.timeSynced = env.timeSynced;
    raw_.hour = env.hour;
    raw_.ownerNear = env.ownerNear;
    if (env.ownerNear && !owner_was_nearby_) sense_.event(BodySense::Event::OwnerArrived, now);
    owner_was_nearby_ = env.ownerNear;
    if (e_.imu.tiltActive()) sense_.event(BodySense::Event::Tilt, now);  // held up or turned over: keep feeling it
    sense_.update(raw_, now, bus);
    feedVision(now);

    // Auditory NN (main thread: ~7 k MACs). Our own voice and servos are
    // flagged so the network can tell the world from the robot.
    {
        const auto self = e_.audio.takeSelfSound();
        if (e_.vocal.active()) self_voice_until_ms_ = now + 300;
        AuditoryNet::Context ctx;
        ctx.selfSpeaking = msWithin(now, self_voice_until_ms_);
        ctx.ownMotion = msWithin(now, own_motion_until_ms_);
        ctx.selfLevel01 = self[0];
        // Events arrive as one-tick pulses; the spiking core was trained on
        // events that fade over ~3 ticks (the same shape the perception
        // networks give), so hold them the same way.
        // Our own servos click and whine loud enough to look like claps (peaks of
        // 20-32k on hardware): while the head moves, and just after, such events
        // are mostly ourselves and are not passed on at all.
        if (ctx.ownMotion) loud_event_ = onset_event_ = 0.0f;
        onset_hold_ = std::max(onset_event_, onset_hold_ * 0.6f);
        loud_hold_ = std::max(loud_event_, loud_hold_ * 0.6f);
        ctx.onset01 = onset_hold_ > 0.05f ? onset_hold_ : 0.0f;
        ctx.loud01 = loud_hold_ > 0.05f ? loud_hold_ : 0.0f;
        onset_event_ = loud_event_ = 0.0f;
        const int64_t t0 = esp_timer_get_time();
        auditory_.run(e_.mic.audioFront(), ctx, bus.audio);
        const uint32_t us = static_cast<uint32_t>(esp_timer_get_time() - t0);
        audio_us_ = audio_us_ ? (audio_us_ * 7 + us) / 8 : us;
        // what the ear makes of it, once per sound (rising edge), for live tests
        const float heard[3] = {bus.audio[AUD_CLAP], bus.audio[AUD_SHARP], bus.audio[AUD_SPEECH]};
        static const char* const names[3] = {"clap", "sharp", "speech"};
        for (int i = 0; i < 3; ++i) {
            if (heard[i] > 0.5f && !heard_[i])
                ESP_LOGI(kTag, "hear: %s %.2f (energy %.2f%s)", names[i], static_cast<double>(heard[i]),
                         static_cast<double>(bus.audio[AUD_ENERGY]), ctx.ownMotion ? ", own motion" : "");
            heard_[i] = heard[i] > 0.35f ? (heard[i] > 0.5f || heard_[i]) : false;
        }
    }

    if (last_tick_ms_ != 0 && now - next_tick_ms_ > kTickIntervalMs * 3 / 2) ++missed_ticks_;
    tick_period_ms_ = last_tick_ms_ == 0 ? 0 : now - last_tick_ms_;
    last_tick_ms_ = now;
    const uint64_t start_us = static_cast<uint64_t>(esp_timer_get_time());
    brain_.tick(now);
    {  // what the robot just learned by itself (a link crossed 0.25 or 0.5)
        int cue = 0, outcome = 0;
        float w = 0.0f;
        bool rising = false;
        if (brain_.assoc().takeEvent(cue, outcome, w, rising))
            ESP_LOGI(kTag, "learn: %s -> %s %.2f (%s)", AssocMemory::cueName(cue), AssocMemory::outcomeName(outcome),
                     static_cast<double>(w), rising ? "learned" : "fading");
    }
    executor_.apply(brain_, sense_, now);
    track_.update(now, frame_this_tick_, bus.vision[VIS_PERSON], bus.vision[VIS_TARGET_X], bus.vision[VIS_TARGET_CONF],
                  raw_.yawDeg);
    frame_this_tick_ = false;
    tick_last_us_ = static_cast<uint64_t>(esp_timer_get_time()) - start_us;
    tick_max_us_ = std::max(tick_max_us_, tick_last_us_);
}

void BrainLoop::step(uint32_t now, const Env& env)
{
    updateBody(now);
    if (pending_touch_ms_) {
        // signed: the touch callback stamps its own, slightly later clock reading
        const int32_t held_s = static_cast<int32_t>(now - pending_touch_ms_);
        const uint32_t held = held_s > 0 ? static_cast<uint32_t>(held_s) : 0;
        // A real finger is a full-strength touch (level 3) of 180-250 ms or more (measured 2026-10-04); a servo phantom is
        // a short blip. Level 3 is believed from 120 ms, a weaker level-2 contact from 250 ms.
        const uint32_t needed = e_.touch.pressLevel() >= 3 ? 120 : 250;
        if (!e_.touch.contactNow() && held < needed) {
            ESP_LOGI(kTag, "head touch ignored: brief, during own motion (%u ms, level %u)", static_cast<unsigned>(held),
                     static_cast<unsigned>(e_.touch.pressLevel()));
            pending_touch_ms_ = 0;
        } else if (held >= needed) {  // still on the pad, or a full-strength tap that lasted long enough
            if (!touchStorm(now)) {
                sense_.event(BodySense::Event::HeadTouch, now);
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

void BrainLoop::afterVoice(uint32_t now)
{
    (void)now;
    // Keep the mic's self-speaker suppression running while a phrase
    // streams so the robot does not hear its own voice as the world.
    if (e_.vocal.active()) e_.mic.suppressFor(300);
}

int BrainLoop::formatTelemetry(char* buf, int size) const
{
    const Bus& b = brain_.bus();
    const auto& a = brain_.action();
    const auto& s = b.snn;
    const auto& v = b.vision;
    const auto& au = b.audio;
    return std::snprintf(
        buf, static_cast<size_t>(size),
        "VISION target=%+.2f,%+.2f conf=%.2f motion=%.2f/%.2f/%.2f person=%.2f face=%.2f loom=%.2f | AUDIO energy=%.2f "
        "speech=%.2f clap=%.2f noise=%.2f nov=%.2f | SNN cur=%.2f aro=%.2f bor=%.2f ann=%.2f soc=%.2f "
        "play=%.2f fat=%.2f slp=%.2f mus=%.2f att=%+.2f,%+.2f/%.2f rate=%.3f | POLICY yaw=%+.2f pitch=%+.2f speed=%.2f "
        "hold=%.2f expl=%.2f snd=%.2f inv=%.2f val=%+.2f aro=%.2f | BODY yaw=%+.2f pitch=%+.2f | PE m=%.2f v=%.2f "
        "| LEARN petting=%.2f rough=%.2f | HEART %.0f bpm",
        v[VIS_TARGET_X], v[VIS_TARGET_Y], v[VIS_TARGET_CONF], v[VIS_MOTION_L], v[VIS_MOTION_C], v[VIS_MOTION_R],
        v[VIS_PERSON], v[VIS_FACE], v[VIS_LOOM], au[AUD_ENERGY], au[AUD_SPEECH], au[AUD_CLAP], au[AUD_CONTINUOUS],
        au[AUD_NOVELTY], s[SNN_CURIOSITY], s[SNN_AROUSAL], s[SNN_BOREDOM], s[SNN_ANNOYANCE],
        s[SNN_SOCIAL], s[SNN_PLAYFUL], s[SNN_FATIGUE], s[SNN_SLEEP], s[SNN_MUSIC], s[SNN_ATT_X], s[SNN_ATT_Y],
        s[SNN_ATT_STRENGTH], static_cast<double>(brain_.snn().meanRate()), a[ACT_YAW], a[ACT_PITCH], a[ACT_SPEED],
        a[ACT_HOLD], a[ACT_EXPLORE], a[ACT_SOUND_ACT], a[ACT_INVITE], a[ACT_FACE_VALENCE],
        a[ACT_FACE_AROUSAL], b.body[BODY_YAW], b.body[BODY_PITCH], b.pe[PE_MOTOR], b.pe[PE_VISION],
        b.learn[LEARN_EXPECT_REWARD], b.learn[LEARN_EXPECT_THREAT], brain_.physiology().heartBpm);
}

int BrainLoop::formatDataset(uint32_t nowMs, char* buf, int size) const
{
    PolicyGru::Input in;
    brain_.bus().policyInput(in);
    int n = std::snprintf(buf, static_cast<size_t>(size), "dataset: %u ", static_cast<unsigned>(nowMs));
    auto put = [&](float v) {
        if (n + 5 >= size) return;
        const int q = static_cast<int>(std::lround(std::clamp(v, -32.0f, 32.0f) * 1000.0f));
        n += std::snprintf(buf + n, static_cast<size_t>(size - n), "%04x", static_cast<unsigned>(q & 0xffff));
    };
    for (float v : in) put(v);
    for (float v : brain_.action()) put(v);
    return n;
}

int BrainLoop::formatAudioRecord(uint32_t nowMs, char* buf, int size) const
{
    int n = std::snprintf(buf, static_cast<size_t>(size), "audioin: %u ", static_cast<unsigned>(nowMs));
    for (float v : auditory_.lastInput()) {
        if (n + 5 >= size) break;
        const int q = static_cast<int>(std::lround(std::clamp(v, -32.0f, 32.0f) * 1000.0f));
        n += std::snprintf(buf + n, static_cast<size_t>(size - n), "%04x", static_cast<unsigned>(q & 0xffff));
    }
    return n;
}
