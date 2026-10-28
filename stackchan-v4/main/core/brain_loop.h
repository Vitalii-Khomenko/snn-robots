#pragma once
#include <algorithm>
#include <array>
#include <cstdint>

#include "../audio/audio_output.h"
#include "../led/led_engine.h"
#include "../motion/motion_engine.h"
#include "../perception/auditory_net.h"
#include "../power/power_status_engine.h"
#include "../sensors/camera_engine.h"
#include "../sensors/imu_engine.h"
#include "../sensors/microphone_engine.h"
#include "../sensors/presence_engine.h"
#include "../sensors/touch_engine.h"
#include "../ui/face_engine.h"
#include "../voice/babble_voice.h"
#include "../v4/body_v4.h"
#include "../v4/brain_v4.h"
#include "../v4/channel_encoder.h"
#include "../v4/contingency_v4.h"
#include "../v4/learned_blob.h"
#include "../v4/v4_executor.h"
#include "../v4/zone_shift.h"

// Everything between the hardware engines and brain v4, shared by the robot (RobotRuntime) and the digital twin
// (SimRuntime) so the two can not drift apart:
//
//   engines -> the ear (AuditoryNet: loudness, speech) and the raw readings -> ChannelEncoder (13 channels + 2 where)
//           -> BrainV4 (spiking analyzer every tick, main network every 2nd) -> V4Executor (face, voice, LEDs, head)
//
// plus the coupling of our own servos and voice into the microphone and the IMU, and the head pad's phantom filters.
class BrainLoop {
public:
    struct Engines {
        MotionEngine& motion;
        FaceEngine& face;
        LedEngine& led;
        AudioOutput& audio;
        BabbleVoice& vocal;
        ImuEngine& imu;
        MicrophoneEngine& mic;
        PresenceEngine& presence;
        CameraEngine& camera;
        PowerStatusEngine& power;
        TouchEngine& touch;
    };
    struct Env {
        bool timeSynced = false;
        float hour = 12.0f;
    };
    static constexpr uint32_t kTickIntervalMs = 50;

    explicit BrainLoop(Engines e);

    // ---- events from the engines' callbacks
    void headTouch(uint32_t now);
    void screenTouch(uint32_t now) { (void)now, screen_ev_ = 1.0f; }
    // intensity as MicrophoneEngine reports it (0.4 .. 1.0)
    void loudSound(float intensity01) { loud_event_ = std::max(loud_event_, intensity01); }
    void soundOnset(float strength01) { onset_event_ = std::max(onset_event_, strength01); }
    void vision(const CameraEngine::VisionFrame& f, uint32_t now);
    // The servo rail was power-cycled: the head jolts as the servos take hold again - not a shake (2026-10-24: shake
    // 0.33 right after a power cycle). The IMU channels are not read for the next 3 s.
    // (And it is the body's one real ache: BodyV4's budget pays for it.)
    void servoRepowered(uint32_t now) { imu_mask_until_ms_ = now + 3000, hurt_ = true; }
    // What the robot learned from its life (stage 6): the main network's action heads and the zone shifts from a blob
    // (net/learned_store.h on the robot, --learned in the simulator); an empty view = innate. Between two ticks only.
    // Returns the number of arrays taken (0, 1 or 5).
    int useLearned(const learned::View& v);
    const ZoneShift& zoneShift() const { return zones_; }

    // ---- per main-loop pass
    // Before the microphone / IMU update(): our own servos make noise.
    void coupleMotion(uint32_t now);
    // After every engine has updated: on the brain's clock the ear, the channels, brain v4 and the executor.
    void step(uint32_t now, const Env& env);
    // After the voice has been stepped: keep the microphone from hearing our voice.
    void afterVoice(uint32_t now);

    // ---- what the last tick saw and did (the diary, the stats line)
    uint32_t tickCount() const { return ticks_; }
    const ChannelEncoder::Raw& raw() const { return raw_; }
    const ChannelEncoder::Out& channels() const { return channels_; }
    const AuditoryNet::Features& ear() const { return ear_; }
    const AuditoryNet& auditory() const { return auditory_; }
    // the ear's own speech class before the voice check, and the voice check (diary ear_*)
    float earSpeech() const { return ear_speech_; }
    float voice() const { return voice_; }
    BrainV4& brain() { return brain_; }
    const BrainV4& brain() const { return brain_; }
    const V4Executor& executor() const { return exec_; }
    bool night() const { return night_; }
    bool driving() const { return brain_.trained(); }  // brain v4 drives the body (false: untrained networks)
    bool saidThisTick() const { return said_tick_; }
    bool decidedThisTick() const { return decided_tick_; }
    bool gestureThisTick() const { return gesture_tick_; }
    // the sense of "I did that" (docs/CONTINGENCY.md): its numbers of the last tick (diary ct_*)
    const ContingencyV4::Out& contingency() const { return contingency_out_; }
    // the body (docs/BODY.md): the sleep pressure (the diary's raw_stamina is 1 - it)
    const BodyV4& body() const { return body_; }
    // diagnostics
    uint64_t tickLastUs() const { return tick_last_us_; }
    uint64_t takeTickMaxUs() { const uint64_t m = tick_max_us_; tick_max_us_ = 0; return m; }
    uint32_t missedTicks() const { return missed_ticks_; }
    uint32_t tickPeriodMs() const { return tick_period_ms_; }
    uint32_t audioUs() const { return audio_us_; }
    uint32_t stormDropped() const { return storm_dropped_; }

private:
    void tick(uint32_t now, const Env& env);
    void hear(uint32_t now, float& sharpEv);
    void updateChannels(uint32_t now, float sharpEv, const Env& env);
    bool touchStorm(uint32_t now);
    void acceptHeadTouch(uint32_t now);

    Engines e_;
    AuditoryNet auditory_;
    AuditoryNet::Features ear_{};
    ChannelEncoder encoder_;
    ChannelEncoder::Raw raw_{};
    ChannelEncoder::Out channels_{};
    BrainV4 brain_;
    V4Executor exec_;
    BrainV4::Executed done_{};
    bool night_ = false, said_tick_ = false, gesture_tick_ = false, decided_tick_ = false;
    ZoneShift zones_;
    ContingencyV4 contingency_;
    ContingencyV4::Out contingency_out_{};
    BodyV4 body_;
    float body_yaw_ = 0.0f, body_pitch_ = 0.0f;
    bool body_head_ = false, hurt_ = false;
    uint32_t lit_ms_ = 0;  // the room was last lit (night rules without a clock)
    static constexpr float kLitRoom = 0.04f;

    // the camera's last frame (~7 fps): motion share and centre
    float motion_[3] = {0, 0, 0};
    float light_spot_[3] = {0, 0, 0};  // the camera's brightest spot: x, y, strength
    bool frame_ok_ = false, light_ok_ = false, frame_this_tick_ = false;
    static constexpr float kLightTurnDegS = 12.0f;  // the bright spot is read in turns slower than this
    // touches
    float head_ev_ = 0.0f, screen_ev_ = 0.0f;
    uint32_t touch_ms_ = 0;          // the last accepted head touch (or the finger still on the pad after it)
    uint32_t pending_touch_ms_ = 0;  // a touch that arrived during fast motion, waiting to be told from a servo phantom
    std::array<uint32_t, 9> touch_times_{};  // the last head touches: a storm with nobody around is pad noise
    uint32_t touch_n_ = 0, storm_dropped_ = 0, storm_logged_ms_ = 0, last_somebody_ms_ = 0;
    // sound
    float loud_event_ = 0.0f, onset_event_ = 0.0f, loud_hold_ = 0.0f, onset_hold_ = 0.0f;
    std::array<bool, 3> heard_{};  // clap / sharp / speech above 0.5 (rising edges are logged)
    // our own motion and voice
    bool was_moving_ = false;
    uint32_t motion_edge_ms_ = 0, own_motion_until_ms_ = 0, self_voice_until_ms_ = 0, imu_mask_until_ms_ = 0;
    // speech must be a voice and not our own (hear())
    static constexpr float kVoiceGate = 0.3f, kSpeechNoVoice = 0.3f;
    static constexpr uint32_t kOwnVoiceTailMs = 1500;
    uint32_t own_voice_tail_until_ms_ = 0;
    float ear_speech_ = 0.0f, voice_ = 0.0f;
    std::array<float, 10> accel_{};  // |acceleration| of the last 10 ticks (0.5 s), g
    uint32_t accel_n_ = 0;
    // timing
    uint32_t ticks_ = 0, next_tick_ms_ = 0, last_tick_ms_ = 0, tick_period_ms_ = 0, missed_ticks_ = 0;
    uint64_t tick_last_us_ = 0, tick_max_us_ = 0;
    uint32_t audio_us_ = 0;
};
