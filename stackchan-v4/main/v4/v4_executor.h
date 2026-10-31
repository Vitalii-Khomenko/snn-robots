#pragma once
#include <cstdint>

#include "../led/led_engine.h"
#include "../motion/motion_engine.h"
#include "../ui/face_engine.h"
#include "../voice/babble_voice.h"
#include "brain_v4.h"
#include "memory_v4.h"

// Brain v4's "muscles" (BRAIN_V4_PLAN.txt section 7): the main network decides WHAT (emotion, voice, LEDs, head
// mode); this only does HOW, inside physical and safety limits:
//   face   - the chosen emotion (a face stays at least 1.5 s so its morph can be seen); the eyes follow attention
//   voice  - BabbleVoice with the network's style, mood, register, length, rate; the call budget (~12 a minute,
//            reactions may spend the last tokens); SILENT at night 21:45-06:00 (the owner sleeps next to the robot)
//   LEDs   - colour, pattern (steady / breathe / pulse / flicker / wave), brightness, rate; dim at night
//   head   - gaze targets and the gesture library: nod "yes" (pitch), shake "no" (yaw), flinch, look down / up,
//            doze. Small amplitudes, at most one nod / shake every 4 s (current peaks: the AXP power-offs, 2026-10-23);
//            no nods, shakes or flinches at night.
class V4Executor {
public:
    V4Executor(FaceEngine& face, LedEngine& led, BabbleVoice& vocal, MotionEngine& motion)
        : face_(face), led_(led), vocal_(vocal), motion_(motion)
    {}
    // Every brain tick (20 Hz) with the latest decision; `decided` when the main network ran this tick.
    // `night`: the owner's sleep window, or the clock not synced yet (it might be night: quiet until it is known). Fills `done` with what really happened (BrainV4::Executed).
    void apply(const BrainV4& brain, bool decided, bool night, uint32_t nowMs, BrainV4::Executed& done);

    uint32_t calls() const { return calls_; }
    uint32_t gestures() const { return gestures_; }
    uint32_t nods() const { return nods_; }
    uint32_t shakes() const { return shakes_; }
    int shownEmotion() const { return static_cast<int>(shown_); }
    // What the executor held back of the network's wishes this tick (2026-10-28, an audit: the night's learning took
    // every wish for a deed; a third of the calls the network wanted never sounded). The diary's `held`, bits:
    enum Held : uint8_t {
        kCallNight = 1,      // a call wanted (say above 0.55, armed), silent: night
        kCallBudget = 2,     // ... the call budget is spent
        kCallBusy = 4,       // ... a phrase still sounding, or too soon after the last
        kGestureNight = 8,   // a nod / shake / flinch wanted (the mode's rising edge), not made: night
        kGestureSoon = 16,   // ... too soon after the last, or another still running
        kGazeOwn = 32,       // the head follows its own reflex (something new in view), a watch or a glance, not the
                             // network's gaze (look / scan / hold by day)
    };
    uint8_t held() const { return held_; }
    // Remembering a moment (MemoryV4, docs/EPISODES.md): the head turns for a moment to where it happened, a little up,
    // and the face shows the remembering face; false when something comes first (night, a gesture, a reaction,
    // something new in view, a glance) - then it is not remembered now.
    bool remember(const MemoryV4::Episode& e, uint32_t nowMs);
    static constexpr uint32_t kMemoryHoldMs = 2500, kMemoryFaceMs = 3500;
    // a move starts once the servo rail is up (MotionEngine::update; the simulator's servos the same): ~1 s
    static constexpr uint32_t kRailUpMs = 1000;
    static constexpr float kMemorySpeedDegS = 40.0f, kMemoryPitchUpDeg = 6.0f;

    static constexpr float kYawLimitDeg = 100.0f, kPitchMinDeg = -20.0f, kPitchMaxDeg = 35.0f;
    static constexpr uint32_t kMinCommandGapMs = 120;  // servo bus: <= ~8 packets/s (gesture keyframes)
    // Gaze moves (look, scan, look up / down, doze): at most one every 2.5 s, and only for a visible change. The servos
    // hang more often the more packets they get (docs/HARDWARE.md); with every small gaze correction sent, the lively
    // network of 2026-10-24 sent 45-80 packets a minute, the servos hung 60-100 times an hour and the head barely moved.
    // 2026-10-25: every mood moves its head now (teacher_nn.py MANNER) and at 2.5 s / 8 deg the robot sent 50-120
    // packets a minute and the servos hung within a minute of every power cycle: one move per 4 s, >= 12 / 8 deg.
    static constexpr uint32_t kGazeGapMs = 4000;
    static constexpr float kGazeDeadYawDeg = 12.0f, kGazeDeadPitchDeg = 8.0f;
    // Wandering gaze (2026-10-25 evening, the owner: "the head is always round the centre, and a short whizz of the
    // motors every few seconds"): the network's gaze drifted within ~+-0.45 (+-27 deg) and every 4 s a 14 deg step went
    // out - 11 moves a minute. With nothing holding its attention (the attention below kAttLook for kWatchMs) the head
    // now keeps a place for kWanderHoldMs (shorter in a lively mood) and turns only for >= kWanderDeadDeg; what it
    // attends to is looked at as before. (Wider is the network's own gaze, teacher_nn.py GAZE_WIDE: spreading it here
    // 2x ran away - the network follows its own head yaw - and the head stayed at -95 deg in the simulator.)
    static constexpr uint32_t kWanderHoldMinMs = 5000, kWanderHoldMaxMs = 11000, kWatchMs = 10000;
    static constexpr float kWanderDeadDeg = 20.0f, kAttLook = 0.12f;
    // Something new in view turns the head there 1.5x faster than the idle wander (still eased): at the wander's
    // speed a lamp had stopped being new before the head got there (2026-10-26, simulator: at the lamp 30 % of the
    // time, 27 % with it off).
    static constexpr float kOrientSpeedK = 1.5f;  // 2.5 until 2026-10-26 evening: a jerk right after a glance
    // The orienting reflex (2026-10-26 evening): the turn to what the analyzer's attention holds is the executor's own,
    // like the teacher's rule (head yaw + att_x x half the camera's width), and the place is watched kWatchMs. It was
    // left to the network; with the Muse's ideas to learn as well it turned to a new lamp no more than by chance.
    // 2026-10-27, the owner again: "sometimes it jerks its head far left and back". The reflex re-aimed every tick
    // from where the head was *now*, while the picture (and the attention) is not renewed during a turn: the camera
    // takes no frames in a fast turn, so a lamp at the picture's edge stayed at att_x -0.95 and each tick added another
    // 30 deg - after a glance to -50..-66 the head ran on to -88..-100 (11:26, 12:03, 12:51; at -100 the servos hung and
    // the rail was cycled), then went back. Now the place is aimed at once, again only after the head has been still
    // kAimStillMs (fresh frames), and the reflex stays within kOrientLimitDeg.
    static constexpr float kCamHalfDeg = 30.0f, kOrientDeadDeg = 8.0f, kOrientLimitDeg = 75.0f;
    static constexpr uint32_t kOrientGapMs = 1500, kAimStillMs = 800;
    // Curious glances (2026-10-26, the owner: "add some creativity and unpredictability"): while it wanders by day, at
    // a random moment every 40-150 s the head turns far to one side (55-90 deg, the other side when it already looks
    // aside; a little up or down), looks there 3-7 s and goes back to the network's gaze. The room's edges, which
    // the network's gaze rarely reaches, get looked at too. (A move it makes by itself, like a nod: the network
    // decides what the robot feels and wants, this only how its idle head moves.)
    // 2026-10-26 evening, the owner: "sometimes it jerks its head far to the side and back" - 55-90 deg at a 20 deg/s
    // peak and back after 3-7 s: now 40-70 deg at 12 deg/s, looking 4-9 s.
    // 2026-10-27: the jerks were the servos' own (docs/HARDWARE.md, the head's own jerks), not the glances. The slow
    // glances came with ~20 hangs an hour instead of 0-3 (a third of them within 8 s of a glance's start, 6x the
    // chance): back to a 20 deg/s peak (kGlanceSpeedDegS 40, halved by the eased move), 40-70 deg - a test.
    // 18:00-19:00 at 20 deg/s: 20 hangs (8.6 % of the switch-ons), as at 12 deg/s - the glance speed does not cause them
    // (a glance's start is simply where the rail comes on). Kept.
    static constexpr uint32_t kGlanceGapMinMs = 40000, kGlanceGapMaxMs = 150000, kGlanceHoldMinMs = 4000,
                              kGlanceHoldMaxMs = 9000;
    static constexpr float kGlanceYawMinDeg = 40.0f, kGlanceYawMaxDeg = 70.0f, kGlanceSpeedDegS = 40.0f;
    // An answer opens a turn (2026-10-28, the owner: answers and the circular reaction): for kTurnMs after the detector's
    // answered pulse rises above kTurnOn a call costs what a reaction does (1 token, not 2.5) - the robot can answer an
    // answer at once, as in a proto-conversation, even when idle babble has spent the budget (calls were at the budget
    // in every simulator scenario, answered or not: the answered call could not come again).
    static constexpr float kTurnOn = 0.3f;
    // The call budget, a piggy bank: kTokensMax tokens, one more every kTokenMs; a call costs one and needs kIdleTokens
    // (a reaction or an answer's turn: kReactionTokens). 2026-10-28, the owner: "if the network wants to call, let it" -
    // it was 5 tokens, one per 10 s, idle 2.5 (~6 calls a minute): the robot called at that ceiling in every scenario and
    // held back 45 % of the calls it wanted (~10 a minute on the robot). Now a ceiling of ~12 a minute, against a stuck
    // voice only.
    static constexpr float kTokensMax = 6.0f, kTokenMs = 5000.0f, kIdleTokens = 1.0f, kReactionTokens = 0.5f;
    static constexpr uint32_t kTurnMs = 4000;
    // the flinch: at most one every kFlinchGapMs by day (2026-10-26: 9 in 13 minutes at the owner's sounds)
    static constexpr uint32_t kFlinchGapMs = 15000;
    static constexpr uint32_t kGestureGapMs = 4000;    // nod / shake at most every 4 s
    static constexpr float kMaxSpeedDegS = 110.0f;

private:
    enum class Gesture : uint8_t { None, Nod, Shake, Flinch };
    void applyFace(const BrainV4& brain, uint32_t now);
    void applyVoice(const BrainV4& brain, bool night, uint32_t now, BrainV4::Executed& done);
    void applyLeds(const BrainV4& brain, bool night);
    void applyHead(const BrainV4& brain, bool night, uint32_t now, BrainV4::Executed& done);
    float rnd01(uint32_t now);
    uint32_t rndMs(uint32_t lo, uint32_t hi, uint32_t now);
    bool command(float yaw, float pitch, float speed, uint32_t now, bool force = false, uint32_t gap_ms = kGazeGapMs,
                 float dead_yaw_deg = kGazeDeadYawDeg);

    FaceEngine& face_;
    LedEngine& led_;
    BabbleVoice& vocal_;
    MotionEngine& motion_;

    // face
    Emotion shown_ = Emotion::Normal, candidate_ = Emotion::Normal;
    int candidate_ticks_ = 0;
    uint32_t face_since_ms_ = 0;
    // voice
    float tokens_ = 3.0f;
    uint32_t last_token_ms_ = 0, calls_ = 0;
    int reaction_ticks_ = 0;
    float prev_touch_ = 0.0f;
    float prev_answered_ = 0.0f;
    uint32_t turn_until_ms_ = 0;  // an answer's turn is open until (kTurnMs)
    // head
    Gesture gesture_ = Gesture::None;
    uint32_t gesture_start_ms_ = 0, last_gesture_ms_ = 0, last_command_ms_ = 0, gestures_ = 0, nods_ = 0, shakes_ = 0;
    int gesture_step_ = -1, gesture_reps_ = 2;
    float gesture_amp_ = 8.0f, gesture_half_s_ = 0.4f, base_yaw_ = 0.0f, base_pitch_ = 5.0f;
    float sent_yaw_ = 0.0f, sent_pitch_ = 5.0f;
    bool sent_valid_ = false;
    int prev_mode_ = 0;
    int head_cand_ = 0, head_cand_ticks_ = 0, head_mode_ = 0;  // the head mode, debounced
    uint32_t attended_ms_ = 0;  // when the attention last held something (kAttLook)
    uint32_t wander_hold_ms_ = kWanderHoldMinMs;  // how long the wandering gaze keeps its present place
    uint32_t next_glance_ms_ = 0, glance_until_ms_ = 0;
    uint32_t rng_ = 0;  // xorshift32 state (0: not seeded yet)
    uint32_t last_flinch_ms_ = 0;
    float watch_yaw_ = 0.0f, watch_pitch_ = 5.0f;
    uint32_t watch_until_ms_ = 0;
    uint32_t still_since_ms_ = 0;  // the head has not moved since
    bool idea_on_[5] = {false, false, false, false, false};  // the Muse's ideas above 0.5 (logged as they come)
    uint32_t idea_log_ms_[5] = {0, 0, 0, 0, 0};  // the next curious glance; the end of the present one (0: none)
    bool attended_ = false;
    uint8_t held_ = 0;  // this tick's Held bits
    uint32_t memory_until_ms_ = 0;  // the remembering glance holds the head until
    bool night_ = false;
};
