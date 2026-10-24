#pragma once
#include <array>
#include <cstdint>

#include "channel_encoder.h"
#include "nn_v4.h"
#include "snn_v4.h"

// Brain v4 (BRAIN_V4_PLAN.txt): the spiking analyzer every 50 ms tick, the main network every second tick (10 Hz),
// and what the main network sees of its own past ("what I just did", v4_spec.h Prev): the shown emotion and for how
// long, when it last spoke or gestured, where the head is, the LEDs, whether the last touch has been answered, and
// three slow pacemakers (its random choices are made visible to it). Mirrors tools/v4/teacher_nn.py
// (Teacher.prev_inputs / apply) and pacemakers().
class BrainV4 {
public:
    // What the executor really did in the main network's tick (v4_executor.h).
    struct Executed {
        int emotion = 0;          // shown face (Emotion enum = v4::EMOTIONS order)
        bool said = false;        // a phrase started
        bool gesture = false;     // nod / shake / flinch started
        float headYawDeg = 0, headPitchDeg = 0;
        float ledLevel = 0.3f;
    };

    BrainV4();
    // One brain tick. true when the main network decided this tick (then nn() is new).
    bool tick(const ChannelEncoder::Out& channels);
    void executed(const Executed& e);

    const SnnV4::Out& snn() const { return snn_out_; }
    const NnV4::Out& nn() const { return nn_out_; }
    bool trained() const { return snn_.trained() && nn_.trained(); }
    bool nnValid() const { return nn_valid_; }
    SnnV4& analyzer() { return snn_; }
    uint32_t nnFaults() const { return nn_faults_; }
    void seed(uint32_t s) { rng_ = s ? s : 1; }

    static constexpr float kNnDt = 0.1f;
    static constexpr uint32_t kSettleTicks = 100;  // 5 s
    static constexpr float kGazeYawDeg = 60.0f, kGazePitchDeg = 30.0f;  // gaze -1..1 -> degrees (v4_executor.h)

private:
    float gauss();
    void prevInputs(float* x) const;

    SnnV4 snn_;
    NnV4 nn_;
    SnnV4::Out snn_out_{};
    NnV4::Out nn_out_{};
    bool nn_valid_ = false;
    uint32_t ticks_ = 0, nn_faults_ = 0, rng_ = 0x2545f491u;
    std::array<float, 3> pace_v_{}, pace_{};
    // the past, as teacher_nn.Teacher keeps it
    int emo_ = 0;
    float emo_age_ = 10.0f, since_sound_ = 10.0f, since_gesture_ = 30.0f, touch_age_ = 0.0f, prev_touch_ = 0.0f;
    bool touch_open_ = false;
    float head_yaw_ = 0.0f, head_pitch_ = 0.0f, led_level_ = 0.3f;
};
