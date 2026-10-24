#pragma once

#include <atomic>
#include <cstdint>

#include <lvgl.h>

#include "face_style.h"

// The robot's animated face. The look of every emotion lives in
// face_style.{h,cpp} (pure geometry, previewable on the PC with
// tools/face_preview); this class animates it on the LCD through LVGL:
//  - an emotion change morphs smoothly (~0.25 s) instead of jumping;
//  - blinks (sometimes double), eyes glide after the gaze;
//  - small idle life when nothing happens (a glance aside, a slow blink);
//  - per-emotion motion: fear trembles, laughter bounces, hearts beat,
//    tears fall, "z" rises.
// CPU: one transparent LVGL object; nothing is redrawn while nothing moves,
// continuous motion is capped at ~15 frames/s, morphs and blinks at ~30.
class FaceEngine {
public:
    void begin();
    void update();  // call from the main loop (every ~20 ms)
    // Brings the panel back after a power dip (the servo rail's power cycle can reset the LCD controller: the
    // picture went black while the robot kept living, 2026-10-11): software reset, init, backlight, full redraw.
    void recoverPanel(bool reset = true);
    void setEmotion(Emotion e);
    // Eye direction, -1..1 (right, up positive).
    void setGaze(float x, float y);
    // A blink on request (the policy's blink output); the idle blink timer keeps running.
    void blink() { blink_request_ = true; }
    // A small person symbol in the top-right corner while the robot recognises a person (green).
    void setPersonSeen(bool seen)
    {
        if (seen == person_seen_) return;
        person_seen_ = seen;
        dirty_ = true;
    }
    bool takeScreenTouch() { return screen_touch_pending_.exchange(false); }
    Emotion emotion() const { return emotion_; }
    // CPU cost for the stats line: frames drawn and the slowest draw (us)
    // since the last call.
    void takeDrawStats(uint32_t& frames, uint32_t& maxUs)
    {
        frames = frames_.exchange(0);
        maxUs = draw_max_us_.exchange(0);
    }

private:
    static void drawEvent(lv_event_t* event);
    static void touchEvent(lv_event_t* event);
    static void renderEvent(lv_event_t* event);
    int64_t render_t0_ = 0;  // LVGL task only
    void draw(lv_layer_t* layer) const;
    bool animate(uint32_t now);  // updates target_anim_/shape_, true if the picture changed

    lv_obj_t* pixel_face_ = nullptr;
    Emotion emotion_ = Emotion::Normal;

    // Morph between expressions.
    face::Shape from_{}, to_{}, shape_{};
    uint32_t morph_start_ms_ = 0;
    static constexpr uint32_t kMorphMs = 240;

    // Animation state (updated in update(), drawn in draw()).
    face::Anim anim_{}, drawn_anim_{};
    face::Shape drawn_shape_{};
    float gaze_x_ = 0.0f, gaze_y_ = 0.0f;       // target
    float eyes_x_ = 0.0f, eyes_y_ = 0.0f;       // smoothed, px

    // Blink.
    enum class Blink : uint8_t { Open, Closing, Closed, Opening };
    Blink blink_ = Blink::Open;
    uint32_t blink_t0_ = 0, next_blink_ms_ = 0;
    bool double_blink_ = false;
    bool blink_request_ = false;

    // Idle glance.
    uint32_t next_quirk_ms_ = 0, quirk_until_ms_ = 0;
    float quirk_x_ = 0.0f;

    bool person_seen_ = false;

    uint32_t last_frame_ms_ = 0;
    bool dirty_ = true;
    bool frame_pending_ = false;  // a frame is ready but the display was busy
    mutable std::atomic<uint32_t> frames_{0}, draw_max_us_{0};
    std::atomic<bool> screen_touch_pending_{false};

    // Simulator bookkeeping (simulator/hal/platform.cpp).
    int gaze_dx_ = 0, gaze_dy_ = 0;
    static constexpr int kGazeRangeX = 3, kGazeRangeY = 2;
};
