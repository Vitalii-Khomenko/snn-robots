#include "face_engine.h"

#include <esp_attr.h>

#include <esp_lcd_panel_ops.h>
#include <esp_lcd_touch.h>
#include "../core/ms_clock.h"

#include <bsp/esp-bsp.h>
#include <esp_random.h>
#include <esp_timer.h>

#include <algorithm>
#include <cmath>

namespace {

uint32_t nowMs() { return static_cast<uint32_t>(esp_timer_get_time() / 1000); }

float frand() { return static_cast<float>(esp_random() % 10000) / 10000.0f; }

// face::Painter on an LVGL draw layer (runs in the LVGL task during
// LV_EVENT_DRAW_MAIN). Coordinates are relative to the face object.
class LvglPainter : public face::Painter {
public:
    LvglPainter(lv_layer_t* layer, int ox, int oy) : layer_(layer), ox_(ox), oy_(oy) {}

    void roundRect(int x1, int y1, int x2, int y2, int radius, uint32_t color, uint8_t opa) override
    {
        lv_draw_rect_dsc_t d;
        lv_draw_rect_dsc_init(&d);
        d.bg_color = lv_color_hex(color);
        d.bg_opa = opa;
        d.radius = radius;
        lv_area_t a{ox_ + x1, oy_ + y1, ox_ + x2, oy_ + y2};
        lv_draw_rect(layer_, &d, &a);
    }
    void circle(int cx, int cy, int r, uint32_t color, uint8_t opa) override
    {
        roundRect(cx - r, cy - r, cx + r, cy + r, LV_RADIUS_CIRCLE, color, opa);
    }
    void triangle(int x1, int y1, int x2, int y2, int x3, int y3, uint32_t color) override
    {
        lv_draw_triangle_dsc_t d;
        lv_draw_triangle_dsc_init(&d);
        d.color = lv_color_hex(color);
        d.opa = LV_OPA_COVER;
        d.p[0].x = ox_ + x1;
        d.p[0].y = oy_ + y1;
        d.p[1].x = ox_ + x2;
        d.p[1].y = oy_ + y2;
        d.p[2].x = ox_ + x3;
        d.p[2].y = oy_ + y3;
        lv_draw_triangle(layer_, &d);
    }
    void line(int x1, int y1, int x2, int y2, int width, uint32_t color, uint8_t opa) override
    {
        lv_draw_line_dsc_t d;
        lv_draw_line_dsc_init(&d);
        d.color = lv_color_hex(color);
        d.opa = opa;
        d.width = width;
        d.round_start = d.round_end = 1;
        d.p1.x = ox_ + x1;
        d.p1.y = oy_ + y1;
        d.p2.x = ox_ + x2;
        d.p2.y = oy_ + y2;
        lv_draw_line(layer_, &d);
    }
    void arc(int cx, int cy, int r, int width, int startDeg, int endDeg, uint32_t color) override
    {
        lv_draw_arc_dsc_t d;
        lv_draw_arc_dsc_init(&d);
        d.color = lv_color_hex(color);
        d.width = width;
        d.rounded = 1;
        d.center.x = ox_ + cx;
        d.center.y = oy_ + cy;
        d.radius = static_cast<uint16_t>(std::max(1, r));
        d.start_angle = ((startDeg % 360) + 360) % 360;
        d.end_angle = ((endDeg % 360) + 360) % 360;
        lv_draw_arc(layer_, &d);
    }

private:
    lv_layer_t* layer_;
    int ox_, oy_;
};

}  // namespace

namespace {
void safeTouchRead(lv_indev_t* indev, lv_indev_data_t* data);
}

void FaceEngine::begin()
{
    bsp_display_lock(0);
    lv_obj_t* screen = lv_screen_active();
    lv_obj_clean(screen);
    lv_obj_set_style_bg_color(screen, lv_color_hex(face::kBackground), 0);
    lv_obj_set_style_bg_opa(screen, LV_OPA_COVER, 0);

    pixel_face_ = lv_obj_create(screen);
    lv_obj_set_size(pixel_face_, LV_PCT(100), LV_PCT(100));
    lv_obj_center(pixel_face_);
    lv_obj_set_style_bg_opa(pixel_face_, LV_OPA_TRANSP, 0);
    lv_obj_set_style_border_width(pixel_face_, 0, 0);
    lv_obj_set_style_pad_all(pixel_face_, 0, 0);
    lv_obj_set_style_radius(pixel_face_, 0, 0);
    lv_obj_set_scrollable(pixel_face_, false);
    // The face covers the full panel. Keep it clickable so the CoreS3 LCD
    // touch controller can reach the runtime; the event only sets an atomic
    // flag and never touches motion/SNN from LVGL's task.
    lv_obj_set_clickable(pixel_face_, true);
    lv_obj_add_event_cb(pixel_face_, &FaceEngine::drawEvent, LV_EVENT_DRAW_MAIN, this);
    lv_obj_add_event_cb(pixel_face_, &FaceEngine::touchEvent, LV_EVENT_PRESSED, this);
    // Frame cost: time from render start to the last chunk flushed.
    lv_display_t* disp = lv_display_get_default();
    lv_display_add_event_cb(disp, &FaceEngine::renderEvent, LV_EVENT_RENDER_START, this);
    lv_display_add_event_cb(disp, &FaceEngine::renderEvent, LV_EVENT_REFR_READY, this);

    emotion_ = Emotion::Normal;
    from_ = to_ = shape_ = drawn_shape_ = face::shapeFor(Emotion::Normal);
    if (lv_indev_t* touch = bsp_display_get_input_dev()) lv_indev_set_read_cb(touch, &safeTouchRead);
    bsp_display_unlock();

    const uint32_t now = nowMs();
    next_blink_ms_ = now + 2500 + esp_random() % 3000;
    next_quirk_ms_ = now + 5000 + esp_random() % 5000;
    dirty_ = true;
}

namespace {
// The first fields of esp_lvgl_port's private lvgl_port_display_ctx_t (esp_lvgl_port_disp.c, lvgl9), the display's
// driver data: the only way to the panel handle that bsp_display_start() keeps to itself. Checked against the
// managed component in this repository; if the port changes, recoverPanel() only does less (a null handle is skipped).
struct LvglPortDispPrefix {
    int disp_type;
    esp_lcd_panel_io_handle_t io_handle;
    esp_lcd_panel_handle_t panel_handle;
};
}  // namespace

namespace {
// The first fields of esp_lvgl_port's private lvgl_port_touch_ctx_t (esp_lvgl_port_touch.c, lvgl9): the touch
// controller's handle and the coordinate scale.
struct LvglPortTouchPrefix {
    esp_lcd_touch_handle_t handle;
    lv_indev_t* indev;
    struct {
        float x, y;
    } scale;
};

// The port's read callback wraps every I2C read in ESP_ERROR_CHECK: one failed read of the screen's touch controller
// on the shared, sometimes glitchy I2C bus aborted the whole robot (panic in taskLVGL, 2026-10-17). This one skips
// the sample instead.
void safeTouchRead(lv_indev_t* indev, lv_indev_data_t* data)
{
    const auto* ctx = static_cast<const LvglPortTouchPrefix*>(lv_indev_get_driver_data(indev));
    data->state = LV_INDEV_STATE_RELEASED;
    if (!ctx || !ctx->handle || esp_lcd_touch_read_data(ctx->handle) != ESP_OK) return;
    esp_lcd_touch_point_data_t pts[CONFIG_ESP_LCD_TOUCH_MAX_POINTS] = {};
    uint8_t n = 0;
    if (esp_lcd_touch_get_data(ctx->handle, pts, &n, CONFIG_ESP_LCD_TOUCH_MAX_POINTS) != ESP_OK || n == 0) return;
    data->point.x = static_cast<int32_t>(ctx->scale.x * pts[0].x);
    data->point.y = static_cast<int32_t>(ctx->scale.y * pts[0].y);
    data->state = LV_INDEV_STATE_PRESSED;
}
}  // namespace

void FaceEngine::recoverPanel(bool reset)
{
    if (!bsp_display_lock(500)) return;  // LVGL is between frames while we hold its lock: no transfer is running
    lv_display_t* disp = lv_display_get_default();
    const auto* ctx = disp ? static_cast<const LvglPortDispPrefix*>(lv_display_get_driver_data(disp)) : nullptr;
    esp_lcd_panel_handle_t panel = ctx ? ctx->panel_handle : nullptr;
    if (panel) {
        if (reset) esp_lcd_panel_reset(panel);  // no reset pin on this board: the software reset command
        esp_lcd_panel_init(panel);  // sleep out, orientation, colour format, the vendor's settings (~0.3 s)
        esp_lcd_panel_invert_color(panel, true);
        esp_lcd_panel_disp_on_off(panel, true);
    }
    lv_obj_invalidate(lv_screen_active());
    bsp_display_unlock();
    bsp_display_backlight_on();
    dirty_ = true;
}

void FaceEngine::setGaze(float x, float y)
{
    gaze_x_ = std::clamp(x, -1.0f, 1.0f);
    gaze_y_ = std::clamp(y, -1.0f, 1.0f);
}

void FaceEngine::setEmotion(Emotion emotion)
{
    if (emotion == emotion_) return;
    from_emotion_ = morph_t_ < 0.5f ? from_emotion_ : emotion_;  // pixel face: dissolve from what mostly shows
    emotion_ = emotion;
    from_ = shape_;  // morph from wherever the face is right now
    to_ = face::shapeFor(emotion);
    morph_start_ms_ = nowMs();
    dirty_ = true;
}

void FaceEngine::drawEvent(lv_event_t* event)
{
    auto* self = static_cast<FaceEngine*>(lv_event_get_user_data(event));
    if (self != nullptr) self->draw(lv_event_get_layer(event));
}

void FaceEngine::renderEvent(lv_event_t* event)
{
    auto* self = static_cast<FaceEngine*>(lv_event_get_user_data(event));
    if (self == nullptr) return;
    const int64_t t = esp_timer_get_time();
    if (lv_event_get_code(event) == LV_EVENT_RENDER_START) {
        if (self->render_t0_ == 0) self->render_t0_ = t;
    } else if (self->render_t0_ != 0) {
        const uint32_t us = static_cast<uint32_t>(t - self->render_t0_);
        self->render_t0_ = 0;
        if (us > self->draw_max_us_.load()) self->draw_max_us_.store(us);
    }
}

void FaceEngine::touchEvent(lv_event_t* event)
{
    auto* self = static_cast<FaceEngine*>(lv_event_get_user_data(event));
    if (self != nullptr) self->screen_touch_pending_.store(true);
}

void FaceEngine::draw(lv_layer_t* layer) const
{
    if (pixel_face_ == nullptr) return;
    lv_area_t area;
    lv_obj_get_coords(pixel_face_, &area);
    const int ox = area.x1 + (lv_obj_get_width(pixel_face_) - face::kScreenW) / 2;
    const int oy = area.y1 + (lv_obj_get_height(pixel_face_) - face::kScreenH) / 2;
    LvglPainter painter(layer, ox, oy);
    if constexpr (face::kPixelFace) {
        // LVGL task only: three grids of 3 KB, in PSRAM (internal RAM is scarce)
        EXT_RAM_BSS_ATTR static face::PixelGrid to, from, mixed;
        face::renderPixelFace(drawn_emotion_, drawn_anim_, to);
        if (drawn_morph_t_ < 1.0f) {
            face::renderPixelFace(drawn_from_, drawn_anim_, from);
            face::dissolve(from, to, drawn_morph_t_, mixed);
            mixed.flush(painter);
        } else {
            to.flush(painter);
        }
        return;
    }
    face::renderFace(drawn_shape_, drawn_anim_, painter);
    if (person_seen_) {  // head and shoulders, green
        painter.roundRect(282, 21, 302, 33, 6, 0x1FD060, LV_OPA_COVER);
        painter.circle(292, 13, 6, 0x1FD060, LV_OPA_COVER);
    }
}

bool FaceEngine::animate(uint32_t now)
{
    bool changed = dirty_;
    dirty_ = false;
    bool continuous = false;  // something keeps moving (limits the frame rate)

    // Morph with ease-out.
    const uint32_t mt = now - morph_start_ms_;
    if (mt < kMorphMs + 40) {
        const float t = std::min(1.0f, static_cast<float>(mt) / kMorphMs);
        const float e = 1.0f - (1.0f - t) * (1.0f - t);
        shape_ = face::blend(from_, to_, e);
        morph_t_ = t;
        changed = true;
    } else {
        shape_ = to_;
        morph_t_ = 1.0f;
    }

    face::Anim a{};

    // Blink (quick close, short hold, slower open; sometimes twice).
    const bool can_blink = emotion_ != Emotion::Sleepy && emotion_ != Emotion::Laughing && emotion_ != Emotion::Love &&
                           emotion_ != Emotion::Dizzy && emotion_ != Emotion::Crying;
    switch (blink_) {
        case Blink::Open:
            if ((msReached(now, next_blink_ms_) || blink_request_) && can_blink) {
                blink_request_ = false;
                blink_ = Blink::Closing;
                blink_t0_ = now;
            }
            break;
        case Blink::Closing:
            a.blink = std::min(1.0f, (now - blink_t0_) / 70.0f);
            if (now - blink_t0_ >= 70) {
                blink_ = Blink::Closed;
                blink_t0_ = now;
            }
            break;
        case Blink::Closed:
            a.blink = 1.0f;
            if (now - blink_t0_ >= 50) {
                blink_ = Blink::Opening;
                blink_t0_ = now;
            }
            break;
        case Blink::Opening:
            a.blink = 1.0f - std::min(1.0f, (now - blink_t0_) / 100.0f);
            if (now - blink_t0_ >= 100) {
                blink_ = Blink::Open;
                if (!double_blink_ && esp_random() % 5 == 0) {
                    double_blink_ = true;
                    next_blink_ms_ = now + 120;
                } else {
                    double_blink_ = false;
                    // Surprise and fear stare; calm and tired blink more.
                    const uint32_t base = emotion_ == Emotion::Surprised || emotion_ == Emotion::Scared ? 5000 : 2800;
                    next_blink_ms_ = now + base + esp_random() % 3500;
                }
            }
            break;
    }
    if (blink_ != Blink::Open) changed = true;

    // Idle life: now and then a glance aside and back.
    if (msReached(now, next_quirk_ms_)) {
        if (emotion_ == Emotion::Normal || emotion_ == Emotion::Bored || emotion_ == Emotion::Curious) {
            quirk_x_ = (esp_random() % 2 ? 1.0f : -1.0f) * (8.0f + 6.0f * frand());
            quirk_until_ms_ = now + 500 + esp_random() % 500;
        }
        next_quirk_ms_ = now + 4000 + esp_random() % 6000;
    }
    const float quirk = msWithin(now, quirk_until_ms_) ? quirk_x_ : 0.0f;

    // Gaze: the eyes glide (no teleporting).
    const float gx = gaze_x_ * 12.0f + quirk, gy = -gaze_y_ * 9.0f;
    eyes_x_ += (gx - eyes_x_) * 0.35f;
    eyes_y_ += (gy - eyes_y_) * 0.35f;
    a.eyesX = std::round(eyes_x_);
    a.eyesY = std::round(eyes_y_);

    // Per-emotion motion.
    // Animation clock in seconds, on a 120 s cycle: now / 1000 as a float
    // loses precision with uptime (0.06 s steps after ~12 days: bouncing
    // and pulsing turned jerky). Every period below divides 120 s, so the
    // wrap is seamless.
    const float ts = static_cast<float>(now % 120000U) / 1000.0f;
    switch (emotion_) {
        case Emotion::Scared:  // trembling, pupils darting
            a.shakeX = static_cast<float>(static_cast<int>(esp_random() % 5) - 2);
            a.shakeY = static_cast<float>(static_cast<int>(esp_random() % 3) - 1);
            a.eyesX += static_cast<float>(static_cast<int>(esp_random() % 5) - 2);
            a.phase = std::fmod(ts / 1.5f, 1.0f);
            continuous = true;
            break;
        case Emotion::Laughing:  // bouncing with the laugh
            a.shakeY = -std::round(5.0f * std::fabs(std::sin(ts * 3.14159f * 3.0f)));
            continuous = true;
            break;
        case Emotion::Love:
        case Emotion::VeryHappy:
        case Emotion::Wink:
            a.pulse = 0.5f + 0.5f * std::sin(ts * 3.14159f * 2.6f);
            continuous = true;
            break;
        case Emotion::Crying:
            a.phase = std::fmod(ts / 1.2f, 1.0f);
            a.shakeY = std::round(1.5f * std::sin(ts * 3.14159f * 4.0f));  // sobbing
            continuous = true;
            break;
        case Emotion::Sleepy:
            a.phase = std::fmod(ts / 2.5f, 1.0f);
            a.shakeY = std::round(2.0f * std::sin(ts * 3.14159f * 0.5f));  // slow breathing
            continuous = true;
            break;
        case Emotion::Sick:
            a.phase = std::fmod(ts / 1.5f, 1.0f);
            continuous = true;
            break;
        default:
            break;
    }
    const bool moved = a.eyesX != drawn_anim_.eyesX || a.eyesY != drawn_anim_.eyesY ||
                       a.blink != drawn_anim_.blink || a.shakeX != drawn_anim_.shakeX ||
                       a.shakeY != drawn_anim_.shakeY || std::fabs(a.pulse - drawn_anim_.pulse) > 0.08f ||
                       std::fabs(a.phase - drawn_anim_.phase) > 0.04f;
    anim_ = a;
    if (!moved && !changed) return false;
    // Continuous motion alone: ~15 frames/s is plenty and keeps the CPU free.
    if (!changed && continuous && blink_ == Blink::Open && now - last_frame_ms_ < 66) return false;
    return true;
}

void FaceEngine::update()
{
    if (pixel_face_ == nullptr) return;
    const uint32_t now = nowMs();
    if (now - last_frame_ms_ < 30) return;  // LVGL refreshes at ~30 Hz anyway
    if (!animate(now) && !frame_pending_) return;
    // Never make the main loop (and the brain) wait for the LVGL task while
    // it renders a frame: if the display is busy, hand over the new
    // picture on the next pass.
    if (!bsp_display_lock(2)) {
        frame_pending_ = true;
        return;
    }
    frame_pending_ = false;
    last_frame_ms_ = now;
    frames_.fetch_add(1);
    drawn_shape_ = shape_;
    drawn_anim_ = anim_;
    drawn_emotion_ = emotion_;
    drawn_from_ = from_emotion_;
    drawn_morph_t_ = morph_t_;
    lv_obj_invalidate(pixel_face_);
    bsp_display_unlock();
}
