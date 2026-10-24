#include "face_style.h"

#include <algorithm>
#include <cmath>

const char* emotionName(Emotion e)
{
    static const char* const names[kEmotionCount] = {
        "Normal", "Happy", "VeryHappy", "Laughing", "Sad",   "Crying",  "Sleepy",   "Surprised", "Angry", "Annoyed", "Love",
        "Shy",    "Wink",  "Curious",   "Confused", "Thinking", "Bored", "Scared", "Dizzy",   "Cool",      "Sick"};
    const int i = static_cast<int>(e);
    return i >= 0 && i < kEmotionCount ? names[i] : "?";
}

namespace face {
namespace {

constexpr float kPi = 3.14159265f;
constexpr int kEyeCx = 74;       // eye centres at 160 -+ 74
constexpr int kEyeCy = 92;
constexpr int kMouthCy = 186;

Shape both(const Eye& e)
{
    Shape s;
    s.left = s.right = e;
    return s;
}

float lerp(float a, float b, float t) { return a + (b - a) * t; }

uint32_t lerpColor(uint32_t a, uint32_t b, float t)
{
    auto ch = [&](int sh) {
        const float v = lerp(static_cast<float>((a >> sh) & 0xFF), static_cast<float>((b >> sh) & 0xFF), t);
        return static_cast<uint32_t>(std::clamp(v, 0.0f, 255.0f)) << sh;
    };
    return ch(16) | ch(8) | ch(0);
}

Eye blendEye(const Eye& a, const Eye& b, float t)
{
    Eye e;
    e.w = lerp(a.w, b.w, t);
    e.h = lerp(a.h, b.h, t);
    e.r = lerp(a.r, b.r, t);
    e.dx = lerp(a.dx, b.dx, t);
    e.dy = lerp(a.dy, b.dy, t);
    e.lidTop = lerp(a.lidTop, b.lidTop, t);
    e.lidTilt = lerp(a.lidTilt, b.lidTilt, t);
    e.lidBot = lerp(a.lidBot, b.lidBot, t);
    e.pupil = lerp(a.pupil, b.pupil, t);
    e.brow = lerp(a.brow, b.brow, t);
    e.browY = lerp(a.browY, b.browY, t);
    e.browTilt = lerp(a.browTilt, b.browTilt, t);
    e.kind = t < 0.5f ? a.kind : b.kind;
    return e;
}

}  // namespace

Shape blend(const Shape& a, const Shape& b, float t)
{
    t = std::clamp(t, 0.0f, 1.0f);
    Shape s;
    s.left = blendEye(a.left, b.left, t);
    s.right = blendEye(a.right, b.right, t);
    s.mouth = t < 0.5f ? a.mouth : b.mouth;
    s.mouthW = lerp(a.mouthW, b.mouthW, t);
    s.mouthCurve = lerp(a.mouthCurve, b.mouthCurve, t);
    s.mouthOpen = lerp(a.mouthOpen, b.mouthOpen, t);
    s.mouthDy = lerp(a.mouthDy, b.mouthDy, t);
    s.color = lerpColor(a.color, b.color, t);
    s.extras = t < 0.5f ? a.extras : b.extras;
    return s;
}

// The expression vocabulary. Rules of thumb that make faces readable:
//  - joy lives in the LOWER lids (smiling eyes) and an upturned mouth;
//  - sadness / worry / fear: inner ends of lids and brows go UP;
//  - anger: inner ends go DOWN, eyes narrowed from the top;
//  - surprise: everything opens (tall eyes, raised brows, round mouth);
//  - fear = surprise + worried brows + tiny trembling pupils + wavy mouth.
Shape shapeFor(Emotion emotion)
{
    Eye base;
    Shape s = both(base);
    switch (emotion) {
        case Emotion::Normal:
            s.mouth = MouthKind::Smile;  // a hint of a smile: friendly, not blank
            s.mouthW = 41;
            s.mouthCurve = 0.2f;
            break;
        case Emotion::Happy:
            s.left.lidBot = s.right.lidBot = 0.58f;
            s.mouth = MouthKind::Smile;
            s.mouthW = 53;
            s.mouthCurve = 0.6f;
            s.color = 0xFFE27A;
            break;
        case Emotion::VeryHappy:
            s.left.lidBot = s.right.lidBot = 0.62f;
            s.left.dy = s.right.dy = -4;
            s.mouth = MouthKind::OpenSmile;
            s.mouthW = 60;
            s.mouthOpen = 0.7f;
            s.color = 0xFFD45A;
            s.extras = kGlint | kSparkles | kBlush;
            break;
        case Emotion::Laughing:
            s.left.kind = s.right.kind = EyeKind::ArcUp;
            s.mouth = MouthKind::OpenSmile;
            s.mouthW = 72;
            s.mouthOpen = 1.0f;
            s.color = 0xFFC94A;
            s.extras = kBlush;
            break;
        case Emotion::Sad:
            s.left.h = s.right.h = 74;
            s.left.dy = s.right.dy = 6;
            s.left.lidTop = s.right.lidTop = 0.25f;
            s.left.lidTilt = s.right.lidTilt = -0.9f;
            s.left.brow = s.right.brow = 1;
            s.left.browTilt = s.right.browTilt = -1;
            s.mouth = MouthKind::Smile;
            s.mouthCurve = -0.55f;
            s.mouthW = 46;
            s.mouthDy = 4;
            s.color = 0x7FB2FF;
            break;
        case Emotion::Crying:
            s.left.kind = s.right.kind = EyeKind::ArcDown;
            s.left.brow = s.right.brow = 1;
            s.left.browTilt = s.right.browTilt = -1;
            s.mouth = MouthKind::Wavy;
            s.mouthW = 53;
            s.color = 0x6FA2FF;
            s.extras = kTears;
            break;
        case Emotion::Sleepy:
            s.left.lidTop = s.right.lidTop = 0.62f;
            s.left.dy = s.right.dy = 8;
            s.mouth = MouthKind::Small;
            s.mouthW = 14;
            s.color = 0x8FA8D8;
            s.extras = kZzz;
            break;
        case Emotion::Surprised:
            s.left.w = s.right.w = 74;
            s.left.h = s.right.h = 101;
            s.left.r = s.right.r = 36;
            s.left.dy = s.right.dy = -6;
            s.left.brow = s.right.brow = 1;
            s.left.browY = s.right.browY = 12;
            s.mouth = MouthKind::O;
            s.mouthOpen = 0.8f;
            s.color = 0xFFFFFF;
            break;
        case Emotion::Scared:
            s.left.kind = s.right.kind = EyeKind::Ring;
            s.left.w = s.right.w = 67;
            s.left.h = s.right.h = 91;
            s.left.r = s.right.r = 34;
            s.left.pupil = s.right.pupil = 7;
            s.left.brow = s.right.brow = 1;
            s.left.browY = s.right.browY = 8;
            s.left.browTilt = s.right.browTilt = -1;
            s.mouth = MouthKind::Wavy;
            s.mouthW = 55;
            s.color = 0xC9A8FF;
            s.extras = kSweat;
            break;
        case Emotion::Angry:
            s.left.lidTop = s.right.lidTop = 0.32f;
            s.left.lidTilt = s.right.lidTilt = 1.0f;
            s.left.brow = s.right.brow = 1;
            s.left.browTilt = s.right.browTilt = 1.2f;
            s.left.browY = s.right.browY = -4;
            s.mouth = MouthKind::Teeth;
            s.mouthW = 55;
            s.color = 0xFF5A4A;
            s.extras = kAnger;
            break;
        case Emotion::Annoyed:
            s.left.lidTop = s.right.lidTop = 0.45f;
            s.left.lidTilt = s.right.lidTilt = 0.35f;
            s.left.dx = s.right.dx = 10;  // side-eye
            s.mouth = MouthKind::Smile;
            s.mouthCurve = -0.3f;
            s.mouthW = 36;
            s.mouthDy = 2;
            s.color = 0xFF9A7A;
            break;
        case Emotion::Love:
            s.left.kind = s.right.kind = EyeKind::Heart;
            s.mouth = MouthKind::Smile;
            s.mouthW = 48;
            s.mouthCurve = 0.7f;
            s.color = 0xFF7FAE;
            s.extras = kBlush;
            break;
        case Emotion::Shy:
            s.left.lidBot = s.right.lidBot = 0.25f;
            s.left.lidTop = s.right.lidTop = 0.15f;
            s.left.dy = s.right.dy = 8;
            s.left.dx = s.right.dx = -8;  // looking away and down
            s.mouth = MouthKind::Small;
            s.mouthW = 17;
            s.color = 0xFFC2D6;
            s.extras = kBlush;
            break;
        case Emotion::Wink:
            s.right.kind = EyeKind::ArcUp;
            s.left.lidBot = 0.3f;
            s.mouth = MouthKind::Side;
            s.mouthW = 43;
            s.color = 0xFFE27A;
            s.extras = kGlint | kSparkles;
            break;
        case Emotion::Curious:
            s.left.h = 94;
            s.left.w = 74;
            s.left.brow = 1;
            s.left.browY = 12;
            s.right.h = 72;
            s.right.lidTop = 0.15f;
            s.left.dy = -4;
            s.mouth = MouthKind::Small;
            s.mouthW = 19;
            s.color = 0x8FFFD0;
            break;
        case Emotion::Confused:
            s.left.h = 89;
            s.right.h = 62;
            s.right.lidTop = 0.2f;
            s.right.lidTilt = 0.5f;
            s.left.brow = 1;
            s.left.browY = 10;
            s.left.browTilt = -0.6f;
            s.mouth = MouthKind::Wavy;
            s.mouthW = 38;
            s.color = 0xD8C8FF;
            s.extras = kQuestion;
            break;
        case Emotion::Thinking:
            s.left.dx = s.right.dx = 12;
            s.left.dy = s.right.dy = -10;
            s.right.lidTop = 0.3f;
            s.right.brow = 1;
            s.right.browY = 6;
            s.right.browTilt = 0.4f;
            s.mouth = MouthKind::Side;
            s.mouthW = 34;
            s.color = 0xA8D8FF;
            break;
        case Emotion::Bored:
            s.left.lidTop = s.right.lidTop = 0.5f;
            s.left.lidBot = s.right.lidBot = 0.12f;
            s.left.dx = s.right.dx = -6;
            s.mouth = MouthKind::Flat;
            s.mouthW = 41;
            s.mouthDy = 4;
            s.color = 0xA8B4C8;
            s.extras = kNone;
            break;
        case Emotion::Dizzy:
            s.left.kind = s.right.kind = EyeKind::Cross;
            s.mouth = MouthKind::Wavy;
            s.mouthW = 48;
            s.color = 0xD9C2FF;
            break;
        case Emotion::Cool:
            s.left.lidTop = s.right.lidTop = 0.45f;
            s.left.lidTilt = s.right.lidTilt = -0.2f;
            s.mouth = MouthKind::Side;
            s.mouthW = 41;
            s.color = 0xB8C4D9;
            break;
        case Emotion::Sick:
            s.left.lidTop = s.right.lidTop = 0.4f;
            s.left.lidTilt = s.right.lidTilt = -0.5f;
            s.mouth = MouthKind::Wavy;
            s.mouthW = 43;
            s.color = 0xB8E8A0;
            s.extras = kSweat;
            break;
    }
    return s;
}

namespace {

void drawHeart(Painter& p, int cx, int cy, float size, uint32_t c)
{
    const int r = static_cast<int>(size * 0.27f);
    const int off = static_cast<int>(size * 0.24f);
    const int top = cy - static_cast<int>(size * 0.12f);
    p.circle(cx - off, top, r, c);
    p.circle(cx + off, top, r, c);
    const int half = static_cast<int>(size * 0.5f);
    p.triangle(cx - half, top + r / 3, cx + half, top + r / 3, cx, cy + static_cast<int>(size * 0.45f), c);
}

void drawEye(Painter& p, const Eye& e, int cx, int cy, bool leftEye, const Anim& a, uint32_t col, uint16_t extras)
{
    cx += static_cast<int>(e.dx + a.eyesX);
    cy += static_cast<int>(e.dy + a.eyesY);
    const float w = e.w, h = e.h;
    const int x1 = cx - static_cast<int>(w / 2), x2 = cx + static_cast<int>(w / 2);
    int y1 = cy - static_cast<int>(h / 2);
    const int y2 = cy + static_cast<int>(h / 2);
    // inner = towards the middle of the face
    const int inner = leftEye ? x2 : x1, outer = leftEye ? x1 : x2;

    // Brow first (above the lid it frames).
    if (e.brow > 0.05f) {
        const float by = static_cast<float>(y1) - 12.0f - e.browY;
        const float tilt = e.browTilt * 9.0f;
        const int bi = static_cast<int>(by + tilt), bo = static_cast<int>(by - tilt * 0.4f);
        const int span = static_cast<int>(w * 0.42f * e.brow);
        const int ix = leftEye ? cx + span : cx - span, ox = leftEye ? cx - span : cx + span;
        p.line(ix, bi, ox, bo, 8, col);
    }

    switch (e.kind) {
        case EyeKind::Heart: {
            const float beat = 1.0f + 0.12f * a.pulse;
            drawHeart(p, cx, cy, w * 1.05f * beat, col);
            return;
        }
        case EyeKind::ArcUp:
            p.arc(cx, cy + static_cast<int>(h * 0.18f), static_cast<int>(w * 0.42f), 9, 200, 340, col);
            return;
        case EyeKind::ArcDown:
            p.arc(cx, cy - static_cast<int>(h * 0.22f), static_cast<int>(w * 0.40f), 9, 20, 160, col);
            return;
        case EyeKind::Line:
            p.roundRect(x1 + 4, cy - 4, x2 - 4, cy + 4, 4, col);
            return;
        case EyeKind::Cross: {
            const int d = static_cast<int>(w * 0.32f);
            p.line(cx - d, cy - d, cx + d, cy + d, 9, col);
            p.line(cx - d, cy + d, cx + d, cy - d, 9, col);
            return;
        }
        default:
            break;
    }

    // Blink closes the eye from the top: at 1 it is a line.
    const float lidTop = std::max(e.lidTop, a.blink);
    if (lidTop > 0.9f) {
        p.roundRect(x1 + 4, y2 - 8, x2 - 4, y2, 4, col);
        return;
    }

    p.roundRect(x1, y1, x2, y2, static_cast<int>(e.r), col);
    if (e.kind == EyeKind::Ring) {
        const int t = 7;
        p.roundRect(x1 + t, y1 + t, x2 - t, y2 - t, std::max(2, static_cast<int>(e.r) - t), kBackground);
        const int px = cx + static_cast<int>(a.eyesX * 0.6f), py = cy + static_cast<int>(a.eyesY * 0.6f);
        p.circle(px, py, static_cast<int>(e.pupil), col);
    } else if ((extras & kGlint) && lidTop < 0.12f && e.lidBot < 0.4f) {
        const int gx = cx + static_cast<int>(w * 0.18f), gy = y1 + static_cast<int>(h * 0.22f);
        p.roundRect(gx - 5, gy - 5, gx + 5, gy + 5, 3, kBackground, 150);
    }

    // Top lid: a background-coloured quad over the eye, slanted.
    if (lidTop > 0.01f) {
        const float base = static_cast<float>(y1) + lidTop * h;
        const float slant = e.lidTilt * h * 0.28f;
        const int yi = static_cast<int>(base + slant), yo = static_cast<int>(base - slant * 0.6f);
        const int top = y1 - 4;
        const int l = x1 - 3, r = x2 + 3;
        const int yl = leftEye ? yo : yi, yr = leftEye ? yi : yo;
        p.triangle(l, top, r, top, r, yr, kBackground);
        p.triangle(l, top, r, yr, l, yl, kBackground);
        (void)inner;
        (void)outer;
    }
    // Lower lid: a big background dome rising from below = smiling eyes.
    if (e.lidBot > 0.01f) {
        // A round cut from below: the eye becomes a crescent, like a smile.
        const int R = static_cast<int>(w * 0.62f);
        const int topY = y2 - static_cast<int>(e.lidBot * h);
        p.circle(cx, topY + R, R, kBackground);
    }
    (void)y1;
}

void drawMouth(Painter& p, const Shape& s, const Anim& a, int cx, int cy)
{
    const uint32_t c = s.color;
    cy += static_cast<int>(s.mouthDy);
    const int hw = static_cast<int>(s.mouthW / 2);
    switch (s.mouth) {
        case MouthKind::None:
            break;
        case MouthKind::Flat:
            p.roundRect(cx - hw, cy - 5, cx + hw, cy + 5, 5, c);
            break;
        case MouthKind::Smile: {
            const float curve = std::clamp(std::fabs(s.mouthCurve), 0.15f, 1.0f);
            const float span = (40.0f + 80.0f * curve) * kPi / 180.0f;  // total angle
            const float r = s.mouthW / 2.0f / std::sin(span / 2.0f);
            const int R = static_cast<int>(r);
            const int half = static_cast<int>(span * 90.0f / kPi);
            if (s.mouthCurve >= 0) {
                const int ccy = cy - static_cast<int>(r * std::cos(span / 2.0f)) - 4;
                p.arc(cx, ccy, R, 11, 90 - half, 90 + half, c);
            } else {
                const int ccy = cy + static_cast<int>(r * std::cos(span / 2.0f)) + 4;
                p.arc(cx, ccy, R, 11, 270 - half, 270 + half, c);
            }
            break;
        }
        case MouthKind::OpenSmile: {
            const int depth = static_cast<int>(10 + 22 * s.mouthOpen);
            p.roundRect(cx - hw, cy - depth / 2, cx + hw, cy + depth, std::min(hw, depth), c);
            p.roundRect(cx - hw - 2, cy - depth / 2 - 2, cx + hw + 2, cy - depth / 2 + 6, 2, kBackground);
            // tongue
            if (s.mouthOpen > 0.6f)
                p.roundRect(cx - hw / 2, cy + depth / 2, cx + hw / 2, cy + depth - 4, 8, 0xFF6E86);
            break;
        }
        case MouthKind::O: {
            const int r = static_cast<int>(12 + 16 * s.mouthOpen);
            p.roundRect(cx - r + 4, cy - r, cx + r - 4, cy + r, r, c);
            p.roundRect(cx - r + 13, cy - r + 9, cx + r - 13, cy + r - 9, r - 9, kBackground);
            break;
        }
        case MouthKind::Wavy: {
            const int n = 6;
            const float step = s.mouthW / n;
            const float amp = 7.0f;
            float px = static_cast<float>(cx - hw), py = static_cast<float>(cy);
            for (int i = 1; i <= n; ++i) {
                const float nx = static_cast<float>(cx - hw) + step * i;
                const float ny = static_cast<float>(cy) + ((i % 2) ? -amp : amp) + a.shakeY * 0.5f;
                p.line(static_cast<int>(px), static_cast<int>(py), static_cast<int>(nx), static_cast<int>(ny), 8, c);
                px = nx;
                py = ny;
            }
            break;
        }
        case MouthKind::Teeth:
            p.roundRect(cx - hw, cy - 10, cx + hw, cy + 10, 6, c);
            p.roundRect(cx - hw + 5, cy - 1, cx + hw - 5, cy + 1, 0, kBackground);
            for (int i = 1; i < 5; ++i) {
                const int x = cx - hw + i * hw * 2 / 5;
                p.roundRect(x - 1, cy - 6, x + 1, cy + 6, 0, kBackground);
            }
            break;
        case MouthKind::Side:
            p.line(cx - hw, cy + 4, cx + hw - 8, cy + 1, 10, c);
            p.line(cx + hw - 8, cy + 1, cx + hw, cy - 9, 10, c);
            break;
        case MouthKind::Small:
            p.circle(cx, cy, static_cast<int>(std::max(8.0f, s.mouthW / 2.5f)), c);
            break;
    }
}

void drawZ(Painter& p, int x, int y, int size, uint32_t c, uint8_t opa)
{
    p.line(x, y, x + size, y, 5, c, opa);
    p.line(x + size, y, x, y + size, 5, c, opa);
    p.line(x, y + size, x + size, y + size, 5, c, opa);
}

void drawSparkle(Painter& p, int x, int y, int size, uint32_t c)
{
    p.line(x - size, y, x + size, y, 4, c);
    p.line(x, y - size, x, y + size, 4, c);
}

void drawNote(Painter& p, int x, int y, uint32_t c)
{
    p.circle(x, y, 6, c);
    p.line(x + 5, y, x + 5, y - 22, 3, c);
    p.line(x + 5, y - 22, x + 14, y - 16, 3, c);
}

}  // namespace

void renderFace(const Shape& s, const Anim& a, Painter& p)
{
    const int ox = kScreenW / 2 + static_cast<int>(a.shakeX);
    const int oy = static_cast<int>(a.shakeY);
    const uint32_t c = s.color;

    drawEye(p, s.left, ox - kEyeCx, oy + kEyeCy, true, a, c, s.extras);
    drawEye(p, s.right, ox + kEyeCx, oy + kEyeCy, false, a, c, s.extras);
    if (s.extras & kBlush) {
        p.roundRect(ox - kEyeCx - 40, oy + kEyeCy + 44, ox - kEyeCx + 0, oy + kEyeCy + 58, 7, 0xFF7A9A, 200);
        p.roundRect(ox + kEyeCx - 0, oy + kEyeCy + 44, ox + kEyeCx + 40, oy + kEyeCy + 58, 7, 0xFF7A9A, 200);
    }
    drawMouth(p, s, a, ox, oy + kMouthCy);

    if (s.extras & kTears) {
        // A drop falls from each eye along the cheek, over and over.
        for (int side = -1; side <= 1; side += 2) {
            const int x = ox + side * (kEyeCx + 10);
            const int y = oy + kEyeCy + 24 + static_cast<int>(a.phase * 70.0f);
            p.circle(x, y, 7, 0x6FC8FF, static_cast<uint8_t>(255 * (1.0f - 0.7f * a.phase)));
            p.triangle(x - 6, y - 2, x + 6, y - 2, x, y - 14, 0x6FC8FF);
        }
    }
    if (s.extras & kSweat) {
        const int x = ox + kEyeCx + 48, y = oy + kEyeCy - 44 + static_cast<int>(a.phase * 12.0f);
        p.circle(x, y, 7, 0x8FD8FF);
        p.triangle(x - 6, y - 2, x + 6, y - 2, x, y - 15, 0x8FD8FF);
    }
    if (s.extras & kZzz) {
        const float ph = a.phase;
        drawZ(p, ox + 104 + static_cast<int>(ph * 16), oy + 40 - static_cast<int>(ph * 26), 16 + static_cast<int>(ph * 8), c,
              static_cast<uint8_t>(255 * (1.0f - 0.8f * ph)));
        drawZ(p, ox + 128, oy + 16, 11, c, 140);
    }
    if (s.extras & kQuestion) {
        const int x = ox + 118, y = oy + 30;
        p.arc(x, y, 12, 6, 180, 360 + 60, c);
        p.line(x + 6, y + 10, x, y + 18, 6, c);
        p.circle(x, y + 30, 4, c);
    }
    if (s.extras & kAnger) {
        const int x = ox + 120, y = oy + 26, d = 9;
        p.line(x - d, y - 3, x - 3, y - 3, 5, 0xFF3030);
        p.line(x - 3, y - 3, x - 3, y - d, 5, 0xFF3030);
        p.line(x + 3, y - d, x + 3, y - 3, 5, 0xFF3030);
        p.line(x + 3, y - 3, x + d, y - 3, 5, 0xFF3030);
        p.line(x - d, y + 3, x - 3, y + 3, 5, 0xFF3030);
        p.line(x - 3, y + 3, x - 3, y + d, 5, 0xFF3030);
        p.line(x + 3, y + d, x + 3, y + 3, 5, 0xFF3030);
        p.line(x + 3, y + 3, x + d, y + 3, 5, 0xFF3030);
    }
    if (s.extras & kSparkles) {
        const int k = 7 + static_cast<int>(4 * a.pulse);
        drawSparkle(p, ox - 128, oy + 34, k, c);
        drawSparkle(p, ox + 128, oy + 150, k, c);
        drawSparkle(p, ox + 116, oy + 30, k - 2, c);
    }
    if (s.extras & kNotes) {
        drawNote(p, ox - 124, oy + 60 - static_cast<int>(a.phase * 20), c);
        drawNote(p, ox + 118, oy + 70 - static_cast<int>((1.0f - a.phase) * 20), c);
    }
}

}  // namespace face
