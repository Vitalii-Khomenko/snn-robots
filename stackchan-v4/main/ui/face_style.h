#pragma once
// The robot's face as numbers, independent of LVGL: every emotion is a
// FaceShape (eyes, lids, brows, mouth, extras), shapes blend smoothly into
// each other, and renderFace() draws one through a tiny Painter interface.
// The firmware paints with LVGL (face_engine.cpp); tools/face_preview paints
// the same faces into a PNG on the PC, so the design can be checked without
// flashing.
#include <cstdint>

enum class Emotion {
    Normal,
    Happy,
    VeryHappy,
    Laughing,
    Sad,
    Crying,
    Sleepy,
    Surprised,
    Angry,
    Annoyed,
    Love,
    Shy,
    Wink,
    Curious,
    Confused,
    Thinking,
    Bored,
    Scared,
    Dizzy,
    Cool,
    Sick,
};
constexpr int kEmotionCount = 21;
const char* emotionName(Emotion e);

namespace face {

enum class EyeKind : uint8_t {
    Solid,     // rounded block (the default robot eye)
    Ring,      // wide open outline with a small pupil (fear, shock)
    ArcUp,     // closed, curved up: ^ ^ (laughing, blissful)
    ArcDown,   // closed, curved down (crying shut)
    Line,      // closed flat (sleep, wink)
    Heart,
    Cross,     // X (dizzy, sick)
};

enum class MouthKind : uint8_t {
    None,
    Flat,       // short straight line
    Smile,      // arc, curve > 0 smile, < 0 frown
    OpenSmile,  // D-shaped open laugh
    O,          // round open (surprise), size from open
    Wavy,       // trembling zig-zag (fear, unease)
    Teeth,      // clenched (anger)
    Side,       // lopsided smirk
    Small,      // tiny dot-mouth (shy, curious)
};

struct Eye {
    float w = 70, h = 84, r = 26;  // size and corner radius, px
    float dx = 0, dy = 0;          // offset from the eye's home position
    float lidTop = 0;              // 0..1 of the height covered from the top
    float lidTilt = 0;             // top lid slant: + inner corner lower (anger), - inner higher (worry)
    float lidBot = 0;              // 0..1 raised lower lid, curved: smiling eyes
    float pupil = 9;               // pupil radius for EyeKind::Ring
    float brow = 0;                // 0 = no brow, 1 = full brow
    float browY = 0;               // brow height above the eye (+ higher)
    float browTilt = 0;            // + inner end lower (anger), - inner end higher (worry, fear)
    EyeKind kind = EyeKind::Solid;
};

enum Extra : uint16_t {
    kNone = 0,
    kBlush = 1 << 0,
    kTears = 1 << 1,
    kSweat = 1 << 2,
    kZzz = 1 << 3,
    kQuestion = 1 << 4,
    kAnger = 1 << 5,
    kSparkles = 1 << 6,
    kNotes = 1 << 7,
    kGlint = 1 << 8,  // small highlight in solid eyes: looks alive
};

struct Shape {
    Eye left, right;  // left = on the viewer's left
    MouthKind mouth = MouthKind::Flat;
    float mouthW = 34, mouthCurve = 0, mouthOpen = 0, mouthDy = 0;
    uint32_t color = 0xBFEFFF;  // eyes, mouth, brows
    uint16_t extras = kGlint;
};

// Moving parts on top of the shape (animation state, px or 0..1 phases).
struct Anim {
    float shakeX = 0, shakeY = 0;  // tremble / bounce of the whole face
    float eyesX = 0, eyesY = 0;    // gaze shift of both eyes
    float blink = 0;               // 0 open .. 1 closed
    float pulse = 0;               // heart / sparkle beat, 0..1
    float phase = 0;               // slow cycle 0..1 for tears, zzz, notes
};

Shape shapeFor(Emotion e);
Shape blend(const Shape& a, const Shape& b, float t);  // t 0 = a, 1 = b

// What the renderer needs from a graphics library. Coordinates in px,
// colours 0xRRGGBB, opacity 0..255.
class Painter {
public:
    virtual ~Painter() = default;
    virtual void roundRect(int x1, int y1, int x2, int y2, int radius, uint32_t color, uint8_t opa = 255) = 0;
    virtual void circle(int cx, int cy, int r, uint32_t color, uint8_t opa = 255) = 0;
    virtual void triangle(int x1, int y1, int x2, int y2, int x3, int y3, uint32_t color) = 0;
    virtual void line(int x1, int y1, int x2, int y2, int width, uint32_t color, uint8_t opa = 255) = 0;
    // Degrees, 0 = 3 o'clock, clockwise (screen y down), rounded ends.
    virtual void arc(int cx, int cy, int r, int width, int startDeg, int endDeg, uint32_t color) = 0;
};

constexpr int kScreenW = 320, kScreenH = 240;
constexpr uint32_t kBackground = 0x05070C;
// Bounding box that renderFace() may touch (for partial redraws).
constexpr int kFaceX1 = 8, kFaceY1 = 8, kFaceX2 = 312, kFaceY2 = 232;

void renderFace(const Shape& s, const Anim& a, Painter& p);

}  // namespace face
