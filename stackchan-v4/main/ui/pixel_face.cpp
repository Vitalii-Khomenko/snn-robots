#include "pixel_face.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace face {
namespace {

// the palette of airwitech.com (its docs/DESIGN.md): paper, amber, cyan, violet, muted; coral is ours, for anger
constexpr uint32_t kPaper = 0xF4F0E9, kAmber = 0xFFB95F, kCyan = 0x62E4FF, kViolet = 0xAB9DFF, kMuted = 0x8F8CA0,
                   kCoral = 0xFF6B5F, kBlushTone = 0x9A6A3C;

// A glyph: rows of '#' (the face's tone), '+' (the glyph's accent), '*' (paper white), 'o' (dark: cuts a pixel out),
// '.' (nothing).
struct Glyph {
    int w, h;
    const char* rows[8];
};

// ---- eyes (left eye; right eyes are mirrored)
constexpr Glyph kEyeBlock{6, 7, {".####.", "###o##", "######", "######", "######", "######", ".####."}};
constexpr Glyph kEyeBig{7, 8, {".#####.", "####o##", "#######", "#######", "#######", "#######", "#######", ".#####."}};
constexpr Glyph kEyeHalf{6, 4, {"######", "######", "######", ".####."}};
constexpr Glyph kEyeSquint{6, 3, {".####.", "######", ".####."}};
constexpr Glyph kEyeArcUp{6, 3, {"..##..", ".#..#.", "#....#"}};
constexpr Glyph kEyeArcDown{6, 3, {"#....#", ".#..#.", "..##.."}};
constexpr Glyph kEyeLine{6, 1, {"######"}};
constexpr Glyph kEyeRing{6, 7, {".####.", "#....#", "#....#", "#.##.#", "#.##.#", "#....#", ".####."}};
constexpr Glyph kEyeRingSmall{6, 7, {".####.", "#....#", "#....#", "#..#.#", "#....#", "#....#", ".####."}};
constexpr Glyph kEyeHeart{7, 6, {".##.##.", "#######", "#######", ".#####.", "..###..", "...#..."}};
constexpr Glyph kEyeHeartSmall{5, 5, {".#.#.", "#####", "#####", ".###.", "..#.."}};
constexpr Glyph kEyeX{5, 5, {"#...#", ".#.#.", "..#..", ".#.#.", "#...#"}};
constexpr Glyph kEyePlus{5, 5, {"..#..", "..#..", "#####", "..#..", "..#.."}};
constexpr Glyph kEyeGt{5, 5, {"#....", ".##..", "...##", ".##..", "#...."}};  // > (mirrored: <)
constexpr Glyph kShades{22, 4, {"######################", "###*####......###*####", "########......########",
                                ".######........######."}};

// ---- brows (over the left eye; mirrored over the right)
constexpr Glyph kBrowFlat{6, 1, {"######"}};
constexpr Glyph kBrowWorry{6, 3, {"....##", "..##..", "##...."}};  // inner end high
constexpr Glyph kBrowAngry{6, 3, {"##....", "..##..", "....##"}};  // inner end low
constexpr Glyph kBrowRaise{6, 2, {".####.", "#....#"}};

// ---- mouths
constexpr Glyph kMouthFlat{8, 1, {"########"}};
constexpr Glyph kMouthSmall{4, 1, {"####"}};
constexpr Glyph kMouthSmile{8, 2, {"#......#", ".######."}};
constexpr Glyph kMouthBigSmile{10, 3, {"#........#", ".#......#.", "..######.."}};
constexpr Glyph kMouthOpen{10, 4, {"##########", "##########", ".########.", "..######.."}};
constexpr Glyph kMouthLaugh{10, 4, {"##########", "#........#", ".#..++..#.", "..######.."}};
constexpr Glyph kMouthFrown{10, 3, {"..######..", ".#......#.", "#........#"}};
constexpr Glyph kMouthO{4, 4, {".##.", "#..#", "#..#", ".##."}};
constexpr Glyph kMouthSmallO{3, 3, {".#.", "#.#", ".#."}};
constexpr Glyph kMouthWavy{10, 2, {".##..##..#", "#..##..##."}};
constexpr Glyph kMouthTeeth{10, 3, {"##########", "#.#.#.#.##", "##########"}};
constexpr Glyph kMouthSide{8, 2, {"......##", ".######."}};
constexpr Glyph kMouthSideDown{8, 2, {".######.", "##......"}};

// ---- extras
constexpr Glyph kBlushBar{4, 1, {"++++"}};
constexpr Glyph kTear{1, 2, {"+", "+"}};
constexpr Glyph kSweatDrop{2, 3, {".+", "++", "++"}};
constexpr Glyph kZBig{4, 4, {"####", "..#.", ".#..", "####"}};
constexpr Glyph kZSmall{3, 3, {"###", ".#.", "###"}};
constexpr Glyph kQuestionMark{4, 6, {".##.", "#..#", "...#", "..#.", "....", "..#."}};
constexpr Glyph kAngerMark{5, 5, {".+.+.", "++.++", ".....", "++.++", ".+.+."}};
constexpr Glyph kSparkle{3, 3, {".*.", "***", ".*."}};
constexpr Glyph kDot{1, 1, {"#"}};

constexpr int kEyeTop = 6, kBrowRow = 3, kMouthTop = 17;

int leftCol(const Glyph& g) { return 10 - g.w / 2; }
int rightCol(const Glyph& g) { return kPixCols - leftCol(g) - g.w; }

void draw(PixelGrid& out, const Glyph& g, int col, int row, uint32_t tone, uint32_t accent = kCyan, bool mirror = false)
{
    for (int r = 0; r < g.h; ++r) {
        const char* s = g.rows[r];
        for (int c = 0; c < g.w; ++c) {
            const char ch = s[mirror ? g.w - 1 - c : c];
            const int x = col + c, y = row + r;
            if (ch == '#') out.put(x, y, tone);
            else if (ch == '+') out.put(x, y, accent);
            else if (ch == '*') out.put(x, y, kPaper);
            else if (ch == 'o' && x >= 0 && x < kPixCols && y >= 0 && y < kPixRows) out.px[y][x] = 0;
        }
    }
}

// both eyes: the same glyph (the right one mirrored), with the gaze shift; a blink closes the open ones to a line
void eyes(PixelGrid& out, const Glyph& l, const Glyph& r, int top, int dx, int dy, bool blink, uint32_t tone)
{
    auto open = [](const Glyph& g) { return g.h >= 3 && &g != &kEyeArcUp && &g != &kEyeArcDown; };
    const Glyph& gl = blink && open(l) ? kEyeLine : l;
    const Glyph& gr = blink && open(r) ? kEyeLine : r;
    const int tl = gl.h == 1 && l.h > 1 ? top + l.h / 2 : top;
    const int tr = gr.h == 1 && r.h > 1 ? top + r.h / 2 : top;
    draw(out, gl, leftCol(gl) + dx, tl + dy, tone);
    draw(out, gr, rightCol(gr) + dx, tr + dy, tone, kCyan, true);
}

void brows(PixelGrid& out, const Glyph& b, int row, int dx, int dy, uint32_t tone)
{
    draw(out, b, leftCol(b) + dx, row + dy, tone);
    draw(out, b, rightCol(b) + dx, row + dy, tone, kCyan, true);
}

void mouth(PixelGrid& out, const Glyph& m, int row, uint32_t tone, uint32_t accent = kCoral, int dx = 0)
{
    draw(out, m, (kPixCols - m.w) / 2 + dx, row, tone, accent);
}

}  // namespace

uint32_t pixelTone(Emotion e)
{
    switch (e) {
        case Emotion::Happy:
        case Emotion::VeryHappy:
        case Emotion::Laughing:
        case Emotion::Love:
        case Emotion::Annoyed: return kAmber;
        case Emotion::Sad:
        case Emotion::Crying:
        case Emotion::Shy:
        case Emotion::Scared:
        case Emotion::Dizzy: return kViolet;
        case Emotion::Surprised:
        case Emotion::Curious:
        case Emotion::Confused:
        case Emotion::Thinking:
        case Emotion::Cool: return kCyan;
        case Emotion::Sleepy:
        case Emotion::Bored:
        case Emotion::Sick: return kMuted;
        case Emotion::Angry: return kCoral;
        default: return kPaper;
    }
}

void PixelGrid::clear() { std::memset(px, 0, sizeof(px)); }

void PixelGrid::put(int col, int row, uint32_t color)
{
    if (col < 0 || col >= kPixCols || row < 0 || row >= kPixRows) return;
    px[row][col] = (color & 0xFFFFFF) | 0x1000000u;
}

int PixelGrid::flush(Painter& out) const
{
    int n = 0;
    for (int r = 0; r < kPixRows; ++r)
        for (int c = 0; c < kPixCols; ++c) {
            if (!px[r][c]) continue;
            out.roundRect(c * kPixCell, r * kPixCell, c * kPixCell + kPixCell - 2, r * kPixCell + kPixCell - 2, 0,
                          px[r][c] & 0xFFFFFF);
            ++n;
        }
    return n;
}

void renderPixelFace(Emotion e, const Anim& a, PixelGrid& out)
{
    out.clear();
    const uint32_t t = pixelTone(e);
    // the whole face shakes / bounces in whole pixels; the gaze moves eyes and brows (up to 2 pixels aside, 1 up/down)
    const int sx = static_cast<int>(std::lround(a.shakeX / 3.0f)), sy = static_cast<int>(std::lround(a.shakeY / 3.0f));
    const int dx = std::clamp(static_cast<int>(std::lround(a.eyesX / 6.0f)), -2, 2) + sx;
    const int dy = std::clamp(static_cast<int>(std::lround(a.eyesY / 6.0f)), -1, 1) + sy;
    const bool blink = a.blink >= 0.5f;
    const bool beat = a.pulse >= 0.5f;
    const int ph8 = static_cast<int>(a.phase * 8.0f);  // 0..7: falling / rising extras
    switch (e) {
        case Emotion::Happy:
            eyes(out, kEyeArcUp, kEyeArcUp, kEyeTop + 2, dx, dy, false, t);
            mouth(out, kMouthBigSmile, kMouthTop + sy, t);
            break;
        case Emotion::VeryHappy:
            eyes(out, kEyeArcUp, kEyeArcUp, kEyeTop + 2, dx, dy, false, t);
            mouth(out, kMouthOpen, kMouthTop - 1 + sy, t);
            draw(out, kBlushBar, 5 + sx, 13 + sy, t, kBlushTone);
            draw(out, kBlushBar, 23 + sx, 13 + sy, t, kBlushTone);
            if (beat) {
                draw(out, kSparkle, 1, 2, t);
                draw(out, kSparkle, 28, 15, t);
            } else {
                draw(out, kSparkle, 27, 1, t);
            }
            break;
        case Emotion::Laughing:
            draw(out, kEyeGt, leftCol(kEyeGt) + sx, kEyeTop + 1 + sy, t);
            draw(out, kEyeGt, rightCol(kEyeGt) + sx, kEyeTop + 1 + sy, t, kCyan, true);
            mouth(out, kMouthLaugh, kMouthTop - 1 + sy, t, kCoral, sx);
            break;
        case Emotion::Sad:
            brows(out, kBrowWorry, kBrowRow, dx, dy, t);
            eyes(out, kEyeBlock, kEyeBlock, kEyeTop + 1, dx, dy, blink, t);
            mouth(out, kMouthFrown, kMouthTop + 1 + sy, t);
            break;
        case Emotion::Crying:
            brows(out, kBrowWorry, kBrowRow + 1, dx, dy, t);
            eyes(out, kEyeArcDown, kEyeArcDown, kEyeTop + 3, dx, dy, false, t);
            mouth(out, kMouthWavy, kMouthTop + 1 + sy, t);
            // a drop from each eye, over and over
            draw(out, kTear, leftCol(kEyeArcDown) + dx, 10 + ph8 + sy, t, kCyan);
            draw(out, kTear, rightCol(kEyeArcDown) + 5 + dx, 10 + ((ph8 + 4) % 8) + sy, t, kCyan);
            break;
        case Emotion::Sleepy:
            eyes(out, kEyeLine, kEyeLine, kEyeTop + 4, 0, sy, false, t);
            mouth(out, kMouthSmall, kMouthTop + 1 + sy, t);
            draw(out, kZBig, 25 + ph8 / 3, 6 - ph8 / 2, t);
            draw(out, kZSmall, 28, 1, t);
            break;
        case Emotion::Surprised:
            brows(out, kBrowFlat, kBrowRow - 1, dx, dy, t);
            eyes(out, kEyeRing, kEyeRing, kEyeTop, dx, dy, blink, t);
            mouth(out, kMouthO, kMouthTop - 1 + sy, t);
            break;
        case Emotion::Angry:
            brows(out, kBrowAngry, kBrowRow + 2, dx, dy, t);
            eyes(out, kEyeHalf, kEyeHalf, kEyeTop + 3, dx, dy, blink, t);
            mouth(out, kMouthTeeth, kMouthTop + sy, t);
            draw(out, kAngerMark, 26, 1, t, kCoral);
            break;
        case Emotion::Annoyed:
            brows(out, kBrowFlat, kBrowRow + 4, dx, dy, t);
            eyes(out, kEyeHalf, kEyeHalf, kEyeTop + 3, dx, dy, blink, t);
            mouth(out, kMouthSideDown, kMouthTop + 1 + sy, t);
            break;
        case Emotion::Love: {
            const Glyph& h = beat ? kEyeHeart : kEyeHeartSmall;
            eyes(out, h, h, kEyeTop + (beat ? 0 : 1), dx, dy, false, t);
            mouth(out, kMouthBigSmile, kMouthTop + sy, t);
            draw(out, kBlushBar, 5 + sx, 13 + sy, t, kBlushTone);
            draw(out, kBlushBar, 23 + sx, 13 + sy, t, kBlushTone);
            break;
        }
        case Emotion::Shy:
            eyes(out, kEyeArcUp, kEyeArcUp, kEyeTop + 3, dx - 1, dy + 1, false, t);
            mouth(out, kMouthSmall, kMouthTop + 1 + sy, t);
            draw(out, kBlushBar, 5 + sx, 13 + sy, t, kBlushTone);
            draw(out, kBlushBar, 23 + sx, 13 + sy, t, kBlushTone);
            break;
        case Emotion::Wink:
            draw(out, blink ? kEyeLine : kEyeBlock, leftCol(kEyeBlock) + dx, kEyeTop + (blink ? 3 : 0) + dy, t);
            draw(out, kEyeArcUp, rightCol(kEyeArcUp) + dx, kEyeTop + 3 + dy, t, kCyan, true);
            mouth(out, kMouthSmile, kMouthTop + sy, t);
            if (beat) draw(out, kSparkle, 27, 3, t);
            break;
        case Emotion::Curious:
            draw(out, kBrowRaise, rightCol(kBrowRaise) + dx, kBrowRow - 1 + dy, t, kCyan, true);
            draw(out, blink ? kEyeLine : kEyeBlock, leftCol(kEyeBlock) + dx, kEyeTop + 1 + (blink ? 3 : 0) + dy, t);
            draw(out, blink ? kEyeLine : kEyeBig, rightCol(kEyeBig) + dx, kEyeTop + (blink ? 4 : 0) + dy, t, kCyan, true);
            mouth(out, kMouthSmallO, kMouthTop + sy, t);
            break;
        case Emotion::Confused:
            draw(out, kBrowWorry, leftCol(kBrowWorry) + dx, kBrowRow + 1 + dy, t);
            draw(out, blink ? kEyeLine : kEyeBlock, leftCol(kEyeBlock) + dx, kEyeTop + (blink ? 3 : 0) + dy, t);
            draw(out, kEyeSquint, rightCol(kEyeSquint) + dx, kEyeTop + 3 + dy, t, kCyan, true);
            mouth(out, kMouthWavy, kMouthTop + sy, t);
            draw(out, kQuestionMark, 27, 1, t);
            break;
        case Emotion::Thinking:
            brows(out, kBrowFlat, kBrowRow, dx + 1, dy - 1, t);
            eyes(out, kEyeBlock, kEyeBlock, kEyeTop - 1, dx + 1, dy, blink, t);
            mouth(out, kMouthSide, kMouthTop + sy, t, kCoral, 1);
            draw(out, kDot, 25, 4, t);
            if (ph8 >= 3) draw(out, kDot, 27, 2, t);
            if (ph8 >= 6) draw(out, kDot, 29, 0, t);
            break;
        case Emotion::Bored:
            eyes(out, kEyeHalf, kEyeHalf, kEyeTop + 3, dx, dy, blink, t);
            mouth(out, kMouthFlat, kMouthTop + 1 + sy, t);
            break;
        case Emotion::Scared:
            brows(out, kBrowWorry, kBrowRow, dx, dy, t);
            eyes(out, kEyeRingSmall, kEyeRingSmall, kEyeTop, dx, dy, blink, t);
            mouth(out, kMouthWavy, kMouthTop + 1 + sy, t, kCoral, sx);
            draw(out, kSweatDrop, 28 + sx, 3 + ph8 / 3 + sy, t, kCyan);
            break;
        case Emotion::Dizzy: {
            const Glyph& g = beat ? kEyeX : kEyePlus;  // the eyes spin
            eyes(out, g, g, kEyeTop + 1, dx, dy, false, t);
            mouth(out, kMouthWavy, kMouthTop + sy, t);
            break;
        }
        case Emotion::Cool:
            draw(out, kShades, (kPixCols - kShades.w) / 2 + sx, kEyeTop + 1 + sy, t);
            mouth(out, kMouthSide, kMouthTop + sy, t, kCoral, 1);
            break;
        case Emotion::Sick:
            eyes(out, kEyeSquint, kEyeSquint, kEyeTop + 3, dx, dy, blink, t);
            mouth(out, kMouthWavy, kMouthTop + sy, t);
            draw(out, kSweatDrop, 27 + sx, 2 + ph8 / 3 + sy, t, kCyan);
            break;
        default:  // Normal
            eyes(out, kEyeBlock, kEyeBlock, kEyeTop, dx, dy, blink, t);
            mouth(out, kMouthSmile, kMouthTop + sy, t);
            break;
    }
}

void dissolve(const PixelGrid& from, const PixelGrid& to, float t, PixelGrid& out)
{
    for (int r = 0; r < kPixRows; ++r)
        for (int c = 0; c < kPixCols; ++c) {
            uint32_t h = static_cast<uint32_t>(r * kPixCols + c) * 2654435761u;  // a fixed shuffle of the pixels
            h ^= h >> 15;
            const float when = static_cast<float>(h & 1023u) / 1024.0f;
            out.px[r][c] = when < t ? to.px[r][c] : from.px[r][c];
        }
}

}  // namespace face
